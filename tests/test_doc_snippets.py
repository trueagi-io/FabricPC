"""Contract test for documentation code snippets.

Extracts every fenced ```python block from docs/user_guides/*.md and README.md
and checks it against the installed fabricpc API:

1. Parse gate: a block tagged ``python`` must parse as Python.
2. Import check: every top-level import statement in a block must execute.
   A missing module is tolerated only when it is one of the project's declared
   optional dependencies (``TOLERATED_MISSING``); any other missing module —
   fabricpc, a core dependency, or a typo'd name — is a failure.
3. Signature check: for calls whose callee resolves to a fabricpc callable
   (through the block's imports, or through a variable assigned from a
   fabricpc class), every keyword argument must exist in the callable's
   signature. When the call has no ``...`` placeholder and no *args/**kwargs
   expansion, the full binding (positional arity and required arguments) is
   checked as well.

A block is exempted by placing ``<!-- doc-snippet: skip -->`` on the nearest
non-blank line above its opening fence.
"""

import ast
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = sorted((REPO_ROOT / "docs" / "user_guides").glob("*.md")) + [
    REPO_ROOT / "README.md"
]

SKIP_MARKER = "<!-- doc-snippet: skip -->"

# Import names of the optional-dependency groups in pyproject.toml ([tfds],
# [experiments], [viz]). Only these may be absent without failing the import
# check; a missing core dependency or a typo'd module name is a doc defect.
TOLERATED_MISSING = {
    "aim",
    "kaleido",
    "pandas",
    "plotly",
    "scipy",
    "tensorflow",
    "tensorflow_datasets",
    "tokenizers",
}

# Ensure repo-root modules used in snippets (e.g. jax_setup) are importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _skip_marker_precedes(lines, fence_idx):
    """True when the nearest non-blank line above lines[fence_idx] is the marker."""
    for j in range(fence_idx - 1, -1, -1):
        stripped = lines[j].strip()
        if stripped:
            return stripped == SKIP_MARKER
    return False


def extract_python_blocks(path):
    """Return [(start_line, source)] for python-tagged fenced blocks.

    Any line whose stripped form starts with ``` toggles a fence, so an
    opening fence carrying an info string (e.g. ```python title=x) cannot
    desynchronize extraction; its language is the first word after the
    backticks.
    """
    blocks = []
    lines = path.read_text().splitlines()
    in_block = False
    block_lang = None
    block_start = 0
    block_skipped = False
    buf = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_block:
            if stripped.startswith("```"):
                in_block = True
                info = stripped[3:].strip()
                block_lang = info.split()[0] if info else ""
                block_start = i + 2  # 1-based line after the fence
                block_skipped = _skip_marker_precedes(lines, i)
                buf = []
        else:
            if stripped.startswith("```"):
                if block_lang == "python" and not block_skipped:
                    blocks.append((block_start, "\n".join(buf)))
                in_block = False
            else:
                buf.append(line)
    return blocks


def _is_fabricpc_callable(obj):
    module = getattr(obj, "__module__", "") or ""
    return module.startswith("fabricpc")


def _exec_import(stmt, namespace, failures, label):
    code = compile(ast.Module(body=[stmt], type_ignores=[]), label, "exec")
    try:
        exec(code, namespace)
    except ModuleNotFoundError as e:
        top = (e.name or "").split(".")[0]
        if top not in TOLERATED_MISSING:
            failures.append(f"{label}: import failed: {e}")
    except ImportError as e:
        # The module imported but the requested symbol is missing: always a
        # doc defect (an absent module raises ModuleNotFoundError above).
        failures.append(f"{label}: import failed: {e}")
    except Exception as e:  # noqa: BLE001 - any other error is a doc defect
        failures.append(f"{label}: import raised {type(e).__name__}: {e}")


def _resolve(node, namespace):
    """Resolve an ast expression to a live object via the block's imports."""
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve(node.value, namespace)
        if base is not None:
            return getattr(base, node.attr, None)
    return None


def _call_is_lenient(call):
    """True when the call uses placeholders that make full binding impossible."""
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            return True
        if isinstance(arg, ast.Constant) and arg.value is Ellipsis:
            return True
    for kw in call.keywords:
        if kw.arg is None:  # **kwargs expansion
            return True
        if isinstance(kw.value, ast.Constant) and kw.value.value is Ellipsis:
            return True
    return False


def _check_call(call, func, is_method_on_instance, failures, label):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return
    params = list(sig.parameters.values())
    if is_method_on_instance and params and params[0].name == "self":
        params = params[1:]
        sig = sig.replace(parameters=params)

    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
    valid_kw = {
        p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    kw_names = [kw.arg for kw in call.keywords if kw.arg is not None]

    if not accepts_var_kw:
        for name in kw_names:
            if name not in valid_kw:
                failures.append(
                    f"{label}: keyword '{name}' not in signature {sig} of "
                    f"{getattr(func, '__qualname__', func)}"
                )

    if _call_is_lenient(call):
        return
    try:
        sig.bind(*[object()] * len(call.args), **{name: object() for name in kw_names})
    except TypeError as e:
        failures.append(
            f"{label}: call does not bind to signature {sig} of "
            f"{getattr(func, '__qualname__', func)}: {e}"
        )


def check_block(source, label):
    failures = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"{label}: block tagged python does not parse: {e}"]

    namespace = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _exec_import(stmt, namespace, failures, label)

    # Map variables assigned from fabricpc class constructors to their class,
    # so method calls on those variables can be signature-checked.
    var_types = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = _resolve(node.value.func, namespace)
            if inspect.isclass(callee) and _is_fabricpc_callable(callee):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_types[target.id] = callee

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        line = f"{label}:{node.lineno}"
        func = _resolve(node.func, namespace)
        if func is not None and _is_fabricpc_callable(func):
            _check_call(
                node, func, is_method_on_instance=False, failures=failures, label=line
            )
            continue
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            cls = var_types.get(node.func.value.id)
            if cls is None:
                continue
            method = getattr(cls, node.func.attr, None)
            if method is None:
                failures.append(
                    f"{line}: {cls.__name__} has no attribute '{node.func.attr}'"
                )
            elif callable(method) and _is_fabricpc_callable(method):
                _check_call(
                    node,
                    method,
                    is_method_on_instance=True,
                    failures=failures,
                    label=line,
                )
    return failures


@pytest.mark.parametrize("doc_path", DOC_FILES, ids=lambda p: p.name)
def test_doc_snippets(doc_path):
    failures = []
    for start_line, source in extract_python_blocks(doc_path):
        rel = doc_path.relative_to(REPO_ROOT)
        failures.extend(check_block(source, f"{rel}:{start_line}"))
    assert not failures, "\n".join(failures)


def test_typoed_import_is_reported():
    failures = check_block("import optx\nopt = optx.adamw(1e-3)\n", "typo")
    assert failures, "a misspelled third-party import must be reported"


def test_info_string_fence_does_not_desync(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text("```python title=example.py\nx = 1\n```\n\n```python\ny = 2\n```\n")
    assert extract_python_blocks(md) == [(2, "x = 1"), (6, "y = 2")]


def test_skip_marker_exempts_only_the_next_block(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        f"{SKIP_MARKER}\n\n```python\nskipped = True\n```\n\n"
        "```python\nchecked = True\n```\n"
    )
    assert extract_python_blocks(md) == [(8, "checked = True")]
