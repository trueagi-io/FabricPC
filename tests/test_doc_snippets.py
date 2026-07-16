"""Contract test for documentation code snippets.

Extracts every fenced ```python block from docs/user_guides/*.md and README.md
and checks it against the installed fabricpc API:

1. Parse gate: a block tagged ``python`` must parse as Python.
2. Import check: every top-level import statement in a block must execute.
   Missing third-party optional dependencies are tolerated; a missing fabricpc
   module or symbol is a failure.
3. Signature check: for calls whose callee resolves to a fabricpc callable
   (through the block's imports, or through a variable assigned from a
   fabricpc class), every keyword argument must exist in the callable's
   signature. When the call has no ``...`` placeholder and no *args/**kwargs
   expansion, the full binding (positional arity and required arguments) is
   checked as well.

A block is exempted by placing ``<!-- doc-snippet: skip -->`` on the line
before its opening fence.
"""

import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = sorted((REPO_ROOT / "docs" / "user_guides").glob("*.md")) + [
    REPO_ROOT / "README.md"
]

SKIP_MARKER = "<!-- doc-snippet: skip -->"
FENCE_RE = re.compile(r"^```(\w*)\s*$")

# Ensure repo-root modules used in snippets (e.g. jax_setup) are importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def extract_python_blocks(path):
    """Return [(start_line, source)] for python-tagged fenced blocks."""
    blocks = []
    lines = path.read_text().splitlines()
    in_block = False
    block_lang = None
    block_start = 0
    buf = []
    skip_next = False
    for i, line in enumerate(lines, start=1):
        m = FENCE_RE.match(line.strip()) if line.strip().startswith("```") else None
        if not in_block:
            if line.strip() == SKIP_MARKER:
                skip_next = True
                continue
            if m:
                in_block = True
                block_lang = m.group(1)
                block_start = i + 1
                buf = []
            elif line.strip():
                skip_next = False
        else:
            if line.strip().startswith("```"):
                if block_lang == "python" and not skip_next:
                    blocks.append((block_start, "\n".join(buf)))
                in_block = False
                skip_next = False
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
        if top == "fabricpc" or top == "jax_setup":
            failures.append(f"{label}: import failed: {e}")
        # Optional third-party dependency absent in this environment: tolerated.
    except ImportError as e:
        src = ast.unparse(stmt)
        if "fabricpc" in src:
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
