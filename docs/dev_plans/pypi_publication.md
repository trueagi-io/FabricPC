# PyPI publication: repository preparation and release of `fabricpc`

## Context

FabricPC is installable today only by cloning the repo. Publishing to pypi.org makes `pip install fabricpc` work for external users. The name `fabricpc` is unclaimed on both pypi.org and test.pypi.org (verified 2026-07-30, both return 404). This plan covers the configuration fixes required before the first upload, the release automation, and the release procedure itself.

**Settled decisions:**

1. **Release mechanism:** GitHub Actions with PyPI Trusted Publishing (OIDC), triggered by a GitHub release.
2. **First published version:** 0.4.0, cut from `main` after the pending muPC output-scaling branch merges.
3. **`jax_setup.py`:** Option A (§2) — move to `fabricpc/jax_config.py`, keep the eager `__init__`, rename to `setup_jax`, and re-contract the helper to "call before the first JAX computation" using JAX's post-import configuration API.
4. **Metadata (§3):** `authors = [{name = "SingularityNET Foundation", email = "info@singularitynet.io"}]`; the `maintainers` metadata field stays empty; the PyPI accounts `MatthewBehrend` and `SingularityNET` get collaborator roles on the project, set on pypi.org after the first upload.
5. **Dependencies:** `flax`, `chex`, `jaxtyping`, and `orbax-checkpoint` stay in core `dependencies`. `jaxlib` is removed, `optuna` moves to the `[experiments]` extra, `jax` gets floor `>=0.7.0`, and `[all]` no longer pulls in `[dev]`.
6. **CI:** add a minimal pytest workflow alongside the publish workflow.

---

## 1. Readiness review of the configuration files

### Already correct

- **Version single-sourced.** `pyproject.toml` holds `version`; `fabricpc/__init__.py` reads it at runtime via `importlib.metadata.version("fabricpc")`. No duplication.
- **Package discovery.** `[tool.setuptools.packages.find] include = ["fabricpc*"]` — `examples/`, `tests/`, `docs/`, `data/`, `runs/` cannot enter the wheel. All 12 subpackages have `__init__.py`. No non-Python files inside `fabricpc/`, so no package-data declarations needed.
- **Optional dependencies properly isolated.** Every extra-gated import in the library is lazy or guarded: `tokenizers` (try/except in `fabricpc/utils/data/dataloader.py`), `tensorflow`/`tensorflow_datasets` (function-local imports), `scipy` (function-local in `fabricpc/experiments/statistics.py`), `aim` (`find_spec` guard in `fabricpc/utils/dashboarding/_aim_available.py`). A bare `pip install fabricpc` followed by `import fabricpc` will not crash on a missing extra.
- **Pure-Python wheel** (`py3-none-any`), setuptools backend, MIT `LICENSE` file present, `CHANGELOG.md` maintained per release.
- **No leaks.** No hardcoded absolute paths, credentials, or private URLs in `fabricpc/`. Large artifacts (`data/` 65M, `.aim/` 105M, `runs/`) live outside the package tree and are git-ignored; the default sdist will not sweep them in.

### Must fix before upload

| # | File / field | Defect | Fix |
|---|---|---|---|
| 1 | `pyproject.toml` classifier `"Private :: Do Not Upload"` | PyPI rejects any distribution carrying this classifier. It is the deliberate upload guard. | Remove in the release PR. The workflow trigger (release-only) remains the second guard. |
| 2 | No `license` field, no license classifier | PyPI page would show no license despite the MIT `LICENSE` file. | `license = "MIT"` (PEP 639 SPDX expression); bump `build-system.requires` to `setuptools>=77`, which emits `License-Expression` metadata and auto-includes the `LICENSE` file. |
| 3 | No `[project.urls]` | PyPI page would have no links to source, docs, or changelog. | Add Homepage/Repository (`https://github.com/trueagi-io/FabricPC`), Documentation (`.../blob/main/docs/user_guides/00_index.md`), Changelog (`.../blob/main/CHANGELOG.md`), Issues. |
| 4 | `dependencies` includes `jaxlib` | Redundant — `jax` pins its own matched `jaxlib`. A bare unpinned `jaxlib` can also fight the coupled wheel set that `jax[cuda12]`/`jax[cuda13]` manage. | Remove `jaxlib`. |
| 5 | `optuna` in core `dependencies` | Imported only by `fabricpc/tuning/bayesian_tuner.py`; `fabricpc/__init__.py` does not import `tuning`. Core users pay optuna's install cost for a subpackage they never touch. | Move to `[experiments]` (the tuning guide already lives in `15_api_experiments.md`). Document that `fabricpc.tuning` requires the extra. |
| 6 | `jax` has no version floor | Nothing stops a resolver from selecting a years-old jax, and `requires-python` does not stand in for a floor — it bounds jax only from above. jax 0.4.2 (2023-01) declares `>=3.8`, so it installs on Python 3.11 unopposed; the first jax to declare `>=3.11` is 0.7.0. | `jax>=0.7.0`, the oldest release whose own Python floor matches this project's, tested by a `test.yml` leg (§4). |
| 7 | `[tool.setuptools] py-modules = ["jax_setup"]` | Ships a generically named top-level module into every user's site-packages (§2). | Delete the table; the module moves inside the package and `packages.find` picks it up. |
| 8 | `all = ["fabricpc[dev,tfds,experiments,viz]"]` | `pip install "fabricpc[all]"` would give end users black, mypy, ruff, and pre-commit. | `all = ["fabricpc[tfds,experiments,viz]"]`; contributor install becomes `pip install -e ".[all,dev]"`. README and `01_installation.md` updated accordingly. |
| 9 | `authors = [{name = "FabricPC Authors"}]`, no email | Generic, inconsistent with LICENSE (Matthew Behrend) and README (SingularityNET). | Per §3. |
| 10 | README as PyPI landing page | Relative links (`docs/user_guides/...`, `examples/...`, `LICENSE`) break on pypi.org, which renders the README standalone. Installation section says "Clone this repo" only. | Convert relative links to absolute GitHub URLs; add a `pip install fabricpc` section for users; keep the clone + editable install path for contributors. |
| 11 | CI has only `lint.yml` | No test gate and no publish pipeline. | Add `publish.yml` (Trusted Publishing) and a minimal `test.yml` (§4). |
| 12 | `[dev]` extra lacks release tooling | `build` and `twine` are not installed anywhere; local verification of the distribution is currently impossible. | Add `build` and `twine` to `[dev]`. |

### Kept as-is

- `flax>=0.7.5`, `chex>=0.1.84`, `jaxtyping>=0.2.23`, `orbax-checkpoint>=0.4.0` stay in core `dependencies` (decision 5). None is imported anywhere in `fabricpc/`, `examples/`, `tests/`, or `scripts/` today, so every user installs four packages the library does not currently use; they are retained for planned checkpointing and typing work rather than removed and re-added later. `pip check` in the smoke job (§4) covers their co-resolution with the jax wheel set.
- `Development Status :: 3 - Alpha` (README states APIs may change until v1.0), classifiers through 3.13.

### Changed during implementation

- Python floor raised from `>=3.10` to `>=3.11`: the 3.10 classifier, the `py310` targets for black and ruff, and the "Python 3.10–3.13" line in README, `01_installation.md`, and `16_troubleshooting.md` all move with it. The CI matrices below run 3.11 and 3.13 instead of 3.10 and 3.13.
- `setup_jax`'s parameter is named `platform`, not `jax_platforms`; the three callsites passing it by keyword are migrated with the rest.
- The module is `fabricpc/jax_config.py`, not `fabricpc/jax_setup.py`: `setup_jax` is the verb, the module is the noun, and "setup" no longer names a temporal constraint that the re-contracted helper does not have. Test file follows: `tests/test_jax_config.py`.
- `import jax` sits at module scope, not inside `setup_jax`. A function-local import defers nothing here: `from fabricpc.jax_config import setup_jax` executes `fabricpc/__init__`, whose eager submodule imports pull in jax before the helper can run. Measured with `python -X importtime`: importing the helper costs 477 ms, of which jax is 380 ms, and `jax` is in `sys.modules` on return. The deferred import only hid a hard core dependency from the module header.
- `setup_jax` guards each XLA flag by name, not by `name=value`: a value already present in `XLA_FLAGS` — from the caller's shell or a previous `setup_jax` call — wins, and repeated calls leave the string unchanged. The inherited assembly appended the autotune flag unconditionally, so every call grew `XLA_FLAGS` and the appended `=1` overrode a user-set level; the `name=value` guards on the other two flags missed user-set opposite values the same way. The comparison is against whole names — `XLA_FLAGS` is split on whitespace and each token truncated at `=` — so a longer flag sharing a prefix (`--xla_gpu_foo_bar` for `--xla_gpu_foo`) does not read as the shorter one being present. `tests/test_jax_config.py` pins all three behaviors.
- `TF_CPP_MIN_LOG_LEVEL` is a second exception to the consumed-at-backend-initialization rule, documented in the module docstring: the native runtime caches the level at its first emitted log message, so the post-import helper suppresses messages from backend initialization onward but not any emitted during `import jax`.
- `publish-testpypi` sets `skip-existing: true`: TestPyPI enforces the same (name, version, filename) reservation as PyPI, so a second rehearsal at an already-uploaded version would otherwise fail at the upload step. With the flag, the rerun verifies `build` + `smoke` and uploads nothing. The production `publish` job never sets it.
- `test.yml` runs pushes only on `main` — PR commits otherwise run twice, once per event — and cancels superseded runs through a per-ref `concurrency` group.

---

## 2. `jax_setup.py`: placement and contract

### Why the current setup works, and what actually constrains it

The concern was that `jax_setup.py` "only works properly at the project level folder." The file's *location* is not what makes it work — the function only writes `os.environ` entries and imports nothing but `os`. Two separate facts create the current behavior:

- **Why it is importable today:** `pip install -e .` installs it as a top-level module (via `py-modules = ["jax_setup"]`). That is how `examples/mnist_demo.py` can do `from jax_setup import ...` even though the script's own directory is `examples/`, not the repo root. The repo-root location is an artifact of packaging, not a functional requirement.
- **The real constraint is temporal, not spatial** — and narrower than the function's name claims. Measured against the repo's jax 0.10.2 (CPU backend):

| Test | Action | Result |
|---|---|---|
| A | `import jax`, then set `JAX_PLATFORMS=bogus_platform` in `os.environ`, then `jax.devices()` | Silently ignored — devices resolve normally. The env var is captured once, at `import jax`. |
| B | `import jax`, then `jax.config.update("jax_platforms", "cpu")`, then `jax.devices()` | Platform applied. This is JAX's documented post-import platform API. |
| C | `import jax`, then set `XLA_FLAGS=--xla_force_host_platform_device_count=4`, then `jax.devices()` | 4 CPU devices — `XLA_FLAGS` is consumed at backend initialization, not at import. `XLA_PYTHON_CLIENT_PREALLOCATE` is read in the same phase (GPU client creation). |

So only the `JAX_PLATFORMS` *environment-variable* path requires pre-import ordering, and JAX provides `jax.config.update("jax_platforms", ...)` precisely so platform selection works post-import. Everything else the helper sets is read at backend initialization, which JAX defers until the first computation or device query.

Test B has since been re-run on a host with a working CUDA driver: `setup_jax("cpu")` after `import jax` yields CPU devices where the default would be `CudaDevice(id=0)`, so the post-import platform API does select between real backends, not only between a backend and itself. `tests/test_jax_config.py::test_platform_selection_works_after_jax_is_imported` holds this.

Backend initialization is the deadline, and JAX enforces it silently — running a computation first and then calling `setup_jax("cpu")` leaves the CUDA device in place and reports nothing. The old name, `set_jax_flags_before_importing_jax`, carried its precondition; `setup_jax` does not, so the helper checks `jax._src.xla_bridge.backends_are_initialized()` and raises a `RuntimeWarning` naming what was discarded. That symbol is private, so it is imported inside `try`/`except ImportError`: if a future jax moves it, the warning is lost and `setup_jax` keeps working.

### How peer libraries structure their package import

Inspected `__init__.py` of the installed versions (`__getattr__` definitions and lazy-loader machinery):

| Library | `__init__` style | Motivation |
|---|---|---|
| jax 0.10.2 | Eager | — |
| flax 0.12.7 | Eager | — |
| optax 0.2.8 | Eager | — |
| chex 0.1.92 | Eager | — |
| torch | Eager (loads the libtorch C++ runtime at `import torch`; by design, not installed here) | — |
| numpy 2.5.1 | Eager core; a single `__getattr__` serving deprecated aliases and submodule shims | Deprecation plumbing, not laziness |
| scipy | Lazy subpackages via `__getattr__` | Import cost of ~20 large compiled subpackages |
| tensorflow | Lazy via its own loader | Same — enormous module tree |

The pattern: the entire JAX ecosystem is eager; lazy loading appears only in libraries whose import cost is dominated by a very large optional-subpackage tree. No peer uses laziness to solve configuration ordering.

### Chosen: Option A — move to `fabricpc/jax_config.py`, keep the eager `__init__`, re-contract the helper

The helper's "before importing jax" contract was always stricter than JAX requires (Tests A–C). The function is rewritten to be correct when called *after* `import jax`, as long as it runs before the first JAX computation:

```python
# fabricpc/jax_config.py
def setup_jax(platform: str | None = None) -> None:
    """Configure JAX for FabricPC. Call before the first JAX computation
    (backend initialization). Import order no longer matters."""
    if backends_are_initialized():
        warnings.warn("setup_jax() ran after the JAX backend was initialized ...",
                      RuntimeWarning, stacklevel=2)

    if platform is not None and "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = platform          # visible to worker subprocesses
        jax.config.update("jax_platforms", platform)    # effective in this process (Test B)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # ... XLA_FLAGS assembly guarded by flag name (Test C: consumed at backend init) ...
```

The `"JAX_PLATFORMS" not in os.environ` guard preserves the current setdefault semantics: a platform set in the user's shell wins over the function argument, exactly as today. The env write plus `config.update` pair is one deterministic path — the env var covers spawned worker processes, the config call covers the already-imported current process.

- **Rename:** `set_jax_flags_before_importing_jax` → `setup_jax`. The old name states a constraint that no longer exists. All callers are migrated in the same change; no alias is kept.
- **Public import path:** `setup_jax` is re-exported from `fabricpc/__init__.py`, so docs, examples, and users write `from fabricpc import setup_jax`. `fabricpc.jax_config` stays as the implementation module and keeps working; the eager `__init__` means the short path costs nothing extra.
- **The relaxed contract needs a guard.** The old name carried its precondition; `setup_jax` does not, and JAX enforces the deadline silently — see §2's re-run of Test B. The helper checks `jax._src.xla_bridge.backends_are_initialized()` behind `try`/`except ImportError` and warns.
- **Why:** matches the eager-`__init__` convention of every JAX-ecosystem peer; no top-level namespace pollution in site-packages; the contract is enforceable by a real test (call after `import jax`, assert the platform took effect); small diff — one function rewrite plus mechanical caller migration.
- **Residual risk:** relies on `jax.config.update("jax_platforms", ...)` (public, documented) and on XLA flags being read at client creation (verified empirically; stable XLA behavior).
- **Clone workflow: unchanged.** `git clone` + `pip install -e ".[all]"` works as before; only the import line changes.
- **Migration scope (all callers, one mechanical line each, no compatibility shim):** 15 files in `examples/`, 2 in `scripts/`, `README.md`, `docs/user_guides/02_quickstart.md`, `docs/user_guides/16_troubleshooting.md`, and `tests/test_doc_snippets.py` (also delete its `sys.path.insert(REPO_ROOT)` workaround at lines 51–52, which exists only to reach the root-level module). Breaking change documented in CHANGELOG with the one-line migration.

### Rejected: Option B — lazy `fabricpc/__init__.py` (PEP 562), keeping the pre-import contract

Convert the package `__init__` to `__getattr__`-based lazy loading so `from fabricpc.jax_config import ...` does not pull in jax, keeping the literal "before importing jax" contract.

- **Pros:** contract stays literally true; `import fabricpc` becomes cheap.
- **Cons:** no precedent among JAX-ecosystem peers (table above — jax, flax, optax, chex, torch are all eager); the lazy mechanisms that do exist (scipy, tensorflow) solve import *cost* at a much larger scale, not configuration ordering; it couples the package's import architecture to the needs of one helper function; error timing moves from import to first attribute access; static-analysis names must be mirrored in a `TYPE_CHECKING` block and maintained. Option A dissolves the constraint that was the only reason to consider this.

### Rejected: Option C — status quo (top-level `jax_setup` module in the wheel)

- **Pros:** zero work.
- **Cons:** every `pip install fabricpc` drops a module named `jax_setup` at the top of site-packages. Any other package shipping the same generic name collides silently (last-installed wins). Permanent once external users import it.

### Rejected: Option D — repo-only file, excluded from the wheel

- **Pros:** no namespace pollution.
- **Cons:** installed users lose the documented configuration helper entirely; examples rely on the module being installed, so the editable install breaks too.

---

## 3. Author and maintainer metadata

### Two different things named "maintainer"

PyPI surfaces two unrelated notions of maintainer, which is why the project page and the package metadata disagree:

- **Metadata fields.** `[project] authors` / `maintainers` in `pyproject.toml` become the `Author`, `Author-email`, `Maintainer`, `Maintainer-email` core-metadata fields and appear in the "Meta" block of the project page. `authors = [{name = ..., email = ...}]` serializes into `Author-email` as `Name <address>`; the bare `Author` field stays empty unless the entry has no email.
- **Account roles.** The project page's sidebar section titled "Maintainers" lists PyPI *accounts* holding the Owner or Maintainer role. These are set on pypi.org (Manage project → Collaborators), never in `pyproject.toml`, and are not exposed by the JSON API. `snet-cli` shows accounts `MatthewBehrend` and `SingularityNET` there while its `Maintainer` metadata field is empty — the sidebar is roles, not metadata.

### Evidence (PyPI JSON API, fetched 2026-08-13)

| Package | `Author` | `Author-email` | `Maintainer-email` |
|---|---|---|---|
| `snet-cli` 3.1.1 | *(empty)* | SingularityNET Foundation \<info@singularitynet.io\> | *(empty)* |
| `snet-sdk` 6.0.0 | *(empty)* | SingularityNET Foundation \<info@singularitynet.io\> | *(empty)* |
| `hyperon` 0.2.10 | *(empty)* | *(empty)* | *(empty)* |
| `jax` 0.11.0 | JAX team | jax-dev@google.com | *(empty)* |
| `flax` 0.12.8 | *(empty)* | Flax team \<flax-dev@google.com\> | *(empty)* |
| `optax` 0.2.8 | *(empty)* | Google DeepMind \<optax-dev@google.com\> | *(empty)* |
| `chex` 0.1.92 | *(empty)* | Google DeepMind \<chex-dev@google.com\> | *(empty)* |
| `torch` 2.13.0 | *(empty)* | PyTorch Team \<packages@pytorch.org\> | *(empty)* |
| `tensorflow` 2.21.0 | Google Inc. | packages@tensorflow.org | *(empty)* |
| `numpy` 2.5.2 | Travis E. Oliphant et al. | *(empty)* | NumPy Developers \<numpy-discussion@python.org\> |
| `scipy` 1.18.0 | *(empty)* | *(empty)* | SciPy Developers \<scipy-dev@python.org\> |

The convention: `authors` names the entity that created and owns the project — an org or team identity with an org contact address. The `maintainers` metadata field is used only where current stewardship is formally distinct from original authorship (numpy's historical author line vs its present collective; scipy, which drops `author` entirely). Individuals are named in the metadata of none of the org-published ML libraries.

### Decision

```toml
authors = [{name = "SingularityNET Foundation", email = "info@singularitynet.io"}]
# no `maintainers` — stewardship is not distinct from authorship
```

Matching `snet-cli` and `snet-sdk` exactly, and the Google/DeepMind/PyTorch pattern generally.

PyPI collaborators on the `fabricpc` project, set on pypi.org after the first upload: `MatthewBehrend` and `SingularityNET`, both as **Owner**. Owner rather than Maintainer for both, so either can manage collaborators and trusted publishers and neither account is a single point of failure — uploads themselves go through OIDC, so neither account needs upload rights. Both then appear in the page's "Maintainers" sidebar, as on `snet-cli`.

---

## 4. Implementation steps

### Step 1 — `pyproject.toml`

All items from §1's table:

```toml
[build-system]
requires = ["setuptools>=77"]

[project]
version = "0.4.0"
license = "MIT"                      # classifiers: remove "Private :: Do Not Upload"
authors = [{name = "SingularityNET Foundation", email = "info@singularitynet.io"}]

dependencies = [
    "jax>=0.7.0",
    "optax>=0.1.7",
    "orbax-checkpoint>=0.4.0",
    "flax>=0.7.5",
    "chex>=0.1.84",
    "jaxtyping>=0.2.23",
    "numpy>=1.24.0",
    "tqdm>=4.65.0",
]

[project.urls]
Homepage = "https://github.com/trueagi-io/FabricPC"
Repository = "https://github.com/trueagi-io/FabricPC"
Documentation = "https://github.com/trueagi-io/FabricPC/blob/main/docs/user_guides/00_index.md"
Changelog = "https://github.com/trueagi-io/FabricPC/blob/main/CHANGELOG.md"
Issues = "https://github.com/trueagi-io/FabricPC/issues"

[project.optional-dependencies]
experiments = ["scipy>=1.10.0", "optuna"]
dev = [..., "build", "twine"]
all = ["fabricpc[tfds,experiments,viz]"]
# [tool.setuptools] py-modules table deleted (§2)
```

The `[tool.ruff.lint.per-file-ignores]` E402 ignore for `examples/` and `scripts/` is removed. With the post-import contract, those scripts place all imports first and call `setup_jax()` after the import block, so no import sits below a statement and E402 passes without exceptions.

### Choosing the `jax` floor

A floor asserts that every API the library calls exists and behaves as specified at that version and above; its value is turning a runtime `AttributeError` in a training loop into a resolver error at install time. Three candidate methods, and why the third wins:

- **Oldest version whose API surface suffices.** The library's jax usage is deliberately conservative — `jax.Array`, `jax.random.PRNGKey/split/normal`, `jax.tree_util.tree_map`, `jax.value_and_grad`, `jax.jit`, `jax.vmap`, `jax.pmap`, `jax.lax.scan/fori_loop/top_k/pmean`, `jax.nn.one_hot/softmax/softplus`. Not `jax.tree.map` (0.4.25), not typed keys (0.4.16), not `shard_map`. This method yields a floor around 0.4.x, which is not a floor anyone should ship. It works for a leaf dependency used narrowly; it fails for a foundation dependency, where the risk is changed semantics rather than a missing symbol.
- **SPEC 0's 24-month window.** Puts the floor at jax 0.4.32 (2024-09). Calibrated for numpy/scipy cadence, far too wide for a library that removes APIs most minors.
- **Chosen — align with jax's own Python floor.** `jax>=0.7.0` (2025-07-22) is the oldest release declaring `requires-python >=3.11`, the same floor this project declares. Self-consistent, no invented number, about 13 months of supported releases.

`requires-python` is not a substitute. It bounds jax only from above — jax 0.11.0 requires `>=3.12`, so a Python 3.11 user is automatically capped at 0.10.2 — while going backwards it excludes nothing: jax 0.4.2 declares `>=3.8` and resolves on Python 3.11.

Verified, not asserted: with `jax==0.7.0` and `jaxlib==0.7.0` resolved alongside the current optax, flax, chex, and orbax (no backtracking needed), the suite runs 345 passed / 4 skipped — the same result as jax 0.10.1 in the same no-`[tfds]`/no-`[viz]` environment. `jax._src.xla_bridge.backends_are_initialized` exists at 0.7.0, so the deadline warning works there, and jaxlib 0.7.0's XLA accepts all three flags `setup_jax` writes. A `test.yml` leg pins this (§4).

### Step 2 — `jax_setup` relocation and re-contract (§2)

- `git mv jax_setup.py fabricpc/jax_config.py`; rewrite the function as sketched in §2 (rename to `setup_jax`, platform via env-write + `jax.config.update`, remaining env vars unchanged).
- Re-export `setup_jax` from `fabricpc/__init__.py` and add it to `__all__`.
- Migrate the 21 caller files listed in §2 (`from jax_setup import set_jax_flags_before_importing_jax` → `from fabricpc import setup_jax`); delete the `sys.path` workaround in `tests/test_doc_snippets.py:51-52`.
- New test `tests/test_jax_config.py`, run in a subprocess: `import jax` first, then `setup_jax("cpu")`, then assert every device in `jax.devices()` has `platform == "cpu"` and that `XLA_FLAGS` contains the deterministic-ops and Triton-GEMM entries. This pins the post-import contract that Option A depends on. Further cases pin the flag guards — values preset in `XLA_FLAGS` survive unchanged, a second `setup_jax()` call leaves `XLA_FLAGS` identical, and a longer flag sharing a prefix does not suppress the shorter one — and the deadline guard: a call placed after a computation warns and leaves `jax.devices()` unchanged, while the same call placed correctly is silent.

### Step 3 — README and installation docs

- README: new user-facing install block (`pip install fabricpc`, `pip install "fabricpc[all]"`, `pip install "fabricpc[all,cuda12]"` / `cuda13`); contributor block keeps clone + `pip install -e ".[all,dev]"`; all relative links converted to absolute `https://github.com/trueagi-io/FabricPC/blob/main/...` URLs; code snippet updated to `from fabricpc import setup_jax`. The README currently has no images; if any are added, they need absolute `raw.githubusercontent.com` URLs — PyPI does not render GitHub blob pages as images.
- Same updates in `docs/user_guides/01_installation.md`, `02_quickstart.md`, `16_troubleshooting.md`.

### Step 4 — CI

- `.github/workflows/publish.yml`:
  - `on: release: types: [published]` plus `workflow_dispatch` for the TestPyPI rehearsal. Workflow-level `permissions: contents: read` (the publish jobs widen to `id-token: write`), a `concurrency` group with `cancel-in-progress: false` so two dispatches cannot race an upload and no upload is interrupted mid-flight, and `timeout-minutes` on every job.
  - Job `build`: `python -m build`, `twine check dist/*`, upload `dist/` as artifact. On release triggers, a tag–version guard: fail unless the release tag equals `v` + the `Version:` in the built wheel's metadata. `importlib.metadata` keeps `__version__` consistent with `pyproject.toml`, but nothing ties the git tag to either; without the guard a `v0.4.1` release can publish a wheel that reports 0.4.0.
  - Job `smoke`: matrix over Python {3.11, 3.13} × backend extra {none, cpu, cuda12, cuda13}: install the built wheel with that extra into a clean environment, `pip check`, `python -c "import fabricpc; print(fabricpc.__version__)"`; the cpu leg additionally asserts every `jax.devices()` platform is `cpu`. The cuda legs download the multi-GB nvidia wheel set (acceptable at release cadence) and verify install + import only — GitHub-hosted runners have no GPU; the device-level GPU check lives in §5's backend install matrix.
  - Job `sdist`: `pip install dist/*.tar.gz` on one Python version, then import. `twine check` validates metadata and README rendering, not that the source distribution builds; the sdist is pure Python, so one leg covers it.
  - Job `publish-testpypi` (`workflow_dispatch` only): environment `testpypi`, `permissions: id-token: write`, `pypa/gh-action-pypi-publish@release/v1` with `repository-url: https://test.pypi.org/legacy/` and `skip-existing: true` (a re-rehearsal at an already-uploaded version verifies `build` + `smoke` and uploads nothing).
  - Both publish jobs reference `pypa/gh-action-pypi-publish@release/v1`, a branch in the pypa repository advanced to each 1.x release, kept as a moving ref by choice. Considered and rejected: pinning to a commit SHA. A SHA is immune to the branch being retargeted (`tj-actions/changed-files`, CVE-2025-30066, where every version tag was repointed at a secret-dumping commit across ~23,000 repositories), but only while someone bumps it; the action ships about 4 releases a year, and without Dependabot the pin goes stale and stops receiving the action's own fixes. A rotting pin is worse than a maintained moving ref. The deciding argument is scope: the `build` job installs `pip`, `build`, and `twine` unpinned from PyPI and produces the artifacts the publish job uploads verbatim, so pinning the action hardens the last mile and leaves the one before it open. Both are pypa code either way. If this is revisited, revisit it together with the build tooling, and add `.github/dependabot.yml` in the same change.
  - Job `publish` (release only): environment `pypi`, `permissions: id-token: write`, `pypa/gh-action-pypi-publish@release/v1`. No `skip-existing` — a duplicate production upload must fail loudly.
- `.github/workflows/test.yml`: pytest on ubuntu, CPU JAX, Python 3.11 and 3.13, on pull requests and pushes to `main`, with a per-ref `concurrency` group cancelling superseded runs. A third leg installs `jax==0.7.0` on Python 3.11 in the same pip invocation as the project, so one resolve settles jax against optax, flax, chex, and orbax together; this is what makes the declared floor a tested fact rather than an assertion.

### Step 5 — CHANGELOG

`## [0.4.0]` entry: first PyPI release; breaking change — `from jax_setup import set_jax_flags_before_importing_jax` becomes `from fabricpc import setup_jax`, callable any time before the first JAX computation; dependency changes (removed `jaxlib`; `optuna` moved to `[experiments]`); `[all]` no longer includes `[dev]`.

### One-time manual setup (repo admin, outside the codebase)

1. On pypi.org, signed in as `MatthewBehrend` or `SingularityNET`: account → Publishing → add a **pending publisher** for project name `fabricpc`: owner `trueagi-io`, repository `FabricPC`, workflow `publish.yml`, environment `pypi`. Pending publishers work for names that do not exist yet; the first trusted-publish claims the name, and the account that created the pending publisher becomes the project's sole initial Owner. Same on test.pypi.org with environment `testpypi`.
2. On GitHub `trueagi-io/FabricPC`: create environments `pypi` and `testpypi` (optionally with required reviewers as a release gate).

### Release procedure and post-publish hygiene

Moved to §6 (Deployment) — the step-by-step runbook for the TestPyPI rehearsal, its verification, the install test, the production release, and post-publish hygiene.

---

## 5. Verification

- `python -m build` produces sdist + `py3-none-any` wheel; `twine check dist/*` passes.
- Archive inspection: `unzip -l dist/*.whl` and `tar tzf dist/*.tar.gz` show `fabricpc/` only — no top-level `jax_setup`, no stray files.
- Clean-venv wheel install: `python -c "import fabricpc; print(fabricpc.__version__)"`; then the README snippet end-to-end (builds graph, initializes params).
- `setup_jax` contract test (Step 2) passes: platform selection and XLA flags take effect when called after `import jax`, before first computation.
- Bare-install boundary: without extras, `import fabricpc.tuning` raises `ModuleNotFoundError: optuna` and nothing else breaks.
- Metadata check: the built wheel's `METADATA` carries `Author-email: SingularityNET Foundation <info@singularitynet.io>`, `License-Expression: MIT`, and the five project URLs.
- Full pytest suite passes, including the migrated `test_doc_snippets.py`.
- Backend install matrix (below): the PyPI path and the clone path each verified for cpu, cuda12, and cuda13.
- Tag–version guard: the publish workflow fails on a release whose tag does not match the built wheel's version.

### Backend install matrix

Both install paths must work for each backend. Six cells, one clean venv each:

| | cpu | cuda12 | cuda13 |
|---|---|---|---|
| **PyPI wheel** | `pip install "fabricpc[cpu]"` | `pip install "fabricpc[cuda12]"` | `pip install "fabricpc[cuda13]"` |
| **Clone (editable)** | `git clone` + `pip install -e ".[all,cpu]"` | `pip install -e ".[all,cuda12]"` | `pip install -e ".[all,cuda13]"` |

Pass criteria per cell:

- Resolution and install complete; `pip check` reports no broken requirements.
- `import fabricpc` succeeds and reports the expected `__version__`.
- cpu column: `setup_jax()`, then every device in `jax.devices()` has `platform == "cpu"`. Runnable on any machine.
- cuda columns: the coupled wheel set is installed (`jax-cuda12-plugin` + `jax-cuda12-pjrt` + the nvidia-* wheels, or the cuda13 equivalents); on a host with a working NVIDIA driver, `jax.devices()[0].platform == "gpu"`. Install, `pip check`, and import run on any machine — without a driver, backend initialization falls back to CPU with a warning. The device check requires GPU hardware; this machine's driver is currently unloadable (§2), so run it on a GPU host or record the cell as install-verified only.

When each row runs:

- **Clone row:** on the packaging PR, before any release — it needs no published package.
- **PyPI row:** at the TestPyPI rehearsal (release step 3) using `-i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple` (jax's coupled wheel sets resolve from pypi.org via the extra index), and again from pypi.org after publishing (release step 5). The publish workflow's smoke job (Step 4) automates the install + import half of this row on every release.

---

## 6. Deployment

The release runbook. Prerequisites, all from §4: the packaging PR is merged to `main` (the `workflow_dispatch` trigger is only listed once `publish.yml` exists on the default branch), the Tests workflow is green on the release commit, and the one-time manual setup is done — pending publishers for `fabricpc` on test.pypi.org (environment `testpypi`) and pypi.org (environment `pypi`), and both GitHub environments created on `trueagi-io/FabricPC`.

The rehearsal (§6.1–6.3) is mandatory before the first production release: PyPI permanently reserves every (name, version, filename) tuple — even a deleted release cannot be re-uploaded — so a botched production upload burns 0.4.0 forever, while a TestPyPI mistake costs only a rehearsal version.

### 6.1 Test the publication on TestPyPI

The `workflow_dispatch` trigger runs the same `build` and `smoke` jobs as a release, then uploads to test.pypi.org instead of pypi.org. The tag–version guard is skipped on dispatch (there is no tag).

1. Confirm `version = "0.4.0"` in `pyproject.toml` on `main`.
2. Trigger the workflow:
   - CLI: `gh workflow run publish.yml --ref main`
   - UI: repository → Actions → Publish → Run workflow → branch `main`.
3. Watch the run (`gh run watch`, or the Actions UI). Jobs in order:
   - `build` — sdist + wheel, `twine check`, `dist/` artifact.
   - `smoke` — 8 legs: Python {3.11, 3.13} × extra {none, cpu, cuda12, cuda13}; each installs the built wheel into a clean runner, runs `pip check` and the import/version check; the cpu legs also assert every `jax.devices()` platform is `cpu`.
   - `publish-testpypi` — environment `testpypi`, OIDC upload. If the environment has required reviewers, the run pauses here; approve the deployment.
4. The first successful upload claims the name: the pending publisher on test.pypi.org becomes the `fabricpc` project's trusted publisher, with the creating account as sole Owner.

**Retry rule.** TestPyPI enforces the same filename reservation as PyPI; the upload step's `skip-existing: true` turns a duplicate into a no-op rather than a failure. A failure in `build` or `smoke` uploads nothing — fix and dispatch again with the same version. If the upload succeeded but §6.2–6.3 verification fails, a rebuilt 0.4.0 cannot replace the reserved files: a green re-dispatch at 0.4.0 verifies `build` and `smoke` but uploads nothing. Bump to `0.4.0rc1` (then `rc2`, …) for the re-rehearsal, and restore `0.4.0` before the production release. TestPyPI reservations do not touch pypi.org; 0.4.0 stays available there.

### 6.2 Verify the test publication

On https://test.pypi.org/project/fabricpc/:

- Version 0.4.0 is live with both files: the sdist and the `py3-none-any` wheel.
- The README renders with working links (all converted to absolute GitHub URLs in Step 3).
- Meta block: license `MIT`, author `SingularityNET Foundation <info@singularitynet.io>`, and the five `[project.urls]` entries (Homepage, Repository, Documentation, Changelog, Issues).

### 6.3 Test the install from TestPyPI

FabricPC's dependencies (jax, optax, …) do not exist on TestPyPI, so pypi.org is added as the extra index; the version pin keeps the resolver on the rehearsal artifact. Each check runs in its own clean venv.

```bash
python3.11 -m venv /tmp/fpc-rehearsal && source /tmp/fpc-rehearsal/bin/activate
pip install -U pip
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "fabricpc==0.4.0"
pip check
python -c "import fabricpc; print(fabricpc.__version__)"   # 0.4.0
python - <<'PY'
import jax
from fabricpc import setup_jax
setup_jax("cpu")
assert all(d.platform == "cpu" for d in jax.devices())
PY
python -c "import fabricpc.tuning"   # must raise ModuleNotFoundError: optuna
```

Then run the README model-building snippet end-to-end, and the PyPI row of §5's backend install matrix — `"fabricpc[cpu]==0.4.0"`, `"fabricpc[cuda12]==0.4.0"`, `"fabricpc[cuda13]==0.4.0"`, each in its own clean venv with the same two index flags. Pass criteria per §5; the cuda device check needs a GPU host, otherwise record those cells as install-verified only.

### 6.4 Publish the release to pypi.org

1. Confirm §6.1–6.3 passed, `pyproject.toml` on `main` reads `0.4.0`, and `CHANGELOG.md` carries the `[0.4.0]` entry.
2. Write the release notes (the CHANGELOG entry) to a temporary file in the project root, then create the GitHub release. The tag must be exactly `v0.4.0`: the `build` job fails the run on any mismatch with the built wheel's `Version`.
   - CLI: `gh release create v0.4.0 --target main --title "v0.4.0" --notes-file release_notes_v0.4.0.md`
   - UI: Releases → Draft a new release → tag `v0.4.0` on `main` → Publish release.
3. Publishing the release fires `publish.yml` on the release trigger: `build` (now with the tag–version guard), `smoke`, then `publish` (environment `pypi`, OIDC). Approve the environment deployment if reviewers are configured.
4. A failure before upload burns nothing: `publish` needs both `build` and `smoke`, so a red run leaves pypi.org untouched. Fix, delete the release and tag (`gh release delete v0.4.0 --cleanup-tag`), and re-create the release with the same version.
5. As on TestPyPI, the first upload converts the pending publisher into the `fabricpc` project with the creating account as sole Owner.

### 6.5 Verify the production release

- https://pypi.org/project/fabricpc/ renders the README, license, and links; sdist and wheel both present.
- In a clean venv with no index flags: `pip install "fabricpc==0.4.0"`, then `pip check`, the import/version check, the README snippet, and the PyPI row of §5's backend install matrix against pypi.org.

### 6.6 Post-publish hygiene

- Immediately after the first upload, add the second account as Owner (§3): a single-owner project is one lost account from unmaintainable, and PyPI requires a second owner before some project-scoped settings can be changed. PyPI mandates 2FA — store the recovery codes in the org's shared secret storage, not on one person's device.
- Bad release: **yank, never delete**. Yanking removes the version from default resolution while keeping it installable for anyone who already pinned it; deletion breaks those installs and still does not free the version for re-upload. Follow a yank with a fixed 0.4.1 release.

---

## Alternatives considered (summary)

- **`jax_setup` placement and contract:** Options A–D in §2. A (move + post-import-safe contract) chosen, grounded in the empirical import-timing tests and the eager-`__init__` convention of all JAX-ecosystem peers. B (lazy `__init__`) rejected — no ecosystem precedent, and the constraint it preserves is unnecessary. C/D rejected for namespace pollution / loss of the helper.
- **Metadata identity:** org authorship (chosen, §3) vs `authors = [{name = "FabricPC Team", email = "info@singularitynet.io"}]` (the jax/flax team-name variant of the same convention) vs individual author + org maintainer (matches the LICENSE copyright line, but no org-published ML library names an individual) vs omitting both, as `hyperon` does.
- **Unused core dependencies:** keeping `flax`, `chex`, `jaxtyping`, `orbax-checkpoint` (chosen) vs removing them until a feature imports them (rejected: the removal would be reversed as soon as checkpointing or typing work lands, and each round trip is a dependency bump and a release).
- **Publish mechanism:** Trusted Publishing CI (chosen) vs manual `twine` upload with an API token (rejected: long-lived token management, hand-run releases) vs manual-first-then-CI (unnecessary — pending publishers let CI claim the unclaimed name directly).
- **Version:** 0.4.0 (chosen: `jax_setup` breaking change + first public release) vs 0.3.3 vs republishing 0.3.2 (rejected: artifact would not match the 0.3.2 changelog entry).
- **Version single-sourcing:** static `project.version` + CI tag–version guard (chosen) vs deriving the version from the git tag with setuptools-scm (rejected: replaces a one-line CI assertion with build-backend machinery and dev-version noise on untagged checkouts).
