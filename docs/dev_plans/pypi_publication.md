# PyPI publication: repository preparation and release of `fabricpc`

## Context

FabricPC is installable today only by cloning the repo. Publishing to pypi.org makes `pip install fabricpc` work for external users. The name `fabricpc` is unclaimed on both pypi.org and test.pypi.org (verified 2026-07-30, both return 404). This plan covers the configuration fixes required before the first upload, the release automation, and the release procedure itself.

**Settled decisions** (from CLI dialogue):
- Release mechanism: GitHub Actions with PyPI Trusted Publishing (OIDC), triggered by a GitHub release.
- First published version: **0.4.0**, cut from `main` after the pending muPC output-scaling branch merges.

**Open decisions — need your answer** (details in §2 and §3):
1. `jax_setup.py` relocation — Option A recommended: move to `fabricpc/jax_setup.py`, keep the eager `__init__`, and re-contract the helper to "call before the first JAX computation" using JAX's post-import configuration API. Lazy module imports were evaluated against jax/flax/optax/torch practice and rejected (§2, Option B).
2. Author/maintainer metadata — conventions surveyed across org-published and community ML libraries in §3.
3. Defaults I will apply unless you object: remove four never-imported core dependencies (`flax`, `chex`, `jaxtyping`, `orbax-checkpoint`); move `optuna` to the `[experiments]` extra; set floor `jax>=0.10`; drop `dev` from the `[all]` extra; add a minimal pytest CI workflow.

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
| 5 | `dependencies` includes `flax`, `chex`, `jaxtyping`, `orbax-checkpoint` | Never imported anywhere in `fabricpc/`, `examples/`, `tests/`, or `scripts/` (verified by grep). Every user would download and install four unused packages. | Remove all four. Re-add individually when a feature actually imports them. |
| 6 | `optuna` in core `dependencies` | Imported only by `fabricpc/tuning/bayesian_tuner.py`; `fabricpc/__init__.py` does not import `tuning`. Core users pay optuna's install cost for a subpackage they never touch. | Move to `[experiments]` (the tuning guide already lives in `15_api_experiments.md`). Document that `fabricpc.tuning` requires the extra. |
| 7 | `jax` has no version floor | A pip resolver could select an ancient jax. Tested version is 0.10.2. | `jax>=0.10`. |
| 8 | `[tool.setuptools] py-modules = ["jax_setup"]` | Ships a generically named top-level module into every user's site-packages (§2). | Per §2 decision. |
| 9 | `all = ["fabricpc[dev,tfds,experiments,viz]"]` | `pip install "fabricpc[all]"` would give end users black, mypy, ruff, and pre-commit. | `all = ["fabricpc[tfds,experiments,viz]"]`; contributor install becomes `pip install -e ".[all,dev]"`. README and `01_installation.md` updated accordingly. |
| 10 | `authors = [{name = "FabricPC Authors"}]`, no email | Generic, inconsistent with LICENSE (Matthew Behrend) and README (SingularityNET). | Per §3 decision. |
| 11 | README as PyPI landing page | Relative links (`docs/user_guides/...`, `examples/...`, `LICENSE`) break on pypi.org, which renders the README standalone. Installation section says "Clone this repo" only. | Convert relative links to absolute GitHub URLs; add a `pip install fabricpc` section for users; keep the clone + editable install path for contributors. |
| 12 | CI has only `lint.yml` | No test gate and no publish pipeline. | Add `publish.yml` (Trusted Publishing) and a minimal `test.yml` (§4). |
| 13 | `[dev]` extra lacks release tooling | `build` and `twine` are not installed anywhere; local verification of the distribution is currently impossible. | Add `build` and `twine` to `[dev]`. |

Kept as-is: `Development Status :: 3 - Alpha` (README states APIs may change until v1.0), Python floor `>=3.10`, classifiers through 3.13.

---

## 2. Open decision 1: where `jax_setup.py` lives, and what its contract is

### Why the current setup works, and what actually constrains it

Your concern was that `jax_setup.py` "only works properly at the project level folder." The file's *location* is not what makes it work — the function only writes `os.environ` entries and imports nothing but `os`. Two separate facts create the current behavior:

- **Why it is importable today:** `pip install -e .` installs it as a top-level module (via `py-modules = ["jax_setup"]`). That is how `examples/mnist_demo.py` can do `from jax_setup import ...` even though the script's own directory is `examples/`, not the repo root. The repo-root location is an artifact of packaging, not a functional requirement.
- **The real constraint is temporal, not spatial** — and narrower than the function's name claims. Measured against the repo's jax 0.10.2 (CPU backend):

| Test | Action | Result |
|---|---|---|
| A | `import jax`, then set `JAX_PLATFORMS=bogus_platform` in `os.environ`, then `jax.devices()` | Silently ignored — devices resolve normally. The env var is captured once, at `import jax`. |
| B | `import jax`, then `jax.config.update("jax_platforms", "cpu")`, then `jax.devices()` | Platform applied. This is JAX's documented post-import platform API. |
| C | `import jax`, then set `XLA_FLAGS=--xla_force_host_platform_device_count=4`, then `jax.devices()` | 4 CPU devices — `XLA_FLAGS` is consumed at backend initialization, not at import. `XLA_PYTHON_CLIENT_PREALLOCATE` is read in the same phase (GPU client creation). |

So only the `JAX_PLATFORMS` *environment-variable* path requires pre-import ordering, and JAX provides `jax.config.update("jax_platforms", ...)` precisely so platform selection works post-import. Everything else the helper sets is read at backend initialization, which JAX defers until the first computation or device query. (Caveat: this machine's GPU driver is currently unloadable, so Test B could not demonstrate CPU-vs-GPU selection; Tests A and C are the discriminating results, and the config API is the documented mechanism.)

### How peer libraries structure their package import (evaluated per your request)

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

### Option A (recommended): move to `fabricpc/jax_setup.py`, keep the eager `__init__`, re-contract the helper

The helper's "before importing jax" contract was always stricter than JAX requires (Tests A–C). Rewrite it to be correct when called *after* `import jax`, as long as it runs before the first JAX computation:

```python
# fabricpc/jax_setup.py
def setup_jax(platform: str | None = None) -> None:
    """Configure JAX for FabricPC. Call before the first JAX computation
    (backend initialization). Import order no longer matters."""
    import jax  # the settings below are consumed at backend init, not at import

    if platform is not None and not os.environ.get("JAX_PLATFORMS"):
        os.environ["JAX_PLATFORMS"] = platform          # visible to worker subprocesses
        jax.config.update("jax_platforms", platform)    # effective in this process (Test B)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # ... XLA_FLAGS assembly unchanged (Test C: consumed at backend init) ...
```

The `not os.environ.get("JAX_PLATFORMS")` guard preserves the current setdefault semantics: a platform set in the user's shell wins over the function argument, exactly as today. The env write plus `config.update` pair is one deterministic path — the env var covers spawned worker processes, the config call covers the already-imported current process.

- **Rename:** `set_jax_flags_before_importing_jax` → `setup_jax`. The old name states a constraint that no longer exists; keeping it would document a false requirement. All callers are being migrated anyway.
- **Pros:** matches the eager-`__init__` convention of every JAX-ecosystem peer; no top-level namespace pollution in site-packages; the contract is enforceable by a real test (call after `import jax`, assert the platform took effect); small diff — one function rewrite plus mechanical caller migration.
- **Cons:** relies on `jax.config.update("jax_platforms", ...)` (public, documented) and on XLA flags being read at client creation (verified empirically; stable XLA behavior). Callers should still invoke it early — after any JAX computation it is too late, same as today.
- **Clone workflow: unchanged.** `git clone` + `pip install -e ".[all]"` works as before; only the import line changes.
- **Migration scope (all callers, one mechanical line each, no compatibility shim):** 15 files in `examples/`, 2 in `scripts/`, `README.md`, `docs/user_guides/02_quickstart.md`, `docs/user_guides/16_troubleshooting.md`, and `tests/test_doc_snippets.py` (also delete its `sys.path.insert(REPO_ROOT)` workaround at lines 51–52, which exists only to reach the root-level module). Breaking change documented in CHANGELOG with the one-line migration.

### Option B: move into `fabricpc/`, make `fabricpc/__init__.py` lazy (PEP 562), keep the pre-import contract

Convert the package `__init__` to `__getattr__`-based lazy loading so `from fabricpc.jax_setup import ...` does not pull in jax, keeping the literal "before importing jax" contract.

- **Pros:** contract stays literally true; `import fabricpc` becomes cheap.
- **Cons:** no precedent among JAX-ecosystem peers (table above — jax, flax, optax, chex, torch are all eager); the lazy mechanisms that do exist (scipy, tensorflow) solve import *cost* at a much larger scale, not configuration ordering; it couples the package's import architecture to the needs of one helper function; error timing moves from import to first attribute access; static-analysis names must be mirrored in a `TYPE_CHECKING` block and maintained. Rejected: Option A dissolves the constraint that was the only reason to consider this.

### Option C: status quo (top-level `jax_setup` module in the wheel)

- **Pros:** zero work.
- **Cons:** every `pip install fabricpc` drops a module named `jax_setup` at the top of site-packages. Any other package shipping the same generic name collides silently (last-installed wins). Permanent once external users import it.

### Option D: repo-only file, excluded from the wheel

- **Pros:** no namespace pollution.
- **Cons:** installed users lose the documented configuration helper entirely; examples rely on the module being installed, so the editable install breaks too.

---

## 3. Open decision 2: author/maintainer metadata

### Evidence (fetched 2026-07-30 from the PyPI JSON API)

Org-published ML libraries — the org or team is the `author`, with a team contact address; the `maintainer` field is left empty:

| Package | Publisher | Author field on PyPI | Maintainer field |
|---|---|---|---|
| `jax` | Google | JAX team \<jax-dev@google.com\> | *(empty)* |
| `flax` | Google | Flax team \<flax-dev@google.com\> | *(empty)* |
| `optax` | DeepMind | Google DeepMind \<optax-dev@google.com\> | *(empty)* |
| `chex` | DeepMind | Google DeepMind \<chex-dev@google.com\> | *(empty)* |
| `torch` | Meta/Linux Fdn | PyTorch Team \<packages@pytorch.org\> | *(empty)* |
| `tensorflow` | Google | Google Inc. \<packages@tensorflow.org\> | *(empty)* |

Community-governed scientific libraries — the collective goes in `maintainer`; `author` is kept only for historical attribution or omitted:

| Package | Author field | Maintainer field |
|---|---|---|
| `numpy` | Travis E. Oliphant et al. | NumPy Developers \<numpy-discussion@python.org\> |
| `scipy` | *(empty)* | SciPy Developers \<scipy-dev@python.org\> |

ASI Alliance packages:

| Package | GitHub org | Author field | Maintainer field |
|---|---|---|---|
| `snet-cli` 3.1.1 | singnet | SingularityNET Foundation \<info@singularitynet.io\> | *(empty)* |
| `snet-sdk` 6.0.0 | singnet | SingularityNET Foundation \<info@singularitynet.io\> | *(empty)* |
| `hyperon` 0.2.10 | trueagi-io (same org as FabricPC) | *(empty)* | *(empty)* |

### The convention

`authors` names the entity that created and owns the project — for org-published libraries that is a team or org identity with a team mailing address, never an individual. `maintainers` is used only when current stewardship is distinct from original authorship (numpy's historical author vs its current collective). Individual names appear in the metadata of none of the major ML libraries.

### Options

- **3a (org convention, recommended):** `authors = [{name = "SingularityNET Foundation", email = "info@singularitynet.io"}]` — matches both the singnet packages and the Google/DeepMind/PyTorch pattern (owning org + org contact).
- **3b (team identity):** `authors = [{name = "FabricPC Team", email = "info@singularitynet.io"}]` — the jax/flax variant of the same convention: project-team name, org address.
- **3c (lead + org):** `authors = [{name = "Matthew Behrend"}]`, `maintainers = [{name = "SingularityNET Foundation", email = "info@singularitynet.io"}]` — matches the LICENSE copyright line, but no major ML library names an individual in its published metadata.
- **3d (none):** omit both, like `hyperon`.

Recommendation: **3a** or **3b** — identical in convention; pick by whether the public identity should read as the foundation or the project team.

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
authors = [...]                      # per §3 decision

dependencies = [
    "jax>=0.10",
    "numpy>=1.24.0",
    "optax>=0.1.7",
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
# [tool.setuptools] py-modules table deleted (per §2 decision)
```

### Step 2 — `jax_setup` relocation and re-contract (per §2 Option A)

- `git mv jax_setup.py fabricpc/jax_setup.py`; rewrite the function as sketched in §2 (rename to `setup_jax`, platform via env-write + `jax.config.update`, remaining env vars unchanged).
- Migrate the 21 caller files listed in §2 (`from jax_setup import set_jax_flags_before_importing_jax` → `from fabricpc.jax_setup import setup_jax`); delete the `sys.path` workaround in `tests/test_doc_snippets.py:51-52`.
- New test `tests/test_jax_setup.py`, run in a subprocess: `import jax` first, then `setup_jax("cpu")`, then assert every device in `jax.devices()` has `platform == "cpu"` and that `XLA_FLAGS` contains the deterministic-ops and Triton-GEMM entries. This pins the post-import contract that Option A depends on.

### Step 3 — README and installation docs

- README: new user-facing install block (`pip install fabricpc`, `pip install "fabricpc[all]"`, `pip install "fabricpc[all,cuda12]"` / `cuda13`); contributor block keeps clone + `pip install -e ".[all,dev]"`; all relative links converted to absolute `https://github.com/trueagi-io/FabricPC/blob/main/...` URLs; code snippet updated to `from fabricpc.jax_setup import setup_jax`. The README currently has no images; if any are added, they need absolute `raw.githubusercontent.com` URLs — PyPI does not render GitHub blob pages as images.
- Same updates in `docs/user_guides/01_installation.md`, `02_quickstart.md`, `16_troubleshooting.md`.

### Step 4 — CI

- `.github/workflows/publish.yml`:
  - `on: release: types: [published]` plus `workflow_dispatch` for the TestPyPI rehearsal.
  - Job `build`: `python -m build`, `twine check dist/*`, upload `dist/` as artifact. On release triggers, a tag–version guard: fail unless the release tag equals `v` + the `Version:` in the built wheel's metadata. `importlib.metadata` keeps `__version__` consistent with `pyproject.toml`, but nothing ties the git tag to either; without the guard a `v0.4.1` release can publish a wheel that reports 0.4.0.
  - Job `smoke`: install the built wheel into a clean environment on Python 3.10 and 3.13, run `python -c "import fabricpc; print(fabricpc.__version__)"`.
  - Job `publish-testpypi` (`workflow_dispatch` only): environment `testpypi`, `permissions: id-token: write`, `pypa/gh-action-pypi-publish@release/v1` with `repository-url: https://test.pypi.org/legacy/`.
  - Job `publish` (release only): environment `pypi`, `permissions: id-token: write`, `pypa/gh-action-pypi-publish@release/v1`.
- `.github/workflows/test.yml`: pytest on ubuntu, CPU JAX, Python 3.10 and 3.13, on push/PR.

### Step 5 — CHANGELOG

`## [0.4.0]` entry: first PyPI release; breaking change — `from jax_setup import set_jax_flags_before_importing_jax` becomes `from fabricpc.jax_setup import setup_jax`, callable any time before the first JAX computation; dependency slimming (removed `jaxlib`/`flax`/`chex`/`jaxtyping`/`orbax-checkpoint`, `optuna` moved to `[experiments]`); `[all]` no longer includes `[dev]`.

### One-time manual setup (repo admin, outside the codebase)

1. On pypi.org: account → Publishing → add a **pending publisher** for project name `fabricpc`: owner `trueagi-io`, repository `FabricPC`, workflow `publish.yml`, environment `pypi`. Pending publishers work for names that do not exist yet; the first trusted-publish claims the name. Same on test.pypi.org with environment `testpypi`.
2. On GitHub `trueagi-io/FabricPC`: create environments `pypi` and `testpypi` (optionally with required reviewers as a release gate).

### Release procedure

1. Merge the pending muPC output-scaling branch to `main`.
2. Open the packaging PR (Steps 1–5) against `main`; test suite green; merge.
3. Rehearse: run `publish.yml` via `workflow_dispatch` → TestPyPI; in a clean venv, `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fabricpc` and smoke-test; repeat with `"fabricpc[cpu]"` to confirm backend extras resolve (`jax[cpu]` comes from pypi.org via the extra index). The rehearsal is mandatory: PyPI permanently reserves every (name, version, filename) tuple — even a deleted release cannot be re-uploaded — so a botched production upload burns 0.4.0 forever, while TestPyPI mistakes cost nothing.
4. Create GitHub release `v0.4.0` on `main` → workflow publishes to pypi.org.
5. Post-publish check: pypi.org/project/fabricpc renders README, license, and links correctly; `pip install fabricpc` in a clean venv; run the README model-building snippet.

### Post-publish hygiene

- Immediately after the first upload, add a second owner on the pypi.org project: a single-owner project is one lost account from unmaintainable, and PyPI requires a second owner before some project-scoped settings can be changed. PyPI mandates 2FA — store the recovery codes in the org's shared secret storage, not on one person's device.
- Bad release: **yank, never delete**. Yanking removes the version from default resolution while keeping it installable for anyone who already pinned it; deletion breaks those installs and still does not free the version for re-upload.

---

## 5. Verification

- `python -m build` produces sdist + `py3-none-any` wheel; `twine check dist/*` passes.
- Archive inspection: `unzip -l dist/*.whl` and `tar tzf dist/*.tar.gz` show `fabricpc/` only — no top-level `jax_setup`, no stray files.
- Clean-venv wheel install: `python -c "import fabricpc; print(fabricpc.__version__)"`; then the README snippet end-to-end (builds graph, initializes params).
- `setup_jax` contract test (Step 2) passes: platform selection and XLA flags take effect when called after `import jax`, before first computation.
- Bare-install boundary: without extras, `import fabricpc.tuning` raises `ModuleNotFoundError: optuna` and nothing else breaks.
- Full pytest suite passes, including the migrated `test_doc_snippets.py`.
- TestPyPI rehearsal install (release procedure step 3), including the `[cpu]` backend-extra resolution check, before the real release.
- Tag–version guard: the publish workflow fails on a release whose tag does not match the built wheel's version.

## Alternatives considered (summary)

- **`jax_setup` placement and contract:** Options A–D in §2. A (move + post-import-safe contract) recommended, grounded in the empirical import-timing tests and the eager-`__init__` convention of all JAX-ecosystem peers. B (lazy `__init__`) rejected — no ecosystem precedent, and the constraint it preserves is unnecessary. C/D rejected for namespace pollution / loss of the helper.
- **Publish mechanism:** Trusted Publishing CI (chosen) vs manual `twine` upload with an API token (rejected: long-lived token management, hand-run releases) vs manual-first-then-CI (unnecessary — pending publishers let CI claim the unclaimed name directly).
- **Version:** 0.4.0 (chosen: `jax_setup` breaking change + first public release) vs 0.3.3 vs republishing 0.3.2 (rejected: artifact would not match the 0.3.2 changelog entry).
- **Version single-sourcing:** static `project.version` + CI tag–version guard (chosen) vs deriving the version from the git tag with setuptools-scm (rejected: replaces a one-line CI assertion with build-backend machinery and dev-version noise on untagged checkouts).
