# Modernize FabricPC Install for Python 3.13 + CUDA 13 + JAX 0.10 (Completed; 3.14 deferred)

> **Status:** Completed. The implementation extended support to Python
> 3.13 and CUDA 13. Python 3.14 was attempted during development (see
> the empirical log in `virt-env/fpc3`, Python 3.14.4) and ultimately
> **deferred** because TensorFlow has no `cp314` wheels, leaving
> `[tfds]` unresolvable. See "Scope reduction: drop Python 3.14
> support" further down for the deferral decision and what remains
> blocking. `requires-python` in `pyproject.toml` is now capped at
> `<3.14`; this doc retains the 3.14 exploration as historical record.

## Summary

Bring FabricPC's default install path up to date so it works on a current
Fedora 43 / CUDA 13 / JAX 0.10 machine (Python 3.13) without requiring
users to pin to older Python or older CUDA toolkits. The pre-change
README told users to create a Python 3.12.x venv (because Aim does not
support 3.13+); the `cuda` extra installed `jax[cuda12]` even though JAX
0.10 ships a first-class `cuda13` extra; and `pyproject.toml` only
advertised 3.10–3.12 in its classifiers and `black` `target-version`.

This plan tracks the changes needed to modernize the install
instructions and `pyproject.toml`, the version pins we settled on
after empirical testing, and which extras had to be deferred because
their upstream wheels lag newer Python (`aim` on 3.13, `tensorflow` on
3.14 — the latter is the reason 3.14 itself was deferred).

---

## Reference Environment

This plan is being driven and validated against:

- **OS**: Fedora 43 (kernel 6.19.x)
- **Python**: 3.14.4 (`virt-env/fpc3/bin/python`)
- **pip**: 25.1.1
- **GPU**: NVIDIA GeForce RTX 5060 (Blackwell, `sm_120`)
- **NVIDIA driver**: 595.71.05 (CUDA 13.2 capability)
- **CUDA toolkit**: 13.2.1 (`/usr/local/cuda`)
- **cuDNN**: not installed system-wide; will come via the `nvidia-cudnn-cu13` PyPI wheel pulled in by `jax[cuda13]`.
- **gcc**: 15.2.1

---

## Findings From Upstream PyPI (as of 2026-05)

| Package | Latest | Has cp314 wheel? | Notes |
|---|---|---|---|
| `jax` | 0.10.0 | yes (pure Python) | Requires Python ≥3.11; pulls `numpy>=2.0`, `scipy>=1.14`. Provides `cuda12`, `cuda13`, `cuda12-local`, `cuda13-local` extras. |
| `jaxlib` | 0.10.0 | yes (`cp314-cp314-manylinux_2_27_x86_64`) | — |
| `jax-cuda13-plugin` | 0.10.0 | yes | Pulls `nvidia-cudnn-cu13`, `nvidia-cuda-runtime-cu13`, etc. when used with `[with-cuda]`. |
| `optax` | 0.2.8 | yes | — |
| `orbax-checkpoint` | 0.11.39 | yes | — |
| `flax` | 0.12.7 | yes | — |
| `chex` | 0.1.91 | yes | — |
| `jaxtyping` | 0.3.9 | yes | — |
| `numpy` | 2.4.4 | yes | JAX 0.10 forces ≥2.0. |
| `scipy` | 1.17.1 | yes | JAX 0.10 forces ≥1.14. |
| `tqdm` | 4.67.3 | yes | — |
| `optuna` | 4.8.0 | yes | — |
| `plotly` | 6.7.0 | yes | — |
| `kaleido` | 1.3.0 | yes | — |
| `pandas` | 3.0.2 | yes | — |
| `pytest` | 9.0.3 | yes | — |
| `hypothesis` | 6.152.4 | yes | — |
| `mypy` | 2.0.0 | yes | — |
| `pre-commit` | 4.6.0 | yes | — |
| `black` | 26.3.1 | yes | `pyproject.toml` currently pins exactly `26.1.0`; relax to `>=26.1.0`. |
| **Deferred** | | | |
| `aim` | 3.29.1 | no (Python <3.13) | Skip on this machine; gated behind a separate extra. |
| `tensorflow` | n/a for cp314 | **no** | TF has no cp314 wheels yet → `[tfds]` extra is unusable on Python 3.14. |
| `tensorflow-datasets` | 4.9.10 | yes by itself | But it requires TF, so blocked transitively. |

---

## Plan of Record

**Backwards-compatibility constraint:** every command that works for
existing users on Python 3.10–3.12 / CUDA 12 must continue to work
unchanged. New-platform support is added by *expanding* the extras
matrix and adding environment markers, never by repurposing or
narrowing existing extra names.

### 1. Add a `cuda13` extra; do not change `cuda` or `cuda12`

`pyproject.toml` currently:

```toml
cuda = [ "jax[cuda12]" ]
```

Proposed:

```toml
cuda   = [ "jax[cuda12]>=0.10.0" ]   # unchanged behavior; just floored to a known-good version
cuda12 = [ "jax[cuda12]>=0.10.0" ]   # explicit alias
cuda13 = [ "jax[cuda13]>=0.10.0" ]   # NEW: opt-in for CUDA 13 systems
```

Rationale: CUDA 12 remains the default so existing `pip install -e
".[cuda]"` and `pip install -e ".[all]"` invocations on CUDA-12 boxes
continue to install a working stack. Modern users (Fedora 43, CUDA 13,
Blackwell GPUs) explicitly opt into `[cuda13]`. The README will steer
new users on CUDA-13 systems toward the new extra.

The minor floor `>=0.10.0` is added because (a) it's the version
validated here, (b) older JAX versions had subtly different
`jax[cuda12]` resolution semantics. If older JAX is still required
somewhere in the wild this floor can be relaxed; it is not strictly
necessary for backwards compat.

### 2. Make `[viz]` and `[tfds]` self-degrade on too-new Python via env markers

Today `[viz]` pulls `aim` (no Python 3.13+ wheel) and `[tfds]` pulls
`tensorflow` (no Python 3.14 wheel). Add environment markers so the
extras keep installing the same things on the platforms where they
were already working, and silently skip the unavailable bits on newer
Python:

```toml
viz = [
  "plotly>=5.0.0",
  "kaleido>=0.2.1",
  "pandas>=2.0.0",
  "aim>=3.0.0; python_version < '3.13'",   # was unconditional
]
tfds = [
  "tensorflow-datasets>=4.9.0; python_version < '3.14'",
  "tensorflow>=2.15.0; python_version < '3.14'",
]
all = [
  "fabricpc[dev,tfds,experiments,viz,cuda]",   # unchanged shape
]
```

Behavior matrix:

| User Python | `pip install -e ".[viz]"` | `pip install -e ".[tfds]"` | `pip install -e ".[all]"` |
|---|---|---|---|
| 3.10–3.12 | plotly+kaleido+pandas+**aim** | tensorflow-datasets+**tensorflow** | full set incl. aim+TF (same as today) |
| 3.13      | plotly+kaleido+pandas (no aim) | tensorflow-datasets+tensorflow | as today minus aim |
| 3.14      | plotly+kaleido+pandas (no aim) | empty (no TF wheel) | as today minus aim+TF |

No existing user loses anything. New-platform users get a clean
install without manual workarounds. This *is* a behavior change in
the silent-skip sense, but it strictly increases the set of platforms
where the install succeeds.

### 3. Update Python classifiers and tooling — additively

In `[project]`:
- **Keep** `requires-python = ">=3.10"` (do not bump to 3.11). Existing
  3.10 users install fine because pip resolves an older `jax` for
  them; bumping the floor would lock them out without cause.
- **Add** `Programming Language :: Python :: 3.13` and `:: 3.14`
  classifiers.

In `[tool.black]`:
- **Add** `py313` and `py314` to `target-version` (keep `py310`,
  `py311`, `py312`).
- **Relax** the dev pin from `black==26.1.0` to `black>=26.1.0` so
  users get latest fixes (26.3.1 at time of writing).

### 4. README quick-start — additive, not replacement

Keep the existing "Python 3.12.x" recipe so anyone copy-pasting it
gets a known-good config. **Add** sections for newer platforms:

```
## Quick Start

The recipe below works on Python 3.10–3.12 with CUDA 12 GPUs (legacy
default) and is the safest copy-paste:

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[all]"
    pre-commit install
    aim up                 # optional Aim server (Python <=3.12 only)
    python examples/mnist_demo.py

### Modern platforms (Python 3.13+ or CUDA 13)

On Python 3.13+ the Aim experiment-tracking server is not yet
supported upstream and will be skipped by `[all]`.

On Python 3.14 TensorFlow has no wheels yet, so `[tfds]` is also a
no-op.

For a CUDA 13 system (e.g. Fedora 43 with the cuda-toolkit-13-2
package and a 5xx-series driver), use the `[cuda13]` extra instead of
the default `[cuda]`:

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev,experiments,viz,cuda13]"

This installs JAX 0.10 with the matching `jax-cuda13-plugin` and
`nvidia-cudnn-cu13` wheels. ~2 GB of CUDA 13 wheels are downloaded
on first install.
```

Existing CUDA-12 / Python 3.12 users see *no change* in instructions.
New-platform users have a clearly labeled second recipe.

### 5. `jax_setup.py`

No code changes expected. The XLA flags it sets are CUDA-version
agnostic. Verify after install that
`set_jax_flags_before_importing_jax("cuda")` still resolves to the
`jax-cuda13-plugin`-backed PJRT.

### 6. (Optional) System cuDNN hint

JAX's `cuda13` extra pulls `nvidia-cudnn-cu13` from PyPI, so a system
cuDNN is not required. If a user sets `jax[cuda13-local]` they need a
system cuDNN matching CUDA 13.2; document this as a footnote.

---

## Validation Steps (in `virt-env/fpc3`)

1. `pip install --upgrade pip` — current is 25.1.1, latest 26.1.1.
2. Install core only: `pip install -e .` — confirms the base
   `dependencies` block resolves on Python 3.14.
3. `pip install -e ".[dev]"`
4. `pip install -e ".[experiments]"`
5. `pip install -e ".[viz]"` (post-restructure; should not pull aim).
6. `pip install -e ".[cuda13]"` and run a one-liner JAX GPU smoke test
   (`jax.devices()` shows `cuda(id=0)` on the RTX 5060).
7. Skip `[aim]` and `[tfds]` on this machine; document the skip.
8. `pytest -q` against the test suite.

Each step's outcome is recorded inline below as we go.

---

## Empirical Log

### 2026-05-09 — Survey
- Plan drafted. CUDA 13.2.1 toolkit confirmed, JAX 0.10.0 cp314 wheel
  confirmed, `jax[cuda13]` extra confirmed. Aim and TF deferred.

### 2026-05-09 — Step-through install in `virt-env/fpc3` (Python 3.14.4)

1. **`pip install --upgrade pip`** → 25.1.1 → 26.1.1. Clean.
2. **`pip install -e .`** (base `dependencies` block) → clean.
   Key resolved versions: `jax 0.10.0`, `jaxlib 0.10.0`, `numpy 2.4.4`,
   `scipy 1.17.1`, `optax 0.2.8`, `orbax-checkpoint 0.11.39`,
   `flax 0.12.7`, `chex 0.1.91`, `jaxtyping 0.3.9`, `tqdm 4.67.3`,
   `optuna 4.8.0`. No version pin in `dependencies` had to be loosened
   to make this resolve on Python 3.14 — every floor is already
   compatible.
3. **`pip install -e ".[dev]"`** → clean. `black==26.1.0` (the exact
   pin) installs fine on 3.14, but the latest is 26.3.1 — the pin
   should be relaxed to `>=26.1.0` per Plan §3 so users get bug fixes.
4. **Skipped `[viz]`** as defined today (because it transitively pulls
   `aim`, which has no Python 3.13/3.14 wheels). Installed
   `plotly 6.7.0`, `kaleido 1.3.0`, `pandas 3.0.2` directly to validate
   the would-be-modernized `[viz]` group. Clean.
5. **Skipped `[aim]` and `[tfds]`** entirely. `tensorflow` has no cp314
   wheel; `pip install tensorflow` reports "No matching distribution".
   `aim` would also fail on cp314.
6. **`pip install "jax[cuda13]"`** → clean. Resolved:
   `jax-cuda13-plugin 0.10.0`, `jax-cuda13-pjrt 0.10.0`,
   `nvidia-cudnn-cu13 9.22.0.52`, `nvidia-cublas 13.4.1.1`,
   `nvidia-cuda-runtime 13.2.75`, `nvidia-cuda-nvcc 13.2.78`,
   `nvidia-nccl-cu13 2.30.4`, `nvidia-cufft 12.2.0.46`,
   `nvidia-cusolver 12.2.0.1`, `nvidia-cusparse 12.7.10.1`,
   `nvidia-nvjitlink 13.2.78`, `nvidia-nvshmem-cu13 3.6.5`,
   `nvidia-cuda-cupti 13.2.75`, `nvidia-cuda-nvrtc 13.2.78`,
   `nvidia-cuda-cccl 13.2.75`, `nvidia-cuda-crt 13.2.78`,
   `nvidia-nvvm 13.2.78`. ~2 GB total download — flag this in the
   README so users aren't surprised.
7. **GPU smoke test**: `jax.devices()` → `[CudaDevice(id=0)]`; a
   `jnp.arange(1e6).sum()` runs on the RTX 5060 and returns the
   correct value. JAX 0.10 + CUDA 13 wheels work on Blackwell `sm_120`
   with driver 595.71.05.
8. **`pytest -q`** (no `JAX_PLATFORMS` override): **127 passed in
   ~74 s.** No code changes to FabricPC required.

### Observations to fold into the plan

- The base `dependencies` block does not need any new pins or floors —
  Python 3.14 just works. No changes to `requires-python` are needed
  either; older Python users keep getting the older JAX their resolver
  picks.
- The right shape for `[viz]` and `[tfds]` is environment markers, not
  splitting into new extras. That keeps every existing command working
  on every platform where it works today, and only changes behavior
  on the new platforms where the old behavior was *broken*.
- Even though `tensorflow-datasets` itself has a cp314 wheel, it
  imports `tensorflow` at runtime, so both deps need
  `python_version < '3.14'` markers.
- Setting `JAX_PLATFORMS=cuda` globally (which `jax_setup.py` would do
  if invoked with `"cuda"`) breaks tests that explicitly request the
  CPU backend. Recommend the README/docs note that
  `set_jax_flags_before_importing_jax("cuda")` should be left as
  `None` (auto-detect) for development/test runs, and only forced for
  benchmarking / production runs. No code change needed.
- No system cuDNN was needed — `nvidia-cudnn-cu13 9.22.0.52` from the
  `jax[cuda13]` resolution provides it. A user who *prefers* the
  system libraries can switch to `jax[cuda13-local]` and install
  cuDNN via `dnf` (Fedora 43 ships a recent enough `cuda-cudnn-13-2`
  in the CUDA repo).

### Backwards-compatibility audit of every proposed change

| Change | Existing CUDA-12 / Py 3.10–3.12 user | Effect |
|---|---|---|
| `cuda = [ "jax[cuda12]>=0.10.0" ]` (unchanged extra, added floor) | `pip install -e ".[cuda]"` continues to pull the CUDA-12 stack | None unless their `jax` was already older than 0.10 — then floor forces an upgrade. Drop the floor if even that is unacceptable. |
| New `cuda13` extra | Untouched unless they opt in | None |
| `aim` markered `< '3.13'` inside `[viz]` | Aim still installed via `[viz]` and `[all]` | None |
| `tensorflow*` markered `< '3.14'` inside `[tfds]` | TF still installed via `[tfds]` and `[all]` | None |
| `requires-python` left at `>=3.10` | — | None |
| New `3.13`/`3.14` classifiers | — | None (informational) |
| `black>=26.1.0` (was `==26.1.0`) | Wider range, latest 26.3.1 | None (more permissive) |
| `target-version` adds `py313`/`py314` | Existing targets retained | None |
| README adds modern-platform section | Legacy instructions unchanged | None |

The plan as revised does **not** break any existing install path. The
only externally-visible change for an old-platform user is the
optional `>=0.10.0` floor on `jax[cuda12]`, which can be omitted if
even that is too aggressive.

### Final venv inventory (selected)

```
black               26.1.0
chex                0.1.91
flax                0.12.7
jax                 0.10.0
jax-cuda13-pjrt     0.10.0
jax-cuda13-plugin   0.10.0
jaxlib              0.10.0
jaxtyping           0.3.9
numpy               2.4.4
optax               0.2.8
orbax-checkpoint    0.11.39
pytest              9.0.3
scipy               1.17.1
```

94 packages total; aim and tensorflow intentionally absent.

---

## Status

- [x] Survey current state (system, PyPI, wheel availability).
- [x] Draft this plan.
- [x] Validate by stepping through installs in `virt-env/fpc3`.
- [x] Validate by running test suite (127 passed).
- [x] Apply `pyproject.toml` changes from §1–§3.
- [x] Apply README quick-start rewrite from §4.
- [x] (Optional) cuDNN footnote — included inline in the modern-platforms README section.
- [x] Re-validate revised pyproject:
      - `pip install -e ".[all]"` on Py 3.14 skipped aim + TF (markers
        worked) and pulled the CUDA 12 stack via unchanged `[cuda]`.
      - `pip install -e ".[cuda13]"` resolved cleanly.
      - `pytest -q` → 127 passed.

## Addendum: auto-detect installer (added after initial implementation)

PEP 508 environment markers do not include a CUDA version, so the
choice between `[cuda12]` and `[cuda13]` cannot be made declaratively
in `pyproject.toml`. To avoid forcing users to download both stacks or
to think about which one matches their driver, added
`scripts/install.py`:

- Runs `nvidia-smi` and parses its `CUDA Version: X.Y` line (the max
  runtime the driver supports).
- Picks `[cuda13]` if max ≥ 13.0, `[cuda12]` if max ≥ 12.0, else
  no CUDA extra (CPU-only install).
- Defaults extras to `dev,experiments,viz` and adds the chosen CUDA
  extra; `--all` adds `tfds`; `--cuda {12,13,none}` overrides
  detection; `--dry-run` prints the resolved pip command.
- Trailing args after `--` are passed through to pip
  (e.g. `--no-cache-dir`).

### Pitfall surfaced and documented

While validating, I tried to "convert" the venv from cuda12 to cuda13
by uninstalling the cu12 wheels in place. That **broke JAX**:
NVIDIA's PyPI wheels share the `nvidia/*` namespace package, and
`pip uninstall nvidia-cudnn-cu12` wipes `nvidia/cudnn/lib/` even
though `nvidia-cudnn-cu13` still claims to own files there. The
recovery was `pip install --force-reinstall --no-deps` of all the
cu13/unsuffixed nvidia wheels.

Both the helper docstring and the README now warn:

> Do not swap CUDA stacks inside one venv — recreate it.

## Implementation summary

`pyproject.toml`:
- Added `Programming Language :: Python :: 3.13` and `:: 3.14` classifiers.
- Relaxed `black[colorama]==26.1.0` → `black[colorama]>=26.1.0`.
- Added `python_version < '3.13'` marker to `aim` inside `[viz]`.
- Added `python_version < '3.14'` markers to both deps inside `[tfds]`.
- Floored `cuda = [ "jax[cuda12]>=0.10.0" ]` (semantics unchanged).
- Added `cuda12 = [ "jax[cuda12]>=0.10.0" ]` explicit alias.
- Added `cuda13 = [ "jax[cuda13]>=0.10.0" ]` (new, opt-in).
- Added `py313`/`py314` to `[tool.black] target-version`.
- `requires-python` left at `>=3.10`. `[all]` shape unchanged.

`README.md`:
- Quick Start now leads with the "safest copy-paste" Python 3.12.x +
  CUDA 12 recipe (legacy behavior preserved).
- New "Modern platforms (Python 3.13+ or CUDA 13)" subsection
  explains marker-driven self-degradation, the `[cuda13]` opt-in, the
  CUDA 13 wheel download size, and the `cuda13-local` /
  `cuda-cudnn-13-2` system-cuDNN alternative.

## Scope reduction: drop Python 3.14 support (2026-05-13)

Capping supported Python at 3.13 because TensorFlow still has no
Python 3.14 wheels and we'd rather have `[tfds]` installable
everywhere we claim support than juggle two-axis brokenness. (Aim is
already gated below 3.13.)

Changes:
- `requires-python` tightened from `>=3.10` to `>=3.10,<3.14` so pip
  refuses to install on 3.14 with a clear error rather than letting
  users discover the `[tfds]` resolution failure after the fact.
- Dropped `Programming Language :: Python :: 3.14` classifier.
- Removed `python_version < '3.14'` markers from both `[tfds]`
  entries (now unreachable thanks to the `requires-python` cap).
- Dropped `py314` from `[tool.black] target-version`.
- README "Modern platforms" subsection retitled `(Python 3.13 or
  CUDA 13)` and the Python-3.14 bullet removed; added an explicit
  3.10–3.13 support statement.

Validation impact: the `virt-env/fpc3` venv (Python 3.14.4) is now
out of scope. Re-validation will require a Python 3.10–3.13 venv;
the empirical log above remains an accurate record of *what worked*
on 3.14 before the scope reduction, even though that configuration
is no longer supported.

## Tighten: forbid simultaneous cuda12/cuda13 install (2026-05-13)

Even with separate `[cuda12]` and `[cuda13]` extras, a user could
land both stacks in one venv via `pip install -e ".[all]"` (which
transitively pulled `[cuda]` = cuda12) followed later by an explicit
`pip install -e ".[cuda13]"`. Both stacks share the `nvidia/*`
namespace package, so co-existence is fragile and uninstalling one
silently wipes the other (see the empirical pitfall above).

Two-part fix:

1. **`pyproject.toml`**: `[all]` no longer transitively pulls any CUDA
   extra. Users now must combine `[all]` with one of
   `[cuda]`/`[cuda12]`/`[cuda13]`, e.g. `pip install -e ".[all,cuda]"`.
   A `pip install -e ".[all]"` alone yields a CPU-only install. This
   removes the only single-command path that could have produced a
   dual-stack venv.

2. **`scripts/install.py`**: Added a guard that uses
   `importlib.metadata` to detect whether `jax-cuda12-plugin` or
   `jax-cuda13-plugin` is already installed in the running
   interpreter's environment. If the selected extra would conflict
   with an existing opposite stack, the script aborts with a clear
   message telling the user to recreate the venv. Verified by:
   - fpc3 (has cuda13) + `--cuda 12` → aborts.
   - fpc3 (has cuda13) + `--cuda 13` → proceeds.
   - fpc3 (has cuda13) + `--cuda none` → proceeds (CPU-only).
   - fpc6 (Py3.13, clean) + auto → detects driver, picks cuda13.

   The guard is defensive only — pip itself does not understand CUDA
   versions, so it cannot enforce mutual exclusivity at resolve time.
   A user determined to break this can still run two raw
   `pip install` commands; the helper is the supported path.

### Behavior matrix after this change

| Command | Result |
|---|---|
| `python scripts/install.py [--all]` | Exactly one CUDA stack (auto) or CPU-only. Guard prevents conflicts. |
| `pip install -e ".[all]"` | CPU-only. |
| `pip install -e ".[all,cuda]"` or `[all,cuda12]"` | cuda12 stack. |
| `pip install -e ".[all,cuda13]"` | cuda13 stack. |
| `pip install -e ".[cuda12,cuda13]"` | **Both stacks** — pip cannot prevent this; the helper is the way to enforce single-stack. |

## Rename: `[all]` → `[cpu]` (2026-05-13)

The `[all]` extras name became misleading once it stopped including
CUDA: a user would reasonably expect `[all]` to mean "everything",
not "everything except the GPU stack". Renamed to `[cpu]` to make the
no-GPU semantics explicit (consistent with how `jax[cpu]` is named).

Updates:
- `pyproject.toml`: `all = [...]` → `cpu = [...]`; comment updated.
- `README.md`: all forward-looking `[all]`/`[all,cuda*]` references
  rewritten to `[cpu]`/`[cpu,cuda*]`.
- `scripts/install.py`: `--all` CLI flag unchanged — it just toggles
  inclusion of `[tfds]` in the helper's hardcoded extras list and was
  never tied to the `[all]` extras name.

Behavior matrix (post-rename, supersedes the table just above):

| Command | Result |
|---|---|
| `python scripts/install.py [--all]` | Exactly one CUDA stack (auto) or CPU-only. Guard prevents conflicts. |
| `pip install -e ".[cpu]"` | CPU-only. |
| `pip install -e ".[cpu,cuda]"` or `".[cpu,cuda12]"` | cuda12 stack. |
| `pip install -e ".[cpu,cuda13]"` | cuda13 stack. |
| `pip install -e ".[cuda12,cuda13]"` | **Both stacks** — pip cannot prevent this; the helper is the way to enforce single-stack. |

This is a **breaking change** for anyone who has documentation or
scripts referencing `pip install -e ".[all]"`. There is no
backwards-compatibility alias — `[all]` simply does not exist after
this rename.

## Revert: `[cpu]` → `[all]` with build-time CUDA detection (2026-05-13, later)

The `[cpu]` rename was reverted. The user's preference is that
`pip install -e ".[all]"` should "just work" on any host — installing
the GPU stack when the host has CUDA 12 or 13 available, and staying
CPU-only otherwise. That is impossible declaratively (PEP 508 has no
CUDA-version marker), so the implementation moved to a build-time
hook.

Implementation:

- `pyproject.toml`:
  - `[project.optional-dependencies]` block removed.
  - `dynamic = ["optional-dependencies"]` added to `[project]`.
  - `[build-system].requires` bumped from `setuptools>=61.0` to
    `setuptools>=64.0` (dynamic optional-dependencies is well-supported
    from 64+).
- `setup.py` (new top-level file):
  - Imports `setuptools.setup`.
  - Defines all static extras (`dev`, `tfds`, `experiments`, `viz`,
    `cuda`, `cuda12`, `cuda13`) — setuptools requires the complete
    map when the field is dynamic.
  - Runs `nvidia-smi` at build time, parses the `CUDA Version: X.Y`
    line, and appends `fabricpc[cuda12]` or `fabricpc[cuda13]` to the
    `all` extra accordingly. Falls back to CPU-only when no usable
    driver is detected.
  - Prints the detection result to stderr so users see what was
    chosen during their `pip install`.
- `README.md`:
  - Quick Start rewritten around `pip install -e ".[all]"` as the
    primary path.
  - Explicit alternatives (force a CUDA major, CPU-only on a GPU
    host, helper script) documented in a follow-up subsection.
  - Note about editable vs built-wheel caveat: built wheels freeze
    the detection result; editable installs re-detect on each run.

### Validation (2026-05-13, host: Fedora 43, CUDA 13.2, RTX 5060)

`python setup.py egg_info` produced a `requires.txt` whose `[all]`
section contains `fabricpc[dev,experiments,tfds,viz]` and
`fabricpc[cuda13]` — exactly the expected dynamic injection. All
static extras are also present. The `viz` extra's aim entry is
correctly split out as a marker-conditional sub-extra
(`[viz:python_version < "3.13"]`).

### Caveats

- The detection runs at *build* time. For an editable install
  (`pip install -e .`) the hook re-runs on each invocation, so
  switching the host's driver and re-running picks the new stack.
  For a built wheel the detection result is frozen at wheel-build
  time — building on a CUDA-13 box and installing the wheel on a
  CUDA-12 box would still claim `[cuda13]`. Editable installs are
  the recommended path for development; users redistributing wheels
  should not rely on `[all]` self-tuning.
- `[all]` can still co-install with an explicit CUDA extra, e.g.
  `pip install -e ".[all,cuda12]"` on a CUDA-13 host would pull
  both cuda12 (explicit) and cuda13 (via `[all]` detection). The
  helper script's importlib.metadata guard still catches this on
  the next run; pip itself cannot.
- `scripts/install.py` remains useful for users who want explicit
  CUDA overrides, the dual-stack pre-flight guard, or a custom
  extras list. Its detection mirrors `setup.py`'s.

## Fixes for the two caveats above (2026-05-13)

### Fix for caveat 1: built-wheel detection freeze

Added `FABRICPC_SKIP_CUDA_DETECT` env-var opt-out to `setup.py`. When
set (any non-empty value), the build hook short-circuits the
`nvidia-smi` probe and leaves `[all]` CPU-only. Use case:
`FABRICPC_SKIP_CUDA_DETECT=1 python -m build` to produce a
CUDA-agnostic wheel for redistribution.

Rationale for the env-var approach: PEP 517 doesn't give setup.py a
clean signal to distinguish "transient local wheel for local install"
from "redistribution wheel" — `bdist_wheel` is invoked for both. An
explicit opt-in env var is more robust than `sys.argv` heuristics.

Verified by:
- `FABRICPC_SKIP_CUDA_DETECT=1 python setup.py egg_info` → `[all]`
  contains only the non-CUDA bundle.
- Default invocation (no env var) → `[all]` contains the detected
  `[cuda13]` extra on this host.

### Fix for caveat 2: explicit + auto-detected CUDA conflict

Added a runtime check in `fabricpc/__init__.py`: on
`import fabricpc`, the package calls `importlib.metadata.version()`
for both `jax-cuda12-plugin` and `jax-cuda13-plugin`; if both are
present, it raises `ImportError` with a message naming the conflict
and pointing to the bypass env var. `FABRICPC_ALLOW_MULTIPLE_CUDA=1`
disables the check.

This catches the conflict regardless of how it was created:
- `pip install -e ".[all,cuda12]"` on a CUDA-13 host (the original
  caveat).
- Two separate pip invocations adding opposite stacks.
- Manual installs that miss the helper's pre-flight guard.

Cost: two `importlib.metadata.version()` calls per process at first
`import fabricpc` (~sub-millisecond, negligible).

Verified by:
- Normal import (single stack present) → succeeds.
- Spoofed dual-stack via `importlib.metadata.version` monkey-patch →
  raises `ImportError` with the expected message.
- Spoofed dual-stack + `FABRICPC_ALLOW_MULTIPLE_CUDA=1` → import
  succeeds.
- Full `pytest -q` → 127 passed (no regressions from the new check
  on the single-stack venv).

### Env-var summary

| Variable | Effect | Set by |
|---|---|---|
| `FABRICPC_SKIP_CUDA_DETECT=1` | `setup.py` build hook skips CUDA detection; `[all]` is CPU-only. | Wheel builder before `pip wheel` / `python -m build`. |
| `FABRICPC_ALLOW_MULTIPLE_CUDA=1` | Bypass the `import fabricpc` runtime check that forbids both cuda12 and cuda13 plugins. | User debugging only. |

## Reverse the black pin relaxation (2026-05-13, later)

The earlier "Relax `black==26.1.0` → `black>=26.1.0`" change caused
**lint CI drift**:

- `.github/workflows/lint.yml` pins `version: "26.1.0"` exactly.
- `.pre-commit-config.yaml` pins `rev: 26.1.0`.
- The dev extra was floored at `>=26.1.0`, so a fresh `pip install
  -e ".[dev]"` resolves to whatever's latest (26.3.1 today).

A contributor who runs `black .` directly from their venv (instead
of going through pre-commit, which uses its own isolated env at the
pinned rev) gets 26.3.1 output. CI then rejects the diff because
its 26.1.0 disagrees. Net effect: contributors pass formatting
locally and fail CI.

**Fix**: re-pin the dev extra to exactly `26.1.0` in `setup.py` so
all three sources agree. When black is bumped in the future, bump
all three (CI workflow, pre-commit-config, dev extra) in the same
commit.

## Back the backwards-compat claim with CI (2026-05-13, later)

The plan's empirical log only covers one configuration (Fedora 43 /
Python 3.14.4 / CUDA 13.2 — which is itself now out of supported
scope after the 3.14 drop). The "doesn't break older platforms"
claim has been asserted but never automated.

**Fix**: added `.github/workflows/test.yml` with a `cpu-smoke` job:

- Runs on `ubuntu-latest` with `actions/setup-python@v5` at
  `python-version: "3.12"`.
- Installs the project via `pip install -e ".[dev,experiments,viz]"`
  (deliberately excludes `[all]` so the auto-CUDA hook is not even
  exercised; the GitHub runner has no NVIDIA driver anyway, so the
  hook would self-skip).
- Runs `pytest -q --no-header`.
- Triggers on every push and PR.
- `concurrency:` group cancels in-progress runs on the same ref.
- `timeout-minutes: 30` caps a hung job.

This gives us a real cross-platform check: Ubuntu (vs. the local
Fedora dev env) + Python 3.12 (vs. the local 3.14/3.13) + no GPU
(vs. local Blackwell with CUDA 13). If any of the pyproject/setup.py
changes break older Pythons or non-Fedora distros, CI will say so.

### Extending coverage later

The job is intentionally minimal. To broaden coverage, convert it
to a matrix:

```yaml
strategy:
  fail-fast: false
  matrix:
    python-version: ["3.10", "3.11", "3.12", "3.13"]
steps:
  - uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python-version }}
      cache: pip
```

I left this as a single 3.12 job for now per the "small CI job"
brief; the four-version matrix is roughly 4× the cost and the
single-version job already catches the most common breakage modes
(import errors, dynamic-extras parsing failures, dependency
resolution conflicts).
