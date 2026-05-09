# Modernize FabricPC Install for Python 3.14 + CUDA 13 + JAX 0.10 (In Progress)

## Summary

Bring FabricPC's default install path up to date so it works on a current
Fedora 43 / Python 3.14 / CUDA 13 / JAX 0.10 machine without requiring
users to pin to older Python or older CUDA toolkits. The current README
tells users to create a Python 3.12.x venv (because Aim does not support
3.13+); the `cuda` extra installs `jax[cuda12]` even though JAX 0.10
ships a first-class `cuda13` extra; and `pyproject.toml` only advertises
3.10–3.12 in its classifiers and `black` `target-version`.

This plan tracks the changes needed to modernize the install
instructions and `pyproject.toml`, the version pins we settle on after
empirical testing in `virt-env/fpc3` (Python 3.14.4), and which extras
must be deferred because their upstream wheels lag Python 3.14.

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
