# Installation

## Requirements

- Python 3.10–3.13
- **Platform: GPU requires Linux (x86_64 or aarch64).** JAX publishes CUDA wheels for
  Linux only. Native Windows and macOS are CPU-only; for GPU on Windows, use WSL2 (JAX
  marks WSL2 GPU support experimental).
- CUDA 12 or CUDA 13 for GPU acceleration (CPU-only works but is significantly slower)

> CUDA 13 wheels require NVIDIA driver ≥580.
>
> The optional Aim experiment tracker (in `[viz]`/`[all]`) is Linux/macOS only (it needs
> aimrocks, which has no Windows wheels) and supports Python ≤3.12. It is skipped
> automatically on Windows and on Python 3.13, so `[viz]`/`[all]` still install everything
> else; experiment tracking is unavailable in those cases.

## Install from Source

Clone the repository and install in editable mode. The one command below pulls FabricPC,
all optional dependencies, and a version-matched JAX backend — pick the line for your hardware:

```bash
# GPU, CUDA 12:
pip install -U -e ".[all,cuda12]"

# GPU, CUDA 13 (needs NVIDIA driver ≥580):
pip install -U -e ".[all,cuda13]"

# CPU only (the base `jax` dependency is the CPU build):
pip install -U -e ".[all]"
```

For a minimal install (core library only - no demos, utils, and dataloaders), omit `[all]`:

```bash
pip install -e .
```

### Why `-U`?

`jax[cuda12]` installs a coupled set of packages — `jax`, `jaxlib`, `jax-cuda12-plugin`,
and `jax-cuda12-pjrt` — whose versions must match (the plugin and pjrt are tied to the
exact `jaxlib` version). FabricPC's base dependencies install the plain CPU `jax`/`jaxlib`
first, so they are already present. Without `-U`, pip treats them as "already satisfied"
and leaves them at the installed version while still pulling the newest
`jax-cuda12-plugin` — a plugin newer than `jaxlib`, which makes JAX fail at import or at
the first GPU operation. `-U` (`--upgrade`) forces pip to upgrade the whole set together
so `jaxlib` and the CUDA plugin/pjrt land on matching versions. The same applies to `cuda13`.

### Optional Dependency Groups

`all` bundles every group except the hardware backend. Combine it with one backend extra
(`cuda12`, `cuda13`, or `cpu`), or omit the backend for the CPU build. Backend extras also
combine with narrower groups for a stripped-down install — e.g. core + GPU only with
`pip install -U -e ".[cuda12]"`, or datasets + GPU with `pip install -U -e ".[tfds,cuda12]"`.

| Group | Contents | Install with |
|-------|----------|--------------|
| `dev` | pytest, hypothesis, black, mypy, pre-commit | `pip install -e ".[dev]"` |
| `tfds` | TensorFlow Datasets for MNIST/CIFAR loaders | `pip install -e ".[tfds]"` |
| `experiments` | SciPy for statistical analysis | `pip install -e ".[experiments]"` |
| `viz` | Plotly, Aim, Pandas for dashboarding | `pip install -e ".[viz]"` |
| `cpu` | JAX CPU build (explicit) | `pip install -e ".[cpu]"` |
| `cuda12` | JAX CUDA 12 backend | `pip install -U -e ".[cuda12]"` |
| `cuda13` | JAX CUDA 13 backend (driver ≥580) | `pip install -U -e ".[cuda13]"` |
| `all` | Everything except the backend | `pip install -U -e ".[all,cuda12]"` |

## Verify Installation

```python
import fabricpc
import jax
print(jax.devices())  # Should show your GPU(s) or CPU
```

## Pre-commit Hooks (Contributors)

```bash
pre-commit install
```

This enables automatic formatting (Black) and code quality checks on each commit.

## Aim Setup (Optional)

Aim provides experiment tracking dashboards and is Linux/macOS only (Python ≤3.12). After installing with `[viz]` on a Linux/macOS, Python ≤3.12 interpreter:

```bash
aim up
```

This starts a web dashboard at `http://localhost:43800`. See the [Experiment Tracking](09_experiment_tracking.md) guide for details.

## Common Issues

**JAX/CUDA version conflict**: If you see CUDA-related errors, install the backend matching your driver (`cuda12`, or `cuda13` for driver ≥580) and re-run with `-U` so the coupled JAX wheels upgrade together.

**GPU install fails on Windows / macOS**: If `pip install -U -e ".[all,cuda12]"` (or `cuda13`) fails with `No matching distribution found for jax-cuda12-plugin`, you are on a platform without JAX CUDA wheels — JAX publishes them for Linux x86_64/aarch64 only. Install CPU-only (`pip install -U -e ".[all]"`), or use WSL2 for GPU on Windows (JAX marks WSL2 GPU support experimental).

**Triton GEMM errors**: If you see XLA errors mentioning Triton:

```bash
export FABRICPC_DISABLE_TRITON_GEMM=1
```

**Aim not available on Windows or Python 3.13+**: Aim is Linux/macOS only (it needs aimrocks, which has no Windows wheels) and supports Python ≤3.12. On Windows or Python 3.13, `[viz]`/`[all]` install everything except Aim (it is skipped automatically). Use a Linux/macOS, Python ≤3.12 environment (or WSL2 on Windows) if you need experiment tracking.

**`[WinError 206] The filename or extension is too long` on Windows**: A required dependency (`orbax-checkpoint`) ships deeply nested files that exceed Windows' default 260-character path limit.

Recommended to use WSL2, which has no `MAX_PATH` limit. You're only seeing this error if you have a general purpose Windows environment, in which case it's not advisable to enable long paths in Windows.
