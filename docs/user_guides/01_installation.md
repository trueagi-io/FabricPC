# Installation

## Requirements

- Python 3.10–3.13
- CUDA 12 or 13 for GPU acceleration (CPU-only works but is significantly slower)

> Python 3.13 still works, but the Aim experiment tracking library has
> no Python 3.13 wheels and is silently skipped by the `[viz]` extra.
> Use Python 3.12 or earlier if you need Aim.

> **macOS on Intel (x86_64):** use Python 3.10 or 3.11. TensorFlow
> dropped macOS-x86_64 wheels after TF 2.16 and never shipped any for
> Python 3.12+, so `[tfds]` (and therefore `[all]`) cannot be
> resolved on Intel Macs running Python 3.12 or 3.13. Homebrew's
> default `python3` formula is currently Python 3.13, so install an
> older interpreter explicitly with `brew install python@3.11` and
> create the venv from `/usr/local/opt/python@3.11/bin/python3.11`.
> Apple Silicon Macs and Linux are unaffected.

## Install from Source

Clone the repository and install in editable mode:

```bash
pip install -e ".[all]"
```

This installs all optional dependencies. For a minimal install, omit `[all]`:

```bash
pip install -e .
```

### Optional Dependency Groups

| Group | Contents | Install with |
|-------|----------|--------------|
| `dev` | pytest, hypothesis, black, mypy, pre-commit | `pip install -e ".[dev]"` |
| `tfds` | TensorFlow Datasets for CIFAR loaders | `pip install -e ".[tfds]"` |
| `experiments` | SciPy for statistical analysis | `pip install -e ".[experiments]"` |
| `viz` | Plotly, Aim, Pandas for dashboarding | `pip install -e ".[viz]"` |
| `all` | Everything above | `pip install -e ".[all]"` |

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

Aim provides experiment tracking dashboards. After installing with `[viz]`:

```bash
aim up
```

This starts a web dashboard at `http://localhost:43800`. See the [Experiment Tracking](09_experiment_tracking.md) guide for details.

## Common Issues

**JAX/CUDA version conflict**: If you see CUDA-related errors, ensure your CUDA 12 drivers match the JAX version. See [JAX installation docs](https://jax.readthedocs.io/en/latest/installation.html).

**Triton GEMM errors**: If you see XLA errors mentioning Triton:

```bash
export FABRICPC_DISABLE_TRITON_GEMM=1
```

**Aim not compatible with Python 3.13+**: Use Python 3.12 or earlier if you need experiment tracking.

**`No matching distribution found for tensorflow` on macOS Intel**:
TensorFlow has no `macosx_x86_64` wheels for Python 3.12 or 3.13, so
the `[tfds]` and `[all]` extras cannot install on an Intel Mac with
those interpreters. Install Python 3.11 via Homebrew and create your
venv from it:

```bash
brew install python@3.11
/usr/local/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

(Apple Silicon Macs and Linux are unaffected.)
