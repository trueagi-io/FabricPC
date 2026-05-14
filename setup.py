"""Build-time hook that computes `optional-dependencies` for FabricPC.

PEP 508 environment markers have no CUDA-version field, so `[all]` cannot
declaratively express "use [cuda12] on a CUDA-12 host and [cuda13] on a
CUDA-13 host". Instead, this `setup.py` runs at build time, reads
`nvidia-smi` for the host's max supported CUDA runtime, and appends the
matching `fabricpc[cudaN]` to the `all` extra. When no NVIDIA driver is
detected, `[all]` stays CPU-only.

All other extras (`dev`, `tfds`, `experiments`, `viz`, `cuda`, `cuda12`,
`cuda13`) are also defined here because setuptools requires the *complete*
optional-dependencies map once the field is marked dynamic in
`pyproject.toml`.
"""

from __future__ import annotations

import os
import sys

from setuptools import setup

# PEP 517 build isolation chdirs into the source tree but does not add it
# to sys.path, so `_cuda_detect` (a sibling, not yet installed) is invisible
# to `import` without this. Affects `pip install .` and `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cuda_detect import detect_driver_cuda_version, pick_cuda_extra

_STATIC_EXTRAS = {
    "dev": [
        "pytest>=7.0.0",
        "hypothesis>=6.0.0",
        # Pinned exact to match `.github/workflows/lint.yml` and
        # `.pre-commit-config.yaml`. Drifting `black` versions across
        # local / pre-commit / CI causes contributors to pass formatting
        # locally and fail CI. Bump all three together.
        "black[colorama]==26.1.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    "tfds": [
        "tensorflow-datasets>=4.9.0",
        "tensorflow>=2.15.0",
        # `tensorflow_datasets.core.dataset_builder` imports `importlib_resources`
        # unconditionally (not gated on Python<3.9), but tfds does not declare it
        # as a dependency. Without this, `tfds.load(...)` raises ModuleNotFoundError.
        "importlib_resources",
    ],
    "experiments": [
        "scipy>=1.10.0",
    ],
    "viz": [
        "plotly>=5.0.0",
        "kaleido>=0.2.1",
        "pandas>=2.0.0",
        # Aim has no Python 3.13+ wheels; the marker keeps [viz] installable
        # on newer Python without dropping Aim for users on Python <=3.12.
        "aim>=3.0.0; python_version < '3.13'",
    ],
    "cuda": ["jax[cuda12]>=0.10.0"],
    "cuda12": ["jax[cuda12]>=0.10.0"],
    "cuda13": ["jax[cuda13]>=0.10.0"],
}


def _build_all_extra() -> list[str]:
    """Compose the `all` extra, injecting the right CUDA stack for the host.

    Set `FABRICPC_SKIP_CUDA_DETECT=1` to skip detection and leave `[all]`
    CPU-only — useful when building a wheel intended for redistribution to
    hosts whose CUDA driver differs from this build host.
    """
    base = ["fabricpc[dev,tfds,experiments,viz]"]
    # Strict `== "1"` (not just truthy) so `=0` / `=false` / `=no` don't
    # accidentally enable the skip. Matches the precedent set by
    # FABRICPC_DISABLE_TRITON_GEMM in jax_setup.py.
    if os.environ.get("FABRICPC_SKIP_CUDA_DETECT") == "1":
        print(
            "setup.py: FABRICPC_SKIP_CUDA_DETECT=1 — [all] left CPU-only.",
            file=sys.stderr,
        )
        return base
    cuda_extra = pick_cuda_extra(detect_driver_cuda_version())
    if cuda_extra is None:
        print(
            "setup.py: no usable NVIDIA driver detected — [all] will be CPU-only.",
            file=sys.stderr,
        )
        return base
    print(
        f"setup.py: detected CUDA driver — [all] will include [{cuda_extra}].",
        file=sys.stderr,
    )
    return base + [f"fabricpc[{cuda_extra}]"]


setup(
    extras_require={
        **_STATIC_EXTRAS,
        "all": _build_all_extra(),
    },
)
