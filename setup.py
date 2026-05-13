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
import re
import shutil
import subprocess
import sys

from setuptools import setup


def _detect_cuda_major() -> int | None:
    """Return 12 or 13 based on `nvidia-smi`'s max CUDA runtime, else None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        return None
    major = int(match.group(1))
    if major >= 13:
        return 13
    if major >= 12:
        return 12
    return None


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
    if os.environ.get("FABRICPC_SKIP_CUDA_DETECT"):
        print(
            "setup.py: FABRICPC_SKIP_CUDA_DETECT set — [all] left CPU-only.",
            file=sys.stderr,
        )
        return base
    cuda_major = _detect_cuda_major()
    if cuda_major == 13:
        print(
            "setup.py: detected CUDA 13.x driver — [all] will include [cuda13].",
            file=sys.stderr,
        )
        return base + ["fabricpc[cuda13]"]
    if cuda_major == 12:
        print(
            "setup.py: detected CUDA 12.x driver — [all] will include [cuda12].",
            file=sys.stderr,
        )
        return base + ["fabricpc[cuda12]"]
    print(
        "setup.py: no usable NVIDIA driver detected — [all] will be CPU-only.",
        file=sys.stderr,
    )
    return base


setup(
    extras_require={
        **_STATIC_EXTRAS,
        "all": _build_all_extra(),
    },
)
