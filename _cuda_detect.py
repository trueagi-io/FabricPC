"""Shared CUDA-driver detection used by setup.py and scripts/install.py.

Lives at the repo root (not inside fabricpc/) so setup.py can import it
during build, before the fabricpc package is installed. Intentionally
not listed in pyproject.toml's py-modules — this is build/install
tooling, not part of the runtime package.
"""

from __future__ import annotations

import re
import shutil
import subprocess


def detect_driver_cuda_version() -> tuple[int, int] | None:
    """Return (major, minor) of the max CUDA runtime the NVIDIA driver supports.

    Reads `nvidia-smi`'s "CUDA Version: X.Y" line. Returns None when
    nvidia-smi is missing, fails, hangs (10s timeout), or doesn't print
    the field.
    """
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
    return int(match.group(1)), int(match.group(2))


def pick_cuda_extra(version: tuple[int, int] | None) -> str | None:
    """Pick the matching `[cudaN]` extra name for a detected driver version.

    Returns 'cuda13' for CUDA >= 13.x, 'cuda12' for CUDA >= 12.x, else None.
    """
    if version is None:
        return None
    major, _minor = version
    if major >= 13:
        return "cuda13"
    if major >= 12:
        return "cuda12"
    return None
