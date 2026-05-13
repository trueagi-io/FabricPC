#!/usr/bin/env python3
"""Detect the host's CUDA driver and run `pip install -e .` with matching extras.

PEP 508 has no CUDA-version marker, so the choice between `[cuda12]` and
`[cuda13]` cannot be expressed declaratively in `pyproject.toml`. This helper
reads `nvidia-smi`'s "CUDA Version" line (the maximum runtime the installed
driver supports) and selects the right extra so users only download one
JAX/CUDA stack.

Usage:
    python scripts/install.py                    # dev + experiments + viz + auto-cuda
    python scripts/install.py --all              # also include [tfds]
    python scripts/install.py --extras a,b,c     # override the extras list
    python scripts/install.py -- --no-cache-dir  # everything after `--` goes to pip

Selection rules:
    driver max CUDA >= 13.0  -> [cuda13]
    driver max CUDA >= 12.0  -> [cuda12]
    no NVIDIA driver         -> no CUDA extra (CPU-only install)

Notes:
- Only one CUDA stack should ever be installed per venv. The script
  refuses to add cuda12 to a venv that already has cuda13 (or vice
  versa).
- On Python 3.13 the script prints an explicit warning that Aim
  experiment tracking is not installable in this environment (no
  upstream wheel). Use Python 3.12 or earlier if you need Aim.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

# `_cuda_detect` lives at the repo root next to `setup.py`. Add the repo
# root to sys.path so this script can be run from any cwd via
# `python scripts/install.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _cuda_detect import detect_driver_cuda_version, pick_cuda_extra  # noqa: E402

_CUDA_PLUGIN_DISTS = {
    "cuda12": "jax-cuda12-plugin",
    "cuda13": "jax-cuda13-plugin",
}


def detect_installed_cuda_extras() -> set[str]:
    """Return the labels ('cuda12'/'cuda13') of jax cuda plugins already
    installed in the running interpreter's environment."""
    installed: set[str] = set()
    for label, dist in _CUDA_PLUGIN_DISTS.items():
        try:
            importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
        installed.add(label)
    return installed


def warn_python_compat() -> None:
    """Emit explicit warnings for Python versions with reduced functionality."""
    if sys.version_info[:2] == (3, 13):
        # Aim has no Python 3.13 wheels; the [viz] env marker silently
        # drops it. Surface that here so the user is not surprised.
        print(
            "WARNING: Python 3.13 detected. Aim experiment tracking is "
            "not installable in this environment (no upstream Python "
            "3.13 wheel) and will be skipped by the [viz] extra. "
            "Use Python 3.12 or earlier if you need Aim.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also install the [tfds] extra (TensorFlow Datasets).",
    )
    parser.add_argument(
        "--extras",
        type=str,
        default=None,
        help="Comma-separated extras list overriding the default "
        "(dev,experiments,viz).",
    )
    parser.add_argument(
        "--cuda",
        choices=("auto", "12", "13", "none"),
        default="auto",
        help="Override CUDA selection. 'auto' (default) reads nvidia-smi.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved pip command without running it.",
    )
    parser.add_argument(
        "pip_args",
        nargs=argparse.REMAINDER,
        help="Trailing args passed through to pip (use `--` to separate).",
    )
    args = parser.parse_args()

    warn_python_compat()

    if args.extras is not None:
        extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    else:
        extras = ["dev", "experiments", "viz"]
        if args.all:
            extras.append("tfds")

    if args.cuda == "auto":
        version = detect_driver_cuda_version()
        cuda_extra = pick_cuda_extra(version)
        if cuda_extra is None:
            if version is None:
                print("No NVIDIA driver detected (nvidia-smi missing or failed).")
                print("Installing CPU-only — no CUDA extra will be added.")
            else:
                print(
                    f"Driver advertises CUDA {version[0]}.{version[1]}, which is "
                    "too old for JAX's cuda12/cuda13 wheels."
                )
                print("Installing CPU-only.")
        else:
            assert version is not None
            print(
                f"NVIDIA driver supports up to CUDA {version[0]}.{version[1]} "
                f"-> selecting [{cuda_extra}]."
            )
    elif args.cuda == "12":
        cuda_extra = "cuda12"
    elif args.cuda == "13":
        cuda_extra = "cuda13"
    else:  # "none"
        cuda_extra = None

    if cuda_extra is not None:
        conflicting = detect_installed_cuda_extras() - {cuda_extra}
        if conflicting:
            other = sorted(conflicting)[0]
            print(
                f"ERROR: this venv already has the [{other}] JAX/CUDA "
                f"stack installed; refusing to also install "
                f"[{cuda_extra}]. Only one CUDA stack should be "
                f"present per venv.",
                file=sys.stderr,
            )
            return 1
        extras.append(cuda_extra)

    extras_spec = ",".join(extras)
    project_dir = Path(__file__).resolve().parent.parent

    cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{extras_spec}]"]
    pass_through = list(args.pip_args)
    if pass_through and pass_through[0] == "--":
        pass_through = pass_through[1:]
    cmd.extend(pass_through)

    print("Running:", " ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(project_dir))


if __name__ == "__main__":
    sys.exit(main())
