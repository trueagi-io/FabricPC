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

Note: do not try to *swap* CUDA stacks inside one venv by uninstalling
[cuda12] and installing [cuda13] (or vice versa). NVIDIA's PyPI wheels
share the `nvidia/*` namespace package, so uninstalling one variant
can wipe shared files of the other. Recreate the venv from scratch.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def detect_driver_cuda_version() -> tuple[int, int] | None:
    """Parse `nvidia-smi` for the max CUDA runtime the driver supports."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
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
    if version is None:
        return None
    major, _minor = version
    if major >= 13:
        return "cuda13"
    if major >= 12:
        return "cuda12"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also install the [tfds] extra (auto-skipped on Python 3.14).",
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
            extras.append(cuda_extra)
    elif args.cuda == "12":
        extras.append("cuda12")
    elif args.cuda == "13":
        extras.append("cuda13")
    # else: 'none' -> add nothing

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
