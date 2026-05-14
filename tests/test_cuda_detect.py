"""Unit tests for `_cuda_detect` and the install/build hooks that use it."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

import _cuda_detect
from _cuda_detect import detect_driver_cuda_version, pick_cuda_extra

# ---- pick_cuda_extra ----------------------------------------------------


@pytest.mark.parametrize(
    "version, expected",
    [
        (None, None),
        ((11, 0), None),  # too old
        ((12, 0), "cuda12"),
        ((12, 9), "cuda12"),
        ((13, 0), "cuda13"),
        ((13, 2), "cuda13"),
        # Drivers newer than the latest wheel stack get pinned to the
        # latest known major (no warning expected; we deliberately don't
        # try to predict JAX's future support).
        ((14, 0), "cuda13"),
        ((99, 0), "cuda13"),
    ],
)
def test_pick_cuda_extra(version, expected):
    assert pick_cuda_extra(version) == expected


# ---- detect_driver_cuda_version: nvidia-smi missing ---------------------


def test_detect_returns_none_when_nvidia_smi_missing():
    with patch.object(_cuda_detect.shutil, "which", return_value=None):
        assert detect_driver_cuda_version() is None


# ---- detect_driver_cuda_version: parsing ---------------------------------


def _mock_nvidia_smi(stdout: str = "", returncode: int = 0):
    """Build a MagicMock that mimics a successful nvidia-smi subprocess.run."""
    return MagicMock(stdout=stdout, returncode=returncode)


def test_detect_parses_cuda13():
    sample = (
        "+-----------------------------------------+\n"
        "| NVIDIA-SMI 575.74  Driver: 595.71.05    "
        "CUDA Version: 13.2                       |\n"
        "+-----------------------------------------+\n"
    )
    with (
        patch.object(_cuda_detect.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        patch.object(
            _cuda_detect.subprocess, "run", return_value=_mock_nvidia_smi(sample)
        ),
    ):
        assert detect_driver_cuda_version() == (13, 2)


def test_detect_parses_cuda12():
    with (
        patch.object(_cuda_detect.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        patch.object(
            _cuda_detect.subprocess,
            "run",
            return_value=_mock_nvidia_smi("CUDA Version: 12.4"),
        ),
    ):
        assert detect_driver_cuda_version() == (12, 4)


def test_detect_returns_none_on_cuda_na():
    """Real-world failure mode: broken driver prints `CUDA Version: N/A`."""
    with (
        patch.object(_cuda_detect.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        patch.object(
            _cuda_detect.subprocess,
            "run",
            return_value=_mock_nvidia_smi("CUDA Version: N/A"),
        ),
    ):
        assert detect_driver_cuda_version() is None


def test_detect_returns_none_on_nonzero_exit():
    with (
        patch.object(_cuda_detect.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        patch.object(
            _cuda_detect.subprocess,
            "run",
            return_value=_mock_nvidia_smi("CUDA Version: 13.2", returncode=9),
        ),
    ):
        assert detect_driver_cuda_version() is None


def test_detect_returns_none_on_timeout():
    with (
        patch.object(_cuda_detect.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        patch.object(
            _cuda_detect.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10),
        ),
    ):
        assert detect_driver_cuda_version() is None


def test_detect_returns_none_on_oserror():
    with (
        patch.object(_cuda_detect.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        patch.object(_cuda_detect.subprocess, "run", side_effect=OSError("boom")),
    ):
        assert detect_driver_cuda_version() is None


# ---- detect_driver_cuda_version: locale forcing (D4) --------------------


def test_detect_forces_c_locale():
    """Regression test for D4: nvidia-smi must be invoked under LANG=C / LC_ALL=C.

    Otherwise the regex against the English "CUDA Version:" label could
    silently fail on a localized system.
    """
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _mock_nvidia_smi("CUDA Version: 13.2")

    with (
        patch.object(_cuda_detect.shutil, "which", return_value="/usr/bin/nvidia-smi"),
        patch.object(_cuda_detect.subprocess, "run", side_effect=fake_run),
    ):
        detect_driver_cuda_version()
    assert captured["env"] is not None, "subprocess.run was not given an env override"
    assert captured["env"]["LANG"] == "C"
    assert captured["env"]["LC_ALL"] == "C"
