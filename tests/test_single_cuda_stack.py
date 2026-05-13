"""Unit tests for fabricpc._check_single_cuda_stack and the install.py guard."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import fabricpc
from fabricpc import _check_single_cuda_stack

# scripts/install.py is not on sys.path by default; add the repo root so we
# can import the helper as a regular module for testing.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import install as install_script  # noqa: E402

# Importlib raises this on missing dists. We import it once for use in
# the fakes below; the alias matches what fabricpc/__init__.py imports.
PackageNotFoundError = fabricpc.PackageNotFoundError


def _fake_version_factory(installed: set[str]):
    """Return a fake `importlib.metadata.version` that knows only `installed`."""

    def fake_version(pkg: str) -> str:
        if pkg in installed:
            return "0.10.0"
        raise PackageNotFoundError(pkg)

    return fake_version


# ---- _check_single_cuda_stack -------------------------------------------


def test_check_passes_when_no_plugins(monkeypatch):
    monkeypatch.setattr(fabricpc, "version", _fake_version_factory(set()))
    monkeypatch.delenv("FABRICPC_ALLOW_MULTIPLE_CUDA", raising=False)
    _check_single_cuda_stack()  # no raise


def test_check_passes_with_only_cuda12(monkeypatch):
    monkeypatch.setattr(
        fabricpc, "version", _fake_version_factory({"jax-cuda12-plugin"})
    )
    monkeypatch.delenv("FABRICPC_ALLOW_MULTIPLE_CUDA", raising=False)
    _check_single_cuda_stack()


def test_check_passes_with_only_cuda13(monkeypatch):
    monkeypatch.setattr(
        fabricpc, "version", _fake_version_factory({"jax-cuda13-plugin"})
    )
    monkeypatch.delenv("FABRICPC_ALLOW_MULTIPLE_CUDA", raising=False)
    _check_single_cuda_stack()


def test_check_raises_when_both_installed(monkeypatch):
    monkeypatch.setattr(
        fabricpc,
        "version",
        _fake_version_factory({"jax-cuda12-plugin", "jax-cuda13-plugin"}),
    )
    monkeypatch.delenv("FABRICPC_ALLOW_MULTIPLE_CUDA", raising=False)
    with pytest.raises(ImportError) as exc:
        _check_single_cuda_stack()
    msg = str(exc.value)
    assert "jax-cuda12-plugin" in msg and "jax-cuda13-plugin" in msg
    # D1: error message must include the venv-recreate recovery recipe.
    assert "rm -rf .venv" in msg
    assert "python -m venv .venv" in msg
    assert "scripts/install.py" in msg
    # ...and a pointer to the design note.
    assert "single_cuda_stack_check.md" in msg


# ---- bypass env var (A1 regression coverage) ----------------------------


@pytest.mark.parametrize(
    "value, should_bypass",
    [
        ("1", True),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("true", False),
    ],
)
def test_check_bypass_only_on_literal_one(monkeypatch, value, should_bypass):
    """A1: FABRICPC_ALLOW_MULTIPLE_CUDA must require literal '1' to bypass."""
    monkeypatch.setattr(
        fabricpc,
        "version",
        _fake_version_factory({"jax-cuda12-plugin", "jax-cuda13-plugin"}),
    )
    monkeypatch.setenv("FABRICPC_ALLOW_MULTIPLE_CUDA", value)
    if should_bypass:
        _check_single_cuda_stack()  # no raise
    else:
        with pytest.raises(ImportError):
            _check_single_cuda_stack()


# ---- scripts/install.py: detect_installed_cuda_extras -------------------


def test_detect_installed_cuda_extras_empty(monkeypatch):
    monkeypatch.setattr(
        install_script.importlib.metadata,
        "version",
        _fake_version_factory(set()),
    )
    assert install_script.detect_installed_cuda_extras() == set()


def test_detect_installed_cuda_extras_cuda12(monkeypatch):
    monkeypatch.setattr(
        install_script.importlib.metadata,
        "version",
        _fake_version_factory({"jax-cuda12-plugin"}),
    )
    assert install_script.detect_installed_cuda_extras() == {"cuda12"}


def test_detect_installed_cuda_extras_both(monkeypatch):
    monkeypatch.setattr(
        install_script.importlib.metadata,
        "version",
        _fake_version_factory({"jax-cuda12-plugin", "jax-cuda13-plugin"}),
    )
    assert install_script.detect_installed_cuda_extras() == {"cuda12", "cuda13"}
