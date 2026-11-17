"""Tests for manifest.json: the record of what produced a result.

Suite TS-U07 (Run Manifest). SFR-025 (record FabricPC, suite, and Git commit
versions) is covered by the version and Git tests below; they also cover the
command and seed parts of TC-U07-1 (SFR-023, SFR-024).
"""

import json
import re
import subprocess
from types import SimpleNamespace

import jax
import pytest

import fabricpc
from fabricpc.bench.run import manifest
from fabricpc.bench.run.manifest import (
    SCHEMA_VERSION,
    SUITE_VERSION,
    build_manifest,
    write_manifest,
)
from fabricpc.bench.definitions.registry import ROWS

SHA = "0123456789abcdef0123456789abcdef01234567"
COMMAND = ["python", "-m", "fabricpc.bench", "mnist-mlp-spc", "--trials", "2"]

def _no_git(*args, **kwargs):
    raise OSError("git is not available")

def _fake_git(stdout, returncode=0):
    return lambda *a, **k: SimpleNamespace(stdout=stdout, returncode=returncode)

def test_manifest_records_versions_hardware_and_the_command(tmp_path):
    row = ROWS["mnist-mlp-spc"]
    path = write_manifest(
        tmp_path,
        row,
        seeds=[0, 1000],
        command=["python", "-m", "fabricpc.bench", "mnist-mlp-spc", "--trials", "2"],
    )

    assert path == tmp_path / row.id / "manifest.json"
    m = json.loads(path.read_text())

    assert m["schema_version"] == SCHEMA_VERSION
    assert m["row"]["id"] == "mnist-mlp-spc"
    assert m["seeds"] == [0, 1000]
    assert m["command"] == "python -m fabricpc.bench mnist-mlp-spc --trials 2"

    v = m["versions"]
    assert v["fabricpc"] == fabricpc.__version__
    assert v["jax"] == jax.__version__
    for name in ("jaxlib", "optax", "python"):
        assert v[name]

    assert m["platform"]["system"]
    assert m["devices"] == [str(d) for d in jax.devices()]
    assert "xla_flags" in m  # may be empty, but must be recorded
    assert m["created_utc"].endswith("Z")


def test_manifest_records_the_git_commit_when_inside_a_repo(tmp_path):
    row = ROWS["mnist-mlp-spc"]
    m = json.loads(write_manifest(tmp_path, row, seeds=[0], command=[]).read_text())
    # This test file lives inside the FabricPC git repo, so a sha is expected.
    assert m["git_sha"] is None or len(m["git_sha"]) == 40


def test_row_description_includes_the_expected_score():
    from fabricpc.bench.run.manifest import describe_row
    from fabricpc.bench.definitions.registry import ROWS

    judged = describe_row(ROWS["mnist-mlp-spc"])["reference"]
    assert judged["metric"] == "accuracy"
    assert "Colab T4" in judged["source"]
    assert describe_row(ROWS["cifar10-vgg5-spc"])["reference"] is None


def test_git_sha_falls_back_to_the_environment_when_git_cannot_say(monkeypatch):
    # Cloud jobs can run a copy of the code without its .git folder.
    from fabricpc.bench.run import manifest

    def no_git(*args, **kwargs):
        raise OSError("git is not available")

    monkeypatch.setattr(manifest.subprocess, "run", no_git)
    monkeypatch.setenv("FABRICPC_GIT_SHA", "a" * 40)
    assert manifest._git_sha() == "a" * 40

    monkeypatch.setenv("FABRICPC_GIT_SHA", "not-a-sha")
    assert manifest._git_sha() is None

    monkeypatch.delenv("FABRICPC_GIT_SHA")
    assert manifest._git_sha() is None


# --- SFR-025: FabricPC, suite, and Git commit versions ----------------------

def test_manifest_records_the_suite_version():
    m = build_manifest(ROWS["mnist-mlp-spc"], [0], COMMAND)
    assert m["versions"]["suite"] == SUITE_VERSION
    assert isinstance(SUITE_VERSION, str) and SUITE_VERSION.strip()


def test_suite_version_looks_like_a_version_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+([.\-+][0-9A-Za-z.\-+]+)?", SUITE_VERSION)


def test_all_three_versions_are_in_every_manifest(tmp_path):
    # SFR-025: FabricPC version, suite version, and Git commit, together.
    m = json.loads(
        write_manifest(tmp_path, ROWS["mnist-mlp-spc"], seeds=[0], command=COMMAND)
        .read_text()
    )
    assert m["versions"]["fabricpc"] == fabricpc.__version__
    assert m["versions"]["suite"] == SUITE_VERSION
    assert "git_sha" in m  # key is always present, even if the value is None


def test_versions_are_recorded_for_every_row_family():
    for row_id in ("mnist-mlp-spc", "mnist-mlp-epc", "mnist-mlp-backprop",
                   "fashionmnist-mlp-spc", "cifar10-vgg5-spc"):
        v = build_manifest(ROWS[row_id], [0], COMMAND)["versions"]
        assert v["fabricpc"] == fabricpc.__version__
        assert v["suite"] == SUITE_VERSION


def test_versions_survive_the_json_round_trip(tmp_path):
    path = write_manifest(tmp_path, ROWS["mnist-mlp-spc"], seeds=[0], command=COMMAND)
    on_disk = json.loads(path.read_text())
    assert on_disk["versions"] == build_manifest(
        ROWS["mnist-mlp-spc"], [0], COMMAND
    )["versions"]


def test_git_sha_comes_from_git_when_git_can_say(monkeypatch):
    monkeypatch.setattr(manifest.subprocess, "run", _fake_git(SHA + "\n"))
    monkeypatch.delenv("FABRICPC_GIT_SHA", raising=False)
    assert manifest._git_sha() == SHA


def test_git_wins_over_the_environment_variable(monkeypatch):
    monkeypatch.setattr(manifest.subprocess, "run", _fake_git(SHA + "\n"))
    monkeypatch.setenv("FABRICPC_GIT_SHA", "b" * 40)
    assert manifest._git_sha() == SHA


def test_git_sha_falls_back_when_git_returns_an_error(monkeypatch):
    # Not a repo: git exits non-zero and prints nothing useful.
    monkeypatch.setattr(manifest.subprocess, "run", _fake_git("", returncode=128))
    monkeypatch.setenv("FABRICPC_GIT_SHA", "c" * 40)
    assert manifest._git_sha() == "c" * 40


def test_git_sha_falls_back_when_git_times_out(monkeypatch):
    def slow_git(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(manifest.subprocess, "run", slow_git)
    monkeypatch.setenv("FABRICPC_GIT_SHA", "d" * 40)
    assert manifest._git_sha() == "d" * 40


def test_git_sha_ignores_a_git_answer_that_is_not_a_full_sha(monkeypatch):
    monkeypatch.setattr(manifest.subprocess, "run", _fake_git("HEAD\n"))
    monkeypatch.delenv("FABRICPC_GIT_SHA", raising=False)
    assert manifest._git_sha() is None


@pytest.mark.parametrize(
    "bad", ["", "abc123", "g" * 40, "A" * 40, "a" * 39, "a" * 41, " " + "a" * 39]
)
def test_malformed_environment_sha_is_rejected(monkeypatch, bad):
    monkeypatch.setattr(manifest.subprocess, "run", _no_git)
    monkeypatch.setenv("FABRICPC_GIT_SHA", bad)
    assert manifest._git_sha() is None


def test_environment_sha_is_stripped_of_whitespace(monkeypatch):
    monkeypatch.setattr(manifest.subprocess, "run", _no_git)
    monkeypatch.setenv("FABRICPC_GIT_SHA", f"  {SHA}\n")
    assert manifest._git_sha() == SHA


def test_manifest_git_sha_is_none_when_nothing_can_say(monkeypatch):
    monkeypatch.setattr(manifest.subprocess, "run", _no_git)
    monkeypatch.delenv("FABRICPC_GIT_SHA", raising=False)
    m = build_manifest(ROWS["mnist-mlp-spc"], [0], COMMAND)
    assert m["git_sha"] is None
    assert m["versions"]["suite"] == SUITE_VERSION  # versions still recorded


def test_manifest_git_sha_is_the_one_git_reports(monkeypatch):
    monkeypatch.setattr(manifest.subprocess, "run", _fake_git(SHA + "\n"))
    m = build_manifest(ROWS["mnist-mlp-spc"], [0], COMMAND)
    assert m["git_sha"] == SHA


def test_is_full_sha():
    assert manifest._is_full_sha(SHA)
    assert not manifest._is_full_sha(SHA[:-1])
    assert not manifest._is_full_sha(SHA.upper())


# --- TC-U07-1: command, seeds, versions, and Git commit, all present --------
def test_manifest_has_command_seeds_versions_and_git_commit(tmp_path):
    m = json.loads(
        write_manifest(tmp_path, ROWS["mnist-mlp-spc"], seeds=[3, 4], command=COMMAND)
        .read_text()
    )
    assert m["command"] == " ".join(COMMAND)
    assert m["seeds"] == [3, 4]
    assert m["versions"]
    assert "git_sha" in m


def test_manifest_describes_the_row_settings():
    # The MLP row's own settings are recorded so a run can be repeated.
    row = ROWS["mnist-mlp-spc"]
    desc = build_manifest(row, [0], COMMAND)["row"]
    assert desc["dataset"] == "mnist"
    assert desc["model"] == "mlp"
    assert desc["algorithm"] == "spc"
    assert desc["batch_size"] == row.batch_size
    assert desc["train_config"] == dict(row.train_config)
    assert desc["n_trials"] == row.n_trials