"""SFR-004: bad number flags are rejected before anything runs."""

import pytest

from fabricpc.bench.__main__ import main


def rejected(argv, capsys):
    """Run the CLI, expect argparse's exit code 2, and return what it printed."""
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_zero_or_negative_trials_is_rejected_by_name(value, capsys):
    err = rejected(["mnist-mlp-spc", "--trials", value], capsys)
    assert "--trials" in err
    assert "nothing was run" in err


def test_rejected_run_does_not_delete_old_results(tmp_path, capsys):
    old = tmp_path / "mnist-mlp-spc" / "trial0.json"
    old.parent.mkdir()
    old.write_text("{}")
    rejected(["mnist-mlp-spc", "--trials", "0", "--out", str(tmp_path)], capsys)
    assert old.exists()


def test_every_bad_flag_is_named_in_one_message(capsys):
    err = rejected(
        ["mnist-mlp-spc", "--trials", "0", "--epochs", "-2", "--timed", "0"], capsys
    )
    for flag in ("--trials", "--epochs", "--timed"):
        assert flag in err


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--epochs", "nan"),
        ("--epochs", "inf"),
        ("--epochs", "0"),
        ("--warmup", "-1"),
        ("--trial", "-1"),
        ("--curve-batches", "-1"),
        ("--floor", "1.5"),
        ("--floor", "nan"),
        ("--price-per-hour", "0"),
    ],
)
def test_other_bad_values_are_rejected(flag, value, capsys):
    assert flag in rejected(["mnist-mlp-spc", flag, value], capsys)


def test_edge_values_that_work_today_are_still_allowed(capsys):
    code = main(
        [
            "mnist-mlp-spc",
            "--dry-run",
            "--trials",
            "1",
            "--epochs",
            "0.5",
            "--warmup",
            "0",
            "--curve-batches",
            "0",
            "--trial",
            "0",
            "--floor",
            "0",
        ]
    )
    assert code == 0
