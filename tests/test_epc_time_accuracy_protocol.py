"""Pure protocol tests for the ePC ResNet time-to-accuracy follow-up."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "epc_resnet18_time_accuracy.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "epc_resnet18_time_accuracy", _SCRIPT_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(_SCRIPT_PATH)
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)


def _observation(steps, learning_rate, seed, accuracy):
    return protocol.TuningObservation(
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        validation_accuracy=accuracy,
        training_seconds=10.0,
    )


def test_candidate_selection_uses_mean_across_tuning_seeds():
    observations = [
        _observation(2, 0.001, 1, 0.30),
        _observation(2, 0.001, 2, 0.40),
        _observation(5, 0.001, 1, 0.36),
        _observation(5, 0.001, 2, 0.36),
    ]

    selected = protocol.select_candidate(observations)

    assert selected.steps == 5
    assert selected.learning_rate == pytest.approx(0.001)
    assert selected.mean_validation_accuracy == pytest.approx(0.36)


def test_candidate_ties_prefer_fewer_steps_then_lower_learning_rate():
    observations = [
        _observation(5, 0.003, 1, 0.4),
        _observation(2, 0.003, 1, 0.4),
        _observation(2, 0.001, 1, 0.4),
    ]

    selected = protocol.select_candidate(observations)

    assert selected.steps == 2
    assert selected.learning_rate == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("parser", "value"),
    [
        (protocol._parse_csv_ints, ""),
        (protocol._parse_csv_ints, "2,2"),
        (protocol._parse_csv_ints, "0,2"),
        (protocol._parse_csv_nonnegative_ints, "-1,2"),
        (protocol._parse_csv_nonnegative_ints, "0,0"),
        (protocol._parse_csv_floats, "nan,0.1"),
        (protocol._parse_csv_floats, "0.001,0.001"),
        (protocol._parse_csv_floats, "-0.1,0.1"),
    ],
)
def test_grid_parsers_reject_empty_nonpositive_nonfinite_or_duplicate_values(
    parser, value
):
    with pytest.raises(argparse.ArgumentTypeError):
        parser(value)


def test_preregistered_splits_and_grids_remain_fixed():
    assert protocol.TRAIN_SPLIT == "train[:80%]"
    assert protocol.VALIDATION_SPLIT == "train[80%:90%]"
    assert protocol.HOLDOUT_SPLIT == "train[90%:]"
    assert protocol.DEFAULT_STEP_GRID == (2, 5, 10)
    assert protocol.DEFAULT_LR_GRID == (3e-4, 1e-3, 3e-3)
    assert set(protocol.DEFAULT_TUNE_SEEDS).isdisjoint(protocol.DEFAULT_FINAL_SEEDS)


def test_seed_parser_accepts_zero():
    assert protocol._parse_csv_nonnegative_ints("0,1000") == (0, 1000)


def test_candidate_summary_rejects_nonfinite_or_unpaired_observations():
    with pytest.raises(ValueError, match="finite probabilities"):
        protocol.select_candidate([_observation(2, 0.001, 1, float("nan"))])

    with pytest.raises(ValueError, match="same seeds"):
        protocol.select_candidate(
            [
                _observation(2, 0.001, 1, 0.3),
                _observation(5, 0.001, 2, 0.4),
            ]
        )


def test_tuning_runs_complete_grid_without_endpoint_evaluation(monkeypatch):
    calls = []

    def fake_training(**kwargs):
        calls.append(kwargs)
        accuracy = 0.3 + kwargs["inference"].config["infer_steps"] / 100
        return protocol.TrainingResult(
            seed=kwargs["seed"],
            name=kwargs["name"],
            training_seconds=1.0,
            curve=(protocol.CurvePoint(3, 1.0, accuracy, 1.0),),
        )

    monkeypatch.setattr(protocol, "_run_training", fake_training)
    args = SimpleNamespace(
        steps_grid=(2, 5),
        lr_grid=(0.001, 0.003),
        tune_seeds=(11, 12),
        epc_eta=0.1,
        tune_epochs=3,
        weight_decay=0.01,
        batch_size=256,
    )

    protocol.run_tuning(args)

    assert len(calls) == 8
    assert all(call["evaluate_holdout"] is False for call in calls)
    assert {
        (
            call["inference"].config["infer_steps"],
            call["learning_rate"],
            call["seed"],
        )
        for call in calls
    } == {
        (steps, learning_rate, seed)
        for steps in args.steps_grid
        for learning_rate in args.lr_grid
        for seed in args.tune_seeds
    }


def test_final_runs_both_arms_with_endpoints_for_each_paired_seed(monkeypatch):
    calls = []

    def fake_training(**kwargs):
        calls.append(kwargs)
        return protocol.TrainingResult(
            seed=kwargs["seed"],
            name=kwargs["name"],
            training_seconds=1.0,
            curve=(protocol.CurvePoint(kwargs["epochs"], 1.0, 0.4, 1.0),),
            holdout_accuracy=0.4,
            official_test_accuracy=0.4,
        )

    summaries = []
    monkeypatch.setattr(protocol, "_run_training", fake_training)
    monkeypatch.setattr(protocol, "_print_final_summary", summaries.append)
    args = SimpleNamespace(
        epc_steps=5,
        epc_lr=0.003,
        epc_eta=0.1,
        epc_epochs=13,
        spc_steps=120,
        spc_lr=0.001,
        spc_eta=0.1,
        spc_epochs=2,
        final_seeds=(0, 1000, 2000),
        weight_decay=0.01,
        batch_size=256,
    )

    protocol.run_final(args)

    assert len(calls) == 6
    assert all(call["evaluate_holdout"] is True for call in calls)
    assert [call["seed"] for call in calls] == [0, 0, 1000, 1000, 2000, 2000]
    assert len(summaries) == 1
    assert len(summaries[0]) == 6
