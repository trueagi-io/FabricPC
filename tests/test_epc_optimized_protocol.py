"""Protocol tests for solver-specific ePC versus sPC optimization."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "epc_resnet18_optimized.py"
)
_SPEC = importlib.util.spec_from_file_location("epc_resnet18_optimized", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(_SCRIPT_PATH)
protocol = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(protocol)


def _result(recipe, seed, accuracies, seconds=None, endpoint=None):
    if seconds is None:
        seconds = tuple(float(index + 1) for index in range(len(accuracies)))
    best = -1.0
    points = []
    best_epoch = 0
    best_seconds = 0.0
    for index, (accuracy, elapsed) in enumerate(zip(accuracies, seconds), start=1):
        if accuracy > best:
            best = accuracy
            best_epoch = index
            best_seconds = elapsed
        points.append(protocol.CurvePoint(index, elapsed, accuracy, 1.0, best))
    return protocol.TrainingResult(
        recipe=recipe,
        seed=seed,
        total_training_seconds=seconds[-1],
        curve=tuple(points),
        best_epoch=best_epoch,
        best_training_seconds=best_seconds,
        best_validation_accuracy=best,
        final_validation_accuracy=accuracies[-1],
        endpoint_accuracy=endpoint,
        endpoint_seconds=1.0 if endpoint is not None else None,
    )


def _args(**overrides):
    values = {
        "epochs": 8,
        "batch_size": 256,
        "weight_decay": 0.01,
        "warmup_epochs": 0.25,
        "floor_ratio": 0.01,
        "stability_tail_epochs": 3,
        "stability_tolerance": 0.01,
        "epc_steps_grid": (2,),
        "epc_lr_grid": (0.0001,),
        "spc_steps_grid": (120,),
        "spc_lr_grid": (0.001,),
        "decay_epochs_grid": (3, 5),
        "epc_eta": 0.1,
        "spc_eta": 0.1,
        "screen_seed": 11,
        "confirm_seed": 12,
        "shortlist_size": 1,
        "final_seeds": (0, 1, 2),
        "epc_steps": 2,
        "epc_lr": 0.0001,
        "epc_decay_epochs": 3,
        "spc_steps": 120,
        "spc_lr": 0.001,
        "spc_decay_epochs": 5,
        "target_accuracy": 0.3,
        "noninferiority_margin": 0.01,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_preregistered_candidate_spaces_are_solver_specific():
    epc = protocol.make_candidate_grid(
        "ePC",
        protocol.EPC_STEP_GRID,
        protocol.EPC_LR_GRID,
        protocol.DECAY_EPOCH_GRID,
        0.1,
    )
    spc = protocol.make_candidate_grid(
        "sPC",
        protocol.SPC_STEP_GRID,
        protocol.SPC_LR_GRID,
        protocol.DECAY_EPOCH_GRID,
        0.1,
    )

    assert len(epc) == 18
    assert len(spc) == 6
    assert {recipe.peak_learning_rate for recipe in epc} == {
        3e-5,
        1e-4,
        3e-4,
    }
    assert {recipe.peak_learning_rate for recipe in spc} == {
        3e-4,
        1e-3,
        3e-3,
    }


def test_schedule_is_defined_by_recipe_not_observation_horizon():
    recipe = protocol.Recipe("ePC", 5, 0.1, 0.0003, 3)
    schedule = protocol._learning_rate_schedule(recipe, 10, 0.25, 0.01)

    assert float(schedule(29)) > recipe.peak_learning_rate * 0.01
    assert float(schedule(30)) == pytest.approx(recipe.peak_learning_rate * 0.01)
    assert float(schedule(80)) == pytest.approx(recipe.peak_learning_rate * 0.01)


def test_stability_requires_a_flat_tail_near_the_overall_best():
    recipe = protocol.Recipe("ePC", 5, 0.1, 0.0001, 3)
    stable = _result(recipe, 1, (0.2, 0.3, 0.35, 0.36, 0.36, 0.355, 0.358, 0.357))
    collapsed = _result(recipe, 2, (0.2, 0.3, 0.35, 0.36, 0.2, 0.12, 0.11, 0.10))

    assert protocol.is_stable(stable)
    assert not protocol.is_stable(collapsed)


def test_shortlist_prioritizes_stability_before_peak_accuracy():
    stable_recipe = protocol.Recipe("ePC", 2, 0.1, 0.0001, 3)
    collapsed_recipe = protocol.Recipe("ePC", 10, 0.1, 0.0003, 5)
    stable = _result(stable_recipe, 1, (0.2, 0.29, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30))
    collapsed = _result(
        collapsed_recipe, 1, (0.2, 0.35, 0.40, 0.30, 0.20, 0.11, 0.10, 0.10)
    )

    selected = protocol.shortlist_candidates((collapsed, stable), 1)

    assert selected == (stable_recipe,)


def test_recipe_selection_uses_mean_best_accuracy_across_both_tuning_seeds():
    first = protocol.Recipe("ePC", 2, 0.1, 0.0001, 3)
    second = protocol.Recipe("ePC", 5, 0.1, 0.0001, 3)
    stable_tail = (0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30)
    results = {
        first: (
            _result(first, 1, stable_tail),
            _result(first, 2, tuple(value + 0.01 for value in stable_tail)),
        ),
        second: (
            _result(second, 1, tuple(value + 0.02 for value in stable_tail)),
            _result(second, 2, tuple(value + 0.02 for value in stable_tail)),
        ),
    }

    selected = protocol.select_recipe(results)

    assert selected.recipe == second
    assert selected.mean_best_validation_accuracy == pytest.approx(0.32)


def test_target_is_floor_of_selected_spc_stable_tail():
    recipe = protocol.Recipe("sPC", 120, 0.1, 0.001, 5)
    first = _result(recipe, 1, (0.2, 0.3, 0.31, 0.32, 0.33, 0.331, 0.329, 0.330))
    second = _result(recipe, 2, (0.2, 0.3, 0.31, 0.32, 0.325, 0.326, 0.324, 0.325))

    target = protocol.stable_plateau_floor((first, second), 3)

    assert target == pytest.approx(0.324)
    assert protocol.time_to_accuracy(first, target) == pytest.approx(5.0)


def test_tuning_runs_screen_and_confirmation_without_endpoint_access(monkeypatch):
    calls = []

    def fake_run(recipe, seed, args, evaluate_endpoint):
        calls.append((recipe, seed, evaluate_endpoint))
        base = 0.31 if recipe.solver == "ePC" else 0.32
        if recipe.decay_epochs == 5:
            base += 0.01
        return _result(recipe, seed, (base,) * args.epochs)

    monkeypatch.setattr(protocol, "_run_recipe", fake_run)

    protocol.run_tuning(_args())

    assert len(calls) == 6
    assert all(evaluate_endpoint is False for _, _, evaluate_endpoint in calls)
    assert {seed for _, seed, _ in calls} == {11, 12}


def test_final_runs_paired_locked_recipes_with_endpoint_enabled(monkeypatch):
    calls = []

    def fake_run(recipe, seed, args, evaluate_endpoint):
        calls.append((recipe, seed, evaluate_endpoint))
        return _result(recipe, seed, (0.3,) * args.epochs, endpoint=0.3)

    summaries = []
    monkeypatch.setattr(protocol, "_run_recipe", fake_run)
    monkeypatch.setattr(
        protocol,
        "_final_summary",
        lambda results, target, margin: summaries.append((results, target, margin)),
    )

    protocol.run_final(_args())

    assert len(calls) == 6
    assert all(evaluate_endpoint is True for _, _, evaluate_endpoint in calls)
    assert [seed for _, seed, _ in calls] == [0, 0, 1, 1, 2, 2]
    assert len(summaries) == 1


def test_endpoint_evaluates_the_best_validation_checkpoint_once(monkeypatch):
    recipe = protocol.Recipe("ePC", 2, 0.1, 0.0001, 3)
    loaded_splits = []
    endpoint_params = []

    class FakeLoader:
        def __init__(self, split, **_):
            self.split = split
            loaded_splits.append(split)

        def __len__(self):
            return 1

    def fake_train_pcn(*, epoch_callback, structure, config, **_):
        for epoch, weight in enumerate((1, 2, 3)):
            epoch_callback(epoch, {"weight": weight}, structure, config, None)

    def fake_evaluate(params, _, loader, __, ___):
        if loader.split == protocol.VALIDATION_SPLIT:
            return {
                "accuracy": {1: 0.2, 2: 0.4, 3: 0.3}[params["weight"]],
                "energy": 1.0,
            }
        endpoint_params.append(params)
        return {"accuracy": 0.42, "energy": 1.0}

    monkeypatch.setattr(protocol, "Cifar10Loader", FakeLoader)
    monkeypatch.setattr(protocol, "_create_model", lambda *_: ({"weight": 0}, object()))
    monkeypatch.setattr(protocol, "_inference", lambda _: object())
    monkeypatch.setattr(protocol, "_optimizer", lambda *_: object())
    monkeypatch.setattr(protocol, "train_pcn", fake_train_pcn)
    monkeypatch.setattr(protocol, "evaluate_pcn", fake_evaluate)

    result = protocol._run_training(
        recipe=recipe,
        seed=1,
        epochs=3,
        weight_decay=0.01,
        batch_size=256,
        warmup_epochs=0.25,
        floor_ratio=0.01,
        stability_tail_epochs=3,
        stability_tolerance=0.01,
        evaluate_endpoint=True,
    )

    assert result.best_epoch == 2
    assert result.best_validation_accuracy == pytest.approx(0.4)
    assert result.endpoint_accuracy == pytest.approx(0.42)
    assert endpoint_params == [{"weight": 2}]
    assert loaded_splits == [
        protocol.TRAIN_SPLIT,
        protocol.VALIDATION_SPLIT,
        protocol.HOLDOUT_SPLIT,
    ]


def test_final_summary_passes_only_with_noninferiority_and_faster_target(capsys):
    epc = protocol.Recipe("ePC", 2, 0.1, 0.0001, 3)
    spc = protocol.Recipe("sPC", 120, 0.1, 0.001, 5)
    results = [
        _result(epc, 1, (0.2, 0.31), (1.0, 2.0), endpoint=0.496),
        _result(spc, 1, (0.2, 0.31), (2.0, 4.0), endpoint=0.500),
        _result(epc, 2, (0.2, 0.31), (1.1, 2.1), endpoint=0.484),
        _result(spc, 2, (0.2, 0.31), (2.1, 4.1), endpoint=0.490),
    ]

    protocol._final_summary(results, target=0.3, margin=0.01)

    output = capsys.readouterr().out
    assert "noninferior=True" in output
    assert "all_reached=True faster=True" in output
    assert "overall_pass=True" in output
