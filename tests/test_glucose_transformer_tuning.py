"""Behavioral coverage for glucose Optuna tuning and GPU admission."""
from __future__ import annotations

import json
import pickle

import numpy as np
import optuna

from examples.glucose_transformer_tuning import (
    TrialSettings,
    _run_trial_attempt,
    admitted_worker_count,
    create_study,
    suggest_pc_dynamics,
)


FIXED_DYNAMICS = {
    "seq_len": 64,
    "depth": 1,
    "num_heads": 2,
    "lr": 0.005,
    "eta_infer": 0.0002,
    "infer_steps": 8,
    "max_infer_norm": 0.5,
    "grad_clip": 1.0,
    "lr_decay_epochs": 5,
    "weight_init_std": 0.02,
    "energy": "gaussian",
    "ipc": False,
    "infer_optimizer": "sgd",
    "huber_delta": 1.0,
}


def test_search_space_and_journal_resume(tmp_path) -> None:
    trial = optuna.trial.FixedTrial(FIXED_DYNAMICS)
    assert suggest_pc_dynamics(trial) == FIXED_DYNAMICS

    journal = tmp_path / "optuna.log"
    first = create_study(journal, "resume-test")
    first.enqueue_trial(FIXED_DYNAMICS)
    resumed = create_study(journal, "resume-test")

    assert resumed.study_name == first.study_name
    assert len(resumed.trials) == 1
    assert resumed.trials[0].state == optuna.trial.TrialState.WAITING
    assert resumed.trials[0].system_attrs["fixed_params"] == FIXED_DYNAMICS


def test_gpu_admission_uses_half_vram_without_extra_safety_factor() -> None:
    admitted = admitted_worker_count(
        allocations={101: 804, 102: 792, 201: 788},
        managed_pids={201},
        active_workers=1,
        gpu_memory_budget_mib=8192,
        estimated_trial_memory_mib=900,
        max_workers=8,
    )

    # External jobs consume 1,596 MiB. Seven 900-MiB slots fit, one is active.
    assert admitted == 6


def test_gpu_admission_adapts_to_observed_worker_peak() -> None:
    admitted = admitted_worker_count(
        allocations={101: 804, 201: 1740, 202: 1738},
        managed_pids={201, 202},
        active_workers=2,
        gpu_memory_budget_mib=8192,
        estimated_trial_memory_mib=900,
        max_workers=8,
    )

    # The measured 1,740-MiB worker footprint supersedes the optimistic estimate.
    assert admitted == 2


def test_real_data_trial_writes_unique_best_artifacts(tmp_path) -> None:
    settings = TrialSettings(
        run_dir=tmp_path,
        seq_len=8,
        horizon=2,
        depth=1,
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        batch_size=256,
        max_epochs=3,
        min_pruning_epochs=1,
        patience=3,
        warmup_steps=1,
        max_batches_per_epoch=1,
        max_validation_batches=1,
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
    )
    study.enqueue_trial(FIXED_DYNAMICS)
    study.optimize(
        lambda trial: _run_trial_attempt(
            trial,
            suggest_pc_dynamics(trial),
            settings,
            batch_size=settings.batch_size,
        ),
        n_trials=1,
    )

    completed = study.trials[0]
    trial_dir = tmp_path / "trials" / "trial_0000"
    config = json.loads((trial_dir / "config.json").read_text())
    history = json.loads((trial_dir / "history.json").read_text())

    assert completed.state == optuna.trial.TrialState.COMPLETE
    assert completed.value is not None and np.isfinite(completed.value)
    assert config["include_output_scaling"] is True
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert all(row["step"] > 0 for row in history)
    assert all(np.isfinite(row["train_mae_mg_dl"]) for row in history)
    assert all(np.isfinite(row["train_mae_std_mg_dl"]) for row in history)
    assert all(
        row["train_mae_min_mg_dl"]
        <= row["train_mae_mg_dl"]
        <= row["train_mae_max_mg_dl"]
        for row in history
    )
    assert (trial_dir / "best_params.pkl").is_file()
    checkpoint_path = trial_dir / "checkpoint.pkl"
    assert checkpoint_path.is_file()
    with checkpoint_path.open("rb") as file:
        checkpoint = pickle.load(file)
    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == history[-1]["step"]
    assert checkpoint["history"] == history


def test_real_data_trial_honors_optuna_pruning(tmp_path) -> None:
    settings = TrialSettings(
        run_dir=tmp_path,
        seq_len=8,
        horizon=2,
        depth=1,
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        batch_size=256,
        max_epochs=3,
        min_pruning_epochs=1,
        patience=3,
        warmup_steps=1,
        max_batches_per_epoch=1,
        max_validation_batches=1,
    )
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.ThresholdPruner(upper=0.0),
    )
    study.enqueue_trial(FIXED_DYNAMICS)
    study.optimize(
        lambda trial: _run_trial_attempt(
            trial,
            suggest_pc_dynamics(trial),
            settings,
            batch_size=settings.batch_size,
        ),
        n_trials=1,
    )

    pruned = study.trials[0]
    assert pruned.state == optuna.trial.TrialState.PRUNED
    assert "ThresholdPruner" in pruned.user_attrs["prune_reason"]
