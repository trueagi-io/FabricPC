"""Behavioral coverage for glucose Hopfield Optuna tuning."""
from __future__ import annotations

import json

import numpy as np
import optuna

from examples.glucose_hopfield_tuning import (
    CHAMPION_PC_PARAMS,
    TARGET_OPTUNA_MAE,
    TrialSettings,
    _parse_hopfield_strength,
    _run_trial_attempt,
    admitted_worker_count,
    create_study,
    enqueue_hopfield_baselines,
    suggest_hopfield_dynamics,
)


FIXED_DYNAMICS = {
    **CHAMPION_PC_PARAMS,
    "variant": "embed-storkey",
    "hopfield_strength": "1.0",
}


def test_search_space_and_journal_resume(tmp_path) -> None:
    trial = optuna.trial.FixedTrial(FIXED_DYNAMICS)
    assert suggest_hopfield_dynamics(trial) == FIXED_DYNAMICS
    assert _parse_hopfield_strength("learnable") is None
    assert _parse_hopfield_strength("2.0") == 2.0
    assert TARGET_OPTUNA_MAE == 19.876

    journal = tmp_path / "optuna.log"
    first = create_study(journal, "hopfield-resume-test")
    enqueue_hopfield_baselines(first)
    resumed = create_study(journal, "hopfield-resume-test")

    assert resumed.study_name == first.study_name
    assert len(resumed.trials) == 8
    assert resumed.trials[0].state == optuna.trial.TrialState.WAITING


def test_gpu_admission_respects_three_workers() -> None:
    admitted = admitted_worker_count(
        allocations={},
        managed_pids=set(),
        active_workers=0,
        gpu_memory_budget_mib=12000,
        estimated_trial_memory_mib=2500,
        max_workers=3,
    )
    assert admitted == 3


def test_real_data_trial_writes_hopfield_artifacts(tmp_path) -> None:
    settings = TrialSettings(
        run_dir=tmp_path,
        seq_len=8,
        horizon=2,
        depth=1,
        embed_dim=8,
        num_heads=1,
        mlp_dim=16,
        batch_size=256,
        max_epochs=3,
        min_pruning_epochs=1,
        patience=3,
        warmup_steps=1,
        max_batches_per_epoch=1,
        max_validation_batches=1,
    )
    dynamics = {
        "seq_len": 8,
        "depth": 1,
        "num_heads": 1,
        "variant": "embed-storkey",
        "hopfield_strength": "1.0",
        "lr": 0.003,
        "eta_infer": 1.5e-5,
        "infer_steps": 8,
        "max_infer_norm": 1.0,
        "grad_clip": 0.5,
        "lr_decay_epochs": 5,
        "weight_init_std": 0.02,
        "seed_offset": 0,
    }
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
    )
    study.enqueue_trial(dynamics)
    study.optimize(
        lambda trial: _run_trial_attempt(
            trial,
            dynamics,
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
    assert config["variant"] == "embed-storkey"
    assert config["include_output_scaling"] is True
    assert config["target_optuna_mae"] == TARGET_OPTUNA_MAE
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert (trial_dir / "best_params.pkl").exists()
