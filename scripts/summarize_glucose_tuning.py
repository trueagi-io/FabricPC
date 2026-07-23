"""Print a compact Optuna + trial-history summary for glucose tuning."""

from __future__ import annotations

import json
from pathlib import Path

from examples.glucose_transformer_tuning_update_budget import create_study

RUN_DIR = Path("runs/glucose_tuning")
STUDY_NAME = "glucose_transformer_pc"
PARAM_KEYS = [
    "seq_len",
    "depth",
    "num_heads",
    "lr",
    "eta_infer",
    "infer_steps",
    "max_infer_norm",
    "grad_clip",
    "weight_init_std",
]


def main() -> None:
    study = create_study(RUN_DIR / "optuna_journal.log", STUDY_NAME)
    terminal = {"COMPLETE", "PRUNED", "FAIL"}
    finished = sum(1 for trial in study.trials if trial.state.name in terminal)
    print(f"trials={len(study.trials)} finished={finished}")
    if any(trial.state.name == "COMPLETE" for trial in study.trials):
        best = study.best_trial
        print(
            f"best_so_far trial={best.number} "
            f"val_mae={best.value:.4f} params={best.params}"
        )

    for trial in study.trials:
        params = {key: trial.params.get(key) for key in PARAM_KEYS}
        print(
            f"trial={trial.number:02d} state={trial.state.name:9s} "
            f"value={trial.value} attrs={dict(trial.user_attrs)}"
        )
        print(f"  params={params}")
        history_path = RUN_DIR / "trials" / f"trial_{trial.number:04d}" / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        maes = [f"{row['update']}:{row['mae_mg_dl']:.2f}" for row in history]
        print(f"  mae_trace={' -> '.join(maes)}")


if __name__ == "__main__":
    main()
