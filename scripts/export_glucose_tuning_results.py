"""Export Optuna glucose tuning trials to a JSON snapshot for reporting."""

from __future__ import annotations

import json
from pathlib import Path

from examples.glucose_transformer_tuning_update_budget import create_study

RUN_DIR = Path("runs/glucose_tuning")
STUDY_NAME = "glucose_transformer_pc"
OUT_PATH = RUN_DIR / "results_snapshot.json"
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
    "weight_decay",
]


def main() -> None:
    study = create_study(RUN_DIR / "optuna_journal.log", STUDY_NAME)
    rows: list[dict] = []
    for trial in study.trials:
        history_path = (
            RUN_DIR / "trials" / f"trial_{trial.number:04d}" / "history.json"
        )
        history = (
            json.loads(history_path.read_text()) if history_path.exists() else []
        )
        best_mae = min((row["mae_mg_dl"] for row in history), default=None)
        rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "optuna_value": trial.value,
                "best_history_mae": best_mae,
                "params": {key: trial.params.get(key) for key in PARAM_KEYS},
                "user_attrs": dict(trial.user_attrs),
                "history": history,
            }
        )

    completed = [row for row in rows if row["state"] == "COMPLETE"]
    completed.sort(
        key=lambda row: (
            row["best_history_mae"]
            if row["best_history_mae"] is not None
            else float("inf")
        )
    )
    snapshot = {
        "study_name": STUDY_NAME,
        "n_trials": len(rows),
        "n_complete": sum(1 for row in rows if row["state"] == "COMPLETE"),
        "n_pruned": sum(1 for row in rows if row["state"] == "PRUNED"),
        "n_fail": sum(1 for row in rows if row["state"] == "FAIL"),
        "n_running": sum(1 for row in rows if row["state"] == "RUNNING"),
        "best_complete": completed[0] if completed else None,
        "trials": rows,
    }
    OUT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"wrote {OUT_PATH}")
    if completed:
        best = completed[0]
        print(
            f"best_complete trial={best['trial']} "
            f"mae={best['best_history_mae']:.4f} params={best['params']}"
        )


if __name__ == "__main__":
    main()
