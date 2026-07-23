"""Generate Hopfield Optuna reports with full trial dumps and cross-run compare.

Writes under the study ``--run-dir`` and copies Markdown/HTML into
``docs/reports/``. Includes **all** trial states (COMPLETE / PRUNED / FAIL /
RUNNING), within-trial epoch stages, and comparison vs the transformer
phase-4 champion (19.876 Optuna val MAE).

```bash
uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all
```
"""
from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    add_completion=False,
    help="Render glucose Hopfield Optuna MD/HTML reports.",
)

DEFAULT_RUN_DIR = Path("runs/glucose_hopfield_tuning_v1")
DEFAULT_STUDY_NAME = "glucose_hopfield_pc_v1"
TRANSFORMER_CHAMPION_MAE = 19.876

PARAM_GLOSSARY: dict[str, str] = {
    "seq_len": "Input sequence length (number of 5-min glucose readings fed to the model). Longer = more history but heavier.",
    "depth": "Number of transformer blocks stacked. More depth = more capacity but slower inference and higher memory.",
    "num_heads": "Number of parallel attention heads in multi-scale self-attention.",
    "variant": "Hopfield memory placement: 'baseline' (no Hopfield), 'embed-storkey' (after embedding), 'forecast-storkey' (before forecast head), 'projection' (linear memory).",
    "hopfield_strength": "Hopfield interaction strength. 'learnable' = optimised during training; a number = fixed scaling factor for the associative memory.",
    "lr": "Outer learning rate for weight updates (Adam/AdamW). Controls how fast weights move each step.",
    "eta_infer": "PC inference learning rate. Step size for the inner-loop SGD that updates latent activations to minimise prediction errors.",
    "infer_steps": "Number of PC inference iterations per forward pass. More steps = tighter energy minimisation but slower training.",
    "max_infer_norm": "Maximum gradient norm during PC inference. Clips the inner-loop update to prevent latent activations from exploding.",
    "grad_clip": "Global gradient clipping threshold for weight updates. Stabilises training by capping large gradients.",
    "lr_decay_epochs": "Epoch at which the learning rate starts cosine decay toward zero. Later = longer warm phase at full LR.",
    "weight_init_std": "Standard deviation for weight initialisation (Normal). Smaller = more conservative start; interacts with depth.",
    "seed_offset": "Random seed offset for reproducibility and diversity across trials with otherwise similar configs.",
}
TRANSFORMER_PHASES: list[dict[str, Any]] = [
    {
        "phase": "1 broad",
        "run_dir": "runs/glucose_tuning",
        "best_val_mae": 20.3776,
    },
    {
        "phase": "2 refined",
        "run_dir": "runs/glucose_tuning_pc_v2",
        "best_val_mae": 20.6780,
    },
    {
        "phase": "3 local",
        "run_dir": "runs/glucose_tuning_pc_local",
        "best_val_mae": 20.8670,
    },
    {
        "phase": "4 breakthrough",
        "run_dir": "runs/glucose_tuning_pc_breakthrough",
        "best_val_mae": 19.8760,
    },
]
PARAM_KEYS = [
    "seq_len",
    "depth",
    "num_heads",
    "variant",
    "hopfield_strength",
    "lr",
    "eta_infer",
    "infer_steps",
    "max_infer_norm",
    "grad_clip",
    "lr_decay_epochs",
    "weight_init_std",
    "seed_offset",
]


def _resolve_study_name(run_dir: Path, study_name: str | None) -> str:
    if study_name:
        return study_name
    config_path = run_dir / "coordinator_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        configured = config.get("study_name")
        if isinstance(configured, str) and configured:
            return configured
    return DEFAULT_STUDY_NAME


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _export_snapshot(run_dir: Path, study_name: str | None = None) -> Path:
    from examples.glucose_hopfield_tuning import create_study

    resolved_name = _resolve_study_name(run_dir, study_name)
    study = create_study(run_dir / "optuna_journal.log", resolved_name)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        history_path = (
            run_dir / "trials" / f"trial_{trial.number:04d}" / "history.json"
        )
        history = (
            json.loads(history_path.read_text()) if history_path.exists() else []
        )
        best_mae = min((row["mae_mg_dl"] for row in history), default=None)
        if best_mae is None and trial.value is not None:
            best_mae = float(trial.value)
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
    completed = [
        row for row in rows if row["state"] == "COMPLETE" and row["best_history_mae"]
    ]
    completed.sort(key=lambda row: float(row["best_history_mae"]))
    snapshot = {
        "study_name": resolved_name,
        "n_trials": len(rows),
        "n_complete": sum(1 for row in rows if row["state"] == "COMPLETE"),
        "n_pruned": sum(1 for row in rows if row["state"] == "PRUNED"),
        "n_fail": sum(1 for row in rows if row["state"] == "FAIL"),
        "n_running": sum(1 for row in rows if row["state"] == "RUNNING"),
        "best_complete": completed[0] if completed else None,
        "target_optuna_mae": TRANSFORMER_CHAMPION_MAE,
        "trials": rows,
    }
    out_path = run_dir / "results_snapshot.json"
    out_path.write_text(json.dumps(snapshot, indent=2))
    typer.echo(f"wrote {out_path}")
    return out_path


def _load_confirm(confirm_dir: Path) -> dict[str, Any] | None:
    config_path = confirm_dir / "config.json"
    history_path = confirm_dir / "history.csv"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    epochs: list[dict[str, float]] = []
    if history_path.exists():
        with history_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                epochs.append(
                    {
                        "epoch": float(row["epoch"]),
                        "mae_mg_dl": float(row["mae_mg_dl"]),
                        "rmse_mg_dl": float(row["rmse_mg_dl"]),
                        "mard_percent": float(row["mard_percent"]),
                    }
                )
    return {"config": config, "epochs": epochs}


def _load_settings(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "coordinator_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _build_payload(
    snapshot: dict[str, Any],
    confirm: dict[str, Any] | None,
    run_dir: Path,
) -> dict[str, Any]:
    settings = _load_settings(run_dir)
    all_trials = snapshot["trials"]
    completed = [
        trial
        for trial in all_trials
        if trial["state"] == "COMPLETE" and trial["best_history_mae"] is not None
    ]
    completed.sort(key=lambda trial: float(trial["best_history_mae"]))
    best = completed[0] if completed else None
    best_mae = float(best["best_history_mae"]) if best else None

    stage_traces = []
    for trial in completed[:6]:
        points = []
        for row in trial["history"]:
            points.append(
                {
                    "epoch": int(row["epoch"]),
                    "mae_mg_dl": float(row["mae_mg_dl"]),
                    "train_mae_mg_dl": float(row.get("train_mae_mg_dl", 0.0)),
                }
            )
        stage_traces.append(
            {
                "trial": trial["trial"],
                "best_mae": float(trial["best_history_mae"]),
                "variant": trial["params"].get("variant"),
                "hopfield_strength": trial["params"].get("hopfield_strength"),
                "params": trial["params"],
                "points": points,
            }
        )

    best_trial_path = run_dir / "best_trial.json"
    best_trial_file = (
        json.loads(best_trial_path.read_text()) if best_trial_path.exists() else None
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_name": snapshot.get("study_name", DEFAULT_STUDY_NAME),
        "settings": settings,
        "counts": {
            "n_trials": snapshot["n_trials"],
            "n_complete": snapshot["n_complete"],
            "n_pruned": snapshot["n_pruned"],
            "n_fail": snapshot["n_fail"],
            "n_running": snapshot["n_running"],
        },
        "target_optuna_mae": TRANSFORMER_CHAMPION_MAE,
        "best_complete": best,
        "best_trial_file": best_trial_file,
        "beat_transformer_champion": (
            None if best_mae is None else best_mae < TRANSFORMER_CHAMPION_MAE
        ),
        "delta_vs_transformer_champion": (
            None if best_mae is None else best_mae - TRANSFORMER_CHAMPION_MAE
        ),
        "transformer_phase_comparison": TRANSFORMER_PHASES,
        "leaderboard": [
            {
                "trial": trial["trial"],
                "best_mae": float(trial["best_history_mae"]),
                "state": trial["state"],
                "params": trial["params"],
                "delta_vs_target": float(trial["best_history_mae"])
                - TRANSFORMER_CHAMPION_MAE,
                "stop_reason": trial.get("user_attrs", {}).get("stop_reason"),
                "prune_reason": trial.get("user_attrs", {}).get("prune_reason"),
            }
            for trial in completed
        ],
        "all_trials": all_trials,
        "stage_traces": stage_traces,
        "confirm": confirm,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    best = payload.get("best_complete")
    settings = payload.get("settings") or {}
    lines: list[str] = [
        "# Glucose Hopfield Optuna progress report",
        "",
        f"Generated: `{payload['generated_at']}`  ",
        f"Study: `{payload['study_name']}`  ",
        "Mode: **predictive coding (PC) only** + Hopfield memory variants",
        "",
        "## Goal",
        "",
        f"Beat transformer phase-4 Optuna champion **"
        f"{TRANSFORMER_CHAMPION_MAE:.3f}** validation MAE "
        "(same `prepare_data` split / comparable protocol).",
        "",
        "## Settings used",
        "",
        "| Setting | Value |",
        "|---------|------:|",
        f"| Protocol | {settings.get('protocol', 'epoch-based Hyperband')} |",
        f"| max_workers | {settings.get('max_workers', '—')} |",
        f"| n_trials | {settings.get('n_trials', '—')} |",
        f"| max_epochs | {settings.get('max_epochs', '—')} |",
        f"| min_pruning_epochs | {settings.get('min_pruning_epochs', '—')} |",
        f"| patience | {settings.get('patience', '—')} |",
        f"| batch_size | {settings.get('batch_size', '—')} |",
        f"| embed_dim / mlp_dim | {settings.get('embed_dim', '—')} / "
        f"{settings.get('mlp_dim', '—')} |",
        f"| gpu_memory_budget_mib | {settings.get('gpu_memory_budget_mib', '—')} |",
        f"| estimated_trial_memory_mib | "
        f"{settings.get('estimated_trial_memory_mib', '—')} |",
        f"| target_optuna_mae | {settings.get('target_optuna_mae', TRANSFORMER_CHAMPION_MAE)} |",
        "",
        f"Search notes: {settings.get('search_notes', '—')}",
        "",
        "## Cross-run comparison (transformer phases vs Hopfield)",
        "",
        "| Run | Best val MAE | Δ vs Hopfield best |",
        "|-----|-------------:|-------------------:|",
    ]
    hopfield_best = (
        float(best["best_history_mae"]) if best is not None else None
    )
    for phase in payload["transformer_phase_comparison"]:
        delta = (
            "—"
            if hopfield_best is None
            else _fmt(float(phase["best_val_mae"]) - hopfield_best, 4)
        )
        lines.append(
            f"| Transformer {phase['phase']} (`{phase['run_dir']}`) | "
            f"{_fmt(phase['best_val_mae'], 4)} | {delta} |"
        )
    if hopfield_best is not None:
        lines.append(
            f"| **Hopfield this study** | **{_fmt(hopfield_best, 4)}** | 0 |"
        )
        beat = payload.get("beat_transformer_champion")
        delta = payload.get("delta_vs_transformer_champion")
        lines.extend(
            [
                "",
                f"Beat 19.876 target: **{'YES' if beat else 'NO'}** "
                f"(Δ {_fmt(delta, 4)}).",
            ]
        )

    lines.extend(
        [
            "",
            "## Study summary (all trial states)",
            "",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Trials recorded | {counts['n_trials']} |",
            f"| Complete | {counts['n_complete']} |",
            f"| Pruned | {counts['n_pruned']} |",
            f"| Failed | {counts['n_fail']} |",
            f"| Running | {counts['n_running']} |",
        ]
    )
    if best is not None:
        params = best["params"]
        lines.extend(
            [
                f"| Best complete trial | {best['trial']} |",
                f"| Best val MAE | {_fmt(best['best_history_mae'], 4)} |",
                f"| Best variant | {params.get('variant')} |",
                f"| Best hopfield_strength | {params.get('hopfield_strength')} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Top model architectures",
            "",
        ]
    )
    for rank, row in enumerate(payload["leaderboard"][:5], 1):
        params = row["params"]
        variant = params.get("variant", "?")
        strength = params.get("hopfield_strength", "?")
        if variant == "baseline":
            arch_desc = "Pure transformer (no Hopfield node)"
        elif variant == "embed-storkey":
            arch_desc = "Storkey Hopfield after embedding layer"
        elif variant == "forecast-storkey":
            arch_desc = "Storkey Hopfield before forecast head"
        elif variant == "projection":
            arch_desc = "Projection Hopfield (linear memory)"
        else:
            arch_desc = str(variant)
        if strength == "learnable":
            strength_desc = "learnable (optimised during training)"
        else:
            strength_desc = f"fixed = {strength}"
        lines.append(f"### #{rank} — Trial {row['trial']} (MAE {_fmt(row['best_mae'], 3)})")
        lines.append(f"")
        lines.append(f"- **Architecture**: {arch_desc}")
        lines.append(f"- **Hopfield strength**: {strength_desc}")
        param_parts = []
        for key in PARAM_KEYS:
            val = params.get(key)
            if val is not None:
                if isinstance(val, float):
                    param_parts.append(f"{key}={val:.4g}")
                else:
                    param_parts.append(f"{key}={val}")
        lines.append(f"- **Params**: {', '.join(param_parts)}")
        lines.append("")

    lines.extend(
        [
            "## Complete-trial leaderboard",
            "",
            "| Trial | Best MAE | Δ vs 19.876 | variant | strength | LR | η_infer | steps | seed_off |",
            "|------:|---------:|------------:|---------|----------|---:|--------:|------:|---------:|",
        ]
    )
    for row in payload["leaderboard"]:
        params = row["params"]
        lines.append(
            "| {trial} | {mae} | {delta} | {variant} | {strength} | {lr:.4g} | "
            "{eta:.4g} | {steps} | {seed} |".format(
                trial=row["trial"],
                mae=_fmt(row["best_mae"], 3),
                delta=_fmt(row["delta_vs_target"], 3),
                variant=params.get("variant"),
                strength=params.get("hopfield_strength"),
                lr=float(params.get("lr") or 0.0),
                eta=float(params.get("eta_infer") or 0.0),
                steps=params.get("infer_steps"),
                seed=params.get("seed_offset"),
            )
        )

    lines.extend(
        [
            "",
            "## All trials (every state)",
            "",
            "| Trial | State | Best MAE | variant | strength | stop / prune |",
            "|------:|-------|---------:|---------|----------|--------------|",
        ]
    )
    for trial in payload["all_trials"]:
        attrs = trial.get("user_attrs") or {}
        reason = attrs.get("stop_reason") or attrs.get("prune_reason") or attrs.get(
            "failure_reason", ""
        )
        if isinstance(reason, str) and len(reason) > 60:
            reason = reason[:57] + "..."
        lines.append(
            "| {trial} | {state} | {mae} | {variant} | {strength} | {reason} |".format(
                trial=trial["trial"],
                state=trial["state"],
                mae=_fmt(trial.get("best_history_mae"), 3),
                variant=(trial.get("params") or {}).get("variant"),
                strength=(trial.get("params") or {}).get("hopfield_strength"),
                reason=reason or "—",
            )
        )

    if payload["stage_traces"]:
        lines.extend(
            [
                "",
                "## Within-trial stage comparison (epoch MAE traces)",
                "",
            ]
        )
        for hist in payload["stage_traces"]:
            params = hist.get("params") or {}
            trace = " → ".join(
                f"e{point['epoch']}:{_fmt(point['mae_mg_dl'], 2)}"
                for point in hist["points"]
            )
            lr = params.get("lr")
            eta = params.get("eta_infer")
            steps = params.get("infer_steps")
            detail_parts = [f"{hist['variant']}", f"str={hist['hopfield_strength']}"]
            if lr is not None:
                detail_parts.append(f"lr={float(lr):.4g}")
            if eta is not None:
                detail_parts.append(f"η={float(eta):.3g}")
            if steps is not None:
                detail_parts.append(f"steps={steps}")
            lines.append(
                f"- **Trial {hist['trial']}** "
                f"({', '.join(detail_parts)}, "
                f"best {_fmt(hist['best_mae'], 3)}): {trace}"
            )

    confirm = payload.get("confirm")
    if confirm is not None:
        cfg = confirm["config"]
        lines.extend(
            [
                "",
                "## Confirmation train (epoch loop)",
                "",
                "| Metric | Value |",
                "|--------|------:|",
                f"| Best val MAE | {_fmt(cfg.get('best_val_mae_mg_dl'), 4)} |",
                f"| Test MAE | {_fmt(cfg.get('test_mae_mg_dl'), 4)} |",
                f"| Test RMSE | {_fmt(cfg.get('test_rmse_mg_dl'), 4)} |",
                f"| Test MARD (%) | {_fmt(cfg.get('test_mard_percent'), 2)} |",
                f"| Epochs run | {cfg.get('final_epoch')} |",
                "",
                "### Epoch validation MAE",
                "",
                "| Epoch | Val MAE | RMSE | MARD (%) |",
                "|------:|--------:|-----:|---------:|",
            ]
        )
        for row in confirm["epochs"]:
            lines.append(
                f"| {int(row['epoch'])} | {_fmt(row['mae_mg_dl'], 3)} | "
                f"{_fmt(row['rmse_mg_dl'], 3)} | {_fmt(row['mard_percent'], 2)} |"
            )

    lines.extend(
        [
            "",
            "## Hyperparameter glossary",
            "",
            "| Parameter | Meaning |",
            "|-----------|---------|",
        ]
    )
    for key, desc in PARAM_GLOSSARY.items():
        lines.append(f"| `{key}` | {desc} |")

    lines.extend(
        [
            "",
            "## How the model works",
            "",
            "This model extends the glucose PC transformer with **Hopfield associative memory** — a",
            "content-addressable memory layer that can store and recall learned glucose patterns.",
            "The tuning searches over where to place the Hopfield node and how strong its influence is.",
            "",
            "### Hopfield variants searched",
            "",
            "| Variant | Where the Hopfield memory sits | Intuition |",
            "|---------|-------------------------------|-----------|",
            "| `baseline` | No Hopfield node | Pure transformer (control group) |",
            "| `embed-storkey` | After the embedding layer | Memory enriches token representations before attention |",
            "| `forecast-storkey` | Before the forecast head | Memory pattern-matches right before making the prediction |",
            "| `projection` | Linear projection memory | Lightweight associative recall with learned projections |",
            "",
            "### Architecture (embed-storkey example)",
            "",
            "```",
            "Glucose Input (batch, seq_len, 1)",
            "       |",
            "  Continuous Embedding",
            "       |",
            "  [Storkey Hopfield Memory]  ← associative recall of learned patterns",
            "       |                       strength = fixed or learnable",
            "  +--[ Transformer Block ] × depth --------+",
            "  |    Multi-Scale Self-Attention (RoPE)    |",
            "  |    LN → MLP expand (GELU)               |",
            "  |    MLP contract + Residual skip          |",
            "  +------------------------------------------+",
            "       |",
            "  Regression Output Head → Glucose Forecast (60 min)",
            "```",
            "",
            "### PC inference loop (runs at every node including Hopfield)",
            "",
            "1. Predict `z_mu` from incoming activations",
            "2. Compute `error = z_latent - z_mu`",
            "3. Compute energy (Gaussian: E = 0.5 ||error||^2)",
            "4. Update `z_latent` via SGD (step size = `eta_infer`, clip = `max_infer_norm`)",
            "5. Repeat for `infer_steps` iterations",
            "",
            "## Files",
            "",
            "| Path | Role |",
            "|------|------|",
            "| `results_snapshot.json` | Full trial dump (all states) |",
            "| `report_data.json` | Structured payload |",
            "| `report.md` / `report.html` | Human-readable views |",
            "| `best_trial.json` | Coordinator winner summary |",
            "| `coordinator_config.json` | Exact settings for this run |",
            "| `trials/trial_XXXX/` | Per-trial config, history, checkpoints |",
            "",
            "Regenerate:",
            "",
            "```bash",
            "uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _svg_hopfield_architecture() -> str:
    """Inline SVG showing the Hopfield glucose PC architecture variants."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 560" '
        'role="img" aria-label="Glucose Hopfield PC Architecture Variants" '
        'style="max-width:100%;height:auto;">'
        '<rect width="920" height="560" rx="12" fill="#0f1419"/>'
        '<defs><marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L8,3 L0,6" fill="#4b5563"/></marker></defs>'
        # Title
        '<text x="460" y="30" fill="#e7ecf3" font-size="16" '
        'font-family="Segoe UI,sans-serif" font-weight="650" text-anchor="middle">'
        'Hopfield Variant Architectures (all share PC inference at every node)</text>'
        # --- Column labels ---
        '<text x="155" y="60" fill="#9aa7b8" font-size="12" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">baseline</text>'
        '<text x="155" y="74" fill="#4b5563" font-size="9.5" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">(no Hopfield — control)</text>'
        '<text x="385" y="60" fill="#f5a524" font-size="12" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">embed-storkey</text>'
        '<text x="385" y="74" fill="#4b5563" font-size="9.5" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">(memory after embedding)</text>'
        '<text x="615" y="60" fill="#f5a524" font-size="12" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">forecast-storkey</text>'
        '<text x="615" y="74" fill="#4b5563" font-size="9.5" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">(memory before head)</text>'
        '<text x="820" y="60" fill="#9353d3" font-size="12" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">projection</text>'
        '<text x="820" y="74" fill="#4b5563" font-size="9.5" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">(linear memory)</text>'

        # ========== Column 1: baseline ==========
        # Input
        '<rect x="80" y="88" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="155" y="108" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Glucose Input</text>'
        '<line x1="155" y1="118" x2="155" y2="132" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Embed
        '<rect x="80" y="132" width="150" height="30" rx="6" fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
        '<text x="155" y="152" fill="#006FEE" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Embedding</text>'
        '<line x1="155" y1="162" x2="155" y2="196" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Transformer block
        '<rect x="60" y="196" width="190" height="120" rx="8" fill="#121820" '
        'stroke="#9aa7b8" stroke-width="1" stroke-dasharray="5,3"/>'
        '<text x="76" y="214" fill="#9aa7b8" font-size="9" font-family="Segoe UI,sans-serif" '
        'font-style="italic">x depth</text>'
        '<rect x="78" y="220" width="154" height="26" rx="5" fill="#1a2332" stroke="#f5a524" stroke-width="0.8"/>'
        '<text x="155" y="238" fill="#f5a524" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">Multi-Scale MHA</text>'
        '<rect x="78" y="254" width="154" height="26" rx="5" fill="#1a2332" stroke="#9353d3" stroke-width="0.8"/>'
        '<text x="155" y="272" fill="#9353d3" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">MLP + Residual</text>'
        '<rect x="78" y="288" width="154" height="20" rx="5" fill="#1a2332" stroke="#9353d3" stroke-width="0.8"/>'
        '<text x="155" y="302" fill="#9353d3" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">MLP contract + skip</text>'
        '<line x1="155" y1="316" x2="155" y2="330" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Output
        '<rect x="80" y="330" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="155" y="350" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Forecast Head</text>'

        # ========== Column 2: embed-storkey ==========
        '<rect x="310" y="88" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="385" y="108" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Glucose Input</text>'
        '<line x1="385" y1="118" x2="385" y2="132" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="310" y="132" width="150" height="30" rx="6" fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
        '<text x="385" y="152" fill="#006FEE" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Embedding</text>'
        '<line x1="385" y1="162" x2="385" y2="176" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Hopfield node (highlighted)
        '<rect x="298" y="176" width="174" height="34" rx="6" fill="#2a1f0a" stroke="#f5a524" stroke-width="2"/>'
        '<text x="385" y="194" fill="#f5a524" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="700">Storkey Hopfield</text>'
        '<text x="385" y="206" fill="#9aa7b8" font-size="8" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">associative memory recall</text>'
        '<line x1="385" y1="210" x2="385" y2="224" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Transformer block
        '<rect x="290" y="224" width="190" height="92" rx="8" fill="#121820" '
        'stroke="#9aa7b8" stroke-width="1" stroke-dasharray="5,3"/>'
        '<text x="306" y="242" fill="#9aa7b8" font-size="9" font-family="Segoe UI,sans-serif" '
        'font-style="italic">x depth</text>'
        '<rect x="308" y="248" width="154" height="24" rx="5" fill="#1a2332" stroke="#f5a524" stroke-width="0.8"/>'
        '<text x="385" y="264" fill="#f5a524" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">Multi-Scale MHA</text>'
        '<rect x="308" y="278" width="154" height="24" rx="5" fill="#1a2332" stroke="#9353d3" stroke-width="0.8"/>'
        '<text x="385" y="294" fill="#9353d3" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">MLP + Residual</text>'
        '<line x1="385" y1="316" x2="385" y2="330" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="310" y="330" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="385" y="350" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Forecast Head</text>'

        # ========== Column 3: forecast-storkey ==========
        '<rect x="540" y="88" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="615" y="108" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Glucose Input</text>'
        '<line x1="615" y1="118" x2="615" y2="132" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="540" y="132" width="150" height="30" rx="6" fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
        '<text x="615" y="152" fill="#006FEE" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Embedding</text>'
        '<line x1="615" y1="162" x2="615" y2="196" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Transformer block
        '<rect x="520" y="196" width="190" height="92" rx="8" fill="#121820" '
        'stroke="#9aa7b8" stroke-width="1" stroke-dasharray="5,3"/>'
        '<text x="536" y="214" fill="#9aa7b8" font-size="9" font-family="Segoe UI,sans-serif" '
        'font-style="italic">x depth</text>'
        '<rect x="538" y="220" width="154" height="24" rx="5" fill="#1a2332" stroke="#f5a524" stroke-width="0.8"/>'
        '<text x="615" y="238" fill="#f5a524" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">Multi-Scale MHA</text>'
        '<rect x="538" y="250" width="154" height="24" rx="5" fill="#1a2332" stroke="#9353d3" stroke-width="0.8"/>'
        '<text x="615" y="268" fill="#9353d3" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">MLP + Residual</text>'
        '<line x1="615" y1="288" x2="615" y2="302" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Hopfield node (highlighted)
        '<rect x="528" y="302" width="174" height="34" rx="6" fill="#2a1f0a" stroke="#f5a524" stroke-width="2"/>'
        '<text x="615" y="320" fill="#f5a524" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="700">Storkey Hopfield</text>'
        '<text x="615" y="332" fill="#9aa7b8" font-size="8" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">pattern match before forecast</text>'
        '<line x1="615" y1="336" x2="615" y2="350" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="540" y="350" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="615" y="370" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Forecast Head</text>'

        # ========== Column 4: projection ==========
        '<rect x="745" y="88" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="820" y="108" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Glucose Input</text>'
        '<line x1="820" y1="118" x2="820" y2="132" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="745" y="132" width="150" height="30" rx="6" fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
        '<text x="820" y="152" fill="#006FEE" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Embedding</text>'
        '<line x1="820" y1="162" x2="820" y2="196" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Transformer block
        '<rect x="725" y="196" width="190" height="92" rx="8" fill="#121820" '
        'stroke="#9aa7b8" stroke-width="1" stroke-dasharray="5,3"/>'
        '<text x="741" y="214" fill="#9aa7b8" font-size="9" font-family="Segoe UI,sans-serif" '
        'font-style="italic">x depth</text>'
        '<rect x="743" y="220" width="154" height="24" rx="5" fill="#1a2332" stroke="#f5a524" stroke-width="0.8"/>'
        '<text x="820" y="238" fill="#f5a524" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">Multi-Scale MHA</text>'
        '<rect x="743" y="250" width="154" height="24" rx="5" fill="#1a2332" stroke="#9353d3" stroke-width="0.8"/>'
        '<text x="820" y="268" fill="#9353d3" font-size="9" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">MLP + Residual</text>'
        '<line x1="820" y1="288" x2="820" y2="302" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        # Projection memory (different color)
        '<rect x="733" y="302" width="174" height="34" rx="6" fill="#1a1232" stroke="#9353d3" stroke-width="2"/>'
        '<text x="820" y="320" fill="#9353d3" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="700">Projection Memory</text>'
        '<text x="820" y="332" fill="#9aa7b8" font-size="8" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">lightweight linear recall</text>'
        '<line x1="820" y1="336" x2="820" y2="350" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="745" y="350" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="820" y="370" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Forecast Head</text>'

        # --- Bottom: PC inference explanation ---
        '<rect x="60" y="400" width="800" height="140" rx="10" fill="#121820" '
        'stroke="#f31260" stroke-width="1"/>'
        '<text x="460" y="424" fill="#f31260" font-size="13" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="650">Predictive Coding (PC) Inference — runs at EVERY node above</text>'
        '<text x="460" y="448" fill="#e7ecf3" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">Each node maintains a latent belief z, predicts z_mu from inputs, computes error = z - z_mu</text>'
        '<text x="460" y="470" fill="#e7ecf3" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">Inner loop: update z via SGD for infer_steps iterations '
        '(step size eta_infer, clip max_infer_norm)</text>'
        '<text x="460" y="492" fill="#e7ecf3" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">Outer loop: update weights via Adam (step size lr, clip grad_clip)</text>'
        '<text x="460" y="516" fill="#9aa7b8" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">hopfield_strength controls how strongly the memory influences activations — '
        '"learnable" lets the model optimise it during training</text>'
        '</svg>'
    )


def _render_html(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    best = payload.get("best_complete")
    best_mae = best["best_history_mae"] if best else None
    settings = payload.get("settings") or {}

    bar_labels = [str(trial["trial"]) for trial in payload["all_trials"]]
    bar_values = [
        float(trial["best_history_mae"])
        if trial.get("best_history_mae") is not None
        else None
        for trial in payload["all_trials"]
    ]
    bar_colors = []
    for trial, value in zip(payload["all_trials"], bar_values):
        if value is None:
            bar_colors.append("#4b5563")
        elif trial["state"] == "PRUNED":
            bar_colors.append("#f5a524")
        elif trial["state"] == "FAIL":
            bar_colors.append("#f31260")
        elif value <= TRANSFORMER_CHAMPION_MAE:
            bar_colors.append("#3dd68c")
        else:
            bar_colors.append("#006FEE")

    line_palette = ["#3dd68c", "#006FEE", "#f5a524", "#9353d3", "#f31260", "#a1a1aa"]
    line_datasets = []
    for index, hist in enumerate(payload["stage_traces"]):
        color = line_palette[index % len(line_palette)]
        params = hist.get("params") or {}
        lr = params.get("lr")
        eta = params.get("eta_infer")
        steps = params.get("infer_steps")
        label_parts = [f"T{hist['trial']}"]
        label_parts.append(f"{hist['variant']}")
        label_parts.append(f"str={hist['hopfield_strength']}")
        if lr is not None:
            label_parts.append(f"lr={float(lr):.4g}")
        if eta is not None:
            label_parts.append(f"η={float(eta):.3g}")
        if steps is not None:
            label_parts.append(f"steps={steps}")
        label_parts.append(f"MAE={hist['best_mae']:.2f}")
        line_datasets.append(
            {
                "label": " ".join(label_parts),
                "data": [
                    {"x": int(point["epoch"]), "y": float(point["mae_mg_dl"])}
                    for point in hist["points"]
                ],
                "borderColor": color,
                "backgroundColor": color,
                "tension": 0.2,
                "pointRadius": 3,
            }
        )

    leaderboard_rows = []
    for row in payload["leaderboard"]:
        params = row["params"]
        leaderboard_rows.append(
            "<tr>"
            f"<td>{row['trial']}</td>"
            f"<td>{_fmt(row['best_mae'], 3)}</td>"
            f"<td>{_fmt(row['delta_vs_target'], 3)}</td>"
            f"<td>{html.escape(str(params.get('variant')))}</td>"
            f"<td>{html.escape(str(params.get('hopfield_strength')))}</td>"
            f"<td>{float(params.get('lr') or 0):.4g}</td>"
            f"<td>{float(params.get('eta_infer') or 0):.4g}</td>"
            f"<td>{params.get('infer_steps')}</td>"
            "</tr>"
        )

    arch_cards = []
    for rank, row in enumerate(payload["leaderboard"][:5], 1):
        params = row["params"]
        variant = params.get("variant", "?")
        strength = params.get("hopfield_strength", "?")
        if variant == "baseline":
            arch_desc = "Pure transformer (no Hopfield node)"
        elif variant == "embed-storkey":
            arch_desc = "Storkey Hopfield after embedding layer"
        elif variant == "forecast-storkey":
            arch_desc = "Storkey Hopfield before forecast head"
        elif variant == "projection":
            arch_desc = "Projection Hopfield (linear memory)"
        else:
            arch_desc = html.escape(str(variant))
        if strength == "learnable":
            strength_desc = "learnable (optimised during training)"
        else:
            strength_desc = f"fixed = {strength}"
        param_rows = []
        for key in PARAM_KEYS:
            val = params.get(key)
            if val is not None:
                if isinstance(val, float):
                    display = f"{val:.4g}"
                else:
                    display = html.escape(str(val))
                param_rows.append(
                    f"<tr><td>{html.escape(key)}</td><td>{display}</td></tr>"
                )
        arch_cards.append(
            f'<div class="arch-card">'
            f'<div class="arch-rank">#{rank}</div>'
            f'<div class="arch-trial">Trial {row["trial"]}</div>'
            f'<div class="arch-mae">MAE {_fmt(row["best_mae"], 3)}</div>'
            f'<div class="arch-desc">{arch_desc}</div>'
            f'<div class="arch-strength">Hopfield strength: {strength_desc}</div>'
            f'<table class="arch-params">{"".join(param_rows)}</table>'
            f"</div>"
        )

    all_rows = []
    for trial in payload["all_trials"]:
        attrs = trial.get("user_attrs") or {}
        reason = attrs.get("stop_reason") or attrs.get("prune_reason") or ""
        all_rows.append(
            "<tr>"
            f"<td>{trial['trial']}</td>"
            f"<td>{html.escape(trial['state'])}</td>"
            f"<td>{_fmt(trial.get('best_history_mae'), 3)}</td>"
            f"<td>{html.escape(str((trial.get('params') or {}).get('variant')))}</td>"
            f"<td>{html.escape(str((trial.get('params') or {}).get('hopfield_strength')))}</td>"
            f"<td>{html.escape(str(reason)[:80])}</td>"
            "</tr>"
        )

    phase_rows = []
    for phase in payload["transformer_phase_comparison"]:
        phase_rows.append(
            "<tr>"
            f"<td>Transformer {html.escape(phase['phase'])}</td>"
            f"<td>{_fmt(phase['best_val_mae'], 4)}</td>"
            "</tr>"
        )
    if best_mae is not None:
        phase_rows.append(
            "<tr>"
            "<td><strong>Hopfield (this study)</strong></td>"
            f"<td><strong>{_fmt(best_mae, 4)}</strong></td>"
            "</tr>"
        )

    charts_json = json.dumps(
        {
            "bar": {
                "labels": bar_labels,
                "values": [v if v is not None else 0 for v in bar_values],
                "colors": bar_colors,
                "target": TRANSFORMER_CHAMPION_MAE,
            },
            "lines": {"datasets": line_datasets},
        }
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Glucose Hopfield Optuna progress</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14; --card: #121820; --text: #e7ecf3;
      --muted: #9aa7b8; --line: #243041; --accent: #3dd68c;
    }}
    body {{ margin:0; font-family:"Segoe UI",system-ui,sans-serif;
      background:var(--bg); color:var(--text); line-height:1.45; }}
    main {{ max-width:1040px; margin:0 auto; padding:32px 20px 64px; }}
    h1,h2 {{ margin:0 0 12px; font-weight:650; }}
    h1 {{ font-size:1.8rem; }} h2 {{ font-size:1.2rem; margin-top:28px; }}
    p,li {{ color:var(--muted); }}
    .stats {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr));
      gap:12px; margin:20px 0; }}
    .stat {{ background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:14px 16px; }}
    .stat strong {{ display:block; font-size:1.35rem; color:var(--text); }}
    .stat span {{ color:var(--muted); font-size:0.85rem; }}
    .chart {{ margin:16px 0; border:1px solid var(--line); border-radius:12px;
      background:var(--card); padding:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; }}
    th {{ color:var(--muted); font-weight:600; }}
    code {{ color:var(--accent); }}
    .arch-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
      gap:14px; margin:16px 0; }}
    .arch-card {{ background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:16px; position:relative; }}
    .arch-rank {{ position:absolute; top:12px; right:16px; font-size:1.6rem;
      font-weight:700; color:var(--line); }}
    .arch-trial {{ font-size:0.85rem; font-weight:600; color:var(--muted);
      margin-bottom:2px; letter-spacing:0.03em; }}
    .arch-mae {{ font-size:1.2rem; font-weight:650; color:var(--accent); margin-bottom:6px; }}
    .arch-desc {{ font-size:0.95rem; color:var(--text); margin-bottom:4px; }}
    .arch-strength {{ font-size:0.85rem; color:var(--muted); margin-bottom:10px; }}
    .arch-params {{ font-size:0.8rem; }}
    .arch-params td {{ padding:3px 8px; border-bottom:1px solid var(--line); }}
    .arch-params td:first-child {{ color:var(--muted); }}
    @media (max-width:900px) {{ .stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  </style>
</head>
<body>
  <main>
    <h1>Glucose Hopfield Optuna progress</h1>
    <p>
      Study <code>{html.escape(payload['study_name'])}</code> ·
      target beat <code>{TRANSFORMER_CHAMPION_MAE}</code> ·
      generated <code>{html.escape(payload['generated_at'])}</code>
    </p>
    <p>
      Protocol: {html.escape(str(settings.get('protocol', 'epoch Hyperband')))} ·
      workers={html.escape(str(settings.get('max_workers', '—')))} ·
      epochs={html.escape(str(settings.get('max_epochs', '—')))} ·
      patience={html.escape(str(settings.get('patience', '—')))}
    </p>
    <div class="stats">
      <div class="stat"><strong>{counts['n_trials']}</strong><span>All trials</span></div>
      <div class="stat"><strong>{counts['n_complete']}</strong><span>Complete</span></div>
      <div class="stat"><strong>{counts['n_pruned']}</strong><span>Pruned</span></div>
      <div class="stat"><strong>{counts['n_fail']}</strong><span>Failed</span></div>
      <div class="stat"><strong>{_fmt(best_mae, 2)}</strong><span>Best MAE</span></div>
    </div>
    <section>
      <h2>Cross-run comparison</h2>
      <table>
        <thead><tr><th>Run</th><th>Best val MAE</th></tr></thead>
        <tbody>{''.join(phase_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>All trials (every state)</h2>
      <p style="font-size:0.85rem;">
        <span style="color:#3dd68c;">&#9632;</span>&nbsp;Beat target&ensp;
        <span style="color:#006FEE;">&#9632;</span>&nbsp;Complete&ensp;
        <span style="color:#f5a524;">&#9632;</span>&nbsp;Pruned&ensp;
        <span style="color:#f31260;">&#9632;</span>&nbsp;Failed&ensp;
        <span style="color:#4b5563;">&#9632;</span>&nbsp;No MAE
      </p>
      <div class="chart"><canvas id="barChart" height="120"></canvas></div>
      <table>
        <thead>
          <tr><th>Trial</th><th>State</th><th>Best MAE</th>
          <th>variant</th><th>strength</th><th>reason</th></tr>
        </thead>
        <tbody>{''.join(all_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>Within-trial epoch stages</h2>
      <div class="chart"><canvas id="lineChart" height="120"></canvas></div>
    </section>
    <section>
      <h2>Complete-trial leaderboard</h2>
      <table>
        <thead>
          <tr>
            <th>Trial</th><th>Best MAE</th><th>Δ vs 19.876</th>
            <th>variant</th><th>strength</th><th>LR</th>
            <th>η_infer</th><th>steps</th>
          </tr>
        </thead>
        <tbody>{''.join(leaderboard_rows)}</tbody>
      </table>
    </section>
    <section>
      <h2>Top model architectures</h2>
      <p style="font-size:0.85rem; color:var(--muted);">
        Full hyperparameter configurations of the top 5 performing models,
        showing which network variants and parameter combinations work best.
      </p>
      <div class="arch-grid">{''.join(arch_cards)}</div>
    </section>
    <section>
      <h2>Hyperparameter glossary</h2>
      <p style="font-size:0.85rem; color:var(--muted);">
        What each tuned parameter controls in the Hopfield predictive-coding model.
      </p>
      <table>
        <thead><tr><th>Parameter</th><th>Meaning</th></tr></thead>
        <tbody>
          {''.join(
              f'<tr><td><code>{html.escape(k)}</code></td>'
              f'<td style="color:var(--muted);">{html.escape(v)}</td></tr>'
              for k, v in PARAM_GLOSSARY.items()
          )}
        </tbody>
      </table>
    </section>
    <section>
      <h2>How the model works</h2>
      <p style="font-size:0.85rem; color:var(--muted);">
        This study adds <strong>Hopfield associative memory</strong> to the glucose PC transformer.
        A Hopfield layer is a content-addressable memory that stores and recalls learned glucose patterns,
        potentially helping the model recognise recurring dynamics (meals, exercise, dawn phenomenon).
        The tuning searches over <strong>where</strong> to place the memory (or omit it entirely) and
        <strong>how strong</strong> its influence is.
      </p>
      <div class="chart">{_svg_hopfield_architecture()}</div>
    </section>
  </main>
  <script>
    const DATA = {charts_json};
    const tickColor = "#9aa7b8";
    const gridColor = "#243041";
    new Chart(document.getElementById("barChart"), {{
      type: "bar",
      data: {{
        labels: DATA.bar.labels,
        datasets: [{{
          label: "Best val MAE",
          data: DATA.bar.values,
          backgroundColor: DATA.bar.colors,
          borderRadius: 4,
        }}],
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ labels: {{ color: tickColor }} }},
          annotation: {{
            annotations: {{
              target: {{
                type: "line",
                yMin: DATA.bar.target,
                yMax: DATA.bar.target,
                borderColor: "#3dd68c",
                borderWidth: 2,
                borderDash: [6, 4],
                label: {{
                  display: true,
                  content: `Target ${{DATA.bar.target}}`,
                  position: "end",
                  backgroundColor: "#121820",
                  color: "#3dd68c",
                }},
              }},
            }},
          }},
        }},
        scales: {{
          x: {{ ticks: {{ color: tickColor }}, grid: {{ color: gridColor }} }},
          y: {{
            ticks: {{ color: tickColor }},
            grid: {{ color: gridColor }},
            suggestedMin: 18,
            suggestedMax: 50,
            title: {{ display: true, text: "MAE (mg/dL)", color: tickColor }},
          }},
        }},
      }},
    }});
    new Chart(document.getElementById("lineChart"), {{
      type: "line",
      data: {{ datasets: DATA.lines.datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: tickColor }} }} }},
        scales: {{
          x: {{
            type: "linear",
            ticks: {{ color: tickColor }},
            grid: {{ color: gridColor }},
            title: {{ display: true, text: "Epoch", color: tickColor }},
          }},
          y: {{
            ticks: {{ color: tickColor }},
            grid: {{ color: gridColor }},
            suggestedMin: 18,
            suggestedMax: 40,
            title: {{ display: true, text: "Val MAE", color: tickColor }},
          }},
        }},
      }},
    }});
  </script>
</body>
</html>
"""


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, help="Hopfield Optuna study dir."),
    study_name: str | None = typer.Option(None, help="Optuna study name."),
    confirm_dir: Path = typer.Option(
        Path("runs/glucose_hopfield_best_confirm"),
        help="Optional confirmation run directory.",
    ),
    format: str = typer.Option("all", "--format", help="md, html, json, or all."),
    refresh: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Re-export results_snapshot.json from the journal first.",
    ),
) -> None:
    """Refresh Hopfield Optuna results and write progress reports."""
    if ctx.invoked_subcommand is not None:
        return

    if refresh and (run_dir / "optuna_journal.log").exists():
        _export_snapshot(run_dir, study_name=study_name)

    snapshot_path = run_dir / "results_snapshot.json"
    if not snapshot_path.exists():
        raise typer.BadParameter(f"Missing snapshot: {snapshot_path}")

    snapshot = json.loads(snapshot_path.read_text())
    confirm = _load_confirm(confirm_dir)
    payload = _build_payload(snapshot, confirm, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    wanted = {format.lower()} if format.lower() != "all" else {"md", "html", "json"}
    if "json" in wanted:
        data_path = run_dir / "report_data.json"
        data_path.write_text(json.dumps(payload, indent=2))
        typer.echo(f"wrote {data_path}")
    if "md" in wanted:
        md_text = _render_markdown(payload)
        md_path = run_dir / "report.md"
        md_path.write_text(md_text, encoding="utf-8")
        typer.echo(f"wrote {md_path}")
        docs_md = Path("docs/reports/glucose_hopfield_optuna_progress.md")
        docs_md.parent.mkdir(parents=True, exist_ok=True)
        docs_md.write_text(md_text, encoding="utf-8")
        typer.echo(f"wrote {docs_md}")
    if "html" in wanted:
        html_text = _render_html(payload)
        html_path = run_dir / "report.html"
        html_path.write_text(html_text, encoding="utf-8")
        typer.echo(f"wrote {html_path}")
        docs_html = Path("docs/reports/glucose_hopfield_optuna_progress.html")
        docs_html.parent.mkdir(parents=True, exist_ok=True)
        docs_html.write_text(html_text, encoding="utf-8")
        typer.echo(f"wrote {docs_html}")

    best = payload.get("best_complete")
    if best is not None:
        typer.echo(
            f"best_complete trial={best['trial']} "
            f"mae={best['best_history_mae']:.4f} "
            f"beat_target={payload.get('beat_transformer_champion')}"
        )


if __name__ == "__main__":
    app()
