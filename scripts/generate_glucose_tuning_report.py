"""Generate Markdown + HTML progress reports from glucose Optuna results.

Refreshes ``runs/glucose_tuning/results_snapshot.json`` from the Optuna journal,
optionally merges a PC confirmation run, then writes:

- ``runs/glucose_tuning/report.md``
- ``runs/glucose_tuning/report.html``  (self-contained SVG charts)
- ``runs/glucose_tuning/report_data.json``  (structured payload for reuse)

Usage::

    uv run python scripts/generate_glucose_tuning_report.py
    uv run python scripts/generate_glucose_tuning_report.py --format md
    uv run python scripts/generate_glucose_tuning_report.py --format html
    uv run python scripts/generate_glucose_tuning_report.py --format all
"""

from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, help="Render glucose Optuna MD/HTML reports.")

DEFAULT_RUN_DIR = Path("runs/glucose_tuning_pc_v2")
CONFIRM_DIR = Path("runs/glucose_pc_best_confirm")
STUDY_NAME = "glucose_transformer_pc_epochs_v2"

# Newbie-friendly theory: short = hover tooltip; longer fields = Theory section.
PARAM_THEORY: dict[str, dict[str, str]] = {
    "seq_len": {
        "short": "How many recent 5-min CGM readings the model sees (e.g. 64 ≈ 5.3 hours).",
        "what": "Sequence length — the history window length fed into the transformer.",
        "why": "Glucose has delayed effects (meals, activity). Too short and the model misses context; too long and it can drown in noise and cost more compute.",
        "effect": "64 often wins here (recent day-part). 128 doubles history but did not reliably help under our budget.",
    },
    "depth": {
        "short": "How many transformer blocks are stacked (model “shallowness”).",
        "what": "Depth — number of identical transformer blocks stacked on top of each other.",
        "why": "Each block can refine representations. Depth is capacity: more layers can learn richer patterns, but also train slower and overfit more easily on small data.",
        "effect": "Shallow (depth=1) means one attention+MLP stage — like a short assembly line. Deeper (2–4) means more stages. On this single-person CGM set, shallow models often generalise better and finish trials sooner.",
    },
    "num_heads": {
        "short": "Parallel attention “viewpoints” inside each block.",
        "what": "Multi-head attention splits attention into several heads that each look for different relationships in the sequence.",
        "why": "One head might track rising trends; another meal-like bumps. More heads = more specialised views, but also more parameters.",
        "effect": "With small embed dims, 1 head is common and stable. Extra heads help only if there is enough data and width to use them.",
    },
    "embed_dim": {
        "short": "Width of the internal vector for each timestep.",
        "what": "Embedding dimension — size of the hidden vector representing each glucose reading inside the network.",
        "why": "Wider vectors can store richer features, but need more data and memory.",
        "effect": "Larger embed_dim → more expressive, heavier. Too large on tiny data can overfit (train looks good, val MAE worse).",
    },
    "lr": {
        "short": "Outer learning rate — how big each weight update is (Adam).",
        "what": "Learning rate for the outer optimiser that updates model weights after each batch/update.",
        "why": "Too high → training jumps around or diverges (MAE explodes). Too low → crawls and never improves within the budget.",
        "effect": "Mid-range ~1e-3–4e-3 often worked for champion-like PC configs. Think of it as step size on a foggy hill: big steps miss the valley; tiny steps never arrive.",
    },
    "eta_infer": {
        "short": "Inner PC step size for refining latent activations.",
        "what": "PC inference learning rate (η). Inside each forward pass, latents are nudged to reduce prediction error.",
        "why": "PC has an inner loop separate from weight LR. η controls how aggressively beliefs are corrected before weights move.",
        "effect": "Too large → unstable energy / wild MAE. Too small → under-inferred latents (model never “settles”). Sweet spot here was roughly 1e-5–2.5e-5.",
    },
    "infer_steps": {
        "short": "How many inner PC refinement iterations per forward pass.",
        "what": "Number of times the inner PC loop updates latents before producing a forecast.",
        "why": "More steps ≈ tighter energy minimum, but each step costs compute. Too few and PC barely runs.",
        "effect": "12–16 steps were common in strong trials. Doubling steps rarely pays for itself if η is already well tuned.",
    },
    "max_infer_norm": {
        "short": "Clip on the size of each PC latent update.",
        "what": "Maximum gradient/update norm allowed during the inner PC loop.",
        "why": "Stops latent activations from exploding when errors are large (e.g. after a sharp glucose swing).",
        "effect": "Lower clips = safer but slower settling. Higher clips = freer movement, more risk of blow-ups.",
    },
    "grad_clip": {
        "short": "Clip on outer weight gradients (training stability).",
        "what": "Global gradient clipping for Adam weight updates.",
        "why": "Occasional huge gradients can wipe a good run. Clipping caps the damage.",
        "effect": "0.5–1.0 were typical. Too tight can stall learning; too loose lets rare spikes destabilise training.",
    },
    "lr_decay_epochs": {
        "short": "When cosine LR decay begins (epoch-based runs).",
        "what": "Epoch index after which the outer learning rate anneals toward zero.",
        "why": "Early high LR explores; later lower LR fine-tunes. Decay timing changes that schedule.",
        "effect": "Later decay = longer aggressive phase. Earlier decay = settle sooner (good if you overshoot).",
    },
    "weight_init_std": {
        "short": "How large random initial weights are.",
        "what": "Standard deviation of Normal weight initialisation.",
        "why": "Starting scale interacts with depth and PC dynamics. Bad init can look like a “broken” hyperparameter set.",
        "effect": "Smaller std = gentler start (often better with PC). Larger std can help or explode depending on η/LR.",
    },
    "weight_decay": {
        "short": "L2 penalty that discourages huge weights.",
        "what": "Weight decay regularisation strength.",
        "why": "On small datasets, unconstrained weights memorise noise. Decay nudges them toward simpler solutions.",
        "effect": "Higher → stronger regularisation (can underfit). Zero → freer fit (can overfit).",
    },
    "readout": {
        "short": "How the sequence is turned into one glucose forecast.",
        "what": "Regression head mode: flatten / mean_pool / last.",
        "why": "The network outputs a sequence of vectors; readout decides how to map that to a single 60‑min-ahead number.",
        "effect": "flatten uses the full sequence (more parameters, often best here). mean_pool averages time; last uses only the newest step — lighter heads that may need their own LR/η retuning.",
    },
    "seed_offset": {
        "short": "Random-seed nudge so similar configs can still differ.",
        "what": "Added to a base seed so trials explore different initialisations/data shuffles.",
        "why": "PC training can be seed-sensitive. Searching a small offset finds lucky (or unlucky) starts.",
        "effect": "Same architecture can move several MAE points just from seed — document the winning seed for replay.",
    },
    "energy": {
        "short": "How PC nodes score prediction error (Gaussian vs Huber).",
        "what": "Energy functional used inside PC nodes.",
        "why": "Gaussian (MSE) punishes large errors hard. Huber becomes linear for outliers — useful for glucose spikes.",
        "effect": "Gaussian is the classic default. Huber can be more robust when a few wild points would otherwise dominate.",
    },
    "huber_delta": {
        "short": "Threshold where Huber switches from quadratic to linear.",
        "what": "Delta parameter for Huber energy (only if energy=huber).",
        "why": "Controls when an error is treated as an “outlier”.",
        "effect": "Smaller delta → more robust / less spike-sensitive. Larger → closer to plain MSE.",
    },
    "ipc": {
        "short": "Update latents layer-by-layer (incremental PC) vs all at once.",
        "what": "Incremental Predictive Coding flag.",
        "why": "Layerwise updates can improve convergence on deep stacks by letting lower layers settle first.",
        "effect": "On shallow nets the difference may be small; on deeper nets IPC can matter more.",
    },
    "infer_optimizer": {
        "short": "Optimiser inside the PC loop: SGD or Adam.",
        "what": "Which optimiser nudges latent activations during inference.",
        "why": "SGD is simple/fast; Adam adapts per-coordinate and may settle in fewer steps (more memory).",
        "effect": "Try SGD first for speed; Adam if inference looks under-converged at the same step count.",
    },
}

PARAM_GLOSSARY: dict[str, str] = {
    key: meta["short"] for key, meta in PARAM_THEORY.items()
}

TRAINING_BASICS_HTML: str = """
<section>
  <h2>How to read this report (newbie guide)</h2>
  <div class="callout">
    <h3>Trials vs epochs (or updates)</h3>
    <p>
      <strong>A trial</strong> is one full experiment with a fixed set of hyperparameters
      (learning rate, depth, η_infer, …). Optuna proposes many trials to search for a good recipe.
    </p>
    <p>
      <strong>Inside a trial</strong>, the model trains over time. Depending on the protocol you will see either:
    </p>
    <ul>
      <li><strong>Epochs</strong> — one pass over the training set = 1 epoch. Validation MAE is logged each epoch.</li>
      <li><strong>Updates</strong> — weight steps (e.g. every 200 updates). Same idea: a learning curve of MAE vs training progress.</li>
    </ul>
    <p>
      So: <em>trial = which recipe</em>, <em>epoch/update = how far that recipe has trained</em>.
      The bar chart compares recipes; the line charts show learning inside a recipe.
    </p>
    <h3>Why a run stops early</h3>
    <ul>
      <li><strong>Early stopping (patience)</strong> — validation MAE stopped improving for N checks → stop; keeps the best checkpoint.</li>
      <li><strong>Pruned</strong> — Optuna’s pruner killed a clearly-losing trial mid-way to save compute (not a crash).</li>
      <li><strong>Failed</strong> — hard error (OOM, NaNs, timeout). Check the reason column.</li>
      <li><strong>Complete</strong> — finished the allowed budget (or early-stopped cleanly) and reported a final score.</li>
    </ul>
    <p>
      Lower <strong>MAE (mg/dL)</strong> is better — average absolute error of the 60‑minute glucose forecast.
      Hover green parameter chips (e.g. <span class="tip" data-tip="Example tooltip">lr</span>) for a one-line explanation;
      the Theory section below goes deeper.
    </p>
  </div>
</section>
"""


def _resolve_study_name(run_dir: Path, study_name: str | None) -> str:
    if study_name:
        return study_name
    config_path = run_dir / "coordinator_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        configured = config.get("study_name")
        if isinstance(configured, str) and configured:
            return configured
    return STUDY_NAME


def _export_snapshot(run_dir: Path, study_name: str | None = None) -> Path:
    """Refresh results_snapshot.json from the Optuna journal."""
    from examples.glucose_transformer_tuning import create_study

    resolved_name = _resolve_study_name(run_dir, study_name)
    param_keys = [
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
        "readout",
        "seed_offset",
    ]
    study = create_study(run_dir / "optuna_journal.log", resolved_name)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        history_path = run_dir / "trials" / f"trial_{trial.number:04d}" / "history.json"
        history = (
            json.loads(history_path.read_text()) if history_path.exists() else []
        )
        best_mae = min((row["mae_mg_dl"] for row in history), default=None)
        best_idx = None
        if history and best_mae is not None:
            best_idx = next(
                i for i, row in enumerate(history)
                if row["mae_mg_dl"] == best_mae
            )
        best_mard = (
            history[best_idx].get("mard_percent")
            if best_idx is not None
            else None
        )
        rows.append(
            {
                "trial": trial.number,
                "state": trial.state.name,
                "optuna_value": trial.value,
                "best_history_mae": best_mae,
                "best_mard_percent": best_mard,
                "params": {key: trial.params.get(key) for key in param_keys},
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
        "study_name": resolved_name,
        "n_trials": len(rows),
        "n_complete": sum(1 for row in rows if row["state"] == "COMPLETE"),
        "n_pruned": sum(1 for row in rows if row["state"] == "PRUNED"),
        "n_fail": sum(1 for row in rows if row["state"] == "FAIL"),
        "n_running": sum(1 for row in rows if row["state"] == "RUNNING"),
        "best_complete": completed[0] if completed else None,
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


def _completed_sorted(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        trial
        for trial in snapshot["trials"]
        if trial["best_history_mae"] is not None
        and trial["state"] in ("COMPLETE", "RUNNING")
    ]
    rows.sort(key=lambda trial: float(trial["best_history_mae"]))
    return rows


def _build_payload(
    snapshot: dict[str, Any],
    confirm: dict[str, Any] | None,
    run_dir: Path,
) -> dict[str, Any]:
    completed = _completed_sorted(snapshot)
    trial_bar = []
    for trial in snapshot["trials"]:
        mae = trial["best_history_mae"]
        if mae is None:
            continue
        trial_bar.append(
            {
                "trial": trial["trial"],
                "state": trial["state"],
                "best_mae": float(mae),
            }
        )

    top_histories = []
    for trial in completed[:4]:
        points = []
        for row in trial["history"]:
            point = {
                "update": int(row.get("update", row.get("epoch", 0))),
                "mae_mg_dl": float(row["mae_mg_dl"]),
            }
            if row.get("train_mae_mg_dl"):
                point["train_mae_mg_dl"] = float(row["train_mae_mg_dl"])
            if row.get("mard_percent") is not None:
                point["mard_percent"] = float(row["mard_percent"])
            points.append(point)
        top_histories.append(
            {
                "trial": trial["trial"],
                "best_mae": float(trial["best_history_mae"]),
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
        "study_name": snapshot.get("study_name", STUDY_NAME),
        "counts": {
            "n_trials": snapshot["n_trials"],
            "n_complete": snapshot["n_complete"],
            "n_pruned": snapshot["n_pruned"],
            "n_fail": snapshot["n_fail"],
            "n_running": snapshot["n_running"],
        },
        "best_complete": snapshot.get("best_complete"),
        "best_trial_file": best_trial_file,
        "trial_bar": trial_bar,
        "leaderboard": [
            {
                "trial": trial["trial"],
                "best_mae": float(trial["best_history_mae"]),
                "best_mard": trial.get("best_mard_percent"),
                "state": trial["state"],
                "params": trial["params"],
                "stop_reason": trial.get("user_attrs", {}).get("stop_reason"),
                "prune_reason": trial.get("user_attrs", {}).get("prune_reason"),
            }
            for trial in completed
        ],
        "top_histories": top_histories,
        "confirm": confirm,
        "all_trials": snapshot["trials"],
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _summarize_what_helped(leaderboard: list[dict[str, Any]]) -> str:
    """Dynamically summarize which parameter values appear in top trials."""
    top = leaderboard[:5]
    if not top:
        return "No completed trials to analyze."
    lines = []
    for key in ["seq_len", "depth", "num_heads", "readout"]:
        values = [row["params"].get(key) for row in top if row["params"].get(key) is not None]
        if not values:
            continue
        from collections import Counter
        counts = Counter(values)
        dominant = counts.most_common(1)[0]
        if dominant[1] >= len(values) * 0.6:
            lines.append(f"- **{key}={dominant[0]}** dominates top 5 ({dominant[1]}/{len(values)})")
        else:
            vals_str = ", ".join(f"{v}×{c}" for v, c in counts.most_common())
            lines.append(f"- **{key}**: mixed ({vals_str})")
    for key in ["lr", "eta_infer", "weight_init_std"]:
        values = [float(row["params"].get(key)) for row in top if row["params"].get(key) is not None]
        if not values:
            continue
        lo, hi = min(values), max(values)
        median = sorted(values)[len(values) // 2]
        lines.append(f"- **{key}**: range {lo:.4g}–{hi:.4g}, median {median:.4g}")
    for key in ["infer_steps", "grad_clip"]:
        values = [row["params"].get(key) for row in top if row["params"].get(key) is not None]
        if not values:
            continue
        from collections import Counter
        counts = Counter(values)
        vals_str = ", ".join(f"{v}×{c}" for v, c in counts.most_common())
        lines.append(f"- **{key}**: {vals_str}")
    return "\n".join(lines) if lines else "Insufficient parameter data to summarize."


def _render_markdown(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    best = payload.get("best_complete")
    lines: list[str] = [
        "# Glucose PC Optuna progress report",
        "",
        f"Generated: `{payload['generated_at']}`  ",
        f"Study: `{payload['study_name']}`  ",
        "Mode: predictive coding (PC) only",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Trials recorded | {counts['n_trials']} |",
        f"| Complete | {counts['n_complete']} |",
        f"| Pruned | {counts['n_pruned']} |",
        f"| Failed | {counts['n_fail']} |",
        f"| Running | {counts['n_running']} |",
    ]
    if best is not None:
        lines.extend(
            [
                f"| Best complete trial | {best['trial']} |",
                f"| Best val MAE (mg/dL) | {_fmt(best['best_history_mae'], 4)} |",
            ]
        )
    helped = _summarize_what_helped(payload["leaderboard"])
    lines.extend(
        [
            "",
            "## What helped (auto-generated from top trials)",
            "",
            helped,
            "",
            "## Top model architectures",
            "",
        ]
    )
    param_keys = [
        "seq_len", "depth", "num_heads", "lr", "eta_infer", "infer_steps",
        "max_infer_norm", "grad_clip", "lr_decay_epochs", "weight_init_std",
        "weight_decay", "readout", "seed_offset",
    ]
    for rank, row in enumerate(payload["leaderboard"][:5], 1):
        params = row["params"]
        geometry = (
            f"seq_len={params.get('seq_len')}, depth={params.get('depth')}, "
            f"heads={params.get('num_heads')}"
        )
        readout = params.get("readout", "flatten")
        lines.append(f"### #{rank} — Trial {row['trial']} (MAE {_fmt(row['best_mae'], 3)})")
        lines.append("")
        lines.append(f"- **Geometry**: {geometry}")
        lines.append(f"- **Readout**: {readout}")
        param_parts = []
        for key in param_keys:
            val = params.get(key)
            if val is not None:
                if isinstance(val, float):
                    param_parts.append(f"{key}={val:.4g}")
                else:
                    param_parts.append(f"{key}={val}")
        lines.append(f"- **All params**: {', '.join(param_parts)}")
        lines.append("")

    lines.extend(
        [
            "## Complete-trial leaderboard",
            "",
            "| Trial | Best MAE | MARD% | Geometry | LR | η_infer | Infer steps | Grad clip |",
            "|------:|---------:|------:|----------|---:|--------:|------------:|----------:|",
        ]
    )
    for row in payload["leaderboard"]:
        params = row["params"]
        geometry = (
            f"{params.get('seq_len')}/d{params.get('depth')}/h{params.get('num_heads')}"
        )
        lines.append(
            "| {trial} | {mae} | {mard} | {geometry} | {lr:.4g} | {eta:.4g} | {steps} | {clip} |".format(
                trial=row["trial"],
                mae=_fmt(row["best_mae"], 3),
                mard=_fmt(row.get("best_mard"), 2),
                geometry=geometry,
                lr=float(params.get("lr") or 0.0),
                eta=float(params.get("eta_infer") or 0.0),
                steps=params.get("infer_steps"),
                clip=params.get("grad_clip"),
            )
        )

    lines.extend(
        [
            "",
            "## Best MAE by trial",
            "",
            "| Trial | State | Best MAE (mg/dL) |",
            "|------:|-------|-----------------:|",
        ]
    )
    for row in payload["trial_bar"]:
        lines.append(
            f"| {row['trial']} | {row['state']} | {_fmt(row['best_mae'], 3)} |"
        )

    if payload["top_histories"]:
        lines.extend(["", "## Top trial MAE traces (every 200 updates)", ""])
        for hist in payload["top_histories"]:
            trace = " → ".join(
                f"{point['update']}:{_fmt(point['mae_mg_dl'], 2)}"
                for point in hist["points"]
            )
            lines.append(
                f"- **Trial {hist['trial']}** (best {_fmt(hist['best_mae'], 3)}): {trace}"
            )

    confirm = payload.get("confirm")
    if confirm is not None:
        cfg = confirm["config"]
        lines.extend(
            [
                "",
                "## PC confirmation train (epoch loop)",
                "",
                "Directory: `runs/glucose_pc_best_confirm`",
                "",
                "| Metric | Value |",
                "|--------|------:|",
                f"| Best val MAE | {_fmt(cfg.get('best_val_mae_mg_dl'), 4)} |",
                f"| Test MAE | {_fmt(cfg.get('test_mae_mg_dl'), 4)} |",
                f"| Test RMSE | {_fmt(cfg.get('test_rmse_mg_dl'), 4)} |",
                f"| Test MARD (%) | {_fmt(cfg.get('test_mard_percent'), 2)} |",
                f"| Epochs run | {cfg.get('final_epoch')} |",
                f"| Wall time (s) | {cfg.get('elapsed_s')} |",
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
            "## How to read this report",
            "",
            "- **Trial** = one hyperparameter recipe Optuna tried.",
            "- **Epoch / update** = training progress *inside* that recipe (learning curve).",
            "- **Pruned** = Optuna stopped a weak trial early to save compute (not a crash).",
            "- **Early stop / patience** = validation MAE stopped improving, so training halted and kept the best checkpoint.",
            "- Lower **MAE (mg/dL)** is better.",
            "",
            "## Complete training report (all trials)",
            "",
            "Each trial below keeps the same layout: summary → params → stage trace → stop reason.",
            "",
        ]
    )
    for trial in payload.get("all_trials") or []:
        params = trial.get("params") or {}
        attrs = trial.get("user_attrs") or {}
        reason = (
            attrs.get("stop_reason")
            or attrs.get("prune_reason")
            or attrs.get("failure_reason")
            or "—"
        )
        geometry = (
            f"{params.get('seq_len')}/d{params.get('depth')}/h{params.get('num_heads')}"
        )
        lines.append(
            f"### Trial {trial['trial']} — {trial['state']} "
            f"(best MAE {_fmt(trial.get('best_history_mae'), 3)})"
        )
        lines.append("")
        lines.append(f"- **Geometry**: `{geometry}` · readout `{params.get('readout', '—')}`")
        lines.append(f"- **Stop / prune reason**: {reason}")
        param_parts = []
        for key in [
            "lr", "eta_infer", "infer_steps", "max_infer_norm", "grad_clip",
            "weight_init_std", "weight_decay", "seed_offset", "energy", "ipc",
        ]:
            val = params.get(key)
            if val is None:
                continue
            if isinstance(val, float):
                param_parts.append(f"{key}={val:.4g}")
            else:
                param_parts.append(f"{key}={val}")
        if param_parts:
            lines.append(f"- **Params**: {', '.join(param_parts)}")
        history = trial.get("history") or []
        if history:
            x_key = "epoch" if "epoch" in history[0] else "update"
            lines.append("")
            lines.append(f"| {x_key.capitalize()} | Val MAE | MARD (%) |")
            lines.append("|------:|--------:|---------:|")
            for row in history:
                x_val = row.get(x_key, row.get("update", row.get("epoch")))
                lines.append(
                    f"| {x_val} | {_fmt(row.get('mae_mg_dl'), 3)} | "
                    f"{_fmt(row.get('mard_percent'), 2)} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Hyperparameter theory (for newbies)",
            "",
            "Short glossary first, then practical intuition for each knob.",
            "",
            "| Parameter | One-line meaning |",
            "|-----------|------------------|",
        ]
    )
    for key, meta in PARAM_THEORY.items():
        lines.append(f"| `{key}` | {meta['short']} |")

    lines.extend(["", "### Deeper explanations", ""])
    for key, meta in PARAM_THEORY.items():
        lines.append(f"#### `{key}`")
        lines.append("")
        lines.append(f"- **What it is**: {meta['what']}")
        lines.append(f"- **Why you care**: {meta['why']}")
        lines.append(f"- **How changes show up**: {meta['effect']}")
        lines.append("")

    lines.extend(
        [
            "## Background",
            "",
            "This work builds on our earlier results with conventional (non-PC) transformers for glucose",
            "forecasting at [GlucoseDAO/glucose-forecasting](https://github.com/GlucoseDAO/glucose-forecasting).",
            "Here we replace the standard forward pass with **predictive coding (PC)** — an inner",
            "optimisation loop where each layer maintains its own \"belief\" about what the input should",
            "look like, computes a prediction error, and iteratively refines its activations before the",
            "outer weight update.",
            "",
            "We also explore a **Hopfield extension** — adding a content-addressable associative memory",
            "layer (Storkey Hopfield network) that can store and recall learned glucose dynamics such as",
            "meal responses, exercise patterns, and dawn phenomenon. The Hopfield memory gives the model",
            "an explicit pattern-recall mechanism beyond what attention alone provides.",
            "",
            "## How the model works",
            "",
            "### Standard PC Transformer",
            "",
            "```",
            "Glucose Input (batch, seq_len, 1)",
            "       |",
            "  Continuous Embedding  — linear projection to embed_dim",
            "       |",
            "  +--[ Transformer Block ] × depth --------+",
            "  |    Multi-Scale Self-Attention (RoPE)    |",
            "  |    at downsampling 1×, 2×, 4×           |",
            "  |    LN → MLP expand (GELU)               |",
            "  |    MLP contract + Residual skip          |",
            "  |    PC Energy Node                        |",
            "  +------------------------------------------+",
            "       |",
            "  Regression Output Head → Glucose Forecast (60 min)",
            "```",
            "",
            "### PC inference loop (runs at every node)",
            "",
            "1. Predict `z_mu` from incoming activations",
            "2. Compute `error = z_latent - z_mu`",
            "3. Compute energy from error (Gaussian or Huber)",
            "4. Update `z_latent` via SGD or Adam (step size = `eta_infer`, clip = `max_infer_norm`)",
            "5. Repeat for `infer_steps` iterations",
            "",
            "## Limitations",
            "",
            "- **Single participant data** — Livia's personal CGM only for the deadline sprint.",
            "- **Glucose-only input** — carbs / HR / steps not included yet.",
            "- **Limited tuning budget** — finite Optuna trials and ranges.",
            "",
            "## How to run",
            "",
            "| Task | Command |",
            "|------|---------|",
            "| Install (CPU) | `uv sync --extra glucose` |",
            "| Install (GPU / WSL) | `uv sync --extra glucose --extra cuda12` |",
            "| Train PC transformer | `uv run glucose-transformer` |",
            "| Start epoch Optuna | `uv run glucose-transformer-tune run` |",
            "| Update-budget Optuna | `uv run glucose-transformer-tune-update-budget run` |",
            "| Regenerate this report | `uv run python scripts/generate_glucose_tuning_report.py --format all` |",
            "| All reports + master | `uv run python scripts/generate_all_glucose_reports.py --format all` |",
            "| Master report only | `uv run python scripts/generate_glucose_master_report.py` |",
            "",
        ]
    )
    return "\n".join(lines)


def _svg_bar_chart(
    values: list[tuple[str, float]],
    *,
    title: str,
    width: int = 920,
    height: int = 320,
    y_min: float = 18.0,
    y_max: float = 50.0,
) -> str:
    left, right, top, bottom = 48, 16, 36, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = max(len(values), 1)
    bar_w = plot_w / n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#0f1419"/>',
        f'<text x="{left}" y="22" fill="#e7ecf3" font-size="14" '
        f'font-family="Segoe UI, sans-serif">{html.escape(title)}</text>',
    ]
    for tick in (20, 30, 40, 50):
        if tick < y_min or tick > y_max:
            continue
        y = top + (1 - (tick - y_min) / (y_max - y_min)) * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#2a3441" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" fill="#9aa7b8" font-size="11" '
            f'text-anchor="end" font-family="Segoe UI, sans-serif">{tick}</text>'
        )
    for index, (label, value) in enumerate(values):
        x = left + index * bar_w + bar_w * 0.15
        w = bar_w * 0.7
        clamped = min(max(value, y_min), y_max)
        h = ((clamped - y_min) / (y_max - y_min)) * plot_h
        y = top + plot_h - h
        color = "#3dd68c" if value <= 21.5 else ("#f5a524" if value <= 30 else "#f31260")
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{color}" rx="2"/>'
        )
        if n <= 40:
            parts.append(
                f'<text x="{x + w / 2:.1f}" y="{height - 14}" fill="#9aa7b8" '
                f'font-size="9" text-anchor="middle" '
                f'font-family="Segoe UI, sans-serif">{html.escape(label)}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_line_chart(
    series: list[dict[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    width: int = 920,
    height: int = 320,
    y_min: float = 18.0,
    y_max: float = 36.0,
    colors: list[str] | None = None,
) -> str:
    palette = colors or ["#3dd68c", "#006FEE", "#a1a1aa", "#f5a524", "#f31260"]
    left, right, top, bottom = 48, 16, 36, 44
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x: list[float] = []
    for item in series:
        all_x.extend(float(point[x_key]) for point in item["points"])
    if not all_x:
        return f"<p>{html.escape(title)}: no data</p>"
    x_min, x_max = min(all_x), max(all_x)
    x_span = max(x_max - x_min, 1.0)

    def map_x(value: float) -> float:
        return left + ((value - x_min) / x_span) * plot_w

    def map_y(value: float) -> float:
        clamped = min(max(value, y_min), y_max)
        return top + (1 - (clamped - y_min) / (y_max - y_min)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="#0f1419"/>',
        f'<text x="{left}" y="22" fill="#e7ecf3" font-size="14" '
        f'font-family="Segoe UI, sans-serif">{html.escape(title)}</text>',
    ]
    for tick in range(int(math.floor(y_min)), int(math.ceil(y_max)) + 1, 2):
        y = map_y(float(tick))
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#2a3441" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" fill="#9aa7b8" font-size="11" '
            f'text-anchor="end" font-family="Segoe UI, sans-serif">{tick}</text>'
        )

    for index, item in enumerate(series):
        color = palette[index % len(palette)]
        points = item["points"]
        if len(points) < 2:
            continue
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{map_x(float(point[x_key])):.1f},"
            f"{map_y(float(point[y_key])):.1f}"
            for i, point in enumerate(points)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        parts.append(
            f'<text x="{width - right - 8}" y="{top + 14 + index * 16}" '
            f'fill="{color}" font-size="11" text-anchor="end" '
            f'font-family="Segoe UI, sans-serif">{html.escape(item["name"])}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_line_chart_detailed(
    series: list[dict[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    width: int = 920,
    height: int = 360,
    y_min: float = 18.0,
    y_max: float = 36.0,
    colors: list[str] | None = None,
) -> str:
    """SVG line chart with a separate legend block below showing full params."""
    palette = colors or ["#3dd68c", "#006FEE", "#f5a524", "#9353d3", "#f31260", "#a1a1aa"]
    left, right, top, bottom = 52, 16, 36, 28
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x: list[float] = []
    for item in series:
        all_x.extend(float(p[x_key]) for p in item["points"])
    if not all_x:
        return f"<p>{html.escape(title)}: no data</p>"
    x_min, x_max = min(all_x), max(all_x)
    x_span = max(x_max - x_min, 1.0)

    def mx(v: float) -> float:
        return left + ((v - x_min) / x_span) * plot_w

    def my(v: float) -> float:
        clamped = min(max(v, y_min), y_max)
        return top + (1 - (clamped - y_min) / (y_max - y_min)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(title)}" '
        f'style="max-width:100%;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#0f1419"/>',
        f'<text x="{left}" y="22" fill="#e7ecf3" font-size="14" '
        f'font-family="Segoe UI, sans-serif">{html.escape(title)}</text>',
    ]
    step = 2 if (y_max - y_min) <= 20 else 5
    for tick in range(int(math.floor(y_min)), int(math.ceil(y_max)) + 1, step):
        y = my(float(tick))
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#2a3441" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" fill="#9aa7b8" font-size="11" '
            f'text-anchor="end" font-family="Segoe UI, sans-serif">{tick}</text>'
        )

    for idx, item in enumerate(series):
        color = palette[idx % len(palette)]
        points = item["points"]
        if len(points) < 2:
            for p in points:
                cx, cy = mx(float(p[x_key])), my(float(p[y_key]))
                parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>'
                )
            continue
        dash = item.get("dash")
        dash_attr = f' stroke-dasharray="{dash[0]},{dash[1]}"' if dash else ""
        path_d = " ".join(
            f"{'M' if i == 0 else 'L'}{mx(float(p[x_key])):.1f},"
            f"{my(float(p[y_key])):.1f}"
            for i, p in enumerate(points)
        )
        parts.append(
            f'<path d="{path_d}" fill="none" stroke="{color}" '
            f'stroke-width="2.2"{dash_attr}/>'
        )
        for p in points:
            cx, cy = mx(float(p[x_key])), my(float(p[y_key]))
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color}"/>'
            )

    parts.append("</svg>")
    svg = "\n".join(parts)

    legend_items = []
    for idx, item in enumerate(series):
        color = palette[idx % len(palette)]
        dash = item.get("dash")
        line_style = "border-top:2px dashed" if dash else "border-top:2px solid"
        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f'margin-right:16px;margin-bottom:4px;font-size:0.8rem;">'
            f'<span style="display:inline-block;width:20px;{line_style} {color};"></span>'
            f'<span style="color:{color};">{html.escape(item["name"])}</span>'
            f'</span>'
        )
    legend_html = (
        f'<div style="margin-top:8px;line-height:1.8;">'
        f'{"".join(legend_items)}</div>'
    )
    return svg + legend_html


def _svg_architecture_diagram() -> str:
    """Inline SVG showing Standard PC Transformer vs Hopfield PC Transformer side by side."""
    f = 'font-family="Segoe UI,sans-serif"'

    # ── Left column: Standard PC Transformer ──
    def _standard_col() -> str:
        cx, lx, w = 230, 130, 200
        return (
            f'<text x="{cx}" y="60" fill="#006FEE" font-size="13" {f} '
            f'font-weight="700" text-anchor="middle">Standard PC Transformer</text>'
            f'<text x="{cx}" y="76" fill="#9aa7b8" font-size="10" {f} '
            f'text-anchor="middle">No associative memory</text>'
            # Input
            f'<rect x="{lx}" y="90" width="{w}" height="32" rx="6" '
            f'fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
            f'<text x="{cx}" y="110" fill="#3dd68c" font-size="10" {f} '
            f'text-anchor="middle" font-weight="600">Glucose Input (seq_len, 1)</text>'
            f'<line x1="{cx}" y1="122" x2="{cx}" y2="134" stroke="#4b5563" '
            f'stroke-width="1" marker-end="url(#ah)"/>'
            # Embedding
            f'<rect x="{lx}" y="134" width="{w}" height="30" rx="6" '
            f'fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
            f'<text x="{cx}" y="154" fill="#006FEE" font-size="10" {f} '
            f'text-anchor="middle" font-weight="600">Continuous Embedding</text>'
            f'<line x1="{cx}" y1="164" x2="{cx}" y2="184" stroke="#4b5563" '
            f'stroke-width="1" marker-end="url(#ah)"/>'
            # Transformer block
            f'<rect x="{lx - 10}" y="184" width="{w + 20}" height="120" rx="8" '
            f'fill="#121820" stroke="#9aa7b8" stroke-width="1" stroke-dasharray="5,3"/>'
            f'<text x="{lx + 6}" y="200" fill="#9aa7b8" font-size="9" {f} '
            f'font-style="italic">× depth</text>'
            f'<rect x="{lx + 4}" y="206" width="{w - 8}" height="26" rx="5" '
            f'fill="#1a2332" stroke="#f5a524" stroke-width="0.8"/>'
            f'<text x="{cx}" y="224" fill="#f5a524" font-size="9.5" {f} '
            f'text-anchor="middle">Multi-Scale MHA + Residual</text>'
            f'<rect x="{lx + 4}" y="238" width="{w - 8}" height="24" rx="5" '
            f'fill="#1a2332" stroke="#9353d3" stroke-width="0.8"/>'
            f'<text x="{cx}" y="254" fill="#9353d3" font-size="9.5" {f} '
            f'text-anchor="middle">LN → MLP → MLP + skip</text>'
            f'<rect x="{lx + 4}" y="268" width="{w - 8}" height="26" rx="5" '
            f'fill="#1a1a2e" stroke="#006FEE" stroke-width="1"/>'
            f'<text x="{cx}" y="286" fill="#006FEE" font-size="9" {f} '
            f'text-anchor="middle" font-weight="600">PC Energy Node</text>'
            f'<line x1="{cx}" y1="304" x2="{cx}" y2="318" stroke="#4b5563" '
            f'stroke-width="1" marker-end="url(#ah)"/>'
            # Readout
            f'<rect x="{lx}" y="318" width="{w}" height="30" rx="6" '
            f'fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
            f'<text x="{cx}" y="338" fill="#3dd68c" font-size="10" {f} '
            f'text-anchor="middle" font-weight="600">Readout → Forecast</text>'
            f'<text x="{cx}" y="366" fill="#9aa7b8" font-size="9" {f} '
            f'text-anchor="middle">Direct path: embed → attend → predict</text>'
        )

    # ── Right column: Hopfield PC Transformer ──
    def _hopfield_col() -> str:
        cx, lx, w = 690, 578, 224
        return (
            f'<text x="{cx}" y="60" fill="#f5a524" font-size="13" {f} '
            f'font-weight="700" text-anchor="middle">Hopfield PC Transformer</text>'
            f'<text x="{cx}" y="76" fill="#9aa7b8" font-size="10" {f} '
            f'text-anchor="middle">Associative memory for pattern recall</text>'
            # Input
            f'<rect x="{lx}" y="90" width="{w}" height="32" rx="6" '
            f'fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
            f'<text x="{cx}" y="110" fill="#3dd68c" font-size="10" {f} '
            f'text-anchor="middle" font-weight="600">Glucose Input (seq_len, 1)</text>'
            f'<line x1="{cx}" y1="122" x2="{cx}" y2="134" stroke="#4b5563" '
            f'stroke-width="1" marker-end="url(#ah)"/>'
            # Embedding
            f'<rect x="{lx}" y="134" width="{w}" height="30" rx="6" '
            f'fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
            f'<text x="{cx}" y="154" fill="#006FEE" font-size="10" {f} '
            f'text-anchor="middle" font-weight="600">Continuous Embedding</text>'
            f'<line x1="{cx}" y1="164" x2="{cx}" y2="176" stroke="#4b5563" '
            f'stroke-width="1" marker-end="url(#ah)"/>'
            # Hopfield memory (highlighted)
            f'<rect x="{lx - 4}" y="176" width="{w + 8}" height="38" rx="6" '
            f'fill="#2a1f0a" stroke="#f5a524" stroke-width="2"/>'
            f'<text x="{cx}" y="194" fill="#f5a524" font-size="11" {f} '
            f'text-anchor="middle" font-weight="700">Storkey Hopfield Memory</text>'
            f'<text x="{cx}" y="208" fill="#9aa7b8" font-size="8.5" {f} '
            f'text-anchor="middle">content-addressable pattern recall</text>'
            f'<line x1="{cx}" y1="214" x2="{cx}" y2="228" stroke="#4b5563" '
            f'stroke-width="1" marker-end="url(#ah)"/>'
            # Transformer block
            f'<rect x="{lx - 10}" y="228" width="{w + 20}" height="76" rx="8" '
            f'fill="#121820" stroke="#9aa7b8" stroke-width="1" stroke-dasharray="5,3"/>'
            f'<text x="{lx + 6}" y="244" fill="#9aa7b8" font-size="9" {f} '
            f'font-style="italic">× depth</text>'
            f'<rect x="{lx + 4}" y="250" width="{w - 8}" height="24" rx="5" '
            f'fill="#1a2332" stroke="#f5a524" stroke-width="0.8"/>'
            f'<text x="{cx}" y="266" fill="#f5a524" font-size="9.5" {f} '
            f'text-anchor="middle">Multi-Scale MHA + Residual</text>'
            f'<rect x="{lx + 4}" y="278" width="{w - 8}" height="22" rx="5" '
            f'fill="#1a2332" stroke="#9353d3" stroke-width="0.8"/>'
            f'<text x="{cx}" y="294" fill="#9353d3" font-size="9.5" {f} '
            f'text-anchor="middle">MLP + skip + PC Energy Node</text>'
            f'<line x1="{cx}" y1="304" x2="{cx}" y2="318" stroke="#4b5563" '
            f'stroke-width="1" marker-end="url(#ah)"/>'
            # Readout
            f'<rect x="{lx}" y="318" width="{w}" height="30" rx="6" '
            f'fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
            f'<text x="{cx}" y="338" fill="#3dd68c" font-size="10" {f} '
            f'text-anchor="middle" font-weight="600">Readout → Forecast</text>'
            f'<text x="{cx}" y="366" fill="#9aa7b8" font-size="9" {f} '
            f'text-anchor="middle">Memory recalls learned glucose dynamics</text>'
        )

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 530" '
        'role="img" aria-label="Standard vs Hopfield PC Transformer" '
        'style="max-width:100%;height:auto;">'
        '<rect width="920" height="530" rx="12" fill="#0f1419"/>'
        '<defs><marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L8,3 L0,6" fill="#4b5563"/></marker></defs>'
        f'<text x="460" y="30" fill="#e7ecf3" font-size="15" {f} font-weight="650" '
        f'text-anchor="middle">Standard PC Transformer vs Hopfield Extension</text>'
        f'<text x="460" y="46" fill="#9aa7b8" font-size="10" {f} text-anchor="middle">'
        f'Same transformer backbone — Hopfield adds associative memory for glucose pattern recall</text>'
        '<line x1="460" y1="56" x2="460" y2="375" stroke="#243041" stroke-width="1" stroke-dasharray="6,4"/>'
        + _standard_col()
        + _hopfield_col()
        # PC inference box (shared, bottom)
        + f'<rect x="100" y="386" width="720" height="130" rx="8" fill="#121820" '
        f'stroke="#f31260" stroke-width="1"/>'
        f'<text x="460" y="408" fill="#f31260" font-size="12" {f} '
        f'text-anchor="middle" font-weight="650">'
        f'PC Inference Loop (runs at every node in both variants)</text>'
        f'<text x="460" y="428" fill="#e7ecf3" font-size="10" {f} text-anchor="middle">'
        f'1. Predict z_mu from inputs  →  2. error = z − z_mu  →  '
        f'3. Compute energy  →  4. Update z via SGD/Adam</text>'
        f'<text x="460" y="448" fill="#9aa7b8" font-size="10" {f} text-anchor="middle">'
        f'Repeat infer_steps× · step size eta_infer · clip max_infer_norm · '
        f'optimizer: sgd or adam (searched)</text>'
        f'<text x="460" y="468" fill="#9aa7b8" font-size="9.5" {f} text-anchor="middle">'
        f'Outer loop: Adam weight update (lr, grad_clip) · cosine LR decay after lr_decay_epochs</text>'
        f'<text x="460" y="492" fill="#f5a524" font-size="10" {f} text-anchor="middle" font-weight="600">'
        f'Why Hopfield? Content-addressable memory stores learned glucose patterns (meals, exercise,</text>'
        f'<text x="460" y="506" fill="#f5a524" font-size="10" {f} text-anchor="middle" font-weight="600">'
        f'dawn phenomenon) and recalls them during inference — giving the model explicit pattern memory.</text>'
        '</svg>'
    )


def _svg_energy_comparison() -> str:
    """Inline SVG comparing Gaussian vs Huber energy for PC nodes."""
    w, h = 920, 340
    # Plot area
    left, right, top, bottom = 60, 60, 50, 50
    pw, ph = w - left - right, h - top - bottom
    cx, cy = left + pw // 2, top + ph // 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'role="img" aria-label="Gaussian vs Huber Energy Comparison" '
        f'style="max-width:100%;height:auto;">',
        f'<rect width="{w}" height="{h}" rx="12" fill="#0f1419"/>',
        f'<text x="{w//2}" y="30" fill="#e7ecf3" font-size="15" '
        f'font-family="Segoe UI,sans-serif" font-weight="650" text-anchor="middle">'
        f'Energy Functions: Gaussian (MSE) vs Huber (robust)</text>',
    ]

    # Axes
    parts.append(
        f'<line x1="{left}" y1="{cy}" x2="{w-right}" y2="{cy}" '
        f'stroke="#4b5563" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{cx}" y1="{top}" x2="{cx}" y2="{h-bottom}" '
        f'stroke="#4b5563" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{w-right+4}" y="{cy+4}" fill="#9aa7b8" font-size="10" '
        f'font-family="Segoe UI,sans-serif">error</text>'
    )
    parts.append(
        f'<text x="{cx+6}" y="{top-4}" fill="#9aa7b8" font-size="10" '
        f'font-family="Segoe UI,sans-serif">Energy</text>'
    )

    scale_x = pw / 8.0
    scale_y = ph / 5.0
    delta = 1.0

    def px(v: float) -> float:
        return cx + v * scale_x

    def py(v: float) -> float:
        return cy - v * scale_y

    # Gaussian curve: E = 0.5 * x^2
    gauss_pts = []
    for i in range(-40, 41):
        x = i * 0.1
        e = 0.5 * x * x
        if e > 4.5:
            continue
        gauss_pts.append(f"{px(x):.1f},{py(e):.1f}")
    parts.append(
        f'<polyline points="{" ".join(gauss_pts)}" fill="none" '
        f'stroke="#006FEE" stroke-width="2.5"/>'
    )

    # Huber curve
    huber_pts = []
    for i in range(-40, 41):
        x = i * 0.1
        ax = abs(x)
        if ax <= delta:
            e = 0.5 * x * x
        else:
            e = delta * (ax - 0.5 * delta)
        if e > 4.5:
            continue
        huber_pts.append(f"{px(x):.1f},{py(e):.1f}")
    parts.append(
        f'<polyline points="{" ".join(huber_pts)}" fill="none" '
        f'stroke="#f5a524" stroke-width="2.5"/>'
    )

    # Delta markers
    for sign in (-1, 1):
        dx = px(sign * delta)
        parts.append(
            f'<line x1="{dx:.1f}" y1="{cy-4}" x2="{dx:.1f}" y2="{cy+4}" '
            f'stroke="#f5a524" stroke-width="1.5"/>'
        )
    parts.append(
        f'<text x="{px(delta):.1f}" y="{cy+16}" fill="#f5a524" font-size="9" '
        f'font-family="Segoe UI,sans-serif" text-anchor="middle">delta</text>'
    )
    parts.append(
        f'<text x="{px(-delta):.1f}" y="{cy+16}" fill="#f5a524" font-size="9" '
        f'font-family="Segoe UI,sans-serif" text-anchor="middle">-delta</text>'
    )

    # Legend
    ly = h - 24
    parts.append(
        f'<rect x="{w//2-220}" y="{ly-12}" width="16" height="3" fill="#006FEE"/>'
        f'<text x="{w//2-200}" y="{ly-6}" fill="#006FEE" font-size="11" '
        f'font-family="Segoe UI,sans-serif">Gaussian: E = 0.5 ||error||'
        f'&#xB2;  — penalises large errors heavily</text>'
    )
    parts.append(
        f'<rect x="{w//2-220}" y="{ly+4}" width="16" height="3" fill="#f5a524"/>'
        f'<text x="{w//2-200}" y="{ly+10}" fill="#f5a524" font-size="11" '
        f'font-family="Segoe UI,sans-serif">Huber: linear past delta — robust to '
        f'glucose spikes / outliers</text>'
    )

    parts.append('</svg>')
    return '\n'.join(parts)


def _tip(key: str, label: str | None = None) -> str:
    """Hoverable parameter label with short glossary text."""
    meta = PARAM_THEORY.get(key)
    text = label if label is not None else key
    if meta is None:
        return f"<code>{html.escape(text)}</code>"
    tip = html.escape(meta["short"], quote=True)
    return (
        f'<span class="tip" data-tip="{tip}">'
        f"<code>{html.escape(text)}</code></span>"
    )


def _fmt_param_value(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4g}"
    return html.escape(str(val))


def _theory_section_html() -> str:
    cards: list[str] = []
    for key, meta in PARAM_THEORY.items():
        cards.append(
            f'<details class="theory-card">'
            f"<summary>{_tip(key)} — {html.escape(meta['short'])}</summary>"
            f"<p><strong>What it is.</strong> {html.escape(meta['what'])}</p>"
            f"<p><strong>Why you care.</strong> {html.escape(meta['why'])}</p>"
            f"<p><strong>How changes show up.</strong> {html.escape(meta['effect'])}</p>"
            f"</details>"
        )
    return (
        "<section><h2>Hyperparameter theory (for newbies)</h2>"
        "<p class=\"hint\">Hover any green parameter chip for a one-liner. "
        "Open a card for the full “what / why / effect” story "
        "(e.g. what “shallow depth” actually means).</p>"
        f"{''.join(cards)}</section>"
    )


def _trial_sections_html(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Per-trial complete report blocks + chart payloads for JS."""
    blocks: list[str] = []
    trial_charts: list[dict[str, Any]] = []
    palette = ["#3dd68c", "#006FEE", "#f5a524", "#9353d3", "#f31260", "#a1a1aa"]
    for trial in payload.get("all_trials") or []:
        params = trial.get("params") or {}
        attrs = trial.get("user_attrs") or {}
        reason = (
            attrs.get("stop_reason")
            or attrs.get("prune_reason")
            or attrs.get("failure_reason")
            or "—"
        )
        history = trial.get("history") or []
        x_key = "epoch" if history and "epoch" in history[0] else "update"
        state = trial["state"]
        state_class = {
            "COMPLETE": "ok",
            "PRUNED": "warn",
            "FAIL": "bad",
            "RUNNING": "run",
        }.get(state, "")
        param_chips = []
        for key in [
            "seq_len", "depth", "num_heads", "readout", "lr", "eta_infer",
            "infer_steps", "max_infer_norm", "grad_clip", "weight_init_std",
            "weight_decay", "seed_offset", "energy", "ipc", "infer_optimizer",
        ]:
            val = params.get(key)
            if val is None:
                continue
            param_chips.append(
                f"<tr><td>{_tip(key)}</td>"
                f"<td><strong>{_fmt_param_value(val)}</strong></td></tr>"
            )
        stage_rows = []
        for row in history:
            x_val = row.get(x_key, row.get("update", row.get("epoch")))
            stage_rows.append(
                "<tr>"
                f"<td>{html.escape(str(x_val))}</td>"
                f"<td>{_fmt(row.get('mae_mg_dl'), 3)}</td>"
                f"<td>{_fmt(row.get('mard_percent'), 2)}</td>"
                "</tr>"
            )
        chart_id = f"trialChart_{trial['trial']}"
        chart_block = ""
        if history:
            points = [
                {
                    "x": int(row.get(x_key, row.get("update", row.get("epoch", 0)))),
                    "y": float(row["mae_mg_dl"]),
                }
                for row in history
                if row.get("mae_mg_dl") is not None
            ]
            color = palette[int(trial["trial"]) % len(palette)]
            trial_charts.append(
                {
                    "id": chart_id,
                    "label": f"Trial {trial['trial']} val MAE",
                    "xTitle": x_key.capitalize(),
                    "data": points,
                    "color": color,
                }
            )
            chart_block = (
                f'<div class="chart"><canvas id="{chart_id}" height="110"></canvas></div>'
            )
        table_block = ""
        if stage_rows:
            table_block = (
                f"<table><thead><tr>"
                f"<th>{html.escape(x_key.capitalize())}</th>"
                f"<th>Val MAE</th><th>MARD (%)</th>"
                f"</tr></thead><tbody>{''.join(stage_rows)}</tbody></table>"
            )
        blocks.append(
            f'<details class="trial-card" id="trial-{trial["trial"]}">'
            f"<summary>"
            f'<span class="badge {state_class}">{html.escape(state)}</span> '
            f"Trial {trial['trial']} · best MAE "
            f"<strong>{_fmt(trial.get('best_history_mae'), 3)}</strong>"
            f"</summary>"
            f"<div class=\"trial-body\">"
            f"<h3>Summary</h3>"
            f"<p>Stop / prune reason: <em>{html.escape(str(reason))}</em></p>"
            f"<h3>Parameters "
            f"<span class=\"hint\">(hover names for explanations)</span></h3>"
            f'<table class="arch-params">{"".join(param_chips) or "<tr><td>—</td></tr>"}</table>'
            f"<h3>Learning curve (stages)</h3>"
            f"{chart_block}"
            f"{table_block}"
            f"</div></details>"
        )
    section = (
        '<details class="fold-block" id="complete">'
        "<summary>Complete training report (all trials)</summary>"
        '<p class="hint">Same layout for every trial: summary → parameters → '
        "stage curve/table. Expand a trial to inspect why it stopped.</p>"
        f"{''.join(blocks)}</details>"
    )
    return section, trial_charts


def _render_html(payload: dict[str, Any]) -> str:
    """Interactive HTML report: Chart.js, param hover tips, full trial dump."""
    counts = payload["counts"]
    best = payload.get("best_complete")
    best_mae = best["best_history_mae"] if best else None

    bar_labels: list[str] = []
    bar_values: list[float | None] = []
    bar_colors: list[str] = []
    for trial in payload.get("all_trials") or []:
        bar_labels.append(str(trial["trial"]))
        mae = trial.get("best_history_mae")
        bar_values.append(float(mae) if mae is not None else None)
        if mae is None:
            bar_colors.append("#4b5563")
        elif trial["state"] == "PRUNED":
            bar_colors.append("#f5a524")
        elif trial["state"] == "FAIL":
            bar_colors.append("#f31260")
        elif float(mae) <= 21.5:
            bar_colors.append("#3dd68c")
        elif float(mae) <= 30:
            bar_colors.append("#f5a524")
        else:
            bar_colors.append("#f31260")

    line_palette = ["#3dd68c", "#006FEE", "#f5a524", "#9353d3"]
    line_datasets: list[dict[str, Any]] = []
    sample_hist = None
    for trial in payload.get("all_trials") or []:
        if trial.get("history"):
            sample_hist = trial["history"][0]
            break
    x_title = (
        "Epoch"
        if sample_hist is not None and "epoch" in sample_hist and "update" not in sample_hist
        else "Optimizer update"
    )
    for index, hist in enumerate(payload["top_histories"]):
        color = line_palette[index % len(line_palette)]
        params = hist.get("params") or {}
        geometry = (
            f"{params.get('seq_len')}/d{params.get('depth')}"
            f"/h{params.get('num_heads')}"
        )
        detail_parts = [f"T{hist['trial']}", geometry]
        for key in ("lr", "eta_infer", "infer_steps"):
            val = params.get(key)
            if val is None:
                continue
            if isinstance(val, float):
                detail_parts.append(f"{key}={val:.4g}")
            else:
                detail_parts.append(f"{key}={val}")
        detail_parts.append(f"MAE={hist['best_mae']:.2f}")
        xy = [
            {"x": int(p["update"]), "y": float(p["mae_mg_dl"])}
            for p in hist["points"]
        ]
        line_datasets.append(
            {
                "label": " ".join(detail_parts),
                "data": xy,
                "borderColor": color,
                "backgroundColor": color,
                "tension": 0.2,
                "pointRadius": 3,
                "pointHoverRadius": 6,
            }
        )

    confirm = payload.get("confirm")
    confirm_block = ""
    confirm_data: dict[str, Any] | None = None
    if confirm is not None and confirm["epochs"]:
        cfg = confirm["config"]
        confirm_data = {
            "labels": [int(r["epoch"]) for r in confirm["epochs"]],
            "values": [float(r["mae_mg_dl"]) for r in confirm["epochs"]],
        }
        confirm_block = f"""
        <section>
          <h2>PC confirmation train</h2>
          <p>
            Best val MAE <strong>{_fmt(cfg.get('best_val_mae_mg_dl'), 4)}</strong>,
            test MAE <strong>{_fmt(cfg.get('test_mae_mg_dl'), 4)}</strong>,
            epochs {html.escape(str(cfg.get('final_epoch')))},
            {html.escape(str(cfg.get('elapsed_s')))}s.
            Hover points for exact levels; click legend to toggle.
          </p>
          <div class="chart"><canvas id="confirmChart" height="120"></canvas></div>
        </section>
        """

    leaderboard_rows = []
    for row in payload["leaderboard"]:
        params = row["params"]
        geometry = (
            f"{params.get('seq_len')}/d{params.get('depth')}/h{params.get('num_heads')}"
        )
        leaderboard_rows.append(
            "<tr>"
            f"<td>{row['trial']}</td>"
            f"<td>{_fmt(row['best_mae'], 3)}</td>"
            f"<td>{_fmt(row.get('best_mard'), 2)}</td>"
            f"<td>{html.escape(geometry)}</td>"
            f"<td>{float(params.get('lr') or 0):.4g}</td>"
            f"<td>{float(params.get('eta_infer') or 0):.4g}</td>"
            f"<td>{params.get('infer_steps')}</td>"
            f"<td>{params.get('grad_clip')}</td>"
            "</tr>"
        )

    arch_param_keys = [
        "seq_len", "depth", "num_heads", "lr", "eta_infer", "infer_steps",
        "max_infer_norm", "grad_clip", "lr_decay_epochs", "weight_init_std",
        "weight_decay", "readout", "seed_offset",
    ]
    arch_cards = []
    for rank, row in enumerate(payload["leaderboard"][:5], 1):
        params = row["params"]
        geometry = (
            f"seq_len={params.get('seq_len')}, depth={params.get('depth')}, "
            f"heads={params.get('num_heads')}"
        )
        readout = params.get("readout") or "flatten"
        param_rows = []
        for key in arch_param_keys:
            val = params.get(key)
            if val is None:
                continue
            param_rows.append(
                f"<tr><td>{_tip(key)}</td><td>{_fmt_param_value(val)}</td></tr>"
            )
        mard_str = (
            f" · MARD {_fmt(row.get('best_mard'), 1)}%"
            if row.get("best_mard") is not None
            else ""
        )
        arch_cards.append(
            f'<div class="arch-card">'
            f'<div class="arch-rank">#{rank}</div>'
            f'<div class="arch-trial">Trial {row["trial"]}</div>'
            f'<div class="arch-mae">MAE {_fmt(row["best_mae"], 3)}{mard_str}</div>'
            f'<div class="arch-desc">{html.escape(geometry)}</div>'
            f'<div class="arch-detail">readout: {_tip("readout", readout)}</div>'
            f'<table class="arch-params">{"".join(param_rows)}</table>'
            f"</div>"
        )

    trial_section, trial_charts = _trial_sections_html(payload)
    charts_json = json.dumps(
        {
            "bar": {
                "labels": bar_labels,
                "values": [v if v is not None else None for v in bar_values],
                "colors": bar_colors,
                "bestMae": best_mae,
            },
            "lines": {"datasets": line_datasets, "xTitle": x_title},
            "confirm": confirm_data,
            "trialCharts": trial_charts,
        }
    )

    th_lr = _tip("lr", "LR")
    th_eta = _tip("eta_infer", "η_infer")
    th_steps = _tip("infer_steps", "Infer steps")
    th_clip = _tip("grad_clip", "Grad clip")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Glucose PC Optuna progress</title>
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
    main {{ max-width:1040px; margin:0 auto; padding:32px 20px 80px; }}
    h1,h2,h3 {{ margin:0 0 12px; font-weight:650; }}
    h1 {{ font-size:1.8rem; }} h2 {{ font-size:1.2rem; margin-top:28px; }}
    h3 {{ font-size:1rem; margin-top:16px; color:var(--text); }}
    p,li {{ color:var(--muted); }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
      gap:12px; margin:20px 0; }}
    .stat {{ background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:14px 16px; }}
    .stat strong {{ display:block; font-size:1.4rem; color:var(--text); }}
    .stat span {{ color:var(--muted); font-size:0.85rem; }}
    .chart {{ margin:16px 0; border:1px solid var(--line); border-radius:12px;
      background:var(--card); padding:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; }}
    th {{ color:var(--muted); font-weight:600; }}
    code {{ color:var(--accent); }}
    .hint {{ font-size:0.85rem; color:var(--muted); }}
    .callout {{ background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:16px 18px; }}
    .callout h3 {{ margin-top:14px; }} .callout h3:first-child {{ margin-top:0; }}
    .tip {{ position:relative; cursor:help; border-bottom:1px dotted var(--accent); }}
    .tip:hover::after, .tip:focus::after {{
      content: attr(data-tip);
      position:absolute; left:0; bottom:calc(100% + 8px); z-index:40;
      min-width:220px; max-width:320px; padding:10px 12px;
      background:#1a2332; color:var(--text); border:1px solid var(--line);
      border-radius:8px; font-size:0.8rem; line-height:1.35; white-space:normal;
      box-shadow:0 8px 24px rgba(0,0,0,0.35); pointer-events:none;
    }}
    .arch-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
      gap:14px; margin:16px 0; }}
    .arch-card {{ background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:16px; position:relative; }}
    .arch-rank {{ position:absolute; top:12px; right:16px; font-size:1.6rem;
      font-weight:700; color:var(--line); }}
    .arch-trial {{ font-size:0.85rem; font-weight:600; color:var(--muted); }}
    .arch-mae {{ font-size:1.2rem; font-weight:650; color:var(--accent); margin:4px 0 6px; }}
    .arch-desc {{ font-size:0.95rem; color:var(--text); }}
    .arch-detail {{ font-size:0.85rem; color:var(--muted); margin-bottom:10px; }}
    .arch-params td {{ padding:3px 8px; border-bottom:1px solid var(--line); }}
    .trial-card, .theory-card, .fold-block {{
      background:var(--card); border:1px solid var(--line); border-radius:12px;
      margin:10px 0; padding:0 14px; }}
    .trial-card > summary, .theory-card > summary, .fold-block > summary {{
      cursor:pointer; list-style:none; padding:14px 0; color:var(--text); font-weight:600; }}
    .trial-card > summary::-webkit-details-marker,
    .theory-card > summary::-webkit-details-marker,
    .fold-block > summary::-webkit-details-marker {{ display:none; }}
    .fold-block > summary::before {{ content:"▸ "; color:var(--accent); }}
    .fold-block[open] > summary::before {{ content:"▾ "; }}
    .trial-body {{ padding:0 0 16px; }}
    .param-list {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
      gap:4px 12px; padding-left:18px; }}
    .badge {{ display:inline-block; font-size:0.75rem; padding:2px 8px;
      border-radius:999px; border:1px solid var(--line); color:var(--muted); }}
    .badge.ok {{ color:#3dd68c; border-color:#3dd68c55; }}
    .badge.warn {{ color:#f5a524; border-color:#f5a52455; }}
    .badge.bad {{ color:#f31260; border-color:#f3126055; }}
    .badge.run {{ color:#006FEE; border-color:#006FEE55; }}
    .toc a {{ color:var(--accent); margin-right:12px; font-size:0.9rem; }}
    @media (max-width:800px) {{
      .stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Glucose PC Optuna progress</h1>
    <p>
      Study <code>{html.escape(payload['study_name'])}</code> ·
      predictive coding only · generated
      <code>{html.escape(payload['generated_at'])}</code>
    </p>
    <p class="hint">
      Interactive charts (Chart.js): hover for exact levels, axes labelled, click legend to show/hide.
      Open in Chrome/Edge/Firefox — IDE HTML previews may block the CDN.
    </p>
    <p class="toc">
      <a href="#overview">Overview</a>
      <a href="#curves">Learning curves</a>
      <a href="#leaderboard">Leaderboard</a>
      <a href="#complete">All trials</a>
      <a href="#theory">Theory</a>
      <a href="#how-to-run">How to run</a>
    </p>
    <div class="stats" id="overview">
      <div class="stat"><strong>{counts['n_trials']}</strong><span>Trials</span></div>
      <div class="stat"><strong>{counts['n_complete']}</strong><span>Complete</span></div>
      <div class="stat"><strong>{counts['n_pruned']}</strong><span>Pruned</span></div>
      <div class="stat"><strong>{_fmt(best_mae, 2)}</strong><span>Best MAE (mg/dL)</span></div>
    </div>
    {TRAINING_BASICS_HTML}
    <section>
      <h2>What helped (auto-generated from top trials)</h2>
      <pre style="font-size:0.85rem; color:var(--muted); white-space:pre-wrap;">{html.escape(_summarize_what_helped(payload['leaderboard']))}</pre>
    </section>
    <section>
      <h2>Best MAE by trial</h2>
      <p class="hint">
        <span style="color:#3dd68c;">■</span> ≤21.5&ensp;
        <span style="color:#f5a524;">■</span> pruned / mid&ensp;
        <span style="color:#f31260;">■</span> high / fail&ensp;
        Dashed line = best complete MAE.
      </p>
      <div class="chart"><canvas id="barChart" height="120"></canvas></div>
    </section>
    <section id="curves">
      <h2>Top trial learning curves</h2>
      <p class="hint">X = training progress inside a trial; Y = validation MAE. Lower is better.</p>
      <div class="chart"><canvas id="lineChart" height="120"></canvas></div>
    </section>
    {confirm_block}
    <details class="fold-block" id="leaderboard">
      <summary>Complete-trial leaderboard</summary>
      <p class="hint">Hover column headers marked with a dotted underline for parameter explainers.</p>
      <table>
        <thead>
          <tr>
            <th>Trial</th><th>Best MAE</th><th>MARD%</th><th>Geometry</th>
            <th>{th_lr}</th><th>{th_eta}</th><th>{th_steps}</th><th>{th_clip}</th>
          </tr>
        </thead>
        <tbody>{''.join(leaderboard_rows)}</tbody>
      </table>
    </details>
    <section>
      <h2>Top model architectures</h2>
      <p class="hint">Hover parameter names for short explainers.</p>
      <div class="arch-grid">{''.join(arch_cards)}</div>
    </section>
    {trial_section}
    <div id="theory">{_theory_section_html()}</div>
    <section>
      <h2>Background</h2>
      <p class="hint">
        Builds on <a href="https://github.com/GlucoseDAO/glucose-forecasting" style="color:var(--accent);">GlucoseDAO/glucose-forecasting</a>.
        Here the forward pass uses <strong>predictive coding (PC)</strong>: each layer holds a belief,
        computes a prediction error, and refines activations before the outer weight update.
      </p>
    </section>
    <section>
      <h2>How the model works</h2>
      <div class="chart">{_svg_architecture_diagram()}</div>
      <div class="chart">{_svg_energy_comparison()}</div>
    </section>
    <section id="how-to-run">
      <h2>How to run</h2>
      <table style="font-size:0.85rem;">
        <tbody>
          <tr><td style="color:var(--muted);">Install (CPU)</td>
            <td><code>uv sync --extra glucose</code></td></tr>
          <tr><td style="color:var(--muted);">Install (GPU / WSL)</td>
            <td><code>uv sync --extra glucose --extra cuda12</code></td></tr>
          <tr><td style="color:var(--muted);">Train PC transformer</td>
            <td><code>uv run glucose-transformer</code></td></tr>
          <tr><td style="color:var(--muted);">Epoch Optuna (default)</td>
            <td><code>uv run glucose-transformer-tune run</code></td></tr>
          <tr><td style="color:var(--muted);">Update-budget Optuna</td>
            <td><code>uv run glucose-transformer-tune-update-budget run</code></td></tr>
          <tr><td style="color:var(--muted);">Regenerate this report</td>
            <td><code>uv run python scripts/generate_glucose_tuning_report.py --format all</code></td></tr>
          <tr><td style="color:var(--muted);">All studies + master</td>
            <td><code>uv run python scripts/generate_all_glucose_reports.py --format all</code></td></tr>
          <tr><td style="color:var(--muted);">Master report only</td>
            <td><code>uv run python scripts/generate_glucose_master_report.py</code></td></tr>
        </tbody>
      </table>
    </section>
  </main>
  <script>
    const DATA = {charts_json};
    const tickColor = "#9aa7b8";
    const gridColor = "#243041";
    const common = {{
      responsive: true,
      maintainAspectRatio: true,
      interaction: {{ mode: "nearest", intersect: false }},
      plugins: {{
        legend: {{
          display: true,
          labels: {{ color: tickColor, boxWidth: 12, usePointStyle: true }},
        }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const y = ctx.parsed.y;
              const x = ctx.parsed.x ?? ctx.label;
              if (y == null) return `${{ctx.dataset.label}}: n/a`;
              return `${{ctx.dataset.label}}: ${{y.toFixed(3)}} mg/dL @ ${{x}}`;
            }},
          }},
        }},
      }},
      scales: {{
        x: {{
          ticks: {{ color: tickColor }},
          grid: {{ color: gridColor }},
          title: {{ display: true, color: tickColor }},
        }},
        y: {{
          ticks: {{ color: tickColor }},
          grid: {{ color: gridColor }},
          title: {{ display: true, text: "MAE (mg/dL)", color: tickColor }},
        }},
      }},
    }};

    new Chart(document.getElementById("barChart"), {{
      type: "bar",
      data: {{
        labels: DATA.bar.labels,
        datasets: [{{
          label: "Best val MAE (mg/dL)",
          data: DATA.bar.values,
          backgroundColor: DATA.bar.colors,
          borderRadius: 4,
        }}],
      }},
      options: {{
        ...common,
        plugins: {{
          ...common.plugins,
          annotation: DATA.bar.bestMae == null ? {{}} : {{
            annotations: {{
              bestLine: {{
                type: "line",
                yMin: DATA.bar.bestMae,
                yMax: DATA.bar.bestMae,
                borderColor: "#3dd68c",
                borderWidth: 2,
                borderDash: [6, 4],
                label: {{
                  display: true,
                  content: `Best ${{DATA.bar.bestMae.toFixed(2)}}`,
                  position: "end",
                  backgroundColor: "#121820",
                  color: "#3dd68c",
                }},
              }},
            }},
          }},
        }},
        scales: {{
          ...common.scales,
          x: {{
            ...common.scales.x,
            title: {{ display: true, text: "Trial", color: tickColor }},
          }},
          y: {{
            ...common.scales.y,
            suggestedMin: 18,
            suggestedMax: 50,
          }},
        }},
      }},
    }});

    new Chart(document.getElementById("lineChart"), {{
      type: "line",
      data: {{ datasets: DATA.lines.datasets }},
      options: {{
        ...common,
        scales: {{
          x: {{
            type: "linear",
            ticks: {{ color: tickColor }},
            grid: {{ color: gridColor }},
            title: {{ display: true, text: DATA.lines.xTitle || "Update", color: tickColor }},
          }},
          y: {{
            ...common.scales.y,
            suggestedMin: 18,
            suggestedMax: 40,
          }},
        }},
      }},
    }});

    if (DATA.confirm) {{
      new Chart(document.getElementById("confirmChart"), {{
        type: "line",
        data: {{
          labels: DATA.confirm.labels,
          datasets: [{{
            label: "Val MAE (mg/dL)",
            data: DATA.confirm.values,
            borderColor: "#3dd68c",
            backgroundColor: "rgba(61,214,140,0.15)",
            fill: true,
            tension: 0.25,
            pointRadius: 4,
            pointHoverRadius: 7,
          }}],
        }},
        options: {{
          ...common,
          scales: {{
            x: {{
              ...common.scales.x,
              title: {{ display: true, text: "Epoch", color: tickColor }},
            }},
            y: {{
              ...common.scales.y,
              suggestedMin: 18,
              suggestedMax: 34,
            }},
          }},
        }},
      }});
    }}

    const trialChartById = {{}};
    (DATA.trialCharts || []).forEach((tc) => {{
      const el = document.getElementById(tc.id);
      if (!el || !tc.data.length) return;
      trialChartById[tc.id] = new Chart(el, {{
        type: "line",
        data: {{
          datasets: [{{
            label: tc.label,
            data: tc.data,
            borderColor: tc.color,
            backgroundColor: tc.color,
            tension: 0.2,
            pointRadius: 3,
            pointHoverRadius: 6,
          }}],
        }},
        options: {{
          ...common,
          plugins: {{
            ...common.plugins,
            legend: {{ display: true, labels: {{ color: tickColor }} }},
          }},
          scales: {{
            x: {{
              type: "linear",
              ticks: {{ color: tickColor }},
              grid: {{ color: gridColor }},
              title: {{ display: true, text: tc.xTitle, color: tickColor }},
            }},
            y: {{
              ...common.scales.y,
              title: {{ display: true, text: "Val MAE (mg/dL)", color: tickColor }},
            }},
          }},
        }},
      }});
    }});
    document.querySelectorAll("details.trial-card").forEach((det) => {{
      det.addEventListener("toggle", () => {{
        if (!det.open) return;
        det.querySelectorAll("canvas").forEach((canvas) => {{
          const chart = trialChartById[canvas.id];
          if (chart) chart.resize();
        }});
      }});
    }});
  </script>
</body>
</html>
"""


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, help="Optuna study directory."),
    study_name: str | None = typer.Option(
        None, help="Optuna study name (default: from coordinator_config.json)."
    ),
    confirm_dir: Path = typer.Option(
        CONFIRM_DIR, help="Optional PC confirmation run directory."
    ),
    format: str = typer.Option(
        "all",
        "--format",
        help="Output format: md, html, json, or all.",
    ),
    refresh: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Re-export results_snapshot.json from the Optuna journal first.",
    ),
) -> None:
    """Refresh results and write human-readable progress reports."""
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
        canvas_path = run_dir / "canvas_payload.json"
        canvas_path.write_text(json.dumps(payload, indent=2))
        typer.echo(f"wrote {canvas_path}")
    if "md" in wanted:
        md_path = run_dir / "report.md"
        md_text = _render_markdown(payload)
        md_path.write_text(md_text, encoding="utf-8")
        typer.echo(f"wrote {md_path}")
        resolved = _resolve_study_name(run_dir, study_name)
        docs_md = Path(f"docs/reports/old/{resolved}_progress.md")
        docs_md.parent.mkdir(parents=True, exist_ok=True)
        docs_md.write_text(md_text, encoding="utf-8")
        typer.echo(f"wrote {docs_md}")
    if "html" in wanted:
        html_path = run_dir / "report.html"
        html_text = _render_html(payload)
        html_path.write_text(html_text, encoding="utf-8")
        typer.echo(f"wrote {html_path}")
        resolved = _resolve_study_name(run_dir, study_name)
        docs_html = Path(f"docs/reports/old/{resolved}_progress.html")
        docs_html.parent.mkdir(parents=True, exist_ok=True)
        docs_html.write_text(html_text, encoding="utf-8")
        typer.echo(f"wrote {docs_html}")

    best = payload.get("best_complete")
    if best is not None:
        typer.echo(
            f"best_complete trial={best['trial']} "
            f"mae={best['best_history_mae']:.4f}"
        )


if __name__ == "__main__":
    app()
