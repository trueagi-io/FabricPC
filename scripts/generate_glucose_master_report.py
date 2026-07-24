"""Generate one master HTML/MD report covering ALL glucose Optuna studies.

Layout (requested):
1. Schema / theory first — Hopfield architecture diagram + hyperparameter glossary
2. Per-study search-space tables (what Optuna was allowed to sample)
3. Cross-study overview + interactive charts (axes labelled, best-MAE line)
4. Confirmation trains

Usage::

    uv run python scripts/generate_glucose_master_report.py
    uv run python scripts/generate_glucose_master_report.py --format html
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
    help="Render one master report across all glucose Optuna studies.",
)

OUT_HTML = Path("docs/reports/glucose_master_progress.html")
OUT_MD = Path("docs/reports/glucose_master_progress.md")
OUT_JSON = Path("docs/reports/glucose_master_progress.json")

# Ordered catalog: category → studies. Missing run dirs are skipped.
CATEGORIES: list[dict[str, Any]] = [
    {
        "id": "transformer_update_budget",
        "title": "1. Transformer PC — update-budget Optuna",
        "blurb": (
            "Archived update-budget search (phases 1–4). Validation every N "
            "optimizer updates; Median/Hyperband-style pruning on update checks."
        ),
        "studies": [
            {
                "run_dir": "runs/glucose_tuning",
                "label": "Phase 1 — broad search",
                "family": "transformer",
                "space_key": "transformer_phase1",
            },
            {
                "run_dir": "runs/glucose_tuning_pc_v2",
                "label": "Phase 2 — refined",
                "family": "transformer",
                "space_key": "transformer_refined",
            },
            {
                "run_dir": "runs/glucose_tuning_pc_local",
                "label": "Phase 3 — local",
                "family": "transformer",
                "space_key": "transformer_local",
            },
            {
                "run_dir": "runs/glucose_tuning_pc_breakthrough",
                "label": "Phase 4 — breakthrough",
                "family": "transformer",
                "space_key": "transformer_breakthrough",
            },
        ],
    },
    {
        "id": "transformer_epochs",
        "title": "2. Transformer PC — epoch Hyperband Optuna",
        "blurb": (
            "Default epoch-based tuner (`glucose-transformer-tune`). Full epochs "
            "with Hyperband pruning; produced the current PC champion (~19.1 MAE)."
        ),
        "studies": [
            {
                "run_dir": "runs/glucose_tuning_epochs_v1",
                "label": "Epochs v1 (Hyperband)",
                "family": "transformer",
                "space_key": "transformer_epochs",
            },
        ],
    },
    {
        "id": "hopfield",
        "title": "3. Hopfield PC Optuna",
        "blurb": (
            "Same PC backbone plus Hopfield associative-memory variants "
            "(baseline / embed-storkey / forecast-storkey / projection)."
        ),
        "studies": [
            {
                "run_dir": "runs/glucose_hopfield_tuning_v1",
                "label": "Hopfield v1 (native / early)",
                "family": "hopfield",
                "space_key": "hopfield_v1",
            },
            {
                "run_dir": "runs/glucose_hopfield_tuning_wsl_v1",
                "label": "Hopfield WSL v1",
                "family": "hopfield",
                "space_key": "hopfield_wsl_v1",
            },
            {
                "run_dir": "runs/glucose_hopfield_tuning_wsl_v2",
                "label": "Hopfield WSL v2",
                "family": "hopfield",
                "space_key": "hopfield_wsl_v2",
            },
        ],
    },
    {
        "id": "confirms",
        "title": "4. Confirmation trains (single-config replay)",
        "blurb": (
            "Longer epoch loops that replay a winning Optuna config and report "
            "validation + held-out test metrics."
        ),
        "confirms": [
            {
                "run_dir": "runs/glucose_pc_best_confirm",
                "label": "Phase-1 champion confirm",
            },
            {
                "run_dir": "runs/glucose_pc_breakthrough_confirm",
                "label": "Phase-4 breakthrough confirm",
            },
        ],
    },
]

# Search spaces each study was allowed to sample (documented from tuners / YAMLs).
SEARCH_SPACES: dict[str, list[dict[str, str]]] = {
    "transformer_phase1": [
        {"param": "seq_len", "range": "categorical {64, 128}"},
        {"param": "depth", "range": "int 1–3"},
        {"param": "num_heads", "range": "categorical {1, 2, 4}"},
        {"param": "lr", "range": "float log 3e-4 – 5e-3"},
        {"param": "eta_infer", "range": "float log 1e-5 – 5e-4"},
        {"param": "infer_steps", "range": "int 8–24"},
        {"param": "max_infer_norm", "range": "categorical {0.5, 1.0, 5.0}"},
        {"param": "grad_clip", "range": "categorical {0.5, 1.0, 2.0}"},
        {"param": "weight_init_std", "range": "float log 0.01 – 0.03"},
        {"param": "weight_decay", "range": "fixed 0.0"},
        {"param": "readout", "range": "fixed flatten"},
    ],
    "transformer_refined": [
        {"param": "seq_len", "range": "categorical {64}"},
        {"param": "depth", "range": "int 1–2"},
        {"param": "num_heads", "range": "categorical {1, 4}"},
        {"param": "lr", "range": "float log 1e-3 – 5e-3"},
        {"param": "eta_infer", "range": "float log 3e-6 – 8e-5"},
        {"param": "infer_steps", "range": "int 10–20"},
        {"param": "max_infer_norm", "range": "categorical {0.5, 1.0}"},
        {"param": "grad_clip", "range": "categorical {0.5, 1.0}"},
        {"param": "weight_init_std", "range": "float log 0.012 – 0.025"},
        {"param": "weight_decay", "range": "float log 1e-6 – 1e-3"},
        {"param": "readout", "range": "fixed flatten"},
    ],
    "transformer_local": [
        {"param": "seq_len", "range": "categorical {64}"},
        {"param": "depth", "range": "categorical {1}"},
        {"param": "num_heads", "range": "categorical {1}"},
        {"param": "lr", "range": "float log 1.5e-3 – 4.0e-3"},
        {"param": "eta_infer", "range": "float log 8e-6 – 3e-5"},
        {"param": "infer_steps", "range": "int 12–18"},
        {"param": "max_infer_norm", "range": "categorical {0.5, 1.0}"},
        {"param": "grad_clip", "range": "categorical {0.5, 1.0}"},
        {"param": "weight_init_std", "range": "float log 0.014 – 0.022"},
        {"param": "weight_decay", "range": "fixed 0.0"},
        {"param": "readout", "range": "fixed flatten"},
    ],
    "transformer_breakthrough": [
        {"param": "seq_len", "range": "categorical {64}"},
        {"param": "depth", "range": "categorical {1}"},
        {"param": "num_heads", "range": "categorical {1}"},
        {"param": "lr", "range": "float log 1.8e-3 – 3.8e-3"},
        {"param": "eta_infer", "range": "float log 9e-6 – 2.5e-5"},
        {"param": "infer_steps", "range": "int 12–18"},
        {"param": "max_infer_norm", "range": "categorical {0.5, 1.0}"},
        {"param": "grad_clip", "range": "categorical {0.5, 1.0}"},
        {"param": "weight_init_std", "range": "float log 0.014 – 0.021"},
        {"param": "weight_decay", "range": "fixed 0.0"},
        {"param": "readout", "range": "categorical {flatten, mean_pool, last}"},
        {"param": "seed_offset", "range": "int 0–40"},
    ],
    "transformer_epochs": [
        {"param": "seq_len", "range": "categorical {64, 128}"},
        {"param": "depth", "range": "int 1–3"},
        {"param": "num_heads", "range": "categorical {1, 2, 4}"},
        {"param": "lr", "range": "float log 3e-4 – 5e-3"},
        {"param": "eta_infer", "range": "float log 1e-5 – 5e-4"},
        {"param": "infer_steps", "range": "int 8–24"},
        {"param": "max_infer_norm", "range": "categorical {0.5, 1.0, 5.0}"},
        {"param": "grad_clip", "range": "categorical {0.5, 1.0, 2.0}"},
        {"param": "lr_decay_epochs", "range": "categorical {5, 10, 15}"},
        {"param": "weight_init_std", "range": "float log 0.01 – 0.03"},
        {"param": "energy", "range": "categorical {gaussian, huber}"},
        {"param": "ipc", "range": "categorical {true, false}"},
        {"param": "infer_optimizer", "range": "categorical {sgd, adam}"},
        {"param": "huber_delta", "range": "float log 0.1 – 2.0 (if energy=huber)"},
    ],
    "hopfield_v1": [
        {"param": "seq_len", "range": "categorical {64}"},
        {"param": "depth", "range": "categorical {1}"},
        {"param": "num_heads", "range": "categorical {1}"},
        {
            "param": "variant",
            "range": "categorical {baseline, embed-storkey, forecast-storkey, projection}",
        },
        {
            "param": "hopfield_strength",
            "range": "categorical {0.5, 1.0, 1.5, 2.0, learnable}",
        },
        {"param": "lr", "range": "float log ~1.8e-3 – 3.8e-3 (breakthrough band)"},
        {"param": "eta_infer", "range": "float log ~9e-6 – 2.5e-5"},
        {"param": "infer_steps", "range": "int ~12–20"},
        {"param": "max_infer_norm", "range": "categorical {0.5, 1.0}"},
        {"param": "grad_clip", "range": "categorical {0.5, 1.0}"},
        {"param": "lr_decay_epochs", "range": "categorical {5, 10, 15}"},
        {"param": "weight_init_std", "range": "float log ~0.013 – 0.022"},
        {"param": "seed_offset", "range": "int 0–40"},
    ],
    "hopfield_wsl_v1": [
        {"param": "seq_len", "range": "categorical {64}"},
        {"param": "depth", "range": "categorical {1}"},
        {"param": "num_heads", "range": "categorical {1}"},
        {
            "param": "variant",
            "range": "categorical {baseline, embed-storkey, forecast-storkey, projection}",
        },
        {
            "param": "hopfield_strength",
            "range": "categorical {0.5, 1.0, 1.5, 2.0, learnable}",
        },
        {"param": "lr", "range": "float log ~1.8e-3 – 3.8e-3"},
        {"param": "eta_infer", "range": "float log ~9e-6 – 2.5e-5"},
        {"param": "infer_steps", "range": "int ~12–20"},
        {"param": "max_infer_norm", "range": "categorical {0.5, 1.0}"},
        {"param": "grad_clip", "range": "categorical {0.5, 1.0}"},
        {"param": "lr_decay_epochs", "range": "categorical {5, 10, 15}"},
        {"param": "weight_init_std", "range": "float log ~0.013 – 0.022"},
        {"param": "seed_offset", "range": "int 0–40"},
    ],
    "hopfield_wsl_v2": [
        {"param": "seq_len", "range": "categorical {64}"},
        {"param": "depth", "range": "categorical {1}"},
        {"param": "num_heads", "range": "categorical {1}"},
        {"param": "variant", "range": "categorical {baseline, projection}"},
        {
            "param": "hopfield_strength",
            "range": "categorical {0.5, 1.0, learnable}",
        },
        {"param": "lr", "range": "float log 1.8e-3 – 3.8e-3"},
        {"param": "eta_infer", "range": "float log 9e-6 – 2.5e-5"},
        {"param": "infer_steps", "range": "int 12–20"},
        {"param": "max_infer_norm", "range": "categorical {0.5, 1.0}"},
        {"param": "grad_clip", "range": "categorical {0.5, 1.0}"},
        {"param": "lr_decay_epochs", "range": "categorical {5, 10, 15}"},
        {"param": "weight_init_std", "range": "float log 0.013 – 0.022"},
        {"param": "seed_offset", "range": "int 0–40"},
    ],
}

PARAM_THEORY: dict[str, dict[str, str]] = {
    "seq_len": {
        "short": "How many recent 5-min CGM readings the model sees.",
        "what": "History window length fed into the network.",
        "why": "Too short misses delayed glucose effects; too long adds noise and cost.",
        "effect": "64 ≈ 5.3 hours of context; longer is not automatically better.",
    },
    "depth": {
        "short": "How many transformer blocks are stacked (shallowness).",
        "what": "Number of attention+MLP stages stacked.",
        "why": "Depth is capacity. Shallow nets are simpler and often better on small CGM sets.",
        "effect": "depth=1 is a short assembly line; deeper stacks can overfit or train slower.",
    },
    "num_heads": {
        "short": "Parallel attention viewpoints in each block.",
        "what": "Multi-head attention count.",
        "why": "More heads can specialise, but need enough width/data.",
        "effect": "1 head is common and stable on small models.",
    },
    "variant": {
        "short": (
            "Architecture family Optuna chooses "
            "(baseline / embed-storkey / forecast-storkey / projection)."
        ),
        "what": (
            "Which graph wiring to train: where Hopfield memory sits, or baseline "
            "(no Hopfield). This is an Optuna categorical choice — each trial "
            "gets one architecture from the search space."
        ),
        "why": (
            "Placement changes when associative recall can influence features vs "
            "the final forecast. Optuna explores variants; it is not fixed by hand "
            "per trial."
        ),
        "effect": (
            "baseline = pure transformer control. embed-* recalls early; "
            "forecast-* recalls late; projection is a lighter linear memory."
        ),
    },
    "hopfield_strength": {
        "short": "How strongly Hopfield memory mixes into activations.",
        "what": "Fixed scale (e.g. 0.5–2.0) or 'learnable'.",
        "why": "Too strong can overwrite useful transformer features; too weak is a no-op.",
        "effect": "learnable lets training pick the mix; fixed values are easier to compare across trials.",
    },
    "lr": {
        "short": "Outer Adam learning rate for weight updates.",
        "what": "Step size for updating model weights.",
        "why": "Too high diverges; too low never improves in budget.",
        "effect": "Mid-range ~1e-3–4e-3 often works with champion-like PC settings.",
    },
    "eta_infer": {
        "short": "Inner PC step size for refining latent beliefs.",
        "what": "Learning rate of the PC inference loop.",
        "why": "Separate from weight LR — controls how hard latents correct prediction error.",
        "effect": "Around 1e-5–2.5e-5 was a healthy band in transformer PC runs.",
    },
    "infer_steps": {
        "short": "Inner PC iterations per forward pass.",
        "what": "How many times latents are refined before forecasting.",
        "why": "More steps → tighter energy, more compute.",
        "effect": "Low teens (12–18) are typical; doubling rarely helps if η is wrong.",
    },
    "max_infer_norm": {
        "short": "Clip on PC latent update size.",
        "what": "Max norm for inner-loop updates.",
        "why": "Prevents exploding activations on sharp glucose swings.",
        "effect": "Lower = safer/slower settle; higher = freer but riskier.",
    },
    "grad_clip": {
        "short": "Clip on outer weight gradients.",
        "what": "Global grad clip for Adam.",
        "why": "Stops rare huge gradients from wrecking a run.",
        "effect": "0.5–1.0 are common stable choices.",
    },
    "lr_decay_epochs": {
        "short": "When cosine LR decay begins.",
        "what": "Epoch index that starts annealing LR.",
        "why": "Balances exploration early vs fine-tuning later.",
        "effect": "Later decay keeps LR high longer.",
    },
    "weight_init_std": {
        "short": "Scale of random initial weights.",
        "what": "Normal init standard deviation.",
        "why": "Interacts with PC dynamics and depth.",
        "effect": "Smaller often safer with PC; larger can help or explode.",
    },
    "weight_decay": {
        "short": "L2 penalty that discourages huge weights.",
        "what": "Weight decay regularisation strength.",
        "why": "On small datasets, unconstrained weights memorise noise.",
        "effect": "Higher → stronger regularisation (can underfit). Zero → freer fit.",
    },
    "readout": {
        "short": "How the sequence is turned into one glucose forecast.",
        "what": "Regression head mode: flatten / mean_pool / last.",
        "why": "Maps a sequence of vectors to a single 60-min-ahead number.",
        "effect": "flatten often best here; mean_pool / last are lighter heads.",
    },
    "seed_offset": {
        "short": "Seed nudge so similar configs can still differ.",
        "what": "Added to the base random seed.",
        "why": "PC runs can be seed-sensitive.",
        "effect": "Document the winning seed for fair replay.",
    },
    "energy": {
        "short": "How PC nodes score prediction error (Gaussian vs Huber).",
        "what": "Energy functional used inside PC nodes.",
        "why": "Gaussian punishes large errors hard; Huber is more robust to spikes.",
        "effect": "Huber can help when a few wild glucose points would dominate.",
    },
    "huber_delta": {
        "short": "Threshold where Huber switches from quadratic to linear.",
        "what": "Delta parameter for Huber energy (only if energy=huber).",
        "why": "Controls when an error is treated as an outlier.",
        "effect": "Smaller delta → more robust; larger → closer to plain MSE.",
    },
    "ipc": {
        "short": "Update latents layer-by-layer (incremental PC) vs all at once.",
        "what": "Incremental Predictive Coding flag.",
        "why": "Layerwise updates can improve convergence on deeper stacks.",
        "effect": "On shallow nets the difference may be small.",
    },
    "infer_optimizer": {
        "short": "Optimiser inside the PC loop: SGD or Adam.",
        "what": "Which optimiser nudges latent activations during inference.",
        "why": "SGD is simple/fast; Adam adapts per-coordinate.",
        "effect": "Try SGD first; Adam if inference looks under-converged.",
    },
}

PARAM_TIPS: dict[str, str] = {
    key: meta["short"] for key, meta in PARAM_THEORY.items()
}

PARAM_DISPLAY_KEYS = [
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
    "weight_decay",
    "readout",
    "seed_offset",
    "energy",
    "ipc",
    "infer_optimizer",
    "huber_delta",
]


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _tip(key: str, label: str | None = None) -> str:
    text = label if label is not None else key
    tip = PARAM_TIPS.get(key)
    if tip is None:
        return f"<code>{html.escape(text)}</code>"
    return (
        f'<span class="tip" data-tip="{html.escape(tip, quote=True)}">'
        f"<code>{html.escape(text)}</code></span>"
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_from_trials_dir(run_dir: Path, study_name: str) -> dict[str, Any] | None:
    trials_root = run_dir / "trials"
    if not trials_root.is_dir():
        return None
    rows: list[dict[str, Any]] = []
    for trial_dir in sorted(trials_root.glob("trial_*")):
        try:
            number = int(trial_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        history_path = trial_dir / "history.json"
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.exists()
            else []
        )
        cfg_path = trial_dir / "config.json"
        params: dict[str, Any] = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            for key in PARAM_DISPLAY_KEYS:
                if key in cfg:
                    params[key] = cfg[key]
        best = min((row["mae_mg_dl"] for row in history), default=None)
        rows.append(
            {
                "trial": number,
                "state": "UNKNOWN",
                "best_history_mae": best,
                "params": params,
                "user_attrs": {},
                "history": history,
            }
        )
    if not rows:
        return None
    with_mae = [r for r in rows if r["best_history_mae"] is not None]
    with_mae.sort(key=lambda r: float(r["best_history_mae"]))
    return {
        "study_name": study_name,
        "n_trials": len(rows),
        "n_complete": 0,
        "n_pruned": 0,
        "n_fail": 0,
        "n_running": 0,
        "best_complete": with_mae[0] if with_mae else None,
        "trials": rows,
    }


def _load_study(run_dir: Path, family: str, space_key: str) -> dict[str, Any] | None:
    if not run_dir.exists():
        return None
    config = _load_json(run_dir / "coordinator_config.json") or {}
    study_name = str(config.get("study_name") or run_dir.name)
    snapshot = _load_json(run_dir / "results_snapshot.json")
    if snapshot is None and (run_dir / "optuna_journal.log").exists():
        report_data = _load_json(run_dir / "report_data.json")
        if report_data and report_data.get("all_trials"):
            trials = report_data["all_trials"]
            completed = [
                t
                for t in trials
                if t.get("state") == "COMPLETE" and t.get("best_history_mae") is not None
            ]
            completed.sort(key=lambda t: float(t["best_history_mae"]))
            snapshot = {
                "study_name": report_data.get("study_name", study_name),
                "n_trials": report_data.get("counts", {}).get("n_trials", len(trials)),
                "n_complete": report_data.get("counts", {}).get("n_complete", 0),
                "n_pruned": report_data.get("counts", {}).get("n_pruned", 0),
                "n_fail": report_data.get("counts", {}).get("n_fail", 0),
                "n_running": report_data.get("counts", {}).get("n_running", 0),
                "best_complete": completed[0] if completed else None,
                "trials": trials,
            }
        else:
            snapshot = _snapshot_from_trials_dir(run_dir, study_name)
    if snapshot is None:
        return None

    trials = snapshot.get("trials") or []
    ranked = [t for t in trials if t.get("best_history_mae") is not None]
    ranked.sort(key=lambda t: float(t["best_history_mae"]))
    best = snapshot.get("best_complete")
    if best is None and ranked:
        complete = [t for t in ranked if t.get("state") == "COMPLETE"]
        best = complete[0] if complete else ranked[0]

    return {
        "run_dir": str(run_dir).replace("\\", "/"),
        "family": family,
        "space_key": space_key,
        "search_space": SEARCH_SPACES.get(space_key, []),
        "study_name": snapshot.get("study_name", study_name),
        "config": config,
        "counts": {
            "n_trials": snapshot.get("n_trials", len(trials)),
            "n_complete": snapshot.get("n_complete", 0),
            "n_pruned": snapshot.get("n_pruned", 0),
            "n_fail": snapshot.get("n_fail", 0),
            "n_running": snapshot.get("n_running", 0),
        },
        "best": best,
        "ranked": ranked,
        "trials": trials,
    }


def _load_confirm(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    epochs: list[dict[str, float]] = []
    history_path = run_dir / "history.csv"
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                epochs.append(
                    {
                        "epoch": float(row["epoch"]),
                        "mae_mg_dl": float(row["mae_mg_dl"]),
                        "rmse_mg_dl": float(row.get("rmse_mg_dl") or 0),
                        "mard_percent": float(row.get("mard_percent") or 0),
                    }
                )
    return {
        "run_dir": str(run_dir).replace("\\", "/"),
        "config": config,
        "epochs": epochs,
    }


def _build_master_payload() -> dict[str, Any]:
    categories_out: list[dict[str, Any]] = []
    overview: list[dict[str, Any]] = []

    for category in CATEGORIES:
        cat: dict[str, Any] = {
            "id": category["id"],
            "title": category["title"],
            "blurb": category["blurb"],
            "studies": [],
            "confirms": [],
        }
        for spec in category.get("studies") or []:
            run_dir = Path(spec["run_dir"])
            study = _load_study(
                run_dir,
                family=str(spec["family"]),
                space_key=str(spec["space_key"]),
            )
            if study is None:
                cat["studies"].append(
                    {
                        "label": spec["label"],
                        "run_dir": spec["run_dir"],
                        "space_key": spec["space_key"],
                        "search_space": SEARCH_SPACES.get(str(spec["space_key"]), []),
                        "missing": True,
                    }
                )
                continue
            study["label"] = spec["label"]
            study["missing"] = False
            cat["studies"].append(study)
            best = study.get("best") or {}
            overview.append(
                {
                    "category": category["title"],
                    "label": spec["label"],
                    "family": study["family"],
                    "study_name": study["study_name"],
                    "run_dir": study["run_dir"],
                    "best_mae": best.get("best_history_mae"),
                    "best_trial": best.get("trial"),
                    "n_trials": study["counts"]["n_trials"],
                    "n_complete": study["counts"]["n_complete"],
                }
            )
        for spec in category.get("confirms") or []:
            confirm = _load_confirm(Path(spec["run_dir"]))
            if confirm is None:
                cat["confirms"].append(
                    {
                        "label": spec["label"],
                        "run_dir": spec["run_dir"],
                        "missing": True,
                    }
                )
                continue
            confirm["label"] = spec["label"]
            confirm["missing"] = False
            cat["confirms"].append(confirm)
            cfg = confirm["config"]
            overview.append(
                {
                    "category": category["title"],
                    "label": spec["label"],
                    "family": "confirm",
                    "study_name": spec["run_dir"],
                    "run_dir": confirm["run_dir"],
                    "best_mae": cfg.get("best_val_mae_mg_dl") or cfg.get("test_mae_mg_dl"),
                    "best_trial": None,
                    "n_trials": None,
                    "n_complete": None,
                    "test_mae": cfg.get("test_mae_mg_dl"),
                }
            )
        categories_out.append(cat)

    with_mae = [row for row in overview if row.get("best_mae") is not None]
    with_mae.sort(key=lambda row: float(row["best_mae"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "categories": categories_out,
        "overview": overview,
        "champion": with_mae[0] if with_mae else None,
        "param_theory": PARAM_THEORY,
        "search_spaces": SEARCH_SPACES,
    }


def _param_bits(params: dict[str, Any] | None) -> str:
    if not params:
        return "—"
    parts: list[str] = []
    for key in (
        "seq_len",
        "depth",
        "num_heads",
        "variant",
        "hopfield_strength",
        "lr",
        "eta_infer",
        "infer_steps",
        "readout",
        "energy",
        "ipc",
    ):
        val = params.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            parts.append(f"{key}={val:.4g}")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts) if parts else "—"


def _svg_hopfield_architecture() -> str:
    """Inline SVG showing Hopfield glucose PC architecture variants."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 560" '
        'role="img" aria-label="Glucose Hopfield PC Architecture Variants" '
        'style="max-width:100%;height:auto;">'
        '<rect width="920" height="560" rx="12" fill="#0f1419"/>'
        '<defs><marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L8,3 L0,6" fill="#4b5563"/></marker></defs>'
        '<text x="460" y="30" fill="#e7ecf3" font-size="16" '
        'font-family="Segoe UI,sans-serif" font-weight="650" text-anchor="middle">'
        "Hopfield Variant Architectures (all share PC inference at every node)</text>"
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
        # baseline
        '<rect x="80" y="88" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="155" y="108" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Glucose Input</text>'
        '<line x1="155" y1="118" x2="155" y2="132" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="80" y="132" width="150" height="30" rx="6" fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
        '<text x="155" y="152" fill="#006FEE" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Embedding</text>'
        '<line x1="155" y1="162" x2="155" y2="196" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
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
        '<rect x="80" y="330" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="155" y="350" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Forecast Head</text>'
        # embed-storkey
        '<rect x="310" y="88" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="385" y="108" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Glucose Input</text>'
        '<line x1="385" y1="118" x2="385" y2="132" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="310" y="132" width="150" height="30" rx="6" fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
        '<text x="385" y="152" fill="#006FEE" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Embedding</text>'
        '<line x1="385" y1="162" x2="385" y2="176" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="298" y="176" width="174" height="34" rx="6" fill="#2a1f0a" stroke="#f5a524" stroke-width="2"/>'
        '<text x="385" y="194" fill="#f5a524" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="700">Storkey Hopfield</text>'
        '<text x="385" y="206" fill="#9aa7b8" font-size="8" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">associative memory recall</text>'
        '<line x1="385" y1="210" x2="385" y2="224" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
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
        # forecast-storkey
        '<rect x="540" y="88" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="615" y="108" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Glucose Input</text>'
        '<line x1="615" y1="118" x2="615" y2="132" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="540" y="132" width="150" height="30" rx="6" fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
        '<text x="615" y="152" fill="#006FEE" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Embedding</text>'
        '<line x1="615" y1="162" x2="615" y2="196" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
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
        '<rect x="528" y="302" width="174" height="34" rx="6" fill="#2a1f0a" stroke="#f5a524" stroke-width="2"/>'
        '<text x="615" y="320" fill="#f5a524" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="700">Storkey Hopfield</text>'
        '<text x="615" y="332" fill="#9aa7b8" font-size="8" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">pattern match before forecast</text>'
        '<line x1="615" y1="336" x2="615" y2="350" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="540" y="350" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="615" y="370" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Forecast Head</text>'
        # projection
        '<rect x="745" y="88" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="820" y="108" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Glucose Input</text>'
        '<line x1="820" y1="118" x2="820" y2="132" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="745" y="132" width="150" height="30" rx="6" fill="#1a2332" stroke="#006FEE" stroke-width="1"/>'
        '<text x="820" y="152" fill="#006FEE" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Embedding</text>'
        '<line x1="820" y1="162" x2="820" y2="196" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
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
        '<rect x="733" y="302" width="174" height="34" rx="6" fill="#1a1232" stroke="#9353d3" stroke-width="2"/>'
        '<text x="820" y="320" fill="#9353d3" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="700">Projection Memory</text>'
        '<text x="820" y="332" fill="#9aa7b8" font-size="8" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">lightweight linear recall</text>'
        '<line x1="820" y1="336" x2="820" y2="350" stroke="#4b5563" stroke-width="1" marker-end="url(#ah)"/>'
        '<rect x="745" y="350" width="150" height="30" rx="6" fill="#1a2332" stroke="#3dd68c" stroke-width="1"/>'
        '<text x="820" y="370" fill="#3dd68c" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="600">Forecast Head</text>'
        # PC box
        '<rect x="60" y="400" width="800" height="140" rx="10" fill="#121820" '
        'stroke="#f31260" stroke-width="1"/>'
        '<text x="460" y="424" fill="#f31260" font-size="13" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle" font-weight="650">'
        "Predictive Coding (PC) Inference — runs at EVERY node above</text>"
        '<text x="460" y="448" fill="#e7ecf3" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">'
        "Each node maintains a latent belief z, predicts z_mu from inputs, computes error = z - z_mu</text>"
        '<text x="460" y="470" fill="#e7ecf3" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">'
        "Inner loop: update z via SGD for infer_steps iterations "
        "(step size eta_infer, clip max_infer_norm)</text>"
        '<text x="460" y="492" fill="#e7ecf3" font-size="11" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">'
        "Outer loop: update weights via Adam (step size lr, clip grad_clip)</text>"
        '<text x="460" y="516" fill="#9aa7b8" font-size="10" font-family="Segoe UI,sans-serif" '
        'text-anchor="middle">'
        'hopfield_strength controls how strongly the memory influences activations — '
        '"learnable" lets the model optimise it during training</text>'
        "</svg>"
    )


def _search_space_html(space: list[dict[str, str]]) -> str:
    if not space:
        return '<p class="hint">Search space not documented for this study.</p>'
    rows = []
    for item in space:
        rows.append(
            "<tr>"
            f"<td>{_tip(item['param'])}</td>"
            f"<td>{html.escape(item['range'])}</td>"
            "</tr>"
        )
    return (
        '<div class="arch-card search-space-card">'
        '<div class="arch-desc">Optuna search space</div>'
        '<p class="hint" style="margin:0 0 8px;">'
        "Compact list of what Optuna was allowed to sample. "
        f"{_tip('variant')} is an architecture choice Optuna picks per trial."
        "</p>"
        f'<table class="arch-params">{"".join(rows)}</table>'
        "</div>"
    )


def _variant_desc(variant: Any) -> str:
    mapping = {
        "baseline": "Pure transformer (no Hopfield node)",
        "embed-storkey": "Storkey Hopfield after embedding layer",
        "forecast-storkey": "Storkey Hopfield before forecast head",
        "projection": "Projection Hopfield (linear memory)",
    }
    key = str(variant) if variant is not None else ""
    return mapping.get(key, html.escape(key or "—"))


def _top_arch_cards_html(study: dict[str, Any], limit: int = 5) -> str:
    ranked = [
        t
        for t in (study.get("ranked") or [])
        if t.get("best_history_mae") is not None
    ]
    complete = [t for t in ranked if t.get("state") == "COMPLETE"]
    chosen = (complete or ranked)[:limit]
    if not chosen:
        return '<p class="hint">No architectures to show yet.</p>'
    cards: list[str] = []
    for rank, trial in enumerate(chosen, 1):
        params = trial.get("params") or {}
        variant = params.get("variant")
        strength = params.get("hopfield_strength")
        geometry = (
            f"seq_len={params.get('seq_len')}, depth={params.get('depth')}, "
            f"heads={params.get('num_heads')}"
        )
        param_rows = []
        for key in PARAM_DISPLAY_KEYS:
            val = params.get(key)
            if val is None:
                continue
            display = f"{val:.4g}" if isinstance(val, float) else html.escape(str(val))
            param_rows.append(f"<tr><td>{_tip(key)}</td><td>{display}</td></tr>")
        strength_line = ""
        if strength is not None:
            if str(strength) == "learnable":
                strength_line = (
                    '<div class="arch-strength">Hopfield strength: '
                    "learnable (optimised during training)</div>"
                )
            else:
                strength_line = (
                    f'<div class="arch-strength">Hopfield strength: fixed = '
                    f"{html.escape(str(strength))}</div>"
                )
        desc = _variant_desc(variant) if variant is not None else html.escape(geometry)
        cards.append(
            f'<div class="arch-card">'
            f'<div class="arch-rank">#{rank}</div>'
            f'<div class="arch-trial">Trial {trial["trial"]}</div>'
            f'<div class="arch-mae">MAE {_fmt(trial.get("best_history_mae"), 3)}</div>'
            f'<div class="arch-desc">{desc}</div>'
            f"{strength_line}"
            f'<table class="arch-params">{"".join(param_rows)}</table>'
            f"</div>"
        )
    return (
        '<p class="hint">Hover parameter names for short explainers. '
        f"{_tip('variant')} = architecture Optuna chose for that trial.</p>"
        f'<div class="arch-grid">{"".join(cards)}</div>'
    )


def _search_space_md(space: list[dict[str, str]]) -> list[str]:
    if not space:
        return ["_Search space not documented._", ""]
    lines = [
        "| Hyperparameter | Search range |",
        "|----------------|--------------|",
    ]
    for item in space:
        lines.append(f"| `{item['param']}` | {item['range']} |")
    lines.append("")
    return lines


def _theory_html() -> str:
    cards = []
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
        '<section id="schema">'
        "<h2>Schema &amp; theory</h2>"
        "<p class=\"hint\">"
        "Start here: how the Hopfield / PC graph is wired, then what each "
        "hyperparameter means. Hover dotted names for one-liners."
        "</p>"
        "<h3>Hopfield architecture variants</h3>"
        '<div class="callout">'
        f"<p><strong>{_tip('variant')}</strong> is Optuna's architecture choice: "
        "each trial samples one wiring from the search space "
        "(baseline / embed-storkey / forecast-storkey / projection). "
        "It is not a continuous knob — it selects which graph to train.</p>"
        "</div>"
        f'<div class="chart">{_svg_hopfield_architecture()}</div>'
        "<h3>Hyperparameter glossary</h3>"
        "<p class=\"hint\">Open a card for the full what / why / effect story.</p>"
        f"{''.join(cards)}"
        "</section>"
    )


def _theory_md() -> list[str]:
    lines = [
        "## Schema & theory",
        "",
        "### Hopfield architecture variants",
        "",
        "All variants share the same PC transformer backbone. Placement of "
        "associative memory differs:",
        "",
        "- **baseline** — no Hopfield node (pure transformer control)",
        "- **embed-storkey** — Storkey Hopfield after embedding (early recall)",
        "- **forecast-storkey** — Storkey Hopfield before the forecast head (late recall)",
        "- **projection** — lightweight linear memory before the forecast head",
        "",
        "PC inference runs at every node: predict `z_mu`, error `z - z_mu`, "
        "inner SGD for `infer_steps` at step size `eta_infer`, outer Adam at `lr`.",
        "",
        "### Hyperparameter glossary",
        "",
        "| Parameter | One-line meaning |",
        "|-----------|------------------|",
    ]
    for key, meta in PARAM_THEORY.items():
        lines.append(f"| `{key}` | {meta['short']} |")
    lines.extend(["", "#### Deeper explanations", ""])
    for key, meta in PARAM_THEORY.items():
        lines.append(f"##### `{key}`")
        lines.append("")
        lines.append(f"- **What it is**: {meta['what']}")
        lines.append(f"- **Why you care**: {meta['why']}")
        lines.append(f"- **How changes show up**: {meta['effect']}")
        lines.append("")
    return lines


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Glucose master Optuna report",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "One page covering every major study category (transformer phases, "
        "epoch Hyperband, Hopfield, confirms).",
        "",
    ]
    lines.extend(_theory_md())

    lines.extend(
        [
            "## Cross-study overview",
            "",
            "| Category | Study | Best MAE | Best trial | Trials |",
            "|----------|-------|---------:|-----------:|-------:|",
        ]
    )
    for row in payload["overview"]:
        lines.append(
            "| {cat} | {label} | {mae} | {trial} | {n} |".format(
                cat=row["category"].split("—")[0].strip()
                if "—" in row["category"]
                else row["category"][:40],
                label=row["label"],
                mae=_fmt(row.get("best_mae"), 4),
                trial=row.get("best_trial") if row.get("best_trial") is not None else "—",
                n=row.get("n_trials") if row.get("n_trials") is not None else "—",
            )
        )
    champ = payload.get("champion")
    if champ:
        lines.extend(
            [
                "",
                f"**Current master best:** {champ['label']} — "
                f"MAE {_fmt(champ.get('best_mae'), 4)} "
                f"(`{champ.get('run_dir')}`)",
                "",
            ]
        )

    for category in payload["categories"]:
        lines.extend(["", f"## {category['title']}", "", category["blurb"], ""])
        for study in category["studies"]:
            lines.append(f"### {study['label']}")
            lines.append("")
            if study.get("missing"):
                lines.append(f"_Missing_: `{study['run_dir']}`")
                lines.append("")
                lines.append("#### Search space (configured)")
                lines.append("")
                lines.extend(_search_space_md(study.get("search_space") or []))
                continue
            counts = study["counts"]
            best = study.get("best") or {}
            lines.append(f"- Study: `{study['study_name']}`")
            lines.append(f"- Run dir: `{study['run_dir']}`")
            lines.append(
                f"- Trials: {counts['n_trials']} "
                f"(complete {counts['n_complete']}, pruned {counts['n_pruned']}, "
                f"fail {counts['n_fail']}, running {counts['n_running']})"
            )
            lines.append(
                f"- Best: trial {best.get('trial')} · "
                f"MAE {_fmt(best.get('best_history_mae'), 4)}"
            )
            lines.append(f"- Best params: `{_param_bits(best.get('params'))}`")
            lines.append("")
            lines.append("#### Search space (what Optuna sampled)")
            lines.append("")
            lines.extend(_search_space_md(study.get("search_space") or []))
            lines.append("| Trial | State | Best MAE | Params |")
            lines.append("|------:|-------|---------:|--------|")
            for trial in study.get("ranked") or []:
                lines.append(
                    "| {t} | {s} | {m} | {p} |".format(
                        t=trial["trial"],
                        s=trial.get("state"),
                        m=_fmt(trial.get("best_history_mae"), 3),
                        p=_param_bits(trial.get("params")),
                    )
                )
            lines.append("")
        for confirm in category.get("confirms") or []:
            lines.append(f"### {confirm['label']}")
            lines.append("")
            if confirm.get("missing"):
                lines.append(f"_Missing_: `{confirm['run_dir']}`")
                lines.append("")
                continue
            cfg = confirm["config"]
            lines.append(f"- Dir: `{confirm['run_dir']}`")
            lines.append(f"- Best val MAE: {_fmt(cfg.get('best_val_mae_mg_dl'), 4)}")
            lines.append(f"- Test MAE: {_fmt(cfg.get('test_mae_mg_dl'), 4)}")
            lines.append(f"- Test MARD %: {_fmt(cfg.get('test_mard_percent'), 2)}")
            lines.append("")
            lines.append("#### Replay hyperparameters")
            lines.append("")
            lines.append("| Hyperparameter | Value |")
            lines.append("|----------------|------:|")
            for key in PARAM_DISPLAY_KEYS + ["epochs", "seed", "batch_size", "horizon"]:
                if key not in cfg:
                    continue
                val = cfg[key]
                display = f"{val:.4g}" if isinstance(val, float) else str(val)
                lines.append(f"| `{key}` | {display} |")
            lines.append("")
    lines.extend(
        [
            "## How to run",
            "",
            "| Task | Command |",
            "|------|---------|",
            "| Install (CPU) | `uv sync --extra glucose` |",
            "| Install (GPU / WSL) | `uv sync --extra glucose --extra cuda12` |",
            "| Check JAX device | `uv run python -c \"import jax; print(jax.devices())\"` |",
            "| Train PC transformer | `uv run glucose-transformer` |",
            "| Epoch Optuna (default) | `uv run glucose-transformer-tune run` |",
            "| Update-budget Optuna (archived) | `uv run glucose-transformer-tune-update-budget run` |",
            "| Hopfield pilot train | `uv run glucose-hopfield` |",
            "| Hopfield Optuna | `uv run glucose-hopfield-tune run` |",
            "| Hopfield Optuna on WSL | `bash scripts/run_hopfield_optuna_wsl.sh` |",
            "| Summarize one study | `uv run python scripts/summarize_glucose_tuning.py` |",
            "| Per-study report | `uv run python scripts/generate_glucose_tuning_report.py --format all` |",
            "| Hopfield study report | `uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all` |",
            "| All studies + master | `uv run python scripts/generate_all_glucose_reports.py --format all` |",
            "| This master report only | `uv run python scripts/generate_glucose_master_report.py` |",
            "",
            "Per-study HTML copies land in `docs/reports/old/`. "
            "The live master report stays at `docs/reports/glucose_master_progress.*`.",
            "",
        ]
    )
    return "\n".join(lines)


def _stage_line_payload(study: dict[str, Any]) -> dict[str, Any]:
    """Top complete (or ranked) trials for within-study learning curves."""
    palette = ["#3dd68c", "#006FEE", "#f5a524", "#9353d3", "#f31260", "#a1a1aa"]
    candidates = [
        t
        for t in (study.get("ranked") or [])
        if t.get("history") and t.get("best_history_mae") is not None
    ]
    complete = [t for t in candidates if t.get("state") == "COMPLETE"]
    chosen = (complete or candidates)[:6]
    datasets = []
    x_title = "Epoch / update"
    for index, trial in enumerate(chosen):
        history = trial.get("history") or []
        if history and "epoch" in history[0]:
            x_key = "epoch"
            x_title = "Epoch"
        else:
            x_key = "update"
            x_title = "Update"
        params = trial.get("params") or {}
        label_bits = [f"T{trial['trial']}"]
        if params.get("variant") is not None:
            label_bits.append(str(params["variant"]))
        if params.get("lr") is not None:
            label_bits.append(f"lr={float(params['lr']):.4g}")
        label_bits.append(f"MAE={float(trial['best_history_mae']):.2f}")
        datasets.append(
            {
                "label": " ".join(label_bits),
                "data": [
                    {
                        "x": int(row.get(x_key, row.get("update", row.get("epoch", 0)))),
                        "y": float(row["mae_mg_dl"]),
                    }
                    for row in history
                    if row.get("mae_mg_dl") is not None
                ],
                "borderColor": palette[index % len(palette)],
                "backgroundColor": palette[index % len(palette)],
                "tension": 0.2,
                "pointRadius": 3,
            }
        )
    return {"datasets": datasets, "xTitle": x_title}


def _study_section_html(study: dict[str, Any], chart_id: str) -> str:
    if study.get("missing"):
        return (
            f'<section class="study missing">'
            f"<h3>{html.escape(study['label'])}</h3>"
            f'<p class="hint">Not found: <code>{html.escape(study["run_dir"])}</code></p>'
            f"<h4>Search space (configured)</h4>"
            f"{_search_space_html(study.get('search_space') or [])}"
            f"</section>"
        )
    counts = study["counts"]
    best = study.get("best") or {}
    params = best.get("params") or {}
    leaderboard_rows = []
    for trial in (study.get("ranked") or [])[:20]:
        p = trial.get("params") or {}
        geometry = f"{p.get('seq_len')}/d{p.get('depth')}/h{p.get('num_heads')}"
        extra = ""
        if p.get("variant") is not None:
            extra = f" · {p.get('variant')}/{p.get('hopfield_strength')}"
        leaderboard_rows.append(
            "<tr>"
            f"<td>{trial['trial']}</td>"
            f"<td>{html.escape(str(trial.get('state')))}</td>"
            f"<td>{_fmt(trial.get('best_history_mae'), 3)}</td>"
            f"<td>{html.escape(geometry)}{html.escape(extra)}</td>"
            f"<td>{_fmt(float(p['lr']) if p.get('lr') is not None else None, 4)}</td>"
            f"<td>{_fmt(float(p['eta_infer']) if p.get('eta_infer') is not None else None, 3)}</td>"
            "</tr>"
        )

    trial_cards = []
    for trial in study.get("trials") or []:
        p = trial.get("params") or {}
        attrs = trial.get("user_attrs") or {}
        reason = (
            attrs.get("stop_reason")
            or attrs.get("prune_reason")
            or attrs.get("failure_reason")
            or "—"
        )
        state = str(trial.get("state") or "?")
        badge = {
            "COMPLETE": "ok",
            "PRUNED": "warn",
            "FAIL": "bad",
            "RUNNING": "run",
        }.get(state, "")
        chips = []
        for key in PARAM_DISPLAY_KEYS:
            val = p.get(key)
            if val is None:
                continue
            display = f"{val:.4g}" if isinstance(val, float) else html.escape(str(val))
            chips.append(f"<tr><td>{_tip(key)}</td><td><strong>{display}</strong></td></tr>")
        history = trial.get("history") or []
        x_key = "epoch" if history and "epoch" in history[0] else "update"
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
        table = ""
        if stage_rows:
            table = (
                f"<table><thead><tr><th>{html.escape(x_key)}</th>"
                f"<th>Val MAE</th><th>MARD%</th></tr></thead>"
                f"<tbody>{''.join(stage_rows)}</tbody></table>"
            )
        params_block = (
            f'<table class="arch-params">{"".join(chips)}</table>'
            if chips
            else "<p>—</p>"
        )
        trial_cards.append(
            f'<details class="trial-card">'
            f'<summary><span class="badge {badge}">{html.escape(state)}</span> '
            f"Trial {trial['trial']} · MAE "
            f"<strong>{_fmt(trial.get('best_history_mae'), 3)}</strong></summary>"
            f"<p>Reason: <em>{html.escape(str(reason))}</em></p>"
            f"{params_block}"
            f"{table}</details>"
        )

    return f"""
    <section class="study" id="{html.escape(chart_id)}">
      <h3>{html.escape(study['label'])}</h3>
      <p class="hint">
        Study <code>{html.escape(study['study_name'])}</code> ·
        <code>{html.escape(study['run_dir'])}</code>
      </p>
      <div class="stats">
        <div class="stat"><strong>{counts['n_trials']}</strong><span>Trials</span></div>
        <div class="stat"><strong>{counts['n_complete']}</strong><span>Complete</span></div>
        <div class="stat"><strong>{counts['n_pruned']}</strong><span>Pruned</span></div>
        <div class="stat"><strong>{_fmt(best.get('best_history_mae'), 2)}</strong><span>Best MAE</span></div>
      </div>
      <p>
        Best trial <strong>{best.get('trial', '—')}</strong> ·
        params <code>{html.escape(_param_bits(params))}</code>
      </p>
      <h4>Search space (what Optuna sampled)</h4>
      {_search_space_html(study.get('search_space') or [])}
      <h4>Top model architectures</h4>
      {_top_arch_cards_html(study)}
      <h4>All trials — best val MAE</h4>
      <p class="hint">Dashed green line = best MAE in this study. Hover bars for exact values.</p>
      <div class="chart"><canvas id="{html.escape(chart_id)}_bar" height="110"></canvas></div>
      <h4>Top-trial learning curves</h4>
      <p class="hint">Hover points; click legend to toggle series. Axes: training progress vs val MAE.</p>
      <div class="chart"><canvas id="{html.escape(chart_id)}_line" height="110"></canvas></div>
      <details class="fold-block">
        <summary>Leaderboard (top 20)</summary>
        <table>
          <thead>
            <tr>
              <th>Trial</th><th>State</th><th>Best MAE</th><th>Geometry</th>
              <th>{_tip('lr','LR')}</th><th>{_tip('eta_infer','η_infer')}</th>
            </tr>
          </thead>
          <tbody>{''.join(leaderboard_rows) or '<tr><td colspan="6">No trials</td></tr>'}</tbody>
        </table>
      </details>
      <details class="fold-block">
        <summary>All trials (expand for full hyperparameter list)</summary>
        {''.join(trial_cards) or '<p class="hint">No trial details.</p>'}
      </details>
    </section>
    """


def _confirm_section_html(confirm: dict[str, Any], chart_id: str) -> str:
    if confirm.get("missing"):
        return (
            f'<section class="study missing">'
            f"<h3>{html.escape(confirm['label'])}</h3>"
            f'<p class="hint">Not found: <code>{html.escape(confirm["run_dir"])}</code></p>'
            f"</section>"
        )
    cfg = confirm["config"]
    rows = []
    for row in confirm.get("epochs") or []:
        rows.append(
            "<tr>"
            f"<td>{int(row['epoch'])}</td>"
            f"<td>{_fmt(row.get('mae_mg_dl'), 3)}</td>"
            f"<td>{_fmt(row.get('rmse_mg_dl'), 3)}</td>"
            f"<td>{_fmt(row.get('mard_percent'), 2)}</td>"
            "</tr>"
        )
    param_rows = []
    for key in PARAM_DISPLAY_KEYS + ["epochs", "seed", "batch_size", "horizon", "embed_dim", "mlp_dim"]:
        if key not in cfg:
            continue
        val = cfg[key]
        display = f"{val:.4g}" if isinstance(val, float) else html.escape(str(val))
        param_rows.append(f"<tr><td>{_tip(key)}</td><td>{display}</td></tr>")
    return f"""
    <section class="study" id="{html.escape(chart_id)}">
      <h3>{html.escape(confirm['label'])}</h3>
      <p class="hint"><code>{html.escape(confirm['run_dir'])}</code></p>
      <div class="stats">
        <div class="stat"><strong>{_fmt(cfg.get('best_val_mae_mg_dl'), 2)}</strong><span>Best val MAE</span></div>
        <div class="stat"><strong>{_fmt(cfg.get('test_mae_mg_dl'), 2)}</strong><span>Test MAE</span></div>
        <div class="stat"><strong>{_fmt(cfg.get('test_mard_percent'), 1)}</strong><span>Test MARD %</span></div>
        <div class="stat"><strong>{html.escape(str(cfg.get('final_epoch', '—')))}</strong><span>Epochs</span></div>
      </div>
      <h4>Replay hyperparameters</h4>
      <div class="arch-card search-space-card">
        <table class="arch-params">{''.join(param_rows) or '<tr><td colspan="2">—</td></tr>'}</table>
      </div>
      <h4>Validation learning curve</h4>
      <p class="hint">Dashed green line = best val MAE. X = epoch, Y = MAE (mg/dL).</p>
      <div class="chart"><canvas id="{html.escape(chart_id)}_line" height="110"></canvas></div>
      <table>
        <thead><tr><th>Epoch</th><th>Val MAE</th><th>RMSE</th><th>MARD%</th></tr></thead>
        <tbody>{''.join(rows) or '<tr><td colspan="4">No epoch history</td></tr>'}</tbody>
      </table>
    </section>
    """


def _render_html(payload: dict[str, Any]) -> str:
    overview_rows = []
    for row in payload["overview"]:
        overview_rows.append(
            "<tr>"
            f"<td>{html.escape(row['category'])}</td>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{html.escape(row['family'])}</td>"
            f"<td>{_fmt(row.get('best_mae'), 4)}</td>"
            f"<td>{row.get('best_trial') if row.get('best_trial') is not None else '—'}</td>"
            f"<td>{row.get('n_trials') if row.get('n_trials') is not None else '—'}</td>"
            f"<td><code>{html.escape(row['run_dir'])}</code></td>"
            "</tr>"
        )

    champ = payload.get("champion") or {}
    champ_html = "—"
    if champ:
        champ_html = (
            f"<strong>{html.escape(champ['label'])}</strong> · "
            f"MAE {_fmt(champ.get('best_mae'), 4)} · "
            f"<code>{html.escape(str(champ.get('run_dir')))}</code>"
        )

    category_blocks: list[str] = []
    charts: dict[str, Any] = {"overview": [], "studies": {}, "confirms": {}}
    for row in payload["overview"]:
        if row.get("best_mae") is None:
            continue
        charts["overview"].append(
            {
                "label": row["label"],
                "mae": float(row["best_mae"]),
                "family": row["family"],
            }
        )

    for category in payload["categories"]:
        toc_links = []
        body_parts = []
        for index, study in enumerate(category["studies"]):
            chart_id = f"{category['id']}_s{index}"
            toc_links.append(
                f'<a href="#{html.escape(chart_id)}">{html.escape(study["label"])}</a>'
            )
            body_parts.append(_study_section_html(study, chart_id))
            if not study.get("missing"):
                best = study.get("best") or {}
                best_mae = best.get("best_history_mae")
                charts["studies"][chart_id] = {
                    "labels": [str(t["trial"]) for t in study.get("trials") or []],
                    "values": [
                        float(t["best_history_mae"])
                        if t.get("best_history_mae") is not None
                        else None
                        for t in study.get("trials") or []
                    ],
                    "colors": [
                        "#3dd68c"
                        if t.get("state") == "COMPLETE"
                        else (
                            "#f5a524"
                            if t.get("state") == "PRUNED"
                            else (
                                "#f31260"
                                if t.get("state") == "FAIL"
                                else "#4b5563"
                            )
                        )
                        for t in study.get("trials") or []
                    ],
                    "bestMae": float(best_mae) if best_mae is not None else None,
                    "lines": _stage_line_payload(study),
                }
        for index, confirm in enumerate(category.get("confirms") or []):
            chart_id = f"{category['id']}_c{index}"
            toc_links.append(
                f'<a href="#{html.escape(chart_id)}">{html.escape(confirm["label"])}</a>'
            )
            body_parts.append(_confirm_section_html(confirm, chart_id))
            if not confirm.get("missing"):
                values = [float(r["mae_mg_dl"]) for r in confirm.get("epochs") or []]
                charts["confirms"][chart_id] = {
                    "labels": [int(r["epoch"]) for r in confirm.get("epochs") or []],
                    "values": values,
                    "bestMae": min(values) if values else None,
                }
        category_blocks.append(
            f'<section class="category" id="{html.escape(category["id"])}">'
            f"<h2>{html.escape(category['title'])}</h2>"
            f"<p>{html.escape(category['blurb'])}</p>"
            f'<p class="toc">{" · ".join(toc_links)}</p>'
            f"{''.join(body_parts)}"
            f"</section>"
        )

    overview_best = (
        min((r["mae"] for r in charts["overview"]), default=None)
        if charts["overview"]
        else None
    )
    charts["overviewBestMae"] = overview_best
    charts_json = json.dumps(charts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Glucose master Optuna report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg:#0b0f14; --card:#121820; --text:#e7ecf3;
      --muted:#9aa7b8; --line:#243041; --accent:#3dd68c;
    }}
    body {{ margin:0; font-family:"Segoe UI",system-ui,sans-serif;
      background:var(--bg); color:var(--text); line-height:1.45; }}
    main {{ max-width:1100px; margin:0 auto; padding:32px 20px 96px; }}
    h1,h2,h3,h4 {{ margin:0 0 12px; font-weight:650; }}
    h1 {{ font-size:1.9rem; }} h2 {{ font-size:1.35rem; margin-top:36px; }}
    h3 {{ font-size:1.1rem; margin-top:24px; }} h4 {{ font-size:0.95rem; margin-top:16px; }}
    p,li {{ color:var(--muted); }}
    .hint {{ font-size:0.85rem; color:var(--muted); }}
    .toc a {{ color:var(--accent); margin-right:10px; font-size:0.9rem; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
      gap:10px; margin:14px 0; }}
    .stat {{ background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:12px 14px; }}
    .stat strong {{ display:block; font-size:1.25rem; color:var(--text); }}
    .stat span {{ color:var(--muted); font-size:0.8rem; }}
    .chart {{ margin:12px 0; border:1px solid var(--line); border-radius:12px;
      background:var(--card); padding:14px; }}
    .study {{ background:rgba(18,24,32,0.55); border:1px solid var(--line);
      border-radius:14px; padding:16px 18px; margin:18px 0; }}
    .study.missing {{ opacity:0.7; }}
    table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
    th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; }}
    th {{ color:var(--muted); font-weight:600; }}
    code {{ color:var(--accent); }}
    .tip {{ position:relative; cursor:help; border-bottom:1px dotted var(--accent); }}
    .tip:hover::after {{
      content:attr(data-tip); position:absolute; left:0; bottom:calc(100% + 8px);
      z-index:40; min-width:200px; max-width:300px; padding:8px 10px;
      background:#1a2332; color:var(--text); border:1px solid var(--line);
      border-radius:8px; font-size:0.78rem; white-space:normal;
    }}
    .trial-card, .theory-card, .fold-block {{ background:var(--card); border:1px solid var(--line);
      border-radius:10px; margin:8px 0; padding:0 12px; }}
    .trial-card > summary, .theory-card > summary, .fold-block > summary {{
      cursor:pointer; list-style:none; padding:12px 0;
      color:var(--text); font-weight:600; }}
    .trial-card > summary::-webkit-details-marker,
    .theory-card > summary::-webkit-details-marker,
    .fold-block > summary::-webkit-details-marker {{ display:none; }}
    .fold-block > summary::before {{
      content:"▸ "; color:var(--accent); }}
    .fold-block[open] > summary::before {{ content:"▾ "; }}
    .fold-block > summary {{ font-size:0.95rem; }}
    .arch-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
      gap:12px; margin:12px 0; }}
    .arch-card {{ background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:14px; position:relative; }}
    .arch-rank {{ position:absolute; top:10px; right:14px; font-size:1.5rem;
      font-weight:700; color:var(--line); }}
    .arch-trial {{ font-size:0.82rem; font-weight:600; color:var(--muted); }}
    .arch-mae {{ font-size:1.15rem; font-weight:650; color:var(--accent); margin:4px 0 6px; }}
    .arch-desc {{ font-size:0.92rem; color:var(--text); margin-bottom:4px; }}
    .arch-strength {{ font-size:0.82rem; color:var(--muted); margin-bottom:8px; }}
    .arch-params {{ width:100%; font-size:0.82rem; }}
    .arch-params td {{ padding:2px 6px; border-bottom:1px solid var(--line); }}
    .search-space-card {{ max-width:520px; }}
    .param-list {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
      gap:4px 10px; padding-left:18px; }}
    .badge {{ display:inline-block; font-size:0.72rem; padding:2px 8px;
      border-radius:999px; border:1px solid var(--line); color:var(--muted); }}
    .badge.ok {{ color:#3dd68c; border-color:#3dd68c55; }}
    .badge.warn {{ color:#f5a524; border-color:#f5a52455; }}
    .badge.bad {{ color:#f31260; border-color:#f3126055; }}
    .badge.run {{ color:#006FEE; border-color:#006FEE55; }}
    .callout {{ background:var(--card); border:1px solid var(--line);
      border-radius:12px; padding:14px 16px; margin:16px 0; }}
    @media (max-width:800px) {{ .stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  </style>
</head>
<body>
  <main>
    <h1>Glucose master Optuna report</h1>
    <p>
      Single page for every major experiment category.
      Generated <code>{html.escape(payload['generated_at'])}</code>.
      Open in Chrome/Edge/Firefox for interactive charts.
    </p>
    <div class="callout">
      <p><strong>Master best right now:</strong> {champ_html}</p>
      <p class="hint">
        Order: schema/theory (Hopfield diagram + glossary) → overview →
        update-budget phases → epoch Hyperband → Hopfield → confirms.
        Every study lists the search space it ran on. Charts are interactive
        with labelled axes and a dashed best-MAE line.
      </p>
    </div>
    <p class="toc">
      <a href="#schema">Schema &amp; theory</a>
      <a href="#overview">Overview</a>
      <a href="#transformer_update_budget">Update-budget</a>
      <a href="#transformer_epochs">Epoch Hyperband</a>
      <a href="#hopfield">Hopfield</a>
      <a href="#confirms">Confirms</a>
      <a href="#how-to-run">How to run</a>
    </p>

    {_theory_html()}

    <section id="overview">
      <h2>Cross-study overview</h2>
      <p class="hint">
        X-axis = study · Y-axis = best validation MAE (mg/dL) ·
        dashed green = master best MAE among plotted studies.
      </p>
      <div class="chart"><canvas id="overviewChart" height="120"></canvas></div>
      <table>
        <thead>
          <tr>
            <th>Category</th><th>Study</th><th>Family</th>
            <th>Best MAE</th><th>Best trial</th><th>Trials</th><th>Path</th>
          </tr>
        </thead>
        <tbody>{''.join(overview_rows)}</tbody>
      </table>
    </section>

    {''.join(category_blocks)}

    <section id="how-to-run">
      <h2>How to run</h2>
      <p class="hint">
        Common commands from the repo root. Prefer WSL2 + CUDA for Optuna.
        Per-study report copies go under <code>docs/reports/old/</code>;
        this master page is the live summary.
      </p>
      <table>
        <thead><tr><th>Task</th><th>Command</th></tr></thead>
        <tbody>
          <tr><td style="color:var(--muted);">Install (CPU)</td>
            <td><code>uv sync --extra glucose</code></td></tr>
          <tr><td style="color:var(--muted);">Install (GPU / WSL)</td>
            <td><code>uv sync --extra glucose --extra cuda12</code></td></tr>
          <tr><td style="color:var(--muted);">Check JAX device</td>
            <td><code>uv run python -c "import jax; print(jax.devices())"</code></td></tr>
          <tr><td style="color:var(--muted);">Train PC transformer</td>
            <td><code>uv run glucose-transformer</code></td></tr>
          <tr><td style="color:var(--muted);">Epoch Optuna (default)</td>
            <td><code>uv run glucose-transformer-tune run</code></td></tr>
          <tr><td style="color:var(--muted);">Update-budget Optuna (archived)</td>
            <td><code>uv run glucose-transformer-tune-update-budget run</code></td></tr>
          <tr><td style="color:var(--muted);">Hopfield pilot train</td>
            <td><code>uv run glucose-hopfield</code></td></tr>
          <tr><td style="color:var(--muted);">Hopfield Optuna</td>
            <td><code>uv run glucose-hopfield-tune run</code></td></tr>
          <tr><td style="color:var(--muted);">Hopfield Optuna on WSL</td>
            <td><code>bash scripts/run_hopfield_optuna_wsl.sh</code></td></tr>
          <tr><td style="color:var(--muted);">Summarize one study</td>
            <td><code>uv run python scripts/summarize_glucose_tuning.py</code></td></tr>
          <tr><td style="color:var(--muted);">Per-study transformer report</td>
            <td><code>uv run python scripts/generate_glucose_tuning_report.py --format all</code></td></tr>
          <tr><td style="color:var(--muted);">Per-study Hopfield report</td>
            <td><code>uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all</code></td></tr>
          <tr><td style="color:var(--muted);">All studies + this master</td>
            <td><code>uv run python scripts/generate_all_glucose_reports.py --format all</code></td></tr>
          <tr><td style="color:var(--muted);">This master report only</td>
            <td><code>uv run python scripts/generate_glucose_master_report.py</code></td></tr>
        </tbody>
      </table>
    </section>
  </main>
  <script>
    const DATA = {charts_json};
    const tick = "#9aa7b8";
    const grid = "#243041";
    const familyColor = {{
      transformer: "#3dd68c",
      hopfield: "#f5a524",
      confirm: "#006FEE",
    }};

    function bestLineAnnotation(bestMae, labelPrefix) {{
      if (bestMae == null) return {{}};
      return {{
        annotations: {{
          bestLine: {{
            type: "line",
            yMin: bestMae,
            yMax: bestMae,
            borderColor: "#3dd68c",
            borderWidth: 2,
            borderDash: [6, 4],
            label: {{
              display: true,
              content: `${{labelPrefix}} ${{bestMae.toFixed(2)}}`,
              position: "end",
              backgroundColor: "#121820",
              color: "#3dd68c",
            }},
          }},
        }},
      }};
    }}

    const commonPlugins = {{
      legend: {{ labels: {{ color: tick, usePointStyle: true, boxWidth: 12 }} }},
      tooltip: {{
        callbacks: {{
          label: (ctx) => {{
            const y = ctx.parsed.y;
            if (y == null) return `${{ctx.dataset.label}}: n/a`;
            return `${{ctx.dataset.label}}: ${{y.toFixed(3)}} mg/dL`;
          }},
        }},
      }},
    }};

    new Chart(document.getElementById("overviewChart"), {{
      type: "bar",
      data: {{
        labels: DATA.overview.map((r) => r.label),
        datasets: [{{
          label: "Best MAE (mg/dL)",
          data: DATA.overview.map((r) => r.mae),
          backgroundColor: DATA.overview.map((r) => familyColor[r.family] || "#a1a1aa"),
          borderRadius: 4,
        }}],
      }},
      options: {{
        responsive: true,
        interaction: {{ mode: "nearest", intersect: false }},
        plugins: {{
          ...commonPlugins,
          annotation: bestLineAnnotation(DATA.overviewBestMae, "Best"),
        }},
        scales: {{
          x: {{
            ticks: {{ color: tick, maxRotation: 45, minRotation: 20 }},
            grid: {{ color: grid }},
            title: {{ display: true, text: "Study", color: tick }},
          }},
          y: {{
            ticks: {{ color: tick }},
            grid: {{ color: grid }},
            title: {{ display: true, text: "Best validation MAE (mg/dL)", color: tick }},
          }},
        }},
      }},
    }});

    Object.entries(DATA.studies || {{}}).forEach(([id, payload]) => {{
      const barEl = document.getElementById(id + "_bar");
      if (barEl) {{
        new Chart(barEl, {{
          type: "bar",
          data: {{
            labels: payload.labels,
            datasets: [{{
              label: "Best val MAE (mg/dL)",
              data: payload.values,
              backgroundColor: payload.colors,
              borderRadius: 3,
            }}],
          }},
          options: {{
            responsive: true,
            interaction: {{ mode: "nearest", intersect: false }},
            plugins: {{
              legend: {{ display: false }},
              tooltip: commonPlugins.tooltip,
              annotation: bestLineAnnotation(payload.bestMae, "Best"),
            }},
            scales: {{
              x: {{
                ticks: {{ color: tick }},
                grid: {{ color: grid }},
                title: {{ display: true, text: "Trial number", color: tick }},
              }},
              y: {{
                ticks: {{ color: tick }},
                grid: {{ color: grid }},
                title: {{ display: true, text: "Best validation MAE (mg/dL)", color: tick }},
              }},
            }},
          }},
        }});
      }}

      const lineEl = document.getElementById(id + "_line");
      const lines = payload.lines || {{}};
      if (lineEl && lines.datasets && lines.datasets.length) {{
        new Chart(lineEl, {{
          type: "line",
          data: {{ datasets: lines.datasets }},
          options: {{
            responsive: true,
            interaction: {{ mode: "nearest", intersect: false }},
            plugins: {{
              ...commonPlugins,
              annotation: bestLineAnnotation(payload.bestMae, "Best"),
            }},
            scales: {{
              x: {{
                type: "linear",
                ticks: {{ color: tick }},
                grid: {{ color: grid }},
                title: {{
                  display: true,
                  text: lines.xTitle || "Training progress",
                  color: tick,
                }},
              }},
              y: {{
                ticks: {{ color: tick }},
                grid: {{ color: grid }},
                title: {{ display: true, text: "Validation MAE (mg/dL)", color: tick }},
              }},
            }},
          }},
        }});
      }}
    }});

    Object.entries(DATA.confirms || {{}}).forEach(([id, payload]) => {{
      const el = document.getElementById(id + "_line");
      if (!el || !payload.values.length) return;
      new Chart(el, {{
        type: "line",
        data: {{
          labels: payload.labels,
          datasets: [{{
            label: "Val MAE (mg/dL)",
            data: payload.values,
            borderColor: "#3dd68c",
            backgroundColor: "rgba(61,214,140,0.15)",
            fill: true,
            tension: 0.25,
            pointRadius: 3,
          }}],
        }},
        options: {{
          responsive: true,
          interaction: {{ mode: "nearest", intersect: false }},
          plugins: {{
            ...commonPlugins,
            annotation: bestLineAnnotation(payload.bestMae, "Best"),
          }},
          scales: {{
            x: {{
              ticks: {{ color: tick }},
              grid: {{ color: grid }},
              title: {{ display: true, text: "Epoch", color: tick }},
            }},
            y: {{
              ticks: {{ color: tick }},
              grid: {{ color: grid }},
              title: {{ display: true, text: "Validation MAE (mg/dL)", color: tick }},
            }},
          }},
        }},
      }});
    }});
  </script>
</body>
</html>
"""


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    format: str = typer.Option("all", "--format", help="md, html, json, or all."),
) -> None:
    """Write the master cross-study glucose report."""
    if ctx.invoked_subcommand is not None:
        return

    payload = _build_master_payload()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    wanted = {format.lower()} if format.lower() != "all" else {"md", "html", "json"}

    if "json" in wanted:
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"wrote {OUT_JSON}")
    if "md" in wanted:
        OUT_MD.write_text(_render_markdown(payload), encoding="utf-8")
        typer.echo(f"wrote {OUT_MD}")
    if "html" in wanted:
        OUT_HTML.write_text(_render_html(payload), encoding="utf-8")
        typer.echo(f"wrote {OUT_HTML}")

    champ = payload.get("champion")
    if champ:
        typer.echo(
            f"master_best label={champ['label']} mae={champ.get('best_mae')}"
        )


if __name__ == "__main__":
    app()
