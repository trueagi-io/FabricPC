"""The table view of a run, and a check that a results folder is whole.

``trials.csv`` puts every trial of a row on one line, so the numbers open
straight in a spreadsheet. The JSON files stay the source of truth; the CSV
is rebuilt from them.

``validate`` reads a results folder the way someone checking our numbers
would, and lists everything that does not add up: a missing manifest, a
file without a schema version, a trial filed under the wrong row, a seed
the manifest never mentions, a summary or CSV that disagrees with the
trials. An empty list means the folder is fine. A failed trial is not a
problem; it is a recorded outcome.
"""

import csv
import json
from pathlib import Path
from typing import List

from fabricpc.bench.results.manifest import SCHEMA_VERSION

# Columns in the order people read them; metric columns go after "status".
_LEAD = ["trial", "seed", "status", "num_epochs"]
_TAIL = [
    "step_time_ms",
    "compile_time_s",
    "train_time_s",
    "achieved_tflops",
    "memory_bytes",
    "peak_memory_bytes",
    "step_memory_bytes",
    "epc_band",
    "epc_f_weighted",
    "eta_final",
    "rate_crossings",
    "stiffness_init",
    "stiffness_final",
    "bp_cos_min",
    "bp_cos_mean",
    "bp_cos_weakest",
    "n_params",
    "checkpoint",
    "error",
]


def _trial_files(row_dir: Path) -> List[Path]:
    return sorted(row_dir.glob("trial*.json"), key=lambda p: int(p.stem[5:]))


def _last_line(text) -> str:
    lines = (text or "").strip().splitlines()
    return lines[-1] if lines else ""


def write_trials_csv(results_dir, row_id: str) -> Path:
    """Write ``<results_dir>/<row id>/trials.csv`` from the trial files."""
    row_dir = Path(results_dir) / row_id
    trials = [json.loads(p.read_text()) for p in _trial_files(row_dir)]
    metric_names = sorted({name for t in trials for name in t.get("metrics", {})})

    path = row_dir / "trials.csv"
    with open(path, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=_LEAD + metric_names + _TAIL)
        out.writeheader()
        for t in trials:
            line = {key: t.get(key) for key in _LEAD + _TAIL}
            line.update(t.get("metrics", {}))
            line["step_memory_bytes"] = (t.get("step_memory") or {}).get("total_bytes")
            final = (t.get("epc_regime") or {}).get("final") or {}
            line["epc_band"] = final.get("band")
            line["epc_f_weighted"] = final.get("f_weighted")
            rate = (t.get("rate_control") or {}).get("summary") or {}
            line["eta_final"] = rate.get("eta_final")
            line["rate_crossings"] = rate.get("crossings")
            diag = t.get("diagnostics") or {}
            for when in ("init", "final"):
                stiff = (diag.get(when) or {}).get("stiffness") or {}
                line[f"stiffness_{when}"] = stiff.get("lambda_max")
            align = (diag.get("final") or {}).get("backprop_alignment") or {}
            line["bp_cos_min"] = align.get("min_cos")
            line["bp_cos_mean"] = align.get("mean_cos")
            line["bp_cos_weakest"] = align.get("weakest")
            # The last line of a traceback says what went wrong.
            line["error"] = _last_line(t.get("error"))
            out.writerow(line)
    return path


def _load(path: Path, problems: List[str]):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        problems.append(f"{path}: cannot read it ({e})")
        return None


def _check_version(path: Path, data: dict, problems: List[str]) -> None:
    version = data.get("schema_version")
    if version is None:
        problems.append(f"{path}: no schema_version")
    elif version != SCHEMA_VERSION:
        problems.append(
            f"{path}: schema_version {version}, this code writes {SCHEMA_VERSION}"
        )


def _validate_row(row_dir: Path, problems: List[str]) -> None:
    row_id = row_dir.name

    manifest_path = row_dir / "manifest.json"
    seeds = None
    if not manifest_path.exists():
        problems.append(f"{manifest_path}: missing")
    else:
        manifest = _load(manifest_path, problems)
        if manifest is not None:
            _check_version(manifest_path, manifest, problems)
            seeds = manifest.get("seeds")
            if manifest.get("row", {}).get("id") != row_id:
                problems.append(f"{manifest_path}: row id does not match {row_id}")

    files = _trial_files(row_dir)
    n_ok = 0
    for path in files:
        trial = _load(path, problems)
        if trial is None:
            continue
        _check_version(path, trial, problems)
        if trial.get("row_id") != row_id:
            problems.append(f"{path}: row_id is {trial.get('row_id')}, not {row_id}")
        if trial.get("trial") != int(path.stem[5:]):
            problems.append(f"{path}: trial number does not match the file name")
        if seeds is not None and trial.get("seed") not in seeds:
            problems.append(f"{path}: seed {trial.get('seed')} is not in the manifest")
        status = trial.get("status")
        if status == "ok":
            n_ok += 1
            if not trial.get("metrics"):
                problems.append(f"{path}: status ok but no metrics")
        elif status != "failed":
            problems.append(f"{path}: status is {status!r}, not 'ok' or 'failed'")

    summary_path = row_dir / "summary.json"
    if summary_path.exists():
        summary = _load(summary_path, problems)
        if summary is not None:
            _check_version(summary_path, summary, problems)
            if summary.get("n_ok") != n_ok:
                problems.append(
                    f"{summary_path}: n_ok is {summary.get('n_ok')}, but {n_ok} "
                    "trial files say ok"
                )

    csv_path = row_dir / "trials.csv"
    if not csv_path.exists():
        problems.append(f"{csv_path}: missing")
    else:
        with open(csv_path, newline="") as f:
            n_lines = len(list(csv.DictReader(f)))
        if n_lines != len(files):
            problems.append(
                f"{csv_path}: {n_lines} lines, but there are {len(files)} trial files"
            )


def validate(results_dir) -> List[str]:
    """Everything wrong with a results folder; an empty list means it is fine."""
    root = Path(results_dir)
    if not root.is_dir():
        return [f"{root}: not a folder"]
    row_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and any(d.glob("trial*.json"))
    )
    if not row_dirs:
        return [f"{root}: no benchmark results (no trial files) found"]
    problems: List[str] = []
    for row_dir in row_dirs:
        _validate_row(row_dir, problems)
    return problems
