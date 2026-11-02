"""Reading a run-queue plan file, and guessing how long it will take.

The queue itself is in ``fabricpc.bench.queue``; this file only turns a
plan's JSON into ``Job``s (with clear errors for a broken plan) and turns
probe files into hour estimates. Nothing here trains anything.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PLANS_DIR = Path(__file__).parent / "plans"

_PLAN_KEYS = {"name", "description", "defaults", "jobs"}
_JOB_KEYS = {"id", "trials", "epochs", "zoo", "note"}


class PlanError(ValueError):
    """The plan file cannot be used as it is. The message says why."""


@dataclass(frozen=True)
class Job:
    """One line of a plan: a row or a family, and how to run it.

    ``trials`` and ``epochs`` of None mean "the row's own".
    """

    id: str
    trials: Optional[int] = None
    epochs: Optional[float] = None
    zoo: bool = False
    note: str = ""


@dataclass(frozen=True)
class Plan:
    name: str
    jobs: Tuple[Job, ...]


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_fields(where: str, fields: dict) -> None:
    """Refuse bad values, and keys we do not know (a typo like "epoch"
    would otherwise be silently ignored and the job run with the default)."""
    extra = set(fields) - _JOB_KEYS
    if extra:
        raise PlanError(
            f"{where}: unknown key(s) {sorted(extra)}; allowed: {sorted(_JOB_KEYS)}"
        )
    trials = fields.get("trials")
    if trials is not None and not (
        isinstance(trials, int) and not isinstance(trials, bool) and trials >= 1
    ):
        raise PlanError(
            f"{where}: 'trials' must be a whole number of at least 1, got {trials!r}"
        )
    epochs = fields.get("epochs")
    if epochs is not None and not (_is_number(epochs) and epochs > 0):
        raise PlanError(f"{where}: 'epochs' must be a number above 0, got {epochs!r}")
    if "zoo" in fields and not isinstance(fields["zoo"], bool):
        raise PlanError(f"{where}: 'zoo' must be true or false, got {fields['zoo']!r}")
    if "note" in fields and not isinstance(fields["note"], str):
        raise PlanError(f"{where}: 'note' must be text")


def parse_plan(data, name: str = "plan") -> Plan:
    """Turn a plan's JSON (already loaded) into a ``Plan``."""
    if not isinstance(data, dict):
        raise PlanError("a plan must be a JSON object with a 'jobs' list")
    extra = set(data) - _PLAN_KEYS
    if extra:
        raise PlanError(
            f"unknown plan key(s) {sorted(extra)}; allowed: {sorted(_PLAN_KEYS)}"
        )
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict) or "id" in defaults:
        raise PlanError("'defaults' must be an object without an 'id'")
    _check_fields("defaults", defaults)
    raw_jobs = data.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise PlanError("the plan has no jobs (it needs a non-empty 'jobs' list)")
    jobs, seen = [], set()
    for n, raw in enumerate(raw_jobs, 1):
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("id"), str)
            or not raw["id"]
        ):
            raise PlanError(f"job {n}: every job needs an 'id' (a row or family)")
        where = f"job {n} ({raw['id']})"
        _check_fields(where, raw)
        if raw["id"] in seen:
            raise PlanError(f"{where}: this id is in the plan twice")
        seen.add(raw["id"])
        fields = {**defaults, **raw}
        epochs = fields.get("epochs")
        jobs.append(
            Job(
                id=fields["id"],
                trials=fields.get("trials"),
                epochs=float(epochs) if epochs is not None else None,
                zoo=fields.get("zoo", False),
                note=fields.get("note", ""),
            )
        )
    return Plan(name=str(data.get("name", name)), jobs=tuple(jobs))


def load_plan(path) -> Plan:
    """Read a plan file. Raises PlanError with a readable message."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise PlanError(f"cannot read {path}: {e}") from e
    try:
        data = json.loads(text)
    except ValueError as e:
        raise PlanError(f"{path} is not valid JSON: {e}") from e
    return parse_plan(data, name=path.stem)


def job_rows(job_id: str) -> Optional[Tuple[str, ...]]:
    """The row ids a job runs, or None if the id is not in the registry."""
    from fabricpc.bench.registry import COMPARISONS, ROWS

    if job_id in ROWS:
        return (job_id,)
    if job_id in COMPARISONS:
        return tuple(COMPARISONS[job_id].rows)
    return None


def unknown_ids(plan: Plan) -> List[str]:
    """Job ids that are neither a row nor a family, in plan order."""
    return [job.id for job in plan.jobs if job_rows(job.id) is None]


# --- estimates --------------------------------------------------------------


def load_probes(paths: Iterable[Path]) -> Dict[str, dict]:
    """Every row found in ``probe-*.json`` files (a file, or a folder
    searched all the way down), keyed by row id. A later file wins."""
    probes: Dict[str, dict] = {}
    for path in paths:
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.rglob("probe-*.json"))
        for f in files:
            try:
                report = json.loads(f.read_text())
                device = report.get("device", {})
                for row in report["rows"]:
                    probes[row["row_id"]] = {**row, "device": device.get("kind", "?")}
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue  # not a probe file we understand; ignore it
    return probes


def estimate_hours(job: Job, probes: Dict[str, dict]) -> Optional[float]:
    """Hours for a job from probe numbers, or None if a row has no probe.

    Training time scales with the epoch count; evaluation is taken as is.
    """
    from fabricpc.bench.registry import ROWS

    rows = job_rows(job.id)
    if not rows or any(r not in probes for r in rows):
        return None
    seconds = 0.0
    for row_id in rows:
        p = probes[row_id]
        row = ROWS[row_id]
        trials = job.trials if job.trials is not None else row.n_trials
        epochs = (
            job.epochs if job.epochs is not None else row.train_config["num_epochs"]
        )
        scale = epochs / p["epochs"] if p.get("epochs") else 1.0
        seconds += trials * (p["train_s_per_seed"] * scale + p["eval_s_per_seed"])
    return seconds / 3600.0
