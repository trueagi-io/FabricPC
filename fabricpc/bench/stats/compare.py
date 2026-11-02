"""Comparisons between rows, computed by fabricpc.experiments.

Issue #59 asks that every benchmark go through
``PlannedMultiContrastExperiment``. That runner trains every arm inside one
process and records only the task metric and wall time, while the suite runs
each trial in its own process and records timing, memory, and compute too.
So the suite does the training, and this module hands the finished trials to
the framework's own results class, whose ``contrast_results()`` does the
paired t-tests and effect sizes. The statistics are the framework's, not a
copy of them.

That is only honest if a trial here trains the same model the framework
would. ``arm_for_row`` builds the framework's view of a row, and the tests
check that both give the same accuracy for the same seed.
"""

import functools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

from fabricpc.bench.stats.ci import paired_ci95
from fabricpc.bench.definitions.registry import BenchmarkRow, Comparison
from fabricpc.bench.run.runner import _TRAINER_ALGORITHM
from fabricpc.experiments import ExperimentArm, PlannedMultiContrastResults
from fabricpc.experiments.ab_experiment import TrialResult as ArmTrial
from fabricpc.training import evaluate, train

MIN_TRIALS = 2


def arm_for_row(
    row: BenchmarkRow, *, steps_per_epoch: int, num_epochs: Optional[float] = None
) -> ExperimentArm:
    """The row as an arm of PlannedMultiContrastExperiment.

    The framework builds the optimizer before it sees a loader, so the caller
    says how many batches an epoch has; the learning-rate schedule needs it.
    """
    algorithm = _TRAINER_ALGORITHM[row.algorithm]
    epochs = num_epochs if num_epochs is not None else row.train_config["num_epochs"]
    return ExperimentArm(
        name=row.id,
        model_factory=row.model_factory,
        train_fn=functools.partial(train, algorithm=algorithm),
        eval_fn=functools.partial(
            evaluate, algorithm=algorithm, metrics=row.metrics_for(algorithm)
        ),
        optimizer=row.optimizer_factory(math.ceil(steps_per_epoch * float(epochs))),
        train_config={**row.train_config, "num_epochs": float(epochs)},
    )


def _good_trials(results_dir, row_id: str):
    row_dir = Path(results_dir) / row_id
    files = sorted(row_dir.glob("trial*.json"), key=lambda p: int(p.stem[5:]))
    if not files:
        raise FileNotFoundError(f"no trial files under {row_dir}")
    trials = [json.loads(p.read_text()) for p in files]
    return [t for t in trials if t.get("status") == "ok"]


def _pair_on_shared_seeds(results_dir, rows: Sequence[str]):
    """Good trials of every row, kept only for seeds that all rows share.

    Returns ``(shared_seeds, trials_by_row, unpaired_by_row)``. Trial i of
    one row and trial i of another are only a pair if they used the same
    seed, so a seed missing from any row (never run, or its trial failed)
    is left out of every row and reported as unpaired.
    """
    by_row = {
        row_id: {t["seed"]: t for t in _good_trials(results_dir, row_id)}
        for row_id in rows
    }
    first = by_row[rows[0]]
    shared = [s for s in first if all(s in by_row[r] for r in rows)]
    unpaired = {
        r: sorted(s for s in by_row[r] if s not in shared)
        for r in rows
        if any(s not in shared for s in by_row[r])
    }
    trials = {r: [by_row[r][s] for s in shared] for r in rows}
    return shared, trials, unpaired


def planned_results(
    results_dir,
    *,
    rows: Sequence[str],
    contrasts: Sequence[Tuple[str, str]],
    metric: str,
) -> PlannedMultiContrastResults:
    """Load finished trials into the framework's results object.

    Pairs on the seeds every row has (see ``_pair_on_shared_seeds``) and
    refuses when fewer than two seeds are shared: unpaired numbers must not
    go into a paired test.
    """
    shared, trials, _ = _pair_on_shared_seeds(results_dir, rows)
    if len(shared) < MIN_TRIALS:
        raise ValueError(
            f"need at least {MIN_TRIALS} seeds that all of {list(rows)} ran "
            f"successfully, found {len(shared)} ({shared})"
        )
    per_arm = {
        row_id: [
            ArmTrial(
                metric_value=float(t["metrics"][metric]),
                train_time=float(t.get("train_time_s", 0.0)),
                all_metrics=dict(t["metrics"]),
            )
            for t in trials[row_id]
        ]
        for row_id in rows
    }
    return PlannedMultiContrastResults(
        arm_names=list(rows),
        contrasts=list(contrasts),
        metric=metric,
        n_trials=len(shared),
        per_arm_trials=per_arm,
        seeds=list(shared),
        total_time=sum(t.train_time for arm in per_arm.values() for t in arm),
        num_epochs=float(trials[rows[0]][0].get("num_epochs", 0.0)),
    )


def _plain(d: dict) -> dict:
    """numpy scalars (the framework returns some) as plain Python values."""
    return {k: v.item() if hasattr(v, "item") else v for k, v in d.items()}


def _with_ci(res: PlannedMultiContrastResults, contrast: dict) -> dict:
    """Add the 95% t interval on the paired difference to one contrast.

    The framework's contrast has the SE of the difference but no interval,
    so we work it out from the same per-seed differences.
    """
    a = res.per_arm_metrics(contrast["arm_a"])
    b = res.per_arm_metrics(contrast["arm_b"])
    low, high = paired_ci95(a, b)
    return {**contrast, "ci95_low": low, "ci95_high": high}


def compare_family(results_dir, comparison: Comparison, *, metric: str) -> dict:
    """Run a family's planned contrasts and write ``compare-<family>.json``."""
    res = planned_results(
        results_dir,
        rows=comparison.rows,
        contrasts=comparison.contrasts,
        metric=metric,
    )
    out = {
        "comparison": comparison.id,
        "metric": metric,
        "seeds": res.seeds,
        "n_trials": res.n_trials,
        "unpaired_seeds": _pair_on_shared_seeds(results_dir, comparison.rows)[2],
        "num_epochs": res.num_epochs,
        "contrasts": [_with_ci(res, _plain(asdict(c))) for c in res.contrast_results()],
    }
    # The family's own metric keeps the plain name; any other metric gets its
    # own file, so comparing on a second metric never overwrites the first.
    suffix = "" if metric == comparison.metric else f"-{metric}"
    path = Path(results_dir) / f"compare-{comparison.id}{suffix}.json"
    path.write_text(json.dumps(out, indent=2))
    return out
