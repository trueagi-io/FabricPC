"""Quick numeric regression checks for CI.

    python -m fabricpc.bench regress [--out DIR] [row-id ...]

The unit tests prove the code runs. ``smoke`` proves one MNIST model still
learns. This goes one step further: it trains a few tiny models, each with
sPC, ePC and backprop, and checks that every score is still on the right
side of a fixed line. A change that quietly makes the library worse (a
wrong sign in a gradient, an inference step that stopped doing anything)
shows up here as a FAIL, even when every unit test still passes.

The checks:

- ``mnist-mlp-*``: half an epoch of MNIST, test accuracy at least 0.85
  (the same floor the smoke run has always used).
- ``mnist-autoencoder-*``: half an epoch, test reconstruction MSE at most
  0.055. Handing back the average training image scores 0.0675, so a model
  that learned nothing useful cannot pass.
- ``patterns64-hopfield-*``: the row's full 30 epochs (it is tiny), bit
  accuracy lift over the noisy probe at 20% flips of at least 0.10. A model
  that learned nothing has a lift of 0.

Each check runs two seeds (trial 0 and 1, the same seeds as every other
run) and compares the mean of the two to its line. The lines were set from
real runs on a Mac with a wide margin, so a slightly different CPU or jax
version does not make CI flaky; see ``MEASURED`` below for what the runs
gave. After the checks, the folder is validated like any other results
folder, and a report page (``report.md``) and ``regress.json`` (every
verdict) are written next to the results.

Every run is short on purpose, so a PASS says "nothing is badly broken",
not "the scores match the paper". The full rows and their bands do that.
"""

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from fabricpc.bench.results.report import run_report
from fabricpc.bench.results.writer import validate


@dataclass(frozen=True)
class Check:
    """One row, one metric, and the line its mean has to stay on the right
    side of. ``limit`` is a floor when higher is better, else a ceiling."""

    row_id: str
    metric: str
    limit: float
    higher_is_better: bool
    epochs: float
    trials: int = 2


@dataclass(frozen=True)
class Verdict:
    row_id: str
    metric: str
    value: Optional[float]
    limit: float
    higher_is_better: bool
    passed: bool
    reason: str


MLP_FLOOR = 0.85
AUTOENCODER_CEILING = 0.055
HOPFIELD_LIFT_FLOOR = 0.10
ALGOS = ("spc", "epc", "backprop")

CI_CHECKS: Tuple[Check, ...] = (
    *(Check(f"mnist-mlp-{a}", "accuracy", MLP_FLOOR, True, 0.5) for a in ALGOS),
    *(
        Check(
            f"mnist-autoencoder-{a}",
            "reconstruction_mse",
            AUTOENCODER_CEILING,
            False,
            0.5,
        )
        for a in ALGOS
    ),
    *(
        Check(
            f"patterns64-hopfield-{a}",
            "bit_accuracy_lift_p20",
            HOPFIELD_LIFT_FLOOR,
            True,
            30,
        )
        for a in ALGOS
    ),
)

# What the checks scored on a Mac (Apple silicon, CPU), seeds 0, 1000,
# 2000, 3000 and 4000 at the check's epochs: (lowest, highest) single trial,
# then the mean of seeds 0 and 1000, which is what CI compares. The same
# seed gave the same number on every repeat, in a child process or not.
MEASURED = {
    "mnist-mlp-spc": ((0.9029, 0.9068), 0.9049),
    "mnist-mlp-epc": ((0.9043, 0.9091), 0.9067),
    "mnist-mlp-backprop": ((0.9043, 0.9093), 0.9068),
    "mnist-autoencoder-spc": ((0.0413, 0.0440), 0.0424),
    "mnist-autoencoder-epc": ((0.0400, 0.0437), 0.0410),
    "mnist-autoencoder-backprop": ((0.0397, 0.0439), 0.0409),
    "patterns64-hopfield-spc": ((0.1496, 0.1538), 0.1521),
    "patterns64-hopfield-epc": ((0.1501, 0.1542), 0.1523),
    "patterns64-hopfield-backprop": ((0.1606, 0.1661), 0.1633),
}


def judge(check: Check, value: Optional[float]) -> Verdict:
    """PASS or FAIL for one score. A missing or non-finite score fails."""
    if value is None or not math.isfinite(value):
        ok, reason = False, f"no usable {check.metric} (got {value})"
    elif check.higher_is_better:
        ok = value >= check.limit
        reason = f"{check.metric} {value:.4f} {'>=' if ok else '<'} {check.limit}"
    else:
        ok = value <= check.limit
        reason = f"{check.metric} {value:.4f} {'<=' if ok else '>'} {check.limit}"
    return Verdict(
        row_id=check.row_id,
        metric=check.metric,
        value=value,
        limit=check.limit,
        higher_is_better=check.higher_is_better,
        passed=ok,
        reason=reason,
    )


RunRow = Callable[[Check], tuple]  # check -> (failed trial count, summary)


def run_checks(checks: Sequence[Check], run_row: RunRow) -> list:
    """Run every check's row and judge it. ``run_row`` does the training."""
    verdicts = []
    for check in checks:
        failed, summary = run_row(check)
        if failed or summary is None:
            verdict = replace(judge(check, None), reason="a trial failed")
        else:
            score = summary.metrics.get(check.metric) or {}
            verdict = judge(check, score.get("mean"))
        print(f"regress: {verdict.row_id}: {_label(verdict)} ({verdict.reason})")
        verdicts.append(verdict)
    return verdicts


def select(checks: Sequence[Check], row_ids: Sequence[str]) -> Tuple[Check, ...]:
    """Only the checks for ``row_ids`` (all of them when it is empty)."""
    if not row_ids:
        return tuple(checks)
    by_row = {c.row_id: c for c in checks}
    for row_id in row_ids:
        if row_id not in by_row:
            raise KeyError(row_id)
    return tuple(by_row[r] for r in row_ids)


def _label(verdict: Verdict) -> str:
    return "PASS" if verdict.passed else "FAIL"


def format_table(verdicts: Sequence[Verdict]) -> str:
    lines = [f"{'row':<28} {'metric':<24} {'mean':>8}  {'line':>8}  result"]
    for v in verdicts:
        value = "-" if v.value is None else f"{v.value:.4f}"
        line = f"{'>=' if v.higher_is_better else '<='}{v.limit}"
        lines.append(
            f"{v.row_id:<28} {v.metric:<24} {value:>8}  {line:>8}  {_label(v)}"
        )
    return "\n".join(lines)


def run_regress(out, run_row: RunRow, checks: Optional[Sequence[Check]] = None) -> int:
    """Run the checks, validate the folder, write the report. 0 if all good."""
    checks = CI_CHECKS if checks is None else checks
    Path(out).mkdir(parents=True, exist_ok=True)
    verdicts = run_checks(checks, run_row)
    print(format_table(verdicts))

    problems = validate(out)
    for problem in problems:
        print(f"validate: {problem}")
    run_report(out, out=Path(out) / "report.md")

    passed = all(v.passed for v in verdicts) and not problems
    Path(out, "regress.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": [asdict(v) for v in verdicts],
                "validate_problems": problems,
            },
            indent=2,
        )
    )
    failed = [v.row_id for v in verdicts if not v.passed]
    if passed:
        print(f"regress: ok ({len(verdicts)} checks passed, {out} validates)")
    else:
        print(
            f"regress: FAIL ({len(failed)} of {len(verdicts)} checks failed"
            f"{', ' + str(len(problems)) + ' validate problem(s)' if problems else ''})"
        )
    return 0 if passed else 1
