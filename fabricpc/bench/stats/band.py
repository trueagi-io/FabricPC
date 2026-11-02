"""Pass or fail: does a row's result land near the score it should?

Issue #59 asks for a task-metric band on every row, so a configuration that
runs fast but trains badly fails instead of looking good in a table. A row
passes when

    |mean - expected| <= max(floor, 2 * SE)

where the mean and standard error (SE) come from the row's trials and the
floor is the smallest miss we allow (half a percentage point of accuracy by
default). Two standard errors lets a noisy row wobble as much as its own
seeds do. This rule and the 5-seed count are our proposal; the sponsor has
not signed off yet, so every verdict says so.

A verdict is only given for a full run: the row's own epoch count and at
least its stated number of seeds. Anything shorter, like the CI smoke run,
is marked not comparable instead of passing or failing by accident.
"""

from dataclasses import asdict, dataclass, replace
from typing import Optional

from fabricpc.bench.definitions.registry import BenchmarkRow
from fabricpc.bench.stats.summary import RowSummary, write_summary

DEFAULT_FLOOR = 0.005
RULE = "max(floor, 2*SE)"


@dataclass(frozen=True)
class BandVerdict:
    status: str  # "pass", "fail", "no_reference", or "not_comparable"
    reason: str
    metric: Optional[str] = None
    expected: Optional[float] = None
    observed: Optional[float] = None
    se: Optional[float] = None
    band: Optional[float] = None
    diff: Optional[float] = None  # observed - expected
    source: Optional[str] = None
    rule: str = RULE
    pending_signoff: bool = True


def check_band(row: BenchmarkRow, summary: RowSummary) -> BandVerdict:
    """Judge one row's summary against the row's expected score."""
    ref = row.reference
    if ref is None:
        return BandVerdict(
            status="no_reference",
            reason="no expected score yet; the first full run will set one",
        )

    base = BandVerdict(
        status="not_comparable",
        reason="",
        metric=ref.metric,
        expected=ref.value,
        source=ref.source,
    )
    full_epochs = float(row.train_config["num_epochs"])
    if summary.num_epochs != full_epochs:
        return replace(
            base,
            reason=f"ran {summary.num_epochs:g} epochs; the expected score is "
            f"for {full_epochs:g}",
        )
    if summary.n_ok < row.n_trials:
        return replace(
            base,
            reason=f"{summary.n_ok} good seeds; the expected score is for "
            f"{row.n_trials}",
        )
    stats = summary.metrics.get(ref.metric)
    if stats is None:
        return replace(base, reason=f"the run did not report {ref.metric}")

    band = max(ref.floor, 2.0 * stats["se"])
    diff = stats["mean"] - ref.value
    passed = abs(diff) <= band
    return replace(
        base,
        status="pass" if passed else "fail",
        reason=f"{ref.metric} {stats['mean']:.4f} vs expected {ref.value:.4f} "
        f"(allowed miss {band:.4f})",
        observed=stats["mean"],
        se=stats["se"],
        band=band,
        diff=diff,
    )


def attach_band(results_dir, row: BenchmarkRow, summary: RowSummary) -> RowSummary:
    """Add the verdict to the summary and rewrite ``summary.json``."""
    judged = replace(summary, band=asdict(check_band(row, summary)))
    write_summary(results_dir, judged)
    return judged
