"""Reusable statistical functions for paired experiment analysis.

All functions operate on numpy arrays and are independent of FabricPC types.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DescriptiveStats:
    """Summary statistics for a single group of measurements."""

    mean: float
    std: float
    se: float
    min: float
    max: float
    n: int


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired t-test between two groups."""

    t_statistic: float
    p_value: float
    mean_difference: float
    significant_at_05: bool
    n: int


@dataclass(frozen=True)
class EffectSize:
    """Cohen's d effect size with interpretation."""

    d: float
    magnitude: str  # "negligible", "small", "medium", "large"


def descriptive_stats(values: np.ndarray) -> DescriptiveStats:
    """Compute descriptive statistics for a 1D array of measurements."""
    n = len(values)
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return DescriptiveStats(
        mean=float(np.mean(values)),
        std=std,
        se=std / np.sqrt(n) if n > 1 else 0.0,
        min=float(np.min(values)),
        max=float(np.max(values)),
        n=n,
    )


def paired_ttest(group_a: np.ndarray, group_b: np.ndarray) -> PairedTestResult:
    """Perform a paired t-test between two groups.

    Args:
        group_a: 1D array of measurements for arm A.
        group_b: 1D array of measurements for arm B (same length).

    Returns:
        PairedTestResult with t-statistic, p-value, and significance.

    Raises:
        ValueError: If arrays have different lengths or fewer than 2 samples.
    """
    from scipy import stats

    if len(group_a) != len(group_b):
        raise ValueError(
            f"Groups must have same length: {len(group_a)} vs {len(group_b)}"
        )
    if len(group_a) < 2:
        raise ValueError("Need at least 2 paired samples for t-test")

    t_stat, p_value = stats.ttest_rel(group_a, group_b)
    diff = float(np.mean(group_a - group_b))
    return PairedTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_value),
        mean_difference=diff,
        significant_at_05=p_value < 0.05,
        n=len(group_a),
    )


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> EffectSize:
    """Compute Cohen's d for paired samples.

    Uses the standard deviation of the differences as the denominator,
    which is appropriate for paired designs.
    """
    diff = group_a - group_b
    sd = float(np.std(diff, ddof=1))
    d = float(np.mean(diff)) / sd if sd > 1e-10 else 0.0

    if abs(d) >= 0.8:
        magnitude = "large"
    elif abs(d) >= 0.5:
        magnitude = "medium"
    elif abs(d) >= 0.2:
        magnitude = "small"
    else:
        magnitude = "negligible"

    return EffectSize(d=d, magnitude=magnitude)


def estimate_required_n(
    observed_d: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Estimate required sample size for a paired t-test.

    Uses the approximation: n = ((z_alpha + z_beta) / d)^2

    Args:
        observed_d: Observed Cohen's d (effect size).
        alpha: Significance level (two-sided).
        power: Desired statistical power.

    Returns:
        Estimated number of paired trials needed. Returns 999999
        if effect is approximately zero.
    """
    if abs(observed_d) < 1e-10:
        return 999999

    from scipy import stats

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / observed_d) ** 2
    return int(np.ceil(n))
