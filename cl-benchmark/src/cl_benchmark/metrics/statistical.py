"""
Statistical tests for comparing continual learning methods.

Provides common statistical tests used in ML research:
- Paired t-test
- Wilcoxon signed-rank test (non-parametric)
- Bootstrap confidence intervals
- Effect size (Cohen's d)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Perform paired t-test to compare two methods.

    Tests the null hypothesis that the mean difference between
    paired observations is zero.

    Args:
        scores_a: Scores from method A (e.g., accuracies across seeds)
        scores_b: Scores from method B

    Returns:
        Tuple of (t_statistic, p_value)

    Example:
        >>> # Compare two methods across 5 random seeds
        >>> method_a_accs = np.array([0.95, 0.94, 0.96, 0.93, 0.95])
        >>> method_b_accs = np.array([0.92, 0.91, 0.93, 0.90, 0.92])
        >>> t_stat, p_val = paired_t_test(method_a_accs, method_b_accs)
        >>> if p_val < 0.05:
        ...     print("Significant difference!")
    """
    try:
        from scipy import stats

        result = stats.ttest_rel(scores_a, scores_b)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        # Fallback: manual implementation
        return _paired_t_test_manual(scores_a, scores_b)


def _paired_t_test_manual(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """Manual paired t-test implementation."""
    differences = scores_a - scores_b
    n = len(differences)

    if n < 2:
        return 0.0, 1.0

    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    if std_diff == 0:
        return float("inf") if mean_diff > 0 else float("-inf"), 0.0

    t_stat = mean_diff / (std_diff / np.sqrt(n))

    # Approximate p-value using normal distribution for large n
    # For small n, this is not accurate but serves as fallback
    from math import erfc, sqrt

    p_value = erfc(abs(t_stat) / sqrt(2))

    return float(t_stat), float(p_value)


def wilcoxon_signed_rank(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).

    Tests the null hypothesis that the distribution of differences
    is symmetric about zero.

    Args:
        scores_a: Scores from method A
        scores_b: Scores from method B

    Returns:
        Tuple of (test_statistic, p_value)

    Note:
        Requires scipy for full implementation. Falls back to
        approximate method if scipy is not available.
    """
    try:
        from scipy import stats

        result = stats.wilcoxon(scores_a, scores_b)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        # Fallback: return approximate values
        differences = scores_a - scores_b
        n_pos = np.sum(differences > 0)
        n_neg = np.sum(differences < 0)
        n_total = n_pos + n_neg

        if n_total == 0:
            return 0.0, 1.0

        # Very rough approximation
        stat = min(n_pos, n_neg)
        # This is not a proper p-value, just a placeholder
        p_value = 2 * min(n_pos, n_neg) / n_total

        return float(stat), float(p_value)


def bootstrap_confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Uses percentile bootstrap method.

    Args:
        scores: Array of scores to analyze
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Example:
        >>> scores = np.array([0.95, 0.94, 0.96, 0.93, 0.95])
        >>> mean, lower, upper = bootstrap_confidence_interval(scores)
        >>> print(f"Mean: {mean:.4f} ({lower:.4f}, {upper:.4f})")
    """
    rng = np.random.default_rng(seed)
    n = len(scores)

    if n == 0:
        return 0.0, 0.0, 0.0

    if n == 1:
        return float(scores[0]), float(scores[0]), float(scores[0])

    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        sample = scores[sample_indices]
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentiles
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    mean = np.mean(scores)

    return float(mean), float(lower), float(upper)


def cohens_d(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size.

    Cohen's d measures the standardized difference between two means:
        d = (mean_a - mean_b) / pooled_std

    Interpretation (Cohen's conventions):
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        scores_a: Scores from method A
        scores_b: Scores from method B

    Returns:
        Cohen's d effect size

    Example:
        >>> method_a = np.array([0.95, 0.94, 0.96, 0.93, 0.95])
        >>> method_b = np.array([0.85, 0.84, 0.86, 0.83, 0.85])
        >>> d = cohens_d(method_a, method_b)
        >>> print(f"Effect size: {d:.2f} (large effect)")
    """
    n_a = len(scores_a)
    n_b = len(scores_b)

    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    var_a = np.var(scores_a, ddof=1)
    var_b = np.var(scores_b, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return (
            float("inf")
            if mean_a > mean_b
            else float("-inf") if mean_a < mean_b else 0.0
        )

    d = (mean_a - mean_b) / pooled_std

    return float(d)


def effect_size_interpretation(d: float) -> str:
    """
    Get interpretation of Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        String interpretation
    """
    abs_d = abs(d)

    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
