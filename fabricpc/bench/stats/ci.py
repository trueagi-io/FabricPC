"""95% confidence intervals for a mean over seeds.

A row has only a handful of seeds (two to five), so the interval uses the
Student t value for n - 1 degrees of freedom, not the 1.96 of a normal
curve. With five seeds t is 2.78; with two it is 12.7, so a two-seed
interval is very wide. That is honest: two seeds say little.

We do not bootstrap. With fewer than about ten seeds a bootstrap interval
comes out too narrow (Hesterberg 2015; Colas et al. 2018).

For two rows, trial i of each used the same seed, so the interval that
matters is the one on the per-seed differences. That interval is the
partner of the paired t-test: zero is outside it exactly when p < 0.05.
"""

import math
from typing import Sequence, Tuple

import numpy as np
from scipy import stats

MIN_VALUES = 2
LEVEL = 0.95


def t_critical(n: int) -> float:
    """The two-sided 95% Student t value for a mean of ``n`` values."""
    return float(stats.t.ppf(0.5 + LEVEL / 2, n - 1))


def mean_ci95(values: Sequence[float]) -> Tuple[float, float]:
    """``(low, high)`` of the 95% t interval for the mean of ``values``."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < MIN_VALUES:
        raise ValueError(
            f"a confidence interval needs at least {MIN_VALUES} values, got {n}"
        )
    mean = float(np.mean(arr))
    half = t_critical(n) * float(np.std(arr, ddof=1)) / math.sqrt(n)
    return mean - half, mean + half


def paired_ci95(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """95% t interval for the mean of ``a - b``, pairing value i with value i."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return mean_ci95(diff)
