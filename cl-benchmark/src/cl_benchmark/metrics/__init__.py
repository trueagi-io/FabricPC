"""
Metrics module for continual learning evaluation.

Provides comprehensive metrics for evaluating continual learning:
- Accuracy matrix computation
- Forgetting measures
- Transfer metrics (backward and forward)
- Statistical tests for comparison
"""

from cl_benchmark.metrics.accuracy import compute_accuracy_matrix
from cl_benchmark.metrics.forgetting import (
    compute_forgetting,
    compute_backward_transfer,
    compute_forward_transfer,
    compute_average_accuracy,
    compute_average_forgetting,
)
from cl_benchmark.metrics.statistical import (
    paired_t_test,
    wilcoxon_signed_rank,
    bootstrap_confidence_interval,
    cohens_d,
)

__all__ = [
    # Accuracy
    "compute_accuracy_matrix",
    # Forgetting
    "compute_forgetting",
    "compute_backward_transfer",
    "compute_forward_transfer",
    "compute_average_accuracy",
    "compute_average_forgetting",
    # Statistical
    "paired_t_test",
    "wilcoxon_signed_rank",
    "bootstrap_confidence_interval",
    "cohens_d",
]
