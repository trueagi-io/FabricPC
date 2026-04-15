"""
Visualization module for continual learning benchmarks.

Provides plotting utilities for visualizing benchmark results.
"""

from cl_benchmark.visualization.plots import (
    plot_accuracy_matrix,
    plot_forgetting_analysis,
    plot_learning_curves,
    plot_method_comparison,
)

__all__ = [
    "plot_accuracy_matrix",
    "plot_forgetting_analysis",
    "plot_learning_curves",
    "plot_method_comparison",
]
