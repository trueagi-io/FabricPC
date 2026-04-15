"""
Metric aggregation utilities for continual learning.

Provides functions for aggregating metrics across runs,
computing summary statistics, and extracting learning curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics across multiple runs.

    Stores mean and standard deviation for key metrics
    computed across multiple random seeds or runs.
    """

    # Number of runs aggregated
    num_runs: int

    # Average accuracy
    average_accuracy_mean: float
    average_accuracy_std: float

    # Forgetting
    forgetting_mean: float
    forgetting_std: float

    # Backward transfer
    bwt_mean: float
    bwt_std: float

    # Forward transfer (if available)
    fwt_mean: float = 0.0
    fwt_std: float = 0.0

    # Learning accuracy (diagonal of accuracy matrix)
    learning_accuracy_mean: float = 0.0
    learning_accuracy_std: float = 0.0

    # Per-task final accuracies
    per_task_accuracy_means: np.ndarray = field(default_factory=lambda: np.array([]))
    per_task_accuracy_stds: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-task forgetting
    per_task_forgetting_means: np.ndarray = field(default_factory=lambda: np.array([]))
    per_task_forgetting_stds: np.ndarray = field(default_factory=lambda: np.array([]))


def aggregate_accuracy_matrices(
    matrices: List[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate multiple accuracy matrices.

    Args:
        matrices: List of accuracy matrices from different runs

    Returns:
        Tuple of (mean_matrix, std_matrix)
    """
    if not matrices:
        return np.array([]), np.array([])

    stacked = np.stack(matrices, axis=0)
    mean_matrix = np.mean(stacked, axis=0)
    std_matrix = np.std(stacked, axis=0)

    return mean_matrix, std_matrix


def aggregate_metrics(
    accuracy_matrices: List[np.ndarray],
    forgetting_arrays: Optional[List[np.ndarray]] = None,
    bwt_values: Optional[List[float]] = None,
    fwt_values: Optional[List[float]] = None,
) -> AggregatedMetrics:
    """
    Aggregate metrics across multiple runs.

    Args:
        accuracy_matrices: List of accuracy matrices
        forgetting_arrays: Optional list of per-task forgetting arrays
        bwt_values: Optional list of BWT values
        fwt_values: Optional list of FWT values

    Returns:
        AggregatedMetrics with mean and std for all metrics
    """
    num_runs = len(accuracy_matrices)

    if num_runs == 0:
        return AggregatedMetrics(
            num_runs=0,
            average_accuracy_mean=0.0,
            average_accuracy_std=0.0,
            forgetting_mean=0.0,
            forgetting_std=0.0,
            bwt_mean=0.0,
            bwt_std=0.0,
        )

    # Compute average accuracy for each run
    avg_accs = [np.mean(m[-1, :]) for m in accuracy_matrices]
    avg_acc_mean = float(np.mean(avg_accs))
    avg_acc_std = float(np.std(avg_accs))

    # Compute learning accuracy (diagonal) for each run
    learning_accs = [np.mean(np.diag(m)) for m in accuracy_matrices]
    learning_acc_mean = float(np.mean(learning_accs))
    learning_acc_std = float(np.std(learning_accs))

    # Per-task accuracies
    final_accs = np.stack([m[-1, :] for m in accuracy_matrices], axis=0)
    per_task_acc_means = np.mean(final_accs, axis=0)
    per_task_acc_stds = np.std(final_accs, axis=0)

    # Forgetting
    if forgetting_arrays is not None and len(forgetting_arrays) > 0:
        forgetting_stacked = np.stack(forgetting_arrays, axis=0)
        avg_forgetting = [
            np.mean(f[:-1]) if len(f) > 1 else 0.0 for f in forgetting_arrays
        ]
        forgetting_mean = float(np.mean(avg_forgetting))
        forgetting_std = float(np.std(avg_forgetting))
        per_task_fgt_means = np.mean(forgetting_stacked, axis=0)
        per_task_fgt_stds = np.std(forgetting_stacked, axis=0)
    else:
        forgetting_mean = 0.0
        forgetting_std = 0.0
        per_task_fgt_means = np.array([])
        per_task_fgt_stds = np.array([])

    # BWT
    if bwt_values is not None and len(bwt_values) > 0:
        bwt_mean = float(np.mean(bwt_values))
        bwt_std = float(np.std(bwt_values))
    else:
        bwt_mean = 0.0
        bwt_std = 0.0

    # FWT
    if fwt_values is not None and len(fwt_values) > 0:
        fwt_mean = float(np.mean(fwt_values))
        fwt_std = float(np.std(fwt_values))
    else:
        fwt_mean = 0.0
        fwt_std = 0.0

    return AggregatedMetrics(
        num_runs=num_runs,
        average_accuracy_mean=avg_acc_mean,
        average_accuracy_std=avg_acc_std,
        forgetting_mean=forgetting_mean,
        forgetting_std=forgetting_std,
        bwt_mean=bwt_mean,
        bwt_std=bwt_std,
        fwt_mean=fwt_mean,
        fwt_std=fwt_std,
        learning_accuracy_mean=learning_acc_mean,
        learning_accuracy_std=learning_acc_std,
        per_task_accuracy_means=per_task_acc_means,
        per_task_accuracy_stds=per_task_acc_stds,
        per_task_forgetting_means=per_task_fgt_means,
        per_task_forgetting_stds=per_task_fgt_stds,
    )


def extract_learning_curve(
    accuracy_matrices: List[np.ndarray],
    task_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract learning curve for a specific task.

    The learning curve shows how accuracy on a task changes
    as more tasks are learned.

    Args:
        accuracy_matrices: List of accuracy matrices
        task_id: Task to extract curve for

    Returns:
        Tuple of (mean_curve, std_curve) where each is array of shape (T,)
    """
    if not accuracy_matrices:
        return np.array([]), np.array([])

    # Get the column for this task from each matrix
    curves = [m[:, task_id] for m in accuracy_matrices]
    stacked = np.stack(curves, axis=0)

    mean_curve = np.mean(stacked, axis=0)
    std_curve = np.std(stacked, axis=0)

    return mean_curve, std_curve


def compute_summary_table(
    results: Dict[str, AggregatedMetrics],
) -> str:
    """
    Generate a summary table comparing multiple methods.

    Args:
        results: Dictionary mapping method names to AggregatedMetrics

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display."

    # Header
    lines = [
        "=" * 80,
        f"{'Method':<20} {'Avg Acc':<12} {'Forgetting':<12} {'BWT':<12} {'Runs':<8}",
        "-" * 80,
    ]

    # Sort by average accuracy
    sorted_methods = sorted(
        results.items(),
        key=lambda x: x[1].average_accuracy_mean,
        reverse=True,
    )

    for method, metrics in sorted_methods:
        acc = f"{metrics.average_accuracy_mean:.4f}±{metrics.average_accuracy_std:.4f}"
        fgt = f"{metrics.forgetting_mean:.4f}±{metrics.forgetting_std:.4f}"
        bwt = f"{metrics.bwt_mean:.4f}±{metrics.bwt_std:.4f}"
        runs = str(metrics.num_runs)

        lines.append(f"{method:<20} {acc:<12} {fgt:<12} {bwt:<12} {runs:<8}")

    lines.append("=" * 80)

    return "\n".join(lines)
