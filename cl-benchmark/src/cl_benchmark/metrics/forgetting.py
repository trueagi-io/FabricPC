"""
Forgetting and transfer metrics for continual learning.

Provides standard metrics from the continual learning literature:
- Per-task forgetting
- Average forgetting
- Backward transfer (BWT)
- Forward transfer (FWT)
- Average accuracy
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_forgetting(accuracy_matrix: np.ndarray) -> np.ndarray:
    """
    Compute per-task forgetting from accuracy matrix.

    Forgetting for task t is defined as:
        F_t = max_{i>=t}(A[i,t]) - A[T,t]

    where A[i,j] is accuracy on task j after training through task i,
    and T is the final task index.

    Args:
        accuracy_matrix: Matrix where [i,j] is accuracy on task j
                        after training up to task i. Shape: (T, T)

    Returns:
        Per-task forgetting array of shape (T,)
        Values are in [0, 1], where 0 means no forgetting
    """
    num_tasks = accuracy_matrix.shape[1]
    forgetting = np.zeros(num_tasks)

    for task in range(num_tasks):
        # Get accuracies on this task over time
        task_accs = accuracy_matrix[:, task]

        # Only consider after task was first learned
        if task < len(task_accs):
            # Peak accuracy from when task was learned onward
            peak_acc = np.max(task_accs[task:])
            # Final accuracy
            final_acc = task_accs[-1] if len(task_accs) > 0 else 0
            # Forgetting is drop from peak (clamped to non-negative)
            forgetting[task] = max(0, peak_acc - final_acc)

    return forgetting


def compute_average_forgetting(accuracy_matrix: np.ndarray) -> float:
    """
    Compute average forgetting across all tasks.

    Args:
        accuracy_matrix: Accuracy matrix of shape (T, T)

    Returns:
        Average forgetting (mean of per-task forgetting)
    """
    forgetting = compute_forgetting(accuracy_matrix)
    # Don't include the last task (can't forget what you just learned)
    if len(forgetting) > 1:
        return float(np.mean(forgetting[:-1]))
    return 0.0


def compute_backward_transfer(accuracy_matrix: np.ndarray) -> float:
    """
    Compute backward transfer (BWT).

    BWT measures the average influence of learning new tasks
    on the performance of previously learned tasks:

        BWT = (1/(T-1)) * sum_{t=1}^{T-1} (A[T,t] - A[t,t])

    where A[i,j] is accuracy on task j after training through task i.

    Interpretation:
    - Negative BWT indicates catastrophic forgetting
    - Zero BWT indicates no interference
    - Positive BWT indicates backward knowledge transfer

    Args:
        accuracy_matrix: Accuracy matrix of shape (T, T)

    Returns:
        Backward transfer score (typically negative for forgetting)
    """
    num_tasks = accuracy_matrix.shape[1]
    if num_tasks < 2:
        return 0.0

    bwt_sum = 0.0
    count = 0

    for task in range(num_tasks - 1):
        if task < accuracy_matrix.shape[0]:
            # Accuracy right after training task
            initial = accuracy_matrix[task, task]
            # Final accuracy after all tasks
            final = accuracy_matrix[-1, task]
            bwt_sum += final - initial
            count += 1

    return bwt_sum / count if count > 0 else 0.0


def compute_forward_transfer(
    accuracy_matrix: np.ndarray,
    baseline_accuracies: Optional[np.ndarray] = None,
) -> float:
    """
    Compute forward transfer (FWT).

    FWT measures how much learning previous tasks helps learning new tasks:

        FWT = (1/(T-1)) * sum_{t=2}^{T} (A[t-1,t] - baseline[t])

    where baseline[t] is the accuracy on task t when trained from scratch.

    If baseline_accuracies is not provided, this metric cannot be computed
    meaningfully and returns 0.0.

    Args:
        accuracy_matrix: Accuracy matrix of shape (T, T)
        baseline_accuracies: Optional baseline accuracies for each task
                            when trained from scratch (no transfer)

    Returns:
        Forward transfer score (positive = beneficial transfer)
    """
    if baseline_accuracies is None:
        # Without baselines, we can't compute FWT properly
        # Return 0 as a placeholder
        return 0.0

    num_tasks = accuracy_matrix.shape[1]
    if num_tasks < 2:
        return 0.0

    fwt_sum = 0.0
    count = 0

    for task in range(1, num_tasks):
        if task - 1 < accuracy_matrix.shape[0]:
            # Accuracy on new task before training on it
            # (after training on previous tasks)
            pre_train_acc = accuracy_matrix[task - 1, task]
            # Baseline accuracy (random init)
            baseline = baseline_accuracies[task]
            fwt_sum += pre_train_acc - baseline
            count += 1

    return fwt_sum / count if count > 0 else 0.0


def compute_average_accuracy(accuracy_matrix: np.ndarray) -> float:
    """
    Compute average accuracy across all tasks at end of training.

    This is the simplest overall performance metric:
        AA = (1/T) * sum_{t=1}^{T} A[T,t]

    Args:
        accuracy_matrix: Accuracy matrix of shape (T, T)

    Returns:
        Average final accuracy across all tasks
    """
    if accuracy_matrix.size == 0:
        return 0.0

    # Get final row (after all training)
    final_accuracies = accuracy_matrix[-1, :]

    return float(np.mean(final_accuracies))


def compute_learning_accuracy(accuracy_matrix: np.ndarray) -> float:
    """
    Compute average learning accuracy (accuracy immediately after learning each task).

    LA = (1/T) * sum_{t=1}^{T} A[t,t]

    Args:
        accuracy_matrix: Accuracy matrix of shape (T, T)

    Returns:
        Average accuracy on each task right after learning it
    """
    if accuracy_matrix.size == 0:
        return 0.0

    # Diagonal elements = accuracy on task t right after training on t
    diagonal = np.diag(accuracy_matrix)

    return float(np.mean(diagonal))


def compute_intransigence(
    accuracy_matrix: np.ndarray,
    joint_accuracies: Optional[np.ndarray] = None,
) -> float:
    """
    Compute intransigence measure.

    Intransigence measures the inability to learn new tasks due to
    previous knowledge:

        I = (1/T) * sum_{t=1}^{T} (joint[t] - A[t,t])

    where joint[t] is accuracy when trained jointly on all data.

    Args:
        accuracy_matrix: Accuracy matrix of shape (T, T)
        joint_accuracies: Optional joint training accuracies

    Returns:
        Intransigence score (higher = harder to learn new tasks)
    """
    if joint_accuracies is None:
        return 0.0

    diagonal = np.diag(accuracy_matrix)
    intransigence = joint_accuracies - diagonal

    return float(np.mean(intransigence))
