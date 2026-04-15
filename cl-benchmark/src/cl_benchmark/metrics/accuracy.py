"""
Accuracy matrix computation for continual learning.

The accuracy matrix is the fundamental measurement for continual learning:
A[i,j] = accuracy on task j after training through task i
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cl_benchmark.protocols import ContinualModel, ContinualDataset


def compute_accuracy_matrix(
    model: "ContinualModel",
    dataset: "ContinualDataset",
    tasks_trained: int,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Build accuracy matrix by evaluating model on all tasks.

    The accuracy matrix A[i,j] represents the accuracy on task j
    after training through task i. This function builds a single
    row of the matrix (after training through tasks_trained tasks).

    Args:
        model: Model implementing ContinualModel protocol
        dataset: Dataset implementing ContinualDataset protocol
        tasks_trained: Number of tasks the model has been trained on (row index + 1)
        batch_size: Batch size for evaluation

    Returns:
        1D array of shape (num_tasks,) with accuracies on each task
    """
    accuracies = []

    for task in dataset:
        acc = evaluate_task_accuracy(model, task, batch_size)
        accuracies.append(acc)

    return np.array(accuracies)


def evaluate_task_accuracy(
    model: "ContinualModel",
    task,  # TaskData
    batch_size: int = 256,
) -> float:
    """
    Evaluate model accuracy on a single task.

    Args:
        model: Model implementing ContinualModel protocol
        task: TaskData containing test data
        batch_size: Batch size for evaluation

    Returns:
        Accuracy as float in [0, 1]
    """
    correct = 0
    total = 0

    for images, labels in task.test_batches(batch_size):
        # Get predictions
        predictions = model.predict(images)

        # Convert to class indices
        pred_classes = np.argmax(predictions, axis=1)

        # Handle one-hot encoded labels
        if labels.ndim > 1:
            true_classes = np.argmax(labels, axis=1)
        else:
            true_classes = labels

        correct += np.sum(pred_classes == true_classes)
        total += len(true_classes)

    return correct / total if total > 0 else 0.0


def build_full_accuracy_matrix(
    model: "ContinualModel",
    dataset: "ContinualDataset",
    train_fn,
    epochs_per_task: int = 1,
    batch_size: int = 256,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build full accuracy matrix by training and evaluating sequentially.

    This function trains the model on each task in sequence and
    evaluates on all tasks after each training phase.

    Args:
        model: Model implementing ContinualModel protocol
        dataset: Dataset implementing ContinualDataset protocol
        train_fn: Function(model, task, epochs) -> None that trains the model
        epochs_per_task: Number of training epochs per task
        batch_size: Batch size for evaluation
        verbose: Whether to print progress

    Returns:
        Matrix of shape (num_tasks, num_tasks) where [i,j] is
        accuracy on task j after training through task i
    """
    num_tasks = dataset.num_tasks
    matrix = np.zeros((num_tasks, num_tasks))

    for task_idx, task in enumerate(dataset):
        if verbose:
            print(f"Training on task {task_idx}...")

        # Train on current task
        train_fn(model, task, epochs_per_task)

        # Evaluate on all tasks
        if verbose:
            print(f"Evaluating after task {task_idx}...")

        row = compute_accuracy_matrix(model, dataset, task_idx + 1, batch_size)
        matrix[task_idx] = row

        if verbose:
            print(f"  Accuracies: {row}")

    return matrix
