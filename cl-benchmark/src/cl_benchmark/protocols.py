"""
Core protocols and data structures for continual learning benchmarks.

This module defines the interfaces that models and datasets must implement
to be compatible with the cl-benchmark evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np


@dataclass
class TaskData:
    """
    Container for a single task's data in continual learning.

    Provides train/test splits and batch iteration utilities.
    All data is stored as NumPy arrays for framework interoperability.

    Attributes:
        task_id: Unique identifier for this task
        classes: Tuple of class labels included in this task
        train_images: Training images, shape (N, ...)
        train_labels: Training labels, shape (N,) or (N, num_classes)
        test_images: Test images, shape (M, ...)
        test_labels: Test labels, shape (M,) or (M, num_classes)
        metadata: Optional additional task information
    """

    task_id: int
    classes: Tuple[int, ...]
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data shapes."""
        assert len(self.train_images) == len(self.train_labels), (
            f"Train images ({len(self.train_images)}) and labels "
            f"({len(self.train_labels)}) must have same length"
        )
        assert len(self.test_images) == len(self.test_labels), (
            f"Test images ({len(self.test_images)}) and labels "
            f"({len(self.test_labels)}) must have same length"
        )

    @property
    def num_train(self) -> int:
        """Number of training samples."""
        return len(self.train_images)

    @property
    def num_test(self) -> int:
        """Number of test samples."""
        return len(self.test_images)

    @property
    def num_classes(self) -> int:
        """Number of classes in this task."""
        return len(self.classes)

    def train_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over training data in batches.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            seed: Random seed for reproducible shuffling

        Yields:
            Tuples of (images, labels) batches
        """
        yield from _batch_iterator(
            self.train_images,
            self.train_labels,
            batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    def test_batches(
        self,
        batch_size: int,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over test data in batches.

        Args:
            batch_size: Number of samples per batch

        Yields:
            Tuples of (images, labels) batches
        """
        yield from _batch_iterator(
            self.test_images,
            self.test_labels,
            batch_size,
            shuffle=False,
        )


def _batch_iterator(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generic batch iterator for numpy arrays.

    Args:
        images: Image array (N, ...)
        labels: Label array (N, ...) or (N,)
        batch_size: Samples per batch
        shuffle: Whether to shuffle indices
        seed: Random seed for shuffling

    Yields:
        (batch_images, batch_labels) tuples
    """
    n_samples = len(images)
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield images[batch_indices], labels[batch_indices]


@runtime_checkable
class ContinualDataset(Protocol):
    """
    Protocol for continual learning datasets.

    Datasets must provide sequential access to tasks and metadata
    about the overall benchmark structure.

    Example implementation:
        class MyDataset:
            @property
            def num_tasks(self) -> int:
                return 5

            @property
            def num_classes(self) -> int:
                return 10

            def get_task(self, task_id: int) -> TaskData:
                return TaskData(...)

            def __iter__(self) -> Iterator[TaskData]:
                for i in range(self.num_tasks):
                    yield self.get_task(i)
    """

    @property
    def num_tasks(self) -> int:
        """Total number of tasks in the benchmark."""
        ...

    @property
    def num_classes(self) -> int:
        """Total number of unique classes across all tasks."""
        ...

    def get_task(self, task_id: int) -> TaskData:
        """
        Get data for a specific task.

        Args:
            task_id: Task index (0 to num_tasks-1)

        Returns:
            TaskData containing train/test splits for the task
        """
        ...

    def __iter__(self) -> Iterator[TaskData]:
        """Iterate over tasks in sequential order."""
        ...


@runtime_checkable
class ContinualModel(Protocol):
    """
    Protocol for models compatible with cl-benchmark.

    Models must implement predict() and train_on_task() methods.
    This protocol works with any deep learning framework.

    Example implementation (PyTorch):
        class MyModel:
            def __init__(self):
                self.net = torch.nn.Linear(784, 10)
                self.optimizer = torch.optim.Adam(self.net.parameters())

            def predict(self, x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    logits = self.net(torch.from_numpy(x).float())
                return logits.numpy()

            def train_on_task(self, task_id, train_data, epochs=1):
                for _ in range(epochs):
                    for x, y in train_data.train_batches(256):
                        x = torch.from_numpy(x).float()
                        y = torch.from_numpy(y).long()
                        loss = F.cross_entropy(self.net(x), y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                return {"loss": loss.item()}
    """

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input batch.

        Args:
            x: Input array of shape (batch_size, ...)
               Shape depends on dataset (e.g., (N, 784) for flattened MNIST)

        Returns:
            Predictions of shape (batch_size, num_classes)
            Can be logits or probabilities (argmax is used for accuracy)
        """
        ...

    def train_on_task(
        self,
        task_id: int,
        train_data: TaskData,
        epochs: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train on a single task.

        Args:
            task_id: Identifier for the current task
            train_data: TaskData containing train/test data
            epochs: Number of training epochs
            **kwargs: Framework-specific options

        Returns:
            Dictionary of training metrics (e.g., {"loss": 0.5, "accuracy": 0.9})
        """
        ...


@runtime_checkable
class StatefulModel(ContinualModel, Protocol):
    """
    Extended protocol for models that support state save/restore.

    Useful for baselines that need to store/restore model state
    (e.g., for computing Fisher information in EWC).
    """

    def save_state(self) -> Dict[str, Any]:
        """
        Save model state.

        Returns:
            Dictionary containing model state that can be restored
        """
        ...

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore model state.

        Args:
            state: State dictionary from save_state()
        """
        ...


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding.

    Args:
        labels: Integer labels of shape (N,)
        num_classes: Total number of classes

    Returns:
        One-hot encoded labels of shape (N, num_classes)
    """
    return np.eye(num_classes, dtype=np.float32)[labels]
