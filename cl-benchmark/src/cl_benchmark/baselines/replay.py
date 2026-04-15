"""
Experience replay buffer for continual learning.

Provides a simple replay buffer that can be used with any
model to implement experience replay.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from cl_benchmark.protocols import TaskData


class ReplayBuffer:
    """
    Simple experience replay buffer for continual learning.

    Stores samples from previous tasks and provides methods
    to sample replay batches during training.

    Example usage:
        >>> buffer = ReplayBuffer(max_samples_per_task=500)
        >>>
        >>> # After training on task 0
        >>> buffer.add_task(task_id=0, task_data=task_data)
        >>>
        >>> # During training on task 1
        >>> for batch_x, batch_y in train_loader:
        ...     # Get replay samples
        ...     replay = buffer.sample(batch_size=32, exclude_task=1)
        ...     if replay is not None:
        ...         replay_x, replay_y = replay
        ...         # Combine current batch with replay
        ...         combined_x = np.concatenate([batch_x, replay_x])
        ...         combined_y = np.concatenate([batch_y, replay_y])
    """

    def __init__(
        self,
        max_samples_per_task: int = 500,
        max_total_samples: int = 5000,
        seed: int = 42,
    ):
        """
        Initialize replay buffer.

        Args:
            max_samples_per_task: Maximum samples to store per task
            max_total_samples: Maximum total samples across all tasks
            seed: Random seed for sampling
        """
        self.max_samples_per_task = max_samples_per_task
        self.max_total_samples = max_total_samples
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Storage: task_id -> (images, labels)
        self._storage: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._task_ids: List[int] = []

    def add_task(
        self,
        task_id: int,
        task_data: TaskData,
        prioritize_recent: bool = True,
    ) -> None:
        """
        Add samples from a task to the buffer.

        Args:
            task_id: Task identifier
            task_data: TaskData containing samples
            prioritize_recent: If True, replace old tasks when buffer is full
        """
        images = task_data.train_images
        labels = task_data.train_labels
        n_samples = len(images)

        # Subsample if needed
        if n_samples > self.max_samples_per_task:
            indices = self._rng.choice(
                n_samples,
                size=self.max_samples_per_task,
                replace=False,
            )
            images = images[indices]
            labels = labels[indices]

        # Check total capacity
        current_total = sum(len(v[0]) for v in self._storage.values())
        new_total = current_total + len(images)

        if new_total > self.max_total_samples:
            # Remove samples from oldest tasks
            samples_to_remove = new_total - self.max_total_samples
            self._remove_samples(samples_to_remove)

        # Add to storage
        self._storage[task_id] = (images.copy(), labels.copy())
        if task_id not in self._task_ids:
            self._task_ids.append(task_id)

    def _remove_samples(self, n_samples: int) -> None:
        """Remove samples from oldest tasks."""
        removed = 0
        for task_id in list(self._task_ids):
            if removed >= n_samples:
                break

            if task_id in self._storage:
                task_size = len(self._storage[task_id][0])
                if task_size <= n_samples - removed:
                    # Remove entire task
                    del self._storage[task_id]
                    self._task_ids.remove(task_id)
                    removed += task_size
                else:
                    # Remove some samples from this task
                    to_keep = task_size - (n_samples - removed)
                    images, labels = self._storage[task_id]
                    indices = self._rng.choice(task_size, size=to_keep, replace=False)
                    self._storage[task_id] = (images[indices], labels[indices])
                    removed = n_samples

    def sample(
        self,
        batch_size: int,
        exclude_task: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample a batch from the replay buffer.

        Args:
            batch_size: Number of samples to return
            exclude_task: Optional task to exclude from sampling

        Returns:
            Tuple of (images, labels) or None if buffer is empty
        """
        # Get available tasks
        available_tasks = [
            tid
            for tid in self._task_ids
            if tid != exclude_task and tid in self._storage
        ]

        if not available_tasks:
            return None

        # Collect all available samples
        all_images = []
        all_labels = []
        for task_id in available_tasks:
            images, labels = self._storage[task_id]
            all_images.append(images)
            all_labels.append(labels)

        all_images = np.concatenate(all_images, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        n_available = len(all_images)
        if n_available == 0:
            return None

        # Sample
        actual_batch_size = min(batch_size, n_available)
        indices = self._rng.choice(n_available, size=actual_batch_size, replace=False)

        return all_images[indices], all_labels[indices]

    def get_task_ids(self) -> List[int]:
        """Get list of task IDs in the buffer."""
        return list(self._task_ids)

    def __len__(self) -> int:
        """Get total number of samples in buffer."""
        return sum(len(v[0]) for v in self._storage.values())

    def clear(self) -> None:
        """Clear all samples from buffer."""
        self._storage.clear()
        self._task_ids.clear()
