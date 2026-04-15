"""
Base class for continual learning baselines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from cl_benchmark.protocols import TaskData


class BaselineModel(ABC):
    """
    Abstract base class for reference baseline implementations.

    Subclasses must implement predict() and train_on_task() methods
    to be compatible with the benchmark evaluation protocol.
    """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input batch.

        Args:
            x: Input array of shape (batch_size, ...)

        Returns:
            Predictions of shape (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def train_on_task(
        self,
        task_id: int,
        train_data: TaskData,
        epochs: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train on a single task.

        Args:
            task_id: Task identifier
            train_data: TaskData with train/test splits
            epochs: Number of training epochs
            **kwargs: Additional arguments

        Returns:
            Dictionary of training metrics
        """
        pass

    def reset(self) -> None:
        """Reset model to initial state (optional)."""
        pass
