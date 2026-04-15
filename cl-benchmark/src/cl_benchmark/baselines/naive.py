"""
Naive fine-tuning baseline for continual learning.

Simple sequential training without any protection against
catastrophic forgetting. Serves as a lower bound baseline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from cl_benchmark.baselines.base import BaselineModel
from cl_benchmark.protocols import TaskData


class NaiveModel(BaselineModel):
    """
    Naive fine-tuning baseline.

    Simply trains on each task sequentially without any mechanism
    to prevent forgetting. This establishes a lower bound for
    continual learning performance.

    This is a minimal reference implementation using a simple
    linear model with softmax. For real experiments, users should
    implement their own model using this as a template.

    Example:
        >>> from cl_benchmark import BenchmarkRunner, BenchmarkConfig
        >>> from cl_benchmark.baselines import NaiveModel
        >>>
        >>> model = NaiveModel(input_dim=784, num_classes=10)
        >>> config = BenchmarkConfig(dataset_name="split-mnist")
        >>> runner = BenchmarkRunner(config)
        >>> results = runner.evaluate(model)
    """

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        hidden_dim: int = 256,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        """
        Initialize naive model.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for gradient descent
            seed: Random seed for initialization
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._init_params()

    def _init_params(self) -> None:
        """Initialize model parameters."""
        # Xavier initialization
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        scale2 = np.sqrt(2.0 / (self.hidden_dim + self.num_classes))

        self.W1 = self._rng.normal(0, scale1, (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = self._rng.normal(0, scale2, (self.hidden_dim, self.num_classes))
        self.b2 = np.zeros(self.num_classes)

    def reset(self) -> None:
        """Reset model to initial state."""
        self._rng = np.random.default_rng(self.seed)
        self._init_params()

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        """ReLU gradient."""
        return (x > 0).astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning intermediate values for backprop."""
        z1 = x @ self.W1 + self.b1
        h1 = self._relu(z1)
        z2 = h1 @ self.W2 + self.b2
        return z1, h1, z2

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input batch.

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Predictions of shape (batch_size, num_classes)
        """
        _, _, z2 = self._forward(x)
        return self._softmax(z2)

    def train_on_task(
        self,
        task_id: int,
        train_data: TaskData,
        epochs: int = 1,
        batch_size: int = 256,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train on a single task using simple SGD.

        Args:
            task_id: Task identifier (unused in naive baseline)
            train_data: TaskData with training samples
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with training metrics
        """
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for x, y in train_data.train_batches(batch_size=batch_size, shuffle=True):
                loss = self._train_step(x, y)
                total_loss += loss
                num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "epochs": epochs,
            "task_id": task_id,
        }

    def _train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Single training step with gradient descent.

        Args:
            x: Input batch (batch_size, input_dim)
            y: One-hot labels (batch_size, num_classes)

        Returns:
            Cross-entropy loss value
        """
        batch_size = x.shape[0]

        # Forward pass
        z1, h1, z2 = self._forward(x)
        probs = self._softmax(z2)

        # Compute cross-entropy loss
        eps = 1e-10
        loss = -np.mean(np.sum(y * np.log(probs + eps), axis=-1))

        # Backward pass
        dz2 = (probs - y) / batch_size

        # Gradients for layer 2
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Backprop through layer 1
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * self._relu_grad(z1)

        # Gradients for layer 1
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        return float(loss)
