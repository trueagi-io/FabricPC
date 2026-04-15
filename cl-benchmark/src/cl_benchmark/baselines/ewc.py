"""
Elastic Weight Consolidation (EWC) baseline for continual learning.

Implements EWC as described in Kirkpatrick et al. 2017:
"Overcoming catastrophic forgetting in neural networks"

Uses Fisher information to identify important parameters and
adds regularization to protect them during training on new tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cl_benchmark.baselines.base import BaselineModel
from cl_benchmark.protocols import TaskData


class EWCModel(BaselineModel):
    """
    Elastic Weight Consolidation baseline.

    Protects important parameters from previous tasks by adding
    a quadratic penalty based on the Fisher information matrix:

        L_total = L_task + (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2

    where F_i is the Fisher information for parameter i and
    theta_i* is the optimal parameter value after previous tasks.

    Example:
        >>> from cl_benchmark import BenchmarkRunner, BenchmarkConfig
        >>> from cl_benchmark.baselines import EWCModel
        >>>
        >>> model = EWCModel(input_dim=784, num_classes=10, ewc_lambda=400)
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
        ewc_lambda: float = 400.0,
        fisher_samples: int = 200,
        seed: int = 42,
    ):
        """
        Initialize EWC model.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for gradient descent
            ewc_lambda: EWC regularization strength
            fisher_samples: Number of samples for Fisher estimation
            seed: Random seed for initialization
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._init_params()

        # EWC storage: list of (Fisher, optimal_params) for each task
        self._ewc_data: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []

    def _init_params(self) -> None:
        """Initialize model parameters."""
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
        self._ewc_data.clear()

    def _get_params(self) -> Dict[str, np.ndarray]:
        """Get current parameters as dictionary."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

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
        Train on a single task with EWC regularization.

        Args:
            task_id: Task identifier
            train_data: TaskData with training samples
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional arguments (ignored)

        Returns:
            Dictionary with training metrics
        """
        total_loss = 0.0
        total_ewc_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for x, y in train_data.train_batches(batch_size=batch_size, shuffle=True):
                loss, ewc_loss = self._train_step(x, y)
                total_loss += loss
                total_ewc_loss += ewc_loss
                num_batches += 1

        # After training, compute Fisher information and store parameters
        fisher = self._compute_fisher(train_data, batch_size)
        optimal_params = self._get_params()
        self._ewc_data.append((fisher, optimal_params))

        return {
            "loss": total_loss / max(num_batches, 1),
            "ewc_loss": total_ewc_loss / max(num_batches, 1),
            "epochs": epochs,
            "task_id": task_id,
        }

    def _compute_fisher(
        self,
        train_data: TaskData,
        batch_size: int,
    ) -> Dict[str, np.ndarray]:
        """
        Compute diagonal Fisher information matrix.

        Uses the squared gradients of the log-likelihood as an
        approximation to the Fisher information.

        Args:
            train_data: Task data for Fisher estimation
            batch_size: Batch size

        Returns:
            Dictionary of Fisher values for each parameter
        """
        fisher = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }

        samples_seen = 0
        for x, y in train_data.train_batches(batch_size=batch_size, shuffle=True):
            if samples_seen >= self.fisher_samples:
                break

            # Compute gradients for this batch
            grads = self._compute_gradients(x, y)

            # Accumulate squared gradients
            for key in fisher:
                fisher[key] += grads[key] ** 2

            samples_seen += x.shape[0]

        # Normalize by number of samples
        for key in fisher:
            fisher[key] /= max(samples_seen, 1)

        return fisher

    def _compute_gradients(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients of log-likelihood."""
        batch_size = x.shape[0]

        # Forward pass
        z1, h1, z2 = self._forward(x)
        probs = self._softmax(z2)

        # Gradient of cross-entropy
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

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def _compute_ewc_loss(self) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute EWC regularization loss and gradients.

        Returns:
            Tuple of (ewc_loss, ewc_gradients)
        """
        if not self._ewc_data:
            return 0.0, {
                "W1": np.zeros_like(self.W1),
                "b1": np.zeros_like(self.b1),
                "W2": np.zeros_like(self.W2),
                "b2": np.zeros_like(self.b2),
            }

        ewc_loss = 0.0
        ewc_grads = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }

        current_params = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

        for fisher, optimal_params in self._ewc_data:
            for key in ewc_grads:
                diff = current_params[key] - optimal_params[key]
                ewc_loss += 0.5 * np.sum(fisher[key] * diff**2)
                ewc_grads[key] += fisher[key] * diff

        return float(ewc_loss), ewc_grads

    def _train_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Single training step with EWC regularization.

        Args:
            x: Input batch (batch_size, input_dim)
            y: One-hot labels (batch_size, num_classes)

        Returns:
            Tuple of (task_loss, ewc_loss)
        """
        batch_size = x.shape[0]

        # Forward pass
        z1, h1, z2 = self._forward(x)
        probs = self._softmax(z2)

        # Compute cross-entropy loss
        eps = 1e-10
        task_loss = -np.mean(np.sum(y * np.log(probs + eps), axis=-1))

        # Compute task gradients
        dz2 = (probs - y) / batch_size
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * self._relu_grad(z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Compute EWC loss and gradients
        ewc_loss, ewc_grads = self._compute_ewc_loss()

        # Add EWC gradients (scaled by lambda)
        dW1 += self.ewc_lambda * ewc_grads["W1"]
        db1 += self.ewc_lambda * ewc_grads["b1"]
        dW2 += self.ewc_lambda * ewc_grads["W2"]
        db2 += self.ewc_lambda * ewc_grads["b2"]

        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        return float(task_loss), float(ewc_loss * self.ewc_lambda)
