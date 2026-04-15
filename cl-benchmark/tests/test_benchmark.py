"""
Tests for cl_benchmark package.

Tests core functionality with a mock model to verify
the evaluation protocol works correctly.
"""

import numpy as np
import pytest

from cl_benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkResults,
    ContinualModel,
    TaskData,
    get_dataset,
)
from cl_benchmark.baselines import NaiveModel, EWCModel, ReplayBuffer
from cl_benchmark.metrics import (
    compute_accuracy_matrix,
    compute_forgetting,
    compute_average_forgetting,
    compute_backward_transfer,
    compute_average_accuracy,
)


class MockModel:
    """Simple mock model for testing the evaluation protocol."""

    def __init__(self, num_classes: int = 10, accuracy: float = 0.9):
        """
        Args:
            num_classes: Number of output classes
            accuracy: Simulated accuracy (used to generate predictions)
        """
        self.num_classes = num_classes
        self.accuracy = accuracy
        self.tasks_trained = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return probabilities that give desired accuracy."""
        batch_size = x.shape[0]
        # Return uniform predictions (random guessing as baseline)
        probs = np.ones((batch_size, self.num_classes)) / self.num_classes
        return probs

    def train_on_task(self, task_id: int, train_data: TaskData, epochs: int = 1):
        """Record that training happened."""
        self.tasks_trained.append(task_id)
        return {"loss": 0.1, "task_id": task_id, "epochs": epochs}


class TestTaskData:
    """Tests for TaskData class."""

    def test_train_batches(self):
        """Test training batch iteration."""
        n_samples = 100
        task = TaskData(
            task_id=0,
            classes=(0, 1),
            train_images=np.random.randn(n_samples, 784).astype(np.float32),
            train_labels=np.eye(10)[np.random.randint(0, 2, n_samples)].astype(
                np.float32
            ),
            test_images=np.random.randn(20, 784).astype(np.float32),
            test_labels=np.eye(10)[np.random.randint(0, 2, 20)].astype(np.float32),
        )

        batch_size = 32
        total_samples = 0
        for x, y in task.train_batches(batch_size=batch_size):
            assert x.shape[0] == y.shape[0]
            assert x.shape[0] <= batch_size
            assert x.shape[1] == 784
            assert y.shape[1] == 10
            total_samples += x.shape[0]

        assert total_samples == n_samples

    def test_test_batches(self):
        """Test test batch iteration."""
        n_train = 100
        n_test = 50
        task = TaskData(
            task_id=0,
            classes=(0, 1),
            train_images=np.random.randn(n_train, 784).astype(np.float32),
            train_labels=np.eye(10)[np.random.randint(0, 2, n_train)].astype(
                np.float32
            ),
            test_images=np.random.randn(n_test, 784).astype(np.float32),
            test_labels=np.eye(10)[np.random.randint(0, 2, n_test)].astype(np.float32),
        )

        total_samples = 0
        for x, y in task.test_batches(batch_size=16):
            total_samples += x.shape[0]

        assert total_samples == n_test


class TestMetrics:
    """Tests for metric computations."""

    def test_compute_forgetting(self):
        """Test forgetting computation."""
        # Create a matrix where accuracy decreases over time
        # A[i,j] = accuracy on task j after training task i
        matrix = np.array(
            [
                [0.95, 0.0, 0.0, 0.0, 0.0],  # After task 0
                [0.80, 0.93, 0.0, 0.0, 0.0],  # After task 1
                [0.70, 0.85, 0.92, 0.0, 0.0],  # After task 2
                [0.60, 0.75, 0.88, 0.94, 0.0],  # After task 3
                [0.50, 0.65, 0.82, 0.90, 0.91],  # After task 4 (final)
            ]
        )

        forgetting = compute_forgetting(matrix)

        # Forgetting for task 0: max(0.95) - 0.50 = 0.45
        assert np.isclose(forgetting[0], 0.45, atol=0.01)

        # Forgetting for task 4: max(0.91) - 0.91 = 0.0 (no forgetting yet)
        assert np.isclose(forgetting[4], 0.0, atol=0.01)

    def test_compute_backward_transfer(self):
        """Test BWT computation."""
        matrix = np.array(
            [
                [0.90, 0.0, 0.0],
                [0.85, 0.88, 0.0],
                [0.80, 0.82, 0.90],
            ]
        )

        bwt = compute_backward_transfer(matrix)

        # BWT = (1/2) * ((A[2,0] - A[0,0]) + (A[2,1] - A[1,1]))
        # = (1/2) * ((0.80 - 0.90) + (0.82 - 0.88))
        # = (1/2) * (-0.10 + -0.06) = -0.08
        expected = (1 / 2) * ((0.80 - 0.90) + (0.82 - 0.88))
        assert np.isclose(bwt, expected, atol=0.01)

    def test_compute_average_accuracy(self):
        """Test average accuracy computation."""
        matrix = np.array(
            [
                [0.90, 0.0, 0.0],
                [0.85, 0.88, 0.0],
                [0.80, 0.82, 0.90],
            ]
        )

        avg_acc = compute_average_accuracy(matrix)

        # Average of final row: (0.80 + 0.82 + 0.90) / 3
        expected = (0.80 + 0.82 + 0.90) / 3
        assert np.isclose(avg_acc, expected, atol=0.01)

    def test_average_forgetting(self):
        """Test average forgetting computation."""
        matrix = np.array(
            [
                [0.95, 0.0, 0.0],
                [0.85, 0.93, 0.0],
                [0.75, 0.83, 0.91],
            ]
        )

        avg_fgt = compute_average_forgetting(matrix)

        # Forgetting: [0.20, 0.10, 0.0]
        # Average of first T-1: (0.20 + 0.10) / 2 = 0.15
        forgetting = compute_forgetting(matrix)
        expected = np.mean(forgetting[:-1])
        assert np.isclose(avg_fgt, expected, atol=0.01)


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_add_and_sample(self):
        """Test adding samples and sampling from buffer."""
        buffer = ReplayBuffer(max_samples_per_task=100, seed=42)

        # Create fake task data
        task = TaskData(
            task_id=0,
            classes=(0, 1),
            train_images=np.random.randn(200, 784).astype(np.float32),
            train_labels=np.eye(10)[np.random.randint(0, 2, 200)].astype(np.float32),
            test_images=np.random.randn(50, 784).astype(np.float32),
            test_labels=np.eye(10)[np.random.randint(0, 2, 50)].astype(np.float32),
        )

        buffer.add_task(task_id=0, task_data=task)

        # Should have subsampled to 100
        assert len(buffer) == 100

        # Sample from buffer
        result = buffer.sample(batch_size=32)
        assert result is not None
        x, y = result
        assert x.shape == (32, 784)
        assert y.shape == (32, 10)

    def test_exclude_task(self):
        """Test sampling with task exclusion."""
        buffer = ReplayBuffer(max_samples_per_task=50, seed=42)

        for task_id in range(3):
            task = TaskData(
                task_id=task_id,
                classes=(task_id * 2, task_id * 2 + 1),
                train_images=np.random.randn(100, 784).astype(np.float32),
                train_labels=np.eye(10)[np.random.randint(0, 2, 100)].astype(
                    np.float32
                ),
                test_images=np.random.randn(20, 784).astype(np.float32),
                test_labels=np.eye(10)[np.random.randint(0, 2, 20)].astype(np.float32),
            )
            buffer.add_task(task_id=task_id, task_data=task)

        # Sample excluding task 1
        result = buffer.sample(batch_size=32, exclude_task=1)
        assert result is not None

        # Buffer should have tasks 0, 1, 2
        assert buffer.get_task_ids() == [0, 1, 2]


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BenchmarkConfig(dataset_name="split-mnist")

        assert config.dataset_name == "split-mnist"
        assert config.epochs_per_task == 5  # Default is 5
        assert config.num_runs == 5  # Default is 5
        assert len(config.seeds) == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = BenchmarkConfig(
            dataset_name="split-mnist",
            epochs_per_task=10,
            num_runs=3,
            batch_size=128,
            seeds=[100, 101, 102],  # Explicit seeds
        )

        assert config.epochs_per_task == 10
        assert config.num_runs == 3
        assert config.batch_size == 128
        assert len(config.seeds) == 3
        assert config.seeds[0] == 100

    def test_config_serialization(self):
        """Test config to_dict and from_dict."""
        config = BenchmarkConfig(
            dataset_name="split-mnist",
            epochs_per_task=5,
            num_runs=3,
        )

        d = config.to_dict()
        restored = BenchmarkConfig.from_dict(d)

        assert restored.dataset_name == config.dataset_name
        assert restored.epochs_per_task == config.epochs_per_task
        assert restored.num_runs == config.num_runs


class TestBenchmarkResults:
    """Tests for BenchmarkResults."""

    def test_results_summary(self):
        """Test computing summary metrics."""
        config = BenchmarkConfig(dataset_name="split-mnist", num_runs=2)
        results = BenchmarkResults(config=config, model_name="test")

        # Add some fake accuracy matrices
        matrix1 = np.array(
            [
                [0.90, 0.0, 0.0],
                [0.85, 0.88, 0.0],
                [0.80, 0.82, 0.90],
            ]
        )
        matrix2 = np.array(
            [
                [0.92, 0.0, 0.0],
                [0.87, 0.90, 0.0],
                [0.82, 0.84, 0.92],
            ]
        )

        results.accuracy_matrices = [matrix1, matrix2]
        results.compute_summary_metrics()

        assert results.accuracy_mean > 0
        assert results.accuracy_std >= 0
        assert results.forgetting_mean >= 0

    def test_results_serialization(self):
        """Test results to_dict and from_dict."""
        config = BenchmarkConfig(dataset_name="split-mnist")
        results = BenchmarkResults(config=config, model_name="test")
        results.accuracy_matrices = [np.eye(3)]
        results.compute_summary_metrics()

        d = results.to_dict()
        restored = BenchmarkResults.from_dict(d)

        assert restored.model_name == results.model_name
        assert len(restored.accuracy_matrices) == 1


class TestNaiveModel:
    """Tests for NaiveModel baseline."""

    def test_predict_shape(self):
        """Test prediction output shape."""
        model = NaiveModel(input_dim=784, num_classes=10)
        x = np.random.randn(32, 784).astype(np.float32)
        y = model.predict(x)

        assert y.shape == (32, 10)
        # Should be valid probabilities
        assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)

    def test_train_on_task(self):
        """Test training updates parameters."""
        model = NaiveModel(input_dim=784, num_classes=10, learning_rate=0.1)

        task = TaskData(
            task_id=0,
            classes=(0, 1),
            train_images=np.random.randn(100, 784).astype(np.float32),
            train_labels=np.eye(10)[np.random.randint(0, 2, 100)].astype(np.float32),
            test_images=np.random.randn(20, 784).astype(np.float32),
            test_labels=np.eye(10)[np.random.randint(0, 2, 20)].astype(np.float32),
        )

        W1_before = model.W1.copy()
        metrics = model.train_on_task(task_id=0, train_data=task, epochs=1)

        # Parameters should have changed
        assert not np.allclose(model.W1, W1_before)
        assert "loss" in metrics

    def test_reset(self):
        """Test model reset."""
        model = NaiveModel(input_dim=784, num_classes=10, seed=42)
        W1_initial = model.W1.copy()

        # Train to change parameters
        task = TaskData(
            task_id=0,
            classes=(0, 1),
            train_images=np.random.randn(50, 784).astype(np.float32),
            train_labels=np.eye(10)[np.random.randint(0, 2, 50)].astype(np.float32),
            test_images=np.random.randn(10, 784).astype(np.float32),
            test_labels=np.eye(10)[np.random.randint(0, 2, 10)].astype(np.float32),
        )
        model.train_on_task(task_id=0, train_data=task, epochs=1)

        # Reset should restore initial parameters
        model.reset()
        assert np.allclose(model.W1, W1_initial)


class TestEWCModel:
    """Tests for EWCModel baseline."""

    def test_ewc_regularization(self):
        """Test that EWC stores Fisher information."""
        model = EWCModel(input_dim=784, num_classes=10, ewc_lambda=100)

        task = TaskData(
            task_id=0,
            classes=(0, 1),
            train_images=np.random.randn(100, 784).astype(np.float32),
            train_labels=np.eye(10)[np.random.randint(0, 2, 100)].astype(np.float32),
            test_images=np.random.randn(20, 784).astype(np.float32),
            test_labels=np.eye(10)[np.random.randint(0, 2, 20)].astype(np.float32),
        )

        # Before training, no EWC data
        assert len(model._ewc_data) == 0

        model.train_on_task(task_id=0, train_data=task, epochs=1)

        # After training, should have Fisher + optimal params
        assert len(model._ewc_data) == 1


class TestIntegration:
    """Integration tests for the full benchmark pipeline."""

    def test_mock_model_evaluation(self):
        """Test running evaluation with mock model."""
        model = MockModel(num_classes=10)

        config = BenchmarkConfig(
            dataset_name="split-mnist",
            epochs_per_task=1,
            num_runs=1,
            verbose=False,
        )

        runner = BenchmarkRunner(config)
        results = runner.evaluate(model, model_name="mock")

        # Should have trained on all 5 tasks
        assert len(model.tasks_trained) == 5
        assert model.tasks_trained == [0, 1, 2, 3, 4]

        # Should have one accuracy matrix
        assert len(results.accuracy_matrices) == 1
        assert results.accuracy_matrices[0].shape == (5, 5)

        # Metrics should be computed
        assert results.accuracy_mean >= 0
        assert results.accuracy_mean <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
