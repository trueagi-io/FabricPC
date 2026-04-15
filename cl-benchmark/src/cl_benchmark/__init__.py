"""
CL-Benchmark: Framework-Agnostic Continual Learning Benchmark

A standardized protocol for evaluating continual learning algorithms
that works with any deep learning framework (PyTorch, JAX, TensorFlow, NumPy).

Example usage:
    from cl_benchmark import get_dataset, BenchmarkRunner, BenchmarkConfig

    # Your model (any framework)
    class MyModel:
        def predict(self, x):
            return self.net(x)

        def train_on_task(self, task_id, train_data, epochs=1):
            for _ in range(epochs):
                for x, y in train_data.train_batches(256):
                    self.optimizer.step(x, y)

    # Run benchmark
    config = BenchmarkConfig(dataset_name="split-mnist", epochs_per_task=5)
    runner = BenchmarkRunner(config)
    results = runner.evaluate(MyModel())

    print(f"Average Accuracy: {results.average_accuracy:.4f}")
    print(f"Forgetting: {results.forgetting_mean:.4f}")
"""

__version__ = "0.1.0"

# Protocols
from cl_benchmark.protocols import ContinualModel, ContinualDataset, TaskData

# Dataset registry
from cl_benchmark.datasets import get_dataset, register_dataset, list_datasets

# Metrics
from cl_benchmark.metrics import (
    compute_accuracy_matrix,
    compute_forgetting,
    compute_backward_transfer,
    compute_forward_transfer,
    compute_average_accuracy,
)

# Evaluation
from cl_benchmark.evaluation import BenchmarkConfig, BenchmarkRunner, BenchmarkResults

# Visualization
from cl_benchmark.visualization import (
    plot_accuracy_matrix,
    plot_forgetting_analysis,
    plot_learning_curves,
)

__all__ = [
    # Version
    "__version__",
    # Protocols
    "ContinualModel",
    "ContinualDataset",
    "TaskData",
    # Datasets
    "get_dataset",
    "register_dataset",
    "list_datasets",
    # Metrics
    "compute_accuracy_matrix",
    "compute_forgetting",
    "compute_backward_transfer",
    "compute_forward_transfer",
    "compute_average_accuracy",
    # Evaluation
    "BenchmarkConfig",
    "BenchmarkRunner",
    "BenchmarkResults",
    # Visualization
    "plot_accuracy_matrix",
    "plot_forgetting_analysis",
    "plot_learning_curves",
]
