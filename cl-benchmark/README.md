# CL-Benchmark

A framework-agnostic continual learning benchmark and evaluation protocol.

## Overview

CL-Benchmark provides a standardized protocol for evaluating continual learning algorithms. It works with any deep learning framework (PyTorch, JAX, TensorFlow, or pure NumPy).

## Features

- **Framework-agnostic**: Works with any model that implements a simple interface
- **Built-in datasets**: Split-MNIST, Permuted-MNIST, Split-CIFAR10/100
- **Comprehensive metrics**: Accuracy matrix, forgetting, backward/forward transfer
- **Statistical tests**: Paired t-test, Wilcoxon, bootstrap confidence intervals
- **Visualization**: Accuracy heatmaps, forgetting analysis, method comparisons

## Installation

```bash
pip install cl-benchmark
```

Or install from source:

```bash
git clone https://github.com/dort/cl-benchmark.git
cd cl-benchmark
pip install -e .
```

## Quick Start

```python
from cl_benchmark import get_dataset, BenchmarkRunner, BenchmarkConfig

# Define your model (must implement predict() and train_on_task())
class MyModel:
    def __init__(self):
        # Initialize your model here
        pass

    def predict(self, x):
        # Return logits or probabilities: (batch_size, num_classes)
        return self.model(x)

    def train_on_task(self, task_id, train_data, epochs=1):
        for _ in range(epochs):
            for x, y in train_data.train_batches(batch_size=256):
                # Your training code here
                pass
        return {"loss": 0.0}  # Return training metrics

# Create benchmark configuration
config = BenchmarkConfig(
    dataset_name="split-mnist",
    epochs_per_task=5,
    num_runs=5,
)

# Run evaluation
runner = BenchmarkRunner(config)
results = runner.evaluate(MyModel())

# Print summary
results.print_summary()
```

## Model Protocol

Your model must implement two methods:

```python
class ContinualModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input batch (batch_size, ...)

        Returns:
            Predictions (batch_size, num_classes) - logits or probabilities
        """
        ...

    def train_on_task(
        self,
        task_id: int,
        train_data: TaskData,
        epochs: int = 1,
    ) -> Dict[str, Any]:
        """
        Args:
            task_id: Current task index
            train_data: TaskData with train_batches() and test_batches()
            epochs: Number of training epochs

        Returns:
            Dictionary of training metrics
        """
        ...
```

## Built-in Datasets

| Dataset | Tasks | Classes | Description |
|---------|-------|---------|-------------|
| `split-mnist` | 5 | 10 | MNIST split into digit pairs: (0,1), (2,3), ... |
| `permuted-mnist` | N | 10 | MNIST with different pixel permutations |
| `split-cifar10` | 5 | 10 | CIFAR-10 split into class pairs |
| `split-cifar100` | 10-20 | 100 | CIFAR-100 split into groups |

```python
from cl_benchmark import get_dataset

# Load Split-MNIST
dataset = get_dataset("split-mnist")

# Load Permuted-MNIST with 20 tasks
dataset = get_dataset("permuted-mnist", num_tasks=20)

# Iterate over tasks
for task in dataset:
    print(f"Task {task.task_id}: classes {task.classes}")
    for x, y in task.train_batches(batch_size=256):
        # x: (batch_size, 784) for flattened MNIST
        # y: (batch_size, 10) one-hot labels
        pass
```

## Metrics

### Accuracy Matrix

The accuracy matrix `A[i,j]` represents the accuracy on task `j` after training through task `i`.

```python
from cl_benchmark.metrics import compute_accuracy_matrix

# After training, evaluate on all tasks
accuracies = compute_accuracy_matrix(model, dataset, tasks_trained=5)
```

### Forgetting

Per-task forgetting measures the drop from peak accuracy:

```
F_t = max(A[t:, t]) - A[T, t]
```

```python
from cl_benchmark.metrics import compute_forgetting, compute_average_forgetting

forgetting = compute_forgetting(accuracy_matrix)
avg_forgetting = compute_average_forgetting(accuracy_matrix)
```

### Transfer Metrics

```python
from cl_benchmark.metrics import compute_backward_transfer, compute_forward_transfer

# Backward transfer (negative = forgetting)
bwt = compute_backward_transfer(accuracy_matrix)

# Forward transfer (requires baseline)
fwt = compute_forward_transfer(accuracy_matrix, baseline_accuracies)
```

## Visualization

```python
from cl_benchmark.visualization import (
    plot_accuracy_matrix,
    plot_forgetting_analysis,
    plot_method_comparison,
)

# Plot accuracy matrix heatmap
plot_accuracy_matrix(results.get_mean_accuracy_matrix())

# Plot forgetting analysis
plot_forgetting_analysis(results.get_mean_accuracy_matrix())

# Compare multiple methods
results_dict = {"Method A": results_a, "Method B": results_b}
plot_method_comparison(results_dict, metric="average_accuracy")
```

## Statistical Testing

```python
from cl_benchmark.metrics import (
    paired_t_test,
    wilcoxon_signed_rank,
    bootstrap_confidence_interval,
    cohens_d,
)

# Compare two methods
t_stat, p_value = paired_t_test(method_a_scores, method_b_scores)

# Non-parametric test
stat, p_value = wilcoxon_signed_rank(method_a_scores, method_b_scores)

# Confidence interval
mean, lower, upper = bootstrap_confidence_interval(scores, confidence=0.95)

# Effect size
d = cohens_d(method_a_scores, method_b_scores)
```

## Custom Datasets

Register your own datasets:

```python
from cl_benchmark.datasets import register_dataset
from cl_benchmark.protocols import TaskData

@register_dataset("my-dataset")
class MyDataset:
    def __init__(self, data_root="./data"):
        # Load your data
        pass

    @property
    def num_tasks(self) -> int:
        return 5

    @property
    def num_classes(self) -> int:
        return 10

    def get_task(self, task_id: int) -> TaskData:
        return TaskData(
            task_id=task_id,
            classes=(0, 1),
            train_images=...,
            train_labels=...,
            test_images=...,
            test_labels=...,
        )

    def __iter__(self):
        for i in range(self.num_tasks):
            yield self.get_task(i)
```

## License

MIT License
