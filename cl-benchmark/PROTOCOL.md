# Continual Learning Benchmark Protocol Specification

Version: 1.0.0

## Overview

This document specifies a standardized protocol for evaluating continual learning algorithms. The protocol is designed to be framework-agnostic and provides reproducible, comparable evaluations across different methods.

## 1. Definitions

### 1.1 Task Sequence

A continual learning benchmark consists of a sequence of T tasks: `{Task_0, Task_1, ..., Task_{T-1}}`.

Each task `Task_t` contains:
- A training set `D_t^train = {(x_i, y_i)}`
- A test set `D_t^test = {(x_i, y_i)}`
- A set of target classes `C_t`

Tasks are presented sequentially. When training on `Task_t`, only `D_t^train` is available (unless explicitly using replay or other memory mechanisms).

### 1.2 Model Interface

A model `M` must implement two operations:

```
predict(x) -> y_hat
    Input: x with shape (batch_size, ...)
    Output: y_hat with shape (batch_size, num_classes)

train_on_task(task_id, train_data, epochs) -> metrics
    Input: task identifier, training data, number of epochs
    Output: dictionary of training metrics
```

## 2. Evaluation Protocol

### 2.1 Training Phase

For each task `t` in sequence `0, 1, ..., T-1`:

1. **Train**: Call `model.train_on_task(t, Task_t, epochs)`
2. **Evaluate**: After training, evaluate on ALL tasks `0, 1, ..., t`
3. **Record**: Store accuracy for each evaluated task

### 2.2 Accuracy Matrix

The accuracy matrix `A` has shape `(T, T)` where:

```
A[i, j] = accuracy on Task_j after training through Task_i
```

The matrix is lower triangular (only `A[i, j]` where `j <= i` is defined).

**Evaluation procedure for `A[i, j]`:**
1. For each sample `(x, y)` in `D_j^test`:
2. Compute `y_hat = argmax(model.predict(x))`
3. `A[i, j] = mean(y_hat == argmax(y))`

### 2.3 Multiple Runs

For statistical validity, run the protocol multiple times (recommended: 5+ runs) with different random seeds. Report mean and standard deviation of all metrics.

## 3. Metrics

### 3.1 Average Accuracy

Final average accuracy across all tasks:

```
Average Accuracy = (1/T) * sum_{t=0}^{T-1} A[T-1, t]
```

### 3.2 Forgetting

Per-task forgetting measures the drop from peak accuracy to final accuracy:

```
F_t = max_{i >= t} A[i, t] - A[T-1, t]
```

Average forgetting (excluding the last task, which has no opportunity to be forgotten):

```
Average Forgetting = (1/(T-1)) * sum_{t=0}^{T-2} F_t
```

### 3.3 Backward Transfer (BWT)

Backward transfer measures how learning new tasks affects performance on previous tasks:

```
BWT = (1/(T-1)) * sum_{t=0}^{T-2} (A[T-1, t] - A[t, t])
```

- `BWT < 0`: Negative transfer (forgetting)
- `BWT > 0`: Positive transfer (learning new tasks improves old ones)
- `BWT = 0`: No transfer effect

### 3.4 Forward Transfer (FWT)

Forward transfer measures how knowledge from previous tasks helps learning new tasks:

```
FWT = (1/(T-1)) * sum_{t=1}^{T-1} (A[t-1, t] - B[t])
```

Where `B[t]` is the baseline accuracy on task `t` (e.g., random initialization performance).

- `FWT > 0`: Positive forward transfer
- `FWT < 0`: Negative forward transfer (interference)

### 3.5 Learning Curve Area

For each task `t`, the learning curve tracks `A[i, t]` for `i >= t`. The area under this curve provides a comprehensive view of both initial learning and retention.

## 4. Standard Datasets

### 4.1 Split-MNIST

- **Tasks**: 5
- **Classes**: 10 (digits 0-9)
- **Task composition**: `{(0,1), (2,3), (4,5), (6,7), (8,9)}`
- **Input shape**: (784,) or (28, 28)
- **Labels**: One-hot encoded (10,)

### 4.2 Permuted-MNIST

- **Tasks**: Configurable (typically 10-20)
- **Classes**: 10 (same digits each task)
- **Task composition**: Same classes, different pixel permutations
- **Input shape**: (784,)
- **Labels**: One-hot encoded (10,)

### 4.3 Split-CIFAR10

- **Tasks**: 5
- **Classes**: 10
- **Task composition**: `{(airplane, automobile), (bird, cat), ...}`
- **Input shape**: (3072,) or (32, 32, 3)
- **Labels**: One-hot encoded (10,)

### 4.4 Split-CIFAR100

- **Tasks**: 10 or 20
- **Classes**: 100
- **Task composition**: Groups of 10 or 5 classes
- **Input shape**: (3072,) or (32, 32, 3)
- **Labels**: One-hot encoded (100,)

## 5. Reporting Standards

### 5.1 Required Information

When reporting results, include:

1. **Dataset**: Name and configuration
2. **Model architecture**: Network structure, number of parameters
3. **Training**: Epochs per task, batch size, learning rate, optimizer
4. **Evaluation**: Number of runs, random seeds used
5. **Metrics**: Average Accuracy, Forgetting, BWT (with standard deviations)

### 5.2 Recommended Format

```
Dataset: Split-MNIST (5 tasks)
Model: MLP (784-256-10)
Training: 5 epochs/task, batch=256, lr=0.01, SGD
Runs: 5 (seeds: 0, 1, 2, 3, 4)

Results:
- Average Accuracy: 0.8534 +/- 0.0123
- Average Forgetting: 0.1245 +/- 0.0089
- Backward Transfer: -0.0987 +/- 0.0056
```

### 5.3 Visualization

Include at minimum:
1. Accuracy matrix heatmap
2. Per-task forgetting bar chart
3. Learning curves (optional but recommended)

## 6. Statistical Significance

### 6.1 Comparing Methods

When comparing two methods A and B:

1. **Paired t-test**: For normally distributed differences
2. **Wilcoxon signed-rank**: For non-parametric comparison
3. **Bootstrap confidence intervals**: For robust estimation

### 6.2 Effect Size

Report Cohen's d for practical significance:

```
d = (mean_A - mean_B) / pooled_std
```

Interpretation:
- `|d| < 0.2`: Negligible
- `0.2 <= |d| < 0.5`: Small
- `0.5 <= |d| < 0.8`: Medium
- `|d| >= 0.8`: Large

## 7. Reference Baselines

### 7.1 Naive Fine-tuning (Lower Bound)

- Train sequentially on each task
- No mechanism to prevent forgetting
- Establishes lower bound for continual learning performance

### 7.2 Joint Training (Upper Bound)

- Train on all tasks simultaneously (all data available)
- Violates continual learning constraints
- Establishes upper bound for achievable performance

### 7.3 Experience Replay

- Maintain buffer of samples from previous tasks
- Mix replay samples with current task training
- Common practical approach

### 7.4 EWC (Elastic Weight Consolidation)

- Compute Fisher information after each task
- Add quadratic penalty to protect important parameters
- Regularization-based approach

## 8. Implementation Notes

### 8.1 Data Format

- Images: NumPy arrays, dtype float32, range [0, 1]
- Labels: One-hot encoded NumPy arrays, dtype float32
- Batching: Provided through iterator interface

### 8.2 Reproducibility

For reproducible results:
1. Set numpy random seed before each run
2. Use deterministic data shuffling
3. Document framework-specific seeds (PyTorch, JAX, TensorFlow)

### 8.3 Evaluation Timing

Evaluate AFTER completing training on each task, not during training epochs. This ensures the accuracy matrix reflects the model state at task boundaries.

## Appendix A: Mathematical Notation

| Symbol | Description |
|--------|-------------|
| T | Total number of tasks |
| t | Task index (0-indexed) |
| A[i,j] | Accuracy on task j after training through task i |
| F_t | Forgetting for task t |
| BWT | Backward Transfer |
| FWT | Forward Transfer |
| D_t^train | Training data for task t |
| D_t^test | Test data for task t |
| C_t | Classes in task t |

## Appendix B: Changelog

### Version 1.0.0
- Initial protocol specification
- Standard metrics definitions
- Built-in dataset specifications
- Reporting guidelines
