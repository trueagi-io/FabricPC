# FabricPC Examples

This directory contains runnable examples for both baseline FabricPC usage and the migrated continual-learning experiments.

## Quick Start

From the repo root:

```bash
pip install -e ".[all]"
python examples/mnist_demo.py
```

If you want a fast end-to-end check of the continual stack:

```bash
python examples/split_mnist_quick.py
```

## Continual Learning Examples

### `split_mnist_quick.py`

Minimal smoke test for the continual-learning pipeline.

- Uses Split-MNIST
- Uses `training_mode="backprop"` for speed
- Trains only the first two tasks
- Good first verification after environment or code changes

### `split_mnist_continual.py`

Full Split-MNIST continual-learning experiment.

- Sequential digit-pair tasks
- Column-based or plain MLP architectures
- Accuracy-matrix, forgetting, and plot generation

Common invocations:

```bash
python examples/split_mnist_continual.py
python examples/split_mnist_continual.py --quick-smoke
python examples/split_mnist_continual.py --training-mode backprop
```

### `split_mnist_causal.py`

Split-MNIST experiment variant focused on causal/support diagnostics for the continual stack.

Use this when you want to inspect the causal-selection and audit-related behavior rather than just the base sequential trainer.

### `split_cifar100_continual.py`

Split-CIFAR-100 continual-learning experiment.

- Default setup is 20 tasks of 5 classes each
- Supports `--quick-smoke` for shorter verification runs
- Writes summaries, plots, and accuracy matrices under `../results/split_cifar100` by default

Common invocations:

```bash
python examples/split_cifar100_continual.py
python examples/split_cifar100_continual.py --quick-smoke
python examples/split_cifar100_continual.py --num-tasks 20 --classes-per-task 5
```

## Existing Core Demos

The rest of this directory contains the core upstream-style demos for MNIST, transformers, multi-GPU training, MU-PC, ResNet/CIFAR, and Storkey Hopfield experiments.

## Outputs

The continual-learning examples create run directories under `../results/` by default, with saved configs, JSON summaries, accuracy matrices, and plots.
