# FabricPC Documentation

FabricPC is a flexible, performant predictive coding library built on JAX. It implements predictive coding networks using a graph-based abstraction of nodes, edges, and iterative inference with local learning rules.

## Getting Started

- [Installation](01_installation.md) — Install FabricPC and verify your setup
- [Quickstart](02_quickstart.md) — Train your first PC network on MNIST in 5 minutes

## Tutorials

Step-by-step guides for building and training predictive coding networks.

- [How Predictive Coding Works](03_how_predictive_coding_works.md) — PC concepts mapped to FabricPC code
- [Building Models](04_building_models.md) — Nodes, edges, graphs, and all node types
- [Initialization and Scaling](05_initialization_and_scaling.md) — Weight init, state init, and muPC scaling
- [Custom Nodes](06_custom_nodes.md) — Writing your own node types
- [Optimizers](07_optimizers.md) — Optax integration, chaining transforms, and natural gradients
- [Training and Evaluation](08_training_and_evaluation.md) — Training loops, evaluation, callbacks, and multi-GPU
- [Experiment Tracking](09_experiment_tracking.md) — Monitoring training with Aim dashboards

## API Reference

Comprehensive reference for all library components.

- [Nodes](10_api_nodes.md) — All node types with full constructor signatures
- [Activations and Energy Functionals](11_api_activations_and_energy.md) — Activation functions and energy formulations
- [Inference Algorithms](12_api_inference.md) — Inference loop algorithms
- [Initializers](13_api_initializers.md) — Weight and state initializers
- [Data Loaders](14_api_data.md) — Built-in datasets and custom data
- [Experiment Framework](15_api_experiments.md) — A/B experiments, statistics, and tuning

## Help

- [Troubleshooting](16_troubleshooting.md) — Common issues and FAQ
