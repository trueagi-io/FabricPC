# FabricPC

**State-of-the-art predictive coding, made easy.**

FabricPC is an easy-to-use, high-performance open-source Python library for building and training predictive coding networks. It is designed to get researchers from idea to running experiment as fast as possible, eliminating algorithm boilerplate. A single directed edge between nodes is all that's needed to define a connection. Local derivatives are built in, following graph topology. The framework handles inference and learning dynamics automatically for whatever you write in a node's `forward()` method.

Built on JAX for GPU and multi-GPU acceleration with local (node-level) automatic differentiation.

## What It Does

FabricPC supports arbitrary graph topologies: feedforward, recurrent, skip connections, and cyclic architectures. Heterogeneous components such as linear, convolutional, and pooling nodes, transformer blocks, and Storkey-Hopfield associative memory coexist within the same energy-minimization graph. The same graph topology can be trained by predictive coding (`train_pcn`) or by backpropagation (`train_backprop`), so controlled PC-vs-backprop comparisons reuse one model definition instead of two. See `examples/PC_backprop_compare.py`.

Internally, everything is organized around three abstractions: nodes (state and computation), edges (connections between nodes), and updates (inference and learning algorithms).

## Installation

Clone this repo and `cd` into the project directory.

Create a virtual environment with Python 3.10–3.13. (The optional Aim experiment tracker in `[viz]`/`[all]` is Linux/macOS only and supports Python ≤3.12; on Windows or Python 3.13 it is skipped automatically and everything else installs normally.)

**Platform:** GPU acceleration requires **Linux** (x86_64 or aarch64) — JAX publishes CUDA wheels for Linux only. On native Windows or macOS, install CPU-only; for GPU on Windows use WSL2 (JAX marks WSL2 GPU support experimental).
```bash
# Verify your CUDA version
nvidia-smi
```

One command installs FabricPC, all optional deps, and a version-matched JAX backend. Pick the line that matches your platform:

```bash
pip install -U -e ".[all,cuda13]"   # GPU, CUDA 13
pip install -U -e ".[all,cuda12]"   # GPU, CUDA 12
pip install -U -e ".[all]"          # CPU only
```

See [`docs/user_guides/01_installation.md`](docs/user_guides/01_installation.md)
for details. For development from this checkout, synchronize the locked
environment, set up hooks, and run an example with `uv`:

```bash
uv sync

# Install pre-commit hooks for code quality
uv run pre-commit install

# Run an example
uv run python examples/mnist_demo.py
```

## Build a Model

Define the graph. Initialize the parameters. Start experimenting.

```python
from jax_setup import set_jax_flags_before_importing_jax
set_jax_flags_before_importing_jax()

import jax
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD

layer1 = Linear(shape=(784,), name="input")
layer2 = Linear(shape=(256,), name="hidden")
layer3 = Linear(shape=(10,), name="output")

structure = graph(
    nodes=[layer1, layer2, layer3],
    edges=[Edge(source=layer1, target=layer2.slot("in")),
           Edge(source=layer2, target=layer3.slot("in"))],
    task_map=TaskMap(x=layer1, y=layer3),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

rng_key = jax.random.PRNGKey(0)
params = initialize_params(structure, rng_key)
```

## Demos

The [`examples`](examples/) folder includes working demonstrations across image classification, sequence modeling, depth scaling (`examples/scaling/`), associative memory, and architectural probes. Start with [`mnist_demo.py`](examples/mnist_demo.py) (over 98% accuracy on MNIST) and explore from there:

- [`mnist_conv_demo.py`](examples/mnist_conv_demo.py) — convolutional MNIST classifier with `ConvNode` and `MaxPool`
- [`resnet18_cifar10_demo.py`](examples/resnet18_cifar10_demo.py) — ResNet-18 as a PC graph, with global average pooling
- [`transformer_v2_demo.py`](examples/transformer_v2_demo.py) — character- or BPE-level language modeling with text generation
- [`transformer_tuning.py`](examples/transformer_tuning.py) — two-phase hyperparameter search minimizing validation perplexity
- [`glucose_transformer.py`](examples/glucose_transformer.py) — GluMind-Uni-style glucose forecasting with predictive coding, backpropagation, or a controlled comparison
- [`glucose_transformer_tuning.py`](examples/glucose_transformer_tuning.py) — resumable, process-isolated Optuna tuning of glucose PC dynamics and architecture
- [`glucose_hopfield.py`](examples/glucose_hopfield.py) — glucose forecasting experiments with baseline, projection, and embedded Storkey-Hopfield variants

### Glucose forecasting

Run the glucose transformer with automatic GPU selection:

```bash
uv run glucose-transformer --mode pc --epochs 30 \
  --out_dir runs/glucose_transformer
```

Use `--mode backprop` for the backpropagation baseline or `--mode compare` to
run both methods on the same data and model geometry. Training validates after
every epoch, saves resumable checkpoints and the best parameters, and stops
early when validation MAE no longer improves. The run directory contains
`config.json`, `history.csv`, checkpoints, and final metrics.

The provisional PC defaults use the best configuration from the original
short, update-budget Optuna study on the Livia validation split: context 64,
depth 2, 1 attention head,
learning rate 0.00327532, 19 inference steps, inference step size 1.44358e-5,
inference norm clipping 1.0, gradient clipping 0.5, and weight initialization
standard deviation 0.0218619.

### Glucose Optuna search

The tuning command runs every trial in a fresh process so JAX compilation
caches and CUDA allocations are released when the trial exits. A shared
journal makes the study resumable, and the coordinator limits parallel workers
against a total GPU-memory budget:

```bash
uv run glucose-transformer-tune run \
  --run-dir runs/glucose_tuning_epochs_v2 \
  --study-name glucose_transformer_pc_epochs_v2 \
  --n-trials 40 \
  --max-workers 3 \
  --gpu-memory-budget-mib 8192 \
  --max-epochs 15 \
  --min-pruning-epochs 3 \
  --patience 4
```

The study minimizes validation MAE while searching context length (64 or 128),
transformer depth, attention heads, learning rate, gradient clipping, inference
step size and count, inference norm clipping, LR-decay horizon in epochs, and
weight initialization scale. Every trial uses the same seed, trains in complete
epochs, and reports validation MAE once per epoch. Hyperband begins pruning
after epoch 3; explicit energy and consecutive-validation-regression guards
remove unstable trials. Live histories are written under
`runs/glucose_tuning_epochs_v2/trials/`, worker logs under
`runs/glucose_tuning_epochs_v2/workers/`, and the final result to
`best_trial.json`. Each epoch records validation MAE and online training MAE,
including its batch-level standard deviation and minimum/maximum. An atomic
epoch checkpoint stores model and optimizer state, RNG, early-stop counters,
and history. Re-running the coordinator recovers dead running trials from
their latest checkpoints while retaining the shared Optuna journal.

## Documentation

User guides, API reference, and tutorials live in
[`docs/user_guides`](docs/user_guides/00_index.md). Development plans and
technical design documents are in [`docs/dev_plans`](docs/dev_plans/).
Glucose forecasting experiment plans, training results, and architecture notes
are grouped under [`docs/glucose_example`](docs/glucose_example/).

## Extending FabricPC

### Custom Nodes

Create custom node types by subclassing `NodeBase`. Implement the `get_slots()`, `initialize_params()`, and `forward()` methods. Nodes have a single output. Slots define incoming connections and are referenced in edges when building the graph.

See [`docs/user_guides/06_custom_nodes.md`](docs/user_guides/06_custom_nodes.md) for the node contract and a Conv2D teaching example (the production node is `fabricpc.nodes.ConvNode`).

## Contributing

Contributions are welcome! Please open issues or pull requests on the GitHub repository.
- Develop on a branch using the convention `username/your_feature_name`.
- Demos must match baseline results, or explain any divergence.
- The test suite must pass.
- Write unit tests and docstrings for new code.
- Use the pre-commit hooks for PEP8 style and code quality.
- Rebase before opening PR.

This is a research-first project.
- APIs may change frequently until the v1.0 release.
- Any breaking changes are documented in the changelog.

## Team

FabricPC is actively maintained by SingularityNET as part of the Artificial Superintelligence Alliance. Project lead: Dr. Matthew Behrend.

## License

This project is licensed under the [MIT License](LICENSE).
