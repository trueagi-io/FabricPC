# FabricPC

**State-of-the-art predictive coding, made easy.**

FabricPC is an easy-to-use, high-performance open-source Python library for building and training predictive coding networks. It is designed to get researchers from idea to running experiment as fast as possible, eliminating the boilerplate of the algorithm. A single directed edge between nodes is all that's needed to define a connection. Local derivatives are built in, following graph topology. The framework handles inference and learning dynamics automatically for whatever you write in the forward method of a node.

Built on JAX for GPU and multi-GPU acceleration with local (node-level) automatic differentiation.

## What It Does

FabricPC supports arbitrary graph topologies: feedforward, recurrent, skip connections, and cyclic architectures. Heterogeneous components such as linear nodes, transformer blocks, and Storkey-Hopfield associative memory coexist within the same energy-minimization graph. The same graph topology can be trained by predictive coding (`train_pcn`) or by backpropagation (`train_backprop`), so controlled PC-vs-backprop comparisons reuse one model definition instead of two. See `examples/PC_backprop_compare.py`.

Internally, everything is organized around three abstractions: nodes (state and computation), edges (connections between nodes), and updates (inference and learning algorithms).

## Installation
Clone this repo and cd to the project path.

Create a virtual environment with Python 3.10–3.13. (The optional Aim experiment tracker in `[viz]`/`[all]` supports Python ≤3.12 only; on Python 3.13 it is skipped automatically and everything else installs normally.)

**Platform:** GPU acceleration requires **Linux** (x86_64 or aarch64) — JAX publishes CUDA wheels for Linux only. On native Windows or macOS, install CPU-only; for GPU on Windows use WSL2 (JAX marks WSL2 GPU support experimental).
```bash
# Verify your cuda version
nvidia-smi

# One command: FabricPC + all optional deps + a version-matched JAX backend.
# GPU, CUDA 13:
pip install -U -e ".[all,cuda13]"
# GPU, CUDA 12:  pip install -U -e ".[all,cuda12]"
# CPU only:      pip install -U -e ".[all]"
#
# See docs/user_guides/01_installation.md for details.

# Install pre-commit hooks for code quality
pre-commit install

# Run an example
python examples/mnist_demo.py
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

The [`examples`](examples/) folder includes working demonstrations across image classification, sequence modeling, depth scaling (`examples/scaling/`), associative memory, and architectural probes. Start with [`mnist_demo.py`](examples/mnist_demo.py) (over 98% accuracy on MNIST) and explore from there.

## Documentation

User guides, API reference, and tutorials live in [`docs/user_guides`](docs/user_guides/00_index.md). Development plans and technical design documents are in [`docs/dev_plans`](docs/dev_plans/).
 
## Extending FabricPC

### Custom Nodes

Create custom node types by subclassing `NodeBase`. Implement the `forward()` and `initialize_params()` methods. Nodes have a single output. Define slots for incoming connections. Slots are named arguments of the node's `forward()` method and are referenced in edges when building the graph.

See `examples/custom_node.py` for a complete Conv2D implementation.

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
