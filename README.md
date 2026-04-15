# FabricPC

**A flexible, performant predictive coding library**

FabricPC implements predictive coding networks using a clean abstraction of:
- **Nodes**: State variables (latents), projection functions, and activations
- **Wires**: Connections (edges) between nodes in the model architecture
- **Updates**: Iterative inference and local learning algorithms

Uses JAX for GPU acceleration and local (node-level) automatic differentiation.

## About Predictive Coding
Predictive coding (PC) is a biologically-inspired framework for perception and learning in the brain. It posits that the brain continuously generates predictions about sensory inputs and updates its internal representations based on local prediction errors. 
PC performs bilevel optimization: an inner loop infers latent activations by minimizing prediction errors, while an outer loop updates weights via local Hebbian-like rules. Under certain conditions, this process is equivalent to backpropagation. While currently slower than backprop on standard hardware, PC offers:
- Potential for faster inference on neuromorphic hardware
- Natural handling of recurrent and arbitrary graph architectures
- Associative memory capabilities
- Potential novel plasticity rules for continual learning

There are various flavors of PC. FabricPC provides a graph-based implementation that focuses on principles:
- Local (Hebbian) learning rules
- Parallel processing of nodes
- Modularity of components
- Arbitrary architectures
- Scalability with JAX
- Extensibility for research
 
## Quick Start

Create a virtual environement with python 3.12.x (higher versions may not work with Aim experiment tracking)
```bash
# Install in editable mode (recommended for development, and running examples)
pip install -e ".[all]"

# Install pre-commit hooks for code quality
pre-commit install     

# Start the Aim visualization server (optional)
aim up

# Run an example
python examples/mnist_demo.py
```

## Features
- Modular node and wire abstractions for flexible model construction
- Inherently supports arbitrary architectures: feedforward, recurrent, skip connections, etc.
- Support for various node types: Linear, Conv1D/2D/3D (planned), Transfomers (in progress)
- Local automatic differentiation for efficient inference and learning
- JAX backend for GPU acceleration and scalability

## Contributions
Contributions are welcome! Please open issues or pull requests on the GitHub repository.
- Develop on a branch using convention "feature/your_feature_name"
- All demos must match baseline results or explain divergence and test suites must pass on rebased PR.
- Write unit tests and docstrings for new code
- Use the pre-commit hooks for PEP8 style and code quality (run `pre-commit install` once after cloning!)
- Follow the rebase instructions in `docs/rebasing_feature_branch.md` before opening a PR.

This is a research-first project.
- APIs may change frequently until v1.0 release.
- Any breaking changes are documented in the changelog.

## License
This project is licensed under the [MIT License](LICENSE).


## Building a model
A model consists of structure and parameters.
```python
from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.inference import InferenceSGD

layer1 = Linear(shape=(784,), name="input")
layer2 = Linear(shape=(256,), name="hidden")
layer3 = Linear(shape=(10,), name="output")

structure = graph(
  nodes=[layer1, layer2, layer3],
  edges=[Edge(layer1, layer2.slot("in")),
         Edge(layer2, layer3.slot("in"))
  ],
  task_map=TaskMap(x=layer1, y=layer3),
  inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)
params = initialize_params(structure, rng_key)
```

## Extending FabricPC

### Custom Nodes

Create custom node types by subclassing `NodeBase`. Implement the `forward()` and `initialize_params()` methods. Nodes have a single output. Define slots for incoming connections. Slots are named arguments of the node's transfer function and are referenced in edges when building the graph.

See `examples/custom_node.py` for a complete Conv2D implementation.

### Inference Algorithms
Create custom inference algorithms by subclassing `InferenceBase` and implementing the `compute_new_latent()` method.

### Learning Algorithms
Weight learning loop algorithm abstraction is planned for a future release. Optimizer chains are fully supported with Optax and can be used directly in the training loop.

### Custom Initializers
Create custom initializers by subclassing `StateInitializerBase` for latent state initialization and implementing the `initialize()` method.

Graph-aware weight initializers are in progress and will be added in a future release.

Node level initializers extend `InitializerBase` and implement `initialize_weights()` method. These are agnostic to acting on node state or parameters.

## Shape Conventions

 All shapes use batch-first, channels-last format (NHWC, NLC, NDHWC) and the batch size is not included in node shape definitions.

 - Consistent with JAX's default conv behavior
 - Linear: shape=(features,) - e.g., (128,) for 128-dimensional vector
 - 1D Conv: shape=(seq_len, channels) - e.g., (100, 32) for 100 timesteps, 32 channels
 - 2D Conv: shape=(H, W, C) - e.g., (28, 28, 64) for 28x28 image, 64 channels (NHWC)
 - 3D Conv: shape=(D, H, W, C) - e.g., (32, 32, 32, 16) for 3D volume

Conv Node Shape Flow (Future Reference)

 - Input:  (batch, H_in, W_in, C_in)   e.g., (32, 28, 28, 1)
 - Kernel: (kH, kW, C_in, C_out)       e.g., (3, 3, 1, 64)
 - Output: (batch, H_out, W_out, C_out) e.g., (32, 26, 26, 64)