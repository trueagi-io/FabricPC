# FabricPC

**A flexible, performant predictive coding library**

FabricPC implements predictive coding networks using a clean abstraction of:
- **Nodes**: State variables (latents), projection functions, and activations
- **Wires**: Connections (edges) between nodes in the model architecture
- **Updates**: Iterative inference and learning algorithms

Uses JAX for GPU acceleration and local (node-level) automatic differentiation.

## About Predictive Coding
Predictive coding (PC) is a biologically-inspired framework for perception and learning in the brain. It posits that the brain continuously generates predictions about sensory inputs and updates its internal representations based on local prediction errors. 
PC performs bilevel optimization: an inner loop infers latent activations by minimizing prediction errors, while an outer loop updates weights via local Hebbian-like rules. Under certain conditions, this process is equivalent to backpropagation. While currently slower than backprop on standard hardware, PC offers:
- Potential for faster inference on specialized hardware
- Natural handling of recurrent and arbitrary architectures
- Associative memory capabilities
- Potential novel plasticity rules for continual learning

There are various flavors of PC. FabricPC provides a graph-based implementation that focuses on principles:
- Local learning rules
- Parallel processing of nodes
- Expectation-maximization style inference.
- Modularity of components
- Arbitrary architectures
- Scalability with JAX
- Extensibility for research
 
## Quick Start
```bash
# Install in editable mode (recommended for development, and running examples)
pip install -e ".[dev,torch,viz]"

# Install pre-commit hooks for code quality
pre-commit install     

# Run an example
python examples/mnist_demo.py
```

## Features
- Modular node and wire abstractions for flexible model construction
- Inherently supports arbitrary architectures: feedforward, recurrent, skip connections, etc.
- Support for various node types: Linear, Conv1D/2D/3D (planned), Transfomers (planned)
- Local automatic differentiation for efficient inference and learning
- JAX backend for GPU acceleration and scalability

## Contributions
Contributions are welcome! Please open issues or pull requests on the GitHub repository.
- Develop on a branch using convention "feature/your_feature_name"
- All demos must match baseline results and test suites must pass before merging new code.
- Write unit tests and docstrings for new code
- Use pre-commit hooks for PEP8 style and code quality (run `pre-commit install` once after cloning!)
- Follow the rebase instructions in `docs/rebasing_feature_branch.md` before merging to main.

This is a research-first project.
- APIs may change frequently until v1.0 release.
- Any breaking changes are documented in the changelog.

## License: private until officially released. Please do not distribute.

## Extending FabricPC

### Custom Nodes

Create custom node types by subclassing `NodeBase`:

```python
from fabricpc.nodes import NodeBase, SlotSpec, register_node
from fabricpc.core.types import NodeParams

@register_node("my_node")  # Decorator to register node type
class MyNode(NodeBase):
    # Required: config schema for node-specific parameters
    CONFIG_SCHEMA = {
        "kernel_size": {"type": tuple, "required": True, "description": "Kernel dimensions"},
        "stride": {"type": tuple, "default": (1, 1)},
        "use_bias": {"type": bool, "default": True},
    }

    # Optional: override default energy/activation
    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}
    DEFAULT_ACTIVATION_CONFIG = {"type": "relu"}

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        # Initialize weights/biases for each input edge
        weights, biases = {}, {}
        for edge_key, in_shape in input_shapes.items():
            # ... initialize parameters
            pass
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        # Compute forward pass, return (energy, updated_state)
        pass
```

**Required methods:**
- `get_slots()`: Define input slots (single or multi-input)
- `initialize_params()`: Create weight/bias parameters
- `forward()`: Compute forward pass and energy

**Registration decorators by base class:**
| Base Class | Decorator | Import |
|------------|-----------|--------|
| `NodeBase` | `@register_node("my_node")` | `from fabricpc.nodes import register_node` |
| `EnergyFunctional` | `@register_energy("custom_energy")` | `from fabricpc.core.energy import register_energy` |
| `ActivationBase` | `@register_activation("custom_activation")` | `from fabricpc.core.activations import register_activation` |

**Required class attributes:**
- `CONFIG_SCHEMA`: Dict defining node parameters with types, defaults, and validation

**Schema field options:**
```python
{
    "field_name": {
        "type": int,              # Required: int, float, str, tuple, dict, list, bool
        "required": True,         # Field must be provided (no default)
        "default": 10,            # Default value if not provided
        "choices": ["a", "b"],    # Allowed values
        "description": "...",     # Documentation
    }
}
```

See `examples/custom_node.py` for a complete Conv2D implementation.

### External Package Registration

External packages can register extensions via entry points instead of decorators (auto-discovered on import):

**`pyproject.toml`:**
```toml
[project.entry-points."fabricpc.nodes"]
myconv2d = "my_package.nodes:MyConv2DNode"

[project.entry-points."fabricpc.energies"]
custom_energy = "my_package.energy:CustomEnergy"

[project.entry-points."fabricpc.activations"]
custom_act = "my_package.activations:CustomActivation"
```

## Shape Conventions

 All shapes use batch-first, channels-last format (NHWC, NLC, NDHWC):

 - Consistent with JAX's default conv behavior
 - Linear: shape=(features,) - e.g., (128,) for 128-dimensional vector
 - 1D Conv: shape=(seq_len, channels) - e.g., (100, 32) for 100 timesteps, 32 channels
 - 2D Conv: shape=(H, W, C) - e.g., (28, 28, 64) for 28x28 image, 64 channels (NHWC)
 - 3D Conv: shape=(D, H, W, C) - e.g., (32, 32, 32, 16) for 3D volume

Linear nodes flatten their inputs for transformation and then reshape their outputs to the specified shape.

Conv Node Shape Flow (Future Reference)

 - Input:  (batch, H_in, W_in, C_in)   e.g., (32, 28, 28, 1)
 - Kernel: (kH, kW, C_in, C_out)       e.g., (3, 3, 1, 64)
 - Output: (batch, H_out, W_out, C_out) e.g., (32, 26, 26, 64)