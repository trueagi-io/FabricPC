# Writing Custom Nodes

FabricPC's built-in nodes cover many common use cases, but you may need custom nodes for specialized computations. This guide walks through creating your own node types.

## When to Create a Custom Node

Built-in nodes include:
- `Linear`: Fully-connected layers
- `IdentityNode`: Passthrough nodes
- `StorkeyHopfield`: Associative memory
- `TransformerBlock`: Multi-head attention and feedforward

Create a custom node when you need:
- Convolutional layers (Conv1D, Conv2D, Conv3D)
- Gating mechanisms (LSTM, GRU)
- Custom transfer functions
- Specialized domain-specific projections
- Any computation not covered by built-in nodes

## The NodeBase Contract

All nodes inherit from `NodeBase` and must implement three static methods:

1. **`get_slots()`**: Define input slots (the connection interface)
2. **`initialize_params()`**: Allocate and initialize weights and biases
3. **`forward()`**: Compute predictions, errors, and energy

These methods are static because FabricPC uses a functional JAX-based design. Node instances hold configuration; the static methods define pure functional transformations.

## Step-by-Step: Conv2D Node

Let's build a 2D convolutional node from scratch. This example is adapted from `examples/custom_node.py`.

### Step 1: Define the Class and Constructor

```python
import jax.numpy as jnp
import numpy as np
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, initialize

class Conv2DNode(NodeBase):
    def __init__(
        self,
        shape,
        name,
        kernel_size,
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(),
        **kwargs
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs
        )
```

Key points:
- Accept standard node parameters (`shape`, `name`, `activation`, etc.)
- Accept custom parameters (`kernel_size`, `stride`, `padding`)
- Pass everything to `super().__init__()` via `**kwargs`
- Custom parameters end up in `node_info.node_config` and are accessible in static methods

### Step 2: Define Input Slots

```python
@staticmethod
def get_slots():
    return {"in": SlotSpec(name="in", is_multi_input=True)}
```

This defines a single input slot named `"in"` that accepts multiple incoming edges. The `is_multi_input=True` flag means contributions from different source nodes will be summed.

For nodes with multiple distinct inputs:

```python
@staticmethod
def get_slots():
    return {
        "in": SlotSpec(name="in", is_multi_input=True),
        "mask": SlotSpec(name="mask", is_multi_input=False),
    }
```

### Step 3: Compute Weight Fan-In (Optional)

For muPC scaling, override `get_weight_fan_in()` to return the correct fan-in:

```python
@staticmethod
def get_weight_fan_in(source_shape, config):
    kernel_size = config.get("kernel_size", (1, 1))
    C_in = source_shape[-1]  # channels-last format
    return C_in * int(np.prod(kernel_size))
```

For a 3x3 kernel with 16 input channels: `fan_in = 16 * 3 * 3 = 144`.

If you don't override this method, the default implementation uses the flattened source shape, which works for fully-connected layers but not for convolutions.

### Step 4: Initialize Parameters

```python
@staticmethod
def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
    """
    Initialize convolutional kernels and biases.

    Args:
        key: JAX random key
        node_shape: Output shape of this node (H, W, C_out)
        input_shapes: Dict mapping edge_key -> source_shape
        weight_init: Weight initializer
        config: Node configuration dict

    Returns:
        NodeParams(weights=weights_dict, biases=biases_dict)
    """
    kernel_size = config.get("kernel_size", (1, 1))
    C_out = node_shape[-1]  # output channels

    weights = {}
    biases = {}

    for edge_key, source_shape in input_shapes.items():
        C_in = source_shape[-1]  # input channels

        # Kernel shape: (kH, kW, C_in, C_out)
        kernel_shape = (*kernel_size, C_in, C_out)

        # Initialize kernel weights
        subkey, key = jax.random.split(key)
        fan_in = C_in * int(np.prod(kernel_size))
        fan_out = C_out * int(np.prod(kernel_size))

        weights[edge_key] = initialize(
            subkey,
            kernel_shape,
            weight_init,
            fan_in=fan_in,
            fan_out=fan_out,
        )

        # Bias shape: (1, 1, 1, C_out) for broadcasting
        biases[edge_key] = jnp.zeros((1, 1, 1, C_out))

    return NodeParams(weights=weights, biases=biases)
```

Key points:
- `input_shapes` is a dict keyed by edge identifiers (e.g., `"conv1->conv2:in"`)
- Each edge gets its own weight matrix and bias vector
- Use the `initialize()` helper function with the provided `weight_init` initializer
- Return `NodeParams` with dicts of weights and biases

### Step 5: Implement Forward Computation

```python
@staticmethod
def forward(params, inputs, state, node_info):
    """
    Forward pass: compute convolution, activation, error, and energy.

    Args:
        params: NodeParams with weights and biases dicts
        inputs: Dict mapping edge_key -> input_array
        state: Current NodeState
        node_info: NodeInfo with configuration

    Returns:
        (total_energy, updated_state)
    """
    # Extract config
    kernel_size = node_info.node_config.get("kernel_size", (1, 1))
    stride = node_info.node_config.get("stride", (1, 1))
    padding = node_info.node_config.get("padding", "SAME")
    activation = node_info.activation

    # Accumulate convolution outputs from all input edges
    pre_activation = None

    for edge_key, input_array in inputs.items():
        kernel = params.weights[edge_key]
        bias = params.biases[edge_key]

        # Perform convolution
        # JAX uses (batch, H, W, C) format
        conv_out = jax.lax.conv_general_dilated(
            lhs=input_array,
            rhs=kernel,
            window_strides=stride,
            padding=padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )

        conv_out = conv_out + bias

        if pre_activation is None:
            pre_activation = conv_out
        else:
            pre_activation = pre_activation + conv_out

    # Apply activation function
    z_mu = type(activation).forward(pre_activation, activation.config)

    # Compute prediction error
    error = state.z_latent - z_mu

    # Update state
    state = state._replace(
        pre_activation=pre_activation,
        z_mu=z_mu,
        error=error,
    )

    # Compute energy using the energy functional
    node_class = type(node_info.activation).__bases__[0]
    state = node_class.energy_functional(state, node_info)

    # Return total energy and updated state
    total_energy = jnp.sum(state.energy)
    return total_energy, state
```

The forward method must:

1. **Accumulate inputs**: Sum contributions from all input edges (if multi-input slot)
2. **Apply transformation**: Perform the node's computation (convolution in this case)
3. **Apply activation**: Use the activation's static `forward()` method
4. **Compute error**: `error = z_latent - z_mu`
5. **Update state**: Replace relevant fields in the `NodeState` namedtuple
6. **Compute energy**: Call the node's energy functional
7. **Return**: `(total_energy, updated_state)`

### Step 6: Use the Custom Node

```python
from fabricpc.graph.builder import graph, Edge
from fabricpc.graph.task_map import TaskMap

# Create nodes
input_node = IdentityNode(shape=(28, 28, 1), name="input")

conv1 = Conv2DNode(
    shape=(26, 26, 16),  # VALID padding: 28-3+1 = 26
    kernel_size=(3, 3),
    stride=(1, 1),
    padding="VALID",
    name="conv1",
)

conv2 = Conv2DNode(
    shape=(24, 24, 32),
    kernel_size=(3, 3),
    stride=(1, 1),
    padding="VALID",
    name="conv2",
)

output_node = Linear(
    shape=(10,),
    flatten_input=True,
    name="output",
)

# Build graph
structure = graph(
    nodes=[input_node, conv1, conv2, output_node],
    edges=[
        Edge(source=input_node, target=conv1.slot("in")),
        Edge(source=conv1, target=conv2.slot("in")),
        Edge(source=conv2, target=output_node.slot("in")),
    ],
    task_map=TaskMap(x=input_node, y=output_node),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    scaling=MuPCConfig(),
)
```

## Useful Patterns

### FlattenInputMixin

For nodes that need dense (fully-connected) behavior with flattened inputs, use the `FlattenInputMixin`:

```python
from fabricpc.nodes.base import FlattenInputMixin

class MyDenseNode(FlattenInputMixin, NodeBase):
    @staticmethod
    def forward(params, inputs, state, node_info):
        # Flatten inputs to 2D: (batch, features)
        flat_inputs = MyDenseNode.flatten_input(inputs, node_info)

        # Perform dense computation
        pre_activation = MyDenseNode.compute_linear(flat_inputs, params, node_info)

        # Apply activation
        z_mu = type(node_info.activation).forward(pre_activation, node_info.activation.config)

        # ... rest of forward computation
```

The mixin provides:
- `flatten_input()`: Flattens multi-dimensional inputs to 2D
- `reshape_output()`: Reshapes outputs back to the node's shape
- `compute_linear()`: Performs `W @ x + b` with proper summing over edges

### Explicit Gradients

By default, JAX autodiff computes gradients during learning. For hand-coded gradients (e.g., for efficiency or control), override `forward_and_latent_grads()` and `forward_and_weight_grads()`:

```python
class MyNode(NodeBase):
    @staticmethod
    def forward_and_latent_grads(params, inputs, state, node_info):
        # Forward pass for inference (same as forward())
        return MyNode.forward(params, inputs, state, node_info)

    @staticmethod
    def forward_and_weight_grads(params, inputs, state, node_info, scaling_factors=None):
        # Forward pass with explicit gradient computation
        # ...
        # Return (total_energy, updated_state, custom_grads)
```

See `LinearExplicitGrad` in the source code for a complete example.

### Multiple Input Slots

For nodes with distinct input types (e.g., data and mask):

```python
@staticmethod
def get_slots():
    return {
        "in": SlotSpec(name="in", is_multi_input=True),
        "mask": SlotSpec(name="mask", is_multi_input=False),
    }

@staticmethod
def forward(params, inputs, state, node_info):
    # Access inputs by slot
    data_inputs = {k: v for k, v in inputs.items() if k.endswith(":in")}
    mask_inputs = {k: v for k, v in inputs.items() if k.endswith(":mask")}

    # Process separately
    # ...
```

### Stateful Nodes (e.g., Recurrent)

For nodes that maintain hidden state across time steps, add fields to the `NodeState` via `_replace()`:

```python
@staticmethod
def forward(params, inputs, state, node_info):
    # Access custom state
    hidden = state.custom_fields.get("hidden", jnp.zeros(node_info.shape))

    # Update hidden state (e.g., RNN-style)
    new_hidden = tanh(W_input @ input + W_hidden @ hidden + b)

    # Store updated hidden state
    custom_fields = state.custom_fields.copy()
    custom_fields["hidden"] = new_hidden

    state = state._replace(
        z_mu=new_hidden,
        custom_fields=custom_fields,
    )

    # ... rest of computation
```

## Testing Your Node

Write tests to verify:

1. **Shape correctness**: Output shapes match the node's declared shape
2. **Energy decreases**: Inference reduces free energy across steps
3. **Gradient flow**: Learning updates weights in the expected direction

Example test:

```python
import jax
import jax.numpy as jnp
from fabricpc.graph.builder import graph, Edge
from fabricpc.graph.task_map import TaskMap
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import initialize_params

def test_conv2d_energy_decreases():
    """Test that inference reduces energy for Conv2D node."""
    # Build graph
    input_node = IdentityNode(shape=(28, 28, 1), name="input")
    conv = Conv2DNode(shape=(28, 28, 16), kernel_size=(3, 3), name="conv")
    output = IdentityNode(shape=(28, 28, 16), name="output")

    structure = graph(
        nodes=[input_node, conv, output],
        edges=[
            Edge(source=input_node, target=conv.slot("in")),
            Edge(source=conv, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )

    # Initialize parameters
    rng_key = jax.random.PRNGKey(0)
    params = initialize_params(structure, rng_key)

    # Create dummy data
    batch_size = 4
    x = jax.random.normal(rng_key, (batch_size, 28, 28, 1))
    y = jax.random.normal(rng_key, (batch_size, 28, 28, 16))

    # Run inference and track energy
    # ... (see existing test examples in tests/)

    # Assert energy decreases
    assert final_energy < initial_energy
```

## Summary

Creating custom nodes involves:

1. **Subclass `NodeBase`**: Define your node class
2. **Implement `get_slots()`**: Specify input slots
3. **Implement `initialize_params()`**: Allocate and initialize weights/biases
4. **Implement `forward()`**: Compute predictions, errors, and energy
5. **Optional overrides**:
   - `get_weight_fan_in()`: For correct muPC scaling
   - `forward_and_latent_grads()` / `forward_and_weight_grads()`: For explicit gradients
6. **Test**: Verify shapes, energy convergence, and gradient flow

With these methods in place, your custom node integrates seamlessly with the rest of FabricPC's infrastructure: graph building, inference, learning, and scaling.
