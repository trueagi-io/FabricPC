# Initialization and Scaling

Proper initialization is critical for training deep predictive coding networks. This guide covers weight initialization strategies, the muPC scaling framework, and state initialization.

## Why Initialization Matters in PC

In predictive coding, the quality of initialization directly affects:

1. **Inference convergence**: If activations or errors are poorly scaled, the inference process (optimizing latent states to minimize energy) may fail to converge or get stuck.

2. **Gradient flow**: During learning, weight gradients are computed from prediction errors. If errors vanish or explode, learning stalls or diverges.

3. **Depth scaling**: In deep networks, these problems compound across layers. Without careful initialization, networks deeper than a few layers become untrainable.

Poor initialization causes:
- Vanishing or exploding activations
- Inference failing to converge (energy stays flat or oscillates)
- Learning stalling (zero or exploding weight gradients)
- Instability across network depths

## Weight Initializers

FabricPC provides several weight initialization strategies.

### XavierInitializer

Glorot initialization, designed to maintain variance across layers.

```python
from fabricpc.core.initializers import XavierInitializer

weight_init = XavierInitializer(gain=1.0)
```

Weights are drawn from:
```
std = gain * sqrt(2 / (fan_in + fan_out))
W ~ N(0, std^2)
```

Best for sigmoid and tanh activations in shallow to medium-depth networks.

### KaimingInitializer

He initialization, optimized for ReLU activations.

```python
from fabricpc.core.initializers import KaimingInitializer

weight_init = KaimingInitializer(gain=1.0)
```

Weights are drawn from:
```
std = gain * sqrt(2 / fan_in)
W ~ N(0, std^2)
```

Best for ReLU and other rectified activations. Does not account for fan_out, focusing on maintaining variance in the forward pass.

### MuPCInitializer

Unit variance initialization for use with muPC scaling.

```python
from fabricpc.core.initializers import MuPCInitializer

weight_init = MuPCInitializer(gain=1.0)
```

Weights are drawn from:
```
W ~ N(0, gain^2)
```

The width and depth scaling is handled separately by the muPC framework during the forward pass. This decoupling is key to muPC's effectiveness.

### NormalInitializer

Direct control over mean and standard deviation.

```python
from fabricpc.core.initializers import NormalInitializer

weight_init = NormalInitializer(mean=0.0, std=0.01)
```

Weights are drawn from:
```
W ~ N(mean, std^2)
```

Useful for custom initialization schemes or debugging.

### UniformInitializer

Uniform distribution over a specified range.

```python
from fabricpc.core.initializers import UniformInitializer

weight_init = UniformInitializer(minval=-0.1, maxval=0.1)
```

Weights are drawn from:
```
W ~ U(minval, maxval)
```

## muPC: The Recommended Default

**muPC (Maximal Update Parameterization for Predictive Coding)** is the recommended initialization and scaling strategy for networks with more than a few layers. It maintains O(1) activations, errors, and gradients across arbitrary graph topologies, from 2 layers to 100+ layers.

### Key Idea

muPC decouples weight initialization from forward pass scaling:

1. **Initialization**: Weights are drawn from a simple distribution (e.g., `N(0, 1)`) independent of network width or depth.

2. **Forward scaling**: Width and depth scaling is applied during the forward pass via per-edge scaling factors computed automatically from the graph topology.

This separation allows the same initialization to work across different network depths and widths without retuning hyperparameters.

### How to Enable muPC

Use `MuPCInitializer` for node weights and `MuPCConfig` in the graph builder:

```python
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.nodes import Linear
from fabricpc.core.activations import TanhActivation

hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    weight_init=MuPCInitializer(),
    name="h1",
)

structure = graph(
    nodes=[input_node, hidden, output_node],
    edges=[...],
    task_map=TaskMap(x=input_node, y=output_node),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    scaling=MuPCConfig(),
)
```

### The Scaling Formula

muPC computes a scaling factor `a` for each edge based on the target node type.

#### Hidden Nodes

For edges into hidden nodes:

```
a = gain / sqrt(fan_in * K)
```

where:
- `gain`: The activation function's variance gain (e.g., `sqrt(5/3)` for tanh, `sqrt(2)` for ReLU)
- `fan_in`: The input dimension of the weight matrix (source node's flattened shape)
- `K`: The target node's in-degree (number of incoming edges)

This scaling ensures:
- Width invariance: O(1) activations as `fan_in` increases
- Depth invariance: O(1) activations as the number of layers increases
- Multi-input handling: Correct normalization when multiple edges sum into a node

#### Output Nodes

For edges into output nodes (when `include_output=True`):

```
a = gain / (fan_in * sqrt(K))
```

The stronger `1/fan_in` scaling (instead of `1/sqrt(fan_in)`) is used because output nodes typically don't have downstream dependencies that would amplify variance. This scaling is optimal for regression tasks with identity activation and Gaussian energy (MSE loss).

### Top-Down Gradient Scaling

muPC also scales gradients flowing backward through the network during learning:

```
c_td = a * jacobian_gain
```

where:
- `a`: The forward scaling factor
- `jacobian_gain`: The activation's average Jacobian magnitude (e.g., `1/sqrt(3)` for tanh)

This scaling combines two corrections:

1. **Chain rule correction**: Compensates for the forward scaling `a` to maintain correct gradients.

2. **Jacobian compensation**: Accounts for the average slope of the activation function, preventing exponential gradient vanishing in deep networks with saturating activations (tanh, GELU, sigmoid).

Together, these factors ensure O(1) gradients across depth.

### The `include_output` Flag

The `MuPCConfig` constructor accepts an `include_output` parameter:

```python
# Exclude output nodes from muPC scaling (default)
scaling = MuPCConfig(include_output=False)

# Include output nodes in muPC scaling
scaling = MuPCConfig(include_output=True)
```

**When to use `include_output=False` (default)**:
- Classification tasks with softmax activation and cross-entropy energy
- Networks trained with standard backprop-style output layers
- When the output layer should use traditional initialization (Xavier/Kaiming)

**When to use `include_output=True`**:
- Regression tasks with identity activation and Gaussian energy (MSE)
- When you want consistent muPC scaling throughout the entire network
- When output dimensionality is large and variance scaling matters

### Result

With muPC, you can train networks from 2 to 100+ layers without tuning the learning rate, weight initialization, or inference parameters based on depth. The same hyperparameters work across different architectures.

Example: A 16-layer muPC network trains as stably as a 2-layer network with the same learning rate.

## State Initialization Strategies

After building the graph and initializing parameters, you need to set initial latent states (`z_latent`) for inference. FabricPC provides several strategies:

### FeedforwardStateInit (Default)

Runs a forward pass through the network and sets `z_latent = z_mu` for each node.

```python
from fabricpc.graph.state_initializer import FeedforwardStateInit

structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=...,
    state_init=FeedforwardStateInit(),
)
```

**Advantages**:
- Fast convergence for feedforward DAGs
- Latent states start close to the forward predictions
- Fewer inference steps needed to reach equilibrium

**Best for**: Feedforward and skip-connection architectures.

### GlobalStateInit

Initializes all nodes with the same distribution.

```python
from fabricpc.graph.state_initializer import GlobalStateInit
from fabricpc.core.initializers import NormalInitializer

structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=...,
    state_init=GlobalStateInit(initializer=NormalInitializer(mean=0.0, std=0.1)),
)
```

**Advantages**:
- Simple and consistent
- No forward pass needed

**Disadvantages**:
- May require more inference steps to converge
- Latent states start far from equilibrium

**Best for**: Debugging, baselines, or when forward initialization is not applicable (e.g., cyclic graphs).

### NodeDistributionStateInit

Each node uses its own `latent_init` initializer specified during node construction.

```python
from fabricpc.graph.state_initializer import NodeDistributionStateInit

hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    latent_init=NormalInitializer(mean=0.0, std=0.1),
    name="hidden",
)

structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=...,
    state_init=NodeDistributionStateInit(),
)
```

**Advantages**:
- Fine-grained control per node
- Can tailor initialization to specific node types

**Best for**: Custom architectures where different nodes need different initialization strategies.

## Choosing Your Initialization

### Recommended Default

For most use cases, especially deep networks:

```python
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.graph.state_initializer import FeedforwardStateInit

# Node definition
hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    weight_init=MuPCInitializer(),
    name="hidden",
)

# Graph construction
structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    scaling=MuPCConfig(),
    state_init=FeedforwardStateInit(),
)
```

### Alternative: Traditional Initialization

For shallow networks (2-3 layers) or when matching non-muPC baselines:

```python
from fabricpc.core.initializers import XavierInitializer

# Use Xavier for tanh/sigmoid
hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    weight_init=XavierInitializer(),
    name="hidden",
)

# Or Kaiming for ReLU
hidden = Linear(
    shape=(256,),
    activation=ReLUActivation(),
    weight_init=KaimingInitializer(),
    name="hidden",
)

# Build graph without muPC scaling
structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    # No scaling parameter
)
```

### Summary

| Network Depth | Recommended Initialization | Scaling | State Init |
|---------------|---------------------------|---------|------------|
| 2-3 layers | Xavier/Kaiming | None | FeedforwardStateInit |
| 4-10 layers | MuPCInitializer | MuPCConfig | FeedforwardStateInit |
| 10+ layers | MuPCInitializer | MuPCConfig | FeedforwardStateInit |
| Regression (MSE) | MuPCInitializer | MuPCConfig(include_output=True) | FeedforwardStateInit |
| Classification (CE) | MuPCInitializer | MuPCConfig(include_output=False) | FeedforwardStateInit |

When in doubt, use muPC. It's designed to work out-of-the-box for arbitrary depths and topologies.
