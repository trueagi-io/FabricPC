# Building Models

This guide covers the core abstractions for building predictive coding networks in FabricPC: nodes, edges, and graphs. You'll learn how to define computational units, connect them together, and assemble them into complete models.

## Core Abstractions

### Nodes

Nodes are computational units with state and learnable parameters. Each node maintains:

- **State variables**:
  - `z_latent`: The latent state (optimized during inference)
  - `z_mu`: The prediction (output of the forward computation)
  - `error`: The prediction error (`z_latent - z_mu`)
  - `energy`: The free energy (computed from the error via the energy functional)

- **Configuration**:
  - `shape`: Output shape (excluding batch dimension)
  - `activation`: Activation function (e.g., `TanhActivation`, `ReLUActivation`)
  - `energy`: Energy functional (defaults to `GaussianEnergy`)
  - **Named input slots**: Interface points where edges can connect

- **Learnable parameters**: Weights and biases (varies by node type)

### Edges

Edges are directed connections from one node's output to another node's input slot. They define the flow of information through the network.

```python
Edge(source=source_node, target=target_node.slot("in"))
```

Each edge specifies:
- A source node (where information comes from)
- A target node and slot (where information goes)

### Graphs

The `GraphStructure` is an immutable container assembled by the `graph()` builder function. It contains:
- Topology information (nodes, edges, adjacency)
- Node execution order
- Configuration (inference algorithm, scaling strategy)
- Task map (mapping data to nodes)

## Defining Nodes

All nodes share a common constructor pattern:

```python
node = Linear(
    shape=(256,),              # output shape (excluding batch)
    activation=TanhActivation(),
    energy=GaussianEnergy(),   # default if omitted
    weight_init=MuPCInitializer(),
    name="hidden1",
)
```

Key parameters:
- `shape`: The output shape of the node, not including the batch dimension
- `activation`: The nonlinearity applied to pre-activation values
- `energy`: The energy functional used to compute free energy from prediction errors
- `weight_init`: Initialization strategy for learnable parameters
- `name`: Unique identifier for the node

## Node Types

FabricPC provides several built-in node types for different computational patterns.

### Linear

A weighted projection node: `z_mu = activation(W @ x + b)`.

**Input slots**: `"in"` (multi-input)

The Linear node applies a learned linear transformation followed by an activation function. When multiple edges connect to the `"in"` slot, their contributions are summed.

```python
from fabricpc.nodes import Linear
from fabricpc.core.activations import TanhActivation

hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    name="hidden1",
)
```

For dense layers that receive multi-dimensional input (e.g., images), set `flatten_input=True`:

```python
dense = Linear(
    shape=(128,),
    activation=ReLUActivation(),
    flatten_input=True,
    name="dense1",
)
```

### IdentityNode

A passthrough node with no learnable parameters: `z_mu = activation(x)`.

**Input slots**: `"in"` (multi-input)

Identity nodes are useful for:
- Input nodes (where data is clamped)
- Merge points in skip connections
- Non-learnable transformations

```python
from fabricpc.nodes import IdentityNode

pixels = IdentityNode(shape=(784,), name="pixels")
```

For muPC scaling purposes, identity nodes typically use `fan_in=1`.

### StorkeyHopfield

An associative memory node that blends a residual path with a learned projection:

```
z_mu = activation(x/(1+s) + (x @ W)*s/(1+s) + bias)
```

**Input slots**: `"in"` (single-input)

The `hopfield_strength` parameter `s` controls the blend between the identity path and the Hopfield projection. It is a learnable parameter wrapped in softplus to ensure positive values.

The energy functional includes both standard PC prediction error and Hopfield attractor energy, enabling the node to learn stable attractors for pattern completion and associative recall.

```python
from fabricpc.nodes import StorkeyHopfield

hopfield = StorkeyHopfield(
    shape=(128,),
    name="memory",
    hopfield_strength=1.0,
)

Edge(source=source_node, target=hopfield.slot("in"))
```

### SkipConnection

A passthrough node for residual/skip paths with no learnable parameters.

**Input slots**: `"in"` (multi-input, `is_variance_scalable=False`)

SkipConnection is functionally identical to IdentityNode — it sums inputs and passes them through. The key difference is that its slot has `is_variance_scalable=False` and `is_skip_connection=True`, which tells muPC to leave incoming edges unscaled (scale 1.0). This preserves the identity mapping that carries signal through deep residual networks.

Use SkipConnection for residual/skip paths. Use IdentityNode for summation points where all inputs are independent and should be variance-scaled.

```
prev ──→ Linear(h1) ──→ SkipConnection(res1) ──→ next
  │       (transform)     (sums at scale 1.0)
  │                              ↑
  └──────────────────────────────┘
           (identity skip, unscaled)
```

```python
from fabricpc.nodes import Linear, SkipConnection

linear = Linear(shape=(128,), activation=TanhActivation(),
                weight_init=MuPCInitializer(), name="h1")
skip = SkipConnection(shape=(128,), name="res1")

edges = [
    Edge(source=prev, target=linear.slot("in")),   # transform path (scaled)
    Edge(source=prev, target=skip.slot("in")),      # skip path (unscaled)
    Edge(source=linear, target=skip.slot("in")),    # transform -> sum (unscaled)
]
```

### LinearResidual

A linear transformation (residual) combined with a skip connection in a single PC node:

```
z_mu = activation(W @ x_in + b) + x_skip
```

**Input slots**:
- `"in"` (multi-input, `is_variance_scalable=True`): Receives the transform path. Has a weight matrix, scaled by muPC.
- `"skip"` (multi-input, `is_skip_connection=True`): Receives the identity skip path. No weight matrix, passes through at scale 1.0.

LinearResidual halves graph depth compared to the Linear + SkipConnection pattern (one PC node per residual block instead of two).

```
              ┌─────────────────────────────────────┐
              │         LinearResidual node         │
              │                                     │
prev ─────────┤  slot("in")  → W @ x + b → act()    │
              │                             ↓       │
              │                            (+) ──→ z_mu ──→ next
              │                             ↑       │
prev ─────────┤  slot("skip") ──────────────┘       │
              │   (identity, unscaled)              │
              └─────────────────────────────────────┘
```

```python
from fabricpc.nodes import LinearResidual

prev = stem
for i in range(num_blocks):
    res = LinearResidual(shape=(W,), activation=TanhActivation(),
                         weight_init=MuPCInitializer(), name=f"res{i}")
    edges += [
        Edge(source=prev, target=res.slot("in")),    # transform path
        Edge(source=prev, target=res.slot("skip")),   # identity skip
    ]
    prev = res
```

### TransformerBlock

A multi-head self-attention block with feedforward MLP.

**Input slots**:
- `"in"`: Token embeddings (required)
- `"mask"`: Attention mask (optional)

The TransformerBlock implements:
- Multi-head self-attention with Rotary Position Embeddings (RoPE)
- Residual connections
- Feedforward MLP

```python
from fabricpc.nodes import TransformerBlock

block = TransformerBlock(
    shape=(128,),           # d_model
    n_heads=8,
    d_model=128,
    d_ff=512,               # feedforward dimension
    name="attn",
)

# Connect token embeddings:
Edge(source=embeddings, target=block.slot("in"))

# Optional attention mask:
Edge(source=mask_node, target=block.slot("mask"))
```

### Decomposed Transformer (v2)

Fine-grained transformer components for deeper PC inference. Instead of a monolithic block, the transformer computation is broken into separate nodes:

- `EmbeddingNode`: Token and position embeddings
- `MhaResidualNode`: Multi-head attention with residual connection
- `LnMlp1Node`: Layer norm and first MLP layer
- `Mlp2ResidualNode`: Second MLP layer with residual connection
- `VocabProjectionNode`: Final projection to vocabulary

```
                         ┌───────────────────────────── one transformer block ────────────┐
                         │                                                                │
tokens → EmbeddingNode ──┼──→ MhaResidualNode(+) ──→ LnMlp1Node ──→ Mlp2ResidualNode(+) ──┼──→ VocabProjectionNode → logits
                         │    │      (skip)   ↑                     │      (skip)    ↑    │
                         │    └───────────────┘                     └────────────────┘    │
                         └────────────────────────────────────────────────────────────────┘
```

This decomposition allows PC inference to optimize each stage separately, potentially enabling more fine-grained credit assignment.

## Connecting Nodes with Edges

Edges define the flow of information between nodes:

```python
Edge(source=pixels, target=hidden.slot("in"))
```

### Slot Specification

Most nodes use the default `"in"` slot:

```python
Edge(source=hidden1, target=hidden2)  # implicit .slot("in")
```

Some nodes require explicit slot specification:

```python
# StorkeyHopfield uses "in" slot (single-input)
Edge(source=encoder, target=hopfield.slot("in"))

# TransformerBlock has "in" and "mask" slots
Edge(source=embeddings, target=block.slot("in"))
Edge(source=mask, target=block.slot("mask"))
```

### Multi-Input Slots

Slots marked as multi-input automatically sum contributions from multiple edges:

```python
# Both edges connect to the same slot
Edge(source=hidden1, target=output.slot("in"))
Edge(source=skip_connection, target=output.slot("in"))

# The output receives: hidden1_output + skip_connection_output
```

## The TaskMap

The `TaskMap` connects your data to nodes in the graph:

```python
from fabricpc.graph_initialization.task_map import TaskMap

task_map = TaskMap(x=input_node, y=output_node)
```

During training:
- Data with key `"x"` is clamped to `input_node.z_latent`
- Data with key `"y"` is clamped to `output_node.z_latent`

The task map tells FabricPC which nodes to clamp during supervised learning.

## Building the Graph

The `graph()` builder function assembles nodes, edges, and configuration into an immutable `GraphStructure`:

```python
from fabricpc.graph_initialization.builder import graph
from fabricpc.graph_initialization.task_map import TaskMap
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.mupc import MuPCConfig

structure = graph(
    nodes=[pixels, hidden1, hidden2, output],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=hidden2.slot("in")),
        Edge(source=hidden2, target=output.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=output),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    scaling=MuPCConfig(),  # optional, recommended for deep networks
)
```

Key parameters:
- `nodes`: List of all nodes in the graph
- `edges`: List of all edges connecting nodes
- `task_map`: Mapping from data keys to nodes
- `inference`: Inference algorithm configuration (e.g., `InferenceSGD`)
- `scaling`: Scaling strategy (e.g., `MuPCConfig` for deep networks)

## Graph Topologies

FabricPC supports arbitrary graph topologies beyond simple feedforward chains.

### Feedforward Chains

Standard layer-by-layer architecture:

```
input ──→ hidden1 ──→ hidden2 ──→ output
```

```python
nodes = [input, hidden1, hidden2, output]
edges = [
    Edge(source=input, target=hidden1.slot("in")),
    Edge(source=hidden1, target=hidden2.slot("in")),
    Edge(source=hidden2, target=output.slot("in")),
]
```

### Skip Connections

Edges that bypass layers. Multi-input slots automatically sum contributions:

```
input ──→ hidden1 ──→ sum_node ──→ output
  └──────────────────────┘
          (skip)
```

```python
# Direct connection from input to hidden2
Edge(source=hidden1, target=sum_node.slot("in"))
Edge(source=pixels, target=sum_node.slot("in"))

# output now receives:
# hidden1_output + pixels_output
```

ResNet-style architectures are straightforward:

```
input ──→ hidden1 ──→ hidden2(+) ──→ output
  └───────────────────────┘
          (skip)
```

```python
# Residual block: hidden2 receives both the transformed signal
# from hidden1 and the original signal from input
edges = [
    Edge(source=input, target=hidden1.slot("in")),
    Edge(source=hidden1, target=hidden2.slot("in")),
    Edge(source=input, target=hidden2.slot("skip")),  # skip connection
]
```

For identity pass though use `SkipConnection` or `LinearResidual` nodes. These nodes mark their skip slots as `is_variance_scalable=False`, preventing muPC from attenuating the identity path — which is essential for training networks with many residual blocks. See the [Initialization and Scaling guide](05_initialization_and_scaling.md#skip-connections-and-residual-networks) for details.

### Lateral Connections

Edges between nodes at the same or different depth level:

```
input ──→ path_a ──→ output_a
             ↕
input ──→ path_b ──→ output_b
```

```python
# Two parallel processing paths that communicate
Edge(source=path_a_node, target=path_b_node.slot("in"))
Edge(source=path_b_node, target=path_a_node.slot("in"))
```
In highly connected graphs, where each in-edge is intended to be transformed to a common embedding space, Linear node accepts multiple inputs for this purpose, allocating a weight matrix to each in-edge.

### Cyclic Graphs

Edges that create loops are supported:

```
input ──→ hidden1 ──→ hidden2 ──→ output
            ↑                   │ 
            └───────────────────┘
```

```python
# Cyclic connection
Edge(source=hidden1, target=hidden2.slot("in"))
Edge(source=hidden2, target=hidden1.slot("in"))
```

The builder will emit a warning about topological sort when cycles are detected. Cyclic graphs may require more inference steps for information to propagate around the loops and reach equilibrium.

## Shape Conventions

FabricPC uses batch-first, channels-last format for all tensors.

**Important**: The batch dimension is NOT included in node shape definitions.

### Examples

**Linear layers** (1D feature vectors):
```python
# Shape: (features,)
Linear(shape=(128,), name="hidden")
# Input/output: (batch, 128)
```

**2D images**:
```python
# Shape: (height, width, channels)
IdentityNode(shape=(28, 28, 1), name="mnist_input")
# Input/output: (batch, 28, 28, 1)
```

**Sequences**:
```python
# Shape: (seq_len, features)
TransformerBlock(shape=(100, 128), n_heads=8, d_model=128, d_ff=512, name="transformer")
# Input/output: (batch, 100, 128)
```

**3D data** (e.g., video, 3D convolutions):
```python
# Shape: (depth, height, width, channels)
node = CustomNode(shape=(16, 64, 64, 3), name="video_input")
# Input/output: (batch, 16, 64, 64, 3)
```

The batch dimension is always implicitly first and handled automatically by the framework.
