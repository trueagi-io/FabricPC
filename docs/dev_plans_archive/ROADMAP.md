# FabricPC Development Roadmap

**A JAX-native Predictive Coding Neural Network Framework**

This roadmap outlines the development plan to transform FabricPC into a production-ready, developer-friendly library for industrial-scale research and development of predictive coding neural networks.

---

## Executive Summary

FabricPC is built on solid foundations:
- Pure functional architecture compatible with JAX transformations
- Clean separation of concerns (core, nodes, graph, training)
- Extensible node-based design with slot connectivity
- Multi-GPU support via pmap
- Local Hebbian learning with Jacobian-based gradient computation

This roadmap addresses:
1. **New Node Types** - Transformer, Conv, Normalization layers (done)
2. **Advanced Solvers** - iPC, dynamic inference scheduling (in progress)
3. **N-Dimensional Tensor Support** - Beyond 2D matrices (done)
4. **Plugin Architecture** - Custom nodes without library modification (done)
5. **Hypergraph Support** - Hierarchical subgraph containers (planned)
6. **Documentation & Demos** - Compelling examples for community adoption (in progress)
7. **Performance Optimization** - Node parallelization
8. **ML Workflow** Flax.linen (nesting module pytrees and nn.scan a block across layers), CLU or Grain (data loaders), and Orbax (checkpointing).

---

## Phase 1: Foundation Strengthening

### 1.1 N-Dimensional Tensor Support

**Current State**: NodeState assumes 2D tensors `(batch, dim)`.

**Requirements**:
- [x] Extend `NodeState` to support arbitrary shapes `(batch, *spatial_dims, channels)`
- [x] Update `NodeInfo` with `shape: Tuple[int, ...]` instead of just `dim: int`
- [x] Modify inference loop to handle tensor reshaping correctly
- [x] Update `Linear` to optionally flatten inputs for matmul, reshape outputs
- [x] Ensure backward compatibility with existing 2D use cases
- [x] Add comprehensive tests for n-dim shapes
- [x] Plan for future convolutional node support using channels-last convention

 | Component             | Contains Batch? | Shape Pattern                       |
 |-----------------------|-----------------|-------------------------------------|
 | NodeInfo.shape        | NO              | (dim1, dim2, ...) e.g., (28, 28, 1) |
 | NodeParams.weights    | NO              | (in_features, out_features)         |
 | NodeParams.biases     | NO              | (1, *out_shape) for broadcasting    |
 | NodeState.z_latent    | YES             | (batch_size, *shape)                |
 | GraphState.batch_size | Stores it       | Scalar integer                      |
---

### 1.2 Plugin Architecture for Custom Nodes

**Goal**: Allow developers to register custom node types without modifying library code.

**Requirements**:
- [x] Create node registry with decorator-based registration - DEPRECATED the registry 2/26/2026
- [x] Support runtime node type discovery
- [x] Validate custom nodes implement required interface
- [x] Provide base mixins for common patterns

**Implementation**:
```python
# Usage in user code:
@register_node("my_custom_layer")
class MyCustomNode(NodeBase):
    ...
```

### 1.3 Configurable Energy Functionals

**Current State**: Hard-coded Gaussian energy `E = ½||z - μ||²`.

**Requirements**:
- [x] Define energy functional interface
- [x] Implement Gaussian (default), Bernoulli, CrossEntrpoy energies
- [x] Per-node energy configuration
- [x] Custom energy functional support

**Implementation**:
```python
# fabricpc/core/energy.py
class EnergyFunctional(ABC):
    """Base class for energy functionals."""

    @staticmethod
    @abstractmethod
    def energy(z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        """Compute energy E(z, μ)."""
        pass

    @staticmethod
    @abstractmethod
    def grad_latent(z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        """Compute ∂E/∂z_latent."""
        pass
class GaussianEnergy(EnergyFunctional):
    """Gaussian energy: E = ½||z - μ||²"""
class BernoulliEnergy(EnergyFunctional):
    """Bernoulli energy for binary outputs."""
class CrossEntropyEnergy(EnergyFunctional):
    """Cross-entropy energy for multi-class outputs."""
```

### 1.4 Graph construction system

Refactor graph construction with unified config validation, and node-level delegation.

- [x] Establish a unified registry and schema validation pattern - DEPRECATED the registry 2/26/2026
- [x] Delegate node construction from graph builder to node classes
- [x] Make all configurable objects (nodes, energy, activations) follow the same extensibility pattern
- [x] Define schemas at all levels: graph, node, subnode (energy, activation, slots)

Architectural advantages:
  - Separation of concerns: Node classes now handle their own construction
  - Extensibility: Custom node types can override from_config(), _build_slots(), or the resolve methods
  - Cleaner code: build_graph_structure() is now focused on graph-level concerns
  - Consistent pattern: All configurable objects (nodes, energy, activations) follow the same pattern with CONFIG_SCHEMA and validation delegation


## Phase 2: Core Node Types

### 2.1 Normalization Nodes

#### 2.1.1 LayerNorm Node

**Purpose**: Normalize across feature dimension, critical for transformers.

```python
# fabricpc/nodes/layer_norm.py
@register_node("layer_norm")
class LayerNormNode(NodeBase):
    """
    Layer Normalization: y = γ * (x - μ) / σ + β

    Normalizes across the last dimension (features).
    Slots: "in" (single input)
    Params: gamma (scale), beta (shift)
    """

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        eps = config.get("eps", 1e-5)
        node_dim = int(jnp.prod(jnp.array(node_shape)))
        return NodeParams(
            weights={"gamma": jnp.ones((1, node_dim))},
            biases={"beta": jnp.zeros((1, node_dim))}
        )

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = list(inputs.values())[0]
        eps = node_info.node_config.get("eps", 1e-5)

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + eps)

        gamma = params.weights["gamma"]
        beta = params.biases["beta"]

        z_mu = gamma * x_norm + beta
        pre_activation = z_mu  # No separate activation

        # Compute error and energy
        error = state.z_latent - z_mu
        energy = 0.5 * jnp.sum(error ** 2, axis=-1)

        # Update state
        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
            energy=energy,
            substructure={"mean": mean, "var": var, "x_norm": x_norm}
        )

        return jnp.sum(energy), state
```

#### 2.1.2 BatchNorm Node

```python
# fabricpc/nodes/batch_norm.py
@register_node("batch_norm")
class BatchNormNode(NodeBase):
    """
    Batch Normalization with running statistics.

    During training: normalize over batch dimension
    During inference: use running mean/var

    Note: Requires special handling in PC since batch statistics
    create cross-batch dependencies in Jacobian.
    """

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        node_dim = int(jnp.prod(jnp.array(node_shape)))
        return NodeParams(
            weights={
                "gamma": jnp.ones((1, node_dim)),
                "running_mean": jnp.zeros((1, node_dim)),
                "running_var": jnp.ones((1, node_dim)),
            },
            biases={"beta": jnp.zeros((1, node_dim))}
        )
```

### 2.2 Softmax Node

```python
# fabricpc/nodes/softmax.py
@register_node("softmax")
class SoftmaxNode(NodeBase):
    """
    Softmax activation node for classification outputs.

    y_i = exp(x_i) / Σ_j exp(x_j)

    For PC networks, provides probabilistic output interpretation.
    Typically used at output layer with cross-entropy energy.
    """

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def forward(params, inputs, state, node_info):
        """
        Forward pass returning (total_energy, updated_state).
        """
        node_out_shape = state.z_latent.shape

        # Sum all inputs (like Linear but without weights)
        pre_activation = jnp.zeros(node_out_shape)
        for edge_key, x in inputs.items():
            if edge_key in params.weights:
                pre_activation = pre_activation + jnp.matmul(x, params.weights[edge_key])
            else:
                pre_activation = pre_activation + x

        if "b" in params.biases:
            pre_activation = pre_activation + params.biases["b"]

        # Stable softmax
        x_max = jnp.max(pre_activation, axis=-1, keepdims=True)
        exp_x = jnp.exp(pre_activation - x_max)
        z_mu = exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

        # Compute error and energy
        error = state.z_latent - z_mu
        energy = 0.5 * jnp.sum(error ** 2, axis=-1)

        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
            energy=energy,
            substructure={}
        )

        return jnp.sum(energy), state
```

### 2.3 Convolutional Node

```python
# fabricpc/nodes/conv.py
@register_node("conv2d")
class Conv2DNode(NodeBase):
    """
    2D Convolutional node for spatial feature extraction.

    Input shape: (batch, height, width, in_channels)
    Output shape: (batch, out_height, out_width, out_channels)

    Supports:
    - Configurable kernel size, stride, padding
    - Multiple input channels from different sources
    - Efficient transposed convolution for gradient computation
    """

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        kernel_size = config.get("kernel_size", (3, 3))
        in_channels = config.get("in_channels", sum(s[-1] for s in input_shapes.values()))
        out_channels = config.get("out_channels", node_shape[-1])

        # Xavier initialization for conv kernels
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        fan_out = out_channels * kernel_size[0] * kernel_size[1]
        std = jnp.sqrt(2.0 / (fan_in + fan_out))

        kernel_shape = (*kernel_size, in_channels, out_channels)
        kernel = jax.random.normal(key, kernel_shape) * std

        return NodeParams(
            weights={"kernel": kernel},
            biases={"b": jnp.zeros((1, 1, 1, out_channels))}
        )

    @staticmethod
    def forward(params, inputs, state, node_info):
        """Forward pass returning (total_energy, updated_state)."""
        config = node_info.node_config
        stride = config.get("stride", (1, 1))
        padding = config.get("padding", "SAME")

        # Concatenate inputs along channel dimension
        x = jnp.concatenate(list(inputs.values()), axis=-1)

        # Convolution
        kernel = params.weights["kernel"]
        pre_activation = jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=stride,
            padding=padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )

        if "b" in params.biases:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation
        activation_fn, activation_deriv = get_activation(node_info.node_config["activation"])
        z_mu = activation_fn(pre_activation)

        # Compute error and energy
        error = state.z_latent - z_mu
        energy = 0.5 * jnp.sum(error ** 2, axis=(-3, -2, -1))  # Sum over H, W, C

        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
            energy=energy,
            substructure={}
        )

        return jnp.sum(energy), state

    """**Example Conv2D Node Config**:
    {
        "name": "conv1",
        "shape": (26, 26, 64),  # Output shape: (H, W, C) channels-last
        "type": "conv2d",
        "activation": {"type": "relu"},
        # Conv-specific config:
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": "valid",
        "weight_init": {"type": "he_normal"},
    }
    """

### 2.4 Attention Nodes

#### 2.4.1 Multi-Head Self-Attention

```python
# fabricpc/nodes/attention.py
@register_node("mha")
class MultiHeadAttentionNode(NodeBase):
    """
    Multi-Head Self-Attention for transformer architectures.

    Input: (batch, seq_len, embed_dim)
    Output: (batch, seq_len, embed_dim)

    Implements scaled dot-product attention with multiple heads.

    Slots:
    - "q": Query input (or uses "in" for self-attention)
    - "k": Key input (optional, defaults to query)
    - "v": Value input (optional, defaults to query)
    - "mask": Optional attention mask
    """

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec(name="in", is_multi_input=False),  # Self-attention input
            "mask": SlotSpec(name="mask", is_multi_input=False),  # Optional mask
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        num_heads = config.get("num_heads", 8)
        embed_dim = node_shape[-1]  # Last dimension is embed_dim
        head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        keys = jax.random.split(key, 4)
        std = 1.0 / jnp.sqrt(embed_dim)

        return NodeParams(
            weights={
                "W_q": jax.random.normal(keys[0], (embed_dim, embed_dim)) * std,
                "W_k": jax.random.normal(keys[1], (embed_dim, embed_dim)) * std,
                "W_v": jax.random.normal(keys[2], (embed_dim, embed_dim)) * std,
                "W_o": jax.random.normal(keys[3], (embed_dim, embed_dim)) * std,
            },
            biases={
                "b_q": jnp.zeros((1, 1, embed_dim)),
                "b_k": jnp.zeros((1, 1, embed_dim)),
                "b_v": jnp.zeros((1, 1, embed_dim)),
                "b_o": jnp.zeros((1, 1, embed_dim)),
            }
        )

    @staticmethod
    def forward(params, inputs, state, node_info):
        """Forward pass returning (total_energy, updated_state)."""
        config = node_info.node_config
        num_heads = config.get("num_heads", 8)

        # Get input (self-attention)
        x = inputs.get(list(inputs.keys())[0])  # Main input
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // num_heads

        # Linear projections
        Q = jnp.matmul(x, params.weights["W_q"]) + params.biases["b_q"]
        K = jnp.matmul(x, params.weights["W_k"]) + params.biases["b_k"]
        V = jnp.matmul(x, params.weights["W_v"]) + params.biases["b_v"]

        # Reshape for multi-head: (batch, seq, heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim)
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale  # (batch, heads, seq, seq)

        # Optional mask
        mask_key = [k for k in inputs if ":mask" in k]
        if mask_key:
            mask = inputs[mask_key[0]]
            scores = jnp.where(mask == 0, -1e9, scores)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, V)  # (batch, heads, seq, head_dim)

        # Reshape back: (batch, seq, embed_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

        # Output projection
        pre_activation = jnp.matmul(attn_output, params.weights["W_o"]) + params.biases["b_o"]

        # Apply activation (typically identity for attention output)
        activation_fn, activation_deriv = get_activation(node_info.node_config["activation"])
        z_mu = activation_fn(pre_activation)

        # Compute error and energy
        error = state.z_latent - z_mu
        energy = 0.5 * jnp.sum(error ** 2, axis=(-2, -1))  # Sum over seq_len and embed_dim

        # Store attention weights for gradient computation
        substructure = {
            "attn_weights": attn_weights,
            "Q": Q, "K": K, "V": V,
        }

        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
            energy=energy,
            substructure=substructure
        )

        return jnp.sum(energy), state
```

#### 2.4.2 Transformer Block

```python
# fabricpc/nodes/transformer.py
@register_node("transformer_block")
class TransformerBlock(NodeBase):
    """
    Complete Transformer Block with attention and FFN.

    Architecture:
    x → LayerNorm → MHA → + → LayerNorm → FFN → +
    └─────────────────────┘ └────────────────────┘
         (residual)              (residual)

    This is a composite node that internally manages its substructure.
    For hypergraph support, this could be decomposed into a subgraph.
    """

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        num_heads = config.get("num_heads", 8)
        embed_dim = node_shape[-1]  # Last dimension is embed_dim
        ffn_dim = config.get("ffn_dim")

        keys = jax.random.split(key, 8)
        std = 1.0 / jnp.sqrt(embed_dim)

        return NodeParams(
            weights={
                # Attention weights
                "W_q": jax.random.normal(keys[0], (embed_dim, embed_dim)) * std,
                "W_k": jax.random.normal(keys[1], (embed_dim, embed_dim)) * std,
                "W_v": jax.random.normal(keys[2], (embed_dim, embed_dim)) * std,
                "W_o": jax.random.normal(keys[3], (embed_dim, embed_dim)) * std,
                # FFN weights
                "W_ff1": jax.random.normal(keys[4], (embed_dim, ffn_dim)) * std,
                "W_ff2": jax.random.normal(keys[5], (ffn_dim, embed_dim)) * std,
                # LayerNorm parameters
                "ln1_gamma": jnp.ones((1, 1, embed_dim)),
                "ln2_gamma": jnp.ones((1, 1, embed_dim)),
            },
            biases={
                "b_q": jnp.zeros((1, 1, embed_dim)),
                "b_k": jnp.zeros((1, 1, embed_dim)),
                "b_v": jnp.zeros((1, 1, embed_dim)),
                "b_o": jnp.zeros((1, 1, embed_dim)),
                "b_ff1": jnp.zeros((1, 1, ffn_dim)),
                "b_ff2": jnp.zeros((1, 1, embed_dim)),
                "ln1_beta": jnp.zeros((1, 1, embed_dim)),
                "ln2_beta": jnp.zeros((1, 1, embed_dim)),
            }
        )
    """
    **Example Transformer Block Config**:
    {
        "name": "transformer_block_1",
        "shape": (128, 512),  # (seq_len, embed_dim)
        "type": "transformer_block",
        "activation": {"type": "gelu"},
        # Transformer-specific config:
        "num_heads": 8,
        "ff_dim": 2048,
        "dropout_rate": 0.1,
        "pre_norm": True,
    }
    # The config is already passed to `initialize_params(key, node_shape, input_shapes, config)` and stored in `node_info.node_config` for access in `forward()`.
    """
```

---

## Phase 3: Advanced Solvers

### 3.1 Incremental Predictive Coding (iPC)

**Background**: Standard PC updates all latents simultaneously. iPC updates latents incrementally node-by-node, which can improve convergence and stability.

```python
# fabricpc/solvers/ipc.py
"""
Incremental Predictive Coding (iPC) Solver

Unlike standard PC which updates all latents simultaneously,
iPC updates latents incrementally following topological order.
This can improve:
- Convergence speed (especially for deep networks)
- Stability (avoids oscillations)
- Memory efficiency (process one node at a time)

Reference: Salvatori et al. "Incremental Predictive Coding" (2022)
"""


def ipc_inference_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        eta_infer: float,
        update_order: str = "forward"  # "forward", "backward", "random"
) -> GraphState:
    """
    Single iPC inference step with sequential node updates.

    Args:
        update_order: Order to update nodes
            - "forward": Input to output (topological)
            - "backward": Output to input (reverse topological)
            - "random": Random order each step
    """
    if update_order == "forward":
        order = structure.node_order
    elif update_order == "backward":
        order = structure.node_order[::-1]
    else:
        order = jax.random.permutation(
            jax.random.PRNGKey(0),
            jnp.array(structure.node_order)
        )

    for node_name in order:
        if node_name in clamps:
            continue  # Skip clamped nodes

        node_info = structure.nodes[node_name]
        if node_info.in_degree == 0:
            continue  # Skip terminal input nodes

        # Get node class and gather inputs
        node_class = get_node_class(node_info.node_type)
        in_edges_data = gather_inputs(node_info, structure, state)

        # Compute forward pass and input gradients for this node
        node_state, inedge_grads = node_class.forward_and_latent_grads(
            params.nodes[node_name], in_edges_data,
            state.nodes[node_name], node_info
        )

        # Update node state with predictions, error, energy
        state = state._replace(nodes={**state.nodes, node_name: node_state})

        # Accumulate gradients to source nodes
        for edge_key, grad in inedge_grads.items():
            source_name = structure.edges[edge_key].source
            latent_grad = state.nodes[source_name].latent_grad + grad
            state = update_node_in_state(state, source_name, latent_grad=latent_grad)

        # Update this node's latent using its accumulated gradient
        new_z = node_state.z_latent - eta_infer * node_state.latent_grad
        state = update_node_in_state(state, node_name, z_latent=new_z)

    return state


def run_ipc_inference(
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        infer_steps: int,
        eta_infer: float = 0.1,
        update_order: str = "forward",
) -> GraphState:
    """
    Run iPC inference for multiple steps.

    The inner loop is not easily vectorizable due to sequential
    dependencies, but can be JIT compiled.
    """

    def body_fn(t, state):
        return ipc_inference_step(
            params, state, clamps, structure, eta_infer, update_order
        )

    return jax.lax.fori_loop(0, infer_steps, body_fn, initial_state)
```

### 3.2 Dynamic Inference Rate Scheduler

**Purpose**: Adapt learning rate per-node based on energy changes during inference.

```python
# fabricpc/solvers/dynamic_scheduler.py
"""
Dynamic Inference Rate Scheduler

Adapts the inference learning rate (eta_infer) per-node based on:
1. Current energy magnitude
2. Energy change rate (convergence speed)
3. Node depth in graph (deeper nodes may need different rates)

This enables:
- Faster convergence for stable nodes
- Careful updates for high-energy nodes
- Automatic tuning without manual hyperparameter search
"""

from typing import NamedTuple
import jax.numpy as jnp


class SchedulerState(NamedTuple):
    """State tracked by the dynamic scheduler."""
    node_etas: Dict[str, float]  # Per-node learning rates
    energy_history: Dict[str, jnp.ndarray]  # Rolling energy window
    step: int


def create_scheduler(
        structure: GraphStructure,
        base_eta: float = 0.1,
        window_size: int = 5,
        min_eta: float = 0.001,
        max_eta: float = 1.0,
) -> SchedulerState:
    """Initialize scheduler with uniform learning rates."""
    node_etas = {name: base_eta for name in structure.nodes}
    energy_history = {
        name: jnp.zeros(window_size)
        for name in structure.nodes
    }
    return SchedulerState(node_etas, energy_history, 0)


def update_scheduler(
        scheduler: SchedulerState,
        state: GraphState,
        config: Dict[str, Any],
) -> SchedulerState:
    """
    Update learning rates based on current energy landscape.

    Strategies:
    1. "energy_proportional": η ∝ 1/sqrt(energy)
    2. "gradient_adaptive": η based on gradient magnitude
    3. "convergence_aware": Increase η if converging, decrease if oscillating
    """
    strategy = config.get("strategy", "convergence_aware")
    base_eta = config.get("base_eta", 0.1)
    min_eta = config.get("min_eta", 0.001)
    max_eta = config.get("max_eta", 1.0)

    new_etas = {}
    new_history = {}

    for node_name, node_state in state.nodes.items():
        energy = sum(node_state.energy)  # sum energy over batch dimension
        history = scheduler.energy_history[node_name]

        # Update history (rolling window)
        new_history[node_name] = jnp.roll(history, -1).at[-1].set(energy)

        if strategy == "energy_proportional":
            # Lower energy = higher learning rate
            eta = base_eta / jnp.sqrt(energy + 1e-8)

        elif strategy == "gradient_adaptive":
            # Scale by gradient magnitude
            grad_norm = jnp.linalg.norm(node_state.latent_grad)
            eta = base_eta / (grad_norm + 1e-8)

        elif strategy == "convergence_aware":
            # Check if energy is decreasing (converging)
            if scheduler.step >= len(history):
                energy_trend = history[-1] - history[0]
                if energy_trend < 0:  # Converging
                    eta = scheduler.node_etas[node_name] * 1.1
                else:  # Oscillating or diverging
                    eta = scheduler.node_etas[node_name] * 0.9
            else:
                eta = scheduler.node_etas[node_name]

        new_etas[node_name] = jnp.clip(eta, min_eta, max_eta)

    return SchedulerState(new_etas, new_history, scheduler.step + 1)


def scheduled_inference_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        scheduler: SchedulerState,
) -> Tuple[GraphState, SchedulerState]:
    """
    Inference step with per-node adaptive learning rates.
    Uses forward_and_latent_grads to compute predictions, errors, and gradients.
    """
    # Zero latent gradients
    for node_name in structure.nodes:
        node_state = state.nodes[node_name]
        zero_grad = jnp.zeros_like(node_state.z_latent)
        state = update_node_in_state(state, node_name, latent_grad=zero_grad)

    # Forward pass: compute predictions, errors, and accumulate gradients
    for node_name in structure.nodes:
        node_info = structure.nodes[node_name]
        if node_info.in_degree == 0:
            continue  # Skip terminal input nodes

        node_class = get_node_class(node_info.node_type)
        in_edges_data = gather_inputs(node_info, structure, state)

        node_state, inedge_grads = node_class.forward_and_latent_grads(
            params.nodes[node_name], in_edges_data,
            state.nodes[node_name], node_info
        )
        state = state._replace(nodes={**state.nodes, node_name: node_state})

        # Accumulate gradients to source nodes
        for edge_key, grad in inedge_grads.items():
            source_name = structure.edges[edge_key].source
            latent_grad = state.nodes[source_name].latent_grad + grad
            state = update_node_in_state(state, source_name, latent_grad=latent_grad)

    # Update latents with per-node learning rates
    for node_name in structure.nodes:
        if node_name in clamps:
            new_z = clamps[node_name]
        else:
            eta = scheduler.node_etas[node_name]
            node_state = state.nodes[node_name]
            new_z = node_state.z_latent - eta * node_state.latent_grad
        state = update_node_in_state(state, node_name, z_latent=new_z)

    # Update scheduler
    scheduler = update_scheduler(scheduler, state, {})

    return state, scheduler
```

### 3.3 Solver Interface

```python
# fabricpc/solvers/base.py
"""
Unified solver interface for different inference algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from fabricpc.core.types import GraphParams, GraphState, GraphStructure

class Solver(ABC):
    """Base class for PC inference solvers."""

    @abstractmethod
    def step(
        self,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """Single inference step."""
        pass

    @abstractmethod
    def run(
        self,
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        num_steps: int,
    ) -> GraphState:
        """Run inference for multiple steps."""
        pass


class StandardPCSolver(Solver):
    """Standard parallel PC inference."""

    def __init__(self, eta_infer: float = 0.1):
        self.eta_infer = eta_infer

    def step(self, params, state, clamps, structure):
        return inference_step(params, state, clamps, structure, self.eta_infer)

    def run(self, params, initial_state, clamps, structure, num_steps):
        return run_inference(
            params, initial_state, clamps, structure,
            num_steps, self.eta_infer
        )


class IPCSolver(Solver):
    """Incremental PC solver."""

    def __init__(self, eta_infer: float = 0.1, update_order: str = "forward"):
        self.eta_infer = eta_infer
        self.update_order = update_order


class AdaptivePCSolver(Solver):
    """PC solver with dynamic learning rate scheduling."""

    def __init__(self, base_eta: float = 0.1, strategy: str = "convergence_aware"):
        self.base_eta = base_eta
        self.strategy = strategy
```

---

## Phase 4: Hypergraph Support

### 4.1 Hierarchical Node Containers

**Goal**: Enable nodes that contain subgraphs, allowing hierarchical network construction while maintaining PC semantics.

```python
# fabricpc/nodes/hypergraph.py
"""
Hypergraph support for hierarchical predictive coding networks.

A HyperNode is a container that:
1. Encapsulates a complete subgraph (GraphStructure, GraphParams, GraphState)
2. Exposes external interface via defined input/output slots
3. Runs internal PC inference during the parent graph's inference
4. Propagates gradients between hierarchy levels

This enables:
- Modular network design (reusable subgraph components)
- Hierarchical representations (cortical columns)
- Memory efficiency (shared subgraph definitions)
- Mixed abstraction levels
"""

from fabricpc.core.types import (
 GraphParams, GraphState, GraphStructure,
 NodeParams, NodeState, NodeInfo
)
from fabricpc.nodes.base import NodeBase, SlotSpec


@register_node("hypernode")
class HyperNode(NodeBase):
 """
 A node that contains a complete subgraph.

 The subgraph runs its own PC inference during each step of the
 parent graph's inference. Input/output boundaries connect the
 subgraph to the parent graph.

 Configuration:
 - subgraph_config: Configuration dict for internal graph
 - input_mapping: Maps parent edges to subgraph input nodes
 - output_mapping: Maps subgraph output nodes to this node's output
 - internal_infer_steps: Number of inference steps per parent step
 """

 @staticmethod
 def get_slots():
  # Slots are dynamically defined based on subgraph inputs
  # Default: single input that feeds subgraph input layer
  return {
   "in": SlotSpec(name="in", is_multi_input=True),
  }

 @staticmethod
 def initialize_params(key, node_shape, input_shapes, config):
  """
  Initialize both the hypernode's own parameters and its subgraph.
  """
  import numpy as np
  from fabricpc.graph_initialization.graph_net import create_pc_graph

  subgraph_config = config["subgraph_config"]
  key_sub, key_boundary = jax.random.split(key)

  # Create internal subgraph
  subgraph_params, subgraph_structure = create_pc_graph(
   subgraph_config, key_sub
  )

  # Boundary transformation weights (optional)
  # Maps from external input shape to subgraph input shape
  boundary_weights = {}
  for slot_name, ext_shape in input_shapes.items():
   internal_input = config["input_mapping"].get(slot_name)
   if internal_input:
    ext_dim = int(np.prod(ext_shape))
    int_shape = subgraph_structure.nodes[internal_input].shape
    int_dim = int(np.prod(int_shape))
    key_boundary, subkey = jax.random.split(key_boundary)
    boundary_weights[f"boundary_{slot_name}"] = (
            jax.random.normal(subkey, (ext_dim, int_dim)) * 0.01
    )

  return NodeParams(
   weights={
    **boundary_weights,
    "_subgraph_weights": subgraph_params,  # Nested params
   },
   biases={}
  )

 @staticmethod
 def forward(params, inputs, state, node_info):
  """
  Forward pass runs internal subgraph inference.
  Returns (total_energy, updated_state).
  """
  config = node_info.node_config
  subgraph_params = params.weights["_subgraph_weights"]
  subgraph_structure = config["_subgraph_structure"]
  internal_steps = config.get("internal_infer_steps", 5)
  eta_internal = config.get("internal_eta", 0.1)

  # Initialize subgraph state
  batch_size = list(inputs.values())[0].shape[0]
  subgraph_state = initialize_graph_state(
   subgraph_structure, batch_size
  )

  # Apply boundary transformations and clamp subgraph inputs
  clamps = {}
  for edge_key, ext_input in inputs.items():
   slot = edge_key.split(":")[-1]
   internal_input = config["input_mapping"].get(slot)
   if internal_input:
    boundary_key = f"boundary_{slot}"
    if boundary_key in params.weights:
     transformed = jnp.matmul(ext_input, params.weights[boundary_key])
    else:
     transformed = ext_input
    clamps[internal_input] = transformed

  # Run internal inference
  subgraph_state = run_inference(
   subgraph_params, subgraph_state, clamps,
   subgraph_structure, internal_steps, eta_internal
  )

  # Extract output from subgraph
  output_node = config["output_mapping"]["out"]
  z_mu = subgraph_state.nodes[output_node].z_latent

  # Compute error and energy
  error = state.z_latent - z_mu
  energy = 0.5 * jnp.sum(error ** 2, axis=-1)

  # Update state
  state = state._replace(
   z_mu=z_mu,
   pre_activation=z_mu,
   error=error,
   energy=energy,
   substructure={"_subgraph_state": subgraph_state}
  )

  return jnp.sum(energy), state

 @staticmethod
 def forward_and_latent_grads(params, inputs, state, node_info):
  """
  Forward pass with gradient computation for inputs.

  Propagates gradients backward through the subgraph to get
  input gradients for the parent graph.
  """
  from fabricpc.nodes import get_node_class
  node_class = get_node_class(node_info.node_type)

  # Run forward to get updated state
  _, state = node_class.forward(params, inputs, state, node_info)

  config = node_info.node_config
  subgraph_state = state.substructure["_subgraph_state"]
  subgraph_structure = config["_subgraph_structure"]
  subgraph_params = params.weights["_subgraph_weights"]

  # Set the output node's error to match this node's error
  output_node = config["output_mapping"]["out"]
  subgraph_state = update_node_in_state(
   subgraph_state, output_node,
   error=state.error,
   latent_grad=state.error  # Start gradient propagation
  )

  # Run one inference step backward through subgraph to propagate gradients
  for node_name in reversed(subgraph_structure.node_order):
   sub_node_info = subgraph_structure.nodes[node_name]
   if sub_node_info.in_degree == 0:
    continue

   sub_node_class = get_node_class(sub_node_info.node_type)
   in_edges_data = gather_inputs(sub_node_info, subgraph_structure, subgraph_state)

   _, inedge_grads = sub_node_class.forward_and_latent_grads(
    subgraph_params.nodes[node_name], in_edges_data,
    subgraph_state.nodes[node_name], sub_node_info
   )

   # Accumulate gradients to source nodes
   for edge_key, grad in inedge_grads.items():
    source_name = subgraph_structure.edges[edge_key].source
    latent_grad = subgraph_state.nodes[source_name].latent_grad + grad
    subgraph_state = update_node_in_state(subgraph_state, source_name, latent_grad=latent_grad)

  # Extract gradients for parent graph's source nodes
  input_grads = {}
  for edge_key, ext_input in inputs.items():
   slot = edge_key.split(":")[-1]
   internal_input = config["input_mapping"].get(slot)

   if internal_input:
    # Get gradient from subgraph input node
    internal_grad = subgraph_state.nodes[internal_input].latent_grad

    # Apply boundary transformation (if any)
    boundary_key = f"boundary_{slot}"
    if boundary_key in params.weights:
     grad = jnp.matmul(internal_grad, params.weights[boundary_key].T)
    else:
     grad = internal_grad

    input_grads[edge_key] = -grad

  return state, input_grads
```

### 4.2 Subgraph Sharing

```python
# fabricpc/graph/subgraph.py
"""
Utilities for subgraph sharing and reuse.

Enables:
- Define a subgraph template once, instantiate multiple times
- Share weights across instances (weight tying)
- Memory-efficient storage of replicated structures
"""

def create_subgraph_template(config: Dict[str, Any], key) -> Tuple[GraphStructure, GraphParams]:
    """Create a reusable subgraph template."""
    params, structure = create_pc_graph(config, key)
    return structure, params


def instantiate_hypernode(
    template_structure: GraphStructure,
    template_params: GraphParams,
    instance_config: Dict[str, Any],
    share_weights: bool = False,
) -> Dict[str, Any]:
    """
    Create a hypernode config from a template.

    Args:
        share_weights: If True, instances share the template's parameters
    """
    return {
        "type": "hypernode",
        "subgraph_config": template_structure,
        "initial_params": template_params if share_weights else None,
        **instance_config
    }
```

---

## Phase 5: Developer Experience

### 5.1 High-Level API

```python
# fabricpc/api.py
"""
High-level API for FabricPC.

Provides a simple, PyTorch-like interface while maintaining
the pure functional internals.
"""

class PCNetwork:
    """
    High-level wrapper for predictive coding networks.

    Example:
        # Define network
        net = PCNetwork()
        net.add_node("input", dim=784, activation="identity")
        net.add_node("hidden1", dim=256, activation="relu")
        net.add_node("hidden2", dim=128, activation="relu")
        net.add_node("output", dim=10, activation="softmax")

        net.connect("input", "hidden1")
        net.connect("hidden1", "hidden2")
        net.connect("hidden2", "output")

        net.set_task_mapping(x="input", y="output")

        # Initialize
        net.init(jax.random.PRNGKey(0))

        # Train
        net.fit(train_loader, epochs=10, infer_steps=20)

        # Predict
        predictions = net.predict(x_test, infer_steps=50)
    """

    def __init__(self):
        self._nodes = []
        self._edges = []
        self._task_map = {}
        self._params = None
        self._structure = None
        self._optimizer_state = None

    def add_node(
        self,
        name: str,
        dim: int = None,
        shape: Tuple[int, ...] = None,
        node_type: str = "linear",
        activation: str = "sigmoid",
        **kwargs
    ):
        """Add a node to the network."""
        if dim is None and shape is None:
            raise ValueError("Must specify either dim or shape")

        self._nodes.append({
            "name": name,
            "dim": dim or int(jnp.prod(jnp.array(shape))),
            "shape": shape or (dim,),
            "type": node_type,
            "activation": {"type": activation},
            **kwargs
        })
        return self

    def connect(self, source: str, target: str, slot: str = "in"):
        """Connect two nodes."""
        self._edges.append({
            "source_name": source,
            "target_name": target,
            "slot": slot
        })
        return self

    def set_task_mapping(self, **kwargs):
        """Map task names to nodes (x=input, y=output)."""
        self._task_map = kwargs
        return self

    def init(self, key):
        """Initialize network parameters."""
        config = {
            "node_list": self._nodes,
            "edge_list": self._edges,
            "task_map": self._task_map
        }
        self._params, self._structure = create_pc_graph(config, key)
        return self

    def fit(
        self,
        train_loader,
        epochs: int = 10,
        infer_steps: int = 20,
        eta_infer: float = 0.1,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        solver: str = "standard",  # "standard", "ipc", "adaptive"
        **kwargs
    ):
        """Train the network."""
        self._params, history, self._optimizer_state = train_pcn(
            self._params, self._structure, train_loader,
            {
                "num_epochs": epochs,
                "infer_steps": infer_steps,
                "eta_infer": eta_infer,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
            },
            jax.random.PRNGKey(0)
        )
        return history

    def predict(self, x, infer_steps: int = 50, eta_infer: float = 0.1):
        """Generate predictions."""
        batch_size = x.shape[0]
        state = initialize_graph_state(self._structure, batch_size)

        # Clamp input
        input_node = self._task_map.get("x")
        clamps = {input_node: x}

        # Run inference
        state = run_inference(
            self._params, state, clamps,
            self._structure, infer_steps, eta_infer
        )

        # Return output
        output_node = self._task_map.get("y")
        return state.nodes[output_node].z_latent
```

### 5.2 Model Serialization

```python
# fabricpc/utils/serialization.py
"""
Save and load FabricPC models.
"""

import pickle
from pathlib import Path

def save_model(
    path: str,
    params: GraphParams,
    structure: GraphStructure,
    optimizer_state = None,
    metadata: Dict = None
):
    """Save model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "params": jax.device_get(params),
        "structure": structure,
        "optimizer_state": optimizer_state,
        "metadata": metadata or {},
        "version": "0.2.0",
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def load_model(path: str) -> Tuple[GraphParams, GraphStructure, Any, Dict]:
    """Load model from disk."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    return (
        checkpoint["params"],
        checkpoint["structure"],
        checkpoint.get("optimizer_state"),
        checkpoint.get("metadata", {})
    )
```

### 5.3 Callbacks and Logging

```python
# fabricpc/training/callbacks.py
"""
Training callbacks for monitoring and control.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class Callback(ABC):
    """Base callback class."""

    def on_epoch_start(self, epoch: int, logs: Dict): pass
    def on_epoch_end(self, epoch: int, logs: Dict): pass
    def on_batch_start(self, batch: int, logs: Dict): pass
    def on_batch_end(self, batch: int, logs: Dict): pass
    def on_inference_step(self, step: int, state: GraphState): pass


class EarlyStopping(Callback):
    """Stop training when loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs):
        loss = logs.get("loss", float('inf'))
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logs["stop_training"] = True


class EnergyMonitor(Callback):
    """Monitor per-node energy during inference."""

    def __init__(self, log_frequency: int = 10):
        self.log_frequency = log_frequency
        self.energy_history = []

    def on_inference_step(self, step, state):
        if step % self.log_frequency == 0:
            energies = {
                name: float(sum(ns.energy))  # sum energy over batch dimension
                for name, ns in state.nodes.items()
            }
            self.energy_history.append(energies)


class TensorBoardLogger(Callback):
    """Log to Aim TensorBoard."""

```

---

## Phase 6: Demos and Documentation

### 6.1 Demo: Transformer for Text Classification

```python
# fabricpc/examples/transformer_text_demo.py
"""
Text Classification with Predictive Coding Transformers

This demo shows how to build a transformer-based text classifier
using FabricPC's predictive coding framework.

Key innovations demonstrated:
1. Self-attention with PC inference
2. Local learning in transformer blocks
3. Efficient inference with adaptive scheduling
"""


def create_pc_transformer(
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 4,
        num_classes: int = 2,
        max_seq_len: int = 512,
):
 """Create a predictive coding transformer for text classification."""

 net = PCNetwork()

 # Input embedding (tokens -> embeddings)
 net.add_node("input", dim=vocab_size, activation="identity")
 net.add_node("embedding", dim=embed_dim, node_type="linear", activation="identity")

 # Positional encoding
 net.add_node("pos_encoding", dim=embed_dim, node_type="positional_encoding",
              max_len=max_seq_len)

 # Transformer layers
 prev_layer = "embedding"
 for i in range(num_blocks):
  layer_name = f"transformer_{i}"
  net.add_node(layer_name, dim=embed_dim, node_type="transformer_block",
               num_heads=num_heads, ffn_dim=embed_dim * 4)
  net.connect(prev_layer, layer_name)
  if i == 0:
   net.connect("pos_encoding", layer_name, slot="position")
  prev_layer = layer_name

 # Classification head
 net.add_node("pool", dim=embed_dim, node_type="mean_pool")
 net.add_node("classifier", dim=num_classes, activation="softmax")

 net.connect("input", "embedding")
 net.connect(prev_layer, "pool")
 net.connect("pool", "classifier")

 net.set_task_mapping(x="input", y="classifier")

 return net


# Training script
if __name__ == "__main__":
 import jax
 from fabricpc.utils.data.data_utils import load_imdb_dataset

 # Load data
 train_loader, test_loader, vocab = load_imdb_dataset(batch_size=32)

 # Create model
 model = create_pc_transformer(
  vocab_size=len(vocab),
  embed_dim=256,
  num_heads=8,
  num_blocks=4,
  num_classes=2,
 )
 model.init(jax.random.PRNGKey(42))

 # Train with adaptive solver
 history = model.fit(
  train_loader,
  epochs=10,
  infer_steps=30,
  solver="adaptive",
  optimizer="adamw",
  learning_rate=1e-4,
 )

 # Evaluate
 accuracy = model.evaluate(test_loader)
 print(f"Test accuracy: {accuracy:.4f}")
```

### 6.2 Demo: ConvNet for Time Series

```python
# fabricpc/examples/conv_timeseries_demo.py
"""
Time Series Classification with Predictive Coding ConvNets

This demo shows how to build a 1D convolutional network for
multivariate time series classification using FabricPC.

Demonstrates:
1. Conv1D nodes for temporal feature extraction
2. Multi-scale feature hierarchies
3. PC inference for temporal prediction
"""


def create_pc_convnet(
        input_channels: int,
        seq_length: int,
        num_classes: int,
        base_filters: int = 32,
):
 """Create a PC-ConvNet for time series classification."""

 net = PCNetwork()

 # Input layer (batch, seq_len, channels)
 net.add_node("input", shape=(seq_length, input_channels), activation="identity")

 # Conv blocks with increasing receptive field
 prev = "input"
 channels = [base_filters, base_filters * 2, base_filters * 4]

 for i, ch in enumerate(channels):
  conv_name = f"conv_{i}"
  bn_name = f"bn_{i}"

  net.add_node(conv_name, shape=(seq_length // (2 ** i), ch),
               node_type="conv1d", kernel_size=5, stride=2,
               activation="relu")
  net.add_node(bn_name, shape=(seq_length // (2 ** i), ch),
               node_type="batch_norm")

  net.connect(prev, conv_name)
  net.connect(conv_name, bn_name)
  prev = bn_name

 # Global average pooling
 net.add_node("gap", dim=channels[-1], node_type="global_avg_pool1d")
 net.connect(prev, "gap")

 # Classification
 net.add_node("fc", dim=64, activation="relu")
 net.add_node("output", dim=num_classes, activation="softmax")

 net.connect("gap", "fc")
 net.connect("fc", "output")

 net.set_task_mapping(x="input", y="output")

 return net


if __name__ == "__main__":
 from fabricpc.utils.data.data_utils import load_ucr_dataset

 # Load UCR dataset (e.g., ECG200)
 train_loader, test_loader, metadata = load_ucr_dataset("ECG200", batch_size=32)

 model = create_pc_convnet(
  input_channels=1,
  seq_length=metadata["seq_length"],
  num_classes=metadata["num_classes"],
 )
 model.init(jax.random.PRNGKey(42))

 # Train with iPC solver (good for deep convnets)
 history = model.fit(
  train_loader,
  epochs=50,
  infer_steps=25,
  solver="ipc",
  optimizer="adam",
  learning_rate=1e-3,
 )

 accuracy = model.evaluate(test_loader)
 print(f"Test accuracy: {accuracy:.4f}")
```

### 6.3 Demo: Deep PC Networks with Hypergraphs

```python
# fabricpc/examples/hypergraph_demo.py
"""
Hierarchical Predictive Coding with Hypergraphs

This demo shows how to build deep hierarchical networks using
FabricPC's hypergraph support. Each level of the hierarchy
is a complete PC network that runs internal inference.

This models cortical columns with:
- Local recurrent processing
- Hierarchical message passing
- Top-down predictions and bottom-up errors
"""

def create_cortical_column(embed_dim: int, name_prefix: str):
    """Create a cortical column subgraph template."""
    return {
        "node_list": [
            {"name": f"{name_prefix}_L4", "dim": embed_dim, "activation": "relu"},
            {"name": f"{name_prefix}_L2/3", "dim": embed_dim, "activation": "relu"},
            {"name": f"{name_prefix}_L5", "dim": embed_dim, "activation": "relu"},
            {"name": f"{name_prefix}_L6", "dim": embed_dim, "activation": "relu"},
        ],
        "edge_list": [
            # Bottom-up pathway
            {"source_name": f"{name_prefix}_L4", "target_name": f"{name_prefix}_L2/3", "slot": "in"},
            # Lateral and feedback
            {"source_name": f"{name_prefix}_L2/3", "target_name": f"{name_prefix}_L5", "slot": "in"},
            {"source_name": f"{name_prefix}_L5", "target_name": f"{name_prefix}_L6", "slot": "in"},
            {"source_name": f"{name_prefix}_L6", "target_name": f"{name_prefix}_L4", "slot": "in"},
        ],
        "task_map": {
            "input": f"{name_prefix}_L4",
            "output": f"{name_prefix}_L2/3",
        }
    }


def create_hierarchical_pc_network(
    input_dim: int,
    num_levels: int = 4,
    embed_dim: int = 128,
    output_dim: int = 10,
):
    """Create a hierarchical PC network with cortical columns."""

    net = PCNetwork()

    # Input
    net.add_node("input", dim=input_dim, activation="identity")

    # Hierarchical cortical columns
    prev = "input"
    for level in range(num_levels):
        column_config = create_cortical_column(embed_dim, f"column_{level}")

        net.add_node(
            f"column_{level}",
            dim=embed_dim,
            node_type="hypernode",
            subgraph_config=column_config,
            input_mapping={"in": f"column_{level}_L4"},
            output_mapping={"out": f"column_{level}_L2/3"},
            internal_infer_steps=5,
        )

        net.connect(prev, f"column_{level}")
        prev = f"column_{level}"

    # Output
    net.add_node("output", dim=output_dim, activation="softmax")
    net.connect(prev, "output")

    net.set_task_mapping(x="input", y="output")

    return net
```

---

## Phase 7: Performance Optimization

Node parallelization to reduce scaling from O(N^2) to O(N)

## Phase 8: ML Workflow

### 8.1 Flax.linen (nesting module pytrees and nn.scan a block across layers)

nn.scan is particularly valuable for transformers - you can define one transformer block and scan it across layers, which dramatically reduces compile time and memory for the computation graph
```python
class TransformerStack(nn.Module):
    num_layers: int
    
    @nn.compact
    def __call__(self, x):
        ScanBlock = nn.scan(
            TransformerBlock,
            variable_axes={'params': 0},
            split_rngs={'dropout': True},
            length=self.num_layers,
        )
        x = ScanBlock()(x)
        return x
```
This creates a single set of per-layer parameters stacked along axis 0, then scans the block computation across the layer dimension. One compiled block, N layers of execution—huge win for compile time and enables gradient checkpointing per-layer trivially.

Flax provides a module system that lets you define models in an object-oriented style while compiling down to pure functional JAX under the hood.
The Linen API (flax.linen, aliased as nn)
Flax's current API is called Linen. You define modules by subclassing nn.Module and implementing a __call__ method, which looks superficially like PyTorch:
```python
import flax.linen as nn
import jax.numpy as jnp

class MLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x
```
The @nn.compact decorator is the key mechanism. It lets you define submodules inline (like those nn.Dense calls) rather than having to declare them in a separate setup() method. Flax intercepts those calls and auto-names them ("Dense_0", "Dense_1") to build the parameter tree.
But here's where it diverges from PyTorch fundamentally: the module doesn't hold parameters. Instead, you do an explicit init step:
```python
model = MLP(hidden_dim=256, output_dim=10)
params = model.init(rng_key, dummy_input)  # returns a frozen pytree
output = model.apply(params, real_input)    # pure function call
```

model.init() traces through the module, collects all parameter shapes and initializations, and returns them as a nested dict (a pytree). model.apply() takes those params as an explicit argument and runs the forward pass. Both are pure functions—no side effects, no hidden state. This means model.apply composes directly with jax.jit, jax.grad, jax.vmap without any wrapping or special handling.
The parameter pytree structure
The params object looks like:
```python
{
    'params': {
        'Dense_0': {'kernel': array(...), 'bias': array(...)},
        'Dense_1': {'kernel': array(...), 'bias': array(...)},
    }
}
```
This is just a nested dict of JAX arrays. You can inspect it, serialize it, slice it, transform it with jax.tree_util—there's no opaque object wrapping. This transparency is a major feature for research: you can write custom initialization schemes, parameter-specific learning rates, or selective freezing by directly manipulating the pytree with standard tree operations.
Mutable state handling
For things like batch norm running statistics or dropout RNG, Flax uses a "variable collections" system. Parameters live in the 'params' collection, batch stats in 'batch_stats', etc. You pass them in and get updated collections back:
```python
variables = model.init(rng_key, dummy_input)
# variables = {'params': {...}, 'batch_stats': {...}}

output, updated_state = model.apply(
    variables, x,
    mutable=['batch_stats']  # which collections can be modified
)
```
This keeps the functional contract: state goes in, updated state comes out. No hidden mutation. The mutable argument explicitly declares what's allowed to change, which makes the dataflow visible.
What Flax adds over raw JAX
The concrete things you'd otherwise have to build yourself:
Hierarchical parameter namespacing — in raw JAX, you'd manage a flat dict or manually nest dicts. Flax's module system gives you automatic hierarchical naming that mirrors your model architecture, making checkpoint loading, parameter inspection, and transfer learning straightforward.
Shape inference — Flax defers parameter creation until the first input is traced (during init), so Dense layers don't need you to specify input dimensions. This is similar to PyTorch's LazyLinear but is the default behavior.
RNG management — JAX's explicit PRNG is powerful but threading different RNG streams (one for dropout, one for initialization) through a deep model is tedious. Flax's module system handles RNG splitting and routing:
```python
output = model.apply(params, x, rngs={'dropout': dropout_key})
```
Each module that needs randomness gets an automatically-split key from the appropriate stream.

### 8.2 Orbax (checkpointing).

In JAX, your training state is a pytree—a nested dict of arrays, potentially sharded across multiple devices or hosts. Saving this isn't as simple as pickle.dump() because you need to handle: arrays distributed across GPUs/TPUs that need to be gathered or saved in-place per-shard, asynchronous writes that don't block training, atomic checkpoint completion (no half-written states), and managing a rolling window of checkpoints over thousands of training steps. Orbax provides a layered abstraction stack for all of this.
The abstraction layers (bottom to top)
At the lowest level, Orbax uses TensorStore as its storage backend. TensorStore handles the actual I/O of array data GitHub, using the OCDBT format by default—a key-value store optimized for large tensors. Each array leaf in your pytree gets serialized independently, which means individual parameters can be read back without loading the entire checkpoint. This is what enables partial restoration and cross-topology loading.
Above that sit TypeHandlers, which define how each leaf type in the pytree gets serialized and deserialized. jax.Array, np.ndarray, Python scalars, and strings all have their own handlers. The handler knows how to take a sharded jax.Array distributed across 8 GPUs and write each shard's data, plus metadata about the sharding specification, dtype, and shape. On restore, the handler reads the stored data and reconstitutes the array with whatever sharding you specify—which can differ from the original save topology.
CheckpointHandler sits above TypeHandlers. StandardCheckpointHandler is the default for pytrees and is capable of saving and restoring pytrees with leaves of type Python scalar, np.ndarray, and jax.Array Readthedocs. The handler walks the pytree, dispatches each leaf to the appropriate TypeHandler, and coordinates the overall save/restore. CompositeCheckpointHandler lets you bundle multiple logically distinct objects—say, training state as a pytree plus metadata as JSON—into a single checkpoint directory with different serialization strategies per item.
Checkpointer wraps a handler and adds atomicity and async support. StandardCheckpointer always saves asynchronously Readthedocs—it copies array data to host memory on a background thread and writes to disk without blocking the training loop. The atomicity mechanism works by writing to a temporary directory first, then doing an atomic rename once everything is committed. If the process crashes mid-write, the temporary directory is detected and cleaned up on the next startup, so you never end up with a corrupted checkpoint.
The basic API looks like:
```python
import orbax.checkpoint as ocp

with ocp.StandardCheckpointer() as ckptr:
    ckptr.save(path / 'step_1000', train_state)
    # non-blocking, returns immediately
    
    ckptr.wait_until_finished()  # block if you need to
    
    restored = ckptr.restore(path / 'step_1000', abstract_state)
```

The `abstract_state` on restore is important mechanistically—you provide a pytree of `jax.ShapeDtypeStruct` objects that tell Orbax what shape, dtype, and *sharding* to restore into. This is how you handle topology changes: you save from 4 GPUs and restore onto 8 by providing an abstract state with the new sharding spec.

**CheckpointManager** is the highest-level API, designed for training loop integration. It keeps track of multiple checkpointable objects with a directory structure organized by step number :
```
checkpoints/
  0/
    params/
    metadata/
  500/
    params/
    metadata/
  1000/
    ...
```
It provides customization options including save_decision_policy (determines when to save), preservation_policy (determines which checkpoints to keep), step_format_fixed_length for formatted step directories, and cleanup_tmp_directories for automatic cleanup of incomplete saves Readthedocs. A typical training loop integration:
```python
options = ocp.CheckpointManagerOptions(
    save_interval_steps=500,
    max_to_keep=3,  # rolling window
)

with ocp.CheckpointManager(ckpt_dir, options=options) as mngr:
    for step in range(num_steps):
        train_state = train_step(train_state, batch)
        
        mngr.save(step, args=ocp.args.StandardSave(train_state))
        # only actually writes every 500 steps per policy
        # automatically deletes old checkpoints beyond max_to_keep
```
The manager handles the bookkeeping: tracking which steps exist, deleting old checkpoints (respecting the max_to_keep window), and providing latest_step() / all_steps() queries for resumption logic.
Composite checkpoints let you save heterogeneous state with different serialization:
```python
mngr.save(step, args=ocp.args.Composite(
    state=ocp.args.StandardSave(train_state),    # pytree → TensorStore
    metadata=ocp.args.JsonSave(run_config),       # dict → JSON file
))
```
Each item gets its own subdirectory and handler, but they're managed atomically as a single checkpoint step.
The async mechanism in detail
The save operation does not block—it copies data from device to host memory and writes asynchronously Readthedocs. Mechanistically, the AsyncCheckpointer launches a background thread that performs the actual I/O. The training loop continues computing while writes happen in the background. If you call save() again before the previous save finishes, Orbax queues the new save and ensures ordering. wait_until_finished() blocks until all pending writes complete—you'd use this before evaluation or at program shutdown.
Multi-host / distributed restore optimization
For large-scale distributed training, rather than every device independently fetching the entire checkpoint from storage (like GCS), Orbax supports single-replica restore where only one replica downloads the checkpoint and then broadcasts it to all other replicas via high-speed interconnects Google Cloud. To handle the memory overhead, Orbax breaks the checkpoint into smaller chunks and broadcasts them sequentially, respecting user-defined memory limits to avoid OOM errors Google Cloud.


### 8.3 Grain (data loaders)

Grain — data loading and preprocessing
Grain is a Python library for reading and processing data for training and evaluating JAX models. It is flexible, fast, and deterministic. GitHub
This is a full data pipeline library—analogous to PyTorch's DataLoader but designed around JAX's functional and distributed paradigms. The core problems it solves:
Deterministic data iteration. Grain emphasizes resilience to preemptions by maintaining minimal checkpoint sizes and enabling seamless resumption. Medium The IndexSampler generates a deterministic sequence of indices given a seed. If training gets preempted, Grain can checkpoint just the sampler state (essentially an integer offset), and on restart, it produces the exact same data sequence from where it left off. This is critical for reproducibility—you get bit-identical training runs regardless of interruptions.
True global shuffling. When paired with a random-access file format like ArrayRecord, Grain can perform true global shuffling across the entire dataset even when it doesn't fit in host memory. Google Developers This is different from the approximate shuffling most pipelines do (shuffle filenames, then shuffle a small buffer). Grain generates shuffled index sequences and reads records by random access, so every element has equal probability of appearing at any position.
Declarative pipeline API. You compose transformations as a chain:
```python
dataset = (
    grain.MapDataset.source(data_source)
    .shuffle(seed=42)
    .map(preprocess_fn)
    .batch(batch_size=32)
)
```
Each operation is a lazy transformation that gets applied per-element during iteration. The DataLoader wraps this with multi-worker prefetching so CPU preprocessing overlaps with GPU computation.
Orbax integration for pipeline checkpointing. Grain provides PyGrainCheckpointHandler specifically for checkpointing the DataLoader's iterator, and it's recommended to use with Orbax, which can checkpoint both the input pipeline and model together in a multi-host environment. Readthedocs
In a typical JAX training loop:
- Grain feeds data in
- Orbax saves checkpoints of both the model state and the Grain iterator state
- Optax handles the optimizer
- Flax provides the model

---

## Implementation Priority

### Immediate (Phase 1-2)
1. N-dimensional tensor support
2. Plugin architecture
3. LayerNorm and BatchNorm nodes
4. Conv2D node
5. Softmax node

### Near-term (Phase 3)
1. Multi-head attention
2. Transformer block
3. iPC solver
4. Dynamic scheduler

### Medium-term (Phase 4-5)
1. Hypergraph support
2. High-level API
3. Serialization
4. Callbacks

### Long-term (Phase 6-7)
1. Comprehensive demos
2. Documentation site
3. Benchmarking suite
4. Community examples

---

## Additional Considerations

### API Stability
- Maintain backward compatibility within major versions
- Deprecation warnings before breaking changes
- Semantic versioning

### Documentation
- API reference (auto-generated from docstrings)
- Tutorials for common use cases
- Theoretical background on PC
- Contributing guidelines

### Community
- GitHub discussions for questions
- Issue templates for bugs/features
- Example gallery (community contributed)
- Citation file for academic use

### Performance Targets
- MNIST training: <10s (single GPU)
- Inference step: <1ms for 1000-node graph
- Memory: Linear in batch size
- Multi-GPU scaling: >80% efficiency

---

## References

1. Rao & Ballard (1999) - Predictive Coding in Visual Cortex
2. Whittington & Bogacz (2017) - Approximation of Backprop by PC
3. Millidge et al. (2022) - Predictive Coding: A Theoretical and Experimental Review
4. Salvatori et al. (2022) - Incremental Predictive Coding
5. Song et al. (2024) - Inferring Neural Activity Before Plasticity

---

*This roadmap is a living document. Update as development progresses.*