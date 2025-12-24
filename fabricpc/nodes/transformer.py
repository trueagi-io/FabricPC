# Transformer block node and components for building transformer architectures.
# Initially, the node implements a transformer block, with latent state representing the output of the block.
# We experiment with a sequence of block nodes and compare to backprop.
# Then we progressively break down the block node into its components for a fully-PC approach: multi-head attention, feedforward, layer norm, residual connections

import jax
import jax.numpy as jnp

from fabricpc.nodes import get_node_class
from fabricpc.nodes.base import NodeBase, NodeParams, SlotSpec
from fabricpc.nodes.registry import register_node
from fabricpc.core.activations import get_activation
from fabricpc.core.types import NodeState, NodeInfo
from typing import Dict, Tuple, Any


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute the frequency tensor for RoPE (Rotary Position Embeddings).

    RoPE encodes position by rotating pairs of dimensions in the embedding space.
    Each pair (2i, 2i+1) is rotated by angle θ_i * position, where θ_i decreases
    geometrically with dimension index.

    θ_i = 1 / (theta^(2i/d))

    Args:
        head_dim: Dimension of each attention head (must be even)
        max_seq_len: Maximum sequence length to precompute
        theta: Base for the geometric progression of frequencies (default 10000)

    Returns:
        Tuple of (cos, sin) arrays of shape (max_seq_len, head_dim // 2)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Compute frequency for each dimension pair: θ_i = 1 / (theta^(2i/d))
    dim_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    freqs = 1.0 / (theta ** (dim_indices / head_dim))

    # Compute position indices
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)

    # Outer product: (seq_len, head_dim // 2)
    angles = jnp.outer(positions, freqs)

    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary_emb(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply rotary position embeddings to input tensor.

    The rotation is applied to pairs of dimensions:
    [x_0, x_1] → [x_0 * cos - x_1 * sin, x_0 * sin + x_1 * cos]

    Args:
        x: Input tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine frequencies of shape (seq_len, head_dim // 2)
        sin: Sine frequencies of shape (seq_len, head_dim // 2)

    Returns:
        Rotated tensor of same shape as input
    """
    # x shape: (batch, num_heads, seq_len, head_dim)
    seq_len = x.shape[2]
    head_dim = x.shape[3]

    # Slice frequencies to match sequence length
    cos = cos[:seq_len, :]  # (seq_len, head_dim // 2)
    sin = sin[:seq_len, :]  # (seq_len, head_dim // 2)

    # Reshape for broadcasting: (1, 1, seq_len, head_dim // 2)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    # Split x into even and odd dimensions
    x_even = x[..., 0::2]  # (batch, heads, seq, head_dim // 2)
    x_odd = x[..., 1::2]   # (batch, heads, seq, head_dim // 2)

    # Apply rotation:
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    # Interleave back: stack and reshape
    # Stack along last axis then reshape to interleave
    x_rot = jnp.stack([x_even_rot, x_odd_rot], axis=-1)
    x_rot = x_rot.reshape(x.shape)

    return x_rot

@register_node("transformer_block")
class TransformerBlockNode(NodeBase):
    """
    Complete Transformer Block with attention and FFN.

    Architecture:
    x → LayerNorm → MHA → + → LayerNorm → FFN →  +
    └─────────────────────┘ └────────────────────┘
         (residual)              (residual)

    This is a composite node that internally manages its substructure.
    For hypergraph support, this could be decomposed into a subgraph.

    Positional Encoding:
    Uses Rotary Position Embeddings (RoPE) by default. This generalizes better than absolute positional encodings.

    **Example Transformer Block Config**:
    {
        "name": "transformer_block_1",
        "shape": (128, 512),  # (seq_len, embed_dim)
        "type": "transformer_block",
        "internal_activation": {"type": "gelu"},
        # Transformer-specific config:
        "num_heads": 8,
        "ff_dim": 2048,
        "dropout_rate": 0.1,
        "pre_norm": True,
    }
    """

    CONFIG_SCHEMA = {
        "num_heads": {
            "type": int,
            "default": 8,
            "description": "Number of attention heads"
        },
        "ff_dim": {
            "type": int,
            "default": None,  # Will be computed as 4 * embed_dim if not specified
            "description": "Feedforward hidden dimension (defaults to 4 * embed_dim)"
        },
        "internal_activation": {
            "type": dict,
            "default": {"type": "gelu"},
            "description": "Internal activation function for the feedforward network"
        },
        "dropout_rate": {
            "type": float,
            "default": 0.0,
            "description": "Dropout rate (currently unused, for future implementation)"
        },
        "pre_norm": {
            "type": bool,
            "default": True,
            "description": "Use pre-norm architecture (LayerNorm before attention/FFN)"
        },
        "use_rope": {
            "type": bool,
            "default": True,
            "description": "Use Rotary Position Embeddings (RoPE)"
        },
        "rope_theta": {
            "type": float,
            "default": 10000.0,
            "description": "Base frequency for RoPE"
        },
        "weight_init": {
            "type": dict,
            "default": {"type": "kaiming", "mode": "fan_out"},
            "description": "Weight initialization config"
        },
    }

    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}
    DEFAULT_ACTIVATION_CONFIG = {"type": "identity"}  # Identity at output of node; internal activations handled in "internal_activation"

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec(name="in", is_multi_input=False)  ,  # Input to the block
            "mask": SlotSpec(name="mask", is_multi_input=False),  # Optional mask
        }

    @staticmethod
    def initialize_params(
            key: jax.Array,  # from jax.random.PRNGKey
            node_shape: Tuple[int, ...],
            input_shapes: Dict[str, Tuple[int, ...]],  # edge_key -> source shape
            config: Dict[str, Any]
    ) -> NodeParams:

        num_heads = config["num_heads"]
        embed_dim = node_shape[-1]
        # Default ff_dim to 4 * embed_dim (standard transformer ratio)
        ff_dim = config.get("ff_dim") or (4 * embed_dim)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

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
                "W_ff1": jax.random.normal(keys[4], (embed_dim, ff_dim)) * std * jnp.sqrt(ff_dim / embed_dim),
                "W_ff2": jax.random.normal(keys[5], (ff_dim, embed_dim)) * std * jnp.sqrt(ff_dim / embed_dim),
                # LayerNorm parameters
                "ln1_gamma": jnp.ones((1, 1, embed_dim)),
                "ln2_gamma": jnp.ones((1, 1, embed_dim)),
            },
            biases={
                "b_q": jnp.zeros((1, 1, embed_dim)),
                "b_k": jnp.zeros((1, 1, embed_dim)),
                "b_v": jnp.zeros((1, 1, embed_dim)),
                "b_o": jnp.zeros((1, 1, embed_dim)),
                "b_ff1": jnp.zeros((1, 1, ff_dim)),
                "b_ff2": jnp.zeros((1, 1, embed_dim)),
                "ln1_beta": jnp.zeros((1, 1, embed_dim)),
                "ln2_beta": jnp.zeros((1, 1, embed_dim)),
            }
        )

    @staticmethod
    def forward(
            params: NodeParams,
            inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
            state: NodeState,  # state object for the present node
            node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        """
        Forward pass for the Transformer Block.
        """
        node_class = get_node_class(node_info.node_type)
        config = node_info.node_config
        num_heads = config["num_heads"]

        # Get input (self-attention)
        # find the input key with slotname "in", key format is "{source_name}->{target_name}:{slot_name}"
        in_edge_key = next(iter(k for k in inputs.keys() if k.endswith(":in")))
        input_tensor = inputs[in_edge_key]  # shape: (batch_size, seq_len, embed_dim)

        batch_size, seq_len, embed_dim = input_tensor.shape
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Find mask input key if provided
        mask_edge_key = next((k for k in inputs.keys() if k.endswith(":mask")), None)
        mask = inputs[mask_edge_key] if mask_edge_key else None
        activation_fn, _ = get_activation(config["internal_activation"])

        # LayerNorm 1
        x_norm1 = TransformerBlockNode._layernorm(
            input_tensor,
            params.weights["ln1_gamma"],
            params.biases["ln1_beta"]
        )

        # Multi-Head Attention
        attn_output, substructure_attn = TransformerBlockNode._mha(
            x_norm1,
            mask,
            num_heads,
            params.weights["W_q"],
            params.weights["W_k"],
            params.weights["W_v"],
            params.weights["W_o"],
            params.biases["b_q"],
            params.biases["b_k"],
            params.biases["b_v"],
            params.biases["b_o"],
            lambda x: x,  # Identity activation for attention output
            use_rope=config["use_rope"],
            rope_theta=config["rope_theta"],
        )

        # Residual connection 1
        x_res1 = input_tensor + attn_output

        # LayerNorm 2
        x_norm2 = TransformerBlockNode._layernorm(
            x_res1,
            params.weights["ln2_gamma"],
            params.biases["ln2_beta"]
        )

        # Feedforward Network
        ff_intermediate = jnp.matmul(x_norm2, params.weights["W_ff1"]) + params.biases["b_ff1"]
        ff_activated = activation_fn(ff_intermediate)
        ff_output = jnp.matmul(ff_activated, params.weights["W_ff2"]) + params.biases["b_ff2"]

        # Residual connection 2
        z_mu = x_res1 + ff_output

        pre_activation = z_mu.copy()
        error = state.z_latent - z_mu

        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
        )

        # Compute energy, accumulate the self-latent gradient
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state

    @staticmethod
    def _layernorm(
            x: jnp.ndarray,
            gamma: jnp.ndarray,
            beta: jnp.ndarray,
            eps: float = 1e-5
    ) -> jnp.ndarray:
        """
        Layer Normalization implementation.
        Normalizes the input across the last dimension.
        """

        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / jnp.sqrt(variance + eps)
        return gamma * x_normalized + beta

    @staticmethod
    def _mha(x: jnp.ndarray,
             mask: jnp.ndarray,
             num_heads: int,         # Number of attention heads
             W_q: jnp.ndarray,  # Query projection
             W_k: jnp.ndarray,  # Key projection
             W_v: jnp.ndarray,  # Value projection
             W_o: jnp.ndarray,  # Output projection
             b_q: jnp.ndarray,  # Query bias
             b_k: jnp.ndarray,  # Key bias
             b_v: jnp.ndarray,  # Value bias
             b_o: jnp.ndarray,  # Output bias
             activation_fn,     # Activation function
             use_rope: bool = True,  # Use Rotary Position Embeddings
             rope_theta: float = 10000.0,  # Base frequency for RoPE
    ) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Multi-head attention implementation.
        Implements scaled dot-product multi-head attention.
        1. Linear projections for Q, K, V
        2. Apply RoPE to Q and K (if enabled)
        3. Scaled dot-product attention
        4. Concatenate heads and final linear projection
        5. Apply activation function

        Variables are named using standard transformer notation; don't change for PEP compatibility.
        """

        # Get input (self-attention)
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads

        # Linear projections
        Q = jnp.matmul(x, W_q) + b_q
        K = jnp.matmul(x, W_k) + b_k
        V = jnp.matmul(x, W_v) + b_v

        # Reshape for multi-head: (batch, seq, heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        # Now for multi-head: (batch, heads, seq, head_dim)

        # Apply Rotary Position Embeddings to Q and K
        if use_rope:
            cos, sin = precompute_freqs_cis(head_dim, seq_len, theta=rope_theta)
            Q = apply_rotary_emb(Q, cos, sin)
            K = apply_rotary_emb(K, cos, sin)

        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim)
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale  # (batch, heads, seq, seq)

        # Optional mask
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        attn_matrix = jax.nn.softmax(scores, axis=-1)  # (batch, heads, seq, seq)
        attn_output = jnp.matmul(attn_matrix, V)  # (batch, heads, seq, head_dim)

        # Reshape back: (batch, seq, embed_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)

        # Output projection
        pre_activation = jnp.matmul(attn_output, W_o) + b_o

        # Apply activation (typically identity for attention output)
        projection = activation_fn(pre_activation)

        # Store attention weights for gradient computation
        substructure = {
            "attn_matrix": attn_matrix,
            "Q": Q,
            "K": K,
            "V": V,
        }

        return projection, substructure

