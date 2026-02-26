# Transformer block node and components for building transformer architectures.
# Initially, the node implements a transformer block, with latent state representing the output of the block.
# We experiment with a sequence of block nodes and compare to backprop.
# Then we progressively break down the block node into its components for a fully-PC approach: multi-head attention, feedforward, layer norm, residual connections

import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, NodeParams, SlotSpec
from fabricpc.core.activations import IdentityActivation, GeluActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.types import NodeState, NodeInfo
from typing import Dict, Tuple, Any

# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================


def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    x_odd = x[..., 1::2]  # (batch, heads, seq, head_dim // 2)

    # Apply rotation:
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    # Interleave back: stack and reshape
    # Stack along last axis then reshape to interleave
    x_rot = jnp.stack([x_even_rot, x_odd_rot], axis=-1)
    x_rot = x_rot.reshape(x.shape)

    return x_rot


class TransformerBlock(NodeBase):
    """
    Complete Transformer Block with attention and FFN.

    Architecture:
    x → LayerNorm → MHA → + → LayerNorm → FFN →  +
    └─────────────────────┘ └────────────────────┘
         (residual)              (residual)

    Positional Encoding:
    Uses Rotary Position Embeddings (RoPE) by default.

    Args:
        shape: (seq_len, embed_dim) tuple
        name: Node name
        activation: Output activation (default: IdentityActivation)
        energy: Energy functional (default: GaussianEnergy)
        internal_activation: FFN internal activation (default: GeluActivation)
        num_heads: Number of attention heads (default: 8)
        ff_dim: FFN hidden dim (default: 4 * embed_dim)
        dropout_rate: Dropout rate, currently unused (default: 0.0)
        pre_norm: Use pre-norm architecture (default: True)
        use_rope: Use Rotary Position Embeddings (default: True)
        rope_theta: Base frequency for RoPE (default: 10000.0)
        weight_init: InitializerBase for weights
    """

    DEFAULT_ACTIVATION = IdentityActivation
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_LATENT_INIT = NormalInitializer

    def __init__(
        self,
        shape,
        name,
        activation=None,
        energy=None,
        internal_activation=None,
        num_heads=8,
        ff_dim=None,
        dropout_rate=0.0,
        pre_norm=True,
        use_rope=True,
        rope_theta=10000.0,
        weight_init=None,
        latent_init=None,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            internal_activation=internal_activation or GeluActivation(),
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            pre_norm=pre_norm,
            use_rope=use_rope,
            rope_theta=rope_theta,
            weight_init=weight_init,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec(name="in", is_multi_input=False),  # Input to the block
            "mask": SlotSpec(name="mask", is_multi_input=False),  # Optional mask
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:

        num_heads = config.get("num_heads", 8)
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
                "W_ff1": jax.random.normal(keys[4], (embed_dim, ff_dim))
                * std
                * jnp.sqrt(ff_dim / embed_dim),
                "W_ff2": jax.random.normal(keys[5], (ff_dim, embed_dim))
                * std
                * jnp.sqrt(ff_dim / embed_dim),
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
            },
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        """Forward pass for the Transformer Block."""
        config = node_info.node_config
        num_heads = config.get("num_heads", 8)

        # Get internal activation from config (stored as ActivationBase instance)
        internal_activation = config.get("internal_activation")
        if internal_activation is not None:
            activation_fn = lambda x: type(internal_activation).forward(
                x, internal_activation.config
            )
        else:
            activation_fn = lambda x: x

        # Get input (self-attention)
        in_edge_key = next(iter(k for k in inputs.keys() if k.endswith(":in")))
        input_tensor = inputs[in_edge_key]

        batch_size, seq_len, embed_dim = input_tensor.shape
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Find mask input key if provided
        mask_edge_key = next((k for k in inputs.keys() if k.endswith(":mask")), None)
        mask = inputs[mask_edge_key] if mask_edge_key else None

        # LayerNorm 1
        x_norm1 = TransformerBlock._layernorm(
            input_tensor, params.weights["ln1_gamma"], params.biases["ln1_beta"]
        )

        # Multi-Head Attention
        attn_output, substructure_attn = TransformerBlock._mha(
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
            use_rope=config.get("use_rope", True),
            rope_theta=config.get("rope_theta", 10000.0),
        )

        # Residual connection 1
        x_res1 = input_tensor + attn_output

        # LayerNorm 2
        x_norm2 = TransformerBlock._layernorm(
            x_res1, params.weights["ln2_gamma"], params.biases["ln2_beta"]
        )

        # Feedforward Network
        ff_intermediate = (
            jnp.matmul(x_norm2, params.weights["W_ff1"]) + params.biases["b_ff1"]
        )
        ff_activated = activation_fn(ff_intermediate)
        ff_output = (
            jnp.matmul(ff_activated, params.weights["W_ff2"]) + params.biases["b_ff2"]
        )

        # Residual connection 2
        z_mu = x_res1 + ff_output

        pre_activation = z_mu
        error = state.z_latent - z_mu

        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
        )

        # Compute energy, accumulate the self-latent gradient
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state

    @staticmethod
    def _layernorm(
        x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5
    ) -> jnp.ndarray:
        """Layer Normalization implementation."""
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / jnp.sqrt(variance + eps)
        return gamma * x_normalized + beta

    @staticmethod
    def _mha(
        x: jnp.ndarray,
        mask: jnp.ndarray,
        num_heads: int,
        W_q: jnp.ndarray,
        W_k: jnp.ndarray,
        W_v: jnp.ndarray,
        W_o: jnp.ndarray,
        b_q: jnp.ndarray,
        b_k: jnp.ndarray,
        b_v: jnp.ndarray,
        b_o: jnp.ndarray,
        activation_fn,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Multi-head attention implementation.
        Variables are named using standard transformer notation for mathematical clarity.
        """
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

        # Apply Rotary Position Embeddings to Q and K
        if use_rope:
            cos, sin = precompute_freqs_cis(head_dim, seq_len, theta=rope_theta)
            Q = apply_rotary_emb(Q, cos, sin)
            K = apply_rotary_emb(K, cos, sin)

        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim)
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale

        # Optional mask
        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        attn_matrix = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_matrix, V)

        # Reshape back: (batch, seq, embed_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, embed_dim
        )

        # Output projection
        pre_activation = jnp.matmul(attn_output, W_o) + b_o
        projection = activation_fn(pre_activation)

        substructure = {
            "attn_matrix": attn_matrix,
            "Q": Q,
            "K": K,
            "V": V,
        }

        return projection, substructure
