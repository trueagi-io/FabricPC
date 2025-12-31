"""
Transformer components for JAX predictive coding networks.

This module implements:
- EmbeddingNode: Learned vector lookup
- TransformerBlockNode: Full transformer layer (Self-Attention + MLP)
- Rotary Positional Embeddings (RoPE)
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from jax import nn

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.registry import register_node
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.initialization import initialize_weights
from fabricpc.core.positional import precompute_freqs_cis, apply_rotary_emb

# ==============================================================================
# EMBEDDING NODE
# ==============================================================================


@register_node("embedding")
class EmbeddingNode(NodeBase):
    """
    Embedding Node: Maps integer indices to dense vectors.

    Slot "in" expects integer indices (usually shape (batch, seq_len)).
    Output shape is (seq_len, embed_dim).
    """

    CONFIG_SCHEMA = {
        "vocab_size": {
            "type": int,
            "required": True,
            "description": "Size of vocabulary",
        },
        "embed_dim": {
            "type": int,
            "required": True,
            "description": "Embedding dimension",
        },
        "weight_init": {
            "type": dict,
            "default": {"type": "normal", "std": 0.02},
            "description": "Initialization config",
        },
    }

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        vocab_size = config["vocab_size"]
        embed_dim = config["embed_dim"]

        # Initialize embedding matrix (vocab_size, embed_dim)
        w_key, _ = jax.random.split(key)
        weights = {
            "embeddings": initialize_weights(
                config.get("weight_init"), w_key, (vocab_size, embed_dim)
            )
        }
        return NodeParams(weights=weights, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        # Get input indices, might be shape (batch, seq_len, 1) or (batch, seq_len)
        edge_key = list(inputs.keys())[0]
        indices = inputs[edge_key]

        if indices.ndim == 3 and indices.shape[-1] == 1:
            indices = jnp.squeeze(indices, axis=-1)

        indices_int = indices.astype(jnp.int32)

        # Lookup: (batch, seq_len) -> (batch, seq_len, embed_dim)
        z_mu = params.weights["embeddings"][indices_int]

        # Standard PC error computation
        error = state.z_latent - z_mu

        # Embedding node usually doesn't have activation derivative gain modulation
        # because the "pre-activation" IS the lookup result.
        gain_mod_error = error

        state = state._replace(z_mu=z_mu, error=error, gain_mod_error=gain_mod_error)

        # Compute Energy
        state = EmbeddingNode.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """
        Forward inference for embedding node.

        Embedding indices are discrete, so we can't compute gradients w.r.t. them.
        We only compute the forward pass and return zero gradients for inputs.
        """
        from fabricpc.nodes import get_node_class

        node_class = get_node_class(node_info.node_type)

        # Forward pass only
        _, new_state = node_class.forward(params, inputs, state, node_info)

        # Return zero gradients for all inputs
        input_grads = {}
        for edge_key, inp in inputs.items():
            input_grads[edge_key] = jnp.zeros_like(inp)

        return new_state, input_grads


# ==============================================================================
# TRANSFORMER BLOCK NODE
# ==============================================================================


@register_node("transformer_block")
class TransformerBlockNode(NodeBase):
    """
    Standard Transformer Block:
    x + Attention(LN(x)) + MLP(LN(x))

    Includes:
    - Multi-Head Self Attention (with RoPE)
    - Feed Forward Network (MLP)
    - Pre-Norm architecture
    """

    CONFIG_SCHEMA = {
        "embed_dim": {"type": int, "required": True},
        "num_heads": {"type": int, "required": True},
        "mlp_dim": {"type": int, "required": True},
        "activation": {"type": str, "default": "gelu"},
        "use_rope": {"type": bool, "default": True},
        "max_seq_len": {
            "type": int,
            "default": 1024,
            "description": "For precomputing RoPE",
        },
        "dropout_rate": {"type": float, "default": 0.0},
    }

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        mlp_dim = config["mlp_dim"]
        head_dim = embed_dim // num_heads

        keys = jax.random.split(key, 10)

        # Initialize weights
        def init_w(k, shape):
            return initialize_weights({"type": "xavier"}, k, shape)

        weights = {
            # Attention Projections
            "attn_q": init_w(keys[0], (embed_dim, embed_dim)),
            "attn_k": init_w(keys[1], (embed_dim, embed_dim)),
            "attn_v": init_w(keys[2], (embed_dim, embed_dim)),
            "attn_o": init_w(keys[3], (embed_dim, embed_dim)),
            # MLP Projections
            "mlp_in": init_w(keys[4], (embed_dim, mlp_dim)),
            "mlp_out": init_w(keys[5], (mlp_dim, embed_dim)),
        }

        biases = {
            # Attention Biases
            "attn_q": jnp.zeros((embed_dim,)),
            "attn_k": jnp.zeros((embed_dim,)),
            "attn_v": jnp.zeros((embed_dim,)),
            "attn_o": jnp.zeros((embed_dim,)),
            # MLP Biases
            "mlp_in": jnp.zeros((mlp_dim,)),
            "mlp_out": jnp.zeros((embed_dim,)),
            # Layer Norms (Scale and Bias)
            "ln1_scale": jnp.ones((embed_dim,)),
            "ln1_bias": jnp.zeros((embed_dim,)),
            "ln2_scale": jnp.ones((embed_dim,)),
            "ln2_bias": jnp.zeros((embed_dim,)),
        }

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        head_dim = embed_dim // num_heads

        # Aggregate Inputs
        x_in = jnp.zeros((state.z_latent.shape[0], *node_info.shape))
        for x in inputs.values():
            x_in = x_in + x

        batch_size, seq_len, _ = x_in.shape

        # ----------------------------------------------------------------------
        # Sub-Block 1: Attention
        # ----------------------------------------------------------------------
        # Layer Norm 1
        mu = jnp.mean(x_in, axis=-1, keepdims=True)
        sigma = jnp.std(x_in, axis=-1, keepdims=True)
        ln1 = (x_in - mu) / (sigma + 1e-6)
        ln1 = ln1 * params.biases["ln1_scale"] + params.biases["ln1_bias"]

        # Projections
        wq = jnp.dot(ln1, params.weights["attn_q"]) + params.biases["attn_q"]
        wk = jnp.dot(ln1, params.weights["attn_k"]) + params.biases["attn_k"]
        wv = jnp.dot(ln1, params.weights["attn_v"]) + params.biases["attn_v"]

        # Reshape to (batch, seq, num_heads, head_dim)
        wq = wq.reshape(batch_size, seq_len, num_heads, head_dim)
        wk = wk.reshape(batch_size, seq_len, num_heads, head_dim)
        wv = wv.reshape(batch_size, seq_len, num_heads, head_dim)

        # RoPE
        if config.get("use_rope", True):
            freqs_cis = precompute_freqs_cis(head_dim, seq_len)
            wq, wk = apply_rotary_emb(wq, wk, freqs_cis)

        # Transpose (batch, seq, head, dim) to (batch, head, seq, dim)
        wq = jnp.transpose(wq, (0, 2, 1, 3))
        wk = jnp.transpose(wk, (0, 2, 1, 3))
        wv = jnp.transpose(wv, (0, 2, 1, 3))

        scores = jnp.matmul(wq, jnp.swapaxes(wk, -1, -2)) / jnp.sqrt(head_dim)

        # Causal Mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask == 0, -1e9, scores)

        attn_probs = nn.softmax(scores, axis=-1)
        attn_out = jnp.matmul(attn_probs, wv)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, embed_dim
        )
        attn_out = jnp.dot(attn_out, params.weights["attn_o"]) + params.biases["attn_o"]

        # Residual 1
        h1 = x_in + attn_out

        # ----------------------------------------------------------------------
        # Sub-Block 2: MLP
        # ----------------------------------------------------------------------
        # Layer Norm 2
        mu2 = jnp.mean(h1, axis=-1, keepdims=True)
        sigma2 = jnp.std(h1, axis=-1, keepdims=True)
        ln2 = (h1 - mu2) / (sigma2 + 1e-6)
        ln2 = ln2 * params.biases["ln2_scale"] + params.biases["ln2_bias"]

        # MLP
        mlp_hidden = jnp.dot(ln2, params.weights["mlp_in"]) + params.biases["mlp_in"]
        mlp_hidden = nn.gelu(mlp_hidden)
        mlp_out = (
            jnp.dot(mlp_hidden, params.weights["mlp_out"]) + params.biases["mlp_out"]
        )

        # Residual 2 (Prediction)
        z_mu = h1 + mlp_out

        # Error & State Update
        error = state.z_latent - z_mu

        # For complex nodes like this, we define gain_mod_error simply as error
        # The backpropagation through the complex 'z_mu' function above handles
        # the specific derivatives (attn, activations) via JAX autodiff
        # in the forward_inference/forward_learning wrappers.
        gain_mod_error = error

        state = state._replace(z_mu=z_mu, error=error, gain_mod_error=gain_mod_error)
        state = TransformerBlockNode.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


# To build deep transformer PC graphs
def create_deep_transformer(
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int = 10,
    vocab_size: int = None,
):
    nodes = []
    edges = []

    # Input Node
    if vocab_size is not None:
        nodes.append(
            {
                "name": "input_embed",
                "shape": (seq_len, embed_dim),
                "type": "embedding",
                "vocab_size": vocab_size,
                "embed_dim": embed_dim,
            }
        )
    else:
        nodes.append(
            {
                "name": "input_embed",
                "shape": (seq_len, embed_dim),
                "type": "linear",
                "activation": {"type": "identity"},
            }
        )

    # Stack Transformer Blocks
    previous_node = "input_embed"

    for i in range(depth):
        block_name = f"block_{i}"

        nodes.append(
            {
                "name": block_name,
                "shape": (seq_len, embed_dim),
                "type": "transformer_block",
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "mlp_dim": mlp_dim,
                "use_rope": True,
            }
        )

        # Connect previous -> current
        edges.append(
            {"source_name": previous_node, "target_name": block_name, "slot": "in"}
        )

        previous_node = block_name

    # Output Head
    nodes.append(
        {
            "name": "output",
            "shape": (seq_len, embed_dim),
            "type": "linear",
            "activation": {"type": "identity"},
        }
    )
    edges.append({"source_name": previous_node, "target_name": "output", "slot": "in"})

    return {
        "node_list": nodes,
        "edge_list": edges,
        "task_map": {"x": "input_embed", "y": "output"},
    }
