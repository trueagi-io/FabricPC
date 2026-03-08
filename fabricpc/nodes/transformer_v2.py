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
from fabricpc.core.initializers import initialize
from fabricpc.core.positional import precompute_freqs_cis, apply_rotary_emb
from fabricpc.core.activations import get_activation
from fabricpc.utils.helpers import layernorm

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
        weights = {
            "embeddings": initialize(
                key, (config["vocab_size"], config["embed_dim"]), config["weight_init"]
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

        # Move gain_mod_error to substructure
        state = state._replace(
            z_mu=z_mu, error=error, substructure={"gain_mod_error": gain_mod_error}
        )

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
# MULTI-HEAD ATTENTION + RESIDUAL NODE
# ==============================================================================


@register_node("mha_residual")
class MhaResidualNode(NodeBase):
    """
    Implements: x + Attention(LayerNorm(x))
    """

    CONFIG_SCHEMA = {
        "embed_dim": {"type": int, "required": True},
        "num_heads": {"type": int, "required": True},
        "use_rope": {"type": bool, "default": True},
        "rope_theta": {"type": float, "default": 10000.0},
        "is_causal": {"type": bool, "default": True},
        "weight_init": {"type": dict, "default": {"type": "xavier"}},
    }
    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False), "mask": SlotSpec("mask", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        dim = config["embed_dim"]
        keys = jax.random.split(key, 6)

        # Use config provided weight_init
        weight_init = config.get("weight_init", {"type": "xavier"})

        def init_w(k, s):
            return initialize(k, s, weight_init)

        weights = {
            "ln_gamma": jnp.ones((dim,)),
            "W_q": init_w(keys[0], (dim, dim)),
            "W_k": init_w(keys[1], (dim, dim)),
            "W_v": init_w(keys[2], (dim, dim)),
            "W_o": init_w(keys[3], (dim, dim)),
        }
        biases = {
            "ln_beta": jnp.zeros((dim,)),
            "b_q": jnp.zeros((dim,)),
            "b_k": jnp.zeros((dim,)),
            "b_v": jnp.zeros((dim,)),
            "b_o": jnp.zeros((dim,)),
        }
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = inputs[next(k for k in inputs if k.endswith(":in"))]
        mask_key = next((k for k in inputs if k.endswith(":mask")), None)
        external_mask = inputs[mask_key] if mask_key else None

        cfg = node_info.node_config
        B, L, D = x.shape
        num_heads = cfg["num_heads"]
        head_dim = D // num_heads

        # LayerNorm
        x_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])

        # Projections
        def proj(h, w_name, b_name):
            return jnp.dot(h, params.weights[w_name]) + params.biases[b_name]

        Q = proj(x_norm, "W_q", "b_q").reshape(B, L, num_heads, head_dim)
        K = proj(x_norm, "W_k", "b_k").reshape(B, L, num_heads, head_dim)
        V = proj(x_norm, "W_v", "b_v").reshape(B, L, num_heads, head_dim)

        # RoPE
        if cfg.get("use_rope"):
            freqs_cis = precompute_freqs_cis(head_dim, L, theta=cfg["rope_theta"])
            Q, K = apply_rotary_emb(Q, K, freqs_cis)

        # Attention
        Q, K, V = (
            Q.transpose(0, 2, 1, 3),
            K.transpose(0, 2, 1, 3),
            V.transpose(0, 2, 1, 3),
        )
        scores = jnp.matmul(Q, K.swapaxes(-1, -2)) / jnp.sqrt(head_dim)

        # Causal Masking
        if cfg.get("is_causal", True):
            causal_mask = jnp.tril(jnp.ones((L, L)))
            scores = jnp.where(causal_mask == 0, -1e9, scores)

        if external_mask is not None:
            scores = jnp.where(external_mask == 0, -1e9, scores)

        attn = nn.softmax(scores, axis=-1)
        mha = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, L, D)

        # Output Projection
        mha = proj(mha, "W_o", "b_o")

        # Residual
        z_mu = x + mha
        error = state.z_latent - z_mu

        # Store internals for gradient computation
        substructure = {
            "gain_mod_error": error,
            "attn_matrix": attn,
            "Q": Q,
            "K": K,
            "V": V,
        }

        state = state._replace(z_mu=z_mu, error=error, substructure=substructure)
        state = MhaResidualNode.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


# ==============================================================================
# LAYERNORM + MLP1 NODE
# ==============================================================================


@register_node("ln_mlp1")
class LnMlp1Node(NodeBase):
    """
    Implements: Activation(LayerNorm(x) * W1 + b1)
    """

    CONFIG_SCHEMA = {
        "embed_dim": {"type": int, "required": True},
        "ff_dim": {"type": int, "required": True},
        "activation": {"type": str, "default": "gelu"},
        "weight_init": {"type": dict, "default": {"type": "kaiming"}},
    }
    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        embed_dim, ff_dim = config["embed_dim"], config["ff_dim"]
        keys = jax.random.split(key, 2)

        # Use config provided weight_init
        weight_init = config.get("weight_init", {"type": "kaiming"})

        weights = {
            "ln_gamma": jnp.ones((embed_dim,)),
            "W_ff1": initialize(keys[0], (embed_dim, ff_dim), weight_init),
        }
        biases = {
            "ln_beta": jnp.zeros((embed_dim,)),
            "b_ff1": jnp.zeros((ff_dim,)),
        }
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = inputs[list(inputs.keys())[0]]
        x_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])

        # Linear 1
        h = jnp.dot(x_norm, params.weights["W_ff1"]) + params.biases["b_ff1"]

        # Activation
        act_fn, activation_deriv = get_activation(node_info.node_config["activation"])
        z_mu = act_fn(h)

        error = state.z_latent - z_mu

        # Compute gain modulation (activations usually require f'(h) * error)
        # Note: In standard PC, this is usually done during back-propagation of gradients,
        # but storing it here helps explicit gradient calculation.
        f_prime = activation_deriv(h)
        gain_mod_error = error * f_prime

        state = state._replace(
            z_mu=z_mu, error=error, substructure={"gain_mod_error": gain_mod_error}
        )
        state = LnMlp1Node.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


# ==============================================================================
# MLP2 + RESIDUAL NODE
# ==============================================================================


@register_node("mlp2_residual")
class Mlp2ResidualNode(NodeBase):
    """
    Implements: residual_input + (mlp1_output * W2 + b2)
    """

    CONFIG_SCHEMA = {
        "embed_dim": {"type": int, "required": True},
        "ff_dim": {"type": int, "required": True},
        "weight_init": {"type": dict, "default": {"type": "xavier"}},
    }
    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False), "residual": SlotSpec("residual", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        embed_dim, ff_dim = config["embed_dim"], config["ff_dim"]

        # Use config provided weight_init
        weight_init = config.get("weight_init", {"type": "xavier"})

        weights = {"W_ff2": initialize(key, (ff_dim, embed_dim), weight_init)}
        biases = {"b_ff2": jnp.zeros((embed_dim,))}
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        # Identify inputs
        mlp1_in = next(val for key, val in inputs.items() if key.endswith(":in"))
        res_in = next(val for key, val in inputs.items() if key.endswith(":residual"))

        # Linear 2
        mlp2 = jnp.dot(mlp1_in, params.weights["W_ff2"]) + params.biases["b_ff2"]

        # Residual
        z_mu = res_in + mlp2

        error = state.z_latent - z_mu
        state = state._replace(
            z_mu=z_mu, error=error, substructure={"gain_mod_error": error}
        )
        state = Mlp2ResidualNode.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


# ==============================================================================
# VOCAB PROJECTION NODE
# ==============================================================================


@register_node("vocab_projection")
class VocabProjectionNode(NodeBase):
    CONFIG_SCHEMA = {
        "vocab_size": {"type": int, "required": True},
        "embed_dim": {"type": int, "required": True},
        "weight_init": {"type": dict, "default": {"type": "xavier"}},
    }
    DEFAULT_ENERGY_CONFIG = {"type": "kl_divergence"}

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, config):
        vocab, dim = config["vocab_size"], config["embed_dim"]

        # Use config provided weight_init
        weight_init = config.get("weight_init", {"type": "xavier"})

        weights = {"W_out": initialize(key, (dim, vocab), weight_init)}
        biases = {"b_out": jnp.zeros((vocab,))}
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = inputs[list(inputs.keys())[0]]
        logits = jnp.dot(x, params.weights["W_out"]) + params.biases["b_out"]

        # Softmax (Prediction is probability distribution)
        z_mu = nn.softmax(logits, axis=-1)

        error = state.z_latent - z_mu
        state = state._replace(
            z_mu=z_mu, error=error, substructure={"gain_mod_error": error}
        )
        state = VocabProjectionNode.energy_functional(state, node_info)
        return jnp.sum(state.energy), state
