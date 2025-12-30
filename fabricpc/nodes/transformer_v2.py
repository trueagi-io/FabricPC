"""
Transformer components for JAX predictive coding networks.

This module implements:
- EmbeddingNode: Learned vector lookup
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.registry import register_node
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.initialization import initialize_weights

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
