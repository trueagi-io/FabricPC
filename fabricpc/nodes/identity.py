"""
Identity node implementation for JAX predictive coding networks.

The IdentityNode passes input through unchanged with no transformation and no
learnable parameters. This is useful for input nodes or auxiliary nodes that
serve as conduits for data without learning.

When multiple inputs are connected, they are summed together.
"""

"""                                                                                                                                                                                                                                                                                             adding enough so as to invoke pre-commit checks which got skipped                                                                                                                                                                                                                               """

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.registry import register_node
from fabricpc.core.types import NodeParams, NodeState, NodeInfo


@register_node("identity")
class IdentityNode(NodeBase):
    """
    Identity node: passes input through unchanged.

    This node type:
    - Has a single multi-input slot named "in"
    - Sums all inputs (if multiple) and passes through as z_mu
    - Has no learnable parameters (no weights, no biases)
    - Useful for input nodes or passthrough connections
    """

    CONFIG_SCHEMA: Dict[str, Any] = {}

    DEFAULT_ENERGY_CONFIG: Dict[str, Any] = {"type": "gaussian"}
    DEFAULT_ACTIVATION_CONFIG: Dict[str, Any] = {"type": "identity"}

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Identity nodes have a single multi-input slot.

        Returns:
            Dictionary with one slot "in" that accepts multiple inputs
        """
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Initialize parameters for identity node (none needed).

        Args:
            key: JAX random key (unused)
            node_shape: Output shape of this node (unused)
            input_shapes: Dictionary with edge keys to source shapes (unused)
            config: Node configuration (unused)

        Returns:
            NodeParams with empty weights and biases
        """
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Identity forward pass: sum inputs and pass through.

        For source nodes (no inputs), z_mu equals z_latent.
        For nodes with inputs, z_mu is the sum of all inputs.

        Args:
            params: Node parameters (empty for identity node)
            inputs: Dictionary mapping edge keys to input tensors
            state: Current node state
            node_info: NodeInfo object

        Returns:
            Tuple of (total_energy, updated NodeState)
        """
        from fabricpc.nodes import get_node_class

        node_class = get_node_class(node_info.node_type)

        if node_info.in_degree == 0:
            # Source node: no inputs, prediction equals latent state
            z_mu = state.z_latent
            pre_activation = jnp.zeros_like(state.z_latent)
            error = jnp.zeros_like(state.z_latent)
        else:
            # Sum all inputs
            z_mu = None
            for edge_key, x in inputs.items():
                if z_mu is None:
                    z_mu = x
                else:
                    z_mu = z_mu + x

            # For identity node, pre_activation equals z_mu (no activation transform)
            pre_activation = z_mu

            # Compute prediction error
            error = state.z_latent - z_mu

        # Update node state
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error,
        )

        # Compute energy using the energy functional
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state
