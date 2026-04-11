"""
Identity node implementation for JAX predictive coding networks.

The IdentityNode passes input through unchanged with no transformation, no
activation and no learnable parameters. This is useful for input nodes or
auxiliary nodes that serve as conduits for data without learning.

When multiple inputs are connected, they are summed together.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class IdentityNode(NodeBase):
    """
    Identity node: passes input through unchanged.

    This node type:
    - Has a single multi-input slot named "in"
    - Sums all inputs (if multiple) and passes through as z_mu
    - Has no learnable parameters (no weights, no biases)
    - Useful for input nodes or passthrough connections
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
        scale: float = 1.0,  # Optional fixed scaling factor of the node output (default 1.0, no scaling)
    ):
        """
        Args:
            shape: Output shape tuple (excluding batch dimension)
            name: Node name
            activation: ActivationBase instance (default: IdentityActivation)
            energy: EnergyFunctional instance (default: GaussianEnergy)
            latent_init: InitializerBase instance for latent states
        """
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            scale=scale,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Identity nodes have a single multi-input slot.

        Returns:
            Dictionary with one slot "in" that accepts multiple inputs
        """
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def has_weights() -> bool:
        """IdentityNode has no learnable weights."""
        return False

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize parameters for identity node (none needed).

        Args:
            key: JAX random key (unused)
            node_shape: Output shape of this node (unused)
            input_shapes: Dictionary with edge keys to source shapes (unused)
            weight_init: InitializerBase instance (unused, identity has no weights)
            config: Node configuration dictionary

        Returns:
            NodeParams with empty weights and biases
        """
        if config is None:
            config = {}
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
        # Sum all inputs
        z_mu = None
        for edge_key, x in inputs.items():
            if z_mu is None:
                z_mu = x
            else:
                z_mu = z_mu + x

        z_mu = (
            z_mu * node_info.node_config["scale"]
        )  # Apply fixed scaling factor (default is 1.0)

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
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state
