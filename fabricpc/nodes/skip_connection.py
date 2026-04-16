"""
Skip connection node for residual architectures.

SkipConnection is identical to IdentityNode in behavior — it sums inputs
and passes them through — but sets ``apply_variance_scaling = False``.
This tells muPC to leave edges into this node unscaled (scale = 1.0),
preserving the identity mapping that carries signal through deep networks.

Use SkipConnection for residual/skip paths in your graph. Use IdentityNode
for summation points where all inputs are independent and should be
variance-scaled (e.g., multi-source aggregation).

Example — a residual block in graph form::

    linear = Linear(shape=(128,), name="h1")
    skip   = SkipConnection(shape=(128,), name="res1")

    edges = [
        Edge(source=prev, target=linear.slot("in")),   # transform path
        Edge(source=prev, target=skip.slot("in")),      # skip path (unscaled)
        Edge(source=linear, target=skip.slot("in")),    # transform -> sum
    ]
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


class SkipConnection(NodeBase):
    """
    Skip connection node: sums inputs without muPC variance scaling.

    Identical to IdentityNode in forward behavior (sums all inputs, no
    learnable parameters), but ``apply_variance_scaling = False`` tells
    muPC to leave incoming edges at scale 1.0.

    This preserves the identity mapping through deep residual networks.
    Without this, muPC's in-degree formula scales skip edges by
    1/sqrt(K), causing exponential signal decay (0.707^L for K=2).
    """

    apply_variance_scaling: bool = False

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """No weight matrix — return 1."""
        return 1

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """Sum all inputs and pass through (no transformation)."""
        z_mu = None
        for edge_key, x in inputs.items():
            if z_mu is None:
                z_mu = x
            else:
                z_mu = z_mu + x

        error = state.z_latent - z_mu
        state = state._replace(
            pre_activation=z_mu,
            z_mu=z_mu,
            error=error,
        )

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        return total_energy, state
