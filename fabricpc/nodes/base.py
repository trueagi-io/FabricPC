"""Base node classes for JAX predictive coding networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from fabricpc.core.types import NodeParams, NodeState, NodeInfo, SlotInfo, EdgeInfo
from fabricpc.core.activations import (
    ActivationBase,
    IdentityActivation,
    ensure_activation,
)
from fabricpc.core.energy import (
    EnergyFunctional,
    GaussianEnergy,
    get_energy_and_gradient,
)
from fabricpc.core.energy import ensure_energy
from fabricpc.core.initializers import (
    InitializerBase,
    NormalInitializer,
    ensure_initializer,
)


@dataclass(frozen=True)
class SlotSpec:
    """Specification for an input slot to a node."""

    name: str
    is_multi_input: bool


@dataclass(frozen=True)
class Slot:
    """Runtime slot information with connected edges."""

    spec: SlotSpec
    in_neighbors: Dict[str, str]


class FlattenInputMixin:
    """Mixin providing flatten/reshape utilities for dense nodes."""

    @staticmethod
    def flatten_input(x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    @staticmethod
    def reshape_output(x_flat: jnp.ndarray, out_shape: Tuple[int, ...]) -> jnp.ndarray:
        batch_size = x_flat.shape[0]
        return x_flat.reshape(batch_size, *out_shape)

    @staticmethod
    def compute_linear(
        inputs: Dict[str, jnp.ndarray],
        weights: Dict[str, jnp.ndarray],
        batch_size: int,
        out_shape: Tuple[int, ...],
    ) -> jnp.ndarray:
        import numpy as np

        out_numel = int(np.prod(out_shape))
        pre_activation_flat = jnp.zeros((batch_size, out_numel))

        for edge_key, x in inputs.items():
            x_flat = FlattenInputMixin.flatten_input(x)
            pre_activation_flat = pre_activation_flat + jnp.matmul(
                x_flat, weights[edge_key]
            )

        return FlattenInputMixin.reshape_output(pre_activation_flat, out_shape)


class NodeBase(ABC):
    """Abstract base class for predictive coding nodes."""

    DEFAULT_ENERGY: EnergyFunctional = GaussianEnergy()
    DEFAULT_ACTIVATION: ActivationBase = IdentityActivation()
    DEFAULT_LATENT_INIT: InitializerBase = NormalInitializer()

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        activation: ActivationBase | None = None,
        energy: EnergyFunctional | None = None,
        latent_init: InitializerBase | None = None,
        **node_config: Any,
    ):
        self.name = name
        self.shape = tuple(shape)
        self.activation = ensure_activation(
            activation if activation is not None else self.DEFAULT_ACTIVATION
        )
        self.energy = ensure_energy(
            energy if energy is not None else self.DEFAULT_ENERGY
        )
        self.latent_init = ensure_initializer(
            latent_init if latent_init is not None else self.DEFAULT_LATENT_INIT
        )
        self.node_config = dict(node_config)

        self.node_info = NodeInfo(
            name=self.name,
            shape=self.shape,
            node_type=self.__class__.__name__,
            node=self,
            activation=self.activation,
            energy=self.energy,
            latent_init=self.latent_init,
            node_config=self.node_config,
            slots={},
            in_degree=0,
            out_degree=0,
            in_edges=(),
            out_edges=(),
        )
        # Backward-compatible alias.
        self.info = self.node_info

    @staticmethod
    @abstractmethod
    def get_slots() -> Dict[str, SlotSpec]:
        pass

    @staticmethod
    @abstractmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        pass

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        node_cls = node_info.node.__class__

        if node_info.in_degree == 0:
            new_state = state._replace(
                z_mu=state.z_latent,
                error=jnp.zeros_like(state.error),
                pre_activation=jnp.zeros_like(state.pre_activation),
            )
            new_state = node_cls.energy_functional(new_state, node_info)
            input_grads = {
                edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs
            }

        elif node_info.out_degree == 0 and not is_clamped:
            _, new_state = node_cls.forward(params, inputs, state, node_info)
            new_state = new_state._replace(
                z_latent=new_state.z_mu,
                error=jnp.zeros_like(new_state.error),
                energy=jnp.zeros_like(new_state.energy),
                latent_grad=jnp.zeros_like(new_state.latent_grad),
            )
            input_grads = {
                edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs
            }

        else:
            (_, new_state), input_grads = jax.value_and_grad(
                node_cls.forward, argnums=1, has_aux=True
            )(params, inputs, state, node_info)

        return new_state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        node_cls = node_info.node.__class__
        (_, new_state), params_grad = jax.value_and_grad(
            node_cls.forward, argnums=0, has_aux=True
        )(params, inputs, state, node_info)
        return new_state, params_grad

    @staticmethod
    @abstractmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        pass

    @staticmethod
    def energy_functional(state: NodeState, node_info: NodeInfo) -> NodeState:
        energy_obj = node_info.energy
        if energy_obj is None:
            raise ValueError(
                f"graph was improperly constructed. Node {node_info.name} is missing energy object."
            )

        energy, grad = get_energy_and_gradient(state.z_latent, state.z_mu, energy_obj)
        latent_grad = state.latent_grad + grad
        return state._replace(energy=energy, latent_grad=latent_grad)

    def build_info(
        self,
        in_edges: Dict[str, EdgeInfo],
        out_edges: Dict[str, EdgeInfo],
    ) -> NodeInfo:
        slots = self._build_slots(self.name, in_edges)

        self.node_info = NodeInfo(
            name=self.name,
            shape=self.shape,
            node_type=self.__class__.__name__,
            node=self,
            activation=self.activation,
            energy=self.energy,
            latent_init=self.latent_init,
            node_config=self.node_config,
            slots=slots,
            in_degree=len(in_edges),
            out_degree=len(out_edges),
            in_edges=tuple(in_edges.keys()),
            out_edges=tuple(out_edges.keys()),
        )
        self.info = self.node_info
        return self.node_info

    @classmethod
    def _build_slots(
        cls, node_name: str, in_edges: Dict[str, EdgeInfo]
    ) -> Dict[str, SlotInfo]:
        slot_specs = cls.get_slots()
        slots: Dict[str, SlotInfo] = {}

        for slot_name, slot_spec in slot_specs.items():
            in_neighbors = [
                edge.source for edge in in_edges.values() if edge.slot == slot_name
            ]

            if not slot_spec.is_multi_input and len(in_neighbors) > 1:
                raise ValueError(
                    f"Slot '{slot_name}' in node '{node_name}' is single-input "
                    f"but has {len(in_neighbors)} connections"
                )

            slots[slot_name] = SlotInfo(
                name=slot_name,
                parent_node=node_name,
                is_multi_input=slot_spec.is_multi_input,
                in_neighbors=tuple(in_neighbors),
            )

        for edge_key, edge in in_edges.items():
            if edge.slot not in slots:
                raise ValueError(
                    f"Edge '{edge_key}' connects to non-existent slot '{edge.slot}' "
                    f"in node '{node_name}'. Available slots: {list(slots.keys())}"
                )

        return slots
