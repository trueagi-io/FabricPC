"""Graph state initialization strategies for predictive coding networks."""

from abc import ABC, abstractmethod
from typing import Dict, Any

import jax
import jax.numpy as jnp

from fabricpc.core.types import GraphState, GraphStructure, GraphParams, NodeState
from fabricpc.core.initializers import InitializerBase, NormalInitializer, initialize


class StateInitBase(ABC):
    """Abstract base class for graph state initialization strategies."""

    @abstractmethod
    def initialize_state(
        self,
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        params: GraphParams = None,
    ) -> GraphState:
        pass


class GlobalStateInit(StateInitBase):
    """Initialize all nodes from one shared initializer."""

    def __init__(self, initializer: InitializerBase | None = None):
        self.initializer = initializer or NormalInitializer()

    def initialize_state(
        self,
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        params: GraphParams = None,
    ) -> GraphState:
        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))
        node_key_map = dict(zip(node_names, node_keys))

        node_state_dict = {}
        for node_name, node_info in structure.nodes.items():
            shape = (batch_size, *node_info.shape)

            if node_name in clamps:
                z_latent = clamps[node_name]
            else:
                z_latent = initialize(node_key_map[node_name], shape, self.initializer)

            node_state_dict[node_name] = NodeState(
                z_latent=z_latent,
                z_mu=jnp.zeros(shape),
                error=jnp.zeros(shape),
                energy=jnp.zeros((batch_size,)),
                pre_activation=jnp.zeros(shape),
                latent_grad=jnp.zeros(shape),
                substructure={},
            )

        return GraphState(nodes=node_state_dict, batch_size=batch_size)


class NodeDistributionStateInit(StateInitBase):
    """Initialize each node from its own node-configured latent initializer."""

    def initialize_state(
        self,
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        params: GraphParams = None,
    ) -> GraphState:
        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))
        node_key_map = dict(zip(node_names, node_keys))

        node_state_dict = {}

        for node_name, node_info in structure.nodes.items():
            shape = (batch_size, *node_info.shape)

            if node_name in clamps:
                z_latent = clamps[node_name]
            else:
                z_latent = initialize(
                    node_key_map[node_name], shape, node_info.latent_init
                )

            node_state_dict[node_name] = NodeState(
                z_latent=z_latent,
                z_mu=jnp.zeros(shape),
                error=jnp.zeros(shape),
                energy=jnp.zeros((batch_size,)),
                pre_activation=jnp.zeros(shape),
                latent_grad=jnp.zeros(shape),
                substructure={},
            )

        return GraphState(nodes=node_state_dict, batch_size=batch_size)


class FeedforwardStateInit(StateInitBase):
    """Initialize states via feedforward propagation through the graph."""

    def initialize_state(
        self,
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        params: GraphParams = None,
    ) -> GraphState:
        from fabricpc.core.inference import gather_inputs

        if params is None:
            raise ValueError("FeedforwardStateInit requires params to be provided")

        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))
        node_key_map = dict(zip(node_names, node_keys))

        node_state_dict = {}
        for node_name, node_info in structure.nodes.items():
            shape = (batch_size, *node_info.shape)

            if node_name in clamps:
                z_latent = clamps[node_name]
            else:
                z_latent = initialize(
                    node_key_map[node_name], shape, node_info.latent_init
                )

            node_state_dict[node_name] = NodeState(
                z_latent=z_latent,
                z_mu=jnp.zeros(shape),
                error=jnp.zeros(shape),
                energy=jnp.zeros((batch_size,)),
                pre_activation=jnp.zeros(shape),
                latent_grad=jnp.zeros(shape),
                substructure={},
            )

        state = GraphState(nodes=node_state_dict, batch_size=batch_size)

        for node_name in structure.node_order:
            node_info = structure.nodes[node_name]

            if node_info.in_degree > 0:
                node_state = state.nodes[node_name]
                node_params = params.nodes[node_name]
                edge_inputs = gather_inputs(node_info, structure, state)

                _, projected = node_info.node.__class__.forward(
                    node_params, edge_inputs, node_state, node_info
                )

                if node_name not in clamps:
                    node_state = node_state._replace(
                        z_latent=projected.z_mu,
                        z_mu=projected.z_mu,
                    )
                else:
                    node_state = node_state._replace(
                        z_latent=clamps[node_name],
                        z_mu=projected.z_mu,
                        error=projected.error,
                        energy=projected.energy,
                    )

                state = state._replace(nodes={**state.nodes, node_name: node_state})

        return state


def initialize_graph_state(
    structure: GraphStructure,
    batch_size: int,
    rng_key: jax.Array,
    clamps: Dict[str, jnp.ndarray] = None,
    state_init_config: Any = None,
    params: GraphParams = None,
) -> GraphState:
    """Initialize graph state using a concrete StateInitBase object."""
    clamps = clamps or {}

    state_init = state_init_config
    if state_init is None:
        state_init = structure.config.get("graph_state_initializer")
    if state_init is None:
        state_init = FeedforwardStateInit()

    if isinstance(state_init, dict):
        raise TypeError(
            "state_init_config must be a StateInitBase instance; string/type lookup is removed"
        )
    if not isinstance(state_init, StateInitBase):
        raise TypeError("state_init_config must be a StateInitBase instance")

    return state_init.initialize_state(
        structure=structure,
        batch_size=batch_size,
        rng_key=rng_key,
        clamps=clamps,
        params=params,
    )
