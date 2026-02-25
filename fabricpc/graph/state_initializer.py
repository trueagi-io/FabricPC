"""
Graph state initialization strategies for predictive coding networks.

This module provides:
- StateInitBase abstract class for graph-level state initialization
- Built-in strategies (Global, NodeDistribution, Feedforward)
- initialize_graph_state() convenience function

State initializers determine how latent states are initialized across
the entire graph before inference begins.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, List

import jax
import jax.numpy as jnp

from fabricpc.core.types import (
    GraphState,
    GraphStructure,
    GraphParams,
    NodeState,
)

# =============================================================================
# State Initializer Base Class
# =============================================================================


class StateInitBase(ABC):
    """
    Abstract base class for graph state initialization strategies.

    State initializers determine how latent states are initialized across
    the entire graph before inference begins.

    All methods are static for JAX compatibility (pure functions, no state).
    """

    @staticmethod
    @abstractmethod
    def initialize_state(
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        config: Dict[str, Any],
        params: GraphParams = None,
    ) -> GraphState:
        """
        Initialize graph state for inference. Apply data clamps to latent states if provided.

        Args:
            structure: Graph structure
            batch_size: Batch size
            rng_key: JAX random key
            clamps: Dictionary of clamped values, keyed on node names
            config: State initialization configuration
            params: GraphParams (may be required for some strategies)

        Returns:
            Initialized GraphState
        """
        pass


# =============================================================================
# State Initializer Registry (simplified)
# =============================================================================

_state_init_registry: Dict[str, Type[StateInitBase]] = {}


def register_state_init(init_type: str):
    """
    Decorator to register a state initializer.

    Args:
        init_type: Unique identifier for this state init type (case-insensitive)
    """

    def decorator(cls):
        _state_init_registry[init_type.lower()] = cls
        return cls

    return decorator


def get_state_init_class(init_type: str) -> Type[StateInitBase]:
    """
    Get a state initializer class by its registered type name.

    Args:
        init_type: The registered state init type (case-insensitive)

    Returns:
        The state initializer class

    Raises:
        ValueError: If state init type is not registered
    """
    cls = _state_init_registry.get(init_type.lower())
    if cls is None:
        available = list(_state_init_registry.keys())
        raise ValueError(
            f"Unknown state init type '{init_type}'. Available: {available}"
        )
    return cls


def list_state_init_types() -> List[str]:
    """Return list of all registered state init types."""
    return list(_state_init_registry.keys())


# =============================================================================
# Built-in State Initialization Strategies
# =============================================================================


@register_state_init("global")
class GlobalStateInit(StateInitBase):
    """
    Initialize states from a distribution.
    Each node's state is initialized using a graph-level initializer applied to all nodes.
    Processes nodes independently (no dependencies between nodes).

    Config options:
        - initializer: Initializer config for all nodes
                              (default: {"type": "normal", "mean": 0.0, "std": 0.05})
    """

    @staticmethod
    def initialize_state(
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        config: Dict[str, Any],
        params: GraphParams = None,
    ) -> GraphState:
        """Initialize states from a distribution."""
        from fabricpc.core.initializers import initialize, NormalInitializer

        global_init_config = config.get("initializer", None)
        if global_init_config is None:
            global_init_config = NormalInitializer(mean=0.0, std=0.05)

        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))
        node_key_map = dict(zip(node_names, node_keys))

        node_state_dict = {}

        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            shape = (batch_size, *node_info.shape)

            if node_name in clamps:
                z_latent = clamps[node_name]
            else:
                z_latent = initialize(
                    node_key_map[node_name], shape, global_init_config
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


@register_state_init("node_distribution")
class NodeDistributionStateInit(StateInitBase):
    """
    Initialize states from a distribution using node level configs for initializer.
    Each node's state is initialized using its configured latent_init.
    Processes nodes independently (no dependencies between nodes).
    """

    @staticmethod
    def initialize_state(
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        config: Dict[str, Any],
        params: GraphParams = None,
    ) -> GraphState:
        """Initialize states from a distribution."""
        from fabricpc.core.initializers import initialize

        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))
        node_key_map = dict(zip(node_names, node_keys))

        node_state_dict = {}

        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            shape = (batch_size, *node_info.shape)

            if node_name in clamps:
                z_latent = clamps[node_name]
            else:
                latent_init = node_info.latent_init
                z_latent = initialize(node_key_map[node_name], shape, latent_init)

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


@register_state_init("feedforward")
class FeedforwardStateInit(StateInitBase):
    """
    Initialize states via feedforward propagation through the network.

    1. Initialize source nodes and recurrency nodes with fallback to node's configured initializer
    2. Process nodes in topological order
    3. For each node, compute z_mu via forward pass and set z_latent = z_mu
    4. Clamps override computed values

    Requires params to be provided to compute projections.
    """

    @staticmethod
    def initialize_state(
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        config: Dict[str, Any],
        params: GraphParams = None,
    ) -> GraphState:
        """Initialize states via feedforward propagation."""
        from fabricpc.core.initializers import initialize
        from fabricpc.core.inference import gather_inputs
        from fabricpc.nodes.base import _get_node_class_from_info

        if params is None:
            raise ValueError("FeedforwardStateInit requires params to be provided")

        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))
        node_key_map = dict(zip(node_names, node_keys))

        # First pass: initialize all nodes with clamps or fallback in case of graph cycles
        node_state_dict = {}
        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            shape = (batch_size, *node_info.shape)

            if node_name in clamps:
                z_latent = clamps[node_name]
            else:
                latent_init = node_info.latent_init
                z_latent = initialize(node_key_map[node_name], shape, latent_init)

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

        # Second pass: feedforward propagation in topological order
        for node_name in structure.node_order:
            node = structure.nodes[node_name]
            node_info = node.node_info

            if node_info.in_degree > 0:
                node_state = state.nodes[node_name]
                node_params = params.nodes[node_name]
                node_class = _get_node_class_from_info(node_info)
                edge_inputs = gather_inputs(node_info, structure, state)

                _, projected = node_class.forward(
                    node_params, edge_inputs, node_state, node_info
                )
                # node forward modifies z_mu, pre_activation, error, and energy

                if node_name not in clamps:
                    # z_latent <- z_mu, error <- 0 (since z_latent = z_mu)
                    node_state = node_state._replace(
                        z_latent=projected.z_mu,
                        z_mu=projected.z_mu,
                    )  # leave energy and error already initialized to zeros

                else:
                    # Respect clamped values, retain newly computed error
                    node_state = node_state._replace(
                        z_latent=clamps[node_name],
                        z_mu=projected.z_mu,
                        error=projected.error,
                        energy=projected.energy,
                    )  # error and energy are valid for clamped nodes

                # Update state
                state = state._replace(nodes={**state.nodes, node_name: node_state})

        return state


# =============================================================================
# Convenience Functions
# =============================================================================


def initialize_graph_state(
    structure: GraphStructure,
    batch_size: int,
    rng_key: jax.Array,
    clamps: Dict[str, jnp.ndarray] = None,
    state_init_config: Dict[str, Any] = None,
    params: GraphParams = None,
) -> GraphState:
    """
    Initialize graph state using the specified strategy.

    Args:
        structure: Graph structure
        batch_size: Batch size
        rng_key: JAX random key
        clamps: Dictionary of clamped values
        state_init_config: State initialization config with "type" for a StateInitBase like object.
        params: GraphParams (required for feedforward init)

    Returns:
        Initialized GraphState

    Example:
        state = initialize_graph_state(
            structure, batch_size, key, clamps,
            {"type": "feedforward"},
            params=params
        )
    """
    clamps = clamps or {}

    if state_init_config is None:
        state_init_config = structure.config["graph_state_initializer"]

    init_type = state_init_config["type"]
    init_class = get_state_init_class(init_type)

    return init_class.initialize_state(
        structure, batch_size, rng_key, clamps, state_init_config, params
    )
