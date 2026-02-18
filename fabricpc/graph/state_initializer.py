"""
Graph state initialization strategies for predictive coding networks.

This module provides:
- StateInitBase abstract class for graph-level state initialization
- Built-in strategies (Distribution, Feedforward)
- TODO add uPC initialization class
- Registry with decorator-based registration for custom strategies
- Entry point discovery for external packages

State initializers determine how latent states are initialized across
the entire graph before inference begins.

User Extensibility
------------------
Users can register custom state initializers in two ways:

1. **Decorator-based registration** (recommended for development):

    @register_state_init("my_strategy")
    class MyStateInit(StateInitBase):
        CONFIG_SCHEMA = {"param": {"type": float, "default": 1.0}}

        @staticmethod
        def initialize_state(structure, batch_size, rng_key, clamps, config, params=None):
            # Custom initialization logic
            ...

2. **Entry point discovery** (recommended for distribution):

    Add to pyproject.toml:
        [project.entry-points."fabricpc.state_initializers"]
        my_strategy = "my_package.state_init:MyStateInit"

Configuration
-------------
State initializers are configured via graph config:

    {
        "node_list": [...],
        "edge_list": [...],
        "graph_state_initializer": {
            "type": "feedforward",
        }
    }

Node-level overrides can be specified in node config:

    {
        "name": "hidden",
        "shape": (256,),
        "type": "linear",
        "latent_init": {"type": "uniform", "min": -0.5, "max": 0.5}
    }
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, List

import jax
import jax.numpy as jnp

from fabricpc.core.registry import Registry, RegistrationError, validate_config_schema
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

    Required methods:
        - initialize_state(): Initialize graph state

    Required attributes:
        - CONFIG_SCHEMA: dict specifying configuration validation

    Example implementation:
        @register_state_init("my_strategy")
        class MyStateInit(StateInitBase):
            CONFIG_SCHEMA = {
                "scale": {"type": float, "default": 1.0}
            }

            @staticmethod
            def initialize_state(structure, batch_size, rng_key, clamps, config, params=None):
                # Custom initialization logic
                ...
    """

    # CONFIG_SCHEMA is required - subclasses must define it
    # Use empty dict {} if no additional config parameters are needed
    CONFIG_SCHEMA: Dict[str, Dict[str, Any]]

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
# State Initializer Registry
# =============================================================================


class StateInitRegistrationError(RegistrationError):
    """Raised when state initializer registration fails."""

    pass


# Create the state initializer registry instance
_state_init_registry = Registry(
    name="state_init",
    entry_point_group="fabricpc.state_initializers",
    required_attrs=["CONFIG_SCHEMA"],
    required_methods=["initialize_state"],
    attr_validators={
        "CONFIG_SCHEMA": validate_config_schema,
    },
)
_state_init_registry.set_error_class(StateInitRegistrationError)


def register_state_init(init_type: str):
    """
    Decorator to register a state initializer with the registry.

    Usage:
        @register_state_init("distribution")
        class GlobalStateInit(StateInitBase):
            ...

    Args:
        init_type: Unique identifier for this state init type (case-insensitive)

    Returns:
        Decorator function

    Raises:
        StateInitRegistrationError: If registration fails (duplicate, missing methods)
    """
    return _state_init_registry.register(init_type)


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
    return _state_init_registry.get(init_type)


def list_state_init_types() -> List[str]:
    """Return list of all registered state init types."""
    return _state_init_registry.list_types()


def unregister_state_init(init_type: str) -> None:
    """
    Remove a state init type from the registry.
    Primarily for testing purposes.

    Args:
        init_type: The state init type to unregister (case-insensitive)
    """
    _state_init_registry.unregister(init_type)


def clear_state_init_registry() -> None:
    """Clear all registrations. For testing only."""
    _state_init_registry.clear()


def validate_state_init_config(
    state_init_class: Type[StateInitBase], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and apply defaults from state_init's CONFIG_SCHEMA.

    Args:
        state_init_class: The state init class with CONFIG_SCHEMA
        config: The user-provided config dict

    Returns:
        Config dict with defaults applied

    Raises:
        ConfigValidationError: If required fields are missing or validation fails
    """
    from fabricpc.core.config import validate_config

    schema = getattr(state_init_class, "CONFIG_SCHEMA", None)
    init_type = config.get("type", "unknown") if config else "unknown"
    return validate_config(schema, config, context=f"state_init '{init_type}'")


def discover_external_state_inits() -> None:
    """
    Discover and register state initializers from installed packages via entry points.

    Looks for packages with entry points in the "fabricpc.state_initializers" group.
    Each entry point should map a state init type name to a StateInitBase subclass.

    Example pyproject.toml for an external package:
        [project.entry-points."fabricpc.state_initializers"]
        custom_init = "my_package.state_init:CustomStateInit"
    """
    _state_init_registry.discover_external()


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
        - initializer: Global initializer config for all nodes
                              (default: {"type": "normal", "mean": 0.0, "std": 0.05})
    """

    CONFIG_SCHEMA = {
        "initializer": {
            "type": dict,
            "default": {"type": "normal", "mean": 0.0, "std": 0.05},
            "description": "Global initializer config for all nodes, any type supported by fabricpc.core.initializers.initialize",
        },
    }

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

        global_init_config = config["initializer"]

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
    Each node's state is initialized using its specified Initializer.
    Processes nodes independently (no dependencies between nodes).

    Config options:
    none
    """

    CONFIG_SCHEMA = {}  # No additional config options

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

        for node_name, node_info in structure.nodes.items():
            shape = (batch_size, *node_info.shape)

            if node_name in clamps:
                z_latent = clamps[node_name]
            else:
                node_init_config = node_info.node_config["latent_init"]
                z_latent = initialize(node_key_map[node_name], shape, node_init_config)

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

    Config options:
    none
    """

    CONFIG_SCHEMA = {}  # No additional config options

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
        from fabricpc.nodes import get_node_class

        if params is None:
            raise ValueError("FeedforwardStateInit requires params to be provided")

        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))
        node_key_map = dict(zip(node_names, node_keys))

        # First pass: initialize all nodes with clamps or fallback in case of graph cycles
        node_state_dict = {}
        for node_name, node_info in structure.nodes.items():
            shape = (batch_size, *node_info.shape)

            if node_name in clamps:
                z_latent = clamps[node_name]
            else:
                fallback_config = node_info.node_config["latent_init"]
                z_latent = initialize(node_key_map[node_name], shape, fallback_config)

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
            node_info = structure.nodes[node_name]

            if node_info.in_degree > 0:
                node_state = state.nodes[node_name]
                node_params = params.nodes[node_name]
                node_class = get_node_class(node_info.node_type)
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
            {"type": "feedforward", "fallback": {"type": "zeros"}},
            params=params
        )
    """
    clamps = clamps or {}

    if state_init_config is None:
        state_init_config = structure.config["graph_state_initializer"]

    init_type = state_init_config["type"]
    init_class = get_state_init_class(init_type)

    validated_config = validate_state_init_config(init_class, state_init_config)

    return init_class.initialize_state(
        structure, batch_size, rng_key, clamps, validated_config, params
    )


# Auto-discover external state initializers on import
discover_external_state_inits()
