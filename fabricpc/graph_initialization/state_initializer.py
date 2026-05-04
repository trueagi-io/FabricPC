"""
Graph state initialization strategies for predictive coding networks.

This module provides:
- StateInitBase abstract class for graph-level state initialization
- Built-in strategies (Global, NodeDistribution, Feedforward)
- initialize_graph_state() convenience function

State initializers determine how latent states are initialized across
the entire graph before inference begins.

Usage:
    from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit

    structure = graph(
        nodes=[...], edges=[...], task_map=...,
        graph_state_initializer=FeedforwardStateInit(),
    )
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import jax
import jax.numpy as jnp

from fabricpc.core.inference import gather_inputs
from fabricpc.core.initializers import initialize, NormalInitializer
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.types import (
    GraphState,
    GraphStructure,
    GraphParams,
    NodeState,
)
from fabricpc.utils.helpers import set_latents_to_clamps

# =============================================================================
# State Initializer Base Class
# =============================================================================


class StateInitBase(ABC):
    """
    Abstract base class for graph state initialization strategies.

    State initializers determine how latent states are initialized across
    the entire graph before inference begins.

    Custom state initializers extend this class:

        class MyInit(StateInitBase):
            def __init__(self, fill_value=0.0):
                super().__init__(fill_value=fill_value)

            @staticmethod
            def initialize_state(structure, batch_size, rng_key, clamps, config, params=None):
                ...

    All computation methods are static for JAX compatibility (pure functions, no state).
    """

    def __init__(self, **config):
        self.config = config

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
            config: State initialization configuration (from instance .config)
            params: GraphParams (may be required for some strategies)

        Returns:
            Initialized GraphState
        """
        pass


# =============================================================================
# Built-in State Initialization Strategies
# =============================================================================


class GlobalStateInit(StateInitBase):
    """
    Initialize states from a distribution.
    Each node's state is initialized using a graph-level initializer applied to all nodes.
    Processes nodes independently (no dependencies between nodes).

    Args:
        initializer: InitializerBase instance for all nodes
                     (default: NormalInitializer(mean=0.0, std=0.05))
    """

    def __init__(self, initializer=None):
        super().__init__(initializer=initializer)

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
        global_init_config = config.get("initializer", None)
        if global_init_config is None:
            global_init_config = NormalInitializer(mean=0.0, std=0.05)

        node_names = list(structure.nodes.keys())
        rng_keys = jax.random.split(rng_key, len(node_names))
        rng_key_map = dict(zip(node_names, rng_keys))

        node_state_dict = {}

        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            shape = (batch_size, *node_info.shape)

            z_latent = initialize(rng_key_map[node_name], shape, global_init_config)
            dtype = (
                jnp.asarray(clamps[node_name]).dtype
                if node_name in clamps
                else z_latent.dtype
            )

            node_state_dict[node_name] = NodeState(
                z_latent=z_latent.astype(dtype),
                z_mu=jnp.zeros(shape, dtype=dtype),
                error=jnp.zeros(shape, dtype=dtype),
                energy=jnp.zeros((batch_size,), dtype=dtype),
                pre_activation=jnp.zeros(shape, dtype=dtype),
                latent_grad=jnp.zeros(shape, dtype=dtype),
            )

        state = GraphState(nodes=node_state_dict, batch_size=batch_size)
        return set_latents_to_clamps(state, clamps)


class NodeDistributionStateInit(StateInitBase):
    """
    Initialize states from a distribution using node level configs for initializer.
    Each node's state is initialized using its configured latent_init.
    Processes nodes independently (no dependencies between nodes).
    """

    def __init__(self):
        super().__init__()

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
        node_names = list(structure.nodes.keys())
        rng_keys = jax.random.split(rng_key, len(node_names))
        rng_key_map = dict(zip(node_names, rng_keys))

        node_state_dict = {}

        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            shape = (batch_size, *node_info.shape)

            latent_init = node_info.latent_init
            z_latent = initialize(rng_key_map[node_name], shape, latent_init)
            dtype = (
                jnp.asarray(clamps[node_name]).dtype
                if node_name in clamps
                else z_latent.dtype
            )

            node_state_dict[node_name] = NodeState(
                z_latent=z_latent.astype(dtype),
                z_mu=jnp.zeros(shape, dtype=dtype),
                error=jnp.zeros(shape, dtype=dtype),
                energy=jnp.zeros((batch_size,), dtype=dtype),
                pre_activation=jnp.zeros(shape, dtype=dtype),
                latent_grad=jnp.zeros(shape, dtype=dtype),
            )

        state = GraphState(nodes=node_state_dict, batch_size=batch_size)
        return set_latents_to_clamps(state, clamps)


class FeedforwardStateInit(StateInitBase):
    """
    Initialize states via feedforward propagation through the network.

    1. Initialize terminal input nodes (in_degree=0) and recurrency nodes with fallback to node's configured initializer
    2. Process nodes in topological order
    3. For each node, compute z_mu via forward pass and set z_latent = z_mu
    4. Clamps override computed values

    Requires params to be provided to compute projections.
    """

    def __init__(self):
        super().__init__()

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
        if params is None:
            raise ValueError("FeedforwardStateInit requires params to be provided")

        node_names = list(structure.nodes.keys())
        rng_keys = jax.random.split(rng_key, len(node_names))
        rng_key_map = dict(zip(node_names, rng_keys))

        # First pass: initialize all nodes with their default initializer (used as
        # fallback for graph cycles). Clamps are overlaid afterwards.
        node_state_dict = {}
        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            shape = (batch_size, *node_info.shape)

            latent_init = node_info.latent_init
            z_latent = initialize(rng_key_map[node_name], shape, latent_init)
            dtype = (
                jnp.asarray(clamps[node_name]).dtype
                if node_name in clamps
                else z_latent.dtype
            )

            node_state_dict[node_name] = NodeState(
                z_latent=z_latent.astype(dtype),
                z_mu=jnp.zeros(shape, dtype=dtype),
                error=jnp.zeros(shape, dtype=dtype),
                energy=jnp.zeros((batch_size,), dtype=dtype),
                pre_activation=jnp.zeros(shape, dtype=dtype),
                latent_grad=jnp.zeros(shape, dtype=dtype),
            )

        state = GraphState(nodes=node_state_dict, batch_size=batch_size)
        state = set_latents_to_clamps(state, clamps)

        # Second pass: feedforward propagation in topological order
        for node_name in structure.node_order:
            node = structure.nodes[node_name]
            node_info = node.node_info

            if node_info.in_degree > 0:
                node_state = state.nodes[node_name]
                node_params = params.nodes[node_name]
                node_class = node_info.node_class
                edge_inputs = gather_inputs(node_info, structure, state)

                # Apply muPC forward scaling (if any) before forward pass,
                # matching what the inference loop does during training.
                scaled_inputs = scale_inputs(edge_inputs, node_info.scaling_config)
                _, projected = node_class.forward(
                    node_params, scaled_inputs, node_state, node_info
                )
                # node forward modifies z_mu, pre_activation, error, and energy

                if node_name not in clamps:
                    # z_latent <- z_mu, error <- 0 (since z_latent = z_mu)
                    node_state = node_state._replace(
                        z_latent=projected.z_mu,
                        z_mu=projected.z_mu,
                    )  # leave energy and error already initialized to zeros

                else:
                    # z_latent already set to clamp by first pass + set_latents_to_clamps;
                    # retain newly computed z_mu/error/energy
                    node_state = node_state._replace(
                        z_mu=projected.z_mu,
                        error=projected.error,
                        energy=projected.energy,
                    )

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
    state_init: StateInitBase = None,
    params: GraphParams = None,
) -> GraphState:
    """
    Initialize graph state using the specified strategy.

    Args:
        structure: Graph structure
        batch_size: Batch size
        rng_key: JAX random key
        clamps: Dictionary of clamped values
        state_init: StateInitBase instance (default: from structure config)
        params: GraphParams (required for feedforward init)

    Returns:
        Initialized GraphState

    Example:
        state = initialize_graph_state(
            structure, batch_size, key, clamps,
            state_init=FeedforwardStateInit(),
            params=params
        )
    """
    clamps = clamps or {}

    if state_init is None:
        state_init = structure.config["graph_state_initializer"]

    return type(state_init).initialize_state(
        structure, batch_size, rng_key, clamps, state_init.config, params
    )
