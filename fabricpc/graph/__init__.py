"""JAX graph for predictive coding networks."""

from fabricpc.graph.graph_net import (
    initialize_params,
    set_latents_to_clamps,
    compute_local_weight_gradients,
)

from fabricpc.graph.state_initializer import (
    StateInitBase,
    GlobalStateInit,
    NodeDistributionStateInit,
    FeedforwardStateInit,
    initialize_graph_state,
)

# TODO rename this module to something more specific about initializing the graph state and parameters, like graph_initialization.py or graph_utils.py.

__all__ = [
    # Graph construction
    "initialize_params",
    "set_latents_to_clamps",
    "compute_local_weight_gradients",
    # State initializers
    "StateInitBase",
    "GlobalStateInit",
    "NodeDistributionStateInit",
    "FeedforwardStateInit",
    "initialize_graph_state",
]
