"""JAX graph for predictive coding networks."""

from fabricpc.graph_initialization.graph_net import (
    initialize_params,
    set_latents_to_clamps,
    compute_local_weight_gradients,
)

from fabricpc.graph_initialization.state_initializer import (
    StateInitBase,
    GlobalStateInit,
    NodeDistributionStateInit,
    FeedforwardStateInit,
    initialize_graph_state,
)

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
