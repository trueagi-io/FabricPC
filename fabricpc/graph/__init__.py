"""JAX graph for predictive coding networks."""

from fabricpc.graph.graph_net import (
    initialize_params,
    set_latents_to_clamps,
    compute_local_weight_gradients,
)

from fabricpc.graph.state_initializer import (
    StateInitBase,
    register_state_init,
    get_state_init_class,
    list_state_init_types,
    initialize_graph_state,
)

__all__ = [
    # Graph construction
    "initialize_params",
    "set_latents_to_clamps",
    "compute_local_weight_gradients",
    # State initializer registry
    "StateInitBase",
    "register_state_init",
    "get_state_init_class",
    "list_state_init_types",
    "initialize_graph_state",
]
