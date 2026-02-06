"""JAX graph for predictive coding networks."""

from fabricpc.graph.graph_net import (
    build_graph_structure,
    initialize_params,
    create_pc_graph,
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
    "build_graph_structure",
    "initialize_params",
    "create_pc_graph",
    "set_latents_to_clamps",
    "compute_local_weight_gradients",
    # State initializer registry
    "StateInitBase",
    "register_state_init",
    "get_state_init_class",
    "list_state_init_types",
    "initialize_graph_state",
]
