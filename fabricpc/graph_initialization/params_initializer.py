from typing import Dict
import jax
from fabricpc.core.types import (
    NodeParams,
    GraphParams,
    GraphStructure,
)


def initialize_params(
    structure: GraphStructure,
    rng_key: jax.Array,  # from jax.random.PRNGKey
) -> GraphParams:
    """
    Initialize model parameters for every node in the graph structure.

    Each node class handles its own parameter initialization,
    supporting complex nodes with multiple internal parameters.

    Args:
        structure: Graph structure
        rng_key: JAX random key

    Returns:
        GraphParams with node-based parameter organization
    """
    node_params = {}  # type: Dict[str, NodeParams]

    # Split key for each node
    num_nodes = len([n for n in structure.nodes.values() if n.node_info.in_degree > 0])
    if num_nodes > 0:
        keys = jax.random.split(rng_key, num_nodes)
    else:
        keys = []
    key_idx = 0

    for node_name, node in structure.nodes.items():
        node_info = node.node_info
        # Skip terminal input nodes (no parameters)
        if node_info.in_degree == 0:
            node_params[node_name] = NodeParams(weights={}, biases={})
            continue

        node_class = node_info.node_class

        # Get the input shapes for each edge (full shapes for conv support)
        input_shapes = {}
        for edge_key in node_info.in_edges:
            edge_info = structure.edges[edge_key]
            source_node = structure.nodes[edge_info.source]
            input_shapes[edge_key] = source_node.node_info.shape

        # Initialize parameters of the node
        params_obj = node_class.initialize_params(
            keys[key_idx],
            node_info.shape,
            input_shapes,
            node_info.weight_init,
            node_info.node_config,
        )
        key_idx += 1
        node_params[node_name] = params_obj

    return GraphParams(nodes=node_params)
