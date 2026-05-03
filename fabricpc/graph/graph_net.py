"""
Graph-based predictive coding network construction for JAX with local Hebbian learning.

This module provides functions to initialize parameters at the node level
and compute local weight gradients for Hebbian learning.

Edges direct the flow of information through the graph in inference and training.
- Node output is passed to input slots of post-synaptic nodes
- Node gets gradient contributions from post-synaptic nodes by querying its out_neighbors
  on each outgoing edge for the gradient contributions of that particular edge.
"""

from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from fabricpc.core.types import (
    NodeParams,
    GraphParams,
    NodeState,
    GraphState,
    GraphStructure,
)
from fabricpc.utils.helpers import update_node_in_state
from fabricpc.core.inference import gather_inputs
from fabricpc.core.scaling import scale_inputs, scale_weight_grads


# TODO move this method to a new file learning.py in the core/ module and remove this file.
def compute_local_weight_gradients(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """
    Compute local weight gradients for each node using its own error signal.

    This implements the local Hebbian learning rule for predictive coding.
    muPC scaling is applied here (pre-scale inputs, post-scale weight
    gradients), keeping node methods (forward_and_weight_grads) scaling-unaware.

    Args:
        params: Current model parameters
        final_state: Converged state after inference
        structure: Graph structure

    Returns:
        GraphParams containing gradients for the parameters
    """
    gradients = {}

    for node_name, node in structure.nodes.items():
        node_info = node.node_info
        # Source nodes have no weights, but need empty gradient dict for consistency
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        in_edges_data = gather_inputs(node_info, structure, final_state)

        node_class = node_info.node_class
        sc = node_info.scaling_config

        # Pre-scale inputs by muPC forward scaling factors
        scaled_inputs = scale_inputs(in_edges_data, sc)

        # Compute local gradients using node's method (pure autodiff)
        node_state, grad_params = node_class.forward_and_weight_grads(
            params.nodes[node_name],
            scaled_inputs,
            final_state.nodes[node_name],
            node_info,
        )

        # Post-scale weight gradients by muPC factors
        grad_params = scale_weight_grads(grad_params, sc)

        # Store gradients
        gradients[node_name] = grad_params

    # convert to GraphParams
    params_gradients = GraphParams(nodes=gradients)

    return params_gradients


# TODO move to a new file params_initializer.py.
def initialize_params(
    structure: GraphStructure,
    rng_key: jax.Array,  # from jax.random.PRNGKey
) -> GraphParams:
    """
    Initialize model parameters at the node level.

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


# TODO - move this to utils. Call this function to replace boilerplate code for clamping latents in training loops.
def set_latents_to_clamps(
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
) -> GraphState:
    """
    Set the latent states of specified nodes to their clamped values.

    Args:
        state: Current graph state
        clamps: Dictionary of clamped values, keyed on node names

    Returns:
        Updated GraphState with latents set to clamped values
    """
    for node_name, clamp_value in clamps.items():
        if node_name in state.nodes:
            state = update_node_in_state(
                state, node_name, z_latent=jnp.asarray(clamp_value, dtype=jnp.float32)
            )
    return state
