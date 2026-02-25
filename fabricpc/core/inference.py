"""
Core inference dynamics for JAX predictive coding networks with local Hebbian learning.

This module implements the functional inference loop that updates latent states
using local gradients computed via Jacobian for true predictive coding.
"""

from typing import Dict
import jax
import jax.numpy as jnp

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.types import NodeInfo
from fabricpc.nodes.base import _get_node_class_from_info
from fabricpc.utils.helpers import update_node_in_state


def gather_inputs(
    node_info: NodeInfo,
    structure: GraphStructure,
    state: GraphState,
) -> Dict[str, jax.Array]:
    """
    Gather inputs for a node from the graph structure.
    """
    in_edges_data = {}
    for edge_key in node_info.in_edges:
        edge_info = structure.edges[edge_key]  # get the edge object
        node = edge_info.source
        in_edges_data[edge_key] = state.nodes[
            node
        ].z_latent  # get the data sent along this edge

    return in_edges_data


def inference_step(
    params: GraphParams,
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    eta_infer: float,
) -> GraphState:
    """
    Single inference step with local gradient computation.

    Args:
        params: Model parameters
        state: Current graph state
        clamps: Dictionary of clamped values
        structure: Graph structure
        eta_infer: Inference learning rate

    Returns:
        Updated graph state
    """

    # TODO parallelize over nodes for efficiency

    # 1. Zero the latent gradients
    for node_name in structure.nodes:
        # Reset gradients
        node_state = state.nodes[node_name]
        zero_grad = jnp.zeros_like(node_state.z_latent)
        state = update_node_in_state(state, node_name, latent_grad=zero_grad)

    # 2. Forward inference pass
    for node_name in structure.nodes:
        # Get node and its info
        node = structure.nodes[node_name]
        node_info = node.node_info
        node_class = _get_node_class_from_info(node_info)
        node_state = state.nodes[node_name]
        node_params = params.nodes[node_name]

        # Gather inputs for each slot
        in_edges_data = gather_inputs(node_info, structure, state)

        # Compute predictions, error, and latent gradient contributions
        node_state, inedge_grads = node_class.forward_inference(
            node_params,
            in_edges_data,
            node_state,
            node_info,
            is_clamped=(node_name in clamps),
        )

        # Update the graph state with node state containing errors and energy
        state = state._replace(nodes={**state.nodes, node_name: node_state})

        # Accumulate gradient contributions to this node's sources (local backward pass to in-neighbors)
        for edge_key, grad in inedge_grads.items():
            source_name = structure.edges[
                edge_key
            ].source  # Look up node name from edge key
            latent_grad = state.nodes[source_name].latent_grad
            latent_grad = (
                latent_grad + grad
            )  # Send gradient contribution to source node
            state = update_node_in_state(state, source_name, latent_grad=latent_grad)

    # 3. Update latent states by gradient descent
    for node_name in structure.nodes:
        node_state = state.nodes[node_name]
        if node_name in clamps:
            # Keep clamped nodes fixed
            new_z_latent = clamps[node_name]
        else:
            # Gradient descent
            new_z_latent = node_state.z_latent - eta_infer * node_state.latent_grad
        # Update state
        state = update_node_in_state(state, node_name, z_latent=new_z_latent)

    for node_name in structure.nodes:
        # Reset substructure
        state = update_node_in_state(state, node_name, substructure={})

    return state


def run_inference(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    infer_steps: int,
    eta_infer: float = 0.1,
) -> GraphState:
    """
    Run inference for infer_steps steps to converge latent states.

    Args:
        params: Model parameters
        initial_state: Initial graph state
        clamps: Dictionary of clamped values
        structure: Graph structure
        infer_steps: Number of inference steps
        eta_infer: Inference learning rate

    Returns:
        Final converged graph state
    """

    def body_fn(t, state):
        return inference_step(params, state, clamps, structure, eta_infer)

    # Use lax.fori_loop for efficiency
    final_state = jax.lax.fori_loop(0, infer_steps, body_fn, initial_state)

    return final_state
