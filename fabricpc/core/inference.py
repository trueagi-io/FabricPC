"""Core inference dynamics for JAX predictive coding networks."""

from typing import Dict
import jax
import jax.numpy as jnp

from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeInfo
from fabricpc.utils.helpers import update_node_in_state


def gather_inputs(
    node_info: NodeInfo,
    structure: GraphStructure,
    state: GraphState,
) -> Dict[str, jax.Array]:
    """Gather inputs for a node from the graph structure."""
    in_edges_data = {}
    for edge_key in node_info.in_edges:
        edge_info = structure.edges[edge_key]
        source_node_name = edge_info.source
        in_edges_data[edge_key] = state.nodes[source_node_name].z_latent
    return in_edges_data


def inference_step(
    params: GraphParams,
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    eta_infer: float,
) -> GraphState:
    """Single inference step with local gradient computation."""

    for node_name in structure.nodes:
        node_state = state.nodes[node_name]
        zero_grad = jnp.zeros_like(node_state.z_latent)
        state = update_node_in_state(state, node_name, latent_grad=zero_grad)

    for node_name in structure.nodes:
        node_info = structure.nodes[node_name]
        node_state = state.nodes[node_name]
        node_params = params.nodes[node_name]

        in_edges_data = gather_inputs(node_info, structure, state)

        node_state, inedge_grads = node_info.node.__class__.forward_inference(
            node_params,
            in_edges_data,
            node_state,
            node_info,
            is_clamped=(node_name in clamps),
        )

        state = state._replace(nodes={**state.nodes, node_name: node_state})

        for edge_key, grad in inedge_grads.items():
            source_name = structure.edges[edge_key].source
            latent_grad = state.nodes[source_name].latent_grad
            latent_grad = latent_grad + grad
            state = update_node_in_state(state, source_name, latent_grad=latent_grad)

    for node_name in structure.nodes:
        node_state = state.nodes[node_name]
        if node_name in clamps:
            new_z_latent = clamps[node_name]
        else:
            new_z_latent = node_state.z_latent - eta_infer * node_state.latent_grad
        state = update_node_in_state(state, node_name, z_latent=new_z_latent)

    for node_name in structure.nodes:
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
    """Run inference for infer_steps steps to converge latent states."""

    def body_fn(t, state):
        return inference_step(params, state, clamps, structure, eta_infer)

    final_state = jax.lax.fori_loop(0, infer_steps, body_fn, initial_state)
    return final_state
