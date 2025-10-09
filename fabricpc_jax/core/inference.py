"""
Core inference dynamics for JAX predictive coding networks.

This module implements the functional inference loop that updates latent states
to minimize prediction error.
"""

from typing import Dict, Tuple
from functools import partial
import jax
import jax.numpy as jnp

from fabricpc_jax.core.types import GraphParams, GraphState, GraphStructure
from fabricpc_jax.core.activations import get_activation


def compute_projection(
    params: GraphParams,
    z_latent: Dict[str, jnp.ndarray],
    node_name: str,
    structure: GraphStructure,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute prediction z_mu for a node from its incoming connections.

    Args:
        params: Model parameters (weights, biases)
        z_latent: Current latent states for all nodes
        node_name: Name of the node to compute prediction for
        structure: Static graph structure

    Returns:
        Tuple of (z_mu, pre_activation):
            - z_mu: Predicted state after activation
            - pre_activation: Pre-activation value (for computing derivatives)
    """
    node_info = structure.nodes[node_name]

    # Source nodes (no incoming edges) have zero prediction
    if node_info.in_degree == 0:
        batch_size = next(iter(z_latent.values())).shape[0]
        zero_pred = jnp.zeros((batch_size, node_info.dim))
        return zero_pred, zero_pred

    # Gather inputs from all incoming edges
    inputs = []
    for edge_key in node_info.in_edges:
        edge_info = structure.edges[edge_key]
        source_z = z_latent[edge_info.source]
        inputs.append(source_z)

    # Concatenate all inputs
    inputs_concat = jnp.concatenate(inputs, axis=-1) if len(inputs) > 1 else inputs[0]

    # Linear transformation: pre_activation = inputs @ W + b
    # Use first edge key to index weights (all incoming edges share same weight matrix)
    first_edge_key = node_info.in_edges[0]
    W = params.weights[first_edge_key]
    b = params.biases[node_name]

    pre_activation = jnp.matmul(inputs_concat, W) + b

    # Apply activation function
    activation_fn, _ = get_activation(node_info.activation_config)
    z_mu = activation_fn(pre_activation)

    return z_mu, pre_activation


def compute_all_projections(
    params: GraphParams,
    z_latent: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Compute predictions for all nodes in the graph.

    Args:
        params: Model parameters
        z_latent: Current latent states
        structure: Graph structure

    Returns:
        Tuple of (z_mu_dict, pre_activation_dict):
            - z_mu_dict: Predictions for all nodes
            - pre_activation_dict: Pre-activation values for all nodes
    """
    z_mu = {}
    pre_activation = {}

    # Use node_order for efficient traversal (topological order)
    for node_name in structure.node_order:
        z_mu[node_name], pre_activation[node_name] = compute_projection(
            params, z_latent, node_name, structure
        )

    return z_mu, pre_activation


def compute_errors(
    z_latent: Dict[str, jnp.ndarray],
    z_mu: Dict[str, jnp.ndarray],
    pre_activation: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Compute prediction errors and gain-modulated errors.

    Args:
        z_latent: Current latent states
        z_mu: Predicted states
        pre_activation: Pre-activation values
        structure: Graph structure

    Returns:
        Tuple of (error, gain_mod_error):
            - error: z_latent - z_mu for all nodes
            - gain_mod_error: error * f'(pre_activation) for all nodes
    """
    error = {}
    gain_mod_error = {}

    for node_name in structure.nodes:
        node_info = structure.nodes[node_name]

        # Compute basic error
        err = z_latent[node_name] - z_mu[node_name]
        error[node_name] = err

        # Compute gain-modulated error (error weighted by activation derivative)
        if node_info.in_degree == 0:
            # Source nodes have no prediction, so no gain modulation
            gain_mod_error[node_name] = jnp.zeros_like(err)
        else:
            _, deriv_fn = get_activation(node_info.activation_config)
            gain = deriv_fn(pre_activation[node_name])
            gain_mod_error[node_name] = err * gain

    return error, gain_mod_error


def compute_latent_gradients(
    error: Dict[str, jnp.ndarray],
    gain_mod_error: Dict[str, jnp.ndarray],
    params: GraphParams,
    structure: GraphStructure,
) -> Dict[str, jnp.ndarray]:
    """
    Compute gradients of energy w.r.t. latent states.

    For each node i:
        grad_i = error_i - sum_j (W_{i->j}^T @ gain_mod_error_j)

    where j ranges over all nodes that node i projects to.

    Args:
        error: Prediction errors for all nodes
        gain_mod_error: Gain-modulated errors for all nodes
        params: Model parameters
        structure: Graph structure

    Returns:
        Dictionary of gradients w.r.t. latent states
    """
    latent_grads = {}

    for node_name in structure.nodes:
        node_info = structure.nodes[node_name]

        # Start with local error contribution
        grad = error[node_name].copy()

        # Subtract backpropagated errors from downstream nodes
        for out_edge_key in node_info.out_edges:
            edge_info = structure.edges[out_edge_key]
            target_name = edge_info.target
            target_gain_mod_error = gain_mod_error[target_name]

            # Get weight matrix for this edge
            W = params.weights[out_edge_key]

            # Compute Jacobian contribution: W^T @ gain_mod_error
            # Need to extract the relevant slice of W for this source node
            target_node_info = structure.nodes[target_name]

            # Find which slice of the weight matrix corresponds to this source
            slice_start = 0
            for in_edge_key in target_node_info.in_edges:
                in_edge_info = structure.edges[in_edge_key]
                source_dim = structure.nodes[in_edge_info.source].dim

                if in_edge_info.source == node_name:
                    # This is our slice
                    W_slice = W[slice_start:slice_start + source_dim, :]
                    jacobian_contrib = jnp.matmul(target_gain_mod_error, W_slice.T)
                    grad -= jacobian_contrib
                    break

                slice_start += source_dim

        latent_grads[node_name] = grad

    return latent_grads


def inference_step(
    params: GraphParams,
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    eta_infer: float,
) -> GraphState:
    """
    Single inference step: update all latent states via gradient descent on energy.

    Args:
        params: Model parameters
        state: Current graph state
        clamps: Dictionary of clamped values {node_name: tensor}
        structure: Graph structure
        eta_infer: Inference learning rate

    Returns:
        Updated graph state
    """
    # 1. Compute predictions for all nodes
    z_mu, pre_activation = compute_all_projections(params, state.z_latent, structure)

    # 2. Compute errors
    error, gain_mod_error = compute_errors(
        state.z_latent, z_mu, pre_activation, structure
    )

    # 3. Compute gradients w.r.t. latent states
    latent_grads = compute_latent_gradients(error, gain_mod_error, params, structure)

    # 4. Update latent states (gradient descent on energy)
    new_z_latent = {}
    for node_name in structure.nodes:
        if node_name in clamps:
            # Keep clamped nodes fixed
            new_z_latent[node_name] = clamps[node_name]
        else:
            # Update via gradient descent
            new_z_latent[node_name] = (
                state.z_latent[node_name] - eta_infer * latent_grads[node_name]
            )

    return GraphState(
        z_latent=new_z_latent,
        z_mu=z_mu,
        error=error,
        pre_activation=pre_activation,
        gain_mod_error=gain_mod_error,
    )


def run_inference(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    T_infer: int,
    eta_infer: float = 0.1,
) -> GraphState:
    """
    Run inference for T_infer steps to converge latent states.

    This function uses lax.fori_loop for efficient iteration.
    Note: JIT compilation happens at the call site (e.g., in train_step).

    Args:
        params: Model parameters
        initial_state: Initial graph state
        clamps: Dictionary of clamped values
        structure: Graph structure
        T_infer: Number of inference steps
        eta_infer: Inference learning rate

    Returns:
        Final converged graph state
    """

    def body_fn(t, state):
        return inference_step(params, state, clamps, structure, eta_infer)

    # Use lax.fori_loop for efficiency (unrolled and optimized by XLA)
    final_state = jax.lax.fori_loop(0, T_infer, body_fn, initial_state)

    return final_state
