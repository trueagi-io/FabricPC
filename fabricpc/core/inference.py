"""
Core inference dynamics for JAX predictive coding networks.

This module provides:
- InferenceBase: Abstract base class for inference algorithms
- InferenceSGD: Default SGD-based inference (z -= eta * grad)
- Backward-compatible module-level functions (run_inference, inference_step)

Inference algorithms control how latent states are updated during the
inference loop. The primary extension point is `latent_update()`.

Usage:
    from fabricpc.core.inference import InferenceSGD

    structure = graph(
        nodes=[...], edges=[...], task_map=...,
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import types

import jax
import jax.numpy as jnp

from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeInfo,
    NodeState,
)
from fabricpc.utils.helpers import update_node_in_state

# =============================================================================
# Utility Functions
# =============================================================================


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


# =============================================================================
# Inference Base Class
# =============================================================================


class InferenceBase(ABC):
    """
    Abstract base class for inference algorithms.

    Inference algorithms control how latent states are updated during the
    inference loop. The primary extension point is `latent_update()`.

    Custom inference algorithms extend this class:

    All computation methods are static for JAX compatibility (pure functions, no state).
    """

    def __init__(self, **config):
        self.config = types.MappingProxyType(config)  # Immutable dictionary

    @staticmethod
    def inference_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """
        Single inference step: forward phase -> latent update.

        Override for algorithms that need a different phase structure
        (e.g., momentum that accumulates across steps).
        """
        inference_obj = structure.config["inference"]
        cls = type(inference_obj)

        # Phase 1: Zero the latent gradients
        state = cls.zero_grads(params, state, clamps, structure)

        # Phase 2: Forward pass to predict expected state (z_mu) & accumulate latent grads
        state = cls.forward_value_and_grad(params, state, clamps, structure)

        # Phase 3: Update the latent (z_state) with gradient
        state = cls.update_latents(params, state, clamps, structure, config)

        return state

    @staticmethod
    def zero_grads(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Phase 1: Zero the latent gradients
        """
        # Phase 1: Zero latent gradients
        for node_name in structure.nodes:
            node_state = state.nodes[node_name]
            zero_grad = jnp.zeros_like(node_state.z_latent)
            state = update_node_in_state(state, node_name, latent_grad=zero_grad)

        return state

    @staticmethod
    def forward_value_and_grad(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Phase 2: Forward pass value & grad; accumulate latent grads.

        This is the universal predictive coding mechanics shared by all inference algorithms.
        Phase 3 (latent update) is handled by the inference algorithm's latent_update() method.
        """
        for node_name in structure.nodes:
            # Get node and its info
            node = structure.nodes[node_name]
            node_info = node.node_info
            node_class = node_info.node_class
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
                source_name = structure.edges[edge_key].source
                latent_grad = state.nodes[source_name].latent_grad + grad
                state = update_node_in_state(
                    state, source_name, latent_grad=latent_grad
                )

        return state

    @staticmethod
    def update_latents(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """
        Update latent states for each node based on the accumulated latent gradients.
        """
        inference_obj = structure.config["inference"]
        cls = type(inference_obj)

        for node_name in structure.nodes:
            node_state = state.nodes[node_name]

            if node_name not in clamps:
                new_z_latent = cls.compute_new_latent(
                    node_name,
                    node_state,
                    config,
                )
                state = update_node_in_state(state, node_name, z_latent=new_z_latent)

        return state

    @staticmethod
    @abstractmethod
    def compute_new_latent(
        node_name: str,
        node_state: NodeState,
        config: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Compute the updated z_latent for a single node.
        This is the primary extension point for custom inference algorithms.

        Returns:
            Updated z_latent array
        """
        pass

    @staticmethod
    def run_inference(
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Outer inference loop using lax.fori_loop.

        Override for scan-based tracking, adaptive stopping, etc.
        infer_steps is read from self.config['infer_steps'].
        """
        inference_obj = structure.config["inference"]
        inference_cls = type(inference_obj)
        config = inference_obj.config
        infer_steps = config["infer_steps"]

        def body_fn(t, state):
            return inference_cls.inference_step(
                params, state, clamps, structure, config
            )

        # Use lax.fori_loop for efficiency
        final_state = jax.lax.fori_loop(0, infer_steps, body_fn, initial_state)
        return final_state


# =============================================================================
# Built-in Inference Algorithms
# =============================================================================


class InferenceSGD(InferenceBase):
    """
    Standard SGD inference: z -= eta * grad.

    This is the default inference algorithm for predictive coding networks.

    Args:
        eta_infer: Inference learning rate (default: 0.1)
        infer_steps: Number of inference iterations (default: 20)
    """

    def __init__(self, eta_infer=0.1, infer_steps=20):
        super().__init__(eta_infer=eta_infer, infer_steps=infer_steps)

    @staticmethod
    def compute_new_latent(node_name, node_state, config):

        eta_infer = config["eta_infer"]

        new_latent = node_state.z_latent - eta_infer * node_state.latent_grad
        return new_latent


class InferenceSGDNormClip(InferenceBase):
    """
    SGD inference with per-node gradient norm clipping: z -= eta * clip(grad).

    Clips the L2 norm of each node's latent gradient independently per sample.
    If ||grad|| > max_norm, scales grad down to max_norm.
    Uses safe division with epsilon to handle zero gradients.

    Args:
        eta_infer: Inference learning rate (default: 0.1)
        infer_steps: Number of inference iterations (default: 20)
        max_norm: Maximum gradient norm per node (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(self, eta_infer=0.1, infer_steps=20, max_norm=1.0, eps=1e-8):
        super().__init__(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            max_norm=max_norm,
            eps=eps,
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        eta_infer = config["eta_infer"]
        max_norm = config["max_norm"]
        eps = config["eps"]

        grad = node_state.latent_grad
        # Per-sample L2 norm (sum over all non-batch dims)
        grad_norm = jnp.sqrt(
            jnp.sum(grad.conj() * grad, axis=tuple(range(1, grad.ndim)), keepdims=True)
        )
        clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + eps))
        clipped_grad = grad * clip_factor

        return node_state.z_latent - eta_infer * clipped_grad
