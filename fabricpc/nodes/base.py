"""
Base node classes for JAX predictive coding networks.

This module provides the abstract base class for all node types, defining the
interface for custom transfer functions, multiple input slots, and local gradient computation.
All node methods are pure functions (no side effects) for JAX compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from fabricpc.core.types import NodeParams, NodeState, NodeInfo


@dataclass(frozen=True)
class SlotSpec:
    """Specification for an input slot to a node."""
    name: str
    is_multi_input: bool  # True = multiple inputs allowed, False = single input only


@dataclass(frozen=True)
class Slot:
    """Runtime slot information with connected edges."""
    spec: SlotSpec
    in_neighbors: Dict[str, str]  # edge_key -> source_node_name mapping

class NodeBase(ABC):
    """
    Abstract base class for all predictive coding nodes.

    All methods are pure functions (no side effects) for JAX compatibility.
    Nodes can have multiple input slots and custom transfer functions.
    """

    @staticmethod
    @abstractmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Define the input slots for this node type.

        Returns:
            Dictionary mapping slot names to SlotSpec objects

        Example:
            return {
                "in": SlotSpec(name="in", is_multi_input=True),
                "gate": SlotSpec(name="gate", is_multi_input=False)
            }
        """
        pass

    @staticmethod
    @abstractmethod
    def initialize_params(
        key: jax.Array,  # from jax.random.PRNGKey
        node_dim: int,
        input_dims: Dict[str, int],  # slot_name -> total input dimension
        config: Dict[str, Any]
    ) -> NodeParams:
        """
        Initialize parameters for this node.
        Describe the weights and biases structure in the docstring.

        Args:
            key: JAX random key
            node_dim: Dimension of this node's output
            input_dims: Dictionary mapping slot names to total input dimensions
            config: Node configuration (may contain initialization settings)

        Returns:
            NodeParams with initialized weights and biases

        Example:
            For a linear node with one multi-input slot "in":
            weights = {"W": initialize_weights(key, (input_dims["in"], node_dim))}
            biases = {"b": jnp.zeros((1, node_dim))}
            return NodeParams(weights=weights, biases=biases)
        """
        pass

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        state: NodeState,  # state object for the present node
        node_info: NodeInfo,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """
        Forward pass: updates node state and computes gradients w.r.t. inputs.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: state object for the present node
            node_info: NodeInfo object (contains activation function, etc.)

        Returns:
            Tuple of (NodeState, gradient_wrt_inputs):
                - NodeState: updated node state (z_mu, pre_activation, etc.)
                - gradient_wrt_inputs: dictionary of gradients w.r.t. each input edge
        """
        from fabricpc.nodes import get_node_class_from_type
        node_class = get_node_class_from_type(node_info.node_type)

        # Use JAX's value_and_grad to compute gradients w.r.t. inputs
        (total_energy, new_state), input_grads = jax.value_and_grad(
            node_class.forward,
            argnums=1,  # inputs
            has_aux=True
        )(params, inputs, state, node_info)

        return new_state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,  # state object for the present node
        node_info: NodeInfo
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass: update state and compute gradients of weights for local learning.

        The local gradient for weights is: -(input.T @ gain_mod_error)

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
            state: state object for the present node
            node_info: NodeInfo object

        Returns:
            Tuple of (NodeState, params_grad):
                - NodeState: updated node state (z_mu, pre_activation, etc.)
                - params_grad: NodeParams containing weight and bias gradients
        """
        from fabricpc.nodes import get_node_class_from_type
        node_class = get_node_class_from_type(node_info.node_type)

        # Use JAX's value_and_grad to compute gradients w.r.t. params
        (total_energy, new_state), params_grad = jax.value_and_grad(
            node_class.forward,
            argnums=0,  # params
            has_aux=True
        )(params, inputs, state, node_info)

        return new_state, params_grad

    @staticmethod
    @abstractmethod
    def forward(
            params: NodeParams,
            inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
            state: NodeState,  # state object for the present node
            node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        """
        Forward pass through the node, returning energy scalar and updated state.
        Computes:
            forward pass -> compute error -> compute energy -> total energy

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: state object for the present node
            node_info: NodeInfo object (contains activation function, etc.)

        Returns:
            Tuple of (total_energy, NodeState):
                - total_energy: scalar energy value for this node
                - NodeState: updated node state (z_mu, pre_activation, etc.)
        """
        pass

    @staticmethod
    def energy_functional(
        state: NodeState,
        node_info: NodeInfo
    ) -> NodeState:
        """
        Compute the energy and the derivative with respect to the node's latent state.

        Args:
            state: NodeState object (contains z_latent, z_mu, etc.)
            node_info: NodeInfo object (may contain energy functional info)

        Returns:
        Updated NodeState with
            energy: energy value per batch element
            latent_grad: derivative of energy w.r.t. z_latent
        """
        # 1. Compute energy: E = 0.5 * ||error||^2
        energy = 0.5 * jnp.sum(state.error ** 2, axis=1)  # sum over latent dimensions, don't sum over batch

        # 2. Compute latent gradient: dE/dz_latent
        grad = state.error  # derivative of energy w.r.t. z_latent
        latent_grad = state.latent_grad + grad  # accumulate with existing latent_grad

        # Update node state
        state = state._replace(energy=energy, latent_grad=latent_grad)

        return state

    @staticmethod
    def get_energy_functional(energy_name: str) -> Tuple[Any, Any, Any]:
        """
        Retrieve the energy functional by name.
        Args:
            energy_name: Name of the energy functional (e.g., "gaussian", "bernoulli")
        Returns:
            Energy functional function, gradient function, and jacobian function
        """
        pass