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
from fabricpc_jax.core.types import NodeParams, NodeState, NodeInfo, GraphStructure

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
    @abstractmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        node_info: NodeInfo,
        node_out_shape: Tuple[int, ...], # shape of the node output
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Forward pass through the node.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            node_info: NodeInfo object (contains activation function, etc.)
            node_out_shape: shape of the node output (same as latent state)

        Returns:
            Tuple of (z_mu, pre_activation, substructure_state):
                - z_mu: Output after activation function
                - pre_activation: Output before activation function
                - substructure_state: dictionary of internal states for complex nodes

        Example:
            pre_act = jnp.matmul(inputs["in"], params.weights["W"]) + params.biases["b"]
            z_mu = activation_fn(pre_act)
            return z_mu, pre_act
        """
        pass

    @staticmethod
    def compute_jacobian_for_edge(
        edge_key: str,
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        node_state: NodeState,  # state object for the present node
        node_info: NodeInfo
    ) -> jnp.ndarray:  # EdgeInfo.key -> Jacobian matrix
        """
        Compute Jacobian of output w.r.t. a specific input source.
        jacobian_{j->i} (dim_j, dim_i), del mu_j / del z_i

        The energy for batch element b is:
        E_b = Σ_nodes ½||error_node,b||²

        The gradient at batch b:
        ∂E_b/∂z_source,b = error_source,b - Σ_targets (error_target,b · J_target←source,b)

        Using Einstein summation:
        grad_contrib[b,s] = Σ_t error[b,t] · jacobian[b,t,s]

        Default implementation uses JAX automatic differentiation.
        Can be overridden for non-differentiable nodes with custom derivatives.

        Args:
            edge_key: Key of the edge for which to compute the Jacobian
            params: Node parameters
            inputs: Dictionary mapping edge keys input tensors
            node_state: state object for the present node
            node_info: NodeInfo object for the present node

        Returns:
            dictionary of Jacobian matrix of shape (input_dim, output_dim) for each edge key
        """
        # Get the concrete node class for this node type
        from fabricpc_jax.nodes import get_node_class_from_type
        node_class = get_node_class_from_type(node_info.node_type)

        # Default: use JAX autodiff
        node_out_shape = node_state.z_latent.shape
        def output_fn(source_input):
            # Replace the specific source input
            modified_inputs = inputs.copy()
            modified_inputs[edge_key] = source_input
            z_mu, _, _ = node_class.forward(params, modified_inputs, node_info, node_out_shape)
            return z_mu

        jacobian = jax.jacobian(output_fn)(inputs[edge_key])
        # jacobian has shape (output_dim, batch_size, input_dim, batch_size)
        return jacobian

    @staticmethod
    def compute_gradient(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        node_state: NodeState,
        node_info: NodeInfo,
        structure: GraphStructure,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute local gradients of latent states for inference updates.
        Compute the contributions of node itself and its source nodes to the energy of this node.
        Use the energy functional specific in NodeInfo to compute the gradients.

        Computes:
        ∂E/∂z_i, where i is in {this node} ∪ {source nodes connected to this node}

        Args:
            params: Current node parameters
            inputs: Dictionary mapping edge key to input tensors
            node_state: Current node state (contains errors, pre-activations, etc.)
            node_info: Node configuration
            structure: GraphStructure object (contains graph topology)

        Returns:
            Dictionary mapping node names to latent gradient contributions
        """

        # Determine the energy functional to use for the node from its config
        energy_functional = "gaussian"  # TODO make configurable per node, node_info.config.get("energy_functional", "gaussian")

        latent_grads = {}

        # Self energy gradient
        latent_grads[node_info.name] = node_state.error

        # Back-synapse gradients for each edge, and accumulate to source nodes
        for edge_key, z in inputs.items():
            source_name = structure.edges[edge_key].source
            if source_name not in latent_grads:
                latent_grads[source_name] = jnp.zeros_like(z)

            # Jacobian from source node
            jacobian = NodeBase.compute_jacobian_for_edge(edge_key, params, inputs, node_state, node_info)

            # Compute gradient contribution based on Jacobian dimensions
            if jacobian.ndim == 3:
                # jacobian: (batch, dim_target, dim_source)
                # error: (batch, dim_target)
                # Result: (batch, dim_source)
                grad_contrib = jnp.einsum('bt,bts->bs', node_state.error, jacobian)
                # Alternative: batched matrix multiply (can hide shape errors, may create intermediates in memory, no speedup versus einsum except initial jit compile)
                # grad_contrib = jnp.matmul(
                #     target_error[:, None, :],  # (batch, 1, dim_target)
                #     jacobian                     # (batch, dim_target, dim_source)
                # ).squeeze(1)                    # (batch, dim_source)

            elif jacobian.ndim == 4:
                # Case 2: Cross-batch dependencies
                # jacobian: (batch_target, dim_target, batch_source, dim_source)
                # error: (batch_target, dim_target)
                # Result: (batch_source, dim_source)
                grad_contrib = jnp.einsum('jt,jtis->is', node_state.error, jacobian)
                # Where indices mean:
                # j: target batch index
                # t: target dimension
                # i: source batch index
                # s: source dimension

            else:
                raise ValueError(f"invalid Jacobian shape {jacobian.shape} for edge {edge_key}")

            # Apply the contributions from projections of target nodes
            # ∂E_b /∂z_this = error_this - Σ_targets(error_target · J_target←this)

            latent_grads[source_name] = latent_grads[source_name] - grad_contrib

        return latent_grads

    @staticmethod
    @abstractmethod
    def compute_params_gradient(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        node_state: NodeState,  # state object for the present node
        node_info: NodeInfo
    ) -> NodeParams:
        """
        Compute gradients of weights for local learning.

        The local gradient for weights is: -(input.T @ gain_mod_error)

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
            node_state: state object for the present node
            node_info: NodeInfo object

        Returns:
            NodeParams containing weight and bias gradients
        """
        # TODO autograd as default, override in subclass for efficiency
        pass

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