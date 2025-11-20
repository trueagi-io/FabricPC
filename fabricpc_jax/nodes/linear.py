"""
Linear node implementation for JAX predictive coding networks.

This implements a linear transformation node with configurable activation functions.
The node has a single multi-input slot that accepts multiple incoming connections.
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp

from fabricpc_jax.nodes.base import NodeBase, SlotSpec
from fabricpc_jax.core.types import NodeParams, NodeState, NodeInfo, GraphStructure
from fabricpc_jax.core.activations import get_activation
from fabricpc_jax.core.initialization import initialize_weights


class LinearNode(NodeBase):
    """
    Linear transformation node: y = activation(W @ x + b)

    This node type:
    - Has a single multi-input slot named "in"
    - Concatenates all inputs and applies a linear transformation
    - Supports various activation functions
    - Implements local Hebbian learning
    """

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Linear nodes have a single multi-input slot.

        Returns:
            Dictionary with one slot "in" that accepts multiple inputs
        """
        return {
            "in": SlotSpec(name="in", is_multi_input=True)
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,  # from jax.random.PRNGKey
        node_dim: int,
        input_dims: Dict[str, int],
        config: Dict[str, Any]
    ) -> NodeParams:
        """
        Initialize weight matrix and bias vector.
        Linear node weights structure:
            NodeParams.weights: is a dict keyed by EdgeInfo.key for each incoming edge.
            NodeParams.biases: Dict['b': bias vector]

        Args:
            key: JAX random key
            node_dim: Output dimension of this node
            input_dims: Dictionary with EdgeInfo.key -> input dimension for that edge
            config: Node configuration with weight_init settings

        Returns:
            NodeParams with initialized W and b
        """
        # Counter for total input dimension from the "in" slot
        total_in_dim = 0

        # Get weight initialization config
        default_cfg = {"type": "normal", "mean": 0.0, "std": 0.05}
        weight_init_config = config.get("weight_init", default_cfg)

        # Split key for weights and biases
        key_w, key_b = jax.random.split(key)

        # Initialize weight matrix
        # this node class uses multi-input "in" slot; create weights for each incoming edge
        weights_dict = {}
        rand_key_w = dict(zip(input_dims.keys(), jax.random.split(key_w, len(input_dims))))
        for edge_key, in_dim in input_dims.items():
            if ":in" not in edge_key:
                raise ValueError(f"linear node requires 'in' slot dimension. got edge key {edge_key}")  # validate that edges correspond to "in" slot
            weights_dict[edge_key] = initialize_weights(weight_init_config, rand_key_w[edge_key], (in_dim, node_dim))
            total_in_dim += in_dim

        # Initialize bias (usually zeros)
        use_bias = config.get("use_bias", True)
        if use_bias:
            b = jnp.zeros((1, node_dim))

        return NodeParams(
            weights=weights_dict,
            biases={"b": b} if use_bias else {}
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray], # EdgeInfo.key -> input tensors
        node_info: NodeInfo,
        node_out_shape: Tuple[int, ...],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Forward pass: linear transformation with activation.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            node_info: NodeInfo object (contains activation function, etc.)
            node_out_shape: shape of the node output (same as latent state)

        Returns:
            Tuple of (z_mu, pre_activation, substructure_state):
        """

        pre_activation = jnp.zeros(node_out_shape)

        # Linear transformation
        for edge_key, x in inputs.items():
            pre_activation = pre_activation + jnp.matmul(x, params.weights[edge_key])

        # Add bias if present
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation function
        activation_fn, _ = get_activation(node_info.activation_config)
        z_mu = activation_fn(pre_activation)

        return z_mu, pre_activation, {}

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
            structure: GraphStructure object

        Returns:
            Dictionary mapping node names to latent gradient contributions
        """

        # Determine the energy functional to use for the node from its config
        energy_functional = "gaussian"  # TODO make configurable per node, node_info.config.get("energy_functional", "gaussian")
        latent_is_preactivation = node_info.node_config.get("latent_type") == "preactivation"
        latent_grads = {}

        # Self energy gradient
        latent_grads[node_info.name] = node_state.error

        # Back-synapse gradients for each edge, and accumulate to source nodes
        # ∂E/∂z_source = -W^T @ gain_mod_error_target
        for edge_key, z in inputs.items():
            source_name = structure.edges[edge_key].source
            if source_name not in latent_grads:
                latent_grads[source_name] = jnp.zeros_like(z)

            if energy_functional == "gaussian":
                if latent_is_preactivation:
                    raise NotImplementedError("pre-activation latent type not implemented for LinearNode with Gaussian energy.")
                    grad_contribution = -jnp.matmul(node_state.error, params.weights[edge_key].T)
                    # error (batch, dim_t)
                    # weights{s->t} (dim_s, dim_t)
                else:
                    # For Gaussian energy, the gradient contribution is W @ gain_mod_error
                    grad_contribution = -jnp.matmul(node_state.gain_mod_error, params.weights[edge_key].T)
                    # gain_mod_error (batch, dim_t) = error * f'(a)
                    # weights{s->t} (dim_s, dim_t)
            else:
                raise NotImplementedError(f"energy functional '{energy_functional}' not implemented in LinearNode.")
                # _, activation_deriv = get_activation(node_info.node_config.get("activation_config"))
                # f_prime = activation_deriv(node_state.pre_activation)  # shape (batch_size, dim_node_latent)

            latent_grads[source_name] = latent_grads[source_name] + grad_contribution

        return latent_grads

    @staticmethod
    def compute_params_gradient(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        node_state: NodeState,  # state object for the present node
        node_info: NodeInfo
    ) -> NodeParams:
        """
        Compute local gradients for weights and biases.

        For linear nodes:
        - Weight gradient: -(input.T @ gain_mod_error)
        - Bias gradient: -sum(gain_mod_error, axis=0)

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
            node_state: state object for the present node
            node_info: NodeInfo object

        Returns:
            NodeParams containing weight and bias gradients
        """

        # fix the test file line 203 to check for params keyed on edge strings

        weight_grads = {}
        bias_grads = {}

        # Weight gradient
        for edge_key, in_tensor in inputs.items():
            grad_w = -jnp.matmul(in_tensor.T, node_state.gain_mod_error)
            weight_grads[edge_key] = grad_w

        # Bias gradient
        if "b" in params.biases:
            grad_b = -jnp.sum(node_state.gain_mod_error, axis=0, keepdims=True)
            bias_grads["b"] = grad_b

        return NodeParams(weights=weight_grads, biases=bias_grads)
