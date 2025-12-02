"""
Linear node implementation for JAX predictive coding networks.

This implements a linear transformation node with configurable activation functions.
The node has a single multi-input slot that accepts multiple incoming connections.
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import get_activation
from fabricpc.core.initialization import initialize_weights


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
            inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
            state: NodeState,  # state object for the present node
            node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        """
        Linear transformation with activation.

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
        from fabricpc.nodes import get_node_class_from_type
        node_class = get_node_class_from_type(node_info.node_type)

        # Initialize pre-activation
        node_out_shape = state.z_latent.shape
        pre_activation = jnp.zeros(node_out_shape)

        if node_info.in_degree == 0:
            # Source nodes: no inputs
            z_mu = state.z_latent  # prediction is the latent state itself
            error = jnp.zeros_like(state.z_latent)
            gain_mod_error = jnp.zeros_like(state.z_latent)
        else:
            # Linear transformation
            for edge_key, x in inputs.items():
                pre_activation = pre_activation + jnp.matmul(x, params.weights[edge_key])

            # Add bias if present
            if "b" in params.biases and params.biases["b"].size > 0:
                pre_activation = pre_activation + params.biases["b"]

            # Apply activation function
            activation_fn, activation_deriv = get_activation(node_info.activation_config)
            z_mu = activation_fn(pre_activation)  # TODO turn off activation if latents represent preactivations

            # Error
            error = state.z_latent - z_mu

            # Gain-modulated error (use newly computed pre_activation, not state.pre_activation)
            f_prime = activation_deriv(pre_activation)  # shape (batch_size, dim_node_latent)
            gain_mod_error = error * f_prime  # element-wise multiplication

        # Update node state
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error,
            gain_mod_error=gain_mod_error)

        # Compute energy, accumulate the self-latent gradient
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        state: NodeState,  # state object for the present node
        node_info: NodeInfo,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """
        Forward pass: updates node state and computes gradients w.r.t. inputs.
        Explicitly compute gradients

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

        # Forward pass to get new state
        _, state = node_class.forward(params, inputs, state, node_info)
        # Note: the self-latent gradient is accumulated in state.latent_grad by the forward method

        # Determine the energy functional to use for the node from its config
        energy_functional = "gaussian"  # TODO make configurable per node, node_info.config.get("energy_functional", "gaussian")
        latent_is_preactivation = node_info.node_config.get("latent_type") == "preactivation"
        input_grads = {}

        # Back-synapse gradients for each edge, and accumulate to source nodes
        # ∂E/∂z_source = -Wcompute_params_gradient^T @ gain_mod_error_target
        for edge_key, z in inputs.items():

            if energy_functional == "gaussian":
                if latent_is_preactivation:
                    raise NotImplementedError("pre-activation latent type not implemented for LinearNode with Gaussian energy.")
                    grad_contribution = -jnp.matmul(state.error, params.weights[edge_key].T)
                    # error (batch, dim_t)
                    # weights{s->t} (dim_s, dim_t)
                else:
                    # For Gaussian energy, the gradient contribution is W @ gain_mod_error
                    grad_contribution = -jnp.matmul(state.gain_mod_error, params.weights[edge_key].T)
                    # gain_mod_error (batch, dim_t) = error * f'(a)
                    # weights{s->t} (dim_s, dim_t)
            else:
                raise NotImplementedError(f"energy functional '{energy_functional}' not implemented in LinearNode.")
                # _, activation_deriv = get_activation(node_info.node_config.get("activation_config"))
                # f_prime = activation_deriv(node_state.pre_activation)  # shape (batch_size, dim_node_latent)

            input_grads[edge_key] = grad_contribution

        return state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,  # state object for the present node
        node_info: NodeInfo
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass: update state and compute gradients of weights for local learning.
        Explicitly compute gradients

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

        # Forward pass to get new state
        _, state = node_class.forward(params, inputs, state, node_info)

        weight_grads = {}
        bias_grads = {}

        # Weight gradient
        for edge_key, in_tensor in inputs.items():
            grad_w = -jnp.matmul(in_tensor.T, state.gain_mod_error)
            weight_grads[edge_key] = grad_w

        # Bias gradient
        if "b" in params.biases:
            grad_b = -jnp.sum(state.gain_mod_error, axis=0, keepdims=True)
            bias_grads["b"] = grad_b

        return state, NodeParams(weights=weight_grads, biases=bias_grads)


class LinearAutoGradNode(LinearNode):
    """
    Linear node that uses NodeBase's autodiff-based gradient computation.

    This class extends LinearNode but delegates compute_gradient to the
    base class implementation, which uses JAX automatic differentiation
    via compute_jacobian_for_edge instead of the manual formula.

    Useful for:
    - Verifying correctness of manual gradient implementations
    - Prototyping new node types before optimizing gradients
    - Debugging gradient computation issues
    """

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        state: NodeState,  # state object for the present node
        node_info: NodeInfo,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """
        Forward pass: updates node state and computes gradients w.r.t. inputs.

        Delegate to NodeBase's implementation which uses JAX autodiff.
        """
        return NodeBase.forward_inference(params, inputs, state, node_info)

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,  # state object for the present node
        node_info: NodeInfo
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass: update state and compute gradients of weights for local learning.

        Delegate to NodeBase's implementation which uses JAX autodiff.
        """
        return NodeBase.forward_learning(params, inputs, state, node_info)