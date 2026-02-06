"""
Linear node implementation for JAX predictive coding networks.

This implements a linear transformation node with configurable activation functions.
The node has a single multi-input slot that accepts multiple incoming connections.

By default, linear nodes apply matrix multiplication on the **last axis only**, which is
standard for embeddings, projections, and transformer layers. This means:
- Input shape: (batch, ..., in_features)
- Weight shape: (in_features, out_features)
- Output shape: (batch, ..., out_features)

For fully-connected (dense) behavior that flattens all dimensions, set `flatten_input: true`
in the node config. This creates weights of shape (total_in_numel, total_out_numel) and reshapes output to the node's shape tuple.
"""

from typing import Dict, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.nodes.registry import register_node
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import get_activation
from fabricpc.core.initializers import initialize


@register_node("linear")
class LinearNode(FlattenInputMixin, NodeBase):
    """
    Linear transformation node: y = activation(W @ x + b)

    This node type:
    - Has a single multi-input slot named "in"
    - Concatenates all inputs and applies a linear transformation
    - Supports various activation functions
    - Implements local Hebbian learning

    Uses FlattenInputMixin for flatten/reshape operations.
    """

    CONFIG_SCHEMA = {
        "weight_init": {
            "type": dict,
            "default": {"type": "normal", "mean": 0.0, "std": 0.05},
            "description": "Weight initialization config",
        },
        "use_bias": {
            "type": bool,
            "default": True,
            "description": "Whether to use bias",
        },
        "flatten_input": {
            "type": bool,
            "default": False,
            "description": "If True, flatten all input dimensions for dense/fully-connected behavior and reshape the output to the node's shape tuple. "
            "If False (default), apply matmul on last axis only (standard for embeddings).",
        },
    }

    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Linear nodes have a single multi-input slot.

        Returns:
            Dictionary with one slot "in" that accepts multiple inputs
        """
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,  # from jax.random.PRNGKey
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Initialize weight matrix and bias vector.

        Weight shape depends on `flatten_input` config:
        - flatten_input=False (default): weights have shape (in_features, out_features)
          where in_features and out_features are the last dimensions of input/output.
          This applies the same weights at each position (standard for embeddings).
        - flatten_input=True: weights have shape (in_numel, out_numel) for dense layers.

        Args:
            key: JAX random key
            node_shape: Output shape of this node (excluding batch dimension)
            input_shapes: Dictionary with EdgeInfo.key -> source shape for that edge
            config: Node configuration with weight_init settings

        Returns:
            NodeParams with initialized W and b
        """
        flatten_input = config["flatten_input"]

        # Get weight initialization config
        default_cfg = {"type": "normal", "mean": 0.0, "std": 0.05}
        weight_init_config = config.get("weight_init", default_cfg)

        # Split key for weights and biases
        key_w, key_b = jax.random.split(key)

        # Initialize weight matrix for each incoming edge
        # this node class uses multi-input "in" slot
        weights_dict = {}
        rand_key_w = dict(
            zip(input_shapes.keys(), jax.random.split(key_w, len(input_shapes)))
        )

        for edge_key, in_shape in input_shapes.items():
            if ":in" not in edge_key:
                raise ValueError(
                    f"linear node requires 'in' slot dimension. got edge key {edge_key}"
                )

            if flatten_input:
                # Dense/fully-connected: flatten all dimensions
                in_numel = int(np.prod(in_shape))
                out_numel = int(np.prod(node_shape))
                weight_shape = (in_numel, out_numel)
            else:
                # Per-position: only last axis (standard for embeddings/projections)
                in_features = in_shape[-1]
                out_features = node_shape[-1]
                weight_shape = (in_features, out_features)

            weights_dict[edge_key] = initialize(
                rand_key_w[edge_key], weight_shape, weight_init_config
            )

        # Initialize bias (usually zeros)
        # Bias shape is (1,) + node_shape for proper broadcasting
        use_bias = config.get("use_bias", True)
        if use_bias:
            bias_shape = (1,) + node_shape
            b = jnp.zeros(bias_shape)

        return NodeParams(weights=weights_dict, biases={"b": b} if use_bias else {})

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

        When flatten_input=False (default): applies matmul on last axis only.
        When flatten_input=True: flattens all dimensions for dense behavior.

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
        from fabricpc.nodes import get_node_class

        node_class = get_node_class(node_info.node_type)

        # Get batch size and output shape
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape  # e.g., (128,) or (28, 28, 1)
        flatten_input = node_info.node_config["flatten_input"]

        if node_info.in_degree == 0:
            # Source nodes: no inputs
            z_mu = state.z_latent  # prediction is the latent state itself
            pre_activation = jnp.zeros_like(state.z_latent)
            error = jnp.zeros_like(state.z_latent)
        else:
            if flatten_input:
                # Dense/fully-connected: flatten all dimensions
                pre_activation = FlattenInputMixin.compute_linear(
                    inputs, params.weights, batch_size, out_shape
                )
            else:
                # Per-position: matmul on last axis only (standard for embeddings)
                # Input shape: (batch, ..., in_features)
                # Weight shape: (in_features, out_features)
                # Output shape: (batch, ..., out_features)
                pre_activation = jnp.zeros((batch_size,) + out_shape)
                for edge_key, x in inputs.items():
                    # jnp.matmul broadcasts over leading dimensions
                    pre_activation = pre_activation + jnp.matmul(
                        x, params.weights[edge_key]
                    )

            # Add bias if present (bias already has shape (1, *out_shape))
            if "b" in params.biases and params.biases["b"].size > 0:
                pre_activation = pre_activation + params.biases["b"]

            # Apply activation function
            activation_fn, _ = get_activation(node_info.node_config["activation"])
            z_mu = activation_fn(pre_activation)

            # Error
            error = state.z_latent - z_mu

        # Update node state
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Compute energy, accumulate the self-latent gradient
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state


@register_node("linear_explicit_grad")
class LinearExplicitGrad(LinearNode):
    """
    Linear node that demonstrates overriding NodeBase's autodiff-based gradient computation.
    Don't write explicit gradients for your node classes unless you have a specific reason to do so (e.g., symbolic functions or deliberately non-differentiable nodes).

    This class extends LinearNode to define explicit gradient computations
    for both inference and learning phases. Use as an example of manual gradient.

    Useful for:
    - Verifying correctness of manual gradient implementations
    - Prototyping optimized gradients
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
        Explicitly compute gradients

        When flatten_input=False: gradients computed via matmul on last axis.
        When flatten_input=True: gradients computed in flat space and reshaped.

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
        from fabricpc.nodes import get_node_class

        node_class = get_node_class(node_info.node_type)

        # Forward pass to get new state
        _, state = node_class.forward(params, inputs, state, node_info)
        # Note: the self-latent gradient is accumulated in state.latent_grad by the forward method

        # Gain-modulated error computation
        state = node_class.compute_gain_mod_error(state, node_info)

        # Determine the energy functional to use for the node from its config
        energy_functional = node_info.node_config.get("energy", {}).get("type", None)
        latent_is_preactivation = (
            node_info.node_config.get("latent_type") == "preactivation"
        )
        flatten_input = node_info.node_config["flatten_input"]
        input_grads = {}

        # Back-synapse gradients for each edge, and accumulate to source nodes
        # ∂E/∂z_source = -W^T @ gain_mod_error_target
        for edge_key, z in inputs.items():
            # Get source shape from input tensor
            source_shape = z.shape[1:]  # exclude batch dimension

            if energy_functional == "gaussian":
                if latent_is_preactivation:
                    raise NotImplementedError(
                        "pre-activation latent type not implemented for LinearExplicitGrad with Gaussian energy."
                    )
                else:
                    if flatten_input:
                        # Flatten gain_mod_error and compute gradient in flat space
                        gain_mod_error_flat = FlattenInputMixin.flatten_input(
                            state.substructure["gain_mod_error"]
                        )
                        grad_flat = -jnp.matmul(
                            gain_mod_error_flat, params.weights[edge_key].T
                        )
                        grad_contribution = FlattenInputMixin.reshape_output(
                            grad_flat, source_shape
                        )
                    else:
                        # Per-position: matmul on last axis (broadcasts over leading dims)
                        # gain_mod_error: (batch, ..., out_features)
                        # weights: (in_features, out_features)
                        # grad: (batch, ..., in_features)
                        grad_contribution = -jnp.matmul(
                            state.substructure["gain_mod_error"],
                            params.weights[edge_key].T,
                        )
            else:
                raise NotImplementedError(
                    f"energy functional '{energy_functional}' not implemented in LinearExplicitGrad."
                )

            input_grads[edge_key] = grad_contribution

        return state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,  # state object for the present node
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass: update state and compute gradients of weights for local learning.
        Explicitly compute gradients

        When flatten_input=False: gradient is -(input.T @ gain_mod_error) on last axes.
        When flatten_input=True: inputs/errors flattened before computing gradients.

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
        from fabricpc.nodes import get_node_class

        node_class = get_node_class(node_info.node_type)

        # Forward pass to get new state
        _, state = node_class.forward(params, inputs, state, node_info)

        # Gain-modulated error computation
        state = node_class.compute_gain_mod_error(state, node_info)

        flatten_input = node_info.node_config["flatten_input"]
        weight_grads = {}
        bias_grads = {}

        # Weight gradient
        for edge_key, in_tensor in inputs.items():
            if flatten_input:
                # Flatten inputs and errors for dense gradient computation
                in_flat = FlattenInputMixin.flatten_input(in_tensor)
                gain_mod_error_flat = FlattenInputMixin.flatten_input(
                    state.substructure["gain_mod_error"]
                )
                # Compute gradient: (in_numel, batch) @ (batch, out_numel) -> (in_numel, out_numel)
                grad_w = -jnp.matmul(in_flat.T, gain_mod_error_flat)
            else:
                # Per-position gradient: contract over batch and position dims
                # in_tensor: (batch, ..., in_features)
                # gain_mod_error: (batch, ..., out_features)
                # grad_w: (in_features, out_features)
                # Reshape to (batch*positions, features) for efficient matmul
                in_shape = in_tensor.shape
                err_shape = state.substructure["gain_mod_error"].shape
                in_flat = in_tensor.reshape(
                    -1, in_shape[-1]
                )  # (batch*pos, in_features)
                err_flat = state.substructure["gain_mod_error"].reshape(
                    -1, err_shape[-1]
                )  # (batch*pos, out_features)
                grad_w = -jnp.matmul(in_flat.T, err_flat)  # (in_features, out_features)
            weight_grads[edge_key] = grad_w

        # Bias gradient - sum over batch, keep shape (1, *out_shape)
        if "b" in params.biases:
            # Sum over batch (axis=0), keepdims to get (1, *out_shape)
            grad_b = -jnp.sum(
                state.substructure["gain_mod_error"], axis=0, keepdims=True
            )
            bias_grads["b"] = grad_b

        return state, NodeParams(weights=weight_grads, biases=bias_grads)

    @staticmethod
    def compute_gain_mod_error(state: NodeState, node_info: NodeInfo) -> NodeState:
        """
        Compute gain-modulated error for this node.

        Args:
            state: NodeState containing current error and pre_activation
            node_info: NodeInfo object
        """
        _, activation_deriv = get_activation(node_info.node_config["activation"])

        # Gain-modulated error computation
        f_prime = activation_deriv(
            state.pre_activation
        )  # shape (batch_size, *out_shape)
        gain_mod_error = state.error * f_prime  # element-wise multiplication

        state = state._replace(substructure={"gain_mod_error": gain_mod_error})
        return state
