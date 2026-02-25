"""
Conv1D, Conv2D, and Conv3D nodes for JAX predictive coding networks.

These nodes implement convolutional operations using JAX's lax.conv_general_dilated.

Input formats:
  - Conv1D: NLC (batch, length, channels)
  - Conv2D: NHWC (batch, height, width, channels)
  - Conv3D: NDHWC (batch, depth, height, width, channels)

Output shapes exclude batch dimension.

Example usage:
    config = {
        "node_list": [
            {
                "name": "conv1",
                "shape": (28, 28, 32),
                "type": "conv2d",
                "activation": "relu",
                "conv2d": {
                    "kernel_size": (3, 3),
                    "stride": (1, 1),
                    "padding": "SAME",
                    "weight_init": {"type": "kaiming"}
                }
            }
        ]
    }
"""

from typing import Dict, Any, Tuple
import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.registry import register_node
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import get_activation
from fabricpc.core.initializers import initialize


# ═══════════════════════════════════════════════════════════════════════════
# 1D CONVOLUTIONAL NODE
# ═══════════════════════════════════════════════════════════════════════════

@register_node("conv1d")
class Conv1DNode(NodeBase):
    """
    1D Convolutional node for sequential/time-series data.
    
    Input format: NLC (batch, length, channels)
    Kernel format: (kernel_len, C_in, C_out)
    Output shape: (L_out, C_out)
    
    Example:
        Input:  (32, 100, 16)  [batch=32, length=100, channels=16]
        Kernel: (3, 16, 32)    [size=3, in_channels=16, out_channels=32]
        Output: (32, 100, 32)  [same length due to SAME padding]
    """

    CONFIG_SCHEMA = {
        "kernel_size": {
            "type": tuple,
            "required": True,
            "description": "Convolution kernel size (kL,)",
        },
        "stride": {
            "type": tuple,
            "default": (1,),
            "description": "Stride for convolution",
        },
        "padding": {
            "type": str,
            "default": "SAME",
            "choices": ["SAME", "VALID"],
            "description": "Padding mode",
        },
        "weight_init": {
            "type": dict,
            "default": {"type": "kaiming"},
            "description": "Weight initialization config",
        },
    }

    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}
    DEFAULT_ACTIVATION_CONFIG = {"type": "relu"}

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv1D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        """Initialize Conv1D kernels and biases."""
        kernel_size = config["kernel_size"]
        out_channels = node_shape[-1]
        weight_init_config = config.get("weight_init", {"type": "kaiming"})

        weights_dict = {}
        for edge_key, in_shape in input_shapes.items():
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)
            key, subkey = jax.random.split(key)
            kernel = initialize(subkey, kernel_shape, weight_init_config)
            weights_dict[edge_key] = kernel

        bias = jnp.zeros((1, 1, out_channels))
        return NodeParams(weights=weights_dict, biases={"b": bias})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: compute prediction and energy.
        
        z_mu = sum of convolutions + bias
        error = z_latent - z_mu
        energy = 0.5 * ||error||²
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        
        # Accumulate convolutions from all inputs
        z_mu = jnp.zeros((batch_size, *out_shape))
        
        for edge_key, input_tensor in inputs.items():
            kernel = params.weights[edge_key]
            conv_out = lax.conv_general_dilated(
                lhs=input_tensor,
                rhs=kernel,
                window_strides=node_info.node_config.get("stride", (1,)),
                padding=node_info.node_config.get("padding", "SAME"),
                dimension_numbers=("NLC", "LIO", "NLC"),
            )
            z_mu = z_mu + conv_out
        
        z_mu = z_mu + params.biases["b"]
        
        # Updated code in convolutional.py
        activation_config = node_info.node_config.get("activation", {"type": "relu"})
        # Unpack the tuple to get the activation function
        activation_fn, _ = get_activation(activation_config)
        # Call the function directly on the tensor
        z_mu_activated = activation_fn(z_mu)
        
        # Compute error and energy
        error = state.z_latent - z_mu_activated
        energy = 0.5 * jnp.sum(error ** 2, axis=tuple(range(1, error.ndim)))
        
        new_state = state._replace(
            z_mu=z_mu_activated,
            error=error,
            energy=energy,
            pre_activation=z_mu,
        )
        
        return jnp.sum(energy), new_state

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool = False,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """Inference pass with automatic differentiation for input gradients."""
        if node_info.in_degree == 0:
            return state, {edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs}
        
        if is_clamped:
            return state, {edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs}
        
        (total_energy, new_state), input_grads = jax.value_and_grad(
            Conv1DNode.forward, argnums=1, has_aux=True
        )(params, inputs, state, node_info)
        
        return new_state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """Learning pass with automatic differentiation for parameter gradients."""
        (total_energy, new_state), params_grad = jax.value_and_grad(
            Conv1DNode.forward, argnums=0, has_aux=True
        )(params, inputs, state, node_info)
        
        return new_state, params_grad


# ═══════════════════════════════════════════════════════════════════════════
# 2D CONVOLUTIONAL NODE
# ═══════════════════════════════════════════════════════════════════════════

@register_node("conv2d")
class Conv2DNode(NodeBase):
    """
    2D Convolutional node for image data.
    
    Input format: NHWC (batch, height, width, channels)
    Kernel format: (kH, kW, C_in, C_out)
    Output shape: (H_out, W_out, C_out)
    
    Example:
        Input:  (32, 28, 28, 3)   [batch=32, 28x28 image, RGB]
        Kernel: (3, 3, 3, 32)     [3x3 kernel, 3 input channels, 32 output]
        Output: (32, 28, 28, 32)  [same size with SAME padding]
    """

    CONFIG_SCHEMA = {
        "kernel_size": {
            "type": tuple,
            "required": True,
            "description": "Convolution kernel size (kH, kW)",
        },
        "stride": {
            "type": tuple,
            "default": (1, 1),
            "description": "Stride for convolution",
        },
        "padding": {
            "type": str,
            "default": "SAME",
            "choices": ["SAME", "VALID"],
            "description": "Padding mode",
        },
        "weight_init": {
            "type": dict,
            "default": {"type": "kaiming"},
            "description": "Weight initialization config",
        },
    }

    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}
    DEFAULT_ACTIVATION_CONFIG = {"type": "relu"}

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv2D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        """Initialize Conv2D kernels and biases."""
        kernel_size = config["kernel_size"]
        out_channels = node_shape[-1]
        weight_init_config = config.get("weight_init", {"type": "kaiming"})

        weights_dict = {}
        for edge_key, in_shape in input_shapes.items():
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)
            key, subkey = jax.random.split(key)
            kernel = initialize(subkey, kernel_shape, weight_init_config)
            weights_dict[edge_key] = kernel

        bias = jnp.zeros((1, 1, 1, out_channels))
        return NodeParams(weights=weights_dict, biases={"b": bias})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: compute prediction and energy.
        
        z_mu = sum of convolutions + bias
        error = z_latent - z_mu
        energy = 0.5 * ||error||²
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        
        # Accumulate convolutions from all inputs
        z_mu = jnp.zeros((batch_size, *out_shape))
        
        for edge_key, input_tensor in inputs.items():
            kernel = params.weights[edge_key]
            conv_out = lax.conv_general_dilated(
                lhs=input_tensor,
                rhs=kernel,
                window_strides=node_info.node_config.get("stride", (1, 1)),
                padding=node_info.node_config.get("padding", "SAME"),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            z_mu = z_mu + conv_out
        
        z_mu = z_mu + params.biases["b"]
        
        # Apply activation
        activation_config = node_info.node_config.get("activation", {"type": "relu"})
        # Inside convolutional.py forward methods
        act_fn, _ = get_activation(activation_config)
        z_mu_activated = act_fn(z_mu) # Call the function directly, not .apply()
        
        # Compute error and energy
        error = state.z_latent - z_mu_activated
        energy = 0.5 * jnp.sum(error ** 2, axis=tuple(range(1, error.ndim)))
        
        new_state = state._replace(
            z_mu=z_mu_activated,
            error=error,
            energy=energy,
            pre_activation=z_mu,
        )
        
        return jnp.sum(energy), new_state

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool = False,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """Inference pass with automatic differentiation for input gradients."""
        if node_info.in_degree == 0:
            return state, {edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs}
        
        if is_clamped:
            return state, {edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs}
        
        (total_energy, new_state), input_grads = jax.value_and_grad(
            Conv2DNode.forward, argnums=1, has_aux=True
        )(params, inputs, state, node_info)
        
        return new_state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """Learning pass with automatic differentiation for parameter gradients."""
        (total_energy, new_state), params_grad = jax.value_and_grad(
            Conv2DNode.forward, argnums=0, has_aux=True
        )(params, inputs, state, node_info)
        
        return new_state, params_grad


# ═══════════════════════════════════════════════════════════════════════════
# 3D CONVOLUTIONAL NODE
# ═══════════════════════════════════════════════════════════════════════════

@register_node("conv3d")
class Conv3DNode(NodeBase):
    """
    3D Convolutional node for volumetric data.
    
    Input format: NDHWC (batch, depth, height, width, channels)
    Kernel format: (kD, kH, kW, C_in, C_out)
    Output shape: (D_out, H_out, W_out, C_out)
    
    Example:
        Input:  (8, 32, 32, 32, 16)  [batch=8, 32x32x32 volume, 16 channels]
        Kernel: (3, 3, 3, 16, 32)    [3x3x3 kernel, 16 input, 32 output]
        Output: (8, 32, 32, 32, 32)  [same volume size]
    """

    CONFIG_SCHEMA = {
        "kernel_size": {
            "type": tuple,
            "required": True,
            "description": "Convolution kernel size (kD, kH, kW)",
        },
        "stride": {
            "type": tuple,
            "default": (1, 1, 1),
            "description": "Stride for convolution",
        },
        "padding": {
            "type": str,
            "default": "SAME",
            "choices": ["SAME", "VALID"],
            "description": "Padding mode",
        },
        "weight_init": {
            "type": dict,
            "default": {"type": "kaiming"},
            "description": "Weight initialization config",
        },
    }

    DEFAULT_ENERGY_CONFIG = {"type": "gaussian"}
    DEFAULT_ACTIVATION_CONFIG = {"type": "relu"}

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv3D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        """Initialize Conv3D kernels and biases."""
        kernel_size = config["kernel_size"]
        out_channels = node_shape[-1]
        weight_init_config = config.get("weight_init", {"type": "kaiming"})

        weights_dict = {}
        for edge_key, in_shape in input_shapes.items():
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)
            key, subkey = jax.random.split(key)
            kernel = initialize(subkey, kernel_shape, weight_init_config)
            weights_dict[edge_key] = kernel

        bias = jnp.zeros((1, 1, 1, 1, out_channels))
        return NodeParams(weights=weights_dict, biases={"b": bias})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: compute prediction and energy.
        
        z_mu = sum of convolutions + bias
        error = z_latent - z_mu
        energy = 0.5 * ||error||²
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        
        # Accumulate convolutions from all inputs
        z_mu = jnp.zeros((batch_size, *out_shape))
        
        for edge_key, input_tensor in inputs.items():
            kernel = params.weights[edge_key]
            conv_out = lax.conv_general_dilated(
                lhs=input_tensor,
                rhs=kernel,
                window_strides=node_info.node_config.get("stride", (1, 1, 1)),
                padding=node_info.node_config.get("padding", "SAME"),
                dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
            )
            z_mu = z_mu + conv_out
        
        z_mu = z_mu + params.biases["b"]
        
        # Apply activation
        activation_config = node_info.node_config.get("activation", {"type": "relu"})
        # Inside convolutional.py forward methods
        act_fn, _ = get_activation(activation_config)
        z_mu_activated = act_fn(z_mu) # Call the function directly, not .apply()

        # Compute error and energy
        error = state.z_latent - z_mu_activated
        energy = 0.5 * jnp.sum(error ** 2, axis=tuple(range(1, error.ndim)))
        
        new_state = state._replace(
            z_mu=z_mu_activated,
            error=error,
            energy=energy,
            pre_activation=z_mu,
        )
        
        return jnp.sum(energy), new_state

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool = False,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """Inference pass with automatic differentiation for input gradients."""
        if node_info.in_degree == 0:
            return state, {edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs}
        
        if is_clamped:
            return state, {edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs}
        
        (total_energy, new_state), input_grads = jax.value_and_grad(
            Conv3DNode.forward, argnums=1, has_aux=True
        )(params, inputs, state, node_info)
        
        return new_state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """Learning pass with automatic differentiation for parameter gradients."""
        (total_energy, new_state), params_grad = jax.value_and_grad(
            Conv3DNode.forward, argnums=0, has_aux=True
        )(params, inputs, state, node_info)
        
        return new_state, params_grad