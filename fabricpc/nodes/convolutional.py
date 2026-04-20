"""
Conv1D, Conv2D, and Conv3D nodes for JAX predictive coding networks.

These nodes implement convolutional operations using JAX's lax.conv_general_dilated.

Input formats:
  - Conv1D: NLC (batch, length, channels)
  - Conv2D: NHWC (batch, height, width, channels)
  - Conv3D: NDHWC (batch, depth, height, width, channels)

Output shapes exclude batch dimension.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, KaimingInitializer, initialize

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


# ═══════════════════════════════════════════════════════════════════════════
# 1D CONVOLUTIONAL NODE
# ═══════════════════════════════════════════════════════════════════════════

class Conv1DNode(NodeBase):
    """
    1D Convolutional node for sequential/time-series data.
    
    Input format: NLC (batch, length, channels)
    Kernel format: (kernel_len, C_in, C_out)
    Output shape: (L_out, C_out)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        kernel_size: Tuple[int],
        stride: Tuple[int] = (1,),
        padding: str = "SAME",
        activation: Optional[ActivationBase] = ReLUActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv1D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Conv1D fan_in = C_in * kL."""
        kernel_size = config.get("kernel_size", (1,))
        C_in = source_shape[-1]
        return C_in * int(np.prod(kernel_size))

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Initialize Conv1D kernels and biases."""
        if config is None:
            config = {}
        
        kernel_size = config.get("kernel_size")
        out_channels = node_shape[-1]
        
        if weight_init is None:
            weight_init = KaimingInitializer()

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes) + 1)

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)
            weights_dict[edge_key] = initialize(keys[i], kernel_shape, weight_init)

        use_bias = config.get("use_bias", True)
        if use_bias:
            bias = jnp.zeros((1, 1, out_channels))
        else:
            bias = jnp.array([])

        return NodeParams(weights=weights_dict, biases={"b": bias} if use_bias else {})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """Forward pass for Conv1D."""
        config = node_info.node_config
        stride = config.get("stride", (1,))
        padding = config.get("padding", "SAME")
        
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        
        pre_activation = jnp.zeros((batch_size, *out_shape))
        
        for edge_key, input_tensor in inputs.items():
            kernel = params.weights[edge_key]
            conv_out = lax.conv_general_dilated(
                lhs=input_tensor,
                rhs=kernel,
                window_strides=stride,
                padding=padding,
                dimension_numbers=("NLC", "LIO", "NLC"),
            )
            pre_activation = pre_activation + conv_out
        
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]
        
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        
        error = state.z_latent - z_mu
        
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error,
        )
        
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        
        return total_energy, state


# ═══════════════════════════════════════════════════════════════════════════
# 2D CONVOLUTIONAL NODE
# ═══════════════════════════════════════════════════════════════════════════

class Conv2DNode(NodeBase):
    """
    2D Convolutional node for image data.
    
    Input format: NHWC (batch, height, width, channels)
    Kernel format: (kH, kW, C_in, C_out)
    Output shape: (H_out, W_out, C_out)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        activation: Optional[ActivationBase] = ReLUActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv2D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Conv2D fan_in = C_in * kH * kW."""
        kernel_size = config.get("kernel_size", (1, 1))
        C_in = source_shape[-1]
        return C_in * int(np.prod(kernel_size))

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Initialize Conv2D kernels and biases."""
        if config is None:
            config = {}
            
        kernel_size = config.get("kernel_size")
        out_channels = node_shape[-1]
        
        if weight_init is None:
            weight_init = KaimingInitializer()

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes) + 1)

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)
            weights_dict[edge_key] = initialize(keys[i], kernel_shape, weight_init)

        use_bias = config.get("use_bias", True)
        if use_bias:
            bias = jnp.zeros((1, 1, 1, out_channels))
        else:
            bias = jnp.array([])

        return NodeParams(weights=weights_dict, biases={"b": bias} if use_bias else {})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """Forward pass for Conv2D."""
        config = node_info.node_config
        stride = config.get("stride", (1, 1))
        padding = config.get("padding", "SAME")
        
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        
        pre_activation = jnp.zeros((batch_size, *out_shape))
        
        for edge_key, input_tensor in inputs.items():
            kernel = params.weights[edge_key]
            conv_out = lax.conv_general_dilated(
                lhs=input_tensor,
                rhs=kernel,
                window_strides=stride,
                padding=padding,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            pre_activation = pre_activation + conv_out
        
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]
        
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        
        error = state.z_latent - z_mu
        
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error,
        )
        
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        
        return total_energy, state


# ═══════════════════════════════════════════════════════════════════════════
# 3D CONVOLUTIONAL NODE
# ═══════════════════════════════════════════════════════════════════════════

class Conv3DNode(NodeBase):
    """
    3D Convolutional node for volumetric data.
    
    Input format: NDHWC (batch, depth, height, width, channels)
    Kernel format: (kD, kH, kW, C_in, C_out)
    Output shape: (D_out, H_out, W_out, C_out)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: str = "SAME",
        activation: Optional[ActivationBase] = ReLUActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv3D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Conv3D fan_in = C_in * kD * kH * kW."""
        kernel_size = config.get("kernel_size", (1, 1, 1))
        C_in = source_shape[-1]
        return C_in * int(np.prod(kernel_size))

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Initialize Conv3D kernels and biases."""
        if config is None:
            config = {}
            
        kernel_size = config.get("kernel_size")
        out_channels = node_shape[-1]
        
        if weight_init is None:
            weight_init = KaimingInitializer()

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes) + 1)

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)
            weights_dict[edge_key] = initialize(keys[i], kernel_shape, weight_init)

        use_bias = config.get("use_bias", True)
        if use_bias:
            bias = jnp.zeros((1, 1, 1, 1, out_channels))
        else:
            bias = jnp.array([])

        return NodeParams(weights=weights_dict, biases={"b": bias} if use_bias else {})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """Forward pass for Conv3D."""
        config = node_info.node_config
        stride = config.get("stride", (1, 1, 1))
        padding = config.get("padding", "SAME")
        
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        
        pre_activation = jnp.zeros((batch_size, *out_shape))
        
        for edge_key, input_tensor in inputs.items():
            kernel = params.weights[edge_key]
            conv_out = lax.conv_general_dilated(
                lhs=input_tensor,
                rhs=kernel,
                window_strides=stride,
                padding=padding,
                dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
            )
            pre_activation = pre_activation + conv_out
        
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]
        
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        
        error = state.z_latent - z_mu
        
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error,
        )
        
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        
        return total_energy, state
