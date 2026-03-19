"""
Linear node implementation for JAX predictive coding networks.

This implements a linear transformation node with configurable activation functions.
The node has a single multi-input slot that accepts multiple incoming connections.

By default, linear nodes apply matrix multiplication on the **last axis only**, which is
standard for embeddings, projections, and transformer layers. This means:
- Input shape: (batch, ..., in_features)
- Weight shape: (in_features, out_features)
- Output shape: (batch, ..., out_features)

For fully-connected (dense) behavior that flattens all dimensions, set `flatten_input=True`.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import (
    NodeBase,
    SlotSpec,
    FlattenInputMixin,
)
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, KaimingInitializer

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class Linear(FlattenInputMixin, NodeBase):
    """
    Linear transformation node: y = activation(W @ x + b)

    This node type:
    - Has a single multi-input slot named "in"
    - Concatenates all inputs and applies a linear transformation
    - Supports various activation functions
    - Implements local Hebbian learning

    Uses FlattenInputMixin for flatten/reshape operations.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
    ):
        """
        Args:
            shape: Output shape tuple (excluding batch dimension)
            name: Node name
            activation: ActivationBase instance (default: IdentityActivation)
            energy: EnergyFunctional instance (default: GaussianEnergy)
            use_bias: Whether to use bias (default: True)
            flatten_input: If True, flatten all input dims for dense behavior (default: False)
            weight_init: InitializerBase instance for weights (default: NormalInitializer)
            latent_init: InitializerBase instance for latent states
        """
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            flatten_input=flatten_input,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Linear nodes have a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        """
        Initialize weight matrix and bias vector.

        Weight shape depends on `flatten_input` config:
        - flatten_input=False (default): weights have shape (in_features, out_features)
        - flatten_input=True: weights have shape (in_numel, out_numel) for dense layers.

        Args:
            key: JAX random key
            node_shape: Output shape of this node (excluding batch dimension)
            input_shapes: Dictionary with EdgeInfo.key -> source shape for that edge
            weight_init: InitializerBase instance for weight initialization, or None
            config: Node configuration with weight_init settings

        Returns:
            NodeParams with initialized W and b
        """
        from fabricpc.core.initializers import NormalInitializer, initialize

        flatten_input = config.get("flatten_input", False)

        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.05)

        # Split key for weights and biases
        key_w, key_b = jax.random.split(key)

        # Initialize weight matrix for each incoming edge
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
                in_numel = int(np.prod(in_shape))
                out_numel = int(np.prod(node_shape))
                weight_shape = (in_numel, out_numel)
            else:
                in_features = in_shape[-1]
                out_features = node_shape[-1]
                weight_shape = (in_features, out_features)

            weights_dict[edge_key] = initialize(
                rand_key_w[edge_key], weight_shape, weight_init
            )

        # Initialize bias (usually zeros)
        # Bias shape for proper broadcasting, prepending batch dimension: (1, ..., 1, out_features)
        use_bias = config.get("use_bias", True)
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            b = jnp.zeros(bias_shape)

        return NodeParams(weights=weights_dict, biases={"b": b} if use_bias else {})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
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
            state: NodeState for this node
            node_info: NodeInfo object (contains activation instance, energy instance, etc.)

        Returns:
            Tuple of (total_energy, NodeState)
        """
        # Get batch size and output shape
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        flatten_input = node_info.node_config.get("flatten_input", False)

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

        # Add bias if present
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation function from node_info
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        # Error
        error = state.z_latent - z_mu

        # Update node state
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Compute energy, accumulate the self-latent gradient
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state


class LinearExplicitGrad(Linear):
    """
    Linear node with explicit (non-autodiff) gradient computation.
    Demonstrates overriding NodeBase's autodiff-based gradient computation.

    Useful for:
    - Verifying correctness of manual gradient implementations
    - Prototyping optimized gradients
    - Debugging gradient computation issuesr
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init: Optional[InitializerBase] = NormalInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            use_bias=use_bias,
            flatten_input=flatten_input,
            weight_init=weight_init,
            latent_init=latent_init,
        )

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool,
        var_inputs: Dict[str, jnp.ndarray] = None,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """Forward pass with explicit gradient computation."""
        node_class = node_info.node_class

        # Forward pass to get new state
        _, state = node_class.forward(params, inputs, state, node_info)

        # Gain-modulated error computation
        gain_mod_error = node_class.compute_gain_mod_error(state, node_info)

        # Determine the energy functional to use
        energy_obj = node_info.energy
        energy_type = type(energy_obj).__name__
        flatten_input = node_info.node_config.get("flatten_input", False)
        input_grads = {}

        for edge_key, z in inputs.items():
            source_shape = z.shape[1:]

            if energy_type == "GaussianEnergy":
                if flatten_input:
                    gain_mod_error_flat = FlattenInputMixin.flatten_input(
                        gain_mod_error
                    )
                    grad_flat = -jnp.matmul(
                        gain_mod_error_flat, params.weights[edge_key].T
                    )
                    grad_contribution = FlattenInputMixin.reshape_output(
                        grad_flat, source_shape
                    )
                else:
                    grad_contribution = -jnp.matmul(
                        gain_mod_error,
                        params.weights[edge_key].T,
                    )
            else:
                raise NotImplementedError(
                    f"energy functional '{energy_type}' not implemented in LinearExplicitGrad."
                )

            input_grads[edge_key] = grad_contribution

        return state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """Forward pass with explicit weight gradient computation."""
        node_class = node_info.node_class

        # Forward pass to get new state
        _, state = node_class.forward(params, inputs, state, node_info)

        # Gain-modulated error computation
        gain_mod_error = node_class.compute_gain_mod_error(state, node_info)

        flatten_input = node_info.node_config.get("flatten_input", False)
        weight_grads = {}
        bias_grads = {}

        # Weight gradient
        for edge_key, in_tensor in inputs.items():
            if flatten_input:
                in_flat = FlattenInputMixin.flatten_input(in_tensor)
                gain_mod_error_flat = FlattenInputMixin.flatten_input(gain_mod_error)
                grad_w = -jnp.matmul(in_flat.T, gain_mod_error_flat)
            else:
                in_shape = in_tensor.shape
                err_shape = gain_mod_error.shape
                in_flat = in_tensor.reshape(-1, in_shape[-1])
                err_flat = gain_mod_error.reshape(-1, err_shape[-1])
                grad_w = -jnp.matmul(in_flat.T, err_flat)
            weight_grads[edge_key] = grad_w

        # Bias gradient
        if "b" in params.biases:
            grad_b = -jnp.sum(gain_mod_error, axis=0, keepdims=True)
            bias_grads["b"] = grad_b

        return state, NodeParams(weights=weight_grads, biases=bias_grads)

    @staticmethod
    def compute_gain_mod_error(state: NodeState, node_info: NodeInfo) -> jnp.ndarray:
        """Compute gain-modulated error for this node.

        Returns:
            gain_mod_error array (error * activation derivative)
        """
        activation = node_info.activation
        f_prime = type(activation).derivative(state.pre_activation, activation.config)
        return state.error * f_prime
