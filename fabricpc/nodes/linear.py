"""Linear node implementation for JAX predictive coding networks."""

from typing import Dict, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ActivationBase, IdentityActivation
from fabricpc.core.energy import EnergyFunctional, GaussianEnergy
from fabricpc.core.initializers import (
    InitializerBase,
    NormalInitializer,
    initialize,
)


class LinearNode(FlattenInputMixin, NodeBase):
    """Linear transformation node: y = activation(W @ x + b)."""

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        activation: ActivationBase | None = None,
        energy: EnergyFunctional | None = None,
        latent_init: InitializerBase | None = None,
        weight_init: InitializerBase | None = None,
        use_bias: bool = True,
        flatten_input: bool = False,
    ):
        super().__init__(
            name,
            shape,
            activation=activation or IdentityActivation(),
            energy=energy or GaussianEnergy(),
            latent_init=latent_init,
            weight_init=weight_init or NormalInitializer(),
            use_bias=use_bias,
            flatten_input=flatten_input,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        flatten_input = bool(config.get("flatten_input", False))
        weight_initializer = config.get("weight_init", NormalInitializer())

        key_w, _ = jax.random.split(key)
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
                rand_key_w[edge_key], weight_shape, weight_initializer
            )

        use_bias = bool(config.get("use_bias", True))
        if use_bias:
            bias_shape = (1,) + node_shape
            b = jnp.zeros(bias_shape)

        return NodeParams(weights=weights_dict, biases={"b": b} if use_bias else {})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        flatten_input = bool(node_info.node_config.get("flatten_input", False))

        if node_info.in_degree == 0:
            z_mu = state.z_latent
            pre_activation = jnp.zeros_like(state.z_latent)
            error = jnp.zeros_like(state.z_latent)
        else:
            if flatten_input:
                pre_activation = FlattenInputMixin.compute_linear(
                    inputs, params.weights, batch_size, out_shape
                )
            else:
                pre_activation = jnp.zeros((batch_size,) + out_shape)
                for edge_key, x in inputs.items():
                    pre_activation = pre_activation + jnp.matmul(
                        x, params.weights[edge_key]
                    )

            if "b" in params.biases and params.biases["b"].size > 0:
                pre_activation = pre_activation + params.biases["b"]

            z_mu = node_info.activation.forward(pre_activation)
            error = state.z_latent - z_mu

        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node.__class__.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state


class LinearExplicitGrad(LinearNode):
    """Linear node variant with explicit gradient computation."""

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        node_cls = node_info.node.__class__
        _, state = node_cls.forward(params, inputs, state, node_info)
        state = node_cls.compute_gain_mod_error(state, node_info)

        flatten_input = bool(node_info.node_config.get("flatten_input", False))
        input_grads = {}

        if not isinstance(node_info.energy, GaussianEnergy):
            raise NotImplementedError(
                "LinearExplicitGrad currently supports GaussianEnergy only"
            )

        for edge_key, z in inputs.items():
            source_shape = z.shape[1:]
            if flatten_input:
                gain_mod_error_flat = FlattenInputMixin.flatten_input(
                    state.substructure["gain_mod_error"]
                )
                grad_flat = -jnp.matmul(gain_mod_error_flat, params.weights[edge_key].T)
                grad_contribution = FlattenInputMixin.reshape_output(
                    grad_flat, source_shape
                )
            else:
                grad_contribution = -jnp.matmul(
                    state.substructure["gain_mod_error"],
                    params.weights[edge_key].T,
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
        node_cls = node_info.node.__class__
        _, state = node_cls.forward(params, inputs, state, node_info)
        state = node_cls.compute_gain_mod_error(state, node_info)

        flatten_input = bool(node_info.node_config.get("flatten_input", False))
        weight_grads = {}
        bias_grads = {}

        for edge_key, in_tensor in inputs.items():
            if flatten_input:
                in_flat = FlattenInputMixin.flatten_input(in_tensor)
                gain_mod_error_flat = FlattenInputMixin.flatten_input(
                    state.substructure["gain_mod_error"]
                )
                grad_w = -jnp.matmul(in_flat.T, gain_mod_error_flat)
            else:
                in_shape = in_tensor.shape
                err_shape = state.substructure["gain_mod_error"].shape
                in_flat = in_tensor.reshape(-1, in_shape[-1])
                err_flat = state.substructure["gain_mod_error"].reshape(
                    -1, err_shape[-1]
                )
                grad_w = -jnp.matmul(in_flat.T, err_flat)
            weight_grads[edge_key] = grad_w

        if "b" in params.biases:
            grad_b = -jnp.sum(
                state.substructure["gain_mod_error"], axis=0, keepdims=True
            )
            bias_grads["b"] = grad_b

        return state, NodeParams(weights=weight_grads, biases=bias_grads)

    @staticmethod
    def compute_gain_mod_error(state: NodeState, node_info: NodeInfo) -> NodeState:
        f_prime = node_info.activation.derivative(state.pre_activation)
        gain_mod_error = state.error * f_prime
        return state._replace(substructure={"gain_mod_error": gain_mod_error})
