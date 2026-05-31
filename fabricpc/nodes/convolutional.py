"""
Unified convolutional node (1D, 2D, 3D) for predictive coding graphs.

Spatial rank is inferred from ``len(shape) - 1`` at construction time.
All gradient computation, energy accumulation, and the activation convention
are handled by NodeBase; only ``get_slots``, ``get_weight_fan_in``,
``initialize_params``, and ``forward`` are implemented here.

Design notes
------------
The user-supplied ``weight_init`` flows through verbatim as
``node_info.weight_init`` and is invoked directly on the conv kernel shape;
the shape-aware ``KaimingInitializer`` / ``XavierInitializer`` derive
fan-in / fan-out from that shape on their own. The conv node does not
inspect the activation or rewrite the initializer — pairing an
appropriate initializer config (e.g. ``nonlinearity="leaky_relu"``) with
the chosen activation is the caller's responsibility.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union, Sequence, TYPE_CHECKING
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    KaimingInitializer,
    ZerosInitializer,
    NormalInitializer,
    initialize,
)

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class ConvNode(NodeBase):
    """
    Unified convolutional node (1D, 2D, 3D) for predictive coding graphs.

    Spatial rank is inferred from the output shape:
        len(shape) == 2  →  1D conv  (L_out, C_out)
        len(shape) == 3  →  2D conv  (H_out, W_out, C_out)
        len(shape) == 4  →  3D conv  (D_out, H_out, W_out, C_out)

    Args:
        shape:       Output shape excluding batch dimension.
        name:        Node name.
        kernel_size: Spatial kernel size per axis. **Required — no default.**
        stride:      Stride per spatial axis. Defaults to all-ones.
        padding:     "SAME" or "VALID", or sequence of (low, high) pad pairs.
        activation:  Default: ReLUActivation().
        energy:      Default: GaussianEnergy().
        use_bias:    Default: True.
        weight_init: Default: KaimingInitializer().
        bias_init:   Default: ZerosInitializer().
        latent_init: Default: NormalInitializer().
        slots:       Custom slot dict for skip connections / gating.
    """

    _DIM_NUMBERS = {
        1: ("NLC", "LIO", "NLC"),
        2: ("NHWC", "HWIO", "NHWC"),
        3: ("NDHWC", "DHWIO", "NDHWC"),
    }

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        kernel_size: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        activation: Optional["ActivationBase"] = None,
        energy: Optional["EnergyFunctional"] = None,
        use_bias: bool = True,
        weight_init: Optional["InitializerBase"] = None,
        bias_init: Optional["InitializerBase"] = None,
        latent_init: Optional["InitializerBase"] = None,
        slots: Optional[Dict[str, SlotSpec]] = None,
    ):
        """
        Args:
            shape: Output shape excluding batch dimension.
            name: Node name.
            kernel_size: Spatial kernel size per axis.
            stride: Stride per spatial axis. Defaults to all-ones.
            padding: "SAME" or "VALID", or sequence of (low, high) pad pairs.
            activation: ActivationBase instance. Default: ReLUActivation.
            energy: EnergyFunctional instance. Default: GaussianEnergy.
            use_bias: Whether to add a learnable bias. Default: True.
            weight_init: Initializer for kernels. Default: KaimingInitializer.
            bias_init: Initializer for bias. Default: ZerosInitializer.
            latent_init: Initializer for latent states. Default: NormalInitializer.
            slots: Custom slot dictionary.
        """
        spatial_rank = len(shape) - 1
        if spatial_rank not in (1, 2, 3):
            raise ValueError(
                f"ConvNode shape must have 2-4 elements (spatial_rank 1/2/3). Got shape={shape}."
            )
        if len(kernel_size) != spatial_rank:
            raise ValueError(
                f"kernel_size length {len(kernel_size)} must equal "
                f"spatial_rank {spatial_rank} inferred from shape={shape}."
            )

        # Fill defaults in the body — never in the signature — to avoid the
        # mutable-default-argument pitfall.
        if activation is None:
            activation = ReLUActivation()
        if energy is None:
            energy = GaussianEnergy()
        if latent_init is None:
            latent_init = NormalInitializer()
        if stride is None:
            stride = (1,) * spatial_rank

        self._slots = slots or {"in": SlotSpec(name="in", is_multi_input=True)}

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            bias_init=bias_init,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """fan_in = C_in × ∏(kernel_size). Overrides NodeBase (which returns only C_in)."""
        kernel_size = config.get("kernel_size")
        return source_shape[-1] * int(np.prod(kernel_size))

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional["InitializerBase"] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialise kernels (*kernel_size, C_in, C_out) and bias (1,…,1, C_out).
        """
        if config is None:
            config = {}

        kernel_size = config.get("kernel_size")
        out_channels = node_shape[-1]
        use_bias = config.get("use_bias", True)
        bias_init = config.get("bias_init")
        if use_bias and bias_init is None:
            bias_init = ZerosInitializer()

        if weight_init is None:
            weight_init = KaimingInitializer()

        keys = jax.random.split(key, len(input_shapes) + 1)

        weights_dict = {}
        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)
            weights_dict[edge_key] = initialize(keys[i], kernel_shape, weight_init)

        if use_bias:
            bias_shape = (1,) * len(node_shape) + (out_channels,)
            return NodeParams(
                weights=weights_dict,
                biases={"b": initialize(keys[-1], bias_shape, bias_init)},
            )
        return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Standard NodeBase forward contract:
        conv sum → bias → activation → error → energy_functional → total_energy.
        """
        config = node_info.node_config
        dim_numbers = ConvNode._DIM_NUMBERS[len(node_info.shape) - 1]

        pre_activation = jnp.zeros((state.z_latent.shape[0], *node_info.shape))
        for edge_key, x in inputs.items():
            pre_activation = pre_activation + lax.conv_general_dilated(
                lhs=x,
                rhs=params.weights[edge_key],
                window_strides=config.get("stride"),
                padding=config.get("padding"),
                dimension_numbers=dim_numbers,
            )

        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state
