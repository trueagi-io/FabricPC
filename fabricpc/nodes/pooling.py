"""
Pooling operations for predictive coding graphs.

Provides ``MaxPool`` (windowed max pooling) and ``AvgPool`` (windowed or
global average pooling), both built on a shared ``_PoolBase``. They are
parameter-free graph elements: you wire them in like any other operation, but
the dropped ``Node`` suffix signals that they carry no learnable weights and
exist only to reduce spatial dimensions — not to learn a representation.

muPC scaling note
-----------------
Pooling is a weightless *transformation*, not an identity skip-connection
path. The framework distinguishes two cases:

  * ``is_variance_scalable=False`` → strictly for identity-mapping skip-connection
    paths (see SkipConnection, transformer mask/residual edges). The base class
    even errors if ``is_skip_connection`` and ``is_variance_scalable=True`` are
    set together.
  * Weightless nodes that still perform a transformation (IdentityNode summing
    multi-input edges, pooling nodes reducing spatial dims) follow the
    IdentityNode convention: override ``get_weight_fan_in`` to return 1. The muPC
    formula ``a = gain / sqrt(fan_in * K_slot * L)`` then reduces to
    ``a = gain / sqrt(K_slot * L)`` — compensating only for multi-edge summation
    variance, not for a non-existent weight matrix.

With a single incoming edge in a non-residual graph (K_slot=1, L=1) and
IdentityActivation (gain=1.0), the scale is exactly 1.0 — a no-op. Returning the
upstream channel count instead (the base default) would silently attenuate
activations and gradients through every pool by ``1/sqrt(C_in)``.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Sequence, Union, TYPE_CHECKING
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class _PoolBase(NodeBase):
    """
    Shared structure for parameter-free pooling operations.

    Subclasses implement the reduction in ``_pool(x_sum, node_info)``. The common
    parts — slots, fan_in=1, empty params, and the forward template (sum inputs →
    pool → activation → error → energy) — live here. Not exported; use
    ``MaxPool`` or ``AvgPool``.
    """

    # -- shared validation for windowed pooling ----------------------------

    @staticmethod
    def _validate_windowed(
        shape: Tuple[int, ...],
        window_shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]],
        node_name: str,
    ) -> Tuple[int, Tuple[int, ...]]:
        """Validate windowed-pool args; return (spatial_rank, resolved_stride).

        ``stride`` defaults to ``window_shape`` (non-overlapping pooling).
        """
        spatial_rank = len(shape) - 1
        if spatial_rank not in (1, 2, 3):
            raise ValueError(
                f"{node_name} shape must have 2-4 elements (spatial_rank 1/2/3). "
                f"Got shape={shape}."
            )
        if len(window_shape) != spatial_rank:
            raise ValueError(
                f"window_shape length {len(window_shape)} must equal "
                f"spatial_rank {spatial_rank} inferred from shape={shape}."
            )
        if stride is None:
            stride = window_shape
        if len(stride) != spatial_rank:
            raise ValueError(
                f"stride length {len(stride)} must equal "
                f"spatial_rank {spatial_rank} inferred from shape={shape}."
            )
        return spatial_rank, stride

    @staticmethod
    def _format_pool_padding(
        padding: Union[str, Sequence[Tuple[int, int]]],
    ) -> Union[str, Tuple[Tuple[int, int], ...]]:
        """Format padding for ``lax.reduce_window``.

        Strings ("SAME"/"VALID") pass through. A sequence of spatial ``(low, high)``
        pairs is wrapped with ``(0, 0)`` for the batch and channel dims, since
        ``reduce_window`` requires a pair for every dimension (unlike
        ``conv_general_dilated``, which takes spatial-only padding).
        """
        if isinstance(padding, str):
            return padding
        spatial = tuple((int(lo), int(hi)) for lo, hi in padding)
        return ((0, 0),) + spatial + ((0, 0),)

    # -- NodeBase contract --------------------------------------------------

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """No weight matrix — return 1, matching the IdentityNode convention.

        With fan_in=1 the muPC formula ``a = gain / sqrt(fan_in * K_slot * L)``
        reduces to ``a = gain / sqrt(K_slot * L)`` — compensating only for
        multi-edge summation variance, not for a non-existent weight matrix.
        """
        return 1

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional["InitializerBase"] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Pooling has no learnable parameters."""
        return NodeParams(weights={}, biases={})

    @staticmethod
    def _pool(x_sum: jnp.ndarray, node_info: NodeInfo) -> jnp.ndarray:
        """Subclass-specific reduction over the summed inputs. Override."""
        raise NotImplementedError

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Sum incoming edges, apply the subclass reduction, apply activation,
        and compute energy. Dispatches the reduction through
        ``node_info.node_class`` so the concrete subclass's ``_pool`` runs.
        """
        x_sum = sum(inputs.values())

        pre_activation = node_info.node_class._pool(x_sum, node_info)

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)

        return jnp.sum(state.energy), state


class MaxPool(_PoolBase):
    """
    Windowed max pooling for spatial downsampling.

    Reduces spatial dimensions by taking the maximum value over a sliding window.
    Channels-last format: (Batch, Spatial..., Channels). No learnable params.

    Args:
        shape:        Output shape excluding batch dimension.
        name:         Node name.
        window_shape: Spatial window size per axis.
        stride:       Stride per spatial axis. Defaults to ``window_shape``.
        padding:      "SAME" / "VALID", or a sequence of spatial (low, high) pairs.
        activation:   Default: IdentityActivation().
        energy:       Default: GaussianEnergy().
        latent_init:  Default: NormalInitializer().
        slots:        Custom slot dict.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        window_shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
        activation: Optional["ActivationBase"] = None,
        energy: Optional["EnergyFunctional"] = None,
        latent_init: Optional["InitializerBase"] = None,
        slots: Optional[Dict[str, SlotSpec]] = None,
    ):
        _, stride = self._validate_windowed(shape, window_shape, stride, "MaxPool")

        if activation is None:
            activation = IdentityActivation()
        if energy is None:
            energy = GaussianEnergy()
        if latent_init is None:
            latent_init = NormalInitializer()

        self._slots = slots or {"in": SlotSpec(name="in", is_multi_input=True)}

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=None,  # No weights
            use_bias=False,  # No bias
            window_shape=window_shape,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def _pool(x_sum: jnp.ndarray, node_info: NodeInfo) -> jnp.ndarray:
        config = node_info.node_config
        window_shape = config["window_shape"]
        stride = config["stride"]
        padding = _PoolBase._format_pool_padding(config["padding"])

        # Batch and channel are not pooled, so their window/stride is 1.
        full_window = (1,) + tuple(window_shape) + (1,)
        full_strides = (1,) + tuple(stride) + (1,)

        return lax.reduce_window(
            x_sum, -jnp.inf, lax.max, full_window, full_strides, padding
        )


class AvgPool(_PoolBase):
    """
    Average pooling — windowed or global.

    * Windowed (``global_pool=False``): averages over a sliding window, keeping
      spatial structure like MaxPool: (B, Spatial..., C) -> (B, Spatial'..., C).
      ``shape`` carries the reduced spatial rank, e.g. (14, 14, 32).
    * Global (``global_pool=True``): averages over *all* spatial dimensions,
      collapsing them: (B, Spatial..., C) -> (B, C). ``shape`` is just (C,) and
      ``window_shape`` is ignored.

    No learnable parameters.

    Args:
        shape:        Output shape excluding batch dimension.
        name:         Node name.
        window_shape: Spatial window size per axis (windowed mode only).
        stride:       Stride per spatial axis. Defaults to ``window_shape``.
        padding:      "SAME" / "VALID", or a sequence of spatial (low, high) pairs.
        global_pool:  If True, average over all spatial dims -> (B, C).
        count_include_pad: Windowed mode only. If True (default, matches
                      PyTorch), divide each window by the full window volume
                      (padded cells count as zeros). If False, divide by the
                      number of real (non-padded) elements. No effect in
                      global mode.
        activation:   Default: IdentityActivation().
        energy:       Default: GaussianEnergy().
        latent_init:  Default: NormalInitializer().
        slots:        Custom slot dict.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        window_shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
        global_pool: bool = False,
        count_include_pad: bool = True,
        activation: Optional["ActivationBase"] = None,
        energy: Optional["EnergyFunctional"] = None,
        latent_init: Optional["InitializerBase"] = None,
        slots: Optional[Dict[str, SlotSpec]] = None,
    ):
        if global_pool:
            # Global mode collapses all spatial dims; window/stride unused.
            window_shape = window_shape or ()
            stride = stride or ()
        else:
            if window_shape is None:
                raise ValueError(
                    "AvgPool: window_shape is required when global_pool=False."
                )
            _, stride = self._validate_windowed(shape, window_shape, stride, "AvgPool")

        if activation is None:
            activation = IdentityActivation()
        if energy is None:
            energy = GaussianEnergy()
        if latent_init is None:
            latent_init = NormalInitializer()

        self._slots = slots or {"in": SlotSpec(name="in", is_multi_input=True)}

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=None,  # No weights
            use_bias=False,  # No bias
            window_shape=window_shape,
            stride=stride,
            padding=padding,
            global_pool=global_pool,
            count_include_pad=count_include_pad,
        )

    @staticmethod
    def _pool(x_sum: jnp.ndarray, node_info: NodeInfo) -> jnp.ndarray:
        config = node_info.node_config

        if config.get("global_pool", False):
            # Average over every spatial axis: (B, Spatial..., C) -> (B, C).
            spatial_axes = tuple(range(1, x_sum.ndim - 1))
            return jnp.mean(x_sum, axis=spatial_axes)

        window_shape = config["window_shape"]
        stride = config["stride"]
        padding = _PoolBase._format_pool_padding(config["padding"])

        full_window = (1,) + tuple(window_shape) + (1,)
        full_strides = (1,) + tuple(stride) + (1,)

        summed = lax.reduce_window(
            x_sum, 0.0, lax.add, full_window, full_strides, padding
        )

        if config.get("count_include_pad", True):
            # Divide by the full window volume; padded cells count as zeros.
            # Exact for the default VALID padding (no padding is added).
            return summed / float(np.prod(window_shape))

        # count_include_pad=False: divide each window by its number of *real*
        # (non-padded) elements. Summing an all-ones array through the identical
        # window yields that count exactly (padding contributes 0). The safe
        # divisor + jnp.where guard the degenerate fully-padded window
        # (count == 0), which would otherwise emit NaN that propagates through
        # the network; matching PyTorch, such windows yield 0.
        counts = lax.reduce_window(
            jnp.ones_like(x_sum), 0.0, lax.add, full_window, full_strides, padding
        )
        safe_counts = jnp.where(counts > 0, counts, 1.0)
        return jnp.where(counts > 0, summed / safe_counts, 0.0)
