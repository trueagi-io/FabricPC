"""
Tensor initializers for predictive coding networks.

This module provides:
- InitializerBase abstract class with constructor-based configuration
- Built-in initializers (Zeros, Ones, Normal, Uniform, Xavier, Kaiming)

Initializers are context-agnostic: they don't know if they're initializing
weights or latent states. The caller determines the context.

User Extensibility
------------------
Users can create custom initializers by extending InitializerBase:

    class MyInitializer(InitializerBase):
        def __init__(self, gain=1.0):
            super().__init__(gain=gain)

        @staticmethod
        def initialize(key, shape, config=None):
            config = config or {}
            gain = config.get("gain", 1.0)
            return gain * jax.random.normal(key, shape)

Usage
-----
Initializers are instantiated with their parameters:

    init = NormalInitializer(mean=0.0, std=0.05)
    init = XavierInitializer(distribution="uniform")
    init = KaimingInitializer(mode="fan_out", nonlinearity="relu")
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.core._frozen import FrozenConfig


def _fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Shape-aware (fan_in, fan_out) for Linear and ND conv kernel layouts.

    Linear ``(in, out)`` recovers fan_in=in / fan_out=out; an ND kernel
    ``(*spatial, C_in, C_out)`` gives fan_in=prod(spatial)*C_in and
    fan_out=prod(spatial)*C_out. A rank-1 shape uses fan_in = fan_out.
    """
    if len(shape) >= 2:
        return int(np.prod(shape[:-1])), int(np.prod(shape[:-2])) * shape[-1]
    return shape[0], shape[0]


# =============================================================================
# Initializer Base Class
# =============================================================================


class InitializerBase(FrozenConfig, ABC):
    """
    Abstract base class for tensor initializers.

    Initializers are context-agnostic: they don't know if they're initializing
    weights or latent states. The caller determines the context.

    All initialize() methods are static for JAX compatibility (pure functions, no state).

    Instances are frozen after construction (see ``FrozenConfig``): attributes
    cannot be set or deleted and ``config`` keys cannot be added, removed, or
    reassigned, so one default instance is safe to share as a signature default.
    The freeze is shallow: construct only with immutable scalar config values.

    Required methods:
        - initialize(): Generate initialized array

    Optional methods:
        - element_variance(): Per-element variance of the distribution, in
          closed form. Needed only by nodes that build a weight matrix
          internally and must report their own muPC variance factor
          (StorkeyHopfield's Hopfield matrix W). All built-in initializers
          implement it.
    """

    @staticmethod
    @abstractmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Initialize array with specified shape.

        Args:
            key: JAX random key
            shape: Shape of array to create
            config: Optional configuration dict for initialization parameters

        Returns:
            Initialized array of specified shape
        """
        pass

    @staticmethod
    def element_variance(
        shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> float:
        """
        Per-element variance of the distribution this initializer draws from.

        Shape-dependent for fan-based schemes (Xavier, Kaiming), constant for
        the rest. This is the quantity muPC's scaling is built on, exposed so a
        node holding an internally-initialized weight matrix can derive how
        that matrix scales input variance.

        Args:
            shape: Shape the array would be initialized with.
            config: Optional configuration dict, as passed to initialize().

        Returns:
            Variance of a single element, in closed form.
        """
        raise NotImplementedError(
            "This initializer does not implement element_variance(). Implement "
            "it to use the initializer with a node that derives its muPC "
            "variance factor from its own weights (e.g. StorkeyHopfield)."
        )


# =============================================================================
# Built-in Initializers
# =============================================================================


class ZerosInitializer(InitializerBase):
    """
    Initialize with zeros.

    Useful for biases or initial states where zero is a sensible default.
    """

    def __init__(self, gain=1.0):
        super().__init__()

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Return array of zeros."""
        return jnp.zeros(shape)

    @staticmethod
    def element_variance(
        shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> float:
        """Constant zero, so no variance."""
        return 0.0


class OnesInitializer(InitializerBase):
    """
    Initialize with ones.

    Useful for scaling factors or multiplicative parameters.
    """

    def __init__(self, gain=1.0):
        super().__init__(gain=gain)

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        config = config or {}
        gain = config.get("gain", 1.0)
        """Return array of ones."""
        return gain * jnp.ones(shape)

    @staticmethod
    def element_variance(
        shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> float:
        """Constant ``gain``, so no variance."""
        return 0.0


class NormalInitializer(InitializerBase):
    """
    Normal (Gaussian) distribution initialization.

    Values are drawn from N(mean, std^2).

    Args:
        mean: Mean of the distribution (default: 0.0)
        std: Standard deviation (default: 0.05)
    """

    def __init__(self, mean=0.0, std=0.05, gain=1.0):
        super().__init__(mean=mean, std=std, gain=gain)

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize from normal distribution: mean + std * N(0, 1)."""
        config = config or {}
        mean = config.get("mean", 0.0)
        std = config.get("std", 0.05)
        gain = config.get("gain", 1.0)
        return mean + gain * std * jax.random.normal(key, shape)

    @staticmethod
    def element_variance(
        shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> float:
        """``(gain * std)^2``. The mean shifts the distribution, not its spread."""
        config = config or {}
        return float(config.get("gain", 1.0) * config.get("std", 0.05)) ** 2


class UniformInitializer(InitializerBase):
    """
    Uniform distribution initialization.

    Values are drawn from U(min, max).

    Args:
        min_val: Minimum value (default: -0.1)
        max_val: Maximum value (default: 0.1)
    """

    def __init__(self, min_val=-0.1, max_val=0.1):
        super().__init__(**{"min": min_val, "max": max_val})

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize from uniform distribution: U(min, max)."""
        min_val = config.get("min", -0.1) if config else -0.1
        max_val = config.get("max", 0.1) if config else 0.1
        return jax.random.uniform(key, shape, minval=min_val, maxval=max_val)

    @staticmethod
    def element_variance(
        shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> float:
        """``(max - min)^2 / 12``, the variance of U(min, max)."""
        min_val = config.get("min", -0.1) if config else -0.1
        max_val = config.get("max", 0.1) if config else 0.1
        return float(max_val - min_val) ** 2 / 12.0


class XavierInitializer(InitializerBase):
    """
    Xavier/Glorot initialization for balanced fan-in/fan-out.

    Optimal for sigmoid and tanh activations. Maintains variance of
    activations across layers.

    For uniform: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
    For normal: N(0, std^2) where std = sqrt(2 / (fan_in + fan_out))

    Shape-aware over any rank: Linear ``(in, out)`` or an ND conv kernel
    ``(*spatial, C_in, C_out)``. fan_in = prod(shape[:-1]),
    fan_out = prod(shape[:-2]) * shape[-1]. A rank-1 shape uses fan_in = fan_out.

    Args:
        distribution: "normal" or "uniform" (default: "normal")
    """

    def __init__(self, distribution="normal", gain=1.0):
        super().__init__(distribution=distribution, gain=gain)

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize using Xavier/Glorot scheme."""
        config = config or {}
        distribution = config.get("distribution", "normal")
        gain = config.get("gain", 1.0)
        # PyTorch convention adapted to FabricPC's HWIO/LIO/DHWIO layout.
        fan_in, fan_out = _fans(shape)

        if distribution == "uniform":
            limit = gain * jnp.sqrt(6.0 / (fan_in + fan_out))
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        else:  # normal
            std = gain * jnp.sqrt(2.0 / (fan_in + fan_out))
            return std * jax.random.normal(key, shape)

    @staticmethod
    def element_variance(
        shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> float:
        """``2 * gain^2 / (fan_in + fan_out)``.

        Both distributions land on the same variance: the normal branch draws
        with that standard deviation directly, and the uniform branch's
        ``limit^2 / 3`` reduces to it.
        """
        config = config or {}
        gain = config.get("gain", 1.0)
        fan_in, fan_out = _fans(shape)
        return 2.0 * float(gain) ** 2 / (fan_in + fan_out)


class KaimingInitializer(InitializerBase):
    """
    Kaiming/He initialization optimized for ReLU networks.

    Maintains variance of activations specifically for ReLU and variants.

    For ReLU: gain = sqrt(2.0)
    For Leaky ReLU: gain = sqrt(2.0 / (1 + a^2))

    For uniform: U(-limit, limit) where limit = gain * sqrt(3 / fan)
    For normal: N(0, std^2) where std = gain / sqrt(fan)

    Shape-aware over any rank: Linear ``(in, out)`` or an ND conv kernel
    ``(*spatial, C_in, C_out)``. fan_in = prod(shape[:-1]),
    fan_out = prod(shape[:-2]) * shape[-1]. A rank-1 shape uses fan_in = fan_out.

    Args:
        mode: "fan_in" or "fan_out" (default: "fan_in")
        nonlinearity: "relu" or "leaky_relu" (default: "relu")
        distribution: "normal" or "uniform" (default: "normal")
        a: Negative slope for leaky_relu (default: 0.01)
    """

    def __init__(
        self,
        mode="fan_in",
        nonlinearity="relu",
        distribution="normal",
        a=0.01,
        gain=1.0,
    ):
        super().__init__(
            mode=mode,
            nonlinearity=nonlinearity,
            distribution=distribution,
            a=a,
            gain=gain,
        )

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize using Kaiming/He scheme."""
        config = config or {}
        mode = config.get("mode", "fan_in")
        nonlinearity = config.get("nonlinearity", "relu")
        distribution = config.get("distribution", "normal")
        gain_scaling = config.get("gain", 1.0)

        fan_in, fan_out = _fans(shape)
        fan = fan_out if mode == "fan_out" else fan_in

        if nonlinearity == "leaky_relu":
            a = config.get("a", 0.01)
            gain = jnp.sqrt(2.0 / (1 + a**2))
        else:  # relu
            gain = jnp.sqrt(2.0)

        if distribution == "uniform":
            limit = gain_scaling * gain * jnp.sqrt(3.0 / fan)
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        else:  # normal
            std = gain_scaling * gain / jnp.sqrt(fan)
            return std * jax.random.normal(key, shape)

    @staticmethod
    def element_variance(
        shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> float:
        """``(gain_scaling * gain)^2 / fan``, with gain set by the nonlinearity.

        Both distributions land on the same variance: the normal branch draws
        with that standard deviation directly, and the uniform branch's
        ``limit^2 / 3`` reduces to it.
        """
        config = config or {}
        mode = config.get("mode", "fan_in")
        nonlinearity = config.get("nonlinearity", "relu")
        gain_scaling = float(config.get("gain", 1.0))

        fan_in, fan_out = _fans(shape)
        fan = fan_out if mode == "fan_out" else fan_in

        if nonlinearity == "leaky_relu":
            a = float(config.get("a", 0.01))
            gain_sq = 2.0 / (1 + a**2)
        else:  # relu
            gain_sq = 2.0

        return gain_scaling**2 * gain_sq / fan


class MuPCInitializer(InitializerBase):
    """
    muPC weight initialization: W ~ N(0, gain^2).

    Weights are drawn from a standard normal distribution (unit variance)
    scaled by an optional gain factor. The actual width/depth scaling is
    NOT baked into the weights -- it is applied during the forward pass
    via per-edge scaling factors computed by the muPC module.

    This decoupling of initialization from forward-pass scaling is the
    key innovation of muPC (Yang et al., Innocenti et al.).

    Args:
        gain: Multiplicative factor applied to the standard normal samples
              (default: 1.0)
    """

    def __init__(self, gain=1.0):
        super().__init__(gain=gain)

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize from standard normal: gain * N(0, 1)."""
        config = config or {}
        gain = config.get("gain", 1.0)
        return gain * jax.random.normal(key, shape)

    @staticmethod
    def element_variance(
        shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> float:
        """``gain^2``: unit variance by construction, shape-independent. The
        width and depth scaling lives in the per-edge forward scale, not here."""
        config = config or {}
        return float(config.get("gain", 1.0)) ** 2


# =============================================================================
# Convenience Functions
# =============================================================================


def initialize(
    key: jax.Array, shape: Tuple[int, ...], initializer: InitializerBase
) -> jnp.ndarray:
    """
    Initialize array using the specified initializer.

    Args:
        key: JAX random key
        shape: Shape of array to create
        initializer: InitializerBase instance

    Returns:
        Initialized array

    Example:
        init = XavierInitializer(distribution="uniform")
        arr = initialize(key, (784, 256), init)
    """
    return type(initializer).initialize(key, shape, initializer.config)


def element_variance(shape: Tuple[int, ...], initializer: InitializerBase) -> float:
    """
    Per-element variance the initializer would draw at this shape.

    Args:
        shape: Shape the array would be initialized with
        initializer: InitializerBase instance

    Returns:
        Variance of a single element, in closed form

    Example:
        init = XavierInitializer()
        v = element_variance((256, 256), init)   # 1/256
    """
    return type(initializer).element_variance(shape, initializer.config)
