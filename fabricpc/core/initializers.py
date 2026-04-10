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

import types
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp

# =============================================================================
# Initializer Base Class
# =============================================================================


class InitializerBase(ABC):
    """
    Abstract base class for tensor initializers.

    Initializers are context-agnostic: they don't know if they're initializing
    weights or latent states. The caller determines the context.

    All initialize() methods are static for JAX compatibility (pure functions, no state).

    Required methods:
        - initialize(): Generate initialized array
    """

    def __init__(self, **config):
        self.config = types.MappingProxyType(config)  # Immutable dictionary

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


class XavierInitializer(InitializerBase):
    """
    Xavier/Glorot initialization for balanced fan-in/fan-out.

    Optimal for sigmoid and tanh activations. Maintains variance of
    activations across layers.

    For uniform: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
    For normal: N(0, std^2) where std = sqrt(2 / (fan_in + fan_out))

    Assumes shape is (fan_in, fan_out) or (fan_in,).

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
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]

        if distribution == "uniform":
            limit = gain * jnp.sqrt(6.0 / (fan_in + fan_out))
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        else:  # normal
            std = gain * jnp.sqrt(2.0 / (fan_in + fan_out))
            return std * jax.random.normal(key, shape)


class KaimingInitializer(InitializerBase):
    """
    Kaiming/He initialization optimized for ReLU networks.

    Maintains variance of activations specifically for ReLU and variants.

    For ReLU: gain = sqrt(2.0)
    For Leaky ReLU: gain = sqrt(2.0 / (1 + a^2))

    For uniform: U(-limit, limit) where limit = gain * sqrt(3 / fan)
    For normal: N(0, std^2) where std = gain / sqrt(fan)

    Assumes shape is (fan_in, fan_out) or (fan_in,).

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

        if mode == "fan_out":
            fan = shape[1] if len(shape) > 1 else shape[0]
        else:  # fan_in
            fan = shape[0]

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
