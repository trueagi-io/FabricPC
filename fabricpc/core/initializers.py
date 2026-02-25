"""Tensor initializers for predictive coding networks.

Direct-construction API only: instantiate initializer classes and pass objects.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp


class InitializerBase(ABC):
    """Base class for initializers."""

    @abstractmethod
    def initialize(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        pass


class ZerosInitializer(InitializerBase):
    def initialize(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        return jnp.zeros(shape)


class OnesInitializer(InitializerBase):
    def initialize(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        return jnp.ones(shape)


class NormalInitializer(InitializerBase):
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = float(mean)
        self.std = float(std)

    def initialize(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        return self.mean + self.std * jax.random.normal(key, shape)


class UniformInitializer(InitializerBase):
    def __init__(self, min_val: float = -0.1, max_val: float = 0.1):
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def initialize(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        return jax.random.uniform(key, shape, minval=self.min_val, maxval=self.max_val)


class XavierInitializer(InitializerBase):
    def __init__(self, distribution: str = "normal"):
        if distribution not in {"normal", "uniform"}:
            raise ValueError("distribution must be 'normal' or 'uniform'")
        self.distribution = distribution

    def initialize(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        if self.distribution == "uniform":
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        std = jnp.sqrt(2.0 / (fan_in + fan_out))
        return std * jax.random.normal(key, shape)


class KaimingInitializer(InitializerBase):
    def __init__(
        self,
        mode: str = "fan_in",
        nonlinearity: str = "relu",
        distribution: str = "normal",
        a: float = 0.01,
    ):
        if mode not in {"fan_in", "fan_out"}:
            raise ValueError("mode must be 'fan_in' or 'fan_out'")
        if nonlinearity not in {"relu", "leaky_relu"}:
            raise ValueError("nonlinearity must be 'relu' or 'leaky_relu'")
        if distribution not in {"normal", "uniform"}:
            raise ValueError("distribution must be 'normal' or 'uniform'")

        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution
        self.a = float(a)

    def initialize(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        fan = shape[1] if (self.mode == "fan_out" and len(shape) > 1) else shape[0]
        if self.nonlinearity == "leaky_relu":
            gain = jnp.sqrt(2.0 / (1 + self.a**2))
        else:
            gain = jnp.sqrt(2.0)

        if self.distribution == "uniform":
            limit = gain * jnp.sqrt(3.0 / fan)
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

        std = gain / jnp.sqrt(fan)
        return std * jax.random.normal(key, shape)


def ensure_initializer(initializer: InitializerBase | None) -> InitializerBase:
    """Normalize optional initializer input to an InitializerBase instance."""
    return initializer if initializer is not None else NormalInitializer()


def initialize(
    key: jax.Array,
    shape: Tuple[int, ...],
    initializer: InitializerBase,
) -> jnp.ndarray:
    """Initialize an array with a concrete initializer object."""
    if not isinstance(initializer, InitializerBase):
        raise TypeError(
            "initializer must be an InitializerBase instance; string/type lookup is removed"
        )
    return initializer.initialize(key, shape)
