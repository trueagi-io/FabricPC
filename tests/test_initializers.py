#!/usr/bin/env python3
"""
Test suite for the Initializer system.

Tests core initializer implementations, custom initializer creation,
and determinism.
"""

from math import prod

import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.initializers import (
    InitializerBase,
    initialize,
    ZerosInitializer,
    NormalInitializer,
    XavierInitializer,
    KaimingInitializer,
)

# Weight shapes spanning every rank the ND fan extension must handle. The fan
# formula is rank-agnostic, so one parametrized test covers them all:
#   fan_in  = prod(shape[:-1])            (e.g. kH*kW*C_in for a 2D conv kernel)
#   fan_out = prod(shape[:-2]) * shape[-1] (e.g. kH*kW*C_out)
FAN_SHAPES = [
    (256, 128),  # Linear / 2D matrix  -> fan_in=256, fan_out=128
    (5, 16, 32),  # 1D conv (kL, C_in, C_out)        -> fan_in=80,  fan_out=160
    (3, 3, 16, 32),  # 2D conv (kH, kW, C_in, C_out) -> fan_in=144, fan_out=288
    (3, 3, 3, 8, 16),  # 3D conv (kD,kH,kW,C_in,C_out)-> fan_in=216, fan_out=432
]


def _expected_fans(shape):
    """fan_in / fan_out exactly as the ND-aware initializers compute them."""
    if len(shape) >= 2:
        return prod(shape[:-1]), prod(shape[:-2]) * shape[-1]
    return shape[0], shape[0]


class TestBuiltinInitializers:
    """Test suite for built-in initializer implementations."""

    def test_zeros_initializer(self, rng_key):
        """Test zeros initializer returns all zeros."""
        shape = (32, 64)
        result = initialize(rng_key, shape, ZerosInitializer())

        assert result.shape == shape
        assert jnp.all(result == 0.0)

    def test_normal_initializer_default(self, rng_key):
        """Test normal initializer with default config."""
        shape = (1000, 100)
        result = initialize(rng_key, shape, NormalInitializer())

        assert result.shape == shape
        assert jnp.abs(jnp.mean(result)) < 0.01
        assert jnp.abs(jnp.std(result) - 0.05) < 0.01

    def test_xavier_initializer_normal(self, rng_key):
        """Test Xavier initializer with normal distribution."""
        shape = (256, 128)
        result = initialize(rng_key, shape, XavierInitializer(distribution="normal"))

        assert result.shape == shape
        expected_std = jnp.sqrt(2.0 / (256 + 128))
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01

    def test_kaiming_initializer_fan_in_relu(self, rng_key):
        """Test Kaiming initializer with fan_in mode and ReLU."""
        shape = (512, 256)
        result = initialize(
            rng_key,
            shape,
            KaimingInitializer(
                mode="fan_in", nonlinearity="relu", distribution="normal"
            ),
        )

        assert result.shape == shape
        expected_std = jnp.sqrt(2.0) / jnp.sqrt(512)
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01

    @pytest.mark.parametrize("shape", FAN_SHAPES)
    def test_kaiming_std_nd(self, rng_key, shape):
        """Kaiming/ReLU std ≈ √2 / √fan_in across Linear (2D) and 1D/2D/3D conv
        kernels. fan_in = prod(shape[:-1]); one body, every rank."""
        result = initialize(rng_key, shape, KaimingInitializer(nonlinearity="relu"))
        assert result.shape == shape
        fan_in, _ = _expected_fans(shape)
        expected_std = jnp.sqrt(2.0) / jnp.sqrt(fan_in)
        assert jnp.abs(jnp.std(result) - expected_std) < 0.02

    @pytest.mark.parametrize("shape", FAN_SHAPES)
    def test_xavier_std_nd(self, rng_key, shape):
        """Xavier std ≈ √(2/(fan_in + fan_out)) across Linear (2D) and 1D/2D/3D
        conv kernels. fan_out = prod(shape[:-2]) * shape[-1]."""
        result = initialize(rng_key, shape, XavierInitializer(distribution="normal"))
        assert result.shape == shape
        fan_in, fan_out = _expected_fans(shape)
        expected_std = jnp.sqrt(2.0 / (fan_in + fan_out))
        assert jnp.abs(jnp.std(result) - expected_std) < 0.02


class TestCustomInitializer:
    """Test creating custom initializers."""

    def test_custom_initializer_subclass(self, rng_key):
        """Test creating and using a custom initializer subclass."""

        class ConstantInitializer(InitializerBase):
            def __init__(self, value=1.0):
                super().__init__(value=value)

            @staticmethod
            def initialize(key, shape, config=None):
                value = config.get("value", 1.0) if config else 1.0
                return jnp.full(shape, value)

        init = ConstantInitializer(value=42.0)
        result = initialize(rng_key, (4, 4), init)
        assert jnp.all(result == 42.0)


class TestInitializerDeterminism:
    """Test that initializers are deterministic with same key."""

    def test_normal_deterministic(self, rng_key):
        """Test normal initializer is deterministic."""
        shape = (64, 64)
        init = NormalInitializer()

        result1 = initialize(rng_key, shape, init)
        result2 = initialize(rng_key, shape, init)

        assert jnp.allclose(result1, result2)
