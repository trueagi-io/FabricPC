#!/usr/bin/env python3
"""
Test suite for the Initializer system.

Tests core initializer implementations, custom initializer creation,
and determinism.
"""

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
