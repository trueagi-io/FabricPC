#!/usr/bin/env python3
"""
Test suite for the Initializer system.

Tests:
- Built-in initializers (Zeros, Ones, Normal, Uniform, Xavier, Kaiming)
- Custom initializer creation
- Determinism with same random key
- Various tensor shapes
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.initializers import (
    InitializerBase,
    initialize,
    ZerosInitializer,
    OnesInitializer,
    NormalInitializer,
    UniformInitializer,
    XavierInitializer,
    KaimingInitializer,
)

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


class TestBuiltinInitializers:
    """Test suite for built-in initializer implementations."""

    def test_zeros_initializer(self, rng_key):
        """Test zeros initializer returns all zeros."""
        shape = (32, 64)
        result = initialize(rng_key, shape, ZerosInitializer())

        assert result.shape == shape
        assert jnp.all(result == 0.0)

    def test_ones_initializer(self, rng_key):
        """Test ones initializer returns all ones."""
        shape = (16, 32)
        result = initialize(rng_key, shape, OnesInitializer())

        assert result.shape == shape
        assert jnp.all(result == 1.0)

    def test_normal_initializer_default(self, rng_key):
        """Test normal initializer with default config."""
        shape = (1000, 100)
        result = initialize(rng_key, shape, NormalInitializer())

        assert result.shape == shape
        # Default mean=0.0, std=0.05
        assert jnp.abs(jnp.mean(result)) < 0.01
        assert jnp.abs(jnp.std(result) - 0.05) < 0.01

    def test_normal_initializer_custom(self, rng_key):
        """Test normal initializer with custom mean and std."""
        shape = (1000, 100)
        result = initialize(rng_key, shape, NormalInitializer(mean=5.0, std=2.0))

        assert result.shape == shape
        assert jnp.abs(jnp.mean(result) - 5.0) < 0.1
        assert jnp.abs(jnp.std(result) - 2.0) < 0.1

    def test_uniform_initializer_default(self, rng_key):
        """Test uniform initializer with default config."""
        shape = (1000, 100)
        result = initialize(rng_key, shape, UniformInitializer())

        assert result.shape == shape
        # Default min=-0.1, max=0.1
        assert jnp.all(result >= -0.1)
        assert jnp.all(result <= 0.1)

    def test_uniform_initializer_custom(self, rng_key):
        """Test uniform initializer with custom min and max."""
        shape = (1000, 100)
        result = initialize(
            rng_key, shape, UniformInitializer(min_val=-1.0, max_val=1.0)
        )

        assert result.shape == shape
        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)
        assert jnp.abs(jnp.mean(result)) < 0.1

    def test_xavier_initializer_normal(self, rng_key):
        """Test Xavier initializer with normal distribution."""
        shape = (256, 128)
        result = initialize(rng_key, shape, XavierInitializer(distribution="normal"))

        assert result.shape == shape
        # Xavier std = sqrt(2 / (fan_in + fan_out))
        expected_std = jnp.sqrt(2.0 / (256 + 128))
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01

    def test_xavier_initializer_uniform(self, rng_key):
        """Test Xavier initializer with uniform distribution."""
        shape = (256, 128)
        result = initialize(rng_key, shape, XavierInitializer(distribution="uniform"))

        assert result.shape == shape
        # Xavier limit = sqrt(6 / (fan_in + fan_out))
        expected_limit = jnp.sqrt(6.0 / (256 + 128))
        assert jnp.all(result >= -expected_limit - 0.01)
        assert jnp.all(result <= expected_limit + 0.01)

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
        # Kaiming std = sqrt(2) / sqrt(fan_in)
        expected_std = jnp.sqrt(2.0) / jnp.sqrt(512)
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01

    def test_kaiming_initializer_fan_out(self, rng_key):
        """Test Kaiming initializer with fan_out mode."""
        shape = (512, 256)
        result = initialize(
            rng_key,
            shape,
            KaimingInitializer(
                mode="fan_out", nonlinearity="relu", distribution="normal"
            ),
        )

        assert result.shape == shape
        # Kaiming std = sqrt(2) / sqrt(fan_out)
        expected_std = jnp.sqrt(2.0) / jnp.sqrt(256)
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01

    def test_kaiming_initializer_leaky_relu(self, rng_key):
        """Test Kaiming initializer with leaky ReLU."""
        shape = (512, 256)
        a = 0.2
        result = initialize(
            rng_key,
            shape,
            KaimingInitializer(
                mode="fan_in",
                nonlinearity="leaky_relu",
                distribution="normal",
                a=a,
            ),
        )

        assert result.shape == shape
        # Kaiming std = sqrt(2 / (1 + a^2)) / sqrt(fan_in)
        gain = jnp.sqrt(2.0 / (1 + a**2))
        expected_std = gain / jnp.sqrt(512)
        assert jnp.abs(jnp.std(result) - expected_std) < 0.01


class TestInitializerConfig:
    """Test that initializer instances store config correctly."""

    def test_normal_config(self):
        """Test NormalInitializer stores config."""
        init = NormalInitializer(mean=1.0, std=0.5)
        assert init.config["mean"] == 1.0
        assert init.config["std"] == 0.5

    def test_uniform_config(self):
        """Test UniformInitializer stores config."""
        init = UniformInitializer(min_val=-2.0, max_val=2.0)
        assert init.config["min"] == -2.0
        assert init.config["max"] == 2.0

    def test_xavier_config(self):
        """Test XavierInitializer stores config."""
        init = XavierInitializer(distribution="normal")
        assert init.config["distribution"] == "normal"

    def test_kaiming_config(self):
        """Test KaimingInitializer stores config."""
        init = KaimingInitializer(mode="fan_out", nonlinearity="relu")
        assert init.config["mode"] == "fan_out"
        assert init.config["nonlinearity"] == "relu"


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

        # Use the custom initializer
        init = ConstantInitializer(value=42.0)
        result = initialize(rng_key, (4, 4), init)
        assert jnp.all(result == 42.0)

        # Different value
        init2 = ConstantInitializer(value=-1.0)
        result2 = initialize(rng_key, (3, 5), init2)
        assert jnp.all(result2 == -1.0)


class TestInitializerDeterminism:
    """Test that initializers are deterministic with same key."""

    def test_normal_deterministic(self, rng_key):
        """Test normal initializer is deterministic."""
        shape = (64, 64)
        init = NormalInitializer()

        result1 = initialize(rng_key, shape, init)
        result2 = initialize(rng_key, shape, init)

        assert jnp.allclose(result1, result2)

    def test_different_keys_different_results(self, rng_key):
        """Test different keys produce different results."""
        shape = (64, 64)
        init = NormalInitializer()

        key1, key2 = jax.random.split(rng_key)
        result1 = initialize(key1, shape, init)
        result2 = initialize(key2, shape, init)

        assert not jnp.allclose(result1, result2)

    def test_xavier_deterministic(self, rng_key):
        """Test Xavier initializer is deterministic."""
        shape = (128, 64)
        init = XavierInitializer()

        result1 = initialize(rng_key, shape, init)
        result2 = initialize(rng_key, shape, init)

        assert jnp.allclose(result1, result2)


class TestInitializerShapes:
    """Test initializers work with various shapes."""

    def test_1d_shape(self, rng_key):
        """Test initializer with 1D shape."""
        result = initialize(rng_key, (128,), NormalInitializer())
        assert result.shape == (128,)

    def test_2d_shape(self, rng_key):
        """Test initializer with 2D shape."""
        result = initialize(rng_key, (64, 128), XavierInitializer())
        assert result.shape == (64, 128)

    def test_3d_shape(self, rng_key):
        """Test initializer with 3D shape."""
        result = initialize(rng_key, (32, 28, 28), UniformInitializer())
        assert result.shape == (32, 28, 28)

    def test_4d_shape(self, rng_key):
        """Test initializer with 4D shape (conv kernel)."""
        result = initialize(rng_key, (3, 3, 32, 64), KaimingInitializer())
        assert result.shape == (3, 3, 32, 64)
