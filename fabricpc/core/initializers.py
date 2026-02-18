"""
Tensor initializers for predictive coding networks.

This module provides:
- InitializerBase abstract class with stateless interface
- Built-in initializers (Zeros, Normal, Uniform, Xavier, Kaiming)
- Registry with decorator-based registration for custom initializers
- Entry point discovery for external packages

Initializers are context-agnostic: they don't know if they're initializing
weights or latent states. The caller determines the context.

User Extensibility
------------------
Users can register custom initializers in two ways:

1. **Decorator-based registration** (recommended for development):

    @register_initializer("my_init")
    class MyInitializer(InitializerBase):
        CONFIG_SCHEMA = {"scale": {"type": float, "default": 1.0}}

        @staticmethod
        def initialize(key, shape, config=None):
            scale = config.get("scale", 1.0) if config else 1.0
            return scale * jax.random.normal(key, shape)

2. **Entry point discovery** (recommended for distribution):

    Add to pyproject.toml:
        [project.entry-points."fabricpc.initializers"]
        my_init = "my_package.initializers:MyInitializer"

Configuration
-------------
Initializers are configured via a dict with "type" and other params:

    {"type": "normal", "mean": 0.0, "std": 0.05}
    {"type": "xavier", "distribution": "uniform"}
    {"type": "kaiming", "mode": "fan_out", "nonlinearity": "relu"}
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, List, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.registry import Registry, RegistrationError, validate_config_schema

# =============================================================================
# Initializer Base Class
# =============================================================================


class InitializerBase(ABC):
    """
    Abstract base class for tensor initializers.

    Initializers are context-agnostic: they don't know if they're initializing
    weights or latent states. The caller determines the context.

    All methods are static for JAX compatibility (pure functions, no state).

    Required methods:
        - initialize(): Generate initialized array

    Required attributes:
        - CONFIG_SCHEMA: dict specifying configuration validation

    Example implementation:
        @register_initializer("my_init")
        class MyInitializer(InitializerBase):
            CONFIG_SCHEMA = {
                "scale": {"type": float, "default": 1.0}
            }

            @staticmethod
            def initialize(key, shape, config=None):
                scale = config.get("scale", 1.0) if config else 1.0
                return scale * jax.random.normal(key, shape)
    """

    # CONFIG_SCHEMA is required - subclasses must define it
    # Use empty dict {} if no additional config parameters are needed
    CONFIG_SCHEMA: Dict[str, Dict[str, Any]]

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
# Initializer Registry
# =============================================================================


class InitializerRegistrationError(RegistrationError):
    """Raised when initializer registration fails."""

    pass


# Create the initializer registry instance
_initializer_registry = Registry(
    name="initializer",
    entry_point_group="fabricpc.initializers",
    required_attrs=["CONFIG_SCHEMA"],
    required_methods=["initialize"],
    attr_validators={
        "CONFIG_SCHEMA": validate_config_schema,
    },
)
_initializer_registry.set_error_class(InitializerRegistrationError)


def register_initializer(init_type: str):
    """
    Decorator to register an initializer with the registry.

    Usage:
        @register_initializer("normal")
        class NormalInitializer(InitializerBase):
            ...

    Args:
        init_type: Unique identifier for this initializer type (case-insensitive)

    Returns:
        Decorator function

    Raises:
        InitializerRegistrationError: If registration fails (duplicate, missing methods)
    """
    return _initializer_registry.register(init_type)


def get_initializer_class(init_type: str) -> Type[InitializerBase]:
    """
    Get an initializer class by its registered type name.

    Args:
        init_type: The registered initializer type (case-insensitive)

    Returns:
        The initializer class

    Raises:
        ValueError: If initializer type is not registered
    """
    return _initializer_registry.get(init_type)


def list_initializer_types() -> List[str]:
    """Return list of all registered initializer types."""
    return _initializer_registry.list_types()


def unregister_initializer(init_type: str) -> None:
    """
    Remove an initializer type from the registry.
    Primarily for testing purposes.

    Args:
        init_type: The initializer type to unregister (case-insensitive)
    """
    _initializer_registry.unregister(init_type)


def clear_initializer_registry() -> None:
    """Clear all registrations. For testing only."""
    _initializer_registry.clear()


def validate_initializer_config(
    initializer_class: Type[InitializerBase], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and apply defaults from initializer's CONFIG_SCHEMA.

    Args:
        initializer_class: The initializer class with CONFIG_SCHEMA
        config: The user-provided config dict

    Returns:
        Config dict with defaults applied

    Raises:
        ConfigValidationError: If required fields are missing or validation fails
    """
    from fabricpc.core.config import validate_config

    schema = getattr(initializer_class, "CONFIG_SCHEMA", None)
    init_type = config.get("type", "unknown") if config else "unknown"
    return validate_config(schema, config, context=f"initializer '{init_type}'")


def discover_external_initializers() -> None:
    """
    Discover and register initializers from installed packages via entry points.

    Looks for packages with entry points in the "fabricpc.initializers" group.
    Each entry point should map an initializer type name to an InitializerBase subclass.

    Example pyproject.toml for an external package:
        [project.entry-points."fabricpc.initializers"]
        orthogonal = "my_package.initializers:OrthogonalInitializer"
    """
    _initializer_registry.discover_external()


# =============================================================================
# Built-in Initializers
# =============================================================================


@register_initializer("zeros")
class ZerosInitializer(InitializerBase):
    """
    Initialize with zeros.

    Useful for biases or initial states where zero is a sensible default.
    """

    CONFIG_SCHEMA = {}

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Return array of zeros."""
        return jnp.zeros(shape)


@register_initializer("ones")
class OnesInitializer(InitializerBase):
    """
    Initialize with ones.

    Useful for scaling factors or multiplicative parameters.
    """

    CONFIG_SCHEMA = {}

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Return array of ones."""
        return jnp.ones(shape)


@register_initializer("normal")
class NormalInitializer(InitializerBase):
    """
    Normal (Gaussian) distribution initialization.

    Values are drawn from N(mean, std^2).

    Config options:
        - mean: Mean of the distribution (default: 0.0)
        - std: Standard deviation (default: 0.05)
    """

    CONFIG_SCHEMA = {
        "mean": {
            "type": (int, float),
            "default": 0.0,
            "description": "Mean of the normal distribution",
        },
        "std": {
            "type": (int, float),
            "default": 0.05,
            "description": "Standard deviation of the normal distribution",
        },
    }

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize from normal distribution: mean + std * N(0, 1)."""
        mean = config.get("mean", 0.0) if config else 0.0
        std = config.get("std", 0.05) if config else 0.05
        return mean + std * jax.random.normal(key, shape)


@register_initializer("uniform")
class UniformInitializer(InitializerBase):
    """
    Uniform distribution initialization.

    Values are drawn from U(min, max).

    Config options:
        - min: Minimum value (default: -0.1)
        - max: Maximum value (default: 0.1)
    """

    CONFIG_SCHEMA = {
        "min": {
            "type": (int, float),
            "default": -0.1,
            "description": "Minimum value of the uniform distribution",
        },
        "max": {
            "type": (int, float),
            "default": 0.1,
            "description": "Maximum value of the uniform distribution",
        },
    }

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize from uniform distribution: U(min, max)."""
        min_val = config.get("min", -0.1) if config else -0.1
        max_val = config.get("max", 0.1) if config else 0.1
        return jax.random.uniform(key, shape, minval=min_val, maxval=max_val)


@register_initializer("xavier")
class XavierInitializer(InitializerBase):
    """
    Xavier/Glorot initialization for balanced fan-in/fan-out.

    Optimal for sigmoid and tanh activations. Maintains variance of
    activations across layers.

    For uniform: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
    For normal: N(0, std^2) where std = sqrt(2 / (fan_in + fan_out))

    Assumes shape is (fan_in, fan_out) or (fan_in,).

    Config options:
        - distribution: "normal" or "uniform" (default: "normal")
    """

    CONFIG_SCHEMA = {
        "distribution": {
            "type": str,
            "default": "normal",
            "choices": ["normal", "uniform"],
            "description": "Distribution type for Xavier initialization",
        },
    }

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize using Xavier/Glorot scheme."""
        distribution = config.get("distribution", "normal") if config else "normal"
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]

        if distribution == "uniform":
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        else:  # normal
            std = jnp.sqrt(2.0 / (fan_in + fan_out))
            return std * jax.random.normal(key, shape)


@register_initializer("kaiming")
class KaimingInitializer(InitializerBase):
    """
    Kaiming/He initialization optimized for ReLU networks.

    Maintains variance of activations specifically for ReLU and variants.

    For ReLU: gain = sqrt(2.0)
    For Leaky ReLU: gain = sqrt(2.0 / (1 + a^2))

    For uniform: U(-limit, limit) where limit = gain * sqrt(3 / fan)
    For normal: N(0, std^2) where std = gain / sqrt(fan)

    Assumes shape is (fan_in, fan_out) or (fan_in,).

    Config options:
        - mode: "fan_in" or "fan_out" (default: "fan_in")
        - nonlinearity: "relu" or "leaky_relu" (default: "relu")
        - distribution: "normal" or "uniform" (default: "normal")
        - a: Negative slope for leaky_relu (default: 0.01)
    """

    CONFIG_SCHEMA = {
        "mode": {
            "type": str,
            "default": "fan_in",
            "choices": ["fan_in", "fan_out"],
            "description": "Which dimension to use for variance scaling",
        },
        "nonlinearity": {
            "type": str,
            "default": "relu",
            "choices": ["relu", "leaky_relu"],
            "description": "Nonlinearity type for gain calculation",
        },
        "distribution": {
            "type": str,
            "default": "normal",
            "choices": ["normal", "uniform"],
            "description": "Distribution type for Kaiming initialization",
        },
        "a": {
            "type": (int, float),
            "default": 0.01,
            "description": "Negative slope for leaky_relu",
        },
    }

    @staticmethod
    def initialize(
        key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """Initialize using Kaiming/He scheme."""
        config = config or {}
        mode = config.get("mode", "fan_in")
        nonlinearity = config.get("nonlinearity", "relu")
        distribution = config.get("distribution", "normal")

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
            limit = gain * jnp.sqrt(3.0 / fan)
            return jax.random.uniform(key, shape, minval=-limit, maxval=limit)
        else:  # normal
            std = gain / jnp.sqrt(fan)
            return std * jax.random.normal(key, shape)


# =============================================================================
# Convenience Functions
# =============================================================================


def initialize(
    key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None
) -> jnp.ndarray:
    """
    Initialize array using the specified initializer.

    Args:
        key: JAX random key
        shape: Shape of array to create
        config: Initializer configuration dict specifying an object implementing InitializerBase.

    Returns:
        Initialized array

    Example:
        arr = initialize(key, (784, 256), {"type": "xavier", "distribution": "uniform"})
    """
    if config is None:
        raise ValueError("Initializer config must be provided.")

    init_type = config["type"]
    init_class = get_initializer_class(init_type)

    validated_config = validate_initializer_config(init_class, config)
    return init_class.initialize(key, shape, validated_config)


# Auto-discover external initializers on import
discover_external_initializers()
