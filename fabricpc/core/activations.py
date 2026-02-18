"""
Activation functions for predictive coding networks in JAX.

This module provides:
- ActivationBase abstract class with schema validation
- Built-in activations (identity, sigmoid, tanh, relu, leaky_relu, hard_tanh)
- Registry with decorator-based registration for custom activations
- Entry point discovery for external packages

All functions are pure and compatible with JAX transformations (jit, vmap, grad).
"""

from abc import ABC, abstractmethod
from typing import Callable, Tuple, Dict, Any, Type, List
import jax.numpy as jnp
from jax import nn

from fabricpc.core.registry import Registry, RegistrationError, validate_config_schema

# =============================================================================
# Activation Base Class
# =============================================================================


class ActivationBase(ABC):
    """
    Abstract base class for activation functions.

    Activation functions define how pre-activation values are transformed.
    Each activation provides both the forward function and its derivative.

    All methods are static for JAX compatibility (pure functions, no state).

    Required methods:
        - forward(): Apply activation function
        - derivative(): Compute derivative w.r.t. pre-activation

    Required attributes:
        - CONFIG_SCHEMA: dict specifying configuration validation

    Example implementation:
        @register_activation("my_activation")
        class MyActivation(ActivationBase):
            CONFIG_SCHEMA = {
                "temperature": {"type": float, "default": 1.0}
            }

            @staticmethod
            def forward(x, config=None):
                temp = config.get("temperature", 1.0) if config else 1.0
                return jnp.tanh(x / temp)

            @staticmethod
            def derivative(x, config=None):
                temp = config.get("temperature", 1.0) if config else 1.0
                t = jnp.tanh(x / temp)
                return (1 - t**2) / temp
    """

    # CONFIG_SCHEMA is required - subclasses must define it
    # Use empty dict {} if no additional config parameters are needed
    CONFIG_SCHEMA: Dict[str, Dict[str, Any]]

    @staticmethod
    @abstractmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        """
        Apply activation function.

        Args:
            x: Pre-activation values, any shape
            config: Optional configuration dict for activation parameters

        Returns:
            Activated values, same shape as x
        """
        pass

    @staticmethod
    @abstractmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        """
        Compute derivative w.r.t. pre-activation.

        Args:
            x: Pre-activation values, any shape
            config: Optional configuration dict for activation parameters

        Returns:
            Derivative values, same shape as x

        Note:
            This is the derivative f'(x) evaluated at x, where f is the activation.
            Used in predictive coding for gain modulation.
        """
        pass


# =============================================================================
# Activation Registry
# =============================================================================


class ActivationRegistrationError(RegistrationError):
    """Raised when activation registration fails."""

    pass


# Create the activation registry instance
_activation_registry = Registry(
    name="activation",
    entry_point_group="fabricpc.activations",
    required_attrs=["CONFIG_SCHEMA"],
    required_methods=["forward", "derivative"],
    attr_validators={
        "CONFIG_SCHEMA": validate_config_schema,
    },
)
_activation_registry.set_error_class(ActivationRegistrationError)


def register_activation(activation_type: str):
    """
    Decorator to register an activation function with the registry.

    Usage:
        @register_activation("softplus")
        class SoftplusActivation(ActivationBase):
            ...

    Args:
        activation_type: Unique identifier for this activation type (case-insensitive)

    Returns:
        Decorator function

    Raises:
        ActivationRegistrationError: If registration fails
    """
    return _activation_registry.register(activation_type)


def get_activation_class(activation_type: str) -> Type[ActivationBase]:
    """
    Get an activation class by its registered type name.

    Args:
        activation_type: The registered activation type (case-insensitive)

    Returns:
        The activation class

    Raises:
        ValueError: If activation type is not registered
    """
    return _activation_registry.get(activation_type)


def list_activation_types() -> List[str]:
    """Return list of all registered activation types."""
    return _activation_registry.list_types()


def unregister_activation(activation_type: str) -> None:
    """
    Remove an activation type from the registry.
    Primarily for testing purposes.

    Args:
        activation_type: The activation type to unregister (case-insensitive)
    """
    _activation_registry.unregister(activation_type)


def clear_activation_registry() -> None:
    """Clear all registrations. For testing only."""
    _activation_registry.clear()


def validate_activation_config(
    activation_class: Type[ActivationBase], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and apply defaults from activation's CONFIG_SCHEMA.

    Args:
        activation_class: The activation class with CONFIG_SCHEMA
        config: The user-provided config dict

    Returns:
        Config dict with defaults applied

    Raises:
        ConfigValidationError: If required fields are missing or validation fails
    """
    from fabricpc.core.config import validate_config

    schema = getattr(activation_class, "CONFIG_SCHEMA", None)
    activation_type = config.get("type", "unknown") if config else "unknown"
    return validate_config(schema, config, context=f"activation '{activation_type}'")


def discover_external_activations() -> None:
    """
    Discover and register activations from installed packages via entry points.

    Looks for packages with entry points in the "fabricpc.activations" group.
    Each entry point should map an activation type name to an ActivationBase subclass.

    Example pyproject.toml for an external package:
        [project.entry-points."fabricpc.activations"]
        swish = "my_package.activations:SwishActivation"
    """
    _activation_registry.discover_external()


# =============================================================================
# Built-in Activations
# =============================================================================


@register_activation("identity")
class IdentityActivation(ActivationBase):
    """Identity activation: f(x) = x"""

    CONFIG_SCHEMA = {}

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return x

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return jnp.ones_like(x)


@register_activation("sigmoid")
class SigmoidActivation(ActivationBase):
    """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))"""

    CONFIG_SCHEMA = {}

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return nn.sigmoid(x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        s = nn.sigmoid(x)
        return s * (1 - s)


@register_activation("tanh")
class TanhActivation(ActivationBase):
    """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""

    CONFIG_SCHEMA = {}

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return jnp.tanh(x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        t = jnp.tanh(x)
        return 1 - t**2


@register_activation("relu")
class ReLUActivation(ActivationBase):
    """ReLU activation: max(0, x)"""

    CONFIG_SCHEMA = {}

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return nn.relu(x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return (x > 0).astype(jnp.float32)


@register_activation("leaky_relu")
class LeakyReLUActivation(ActivationBase):
    """
    Leaky ReLU activation: max(alpha * x, x)

    Config options:
        - alpha: Negative slope (default: 0.01)
    """

    CONFIG_SCHEMA = {
        "alpha": {
            "type": (int, float),
            "default": 0.01,
            "description": "Negative slope for x < 0",
        }
    }

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        alpha = config.get("alpha", 0.01) if config else 0.01
        return jnp.where(x > 0, x, alpha * x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        alpha = config.get("alpha", 0.01) if config else 0.01
        return jnp.where(x > 0, 1.0, alpha)


@register_activation("gelu")
class GeluActivation(ActivationBase):
    """GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))"""

    CONFIG_SCHEMA = {}

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return nn.gelu(x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        # Gelu(x): = x * normal_CDF
        sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
        x_cubed = x**3
        apx_norm_cdf = 0.5 * (
            1 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * x_cubed))
        )  # Approximation of normal CDF via tanh
        norm_cdf_prime = (0.5 * sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)) * (
            1 - jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * x_cubed)) ** 2
        )
        return apx_norm_cdf + x * norm_cdf_prime


@register_activation("softmax")
class SoftmaxActivation(ActivationBase):
    """Softmax activation: exp(x) / sum(exp(x)) along the last axis"""

    CONFIG_SCHEMA = {}

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        exp_x = jnp.exp(
            x - jnp.max(x, axis=-1, keepdims=True)
        )  # relative to max value for numerical stability
        return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        s = SoftmaxActivation.forward(x)
        return s * (
            1 - s
        )  # Note: This is a simplification; full Jacobian is more complex


@register_activation("hard_tanh")
class HardTanhActivation(ActivationBase):
    """
    Hard tanh activation: clip(x, min_val, max_val)

    Config options:
        - min_val: Minimum output value (default: -1.0)
        - max_val: Maximum output value (default: 1.0)
    """

    CONFIG_SCHEMA = {
        "min_val": {
            "type": (int, float),
            "default": -1.0,
            "description": "Minimum output value",
        },
        "max_val": {
            "type": (int, float),
            "default": 1.0,
            "description": "Maximum output value",
        },
    }

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        min_val = config.get("min_val", -1.0) if config else -1.0
        max_val = config.get("max_val", 1.0) if config else 1.0
        return jnp.clip(x, min_val, max_val)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        min_val = config.get("min_val", -1.0) if config else -1.0
        max_val = config.get("max_val", 1.0) if config else 1.0
        return ((x > min_val) & (x < max_val)).astype(jnp.float32)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_activation(config: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Get activation function and its derivative from config.

    Args:
        config: Dictionary with 'type' key and optional parameters.
                Supported types: 'identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'hard_tanh'
                - leaky_relu: requires 'alpha' parameter (default 0.01)
                - hard_tanh: requires 'min_val' and 'max_val' parameters (default -1 and 1)

    Returns:
        Tuple of (activation_fn, derivative_fn)

    Example:
        >>> config = {"type": "sigmoid"}
        >>> act_fn, deriv_fn = get_activation(config)
        >>> x = jnp.array([0.0, 1.0, -1.0])
        >>> act_fn(x)
        DeviceArray([0.5, 0.731, 0.269], dtype=float32)
    """
    from fabricpc.core.config import transform_shorthand

    # Handle shorthand
    if isinstance(config, str):
        config = {"type": config}

    if "type" not in config:
        raise ValueError("config['type'] is required")

    act_type = config["type"].lower()
    act_class = get_activation_class(act_type)

    # Validate and apply defaults
    validated_config = validate_activation_config(act_class, config)

    # Return closures that capture the validated config
    def forward_fn(x):
        return act_class.forward(x, validated_config)

    def derivative_fn(x):
        return act_class.derivative(x, validated_config)

    return (forward_fn, derivative_fn)


def get_activation_fn(config: Dict[str, Any]) -> Callable:
    """Get just the activation function (without derivative)."""
    return get_activation(config)[0]


def get_activation_deriv(config: Dict[str, Any]) -> Callable:
    """Get just the activation derivative function."""
    return get_activation(config)[1]


# Auto-discover external activations on import
discover_external_activations()
