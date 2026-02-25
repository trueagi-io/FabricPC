"""
Activation functions for predictive coding networks in JAX.

This module provides:
- ActivationBase abstract class with constructor-based configuration
- Built-in activations (Identity, Sigmoid, Tanh, ReLU, LeakyReLU, GELU, Softmax, HardTanh)

All functions are pure and compatible with JAX transformations (jit, vmap, grad).

User Extensibility
------------------
Users can create custom activations by extending ActivationBase:

    class MyActivation(ActivationBase):
        def __init__(self, temperature=1.0):
            super().__init__(temperature=temperature)

        @staticmethod
        def forward(x, config=None):
            temp = config.get("temperature", 1.0) if config else 1.0
            return jnp.tanh(x / temp)

        @staticmethod
        def derivative(x, config=None):
            temp = config.get("temperature", 1.0) if config else 1.0
            t = jnp.tanh(x / temp)
            return (1 - t**2) / temp

Usage
-----
Activations are instantiated with their parameters:

    act = SigmoidActivation()
    act = LeakyReLUActivation(alpha=0.02)
    z_mu = type(act).forward(x, act.config)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import jax.numpy as jnp
from jax import nn

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

    Example implementation:
        class MyActivation(ActivationBase):
            def __init__(self, temperature=1.0):
                super().__init__(temperature=temperature)

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

    def __init__(self, **config):
        self.config = config

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
# Built-in Activations
# =============================================================================


class IdentityActivation(ActivationBase):
    """Identity activation: f(x) = x"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return x

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return jnp.ones_like(x)


class SigmoidActivation(ActivationBase):
    """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return nn.sigmoid(x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        s = nn.sigmoid(x)
        return s * (1 - s)


class TanhActivation(ActivationBase):
    """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return jnp.tanh(x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        t = jnp.tanh(x)
        return 1 - t**2


class ReLUActivation(ActivationBase):
    """ReLU activation: max(0, x)"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return nn.relu(x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        return (x > 0).astype(jnp.float32)


class LeakyReLUActivation(ActivationBase):
    """
    Leaky ReLU activation: max(alpha * x, x)

    Args:
        alpha: Negative slope (default: 0.01)
    """

    def __init__(self, alpha=0.01):
        super().__init__(alpha=alpha)

    @staticmethod
    def forward(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        alpha = config.get("alpha", 0.01) if config else 0.01
        return jnp.where(x > 0, x, alpha * x)

    @staticmethod
    def derivative(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        alpha = config.get("alpha", 0.01) if config else 0.01
        return jnp.where(x > 0, 1.0, alpha)


class GeluActivation(ActivationBase):
    """GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))"""

    def __init__(self):
        super().__init__()

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


class SoftmaxActivation(ActivationBase):
    """Softmax activation: exp(x) / sum(exp(x)) along the last axis"""

    def __init__(self):
        super().__init__()

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


class HardTanhActivation(ActivationBase):
    """
    Hard tanh activation: clip(x, min_val, max_val)

    Args:
        min_val: Minimum output value (default: -1.0)
        max_val: Maximum output value (default: 1.0)
    """

    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__(min_val=min_val, max_val=max_val)

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
