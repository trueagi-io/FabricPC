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

import math
import types
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
        self.config = types.MappingProxyType(config)  # Immutable dictionary

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
            This method is NOT called by the default framework path — autodiff
            computes gradient automatically via
            ``jax.value_and_grad`` in ``forward_inference()``.

            It is provided as a convenience for node subclasses that override
            ``forward_inference()`` with explicit (non-autodiff) gradient
            computation (see ``LinearExplicitGrad`` for the pattern).
        """
        pass

    @staticmethod
    def variance_gain(config: Dict[str, Any] = None) -> float:
        """
        Return the Kaiming-style gain for variance preservation.

        This is the factor g such that when pre-activations have
        Var(z) = g^2, the post-activation output has Var(f(z)) ≈ 1.
        Used by muPC to compensate for activation-induced variance
        contraction in the forward scaling formula:

            a = gain / sqrt(fan_in * K)

        Subclasses should override with activation-specific values.
        Default returns 1.0 (no correction, appropriate for identity).
        """
        return 1.0

    @staticmethod
    def jacobian(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        """
        Full Jacobian matrix J_{ij} = df_i/dx_j, shape (..., D, D).

        For element-wise activations, the Jacobian is diag(derivative(x)) —
        use derivative() directly as the diagonal. Override this method for
        non-element-wise activations (e.g., softmax) where off-diagonal
        terms are needed.

        This is a convenience for explicit (non-autodiff) gradient
        implementations that need the full Jacobian.

        Args:
            x: Pre-activation values, shape (..., D)
            config: Optional configuration dict

        Returns:
            Jacobian matrix, shape (..., D, D)
        """
        raise NotImplementedError(
            "jacobian() not implemented for this activation. "
            "For element-wise activations, use derivative() as the diagonal."
        )

    @staticmethod
    def jacobian_gain(config: Dict[str, Any] = None) -> float:
        """
        Return the Jacobian compensation factor for muPC topdown gradient scaling.

        In local PC inference, topdown gradients propagate one hop per step
        through the Jacobian diag(act'(z)) @ (a*W). The per-hop gradient
        attenuation is approximately variance_gain * rms(act'(z)), where
        z ~ N(0, variance_gain^2). This factor compensates:

            jacobian_gain = 1 / (variance_gain * rms(act'(z)))

        so that topdown_grad_scale = a * jacobian_gain yields ~1.0 per-hop
        gradient propagation factor.

        Default returns 1.0 (exact for identity, ReLU, LeakyReLU where
        variance_gain * rms(act') = 1.0 by construction).
        """
        return 1.0


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

    @staticmethod
    def variance_gain(config: Dict[str, Any] = None) -> float:
        return math.sqrt(5.0 / 3.0)

    @staticmethod
    def jacobian_gain(config: Dict[str, Any] = None) -> float:
        # rms(tanh'(z)) ≈ 0.6144 for z ~ N(0, 5/3)
        # jacobian_gain = 1 / (sqrt(5/3) * 0.6144) ≈ 1.261
        return 1.261


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

    @staticmethod
    def variance_gain(config: Dict[str, Any] = None) -> float:
        return math.sqrt(2.0)


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

    @staticmethod
    def variance_gain(config: Dict[str, Any] = None) -> float:
        alpha = config.get("alpha", 0.01) if config else 0.01
        return math.sqrt(2.0 / (1.0 + alpha**2))


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

    @staticmethod
    def variance_gain(config: Dict[str, Any] = None) -> float:
        return math.sqrt(2.0)

    @staticmethod
    def jacobian_gain(config: Dict[str, Any] = None) -> float:
        # rms(gelu'(z)) ≈ 0.605 for z ~ N(0, 2)
        # jacobian_gain = 1 / (sqrt(2) * 0.605) ≈ 1.168
        return 1.168


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
        # Diagonal of the Jacobian diag(s) - s @ s.T.
        # The off-diagonal terms are omitted; valid for element-wise PC gradients.
        return s * (1 - s)

    @staticmethod
    def jacobian(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
        """Full softmax Jacobian: J_{ij} = s_i * (delta_{ij} - s_j)."""
        s = SoftmaxActivation.forward(x, config)
        eye = jnp.eye(s.shape[-1])
        return jnp.expand_dims(s, -1) * (eye - jnp.expand_dims(s, -2))


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

    @staticmethod
    def variance_gain(config: Dict[str, Any] = None) -> float:
        return math.sqrt(5.0 / 3.0)

    @staticmethod
    def jacobian_gain(config: Dict[str, Any] = None) -> float:
        # rms(hardtanh'(z)) ≈ 0.749 for z ~ N(0, 5/3)
        # jacobian_gain = 1 / (sqrt(5/3) * 0.749) ≈ 1.035
        return 1.035
