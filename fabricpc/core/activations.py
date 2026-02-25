"""Activation functions for predictive coding networks in JAX.

Direct-construction API only: instantiate activation classes and pass objects.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import nn


class ActivationBase(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)


class IdentityActivation(ActivationBase):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(x)


class SigmoidActivation(ActivationBase):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.sigmoid(x)

    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        s = nn.sigmoid(x)
        return s * (1 - s)


class TanhActivation(ActivationBase):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(x)

    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        t = jnp.tanh(x)
        return 1 - t**2


class ReLUActivation(ActivationBase):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.relu(x)

    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x > 0).astype(jnp.float32)


class LeakyReLUActivation(ActivationBase):
    def __init__(self, alpha: float = 0.01):
        self.alpha = float(alpha)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x > 0, x, self.alpha * x)

    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x > 0, 1.0, self.alpha)


class GeluActivation(ActivationBase):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.gelu(x)

    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
        x_cubed = x**3
        apx_norm_cdf = 0.5 * (1 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * x_cubed)))
        norm_cdf_prime = (0.5 * sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)) * (
            1 - jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * x_cubed)) ** 2
        )
        return apx_norm_cdf + x * norm_cdf_prime


class SoftmaxActivation(ActivationBase):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
        return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class HardTanhActivation(ActivationBase):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(x, self.min_val, self.max_val)

    def derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        return ((x > self.min_val) & (x < self.max_val)).astype(jnp.float32)


def ensure_activation(activation: ActivationBase | None) -> ActivationBase:
    """Normalize optional activation input to an ActivationBase instance."""
    return activation if activation is not None else IdentityActivation()


def get_activation(activation: ActivationBase):
    """Backward-compatible helper returning (forward_fn, derivative_fn)."""
    if not isinstance(activation, ActivationBase):
        raise TypeError(
            "activation must be an ActivationBase instance; string/type lookup is removed"
        )
    return activation.forward, activation.derivative
