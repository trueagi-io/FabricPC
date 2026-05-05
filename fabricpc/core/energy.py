"""
Energy functionals for predictive coding networks.

This module provides:
- EnergyFunctional base class with constructor-based configuration
- Built-in energy functionals (Gaussian, Bernoulli, CrossEntropy, Laplacian, Huber, KLDivergence)

Energy functionals define how prediction errors are quantified into scalar energy
values, which drives both inference (latent state updates) and learning (weight updates).

User Extensibility
------------------
Users can create custom energy functionals by extending EnergyFunctional:

    class MyEnergy(EnergyFunctional):
        def __init__(self, temperature=1.0):
            super().__init__(temperature=temperature)

        @staticmethod
        def energy(z_latent, z_mu, config=None):
            temp = config.get("temperature", 1.0) if config else 1.0
            diff = z_latent - z_mu
            return 0.5 * jnp.sum(diff ** 2, axis=-1) / temp

        @staticmethod
        def grad_latent(z_latent, z_mu, config=None):
            temp = config.get("temperature", 1.0) if config else 1.0
            return (z_latent - z_mu) / temp

Usage
-----
Energy functionals are instantiated with their parameters:

    energy = GaussianEnergy(precision=2.0)
    energy = CrossEntropyEnergy(eps=1e-7)
"""

import types
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import jax.numpy as jnp

# =============================================================================
# Energy Functional Base Class
# =============================================================================


class EnergyFunctional(ABC):
    """
    Abstract base class for energy functionals.

    Energy functionals define how prediction errors are converted to scalar
    energy values. The energy drives inference (minimizing E w.r.t. z_latent)
    and parameter learning (minimizing E w.r.t. params).

    All methods are static for JAX compatibility (pure functions, no state).

    Required methods:
        - energy(): Compute E(z_latent, z_mu) per sample
        - grad_latent(): Compute dE/dz_latent

    Example implementation:
        class MyEnergy(EnergyFunctional):
            def __init__(self, temperature=1.0):
                super().__init__(temperature=temperature)

            @staticmethod
            def energy(z_latent, z_mu, config=None):
                temp = config.get("temperature", 1.0) if config else 1.0
                diff = z_latent - z_mu
                return 0.5 * jnp.sum(diff ** 2, axis=-1) / temp

            @staticmethod
            def grad_latent(z_latent, z_mu, config=None):
                temp = config.get("temperature", 1.0) if config else 1.0
                return (z_latent - z_mu) / temp
    """

    def __init__(self, **config):
        self.config = types.MappingProxyType(config)  # Immutable dictionary

    @staticmethod
    @abstractmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute energy E(z_latent, z_mu).

        Args:
            z_latent: Latent states, shape (batch, *dims)
            z_mu: Predicted expectations, shape (batch, *dims)
            config: Optional configuration dict for energy parameters

        Returns:
            Energy per sample, shape (batch,)

        Note:
            Should sum over all non-batch dimensions to produce per-sample energy.
        """
        pass

    @staticmethod
    @abstractmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient dE/dz_latent (element-wise / diagonal form).

        This method is NOT called by the default framework path — autodiff
        computes the self-latent gradient automatically via
        ``jax.value_and_grad`` in ``forward_and_latent_grads()``.

        It is provided as a convenience for node subclasses that override
        ``forward_and_latent_grads()`` with explicit (non-autodiff) gradient
        computation (see ``LinearExplicitGrad`` for the pattern).

        Args:
            z_latent: Latent states, shape (batch, *dims)
            z_mu: Predicted expectations, shape (batch, *dims)
            config: Optional configuration dict for energy parameters

        Returns:
            Gradient w.r.t. z_latent, same shape as z_latent
        """
        pass


# =============================================================================
# Built-in Energy Functionals
# =============================================================================


class GaussianEnergy(EnergyFunctional):
    """
    Gaussian (quadratic) energy functional.

    E = (1/2sigma^2) * ||z - mu||^2

    This is the standard MSE-based energy, equivalent to assuming Gaussian
    distributions for predictions with fixed variance.

    This is the DEFAULT energy functional if none is specified.

    Args:
        precision: 1/sigma^2 (default: 1.0). Higher values = sharper distributions.
    """

    def __init__(self, precision=1.0):
        super().__init__(precision=precision)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Gaussian energy: E = (precision/2) * ||z - mu||^2

        Sums over all non-batch dimensions.
        """
        precision = config.get("precision", 1.0) if config else 1.0
        diff = z_latent - z_mu
        # Sum over all non-batch dimensions
        axes_to_sum = tuple(range(1, len(diff.shape)))
        return 0.5 * precision * jnp.sum(diff**2, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: dE/dz = precision * (z - mu)
        """
        precision = config.get("precision", 1.0) if config else 1.0
        return precision * (z_latent - z_mu)


class BernoulliEnergy(EnergyFunctional):
    """
    Bernoulli (binary cross-entropy) energy functional.

    E = -sum[z*log(mu) + (1-z)*log(1-mu)]

    Use for binary outputs where mu represents probabilities in [0, 1].
    The target z_latent should be clamped to binary values (0 or 1).

    Args:
        eps: Small constant for numerical stability (default: 1e-7)
    """

    def __init__(self, eps=1e-7):
        super().__init__(eps=eps)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Bernoulli (BCE) energy: E = -sum[z*log(mu) + (1-z)*log(1-mu)]
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1 - eps)

        bce = -(z_latent * jnp.log(z_mu_safe) + (1 - z_latent) * jnp.log(1 - z_mu_safe))

        # Sum over all non-batch dimensions
        axes_to_sum = tuple(range(1, len(bce.shape)))
        return jnp.sum(bce, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: dE/dz = -log(mu) + log(1-mu) = log((1-mu)/mu)

        Note: In standard PC with clamped targets, this gradient is used
        to propagate errors backward. For binary targets, the gradient
        pushes z toward z_mu.
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1 - eps)

        # dBCE/dz = -log(mu) + log(1-mu)
        return -jnp.log(z_mu_safe) + jnp.log(1 - z_mu_safe)


class CrossEntropyEnergy(EnergyFunctional):
    """
    Categorical (cross-entropy) energy functional.

    E = -sum z_i * log(mu_i)

    Use for classification where:
    - z_latent is one-hot encoded targets
    - z_mu is softmax probabilities (should sum to 1 along last axis)

    Args:
        eps: Small constant for numerical stability (default: 1e-7)
        axis: Axis along which probabilities sum to 1 (default: -1)
    """

    def __init__(self, eps=1e-7, axis=-1):
        super().__init__(eps=eps, axis=axis)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute cross_entropy (CE) energy: E = -sum z_i * log(mu_i)
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        ce = -z_latent * jnp.log(z_mu_safe)

        # Sum over all non-batch dimensions
        axes_to_sum = tuple(range(1, len(ce.shape)))
        return jnp.sum(ce, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: dE/dz = -log(mu)

        For one-hot targets with clamped latents, this gradient is used
        to propagate classification errors backward through the network.
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        return -jnp.log(z_mu_safe)


class LaplacianEnergy(EnergyFunctional):
    """
    Laplacian (L1) energy functional.

    E = (1/b) * sum|z - mu|

    More robust to outliers than Gaussian. Corresponds to assuming
    Laplace distributions for predictions.

    Args:
        scale: b parameter (default: 1.0). Larger = more tolerance.
    """

    def __init__(self, scale=1.0):
        super().__init__(scale=scale)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Laplacian energy: E = (1/b) * sum|z - mu|
        """
        scale = config.get("scale", 1.0) if config else 1.0
        diff = jnp.abs(z_latent - z_mu)

        axes_to_sum = tuple(range(1, len(diff.shape)))
        return jnp.sum(diff, axis=axes_to_sum) / scale

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: dE/dz = (1/b) * sign(z - mu)
        """
        scale = config.get("scale", 1.0) if config else 1.0
        return jnp.sign(z_latent - z_mu) / scale


class HuberEnergy(EnergyFunctional):
    """
    Huber energy functional (smooth L1).

    E = {  0.5 * (z - mu)^2           if |z - mu| <= delta
        {  delta * (|z - mu| - 0.5*delta)    if |z - mu| > delta

    Combines advantages of L2 (smooth gradients) and L1 (robustness).

    Args:
        delta: Transition threshold (default: 1.0)
    """

    def __init__(self, delta=1.0):
        super().__init__(delta=delta)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Huber energy.
        """
        delta = config.get("delta", 1.0) if config else 1.0
        diff = z_latent - z_mu
        abs_diff = jnp.abs(diff)

        # Quadratic region
        quadratic = 0.5 * diff**2
        # Linear region
        linear = delta * (abs_diff - 0.5 * delta)

        huber = jnp.where(abs_diff <= delta, quadratic, linear)

        axes_to_sum = tuple(range(1, len(huber.shape)))
        return jnp.sum(huber, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: clipped to [-delta, delta]
        """
        delta = config.get("delta", 1.0) if config else 1.0
        diff = z_latent - z_mu

        return jnp.clip(diff, -delta, delta)


class KLDivergenceEnergy(EnergyFunctional):
    """
    KL Divergence energy functional.

    E = sum z * log(z / mu) = sum z * (log(z) - log(mu))

    Computes D_KL(z || mu), the Kullback-Leibler divergence from mu to z.
    Both z_latent and z_mu should be valid probability distributions
    (non-negative, summing to 1 along the specified axis).

    Use for:
    - Matching probability distributions
    - Variational inference objectives
    - Information-theoretic losses

    Note:
        KL divergence is asymmetric: D_KL(z || mu) != D_KL(mu || z).
        This implementation computes D_KL(z_latent || z_mu), penalizing
        cases where z_latent has mass but z_mu does not.

    Args:
        eps: Small constant for numerical stability (default: 1e-7)
        axis: Axis along which probabilities sum to 1 (default: -1)
    """

    def __init__(self, eps=1e-7, axis=-1):
        super().__init__(eps=eps, axis=axis)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute KL divergence energy: E = sum z * log(z / mu)

        For numerical stability, uses: z * log(z) - z * log(mu)
        with clipping to avoid log(0).
        """
        eps = config.get("eps", 1e-7) if config else 1e-7

        # Clip for numerical stability
        z_latent_safe = jnp.clip(z_latent, eps, 1.0)
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        # KL divergence: z * log(z) - z * log(mu)
        # Note: z * log(z) term handles the case where z -> 0 (gives 0, not -inf)
        kl = z_latent_safe * (jnp.log(z_latent_safe) - jnp.log(z_mu_safe))

        # Handle z = 0 case: 0 * log(0) should be 0, not nan
        kl = jnp.where(z_latent < eps, 0.0, kl)

        # Sum over all non-batch dimensions
        axes_to_sum = tuple(range(1, len(kl.shape)))
        return jnp.sum(kl, axis=axes_to_sum)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute gradient: dE/dz = log(z / mu) + 1 = log(z) - log(mu) + 1

        For KL(z || mu):
            d/dz [z * log(z) - z * log(mu)] = log(z) + 1 - log(mu)
        """
        eps = config.get("eps", 1e-7) if config else 1e-7

        z_latent_safe = jnp.clip(z_latent, eps, 1.0)
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        # Gradient: log(z) - log(mu) + 1
        grad = jnp.log(z_latent_safe) - jnp.log(z_mu_safe) + 1.0

        # For z near 0, gradient should push toward matching mu
        # Use a smooth approximation
        grad = jnp.where(z_latent < eps, -jnp.log(z_mu_safe), grad)

        return grad


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_energy(
    z_latent: jnp.ndarray, z_mu: jnp.ndarray, energy: EnergyFunctional = None
) -> jnp.ndarray:
    """
    Compute energy using the specified energy functional.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy: EnergyFunctional instance. If None, uses GaussianEnergy with defaults.

    Returns:
        Energy per sample, shape (batch,)
    """
    if energy is None:
        energy = GaussianEnergy()

    return type(energy).energy(z_latent, z_mu, energy.config)


def compute_energy_gradient(
    z_latent: jnp.ndarray, z_mu: jnp.ndarray, energy: EnergyFunctional = None
) -> jnp.ndarray:
    """
    Compute energy gradient w.r.t. z_latent.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy: EnergyFunctional instance. If None, uses GaussianEnergy with defaults.

    Returns:
        Gradient dE/dz_latent, same shape as z_latent
    """
    if energy is None:
        energy = GaussianEnergy()

    return type(energy).grad_latent(z_latent, z_mu, energy.config)


def get_energy_and_gradient(
    z_latent: jnp.ndarray, z_mu: jnp.ndarray, energy: EnergyFunctional = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute both energy and gradient efficiently.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy: EnergyFunctional instance. If None, uses GaussianEnergy with defaults.

    Returns:
        Tuple of (energy, gradient):
            - energy: per-sample energy, shape (batch,)
            - gradient: dE/dz_latent, same shape as z_latent
    """
    if energy is None:
        energy = GaussianEnergy()

    energy_cls = type(energy)
    config = energy.config

    e = energy_cls.energy(z_latent, z_mu, config)
    g = energy_cls.grad_latent(z_latent, z_mu, config)

    return e, g
