"""
Energy functionals for predictive coding networks.

This module provides:
- EnergyFunctional base class with abstract interface
- Built-in energy functionals (Gaussian, Bernoulli, Cross Entrpoy)
- Registry with decorator-based registration for custom energy functionals
- Entry point discovery for external packages

Energy functionals define how prediction errors are quantified into scalar energy
values, which drives both inference (latent state updates) and learning (weight updates).

User Extensibility
------------------
Users can register custom energy functionals in two ways:

1. **Decorator-based registration** (recommended for development):

    @register_energy("my_energy")
    class MyEnergyFunctional(EnergyFunctional):
        @staticmethod
        def energy(z_latent, z_mu, config=None):
            ...
        @staticmethod
        def grad_latent(z_latent, z_mu, config=None):
            ...

2. **Entry point discovery** (recommended for distribution):

    Add to pyproject.toml:
        [project.entry-points."fabricpc.energy"]
        my_energy = "my_package.energy:MyEnergyFunctional"

Configuration
-------------
Energy functionals can be configured per-node via node_config:

    {
        "name": "output",
        "shape": (10,),
        "type": "linear",
        "energy": {
            "type": "cross_entropy",  # Name of registered energy functional
            "temperature": 1.0      # Energy-specific parameters
        }
    }

If no energy config is specified, defaults to "gaussian".
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, List, Tuple

import jax.numpy as jnp

from fabricpc.core.registry import Registry, RegistrationError, validate_config_schema

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
        - grad_latent(): Compute ∂E/∂z_latent

    Required attributes:
        - CONFIG_SCHEMA: dict specifying configuration validation

    Example implementation:
        @register_energy("my_energy")
        class MyEnergy(EnergyFunctional):
            CONFIG_SCHEMA = {
                "temperature": {"type": float, "default": 1.0}
            }

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

    # CONFIG_SCHEMA is required - subclasses must define it
    # Use empty dict {} if no additional config parameters are needed
    CONFIG_SCHEMA: Dict[str, Dict[str, Any]]

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
        Compute gradient ∂E/∂z_latent of the node's self latent state.

        Args:
            z_latent: Latent states, shape (batch, *dims)
            z_mu: Predicted expectations, shape (batch, *dims)
            config: Optional configuration dict for energy parameters

        Returns:
            Gradient w.r.t. z_latent, same shape as z_latent

        Note:
            This is the signal used to update latent states during inference:
            z_latent_new = z_latent - eta * grad_latent(z_latent, z_mu)
        """
        pass


# =============================================================================
# Energy Registry
# =============================================================================


class EnergyRegistrationError(RegistrationError):
    """Raised when energy functional registration fails."""

    pass


# Create the energy registry instance
_energy_registry = Registry(
    name="energy",
    entry_point_group="fabricpc.energy",
    required_attrs=["CONFIG_SCHEMA"],
    required_methods=["energy", "grad_latent"],
    attr_validators={
        "CONFIG_SCHEMA": validate_config_schema,
    },
)
_energy_registry.set_error_class(EnergyRegistrationError)

# Expose the internal registry dict for backward compatibility
_ENERGY_REGISTRY = _energy_registry._registry


def register_energy(energy_type: str):
    """
    Decorator to register an energy functional with the registry.

    Usage:
        @register_energy("bernoulli")
        class BernoulliEnergy(EnergyFunctional):
            ...

    Args:
        energy_type: Unique identifier for this energy type (case-insensitive)

    Returns:
        Decorator function

    Raises:
        EnergyRegistrationError: If registration fails (duplicate, missing methods)
    """
    return _energy_registry.register(energy_type)


def get_energy_class(energy_type: str) -> Type[EnergyFunctional]:
    """
    Get an energy functional class by its registered type name.

    Args:
        energy_type: The registered energy type (case-insensitive)

    Returns:
        The energy functional class

    Raises:
        ValueError: If energy type is not registered
    """
    return _energy_registry.get(energy_type)


def list_energy_types() -> List[str]:
    """Return list of all registered energy types."""
    return _energy_registry.list_types()


def unregister_energy(energy_type: str) -> None:
    """
    Remove an energy type from the registry.
    Primarily for testing purposes.

    Args:
        energy_type: The energy type to unregister (case-insensitive)
    """
    _energy_registry.unregister(energy_type)


def clear_energy_registry() -> None:
    """Clear all registrations. For testing only."""
    _energy_registry.clear()


def validate_energy_config(
    energy_class: Type[EnergyFunctional], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and apply defaults from energy's CONFIG_SCHEMA.

    Args:
        energy_class: The energy class with CONFIG_SCHEMA
        config: The user-provided config dict

    Returns:
        Config dict with defaults applied

    Raises:
        ConfigValidationError: If required fields are missing or validation fails
    """
    from fabricpc.core.config import validate_config

    schema = getattr(energy_class, "CONFIG_SCHEMA", None)
    energy_type = config.get("type", "unknown") if config else "unknown"
    return validate_config(schema, config, context=f"energy '{energy_type}'")


def discover_external_energy() -> None:
    """
    Discover and register energy functionals from installed packages via entry points.

    Looks for packages with entry points in the "fabricpc.energy" group.
    Each entry point should map an energy type name to an EnergyFunctional subclass.

    Example pyproject.toml for an external package:
        [project.entry-points."fabricpc.energy"]
        poisson = "my_package.energy:PoissonEnergy"
    """
    _energy_registry.discover_external()


# =============================================================================
# Built-in Energy Functionals
# =============================================================================


@register_energy("gaussian")
class GaussianEnergy(EnergyFunctional):
    """
    Gaussian (quadratic) energy functional.

    E = (1/2σ²) * ||z - μ||²

    This is the standard MSE-based energy, equivalent to assuming Gaussian
    distributions for predictions with fixed variance.

    Config options:
        - precision: 1/σ² (default: 1.0). Higher values = sharper distributions.

    This is the DEFAULT energy functional if none is specified.
    """

    CONFIG_SCHEMA = {
        "precision": {
            "type": (int, float),
            "default": 1.0,
            "description": "Precision (1/variance) of the Gaussian. Higher = tighter fit.",
        }
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Gaussian energy: E = (precision/2) * ||z - μ||²

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
        Compute gradient: ∂E/∂z = precision * (z - μ)
        """
        precision = config.get("precision", 1.0) if config else 1.0
        return precision * (z_latent - z_mu)


@register_energy("bernoulli")
class BernoulliEnergy(EnergyFunctional):
    """
    Bernoulli (binary cross-entropy) energy functional.

    E = -Σ[z*log(μ) + (1-z)*log(1-μ)]

    Use for binary outputs where μ represents probabilities in [0, 1].
    The target z_latent should be clamped to binary values (0 or 1).

    Config options:
        - eps: Small constant for numerical stability (default: 1e-7)
    """

    CONFIG_SCHEMA = {
        "eps": {
            "type": (int, float),
            "default": 1e-7,
            "description": "Small constant for numerical stability in log",
        }
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Bernoulli (BCE) energy: E = -Σ[z*log(μ) + (1-z)*log(1-μ)]
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
        Compute gradient: ∂E/∂z = -log(μ) + log(1-μ) = log((1-μ)/μ)

        Note: In standard PC with clamped targets, this gradient is used
        to propagate errors backward. For binary targets, the gradient
        pushes z toward z_mu.
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1 - eps)

        # ∂BCE/∂z = -log(μ) + log(1-μ)
        return -jnp.log(z_mu_safe) + jnp.log(1 - z_mu_safe)


@register_energy("cross_entropy")
class CrossEntropyEnergy(EnergyFunctional):
    """
    Categorical (cross-entropy) energy functional.

    E = -Σ z_i * log(μ_i)

    Use for classification where:
    - z_latent is one-hot encoded targets
    - z_mu is softmax probabilities (should sum to 1 along last axis)

    Config options:
        - eps: Small constant for numerical stability (default: 1e-7)
        - axis: Axis along which probabilities sum to 1 (default: -1)
    """

    CONFIG_SCHEMA = {
        "eps": {
            "type": (int, float),
            "default": 1e-7,
            "description": "Small constant for numerical stability in log",
        },
        "axis": {
            "type": int,
            "default": -1,
            "description": "Axis along which probabilities sum to 1",
        },
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute cross_entropy (CE) energy: E = -Σ z_i * log(μ_i)
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
        Compute gradient: ∂E/∂z = -log(μ)

        For one-hot targets with clamped latents, this gradient is used
        to propagate classification errors backward through the network.
        """
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        return -jnp.log(z_mu_safe)


@register_energy("laplacian")
class LaplacianEnergy(EnergyFunctional):
    """
    Laplacian (L1) energy functional.

    E = (1/b) * Σ|z - μ|

    More robust to outliers than Gaussian. Corresponds to assuming
    Laplace distributions for predictions.

    Config options:
        - scale: b parameter (default: 1.0). Larger = more tolerance.
    """

    CONFIG_SCHEMA = {
        "scale": {
            "type": (int, float),
            "default": 1.0,
            "description": "Scale parameter b of Laplace distribution",
        }
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute Laplacian energy: E = (1/b) * Σ|z - μ|
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
        Compute gradient: ∂E/∂z = (1/b) * sign(z - μ)
        """
        scale = config.get("scale", 1.0) if config else 1.0
        return jnp.sign(z_latent - z_mu) / scale


@register_energy("huber")
class HuberEnergy(EnergyFunctional):
    """
    Huber energy functional (smooth L1).

    E = {  0.5 * (z - μ)²           if |z - μ| ≤ δ
        {  δ * (|z - μ| - 0.5*δ)    if |z - μ| > δ

    Combines advantages of L2 (smooth gradients) and L1 (robustness).

    Config options:
        - delta: Transition threshold (default: 1.0)
    """

    CONFIG_SCHEMA = {
        "delta": {
            "type": (int, float),
            "default": 1.0,
            "description": "Threshold for switching from quadratic to linear",
        }
    }

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
        Compute gradient: clipped to [-δ, δ]
        """
        delta = config.get("delta", 1.0) if config else 1.0
        diff = z_latent - z_mu

        return jnp.clip(diff, -delta, delta)


@register_energy("kl_divergence")
class KLDivergenceEnergy(EnergyFunctional):
    """
    KL Divergence energy functional.

    E = Σ z * log(z / μ) = Σ z * (log(z) - log(μ))

    Computes D_KL(z || μ), the Kullback-Leibler divergence from μ to z.
    Both z_latent and z_mu should be valid probability distributions
    (non-negative, summing to 1 along the specified axis).

    Use for:
    - Matching probability distributions
    - Variational inference objectives
    - Information-theoretic losses

    Config options:
        - eps: Small constant for numerical stability (default: 1e-7)
        - axis: Axis along which probabilities sum to 1 (default: -1)

    Note:
        KL divergence is asymmetric: D_KL(z || μ) ≠ D_KL(μ || z).
        This implementation computes D_KL(z_latent || z_mu), penalizing
        cases where z_latent has mass but z_mu does not.
    """

    CONFIG_SCHEMA = {
        "eps": {
            "type": (int, float),
            "default": 1e-7,
            "description": "Small constant for numerical stability in log",
        },
        "axis": {
            "type": int,
            "default": -1,
            "description": "Axis along which probabilities sum to 1",
        },
    }

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] = None
    ) -> jnp.ndarray:
        """
        Compute KL divergence energy: E = Σ z * log(z / μ)

        For numerical stability, uses: z * log(z) - z * log(μ)
        with clipping to avoid log(0).
        """
        eps = config.get("eps", 1e-7) if config else 1e-7

        # Clip for numerical stability
        z_latent_safe = jnp.clip(z_latent, eps, 1.0)
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        # KL divergence: z * log(z) - z * log(μ)
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
        Compute gradient: ∂E/∂z = log(z / μ) + 1 = log(z) - log(μ) + 1

        For KL(z || μ):
            ∂/∂z [z * log(z) - z * log(μ)] = log(z) + 1 - log(μ)
        """
        eps = config.get("eps", 1e-7) if config else 1e-7

        z_latent_safe = jnp.clip(z_latent, eps, 1.0)
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)

        # Gradient: log(z) - log(μ) + 1
        grad = jnp.log(z_latent_safe) - jnp.log(z_mu_safe) + 1.0

        # For z near 0, gradient should push toward matching μ
        # Use a smooth approximation
        grad = jnp.where(z_latent < eps, -jnp.log(z_mu_safe), grad)

        return grad


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_energy(
    z_latent: jnp.ndarray, z_mu: jnp.ndarray, energy_config: Dict[str, Any] = None
) -> jnp.ndarray:
    """
    Compute energy using the specified energy functional.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy_config: Energy configuration dict with "type" and other params.
                      If None, uses Gaussian energy with defaults.

    Returns:
        Energy per sample, shape (batch,)

    Example:
        energy = compute_energy(z, mu, {"type": "bernoulli", "eps": 1e-6})
    """
    if energy_config is None:
        energy_config = {"type": "gaussian"}

    energy_type = energy_config.get("type", "gaussian")
    energy_class = get_energy_class(energy_type)

    # Validate and apply defaults
    config = validate_energy_config(energy_class, energy_config)

    return energy_class.energy(z_latent, z_mu, config)


def compute_energy_gradient(
    z_latent: jnp.ndarray, z_mu: jnp.ndarray, energy_config: Dict[str, Any] = None
) -> jnp.ndarray:
    """
    Compute energy gradient w.r.t. z_latent.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy_config: Energy configuration dict with "type" and other params.
                      If None, uses Gaussian energy with defaults.

    Returns:
        Gradient ∂E/∂z_latent, same shape as z_latent

    Example:
        grad = compute_energy_gradient(z, mu, {"type": "cross_entropy"})
    """
    if energy_config is None:
        energy_config = {"type": "gaussian"}

    energy_type = energy_config.get("type", "gaussian")
    energy_class = get_energy_class(energy_type)

    # Validate and apply defaults
    config = validate_energy_config(energy_class, energy_config)

    return energy_class.grad_latent(z_latent, z_mu, config)


def get_energy_and_gradient(
    z_latent: jnp.ndarray, z_mu: jnp.ndarray, energy_config: Dict[str, Any] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute both energy and gradient efficiently.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy_config: Energy configuration dict

    Returns:
        Tuple of (energy, gradient):
            - energy: per-sample energy, shape (batch,)
            - gradient: ∂E/∂z_latent, same shape as z_latent
    """
    if energy_config is None:
        energy_config = {"type": "gaussian"}

    energy_type = energy_config.get("type", "gaussian")
    energy_class = get_energy_class(energy_type)
    config = validate_energy_config(energy_class, energy_config)

    energy = energy_class.energy(z_latent, z_mu, config)
    gradient = energy_class.grad_latent(z_latent, z_mu, config)

    return energy, gradient


# Auto-discover external energy functionals on import
discover_external_energy()
