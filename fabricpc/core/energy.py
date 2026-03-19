"""
Energy functionals for predictive coding networks.

This module provides:
- EnergyFunctional base class with constructor-based configuration
- Built-in energy functionals (Gaussian, Bernoulli, CrossEntropy, Laplacian, Huber, KLDivergence, NavierStokes)

Energy functionals define how prediction errors are quantified into scalar energy
values, which drives both inference (latent state updates) and learning (weight updates).

User Extensibility
------------------
Users can create custom energy functionals by extending EnergyFunctional:

    class MyEnergy(EnergyFunctional):
        def __init__(self, temperature=1.0):
            super().__init__(temperature=temperature)

        @staticmethod
        def energy(z_latent, z_mu, config=None, context=None):
            temp = config.get("temperature", 1.0) if config else 1.0
            diff = z_latent - z_mu
            return 0.5 * jnp.sum(diff ** 2, axis=-1) / temp

        @staticmethod
        def grad_latent(z_latent, z_mu, config=None, context=None):
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

import jax
import jax.numpy as jnp


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
    """

    def __init__(self, **config):
        self.config = types.MappingProxyType(config)

    @staticmethod
    @abstractmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        """
        Compute energy E(z_latent, z_mu).

        Args:
            z_latent: Latent states, shape (batch, *dims)
            z_mu: Predicted expectations, shape (batch, *dims)
            config: Optional configuration dict for energy parameters
            context: Optional runtime metadata (for example node_info)

        Returns:
            Energy per sample, shape (batch,)
        """
        pass

    @staticmethod
    @abstractmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        """
        Compute gradient dE/dz_latent of the node's self latent state.

        Args:
            z_latent: Latent states, shape (batch, *dims)
            z_mu: Predicted expectations, shape (batch, *dims)
            config: Optional configuration dict for energy parameters
            context: Optional runtime metadata (for example node_info)

        Returns:
            Gradient w.r.t. z_latent, same shape as z_latent
        """
        pass


def _sum_non_batch(x: jnp.ndarray) -> jnp.ndarray:
    axes_to_sum = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes_to_sum)


class GaussianEnergy(EnergyFunctional):
    """Gaussian (quadratic) energy functional."""

    def __init__(self, precision=1.0):
        super().__init__(precision=precision)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        precision = config.get("precision", 1.0) if config else 1.0
        diff = z_latent - z_mu
        return 0.5 * precision * _sum_non_batch(diff**2)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        precision = config.get("precision", 1.0) if config else 1.0
        return precision * (z_latent - z_mu)


class BernoulliEnergy(EnergyFunctional):
    """Bernoulli (binary cross-entropy) energy functional."""

    def __init__(self, eps=1e-7):
        super().__init__(eps=eps)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1 - eps)
        bce = -(z_latent * jnp.log(z_mu_safe) + (1 - z_latent) * jnp.log(1 - z_mu_safe))
        return _sum_non_batch(bce)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1 - eps)
        return -jnp.log(z_mu_safe) + jnp.log(1 - z_mu_safe)


class CrossEntropyEnergy(EnergyFunctional):
    """Categorical cross-entropy energy functional."""

    def __init__(self, eps=1e-7, axis=-1):
        super().__init__(eps=eps, axis=axis)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)
        ce = -z_latent * jnp.log(z_mu_safe)
        return _sum_non_batch(ce)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)
        return -jnp.log(z_mu_safe)


class LaplacianEnergy(EnergyFunctional):
    """Laplacian (L1) energy functional."""

    def __init__(self, scale=1.0):
        super().__init__(scale=scale)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        scale = config.get("scale", 1.0) if config else 1.0
        diff = jnp.abs(z_latent - z_mu)
        return _sum_non_batch(diff) / scale

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        scale = config.get("scale", 1.0) if config else 1.0
        return jnp.sign(z_latent - z_mu) / scale


class HuberEnergy(EnergyFunctional):
    """Huber energy functional."""

    def __init__(self, delta=1.0):
        super().__init__(delta=delta)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        delta = config.get("delta", 1.0) if config else 1.0
        diff = z_latent - z_mu
        abs_diff = jnp.abs(diff)
        quadratic = 0.5 * diff**2
        linear = delta * (abs_diff - 0.5 * delta)
        huber = jnp.where(abs_diff <= delta, quadratic, linear)
        return _sum_non_batch(huber)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        delta = config.get("delta", 1.0) if config else 1.0
        diff = z_latent - z_mu
        return jnp.clip(diff, -delta, delta)


class KLDivergenceEnergy(EnergyFunctional):
    """KL divergence energy functional."""

    def __init__(self, eps=1e-7, axis=-1):
        super().__init__(eps=eps, axis=axis)

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_latent_safe = jnp.clip(z_latent, eps, 1.0)
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)
        kl = z_latent_safe * (jnp.log(z_latent_safe) - jnp.log(z_mu_safe))
        kl = jnp.where(z_latent < eps, 0.0, kl)
        return _sum_non_batch(kl)

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        eps = config.get("eps", 1e-7) if config else 1e-7
        z_latent_safe = jnp.clip(z_latent, eps, 1.0)
        z_mu_safe = jnp.clip(z_mu, eps, 1.0)
        grad = jnp.log(z_latent_safe) - jnp.log(z_mu_safe) + 1.0
        grad = jnp.where(z_latent < eps, -jnp.log(z_mu_safe), grad)
        return grad


def _channel_config(config: Dict[str, Any]) -> Dict[str, int]:
    channel_map = dict(config.get("channel_map", {"u": 0, "v": 1, "p": 2}))
    required = {"u", "v", "p"}
    missing = required.difference(channel_map)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"NavierStokesEnergy channel_map must define channels for {missing_str}"
        )
    return channel_map


def _validate_navier_stokes_field(
    x: jnp.ndarray,
    config: Dict[str, Any],
    context: Dict[str, Any] = None,
) -> Dict[str, int]:
    if x.ndim != 4:
        raise ValueError(
            f"NavierStokesEnergy requires rank-4 tensors (batch, H, W, C); got {x.shape}"
        )

    _, height, width, channels = x.shape
    if height < 3 or width < 3:
        raise ValueError(
            f"NavierStokesEnergy requires H >= 3 and W >= 3; got {(height, width)}"
        )

    channel_map = _channel_config(config)
    max_channel = max(channel_map.values())
    if channels <= max_channel:
        raise ValueError(
            "NavierStokesEnergy requires channels for u, v, and p; "
            f"got C={channels} with channel_map={channel_map}"
        )

    return channel_map


def _periodic_central_diff_x(field: jnp.ndarray, dx: float) -> jnp.ndarray:
    return (jnp.roll(field, -1, axis=2) - jnp.roll(field, 1, axis=2)) / (2.0 * dx)


def _periodic_central_diff_y(field: jnp.ndarray, dy: float) -> jnp.ndarray:
    return (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2.0 * dy)


def _periodic_laplacian(field: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    d2dx2 = (jnp.roll(field, -1, axis=2) - 2.0 * field + jnp.roll(field, 1, axis=2)) / (
        dx**2
    )
    d2dy2 = (jnp.roll(field, -1, axis=1) - 2.0 * field + jnp.roll(field, 1, axis=1)) / (
        dy**2
    )
    return d2dx2 + d2dy2


def _navier_stokes_residual_energy(
    x: jnp.ndarray,
    config: Dict[str, Any],
    context: Dict[str, Any] = None,
) -> jnp.ndarray:
    channel_map = _validate_navier_stokes_field(x, config, context)
    viscosity = config["viscosity"]
    dx = config.get("dx", 1.0)
    dy = config.get("dy", 1.0)
    momentum_weight = config.get("momentum_weight", 1.0)
    divergence_weight = config.get("divergence_weight", 1.0)

    u = x[..., channel_map["u"]]
    v = x[..., channel_map["v"]]
    p = x[..., channel_map["p"]]

    dudx = _periodic_central_diff_x(u, dx)
    dudy = _periodic_central_diff_y(u, dy)
    dvdx = _periodic_central_diff_x(v, dx)
    dvdy = _periodic_central_diff_y(v, dy)
    dpdx = _periodic_central_diff_x(p, dx)
    dpdy = _periodic_central_diff_y(p, dy)

    lap_u = _periodic_laplacian(u, dx, dy)
    lap_v = _periodic_laplacian(v, dx, dy)

    residual_u = u * dudx + v * dudy + dpdx - viscosity * lap_u
    residual_v = u * dvdx + v * dvdy + dpdy - viscosity * lap_v
    divergence = dudx + dvdy

    momentum_energy = (
        0.5 * momentum_weight * _sum_non_batch(residual_u**2 + residual_v**2)
    )
    divergence_energy = 0.5 * divergence_weight * _sum_non_batch(divergence**2)
    return momentum_energy + divergence_energy


class NavierStokesEnergy(EnergyFunctional):
    """
    2D incompressible steady-state Navier-Stokes energy for NHWC tensors.

    Expects tensors shaped (batch, H, W, C) with channels mapped to u, v, and p.
    Uses periodic finite differences and combines:
    - data alignment between z_latent and z_mu
    - momentum residual on z_latent
    - momentum residual on z_mu
    - divergence penalties on z_latent and z_mu
    """

    def __init__(
        self,
        viscosity: float,
        dx: float = 1.0,
        dy: float = 1.0,
        data_weight: float = 1.0,
        latent_ns_weight: float = 1.0,
        prediction_ns_weight: float = 1.0,
        momentum_weight: float = 1.0,
        divergence_weight: float = 1.0,
        channel_map: Dict[str, int] = None,
    ):
        super().__init__(
            viscosity=viscosity,
            dx=dx,
            dy=dy,
            data_weight=data_weight,
            latent_ns_weight=latent_ns_weight,
            prediction_ns_weight=prediction_ns_weight,
            momentum_weight=momentum_weight,
            divergence_weight=divergence_weight,
            channel_map=channel_map or {"u": 0, "v": 1, "p": 2},
        )

    @staticmethod
    def energy(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        if config is None:
            raise ValueError("NavierStokesEnergy requires a config")

        _validate_navier_stokes_field(z_latent, config, context)
        _validate_navier_stokes_field(z_mu, config, context)

        data_weight = config.get("data_weight", 1.0)
        latent_ns_weight = config.get("latent_ns_weight", 1.0)
        prediction_ns_weight = config.get("prediction_ns_weight", 1.0)

        diff = z_latent - z_mu
        data_energy = 0.5 * data_weight * _sum_non_batch(diff**2)
        latent_energy = latent_ns_weight * _navier_stokes_residual_energy(
            z_latent, config, context
        )
        prediction_energy = prediction_ns_weight * _navier_stokes_residual_energy(
            z_mu, config, context
        )
        return data_energy + latent_energy + prediction_energy

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray,
        z_mu: jnp.ndarray,
        config: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> jnp.ndarray:
        if config is None:
            raise ValueError("NavierStokesEnergy requires a config")

        def total_energy(latent):
            return jnp.sum(
                NavierStokesEnergy.energy(latent, z_mu, config=config, context=context)
            )

        return jax.grad(total_energy)(z_latent)


def compute_energy(
    z_latent: jnp.ndarray,
    z_mu: jnp.ndarray,
    energy: EnergyFunctional = None,
    context: Dict[str, Any] = None,
) -> jnp.ndarray:
    """
    Compute energy using the specified energy functional.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy: EnergyFunctional instance. If None, uses GaussianEnergy with defaults.
        context: Optional runtime metadata (for example node_info)

    Returns:
        Energy per sample, shape (batch,)
    """
    if energy is None:
        energy = GaussianEnergy()

    return type(energy).energy(z_latent, z_mu, energy.config, context=context)


def compute_energy_gradient(
    z_latent: jnp.ndarray,
    z_mu: jnp.ndarray,
    energy: EnergyFunctional = None,
    context: Dict[str, Any] = None,
) -> jnp.ndarray:
    """
    Compute energy gradient w.r.t. z_latent.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy: EnergyFunctional instance. If None, uses GaussianEnergy with defaults.
        context: Optional runtime metadata (for example node_info)

    Returns:
        Gradient dE/dz_latent, same shape as z_latent
    """
    if energy is None:
        energy = GaussianEnergy()

    return type(energy).grad_latent(z_latent, z_mu, energy.config, context=context)


def get_energy_and_gradient(
    z_latent: jnp.ndarray,
    z_mu: jnp.ndarray,
    energy: EnergyFunctional = None,
    context: Dict[str, Any] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute both energy and gradient efficiently.

    Args:
        z_latent: Latent states, shape (batch, *dims)
        z_mu: Predicted expectations, shape (batch, *dims)
        energy: EnergyFunctional instance. If None, uses GaussianEnergy with defaults.
        context: Optional runtime metadata (for example node_info)

    Returns:
        Tuple of (energy, gradient)
    """
    if energy is None:
        energy = GaussianEnergy()

    energy_cls = type(energy)
    config = energy.config
    e = energy_cls.energy(z_latent, z_mu, config, context=context)
    g = energy_cls.grad_latent(z_latent, z_mu, config, context=context)
    return e, g
