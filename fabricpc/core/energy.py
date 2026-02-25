"""Energy functionals for predictive coding networks.

Direct-construction API only: instantiate energy classes and pass objects.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp


class EnergyFunctional(ABC):
    """Base class for node energy functionals."""

    @abstractmethod
    def energy(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def grad_latent(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        pass


class GaussianEnergy(EnergyFunctional):
    def __init__(self, precision: float = 1.0):
        self.precision = float(precision)

    def energy(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        diff = z_latent - z_mu
        axes_to_sum = tuple(range(1, len(diff.shape)))
        return 0.5 * self.precision * jnp.sum(diff**2, axis=axes_to_sum)

    def grad_latent(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        return self.precision * (z_latent - z_mu)


class BernoulliEnergy(EnergyFunctional):
    def __init__(self, eps: float = 1e-7):
        self.eps = float(eps)

    def energy(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        z_mu_safe = jnp.clip(z_mu, self.eps, 1 - self.eps)
        bce = -(z_latent * jnp.log(z_mu_safe) + (1 - z_latent) * jnp.log(1 - z_mu_safe))
        axes_to_sum = tuple(range(1, len(bce.shape)))
        return jnp.sum(bce, axis=axes_to_sum)

    def grad_latent(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        z_mu_safe = jnp.clip(z_mu, self.eps, 1 - self.eps)
        return -jnp.log(z_mu_safe) + jnp.log(1 - z_mu_safe)


class CrossEntropyEnergy(EnergyFunctional):
    def __init__(self, eps: float = 1e-7):
        self.eps = float(eps)

    def energy(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        z_mu_safe = jnp.clip(z_mu, self.eps, 1.0)
        ce = -z_latent * jnp.log(z_mu_safe)
        axes_to_sum = tuple(range(1, len(ce.shape)))
        return jnp.sum(ce, axis=axes_to_sum)

    def grad_latent(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        z_mu_safe = jnp.clip(z_mu, self.eps, 1.0)
        return -jnp.log(z_mu_safe)


class LaplacianEnergy(EnergyFunctional):
    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    def energy(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        diff = jnp.abs(z_latent - z_mu)
        axes_to_sum = tuple(range(1, len(diff.shape)))
        return jnp.sum(diff, axis=axes_to_sum) / self.scale

    def grad_latent(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        return jnp.sign(z_latent - z_mu) / self.scale


class HuberEnergy(EnergyFunctional):
    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def energy(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        diff = z_latent - z_mu
        abs_diff = jnp.abs(diff)
        quadratic = 0.5 * diff**2
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        huber = jnp.where(abs_diff <= self.delta, quadratic, linear)
        axes_to_sum = tuple(range(1, len(huber.shape)))
        return jnp.sum(huber, axis=axes_to_sum)

    def grad_latent(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(z_latent - z_mu, -self.delta, self.delta)


class KLDivergenceEnergy(EnergyFunctional):
    def __init__(self, eps: float = 1e-7):
        self.eps = float(eps)

    def energy(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        z_latent_safe = jnp.clip(z_latent, self.eps, 1.0)
        z_mu_safe = jnp.clip(z_mu, self.eps, 1.0)
        kl = z_latent_safe * (jnp.log(z_latent_safe) - jnp.log(z_mu_safe))
        kl = jnp.where(z_latent < self.eps, 0.0, kl)
        axes_to_sum = tuple(range(1, len(kl.shape)))
        return jnp.sum(kl, axis=axes_to_sum)

    def grad_latent(self, z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        z_latent_safe = jnp.clip(z_latent, self.eps, 1.0)
        z_mu_safe = jnp.clip(z_mu, self.eps, 1.0)
        grad = jnp.log(z_latent_safe) - jnp.log(z_mu_safe) + 1.0
        return jnp.where(z_latent < self.eps, -jnp.log(z_mu_safe), grad)


def ensure_energy(energy: EnergyFunctional | None) -> EnergyFunctional:
    """Normalize optional energy input to an EnergyFunctional instance."""
    return energy if energy is not None else GaussianEnergy()


def get_energy_and_gradient(
    z_latent: jnp.ndarray,
    z_mu: jnp.ndarray,
    energy: EnergyFunctional,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute energy and latent gradient for a node."""
    if not isinstance(energy, EnergyFunctional):
        raise TypeError(
            "energy must be an EnergyFunctional instance; string/type lookup is removed"
        )
    e = energy.energy(z_latent, z_mu)
    g = energy.grad_latent(z_latent, z_mu)
    return e, g
