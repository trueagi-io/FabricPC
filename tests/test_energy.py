"""
Test suite for energy functional module.

Tests built-in energy functionals, custom energy creation,
and integration with graph construction.
"""

import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.energy import (
    EnergyFunctional,
    GaussianEnergy,
    BernoulliEnergy,
    CrossEntropyEnergy,
    LaplacianEnergy,
    KLDivergenceEnergy,
)
from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.inference import InferenceSGD


class TestGaussianEnergy:
    """Test Gaussian energy functional."""

    def test_gaussian_energy_computation(self):
        """Test Gaussian energy: E = 0.5 * ||z - mu||^2"""
        z_latent = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        energy = GaussianEnergy.energy(z_latent, z_mu)

        assert energy.shape == (2,)
        assert jnp.allclose(energy[0], 7.0)
        assert jnp.allclose(energy[1], 38.5)

    def test_gaussian_gradient_computation(self):
        """Test Gaussian gradient: dE/dz = z - mu"""
        z_latent = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        z_mu = jnp.array([[0.5, 1.0], [1.5, 2.0]])

        grad = GaussianEnergy.grad_latent(z_latent, z_mu)

        expected = z_latent - z_mu
        assert jnp.allclose(grad, expected)


class TestBernoulliEnergy:
    """Test Bernoulli (BCE) energy functional."""

    def test_bernoulli_energy_computation(self):
        """Test Bernoulli cross-entropy energy."""
        z_latent = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        z_mu = jnp.array([[0.9, 0.1], [0.2, 0.8]])

        energy = BernoulliEnergy.energy(z_latent, z_mu)

        assert energy.shape == (2,)
        assert energy[0] > 0
        assert energy[1] > 0


class TestCrossEntropyEnergy:
    """Test Cross Entropy energy functional."""

    def test_categorical_energy_computation(self):
        """Test cross-entropy energy."""
        z_latent = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        z_mu = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])

        energy = CrossEntropyEnergy.energy(z_latent, z_mu)

        assert energy.shape == (2,)
        expected_0 = -jnp.log(0.8)
        expected_1 = -jnp.log(0.7)
        assert jnp.allclose(energy[0], expected_0, atol=1e-5)
        assert jnp.allclose(energy[1], expected_1, atol=1e-5)


class TestLaplacianEnergy:
    """Test Laplacian (L1) energy functional."""

    def test_laplacian_energy_computation(self):
        """Test Laplacian energy: E = ||z - mu||_1"""
        z_latent = jnp.array([[1.0, -2.0, 3.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0]])

        energy = LaplacianEnergy.energy(z_latent, z_mu)

        assert jnp.allclose(energy[0], 6.0)


class TestKLDivergenceEnergy:
    """Test KL Divergence energy functional."""

    def test_kl_divergence_batch_computation(self):
        """Test KL divergence computes correct numerical values."""
        z_latent = jnp.array(
            [
                [0.7, 0.2, 0.1],
                [0.3, 0.3, 0.4],
            ]
        )
        z_mu = jnp.array(
            [
                [0.6, 0.3, 0.1],
                [0.5, 0.25, 0.25],
            ]
        )

        energy = KLDivergenceEnergy.energy(z_latent, z_mu)

        assert energy.shape == (2,)

        expected_0 = (
            0.7 * jnp.log(0.7 / 0.6)
            + 0.2 * jnp.log(0.2 / 0.3)
            + 0.1 * jnp.log(0.1 / 0.1)
        )
        expected_1 = (
            0.3 * jnp.log(0.3 / 0.5)
            + 0.3 * jnp.log(0.3 / 0.25)
            + 0.4 * jnp.log(0.4 / 0.25)
        )

        assert jnp.allclose(energy[0], expected_0, atol=1e-5)
        assert jnp.allclose(energy[1], expected_1, atol=1e-5)


class TestCustomEnergy:
    """Test creating custom energy functionals."""

    def test_custom_energy_subclass(self):
        """Test creating and using a custom energy subclass."""

        class L1Energy(EnergyFunctional):
            def __init__(self):
                super().__init__()

            @staticmethod
            def energy(z_latent, z_mu, config=None):
                diff = z_latent - z_mu
                axes = tuple(range(1, len(diff.shape)))
                return jnp.sum(jnp.abs(diff), axis=axes)

            @staticmethod
            def grad_latent(z_latent, z_mu, config=None):
                return jnp.sign(z_latent - z_mu)

        z_latent = jnp.array([[1.0, -2.0, 3.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0]])

        energy = L1Energy.energy(z_latent, z_mu)
        assert jnp.allclose(energy[0], 6.0)

        grad = L1Energy.grad_latent(z_latent, z_mu)
        assert jnp.allclose(grad, jnp.array([[1.0, -1.0, 1.0]]))


class TestIntegration:
    """Integration tests with graph construction."""

    def test_graph_with_custom_energy(self):
        """Test that graphs can use custom energy."""
        input_node = Linear(shape=(8,), name="input")
        output_node = Linear(shape=(4,), energy=BernoulliEnergy(), name="output")

        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        assert isinstance(structure.nodes["input"].node_info.energy, GaussianEnergy)
        assert isinstance(structure.nodes["output"].node_info.energy, BernoulliEnergy)
