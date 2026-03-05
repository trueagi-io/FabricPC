"""
Test suite for energy functional module.

Tests built-in energy functionals, custom energy creation,
and integration with graph construction.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.energy import (
    EnergyFunctional,
    GaussianEnergy,
    BernoulliEnergy,
    CrossEntropyEnergy,
    LaplacianEnergy,
    HuberEnergy,
    KLDivergenceEnergy,
)
from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.inference import InferenceSGD

jax.config.update("jax_platform_name", "cpu")


class TestGaussianEnergy:
    """Test Gaussian energy functional."""

    def test_gaussian_energy_computation(self):
        """Test Gaussian energy: E = 0.5 * ||z - mu||^2"""
        z_latent = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        energy = GaussianEnergy.energy(z_latent, z_mu)

        # Expected: 0.5 * (1^2 + 2^2 + 3^2) = 0.5 * 14 = 7.0 for first sample
        # Expected: 0.5 * (4^2 + 5^2 + 6^2) = 0.5 * 77 = 38.5 for second sample
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

    def test_gaussian_precision_parameter(self):
        """Test Gaussian energy with precision parameter."""
        z_latent = jnp.array([[1.0, 2.0]])
        z_mu = jnp.array([[0.0, 0.0]])
        config = {"precision": 2.0}

        energy = GaussianEnergy.energy(z_latent, z_mu, config)
        grad = GaussianEnergy.grad_latent(z_latent, z_mu, config)

        # With precision=2: E = precision/2 * ||z - mu||^2 = 1.0 * 5 = 5.0
        assert jnp.allclose(energy[0], 5.0)
        # Gradient scaled by precision
        assert jnp.allclose(grad, 2.0 * (z_latent - z_mu))

    def test_gaussian_energy_instance_config(self):
        """Test GaussianEnergy instance stores config correctly."""
        energy_obj = GaussianEnergy(precision=2.0)
        assert energy_obj.config == {"precision": 2.0}

        # Use the instance's config
        z_latent = jnp.array([[1.0, 2.0]])
        z_mu = jnp.array([[0.0, 0.0]])
        energy = GaussianEnergy.energy(z_latent, z_mu, energy_obj.config)
        assert jnp.allclose(energy[0], 5.0)


class TestBernoulliEnergy:
    """Test Bernoulli (BCE) energy functional."""

    def test_bernoulli_energy_computation(self):
        """Test Bernoulli cross-entropy energy."""
        # Binary targets
        z_latent = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        # Predicted probabilities
        z_mu = jnp.array([[0.9, 0.1], [0.2, 0.8]])

        energy = BernoulliEnergy.energy(z_latent, z_mu)

        # BCE = -[z*log(mu) + (1-z)*log(1-mu)]
        assert energy.shape == (2,)
        assert energy[0] > 0  # Should be positive
        assert energy[1] > 0

    def test_bernoulli_perfect_prediction(self):
        """Test that perfect prediction gives near-zero energy."""
        z_latent = jnp.array([[1.0, 0.0]])
        z_mu = jnp.array([[0.9999, 0.0001]])  # Near-perfect

        energy = BernoulliEnergy.energy(z_latent, z_mu)
        assert energy[0] < 0.01  # Should be very small


class TestCrossEntropyEnergy:
    """Test Cross Entropy energy functional."""

    def test_categorical_energy_computation(self):
        """Test cross-entropy energy."""
        # One-hot targets
        z_latent = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Softmax probabilities
        z_mu = jnp.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])

        energy = CrossEntropyEnergy.energy(z_latent, z_mu)

        # CE = -sum(z * log(mu))
        assert energy.shape == (2,)
        expected_0 = -jnp.log(0.8)  # Only the correct class contributes
        expected_1 = -jnp.log(0.7)
        assert jnp.allclose(energy[0], expected_0, atol=1e-5)
        assert jnp.allclose(energy[1], expected_1, atol=1e-5)

    def test_cross_entropy_instance_config(self):
        """Test CrossEntropyEnergy instance stores config correctly."""
        energy_obj = CrossEntropyEnergy(eps=1e-6)
        assert energy_obj.config["eps"] == 1e-6


class TestLaplacianEnergy:
    """Test Laplacian (L1) energy functional."""

    def test_laplacian_energy_computation(self):
        """Test Laplacian energy: E = ||z - mu||_1"""
        z_latent = jnp.array([[1.0, -2.0, 3.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0]])

        energy = LaplacianEnergy.energy(z_latent, z_mu)

        # Expected: |1| + |-2| + |3| = 6.0
        assert jnp.allclose(energy[0], 6.0)

    def test_laplacian_gradient_is_sign(self):
        """Test Laplacian gradient: dE/dz = sign(z - mu)"""
        z_latent = jnp.array([[1.0, -2.0, 0.0]])
        z_mu = jnp.array([[0.0, 0.0, 0.0]])

        grad = LaplacianEnergy.grad_latent(z_latent, z_mu)

        expected = jnp.array([[1.0, -1.0, 0.0]])
        assert jnp.allclose(grad, expected)


class TestHuberEnergy:
    """Test Huber (smooth L1) energy functional."""

    def test_huber_quadratic_region(self):
        """Test Huber energy in quadratic region (|diff| <= delta)."""
        z_latent = jnp.array([[0.5]])
        z_mu = jnp.array([[0.0]])
        config = {"delta": 1.0}

        energy = HuberEnergy.energy(z_latent, z_mu, config)

        # In quadratic region: E = 0.5 * diff^2 = 0.5 * 0.25 = 0.125
        assert jnp.allclose(energy[0], 0.125)

    def test_huber_linear_region(self):
        """Test Huber energy in linear region (|diff| > delta)."""
        z_latent = jnp.array([[2.0]])
        z_mu = jnp.array([[0.0]])
        config = {"delta": 1.0}

        energy = HuberEnergy.energy(z_latent, z_mu, config)

        # In linear region: E = delta * (|diff| - 0.5 * delta) = 1.0 * (2.0 - 0.5) = 1.5
        assert jnp.allclose(energy[0], 1.5)

    def test_huber_instance_config(self):
        """Test HuberEnergy instance stores config."""
        energy_obj = HuberEnergy(delta=0.5)
        assert energy_obj.config == {"delta": 0.5}


class TestKLDivergenceEnergy:
    """Test KL Divergence energy functional."""

    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence is zero for identical distributions."""
        z_latent = jnp.array([[0.2, 0.3, 0.4, 0.1]])
        z_mu = jnp.array([[0.2, 0.3, 0.4, 0.1]])

        energy = KLDivergenceEnergy.energy(z_latent, z_mu)

        assert energy.shape == (1,)
        assert jnp.allclose(energy[0], 0.0, atol=1e-6)

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

        # Manually compute expected values
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

    def test_kl_divergence_always_non_negative(self):
        """Test that KL divergence is always >= 0 (Gibbs' inequality)."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        raw_z = jax.random.uniform(key1, (10, 5)) + 0.01
        raw_mu = jax.random.uniform(key2, (10, 5)) + 0.01
        z_latent = raw_z / raw_z.sum(axis=-1, keepdims=True)
        z_mu = raw_mu / raw_mu.sum(axis=-1, keepdims=True)

        energy = KLDivergenceEnergy.energy(z_latent, z_mu)

        assert jnp.all(energy >= -1e-6)

    def test_kl_divergence_zero_probability_handling(self):
        """Test that zero probabilities are handled correctly."""
        z_latent = jnp.array([[1.0, 0.0, 0.0]])
        z_mu = jnp.array([[0.8, 0.1, 0.1]])

        energy = KLDivergenceEnergy.energy(z_latent, z_mu)

        expected = 1.0 * jnp.log(1.0 / 0.8)

        assert jnp.isfinite(energy[0])
        assert jnp.allclose(energy[0], expected, atol=1e-5)


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


class TestNDimensionalShapes:
    """Test energy computation with various tensor shapes."""

    def test_1d_tensors(self):
        """Test energy with 1D tensors (batch, features)."""
        z_latent = jnp.ones((4, 10))
        z_mu = jnp.zeros((4, 10))

        energy = GaussianEnergy.energy(z_latent, z_mu)
        assert energy.shape == (4,)
        assert jnp.allclose(energy, 5.0)  # 0.5 * 10 = 5.0

    def test_2d_tensors(self):
        """Test energy with 2D tensors (batch, h, w)."""
        z_latent = jnp.ones((2, 4, 4))
        z_mu = jnp.zeros((2, 4, 4))

        energy = GaussianEnergy.energy(z_latent, z_mu)
        assert energy.shape == (2,)
        assert jnp.allclose(energy, 8.0)  # 0.5 * 16 = 8.0

    def test_3d_tensors(self):
        """Test energy with 3D tensors (batch, h, w, c)."""
        z_latent = jnp.ones((2, 4, 4, 3))
        z_mu = jnp.zeros((2, 4, 4, 3))

        energy = GaussianEnergy.energy(z_latent, z_mu)
        assert energy.shape == (2,)
        assert jnp.allclose(energy, 24.0)  # 0.5 * 48 = 24.0


class TestIntegration:
    """Integration tests with graph construction."""

    def test_graph_with_default_energy(self):
        """Test that graphs use default energy (GaussianEnergy)."""
        key = jax.random.PRNGKey(0)

        input_node = Linear(shape=(8,), name="input")
        output_node = Linear(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        # Both nodes should have GaussianEnergy by default
        assert isinstance(structure.nodes["input"].node_info.energy, GaussianEnergy)
        assert isinstance(structure.nodes["output"].node_info.energy, GaussianEnergy)

    def test_graph_with_custom_energy(self):
        """Test that graphs can use custom energy."""
        key = jax.random.PRNGKey(0)

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

    def test_graph_with_cross_entropy_energy(self):
        """Test graph creation with CrossEntropyEnergy."""
        key = jax.random.PRNGKey(0)

        input_node = Linear(shape=(8,), name="input")
        output_node = Linear(shape=(4,), energy=CrossEntropyEnergy(), name="output")

        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        assert isinstance(
            structure.nodes["output"].node_info.energy, CrossEntropyEnergy
        )
