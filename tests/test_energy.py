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
    NavierStokesEnergy,
)
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params, initialize_graph_state
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.graph.graph_net import compute_local_weight_gradients


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
            def energy(z_latent, z_mu, config=None, context=None):
                diff = z_latent - z_mu
                axes = tuple(range(1, len(diff.shape)))
                return jnp.sum(jnp.abs(diff), axis=axes)

            @staticmethod
            def grad_latent(z_latent, z_mu, config=None, context=None):
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


class TestNavierStokesEnergy:
    """Test Navier-Stokes energy functional."""

    @staticmethod
    def _field_from_uvp(u, v, p):
        return jnp.stack([u, v, p], axis=-1)[None, ...]

    def test_zero_energy_for_matching_constant_divergence_free_field(self):
        zeros = jnp.zeros((4, 4))
        ones = jnp.ones((4, 4))
        field = self._field_from_uvp(ones, zeros, zeros)
        energy_obj = NavierStokesEnergy(viscosity=0.1)

        energy = energy_obj.energy(field, field, energy_obj.config)

        assert energy.shape == (1,)
        assert jnp.allclose(energy, 0.0, atol=1e-6)

    def test_positive_divergence_penalty(self):
        xs = jnp.arange(4, dtype=jnp.float32)
        u = jnp.tile(xs[None, :], (4, 1))
        zeros = jnp.zeros((4, 4))
        field = self._field_from_uvp(u, zeros, zeros)

        energy_obj = NavierStokesEnergy(
            viscosity=0.1,
            data_weight=0.0,
            latent_ns_weight=1.0,
            prediction_ns_weight=0.0,
        )
        energy = energy_obj.energy(field, field, energy_obj.config)

        assert energy[0] > 0

    def test_positive_momentum_residual(self):
        ys = jnp.arange(4, dtype=jnp.float32)
        u = jnp.tile(jnp.sin(2 * jnp.pi * ys / 4)[:, None], (1, 4))
        zeros = jnp.zeros((4, 4))
        field = self._field_from_uvp(u, zeros, zeros)

        energy_obj = NavierStokesEnergy(
            viscosity=0.1,
            data_weight=0.0,
            latent_ns_weight=1.0,
            prediction_ns_weight=0.0,
        )
        energy = energy_obj.energy(field, field, energy_obj.config)

        assert energy[0] > 0

    def test_zero_energy_when_all_weights_disabled(self):
        zeros = jnp.zeros((4, 4, 3))
        ones = jnp.ones((4, 4, 3))
        z_latent = zeros[None, ...]
        z_mu = ones[None, ...]

        energy_obj = NavierStokesEnergy(
            viscosity=0.1,
            data_weight=0.0,
            latent_ns_weight=0.0,
            prediction_ns_weight=0.0,
        )

        energy = energy_obj.energy(z_latent, z_mu, energy_obj.config)
        grad = energy_obj.grad_latent(z_latent, z_mu, energy_obj.config)

        assert jnp.allclose(energy, 0.0, atol=1e-6)
        assert jnp.allclose(grad, 0.0, atol=1e-6)

    def test_validation_errors(self):
        energy_obj = NavierStokesEnergy(viscosity=0.1)

        with pytest.raises(ValueError, match="rank-4 tensors"):
            energy_obj.energy(
                jnp.zeros((4, 4, 3)),
                jnp.zeros((4, 4, 3)),
                energy_obj.config,
            )

        with pytest.raises(ValueError, match="channels for u, v, and p"):
            energy_obj.energy(
                jnp.zeros((1, 4, 4, 2)),
                jnp.zeros((1, 4, 4, 2)),
                energy_obj.config,
            )

        with pytest.raises(ValueError, match="H >= 3 and W >= 3"):
            energy_obj.energy(
                jnp.zeros((1, 2, 4, 3)),
                jnp.zeros((1, 2, 4, 3)),
                energy_obj.config,
            )


class TestNavierStokesIntegration:
    """Integration tests for Navier-Stokes energy in graph training."""

    def test_graph_runs_inference_and_local_gradients(self):
        input_node = IdentityNode(shape=(4, 4, 3), name="input")
        output_node = Linear(
            shape=(4, 4, 3),
            name="output",
            activation=IdentityActivation(),
            energy=NavierStokesEnergy(viscosity=0.1),
        )

        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(eta_infer=0.01, infer_steps=2),
        )

        params = initialize_params(structure, jax.random.PRNGKey(0))
        clamps = {
            structure.task_map["x"]: jnp.ones((2, 4, 4, 3), dtype=jnp.float32),
            structure.task_map["y"]: jnp.ones((2, 4, 4, 3), dtype=jnp.float32) * 0.5,
        }

        state = initialize_graph_state(
            structure,
            batch_size=2,
            rng_key=jax.random.PRNGKey(1),
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)
        grads = compute_local_weight_gradients(params, final_state, structure)

        assert final_state.nodes["output"].energy.shape == (2,)
        assert grads.nodes["output"].weights
        for key, weight in params.nodes["output"].weights.items():
            assert grads.nodes["output"].weights[key].shape == weight.shape
