"""Tests for StorkeyHopfield associative memory node."""

import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear, IdentityNode, StorkeyHopfield
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.graph.graph_net import compute_local_weight_gradients
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.activations import (
    TanhActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import GaussianEnergy, CrossEntropyEnergy
from fabricpc.core.initializers import XavierInitializer, NormalInitializer
from fabricpc.training import train_step
from conftest import with_inference
import optax

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


def _build_hopfield_graph_1d(D=32, infer_steps=10, eta_infer=0.1):
    """Build a simple Linear -> StorkeyHopfield -> Linear graph with 1D shapes."""
    input_node = IdentityNode(shape=(D,), name="input")
    hopfield = StorkeyHopfield(shape=(D,), name="hopfield")
    output = Linear(shape=(5,), activation=SigmoidActivation(), name="output")

    structure = graph(
        nodes=[input_node, hopfield, output],
        edges=[
            Edge(source=input_node, target=hopfield.slot("in")),
            Edge(source=hopfield, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )
    return structure


def _build_hopfield_graph_2d(seq=4, D=16, infer_steps=10, eta_infer=0.1):
    """Build a graph with 2D shape (seq, D) through StorkeyHopfield."""
    input_node = IdentityNode(shape=(seq, D), name="input")
    hopfield = StorkeyHopfield(shape=(seq, D), name="hopfield")
    output = Linear(
        shape=(seq, 5),
        activation=SigmoidActivation(),
        name="output",
        flatten_input=True,
    )

    structure = graph(
        nodes=[input_node, hopfield, output],
        edges=[
            Edge(source=input_node, target=hopfield.slot("in")),
            Edge(source=hopfield, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )
    return structure


# =========================================================================
# Shape tests
# =========================================================================


class TestShapes:
    """Verify parameter and state shapes for 1D and 2D node shapes."""

    def test_init_params_1d(self, rng_key):
        """1D shape: params have correct shapes."""
        D = 32
        structure = _build_hopfield_graph_1d(D=D)
        params = initialize_params(structure, rng_key)

        hop_params = params.nodes["hopfield"]
        # Single weight matrix (W) stored under the edge key, shape (D, D)
        edge_keys = list(hop_params.weights.keys())
        assert len(edge_keys) == 1
        assert hop_params.weights[edge_keys[0]].shape == (D, D)
        # Bias
        assert "b" in hop_params.biases
        # Learnable hopfield_strength
        assert "hopfield_strength" in hop_params.biases
        assert hop_params.biases["hopfield_strength"].shape == ()

    def test_init_params_2d(self, rng_key):
        """2D shape: Hopfield W is (D, D) on last axis."""
        seq, D = 4, 16
        structure = _build_hopfield_graph_2d(seq=seq, D=D)
        params = initialize_params(structure, rng_key)

        hop_params = params.nodes["hopfield"]
        edge_key = list(hop_params.weights.keys())[0]
        assert hop_params.weights[edge_key].shape == (D, D)

    def test_forward_1d(self, rng_key):
        """Forward pass produces correctly shaped state."""
        D = 32
        structure = _build_hopfield_graph_1d(D=D)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, state_key = jax.random.split(rng_key)
        x = jax.random.normal(state_key, (batch_size, D))
        y = jax.random.normal(state_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )

        hop_state = state.nodes["hopfield"]
        assert hop_state.z_latent.shape == (batch_size, D)
        assert hop_state.z_mu.shape == (batch_size, D)


# =========================================================================
# Energy tests
# =========================================================================


class TestEnergy:
    """Verify combined energy behavior during inference."""

    def test_energy_finite(self, rng_key):
        """Combined energy is finite after forward pass."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=1)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, x_key = jax.random.split(rng_key)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)

        hop_energy = final_state.nodes["hopfield"].energy
        assert jnp.all(jnp.isfinite(hop_energy))

    def test_energy_decreases_during_inference(self, rng_key):
        """Energy should decrease (or not increase) over inference steps."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=50, eta_infer=0.05)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, x_key = jax.random.split(rng_key)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        # Run with few steps
        struct_few = with_inference(structure, infer_steps=5, eta_infer=0.05)
        state_init = initialize_graph_state(
            struct_few,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        state_few = run_inference(params, state_init, clamps, struct_few)
        energy_few = sum(
            float(jnp.sum(state_few.nodes[n].energy))
            for n in structure.nodes
            if structure.nodes[n].node_info.in_degree > 0
        )

        # Run with more steps
        struct_many = with_inference(structure, infer_steps=50, eta_infer=0.05)
        state_init = initialize_graph_state(
            struct_many,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        state_many = run_inference(params, state_init, clamps, struct_many)
        energy_many = sum(
            float(jnp.sum(state_many.nodes[n].energy))
            for n in structure.nodes
            if structure.nodes[n].node_info.in_degree > 0
        )

        assert (
            energy_many <= energy_few + 1e-3
        ), f"Energy did not decrease: {energy_few} -> {energy_many}"


# =========================================================================
# Gradient tests
# =========================================================================


class TestGradients:
    """Verify autodiff weight gradients through combined energy."""

    def test_gradient_shapes_match_params(self, rng_key):
        """Weight gradients have same shapes as parameters."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=5)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, x_key = jax.random.split(rng_key)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)

        gradients = compute_local_weight_gradients(params, final_state, structure)

        hop_grads = gradients.nodes["hopfield"]
        hop_params = params.nodes["hopfield"]

        for key in hop_params.weights:
            assert key in hop_grads.weights, f"Missing gradient for weight {key}"
            assert hop_grads.weights[key].shape == hop_params.weights[key].shape

        for key in hop_params.biases:
            assert key in hop_grads.biases, f"Missing gradient for bias {key}"
            assert hop_grads.biases[key].shape == hop_params.biases[key].shape

    def test_hopfield_w_gradient_nonzero(self, rng_key):
        """Hopfield W receives nonzero gradients."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=5)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, x_key = jax.random.split(rng_key)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)
        gradients = compute_local_weight_gradients(params, final_state, structure)

        hop_weight_keys = list(gradients.nodes["hopfield"].weights.keys())
        w_grad = gradients.nodes["hopfield"].weights[hop_weight_keys[0]]
        assert jnp.any(w_grad != 0.0), "Hopfield W gradient is all zeros"

    def test_hopfield_strength_gradient(self, rng_key):
        """Learnable hopfield_strength receives a gradient."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=5)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, x_key = jax.random.split(rng_key)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)
        gradients = compute_local_weight_gradients(params, final_state, structure)

        strength_grad = gradients.nodes["hopfield"].biases["hopfield_strength"]
        assert jnp.isfinite(strength_grad), "hopfield_strength gradient is not finite"


# =========================================================================
# Symmetry tests
# =========================================================================


class TestSymmetry:
    """Verify symmetry enforcement on W."""

    def test_prepare_w_symmetric(self, rng_key):
        """_prepare_W produces a symmetric matrix when enforce_symmetry=True."""
        D = 16
        W = jax.random.normal(rng_key, (D, D))
        config = {"enforce_symmetry": True, "zero_diagonal": False}
        W_prepared = StorkeyHopfield._prepare_W(W, config)
        assert jnp.allclose(W_prepared, W_prepared.T, atol=1e-6)

    def test_prepare_w_zero_diagonal(self, rng_key):
        """_prepare_W zeros the diagonal when zero_diagonal=True."""
        D = 16
        W = jax.random.normal(rng_key, (D, D))
        config = {"enforce_symmetry": False, "zero_diagonal": True}
        W_prepared = StorkeyHopfield._prepare_W(W, config)
        assert jnp.allclose(jnp.diag(W_prepared), 0.0, atol=1e-6)

    def test_init_w_symmetric(self, rng_key):
        """Initialized W is symmetric by default."""
        D = 32
        structure = _build_hopfield_graph_1d(D=D)
        params = initialize_params(structure, rng_key)
        edge_key = list(params.nodes["hopfield"].weights.keys())[0]
        W = params.nodes["hopfield"].weights[edge_key]
        assert jnp.allclose(W, W.T, atol=1e-6)


# =========================================================================
# Config tests
# =========================================================================


class TestConfig:
    """Verify constructor config options."""

    def test_fixed_hopfield_strength(self, rng_key):
        """Fixed hopfield_strength is not in params.biases."""
        D = 16
        input_node = IdentityNode(shape=(D,), name="input")
        hopfield = StorkeyHopfield(shape=(D,), name="hopfield", hopfield_strength=2.0)
        output = Linear(shape=(5,), activation=SigmoidActivation(), name="output")

        structure = graph(
            nodes=[input_node, hopfield, output],
            edges=[
                Edge(source=input_node, target=hopfield.slot("in")),
                Edge(source=hopfield, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)

        # Fixed strength: not a learnable parameter
        assert "hopfield_strength" not in params.nodes["hopfield"].biases

    def test_no_bias(self, rng_key):
        """use_bias=False omits bias from params."""
        D = 16
        input_node = IdentityNode(shape=(D,), name="input")
        hopfield = StorkeyHopfield(shape=(D,), name="hopfield", use_bias=False)
        output = Linear(shape=(5,), activation=SigmoidActivation(), name="output")

        structure = graph(
            nodes=[input_node, hopfield, output],
            edges=[
                Edge(source=input_node, target=hopfield.slot("in")),
                Edge(source=hopfield, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)
        assert "b" not in params.nodes["hopfield"].biases


# =========================================================================
# Integration tests
# =========================================================================


class TestIntegration:
    """End-to-end tests with full PC graph."""

    def test_full_inference_and_training_step(self, rng_key):
        """Run inference + one training step with Linear -> Hopfield -> Linear."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=10, eta_infer=0.05)
        params = initialize_params(structure, rng_key)
        batch_size = 8

        rng_key, x_key, train_key = jax.random.split(rng_key, 3)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        batch = {"input": x, "output": y}

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        # One training step
        new_params, opt_state, energy, _ = train_step(
            params, opt_state, batch, structure, optimizer, train_key
        )

        assert jnp.isfinite(energy), f"Training energy is not finite: {energy}"
        # Params should have changed
        edge_key = list(params.nodes["hopfield"].weights.keys())[0]
        old_w = params.nodes["hopfield"].weights[edge_key]
        new_w = new_params.nodes["hopfield"].weights[edge_key]
        assert not jnp.allclose(old_w, new_w, atol=1e-8), "Hopfield W did not update"

    def test_four_node_graph(self, rng_key):
        """4-node graph: input -> Linear -> Hopfield -> Linear (classifier)."""
        D_hidden = 16
        D_hop = 16
        n_classes = 5

        input_node = IdentityNode(shape=(32,), name="input")
        hidden = Linear(
            shape=(D_hidden,),
            activation=SigmoidActivation(),
            name="hidden",
            weight_init=XavierInitializer(),
        )
        hopfield = StorkeyHopfield(
            shape=(D_hop,),
            name="hopfield",
            weight_init=NormalInitializer(mean=0.0, std=0.01),
        )
        output = Linear(
            shape=(n_classes,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
            weight_init=XavierInitializer(),
        )

        structure = graph(
            nodes=[input_node, hidden, hopfield, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("in")),
                Edge(source=hidden, target=hopfield.slot("in")),
                Edge(source=hopfield, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
        )

        params = initialize_params(structure, rng_key)
        batch_size = 8

        rng_key, x_key, train_key = jax.random.split(rng_key, 3)
        x = jax.random.normal(x_key, (batch_size, 32))
        y = jax.nn.one_hot(
            jax.random.randint(x_key, (batch_size,), 0, n_classes), n_classes
        )
        batch = {"input": x, "class": y}

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        new_params, opt_state, energy, _ = train_step(
            params, opt_state, batch, structure, optimizer, train_key
        )
        assert jnp.isfinite(energy)
