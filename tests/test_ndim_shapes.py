#!/usr/bin/env python3
"""
Test suite for N-Dimensional Tensor Support (Section 1.1).

This test suite verifies:
1. Node with 1D shape: shape=(784,) - current behavior
2. Node with 2D shape: shape=(28, 28) - image without channels
3. Node with 3D shape: shape=(28, 28, 1) - image with channels
4. Mixed shapes in graph: input (784,) -> hidden (128,) -> output (10,)
5. Verify same params work with batch_size=1, 32, 128
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference
import optax
from fabricpc.training import train_step
from fabricpc.core.activations import ReLUActivation, TanhActivation, SigmoidActivation

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


class TestNDimShapes:
    """Test suite for n-dimensional tensor shapes."""

    def test_1d_shape(self, rng_key):
        """Test node with 1D shape: shape=(784,) - standard vector output."""
        node_input = Linear(shape=(784,), name="input")
        node_hidden = Linear(shape=(256,), activation=ReLUActivation(), name="hidden")
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_input, node_hidden, node_output],
            edges=[
                Edge(source=node_input, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_input, y=node_output),
        )
        params = initialize_params(structure, rng_key)

        # Verify shapes
        assert structure.nodes["input"].node_info.shape == (784,)
        assert structure.nodes["hidden"].node_info.shape == (256,)
        assert structure.nodes["output"].node_info.shape == (10,)

        # Verify weight shapes (flattened for linear)
        hidden_weights = params.nodes["hidden"].weights["input->hidden:in"]
        assert hidden_weights.shape == (
            784,
            256,
        ), f"Expected (784, 256), got {hidden_weights.shape}"

        output_weights = params.nodes["output"].weights["hidden->output:in"]
        assert output_weights.shape == (
            256,
            10,
        ), f"Expected (256, 10), got {output_weights.shape}"

        # Run inference
        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(
            params, state, clamps, structure, infer_steps=5, eta_infer=0.1
        )

        # Verify output shapes
        assert final_state.nodes["hidden"].z_latent.shape == (batch_size, 256)
        assert final_state.nodes["output"].z_latent.shape == (batch_size, 10)

    def test_2d_shape(self, rng_key):
        """Test node with 2D shape: shape=(28, 28) - image without channels."""
        node_image = Linear(shape=(28, 28), name="image")
        node_hidden = Linear(
            shape=(128,), activation=ReLUActivation(), flatten_input=True, name="hidden"
        )
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
        )
        params = initialize_params(structure, rng_key)

        # Verify shapes
        assert structure.nodes["image"].node_info.shape == (28, 28)
        assert structure.nodes["hidden"].node_info.shape == (128,)

        # Verify weight shapes - input is flattened (28*28=784)
        hidden_weights = params.nodes["hidden"].weights["image->hidden:in"]
        assert hidden_weights.shape == (
            784,
            128,
        ), f"Expected (784, 128), got {hidden_weights.shape}"

        # Verify bias shape for 2D output
        # (Note: hidden has 1D shape, so bias is (1, 128))
        hidden_bias = params.nodes["hidden"].biases["b"]
        assert hidden_bias.shape == (
            1,
            128,
        ), f"Expected (1, 128), got {hidden_bias.shape}"

        # Run inference with 2D input
        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 28, 28))  # 2D image input
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"image": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(
            params, state, clamps, structure, infer_steps=5, eta_infer=0.1
        )

        # Verify state shapes
        assert final_state.nodes["image"].z_latent.shape == (batch_size, 28, 28)
        assert final_state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert final_state.nodes["output"].z_latent.shape == (batch_size, 10)

    def test_3d_shape(self, rng_key):
        """Test node with 3D shape: shape=(28, 28, 1) - image with channels (NHWC)."""
        node_image = Linear(shape=(28, 28, 1), name="image")
        node_hidden = Linear(
            shape=(64,), activation=TanhActivation(), flatten_input=True, name="hidden"
        )
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
        )
        params = initialize_params(structure, rng_key)

        # Verify shapes
        assert structure.nodes["image"].node_info.shape == (28, 28, 1)

        # Verify weight shapes - input is flattened (28*28*1=784)
        hidden_weights = params.nodes["hidden"].weights["image->hidden:in"]
        assert hidden_weights.shape == (
            784,
            64,
        ), f"Expected (784, 64), got {hidden_weights.shape}"

        # Run inference with 3D input (NHWC format)
        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 28, 28, 1))  # 3D image input
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"image": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(
            params, state, clamps, structure, infer_steps=5, eta_infer=0.1
        )

        # Verify state shapes
        assert final_state.nodes["image"].z_latent.shape == (batch_size, 28, 28, 1)
        assert final_state.nodes["hidden"].z_latent.shape == (batch_size, 64)

    def test_3d_shape_multichannel(self, rng_key):
        """Test node with 3D shape: shape=(32, 32, 3) - RGB image."""
        node_rgb_image = Linear(shape=(32, 32, 3), name="rgb_image")
        node_hidden = Linear(
            shape=(256,),
            activation=ReLUActivation(),
            flatten_input=True,
            name="hidden",
        )
        node_output = Linear(shape=(100,), name="output")

        structure = graph(
            nodes=[node_rgb_image, node_hidden, node_output],
            edges=[
                Edge(source=node_rgb_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_rgb_image, y=node_output),
        )
        params = initialize_params(structure, rng_key)

        # Verify shapes
        assert structure.nodes["rgb_image"].node_info.shape == (32, 32, 3)

        # Verify weight shapes - input is flattened (32*32*3=3072)
        hidden_weights = params.nodes["hidden"].weights["rgb_image->hidden:in"]
        assert hidden_weights.shape == (
            3072,
            256,
        ), f"Expected (3072, 256), got {hidden_weights.shape}"

        # Run inference
        batch_size = 2
        x = jax.random.normal(rng_key, (batch_size, 32, 32, 3))
        y = jax.random.normal(rng_key, (batch_size, 100))
        clamps = {"rgb_image": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(
            params, state, clamps, structure, infer_steps=3, eta_infer=0.1
        )

        assert final_state.nodes["rgb_image"].z_latent.shape == (batch_size, 32, 32, 3)

    def test_mixed_shapes_in_graph(self, rng_key):
        """Test mixed shapes: 2D input -> 1D hidden -> 1D output."""
        node_image = Linear(shape=(28, 28), name="image")
        node_hidden1 = Linear(
            shape=(256,),
            activation=ReLUActivation(),
            flatten_input=True,
            name="hidden1",
        )
        node_hidden2 = Linear(shape=(128,), activation=ReLUActivation(), name="hidden2")
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden1, node_hidden2, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden1.slot("in")),
                Edge(source=node_hidden1, target=node_hidden2.slot("in")),
                Edge(source=node_hidden2, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
        )
        params = initialize_params(structure, rng_key)

        # Verify weight shapes for each transition
        # image (28, 28) -> hidden1 (256): weights (784, 256)
        w1 = params.nodes["hidden1"].weights["image->hidden1:in"]
        assert w1.shape == (784, 256)

        # hidden1 (256) -> hidden2 (128): weights (256, 128)
        w2 = params.nodes["hidden2"].weights["hidden1->hidden2:in"]
        assert w2.shape == (256, 128)

        # hidden2 (128) -> output (10): weights (128, 10)
        w3 = params.nodes["output"].weights["hidden2->output:in"]
        assert w3.shape == (128, 10)

        # Run inference
        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 28, 28))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"image": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(
            params, state, clamps, structure, infer_steps=5, eta_infer=0.1
        )

        # Verify all intermediate shapes
        assert final_state.nodes["image"].z_latent.shape == (batch_size, 28, 28)
        assert final_state.nodes["hidden1"].z_latent.shape == (batch_size, 256)
        assert final_state.nodes["hidden2"].z_latent.shape == (batch_size, 128)
        assert final_state.nodes["output"].z_latent.shape == (batch_size, 10)


class TestSameParamsDifferentBatchSizes:
    """Test that same params work with different batch sizes."""

    def test_same_params_multiple_batch_sizes(self, rng_key):
        """Verify same params work with batch_size=1, 32, 128."""
        node_input = Linear(shape=(784,), name="input")
        node_hidden = Linear(shape=(128,), activation=ReLUActivation(), name="hidden")
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_input, node_hidden, node_output],
            edges=[
                Edge(source=node_input, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_input, y=node_output),
        )

        # Create params ONCE
        params = initialize_params(structure, rng_key)

        # Test with different batch sizes using the SAME params
        for batch_size in [1, 32, 128]:
            key = jax.random.fold_in(rng_key, batch_size)
            x = jax.random.normal(key, (batch_size, 784))
            y = jax.random.normal(key, (batch_size, 10))
            clamps = {"input": x, "output": y}

            # Initialize state with this batch size
            state = initialize_graph_state(
                structure, batch_size, key, clamps=clamps, params=params
            )

            # Run inference
            final_state = run_inference(
                params, state, clamps, structure, infer_steps=5, eta_infer=0.1
            )

            # Verify shapes
            assert final_state.nodes["input"].z_latent.shape == (
                batch_size,
                784,
            ), f"Failed for batch_size={batch_size}"
            assert final_state.nodes["hidden"].z_latent.shape == (
                batch_size,
                128,
            ), f"Failed for batch_size={batch_size}"
            assert final_state.nodes["output"].z_latent.shape == (
                batch_size,
                10,
            ), f"Failed for batch_size={batch_size}"

            # Verify no NaN values
            assert not jnp.any(
                jnp.isnan(final_state.nodes["hidden"].z_latent)
            ), f"NaN values for batch_size={batch_size}"

    def test_same_params_2d_input_multiple_batch_sizes(self, rng_key):
        """Verify same params work with 2D inputs and different batch sizes."""
        node_image = Linear(shape=(28, 28), name="image")
        node_hidden = Linear(
            shape=(64,), activation=TanhActivation(), flatten_input=True, name="hidden"
        )
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
        )

        # Create params ONCE
        params = initialize_params(structure, rng_key)

        # Test with different batch sizes
        for batch_size in [1, 16, 64]:
            key = jax.random.fold_in(rng_key, batch_size)
            x = jax.random.normal(key, (batch_size, 28, 28))  # 2D input
            y = jax.random.normal(key, (batch_size, 10))
            clamps = {"image": x, "output": y}

            state = initialize_graph_state(
                structure, batch_size, key, clamps=clamps, params=params
            )
            final_state = run_inference(
                params, state, clamps, structure, infer_steps=5, eta_infer=0.1
            )

            # Verify shapes preserved
            assert final_state.nodes["image"].z_latent.shape == (batch_size, 28, 28)
            assert final_state.nodes["hidden"].z_latent.shape == (batch_size, 64)
            assert final_state.nodes["output"].z_latent.shape == (batch_size, 10)


class TestNDimTraining:
    """Test that training works with n-dimensional shapes."""

    def test_training_with_2d_input(self, rng_key):
        """Test complete training step with 2D image input."""
        node_image = Linear(shape=(28, 28), name="image")
        node_hidden = Linear(
            shape=(64,),
            activation=SigmoidActivation(),
            flatten_input=True,
            name="hidden",
        )
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
        )
        params = initialize_params(structure, rng_key)

        # Create optimizer
        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(params)

        # Create batch with 2D images
        batch_size = 8
        batch = {
            "x": jax.random.normal(rng_key, (batch_size, 28, 28)),
            "y": jax.random.normal(rng_key, (batch_size, 10)),
        }

        # Run training step
        new_params, new_opt_state, energy, final_state = train_step(
            params,
            opt_state,
            batch,
            structure,
            optimizer,
            rng_key,
            infer_steps=5,
            eta_infer=0.1,
        )

        # Verify energy is valid
        assert not jnp.isnan(energy), "Energy should not be NaN"
        assert energy > 0, "Energy should be positive"

        # Verify weights were updated
        old_w = params.nodes["hidden"].weights["image->hidden:in"]
        new_w = new_params.nodes["hidden"].weights["image->hidden:in"]
        assert not jnp.allclose(old_w, new_w), "Weights should be updated"

    def test_training_with_3d_input(self, rng_key):
        """Test complete training step with 3D image input (NHWC)."""
        node_image = Linear(shape=(16, 16, 3), name="image")
        node_hidden = Linear(
            shape=(32,), activation=ReLUActivation(), flatten_input=True, name="hidden"
        )
        node_output = Linear(shape=(5,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
        )
        params = initialize_params(structure, rng_key)

        optimizer = optax.sgd(0.01)
        opt_state = optimizer.init(params)

        batch_size = 4
        batch = {
            "x": jax.random.normal(rng_key, (batch_size, 16, 16, 3)),
            "y": jax.random.normal(rng_key, (batch_size, 5)),
        }

        new_params, _, energy, _ = train_step(
            params,
            opt_state,
            batch,
            structure,
            optimizer,
            rng_key,
            infer_steps=3,
            eta_infer=0.1,
        )

        assert not jnp.isnan(energy)
        assert energy > 0


class TestEnergyWithNDimShapes:
    """Test that energy computation works correctly with n-dim shapes."""

    def test_energy_decreases_2d_input(self, rng_key):
        """Test that energy decreases during inference with 2D input."""
        node_image = Linear(shape=(14, 14), name="image")
        node_hidden = Linear(
            shape=(32,), activation=TanhActivation(), flatten_input=True, name="hidden"
        )
        node_output = Linear(shape=(5,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
        )
        params = initialize_params(structure, rng_key)

        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 14, 14))
        y = jax.random.normal(rng_key, (batch_size, 5))
        clamps = {"image": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )

        # Run 1 step to get initial energy (energy is computed during inference, not initialization)
        initial_state = run_inference(
            params, state, clamps, structure, infer_steps=1, eta_infer=0.1
        )
        initial_energy = sum(
            jnp.sum(initial_state.nodes[name].energy)
            for name in structure.nodes
            if structure.nodes[name].node_info.in_degree > 0
        )

        # Run more inference steps
        final_state = run_inference(
            params, state, clamps, structure, infer_steps=20, eta_infer=0.1
        )

        # Get final energy
        final_energy = sum(
            jnp.sum(final_state.nodes[name].energy)
            for name in structure.nodes
            if structure.nodes[name].node_info.in_degree > 0
        )

        assert (
            final_energy < initial_energy
        ), f"Energy should decrease: initial={initial_energy}, final={final_energy}"
