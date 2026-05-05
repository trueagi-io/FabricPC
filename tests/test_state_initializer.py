#!/usr/bin/env python3
"""
Test suite for the State Initializer system.

Tests distribution-based init, feedforward init, clamp handling,
and the zero-error invariant of feedforward initialization.
"""

import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear
from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
    GeluActivation,
)
from fabricpc.core.initializers import NormalInitializer
from fabricpc.graph_initialization.state_initializer import (
    GlobalStateInit,
    NodeDistributionStateInit,
    FeedforwardStateInit,
    initialize_graph_state,
)


@pytest.fixture
def simple_graph_structure(rng_key):
    """Simple 3-layer graph structure for testing."""
    input_node = Linear(shape=(784,), name="input")
    hidden_node = Linear(shape=(128,), activation=ReLUActivation(), name="hidden")
    output_node = Linear(shape=(10,), name="output")

    structure = graph(
        nodes=[input_node, hidden_node, output_node],
        edges=[
            Edge(source=input_node, target=hidden_node.slot("in")),
            Edge(source=hidden_node, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
        inference=InferenceSGD(),
    )
    return structure


class TestDistributionStateInit:
    """Test suite for GlobalStateInit."""

    def test_distribution_init_graph_level_config(
        self, simple_graph_structure, rng_key
    ):
        """Test distribution init with graph-level default initializer."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=GlobalStateInit(initializer=NormalInitializer(std=0.1)),
        )

        assert state.batch_size == batch_size
        assert "input" in state.nodes
        assert "hidden" in state.nodes
        assert "output" in state.nodes

        assert state.nodes["input"].z_latent.shape == (batch_size, 784)
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)

        hidden_std = jnp.std(state.nodes["hidden"].z_latent)
        assert hidden_std > 0.05 and hidden_std < 0.2


class TestFeedforwardStateInit:
    """Test suite for FeedforwardStateInit."""

    def test_feedforward_init_requires_params(self, simple_graph_structure, rng_key):
        """Test that feedforward init raises error without params."""
        structure = simple_graph_structure

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        with pytest.raises(ValueError, match="requires params"):
            initialize_graph_state(
                structure,
                batch_size,
                rng_key,
                clamps,
                state_init=FeedforwardStateInit(),
                params=None,
            )

    def test_feedforward_init_with_params(self, simple_graph_structure, rng_key):
        """Test feedforward init propagates through network."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        assert state.batch_size == batch_size
        assert state.nodes["input"].z_latent.shape == (batch_size, 784)
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)

        assert not jnp.allclose(state.nodes["hidden"].z_latent, 0.0)


class TestClampHandling:
    """Test clamp handling in state initialization."""

    def test_distribution_init_respects_clamps(self, simple_graph_structure, rng_key):
        """Test that distribution init respects clamped values."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jnp.ones((batch_size, 784)) * 5.0
        y = jnp.ones((batch_size, 10)) * -3.0
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=NodeDistributionStateInit(),
        )

        assert jnp.allclose(state.nodes["input"].z_latent, x)
        assert jnp.allclose(state.nodes["output"].z_latent, y)


class TestFeedforwardZeroError:
    """Test that feedforward initialization produces zero error at all nodes."""

    def test_feedforward_zero_error_mlp(self, rng_key):
        """Test that feedforward init produces zero error for MLP architecture."""
        input_node = Linear(shape=(32,), name="input")
        h1_node = Linear(shape=(64,), activation=ReLUActivation(), name="h1")
        h2_node = Linear(shape=(32,), activation=ReLUActivation(), name="h2")
        output_node = Linear(shape=(10,), activation=SoftmaxActivation(), name="output")

        structure = graph(
            nodes=[input_node, h1_node, h2_node, output_node],
            edges=[
                Edge(source=input_node, target=h1_node.slot("in")),
                Edge(source=h1_node, target=h2_node.slot("in")),
                Edge(source=h2_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 32))
        clamps = {"input": x}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        for node_name in structure.nodes:
            error = state.nodes[node_name].error
            assert jnp.allclose(
                error, 0.0, atol=1e-6
            ), f"Node {node_name} has non-zero error after feedforward init: max={jnp.max(jnp.abs(error))}"

            if node_name not in clamps:
                z_latent = state.nodes[node_name].z_latent
                z_mu = state.nodes[node_name].z_mu
                assert jnp.allclose(
                    z_latent, z_mu, atol=1e-6
                ), f"Node {node_name}: z_latent != z_mu after feedforward init"

    def test_feedforward_zero_error_transformer(self, rng_key):
        """Test that feedforward init produces zero error for transformer architecture."""
        seq_len = 8
        embed_dim = 16
        vocab_size = 10

        input_node = Linear(
            shape=(seq_len, vocab_size),
            activation=IdentityActivation(),
            name="input",
        )
        embed_node = Linear(
            shape=(seq_len, embed_dim),
            activation=IdentityActivation(),
            name="embed",
        )
        mask_node = Linear(
            shape=(1, seq_len, seq_len),
            activation=IdentityActivation(),
            name="mask",
        )
        transformer_node = TransformerBlock(
            shape=(seq_len, embed_dim),
            num_heads=2,
            ff_dim=32,
            internal_activation=GeluActivation(),
            rope_theta=100.0,
            name="transformer_0",
        )
        output_node = Linear(
            shape=(seq_len, vocab_size),
            activation=SoftmaxActivation(),
            name="output",
        )

        structure = graph(
            nodes=[input_node, embed_node, mask_node, transformer_node, output_node],
            edges=[
                Edge(source=input_node, target=embed_node.slot("in")),
                Edge(source=embed_node, target=transformer_node.slot("in")),
                Edge(source=mask_node, target=transformer_node.slot("mask")),
                Edge(source=transformer_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node, causal_mask=mask_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 2
        x_indices = jax.random.randint(rng_key, (batch_size, seq_len), 0, vocab_size)
        x = jax.nn.one_hot(x_indices, vocab_size)

        causal_mask = jnp.tril(jnp.ones((1, seq_len, seq_len)))
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))

        clamps = {"input": x, "mask": causal_mask}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        for node_name in structure.nodes:
            error = state.nodes[node_name].error
            max_error = jnp.max(jnp.abs(error))
            assert jnp.allclose(
                error, 0.0, atol=1e-5
            ), f"Node {node_name} has non-zero error after feedforward init: max={max_error}"

            if node_name not in clamps:
                z_latent = state.nodes[node_name].z_latent
                z_mu = state.nodes[node_name].z_mu
                assert jnp.allclose(
                    z_latent, z_mu, atol=1e-5
                ), f"Node {node_name}: z_latent != z_mu after feedforward init"

    def test_feedforward_no_change_after_inference_without_output_clamp(self, rng_key):
        """
        Test that inference with no output clamp does not change latent states
        when error is zero after feedforward init.
        """
        input_node = Linear(shape=(16,), name="input")
        hidden_node = Linear(shape=(32,), activation=ReLUActivation(), name="hidden")
        output_node = Linear(shape=(8,), name="output")

        structure = graph(
            nodes=[input_node, hidden_node, output_node],
            edges=[
                Edge(source=input_node, target=hidden_node.slot("in")),
                Edge(source=hidden_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=10),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 2
        x = jax.random.normal(rng_key, (batch_size, 16))
        clamps = {"input": x}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        original_latents = {
            name: state.nodes[name].z_latent for name in structure.nodes
        }

        final_state = run_inference(params, state, clamps, structure)

        for node_name in structure.nodes:
            original = original_latents[node_name]
            final = final_state.nodes[node_name].z_latent
            max_diff = jnp.max(jnp.abs(original - final))
            assert jnp.allclose(
                original, final, atol=1e-5
            ), f"Node {node_name} changed after inference despite zero error: max_diff={max_diff}"
