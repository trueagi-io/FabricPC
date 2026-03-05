#!/usr/bin/env python3
"""
Test suite for the State Initializer system.

Tests:
- GlobalStateInit with graph-level config
- NodeDistributionStateInit with node-level override
- FeedforwardStateInit requires params
- FeedforwardStateInit topological propagation
- Clamp handling in both strategies
- Custom state initializer via subclassing
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear
from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
    GeluActivation,
)
from fabricpc.core.initializers import (
    NormalInitializer,
    UniformInitializer,
    ZerosInitializer,
)
from fabricpc.graph.state_initializer import (
    StateInitBase,
    GlobalStateInit,
    NodeDistributionStateInit,
    FeedforwardStateInit,
    initialize_graph_state,
)

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


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

        # Verify state structure
        assert state.batch_size == batch_size
        assert "input" in state.nodes
        assert "hidden" in state.nodes
        assert "output" in state.nodes

        # Verify shapes
        assert state.nodes["input"].z_latent.shape == (batch_size, 784)
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)

        # Hidden node should be initialized with normal distribution
        # (not clamped, so should have non-zero values with std ~0.1)
        hidden_std = jnp.std(state.nodes["hidden"].z_latent)
        assert hidden_std > 0.05 and hidden_std < 0.2

    def test_distribution_init_with_zeros(self, simple_graph_structure, rng_key):
        """Test distribution init with zeros initializer."""
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
            state_init=GlobalStateInit(initializer=ZerosInitializer()),
        )

        # Hidden should be all zeros
        assert jnp.all(state.nodes["hidden"].z_latent == 0.0)

    def test_distribution_init_node_level_override(self, rng_key):
        """Test distribution init with node-level latent_init override."""
        input_node = Linear(shape=(32,), name="input")
        hidden_node = Linear(
            shape=(16,),
            activation=ReLUActivation(),
            latent_init=UniformInitializer(min_val=-1.0, max_val=1.0),
            name="hidden",
        )
        output_node = Linear(shape=(8,), name="output")

        structure = graph(
            nodes=[input_node, hidden_node, output_node],
            edges=[
                Edge(source=input_node, target=hidden_node.slot("in")),
                Edge(source=hidden_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 32))
        y = jax.random.normal(rng_key, (batch_size, 8))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=NodeDistributionStateInit(),
        )

        # Hidden should be uniform(-1, 1) due to node-level override
        assert jnp.all(state.nodes["hidden"].z_latent >= -1.0)
        assert jnp.all(state.nodes["hidden"].z_latent <= 1.0)
        # Should not be all zeros
        assert not jnp.all(state.nodes["hidden"].z_latent == 0.0)


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
                params=None,  # No params provided
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

        # Verify state structure
        assert state.batch_size == batch_size
        assert state.nodes["input"].z_latent.shape == (batch_size, 784)
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)

        # Hidden should have z_latent = z_mu (feedforward propagation)
        # So z_latent should not be zero for a feedforward init
        assert not jnp.allclose(state.nodes["hidden"].z_latent, 0.0)

    def test_feedforward_init_topological_order(self, rng_key):
        """Test feedforward init processes nodes in topological order."""
        # Create a deeper network to test ordering
        input_node = Linear(shape=(32,), name="input")
        h1_node = Linear(shape=(16,), activation=ReLUActivation(), name="h1")
        h2_node = Linear(shape=(16,), activation=ReLUActivation(), name="h2")
        h3_node = Linear(shape=(8,), activation=ReLUActivation(), name="h3")
        output_node = Linear(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, h1_node, h2_node, h3_node, output_node],
            edges=[
                Edge(source=input_node, target=h1_node.slot("in")),
                Edge(source=h1_node, target=h2_node.slot("in")),
                Edge(source=h2_node, target=h3_node.slot("in")),
                Edge(source=h3_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 2
        x = jax.random.normal(rng_key, (batch_size, 32))
        y = jax.random.normal(rng_key, (batch_size, 4))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        # All intermediate nodes should have non-trivial values
        for name in ["h1", "h2", "h3"]:
            assert not jnp.allclose(
                state.nodes[name].z_latent, 0.0
            ), f"Node {name} should have non-zero z_latent after feedforward init"


class TestClampHandling:
    """Test clamp handling in state initialization."""

    def test_distribution_init_respects_clamps(self, simple_graph_structure, rng_key):
        """Test that distribution init respects clamped values."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jnp.ones((batch_size, 784)) * 5.0  # Specific clamped value
        y = jnp.ones((batch_size, 10)) * -3.0  # Specific clamped value
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=NodeDistributionStateInit(),
        )

        # Clamped nodes should have exact clamped values
        assert jnp.allclose(state.nodes["input"].z_latent, x)
        assert jnp.allclose(state.nodes["output"].z_latent, y)

    def test_feedforward_init_respects_clamps(self, simple_graph_structure, rng_key):
        """Test that feedforward init respects clamped values."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jnp.ones((batch_size, 784)) * 2.0
        y = jnp.ones((batch_size, 10)) * -1.0
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        # Clamped nodes should have exact clamped values
        assert jnp.allclose(state.nodes["input"].z_latent, x)
        assert jnp.allclose(state.nodes["output"].z_latent, y)

    def test_partial_clamps(self, simple_graph_structure, rng_key):
        """Test initialization with only some nodes clamped."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jnp.ones((batch_size, 784)) * 3.0
        clamps = {"input": x}  # Only input clamped

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        # Input should be clamped
        assert jnp.allclose(state.nodes["input"].z_latent, x)

        # Other nodes should be initialized but not clamped
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_initialize_graph_state_function(self, simple_graph_structure, rng_key):
        """Test that initialize_graph_state convenience function works correctly."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        # Use the convenience function (defaults to structure config)
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )

        assert state.batch_size == batch_size
        assert "input" in state.nodes
        assert "hidden" in state.nodes
        assert "output" in state.nodes


class TestCustomStateInit:
    """Test custom state initializer via subclassing."""

    def test_custom_state_init_subclass(self, simple_graph_structure, rng_key):
        """Test creating and using a custom state initializer by subclassing."""
        from fabricpc.core.types import GraphState, NodeState

        class ConstantFillStateInit(StateInitBase):
            def __init__(self, fill_value=99.0):
                super().__init__(fill_value=fill_value)

            @staticmethod
            def initialize_state(
                structure, batch_size, rng_key, clamps, config, params=None
            ):
                fill_value = config.get("fill_value", 99.0)
                node_state_dict = {}

                for node_name, node in structure.nodes.items():
                    shape = (batch_size, *node.node_info.shape)

                    if node_name in clamps:
                        z_latent = clamps[node_name]
                    else:
                        z_latent = jnp.full(shape, fill_value)

                    node_state_dict[node_name] = NodeState(
                        z_latent=z_latent,
                        z_mu=jnp.zeros(shape),
                        error=jnp.zeros(shape),
                        energy=jnp.zeros((batch_size,)),
                        pre_activation=jnp.zeros(shape),
                        latent_grad=jnp.zeros(shape),
                    )

                return GraphState(nodes=node_state_dict, batch_size=batch_size)

        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 2
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=ConstantFillStateInit(fill_value=42.0),
        )

        # Hidden should be filled with 42.0
        assert jnp.all(state.nodes["hidden"].z_latent == 42.0)

        # Clamped nodes should have clamped values
        assert jnp.allclose(state.nodes["input"].z_latent, x)
        assert jnp.allclose(state.nodes["output"].z_latent, y)


class TestFeedforwardZeroError:
    """Test that feedforward initialization produces zero error at all nodes."""

    def test_feedforward_zero_error_mlp(self, rng_key):
        """Test that feedforward init produces zero error for MLP architecture."""
        # Create a 4-layer MLP
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
        clamps = {"input": x}  # Only clamp input, not output

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        # Verify that error is zero at all nodes after feedforward init
        for node_name in structure.nodes:
            error = state.nodes[node_name].error
            assert jnp.allclose(
                error, 0.0, atol=1e-6
            ), f"Node {node_name} has non-zero error after feedforward init: max={jnp.max(jnp.abs(error))}"

            # Also verify z_latent == z_mu for non-clamped nodes
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
        # Create one-hot input
        x_indices = jax.random.randint(rng_key, (batch_size, seq_len), 0, vocab_size)
        x = jax.nn.one_hot(x_indices, vocab_size)

        # Create causal mask
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

        # Verify that error is zero at all nodes after feedforward init
        for node_name in structure.nodes:
            error = state.nodes[node_name].error
            max_error = jnp.max(jnp.abs(error))
            assert jnp.allclose(
                error, 0.0, atol=1e-5
            ), f"Node {node_name} has non-zero error after feedforward init: max={max_error}"

            # Also verify z_latent == z_mu for non-clamped nodes
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
        clamps = {"input": x}  # Only clamp input, no output clamp

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        # Save original latent states
        original_latents = {
            name: state.nodes[name].z_latent for name in structure.nodes
        }

        # Run inference with no output clamp
        final_state = type(structure.config["inference"]).run_inference(
            params, state, clamps, structure
        )

        # Latent states should not have changed since error was zero
        for node_name in structure.nodes:
            original = original_latents[node_name]
            final = final_state.nodes[node_name].z_latent
            max_diff = jnp.max(jnp.abs(original - final))
            assert jnp.allclose(
                original, final, atol=1e-5
            ), f"Node {node_name} changed after inference despite zero error: max_diff={max_diff}"


class TestStateInitDeterminism:
    """Test that state initialization is deterministic."""

    def test_distribution_init_deterministic(self, simple_graph_structure, rng_key):
        """Test distribution init is deterministic with same key."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        clamps = {}

        state1 = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=NodeDistributionStateInit(),
        )
        state2 = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=NodeDistributionStateInit(),
        )

        # Should produce identical results
        assert jnp.allclose(
            state1.nodes["hidden"].z_latent, state2.nodes["hidden"].z_latent
        )

    def test_feedforward_init_deterministic(self, simple_graph_structure, rng_key):
        """Test feedforward init is deterministic with same key."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state1 = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )
        state2 = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        # Should produce identical results
        assert jnp.allclose(
            state1.nodes["hidden"].z_latent, state2.nodes["hidden"].z_latent
        )
