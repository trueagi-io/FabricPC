#!/usr/bin/env python3
"""
Extended test suite for FabricPC-JAX to match coverage with PyTorch version.

This file adds tests that are present in the PyTorch version but missing
in the JAX version, ensuring feature parity and comprehensive testing.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from hypothesis import given, strategies as st, settings

from fabricpc.core.types import NodeState, GraphState
from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SigmoidActivation,
    ReLUActivation,
    TanhActivation,
    IdentityActivation,
)
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference

# Set up JAX
jax.config.update("jax_platform_name", "cpu")


class TestValidation:
    """Test suite for validation and error handling."""

    def test_duplicate_node_names_raise(self):
        """Test that duplicate node names raise an error."""
        x1 = Linear(shape=(2,), activation=SigmoidActivation(), name="x")
        x2 = Linear(shape=(2,), activation=SigmoidActivation(), name="x")

        with pytest.raises(ValueError, match="Duplicate node name"):
            graph(
                nodes=[x1, x2],
                edges=[],
                task_map=TaskMap(),
            )

    def test_self_edge_disallowed(self):
        """Test that self-edges are not allowed."""
        n1 = Linear(shape=(2,), activation=SigmoidActivation(), name="n1")

        with pytest.raises(ValueError, match="Self-edge"):
            graph(
                nodes=[n1],
                edges=[Edge(source=n1, target=n1.slot("in"))],
                task_map=TaskMap(),
            )

    def test_nonexistent_node_in_edge(self):
        """Test that edges referencing non-existent nodes raise an error."""
        a = Linear(shape=(2,), name="a")
        nonexistent = Linear(shape=(2,), name="nonexistent")

        with pytest.raises(ValueError, match="not found|does not exist"):
            graph(
                nodes=[a],  # Only 'a' is in the nodes list
                edges=[
                    Edge(source=a, target=nonexistent.slot("in"))
                ],  # But edge references 'nonexistent'
                task_map=TaskMap(),
            )


class TestShapeConsistency:
    """Test suite for shape validation and consistency."""

    @pytest.fixture
    def simple_chain(self):
        """Minimal directed chain: a -> b."""
        a = Linear(shape=(4,), activation=SigmoidActivation(), name="a")
        b = Linear(shape=(3,), activation=SigmoidActivation(), name="b")

        structure = graph(
            nodes=[a, b],
            edges=[Edge(source=a, target=b.slot("in"))],
            task_map=TaskMap(x=a, y=b),
        )
        return structure

    def test_allocate_and_tensor_shapes(self, simple_chain):
        """Test that allocated tensors have correct shapes."""
        structure = simple_chain
        rng_key = jax.random.PRNGKey(42)
        params = initialize_params(structure, rng_key)

        batch_size = 5

        # Create dummy clamps
        x_data = jnp.zeros((batch_size, 4))
        y_data = jnp.zeros((batch_size, 3))
        clamps = {"a": x_data, "b": y_data}

        # Initialize state
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )

        # Check shapes for each node
        for node_name, node in structure.nodes.items():
            node_state = state.nodes[node_name]
            expected_shape = (batch_size, *node.node_info.shape)

            assert (
                node_state.z_latent.shape == expected_shape
            ), f"z_latent shape mismatch for {node_name}"
            assert (
                node_state.error.shape == expected_shape
            ), f"error shape mismatch for {node_name}"
            assert (
                node_state.z_mu.shape == expected_shape
            ), f"z_mu shape mismatch for {node_name}"
            assert (
                node_state.pre_activation.shape == expected_shape
            ), f"pre_activation shape mismatch for {node_name}"

    def test_projection_shapes_match(self, simple_chain):
        """Test that projections maintain correct shapes."""
        structure = simple_chain
        rng_key = jax.random.PRNGKey(42)
        params = initialize_params(structure, rng_key)

        batch_size = 3
        x = jax.random.normal(rng_key, (batch_size, 4))
        y = jax.random.normal(rng_key, (batch_size, 3))
        clamps = {"a": x, "b": y}

        # Initialize and run one inference step
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        state = run_inference(
            params, state, clamps, structure, infer_steps=1, eta_infer=0.1
        )

        # After projection, z_mu for node b should be (batch_size, 3)
        assert state.nodes["b"].z_mu.shape == (
            batch_size,
            3,
        ), "z_mu shape incorrect after projection"


class TestPropertyBased:
    """Property-based testing using Hypothesis."""

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        dim_a=st.integers(min_value=2, max_value=6),
        dim_b=st.integers(min_value=2, max_value=6),
    )
    @settings(deadline=None)
    def test_allocate_respects_batch_and_dims(self, batch_size, dim_a, dim_b):
        """Test that allocation respects arbitrary batch sizes and dimensions."""
        a = Linear(shape=(dim_a,), activation=SigmoidActivation(), name="a")
        b = Linear(shape=(dim_b,), activation=SigmoidActivation(), name="b")

        structure = graph(
            nodes=[a, b],
            edges=[Edge(source=a, target=b.slot("in"))],
            task_map=TaskMap(x=a, y=b),
        )

        rng_key = jax.random.PRNGKey(42)
        params = initialize_params(structure, rng_key)

        # Create dummy clamps
        x_data = jnp.zeros((batch_size, dim_a))
        y_data = jnp.zeros((batch_size, dim_b))
        clamps = {"a": x_data, "b": y_data}

        # Initialize state
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )

        # Verify shapes
        for node_name, node in structure.nodes.items():
            node_state = state.nodes[node_name]
            expected_shape = (batch_size, *node.node_info.shape)

            assert node_state.z_latent.shape == expected_shape
            assert node_state.error.shape == expected_shape
            assert node_state.z_mu.shape == expected_shape
            assert node_state.pre_activation.shape == expected_shape

    @given(
        infer_steps=st.integers(min_value=1, max_value=20),
        eta_infer=st.floats(min_value=0.001, max_value=0.5),
    )
    @settings(deadline=None)
    def test_inference_parameters(self, infer_steps, eta_infer):
        """Test that inference works with various parameter settings."""
        input_node = Linear(shape=(3,), name="input")
        output_node = Linear(shape=(2,), name="output")

        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
        )

        rng_key = jax.random.PRNGKey(42)
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 3))
        y = jax.random.normal(rng_key, (batch_size, 2))
        clamps = {"input": x, "output": y}

        # Initialize state
        initial_state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )

        # Run inference - should not raise
        final_state = run_inference(
            params, initial_state, clamps, structure, infer_steps, eta_infer
        )

        # Verify state is valid
        assert final_state is not None
        assert not jnp.any(jnp.isnan(final_state.nodes["output"].z_mu))


class TestComplexGraphs:
    """Test more complex graph structures."""

    def test_skip_connection_graph(self):
        """Test graph with skip connections."""
        input_node = Linear(shape=(10,), name="input")
        h1 = Linear(shape=(20,), activation=ReLUActivation(), name="h1")
        h2 = Linear(shape=(15,), activation=ReLUActivation(), name="h2")
        output_node = Linear(shape=(5,), name="output")

        structure = graph(
            nodes=[input_node, h1, h2, output_node],
            edges=[
                Edge(source=input_node, target=h1.slot("in")),
                Edge(source=h1, target=h2.slot("in")),
                Edge(source=h2, target=output_node.slot("in")),
                # Skip connection
                Edge(source=input_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
        )

        rng_key = jax.random.PRNGKey(42)
        params = initialize_params(structure, rng_key)

        # Verify skip connection exists
        output_edges = structure.nodes["output"].node_info.in_edges
        assert (
            len(output_edges) == 2
        ), "Output should have 2 incoming edges (including skip)"

        # Test inference runs without issues
        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 10))
        y = jax.random.normal(rng_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps=clamps,
            params=params,
        )

        # Run 1 step to get initial energy (energy is computed during inference)
        state_after_1_step = run_inference(
            params, state, clamps, structure, infer_steps=1, eta_infer=0.1
        )
        final_state = run_inference(
            params, state, clamps, structure, infer_steps=10, eta_infer=0.1
        )

        # Verify convergence (comparing 1 step vs 10 steps)
        initial_energy = sum(
            jnp.sum(state_after_1_step.nodes[name].energy)
            for name in structure.nodes
            if structure.nodes[name].node_info.in_degree > 0
        )
        final_energy = sum(
            jnp.sum(final_state.nodes[name].energy)
            for name in structure.nodes
            if structure.nodes[name].node_info.in_degree > 0
        )
        assert final_energy < initial_energy

    def test_multi_input_node(self):
        """Test node with multiple inputs from different sources."""
        a = Linear(shape=(5,), name="a")
        b = Linear(shape=(4,), name="b")
        c = Linear(shape=(3,), name="c")
        merger = Linear(shape=(6,), name="merger")

        structure = graph(
            nodes=[a, b, c, merger],
            edges=[
                Edge(source=a, target=merger.slot("in")),
                Edge(source=b, target=merger.slot("in")),
                Edge(source=c, target=merger.slot("in")),
            ],
            task_map=TaskMap(x=a, y=merger),
        )

        rng_key = jax.random.PRNGKey(42)
        params = initialize_params(structure, rng_key)

        # Verify merger node has 3 inputs
        assert structure.nodes["merger"].node_info.in_degree == 3
        assert len(structure.nodes["merger"].node_info.in_edges) == 3

        # Verify weights exist for all connections
        merger_params = params.nodes["merger"]
        assert len(merger_params.weights) == 3


class TestEnergyDynamics:
    """Test energy dynamics during inference."""

    @pytest.fixture
    def energy_test_graph(self):
        x = Linear(shape=(5,), name="x")
        h = Linear(shape=(10,), activation=TanhActivation(), name="h")
        y = Linear(shape=(3,), name="y")

        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(input=x, output=y),
        )
        return structure

    def test_energy_monotonic_decrease(self, energy_test_graph):
        """Test that energy decreases monotonically during inference."""
        structure = energy_test_graph
        rng_key = jax.random.PRNGKey(42)
        params = initialize_params(structure, rng_key)

        batch_size = 16
        x = jax.random.normal(rng_key, (batch_size, 5))
        y = jax.random.normal(rng_key, (batch_size, 3))
        clamps = {"x": x, "y": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )

        # Track energy over multiple inference steps
        energies = []
        current_state = state

        for _ in range(10):
            current_state = run_inference(
                params, current_state, clamps, structure, infer_steps=1, eta_infer=0.1
            )
            energy = sum(
                jnp.sum(current_state.nodes[name].energy)
                for name in structure.nodes
                if structure.nodes[name].node_info.in_degree > 0
            )
            energies.append(float(energy))

        # Energy should generally decrease (allowing small fluctuations)
        for i in range(1, len(energies)):
            # Allow small increase due to numerical precision
            assert (
                energies[i] <= energies[i - 1] * 1.01
            ), f"Energy increased significantly at step {i}: {energies[i-1]} -> {energies[i]}"


@pytest.mark.parametrize(
    "node_type", ["LinearNode"]
)  # Add more types as they're implemented
def test_different_node_types(node_type):
    """Test graph construction with different node types."""
    a = Linear(shape=(4,), name="a")
    b = Linear(shape=(3,), name="b")

    structure = graph(
        nodes=[a, b],
        edges=[Edge(source=a, target=b.slot("in"))],
        task_map=TaskMap(x=a, y=b),
    )

    rng_key = jax.random.PRNGKey(42)
    params = initialize_params(structure, rng_key)

    assert structure.nodes["a"].node_info.node_type == node_type
    assert structure.nodes["b"].node_info.node_type == node_type


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_various_batch_sizes(batch_size):
    """Test that the system works with various batch sizes."""
    input_node = Linear(shape=(5,), name="input")
    output_node = Linear(shape=(3,), name="output")

    structure = graph(
        nodes=[input_node, output_node],
        edges=[Edge(source=input_node, target=output_node.slot("in"))],
        task_map=TaskMap(x=input_node, y=output_node),
    )

    rng_key = jax.random.PRNGKey(42)
    params = initialize_params(structure, rng_key)

    x = jax.random.normal(rng_key, (batch_size, 5))
    y = jax.random.normal(rng_key, (batch_size, 3))
    clamps = {"input": x, "output": y}

    state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )
    final_state = run_inference(
        params, state, clamps, structure, infer_steps=5, eta_infer=0.1
    )

    # Verify shapes are maintained
    assert final_state.nodes["input"].z_latent.shape[0] == batch_size
    assert final_state.nodes["output"].z_latent.shape[0] == batch_size
