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

import pytest
import jax
import jax.numpy as jnp
from hypothesis import given, strategies as st, settings

from fabricpc.core.types import NodeState, GraphState
from fabricpc.graph.graph_net import create_pc_graph, build_graph_structure, initialize_state
from fabricpc.core.inference import run_inference

# Set up JAX
jax.config.update("jax_platform_name", "cpu")


class TestValidation:
    """Test suite for validation and error handling."""

    def test_duplicate_node_names_raise(self):
        """Test that duplicate node names raise an error."""
        config = {
            "node_list": [
                {
                    "name": "x",
                    "dim": 2,
                    "type": "linear",
                    "activation": {"type": "sigmoid"},
                },
                {
                    "name": "x",  # Duplicate name
                    "dim": 2,
                    "type": "linear",
                    "activation": {"type": "sigmoid"},
                },
            ],
            "edge_list": [],
            "task_map": {},
        }

        with pytest.raises(ValueError, match="duplicate.*node"):
            build_graph_structure(config)

    def test_self_edge_disallowed(self):
        """Test that self-edges are not allowed."""
        config = {
            "node_list": [
                {
                    "name": "n1",
                    "dim": 2,
                    "type": "linear",
                    "activation": {"type": "sigmoid"},
                },
            ],
            "edge_list": [
                {"source_name": "n1", "target_name": "n1", "slot": "in"},  # Self-edge
            ],
            "task_map": {},
        }

        with pytest.raises(ValueError, match="self.*edge|same.*node"):
            build_graph_structure(config)

    def test_nonexistent_node_in_edge(self):
        """Test that edges referencing non-existent nodes raise an error."""
        config = {
            "node_list": [
                {"name": "a", "dim": 2, "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "a", "target_name": "nonexistent", "slot": "in"},
            ],
            "task_map": {},
        }

        with pytest.raises(ValueError, match="not found|does not exist"):
            build_graph_structure(config)


class TestShapeConsistency:
    """Test suite for shape validation and consistency."""

    @pytest.fixture
    def simple_chain_config(self):
        """Minimal directed chain: a -> b."""
        return {
            "node_list": [
                {
                    "name": "a",
                    "dim": 4,
                    "type": "linear",
                    "activation": {"type": "sigmoid"},
                },
                {
                    "name": "b",
                    "dim": 3,
                    "type": "linear",
                    "activation": {"type": "sigmoid"},
                },
            ],
            "edge_list": [
                {"source_name": "a", "target_name": "b", "slot": "in"},
            ],
            "task_map": {"x": "a", "y": "b"},
        }

    def test_allocate_and_tensor_shapes(self, simple_chain_config):
        """Test that allocated tensors have correct shapes."""
        rng_key = jax.random.PRNGKey(42)
        params, structure = create_pc_graph(simple_chain_config, rng_key)

        batch_size = 5

        # Create dummy clamps
        x_data = jnp.zeros((batch_size, 4))
        y_data = jnp.zeros((batch_size, 3))
        clamps = {"a": x_data, "b": y_data}

        # Initialize state
        state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)

        # Check shapes for each node
        for node_name, node_info in structure.nodes.items():
            node_state = state.nodes[node_name]
            expected_shape = (batch_size, node_info.dim)

            assert node_state.z_latent.shape == expected_shape, \
                f"z_latent shape mismatch for {node_name}"
            assert node_state.error.shape == expected_shape, \
                f"error shape mismatch for {node_name}"
            assert node_state.z_mu.shape == expected_shape, \
                f"z_mu shape mismatch for {node_name}"
            assert node_state.pre_activation.shape == expected_shape, \
                f"pre_activation shape mismatch for {node_name}"
            assert node_state.gain_mod_error.shape == expected_shape, \
                f"gain_mod_error shape mismatch for {node_name}"

    def test_projection_shapes_match(self, simple_chain_config):
        """Test that projections maintain correct shapes."""
        rng_key = jax.random.PRNGKey(42)
        params, structure = create_pc_graph(simple_chain_config, rng_key)

        batch_size = 3
        x = jax.random.normal(rng_key, (batch_size, 4))
        y = jax.random.normal(rng_key, (batch_size, 3))
        clamps = {"a": x, "b": y}

        # Initialize and run one inference step
        state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)
        state = run_inference(params, state, clamps, structure, infer_steps=1, eta_infer=0.1)

        # After projection, z_mu for node b should be (batch_size, 3)
        assert state.nodes["b"].z_mu.shape == (batch_size, 3), \
            "z_mu shape incorrect after projection"


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
        config = {
            "node_list": [
                {
                    "name": "a",
                    "dim": dim_a,
                    "type": "linear",
                    "activation": {"type": "sigmoid"},
                },
                {
                    "name": "b",
                    "dim": dim_b,
                    "type": "linear",
                    "activation": {"type": "sigmoid"},
                },
            ],
            "edge_list": [
                {"source_name": "a", "target_name": "b", "slot": "in"},
            ],
            "task_map": {"x": "a", "y": "b"},
        }

        rng_key = jax.random.PRNGKey(42)
        params, structure = create_pc_graph(config, rng_key)

        # Create dummy clamps
        x_data = jnp.zeros((batch_size, dim_a))
        y_data = jnp.zeros((batch_size, dim_b))
        clamps = {"a": x_data, "b": y_data}

        # Initialize state
        state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)

        # Verify shapes
        for node_name, node_info in structure.nodes.items():
            node_state = state.nodes[node_name]
            expected_shape = (batch_size, node_info.dim)

            assert node_state.z_latent.shape == expected_shape
            assert node_state.error.shape == expected_shape
            assert node_state.z_mu.shape == expected_shape
            assert node_state.pre_activation.shape == expected_shape
            assert node_state.gain_mod_error.shape == expected_shape

    @given(
        infer_steps=st.integers(min_value=1, max_value=20),
        eta_infer=st.floats(min_value=0.001, max_value=0.5),
    )
    @settings(deadline=None)
    def test_inference_parameters(self, infer_steps, eta_infer):
        """Test that inference works with various parameter settings."""
        config = {
            "node_list": [
                {"name": "input", "dim": 3, "type": "linear"},
                {"name": "output", "dim": 2, "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        rng_key = jax.random.PRNGKey(42)
        params, structure = create_pc_graph(config, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 3))
        y = jax.random.normal(rng_key, (batch_size, 2))
        clamps = {"input": x, "output": y}

        # Initialize state
        initial_state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)

        # Run inference - should not raise
        final_state = run_inference(
            params, initial_state, clamps, structure, infer_steps, eta_infer
        )

        # Verify state is valid
        assert final_state is not None
        assert not jnp.any(jnp.isnan(final_state.nodes["output"].z_latent))


class TestComplexGraphs:
    """Test more complex graph structures."""

    def test_skip_connection_graph(self):
        """Test graph with skip connections."""
        config = {
            "node_list": [
                {"name": "input", "dim": 10, "type": "linear"},
                {"name": "h1", "dim": 20, "type": "linear", "activation": {"type": "relu"}},
                {"name": "h2", "dim": 15, "type": "linear", "activation": {"type": "relu"}},
                {"name": "output", "dim": 5, "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "h1", "slot": "in"},
                {"source_name": "h1", "target_name": "h2", "slot": "in"},
                {"source_name": "h2", "target_name": "output", "slot": "in"},
                # Skip connection
                {"source_name": "input", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        rng_key = jax.random.PRNGKey(42)
        params, structure = create_pc_graph(config, rng_key)

        # Verify skip connection exists
        output_edges = structure.nodes["output"].in_edges
        assert len(output_edges) == 2, "Output should have 2 incoming edges (including skip)"

        # Test inference runs without issues
        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 10))
        y = jax.random.normal(rng_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)
        final_state = run_inference(params, state, clamps, structure, infer_steps=5, eta_infer=0.1)

        # Verify convergence
        initial_energy = sum(
            jnp.sum(state.nodes[name].error ** 2)
            for name in structure.nodes
            if structure.nodes[name].in_degree > 0
        )
        final_energy = sum(
            jnp.sum(final_state.nodes[name].error ** 2)
            for name in structure.nodes
            if structure.nodes[name].in_degree > 0
        )
        assert final_energy < initial_energy

    def test_multi_input_node(self):
        """Test node with multiple inputs from different sources."""
        config = {
            "node_list": [
                {"name": "a", "dim": 5, "type": "linear"},
                {"name": "b", "dim": 4, "type": "linear"},
                {"name": "c", "dim": 3, "type": "linear"},
                {"name": "merger", "dim": 6, "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "a", "target_name": "merger", "slot": "in"},
                {"source_name": "b", "target_name": "merger", "slot": "in"},
                {"source_name": "c", "target_name": "merger", "slot": "in"},
            ],
            "task_map": {"x": "a", "y": "merger"},
        }

        rng_key = jax.random.PRNGKey(42)
        params, structure = create_pc_graph(config, rng_key)

        # Verify merger node has 3 inputs
        assert structure.nodes["merger"].in_degree == 3
        assert len(structure.nodes["merger"].in_edges) == 3

        # Verify weights exist for all connections
        merger_params = params.nodes["merger"]
        assert len(merger_params.weights) == 3


class TestEnergyDynamics:
    """Test energy dynamics during inference."""

    @pytest.fixture
    def energy_test_config(self):
        return {
            "node_list": [
                {"name": "x", "dim": 5, "type": "linear"},
                {"name": "h", "dim": 10, "type": "linear", "activation": {"type": "tanh"}},
                {"name": "y", "dim": 3, "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "x", "target_name": "h", "slot": "in"},
                {"source_name": "h", "target_name": "y", "slot": "in"},
            ],
            "task_map": {"input": "x", "output": "y"},
        }

    def test_energy_monotonic_decrease(self, energy_test_config):
        """Test that energy decreases monotonically during inference."""
        rng_key = jax.random.PRNGKey(42)
        params, structure = create_pc_graph(energy_test_config, rng_key)

        batch_size = 16
        x = jax.random.normal(rng_key, (batch_size, 5))
        y = jax.random.normal(rng_key, (batch_size, 3))
        clamps = {"x": x, "y": y}

        state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)

        # Track energy over multiple inference steps
        energies = []
        current_state = state

        for _ in range(10):
            current_state = run_inference(
                params, current_state, clamps, structure, infer_steps=1, eta_infer=0.1
            )
            energy = sum(
                jnp.sum(current_state.nodes[name].error ** 2)
                for name in structure.nodes
                if structure.nodes[name].in_degree > 0
            )
            energies.append(float(energy))

        # Energy should generally decrease (allowing small fluctuations)
        for i in range(1, len(energies)):
            # Allow small increase due to numerical precision
            assert energies[i] <= energies[i-1] * 1.01, \
                f"Energy increased significantly at step {i}: {energies[i-1]} -> {energies[i]}"


@pytest.mark.parametrize("node_type", ["linear"])  # Add more types as they're implemented
def test_different_node_types(node_type):
    """Test graph construction with different node types."""
    config = {
        "node_list": [
            {"name": "a", "dim": 4, "type": node_type},
            {"name": "b", "dim": 3, "type": node_type},
        ],
        "edge_list": [
            {"source_name": "a", "target_name": "b", "slot": "in"},
        ],
        "task_map": {"x": "a", "y": "b"},
    }

    rng_key = jax.random.PRNGKey(42)
    params, structure = create_pc_graph(config, rng_key)

    assert structure.nodes["a"].node_type == node_type
    assert structure.nodes["b"].node_type == node_type


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_various_batch_sizes(batch_size):
    """Test that the system works with various batch sizes."""
    config = {
        "node_list": [
            {"name": "input", "dim": 5, "type": "linear"},
            {"name": "output", "dim": 3, "type": "linear"},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }

    rng_key = jax.random.PRNGKey(42)
    params, structure = create_pc_graph(config, rng_key)

    x = jax.random.normal(rng_key, (batch_size, 5))
    y = jax.random.normal(rng_key, (batch_size, 3))
    clamps = {"input": x, "output": y}

    state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)
    final_state = run_inference(params, state, clamps, structure, infer_steps=5, eta_infer=0.1)

    # Verify shapes are maintained
    assert final_state.nodes["input"].z_latent.shape[0] == batch_size
    assert final_state.nodes["output"].z_latent.shape[0] == batch_size