#!/usr/bin/env python3
"""
Test suite for the redesigned FabricPC-JAX implementation with local Hebbian learning.

This test suite verifies:
1. Node class architecture with slots and pure functions
2. Node-based parameter organization
3. Local gradient computation using Jacobian
4. Slot validation for edges
5. Parallel inference capability
"""

import os
# Configure JAX to avoid preallocating all device memory (helps memory monitors)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")  # set filtering to "off" for better error messages in traceback

import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeState, NodeParams, GraphState
from fabricpc.graph.graph_net import create_pc_graph, build_graph_structure, initialize_state
from fabricpc.core.inference import run_inference
from fabricpc.training import train_step, compute_local_weight_gradients
from fabricpc.training.optimizers import create_optimizer
from fabricpc.nodes import get_node_class_from_type

# Set up JAX
jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_config():
    """Fixture providing a sample graph configuration."""
    return {
        "node_list": [
            {
                "name": "input",
                "dim": 10,
                "type": "linear",
                "activation": {"type": "identity"},
            },
            {
                "name": "hidden1",
                "dim": 20,
                "type": "linear",
                "activation": {"type": "relu"},
                "weight_init": {"type": "xavier"},
            },
            {
                "name": "hidden2",
                "dim": 15,
                "type": "linear",
                "activation": {"type": "tanh"},
            },
            {
                "name": "output",
                "dim": 5,
                "type": "linear",
                "activation": {"type": "sigmoid"},
            },
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden1", "slot": "in"},
            {"source_name": "hidden1", "target_name": "hidden2", "slot": "in"},
            {"source_name": "hidden2", "target_name": "output", "slot": "in"},
            # Test skip connection
            {"source_name": "hidden1", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }


@pytest.fixture
def graph(sample_config, rng_key):
    """Fixture providing a constructed graph with parameters and structure."""
    params, structure = create_pc_graph(sample_config, rng_key)
    return params, structure


class TestGraphConstruction:
    """Test suite for graph construction and validation."""

    def test_graph_construction_with_slots(self, sample_config, rng_key):
        """Test building a graph with slot validation and node classes."""
        params, structure = create_pc_graph(sample_config, rng_key)

        # Verify structure
        assert len(structure.nodes) == 4, "Should have 4 nodes"
        assert len(structure.edges) == 4, "Should have 4 edges"

        # Check slots
        hidden1_node = structure.nodes["hidden1"]
        assert "in" in hidden1_node.slots, "hidden1 should have 'in' slot"
        assert hidden1_node.slots["in"].is_multi_input, "Linear node slots should be multi-input"

        # Check that output node has two incoming connections
        output_node = structure.nodes["output"]
        assert output_node.in_degree == 2, "Output should have 2 incoming edges"

        # Verify node-based parameters
        assert "hidden1" in params.nodes, "Params should be organized by node"
        node_params = params.nodes["hidden1"]
        assert isinstance(node_params, NodeParams), "Node params should be NodeParams object"
        assert "weights" in node_params._fields, "NodeParams should have weights field"
        assert "biases" in node_params._fields, "NodeParams should have biases field"
        assert "input->hidden1:in" in node_params.weights, "Linear node should have weight matrix named by edge key"

    def test_invalid_slot_rejection(self):
        """Test that invalid slot connections are rejected."""
        config = {
            "node_list": [
                {"name": "a", "dim": 10, "type": "linear"},
                {"name": "b", "dim": 5, "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "a", "target_name": "b", "slot": "invalid_slot"},
            ],
            "task_map": {"x": "a"},
        }

        with pytest.raises(ValueError, match="non-existent slot"):
            build_graph_structure(config)


class TestInference:
    """Test suite for inference and gradient computation."""

    @pytest.fixture
    def inference_data(self, graph, rng_key):
        """Prepare data for inference tests."""
        params, structure = graph
        batch_size = 32
        input_dim = structure.nodes["input"].dim
        output_dim = structure.nodes["output"].dim

        # Split rng_key for data generation and state initialization
        rng_key, x_key, y_key, state_key = jax.random.split(rng_key, 4)

        # Create dummy data
        x_data = jax.random.normal(x_key, (batch_size, input_dim))
        y_data = jax.random.normal(y_key, (batch_size, output_dim))

        # Create clamps
        clamps = {
            "input": x_data,
            "output": y_data,
        }

        return {
            "params": params,
            "structure": structure,
            "batch_size": batch_size,
            "clamps": clamps,
            "state_key": state_key,
            "x_data": x_data,
            "y_data": y_data
        }

    def test_inference_with_local_gradients(self, inference_data):
        """Test inference loop with local Jacobian-based gradients."""
        params = inference_data["params"]
        structure = inference_data["structure"]
        batch_size = inference_data["batch_size"]
        clamps = inference_data["clamps"]
        state_key = inference_data["state_key"]

        # Initialize state
        initial_state = initialize_state(
            structure, batch_size, state_key, clamps=clamps, params=params
        )

        # Verify that energy field exists and is initialized
        assert "nodes" in initial_state._fields, "GraphState should have nodes field"
        assert isinstance(initial_state.nodes, dict), "nodes should be a dict"

        # Run inference
        infer_steps = 10
        eta_infer = 0.1
        final_state = run_inference(
            params, initial_state, clamps, structure, infer_steps, eta_infer
        )

        # Verify that latent gradients were computed
        assert "hidden1" in final_state.nodes, "Should have node hidden1"
        assert "latent_grad" in final_state.nodes["hidden1"]._fields, "Should have latent gradients"
        assert final_state.nodes["hidden1"].latent_grad.shape == (batch_size, 20), "Gradient shape mismatch"

        # Check that energy decreased
        initial_energy = sum(
            jnp.sum(initial_state.nodes[name].error ** 2)
            for name in structure.nodes
            if structure.nodes[name].in_degree > 0
        )
        final_energy = sum(
            jnp.sum(final_state.nodes[name].error ** 2)
            for name in structure.nodes
            if structure.nodes[name].in_degree > 0
        )

        assert final_energy < initial_energy, "Energy should decrease during inference"
        reduction_percentage = (initial_energy - final_energy) / initial_energy * 100
        assert reduction_percentage > 0, "Should have positive energy reduction"

    def test_local_weight_updates(self, inference_data):
        """Test that weight updates use local gradients."""
        params = inference_data["params"]
        structure = inference_data["structure"]
        batch_size = inference_data["batch_size"]
        clamps = inference_data["clamps"]
        state_key = inference_data["state_key"]

        # Initialize and run inference
        initial_state = initialize_state(
            structure, batch_size, state_key, clamps=clamps, params=params
        )
        final_state = run_inference(
            params, initial_state, clamps, structure, infer_steps=10, eta_infer=0.1
        )

        # Compute local gradients
        grads = compute_local_weight_gradients(params, final_state, structure)

        # Verify gradient structure matches params
        assert grads.nodes.keys() == params.nodes.keys(), "Gradient structure mismatch"
        assert grads.nodes.keys() == structure.nodes.keys(), "Gradient structure mismatch"

        # Check that gradients are computed for non-source nodes
        for node_name, node_info in structure.nodes.items():
            if node_info.in_degree > 0:
                node_grads = grads.nodes[node_name]

                # Check if node_grads is a NodeParams
                assert isinstance(node_grads, NodeParams), f"Expected NodeParams for {node_name}"

                # Each key in params should also exist in gradients
                for edge_key in params.nodes[node_name].weights.keys():
                    assert edge_key in node_grads.weights.keys(), f"Missing gradient for {edge_key} in {node_name}"
                    weight_grad = node_grads.weights[edge_key]

                    # Verify gradient shape
                    w = params.nodes[node_name].weights[edge_key]
                    assert weight_grad.shape == w.shape, f"Gradient shape mismatch for {node_name} edge {edge_key}"


class TestTraining:
    """Test suite for training functionality."""

    def test_complete_training_step(self, graph, rng_key):
        """Test a complete training step with local learning."""
        params, structure = graph
        batch_size = 8
        input_dim = structure.nodes["input"].dim
        output_dim = structure.nodes["output"].dim

        # Split rng_key for data generation
        rng_key, x_key, y_key = jax.random.split(rng_key, 3)

        # Create dummy batch
        batch = {
            "x": jax.random.normal(x_key, (batch_size, input_dim)),
            "y": jax.random.normal(y_key, (batch_size, output_dim)),
        }

        # Create optimizer
        optimizer = create_optimizer({"type": "adam", "lr": 0.01})
        opt_state = optimizer.init(params)

        # Run training step
        infer_steps = 10
        eta_infer = 0.1
        new_params, new_opt_state, loss, final_state = train_step(
            params, opt_state, batch, structure, optimizer, rng_key, infer_steps, eta_infer
        )

        # Verify parameters were updated
        for node_name in ["hidden1", "hidden2", "output"]:
            edge_key = next(iter(structure.nodes[node_name].in_edges))  # Get one incoming edge
            w_old = params.nodes[node_name].weights[edge_key]
            w_new = new_params.nodes[node_name].weights[edge_key]
            diff = jnp.max(jnp.abs(w_new - w_old))
            assert diff > 0, f"Weights not updated for {node_name}"

        # Verify loss is a valid number
        assert not jnp.isnan(loss), "Loss should not be NaN"
        assert loss > 0, "Loss should be positive"


class TestJacobian:
    """Test suite for Jacobian computation."""

    @pytest.fixture
    def jacobian_setup(self, graph, rng_key):
        """Setup for Jacobian tests."""
        params, structure = graph
        batch_size = 4

        # Split rng_key for each node
        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))

        # Create dummy latent states
        nodes = {}
        for i, (node_name, node_info) in enumerate(structure.nodes.items()):
            nodes[node_name] = NodeState(
                z_latent=jax.random.normal(node_keys[i], (batch_size, node_info.dim)),
                latent_grad=jnp.zeros((batch_size, node_info.dim)),
                z_mu=jnp.zeros((batch_size, node_info.dim)),
                error=jnp.zeros((batch_size, node_info.dim)),
                energy=jnp.zeros(()),
                pre_activation=jnp.zeros((batch_size, node_info.dim)),
                gain_mod_error=jnp.zeros((batch_size, node_info.dim)),
                substructure={}
            )

        # Create dummy GraphState
        state = GraphState(nodes=nodes, batch_size=batch_size)

        return params, structure, state

    def test_jacobian_computation(self, jacobian_setup):
        """Test Jacobian computation for local gradients."""
        params, structure, state = jacobian_setup

        # Test Jacobian for linear nodes
        for node_name, node_info in structure.nodes.items():
            if node_info.in_degree > 0:  # Skip source nodes
                node_class = get_node_class_from_type(node_info.node_type)
                node_state = state.nodes[node_name]

                # Collect edge inputs for Jacobian computation
                edge_inputs = {}
                for edge_key in node_info.in_edges:
                    in_edge_info = structure.edges[edge_key]
                    edge_inputs[edge_key] = state.nodes[in_edge_info.source].z_latent

                # Test Jacobian computation for each edge
                for edge_key in node_info.in_edges:
                    edge_info = structure.edges[edge_key]
                    source_dim = structure.nodes[edge_info.source].dim
                    target_dim = structure.nodes[edge_info.target].dim

                    # Compute Jacobian
                    jacobian = node_class.compute_jacobian_for_edge(
                        edge_key, params.nodes[node_name], edge_inputs, node_state, node_info
                    )

                    # Verify Jacobian shape
                    expected_shape = (state.batch_size, target_dim, state.batch_size, source_dim)
                    assert jacobian.shape == expected_shape, \
                        f"Jacobian shape mismatch for {edge_key}: got {jacobian.shape}, expected {expected_shape}"


@pytest.mark.parametrize("activation_type", ["identity", "relu", "tanh", "sigmoid"])
def test_different_activations(activation_type, rng_key):
    """Test graph construction with different activation functions."""
    config = {
        "node_list": [
            {"name": "input", "dim": 10, "type": "linear", "activation": {"type": "identity"}},
            {"name": "hidden", "dim": 20, "type": "linear", "activation": {"type": activation_type}},
            {"name": "output", "dim": 5, "type": "linear", "activation": {"type": "identity"}},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }

    params, structure = create_pc_graph(config, rng_key)
    assert structure.nodes["hidden"].activation_config["type"] == activation_type


@pytest.mark.parametrize("weight_init_type", ["uniform", "normal", "xavier", "kaiming"])
def test_different_weight_initializations(weight_init_type, rng_key):
    """Test graph construction with different weight initialization methods."""
    config = {
        "node_list": [
            {"name": "input", "dim": 10, "type": "linear"},
            {
                "name": "hidden",
                "dim": 20,
                "type": "linear",
                "weight_init": {"type": weight_init_type}
            },
            {"name": "output", "dim": 5, "type": "linear"},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }

    params, structure = create_pc_graph(config, rng_key)

    # Check that weights are initialized (not zero or NaN)
    for edge_key, weight in params.nodes["hidden"].weights.items():
        assert not jnp.allclose(weight, 0), f"Weights should not be all zeros for {weight_init_type}"
        assert not jnp.any(jnp.isnan(weight)), f"Weights should not contain NaN for {weight_init_type}"