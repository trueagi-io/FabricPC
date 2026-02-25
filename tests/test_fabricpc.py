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
os.environ.setdefault(
    "JAX_TRACEBACK_FILTERING", "off"
)  # set filtering to "off" for better error messages in traceback

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeState, NodeParams, GraphState
from fabricpc.graph.graph_net import (
    create_pc_graph as _create_pc_graph,
    build_graph_structure as _build_graph_structure,
    compute_local_weight_gradients,
)
from fabricpc.graph.state_initializer import (
    initialize_graph_state as _initialize_graph_state,
)
from tests.new_style_graph import graph_kwargs_from_legacy_config, state_init_from_spec
from fabricpc.core.inference import run_inference
from fabricpc.training import train_step
from fabricpc.training.optimizers import create_optimizer

# Set up JAX
jax.config.update("jax_platform_name", "cpu")


def initialize_graph_state(*args, state_init_config=None, **kwargs):
    if isinstance(state_init_config, dict):
        state_init_config = state_init_from_spec(state_init_config)
    return _initialize_graph_state(*args, state_init_config=state_init_config, **kwargs)


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
                "shape": (10,),
                "type": "linear",
                "activation": {"type": "identity"},
            },
            {
                "name": "hidden1",
                "shape": (20,),
                "type": "linear",
                "activation": {"type": "relu"},
                "weight_init": {"type": "xavier"},
            },
            {
                "name": "hidden2",
                "shape": (15,),
                "type": "linear",
                "activation": {"type": "tanh"},
            },
            {
                "name": "output",
                "shape": (5,),
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
    params, structure = _create_pc_graph(
        rng_key=rng_key, **graph_kwargs_from_legacy_config(sample_config)
    )
    return params, structure


class TestGraphConstruction:
    """Test suite for graph construction and validation."""

    def test_graph_construction_with_slots(self, sample_config, rng_key):
        """Test building a graph with slot validation and node classes."""
        params, structure = _create_pc_graph(
            rng_key=rng_key, **graph_kwargs_from_legacy_config(sample_config)
        )

        # Verify structure
        assert len(structure.nodes) == 4, "Should have 4 nodes"
        assert len(structure.edges) == 4, "Should have 4 edges"

        # Check slots
        hidden1_node = structure.nodes["hidden1"]
        assert "in" in hidden1_node.slots, "hidden1 should have 'in' slot"
        assert hidden1_node.slots[
            "in"
        ].is_multi_input, "Linear node slots should be multi-input"

        # Check that output node has two incoming connections
        output_node = structure.nodes["output"]
        assert output_node.in_degree == 2, "Output should have 2 incoming edges"

        # Verify node-based parameters
        assert "hidden1" in params.nodes, "Params should be organized by node"
        node_params = params.nodes["hidden1"]
        assert isinstance(
            node_params, NodeParams
        ), "Node params should be NodeParams object"
        assert "weights" in node_params._fields, "NodeParams should have weights field"
        assert "biases" in node_params._fields, "NodeParams should have biases field"
        assert (
            "input->hidden1:in" in node_params.weights
        ), "Linear node should have weight matrix named by edge key"

    def test_invalid_slot_rejection(self):
        """Test that invalid slot connections are rejected."""
        config = {
            "node_list": [
                {"name": "a", "shape": (10,), "type": "linear"},
                {"name": "b", "shape": (5,), "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "a", "target_name": "b", "slot": "invalid_slot"},
            ],
            "task_map": {"x": "a"},
        }

        with pytest.raises(ValueError, match="non-existent slot"):
            _build_graph_structure(**graph_kwargs_from_legacy_config(config))


class TestInference:
    """Test suite for inference and gradient computation."""

    @pytest.fixture
    def inference_data(self, graph, rng_key):
        """Prepare data for inference tests."""
        params, structure = graph
        batch_size = 32
        input_shape = structure.nodes["input"].shape
        output_shape = structure.nodes["output"].shape

        # Split rng_key for data generation and state initialization
        rng_key, x_key, y_key, state_key = jax.random.split(rng_key, 4)

        # Create dummy data with full shape (batch, *shape)
        x_data = jax.random.normal(x_key, (batch_size, *input_shape))
        y_data = jax.random.normal(y_key, (batch_size, *output_shape))

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
            "y_data": y_data,
        }

    def test_inference_with_local_gradients(self, inference_data):
        """Test inference loop with local Jacobian-based gradients."""
        params = inference_data["params"]
        structure = inference_data["structure"]
        batch_size = inference_data["batch_size"]
        clamps = inference_data["clamps"]
        state_key = inference_data["state_key"]

        # Initialize state with feedforward initialization
        initial_state = initialize_graph_state(
            structure,
            batch_size,
            state_key,
            clamps=clamps,
            state_init_config=structure.config["graph_state_initializer"],
            params=params,
        )

        # Verify that energy field exists and is initialized
        assert "nodes" in initial_state._fields, "GraphState should have nodes field"
        assert isinstance(initial_state.nodes, dict), "nodes should be a dict"

        # Run 1 step to get initial energy (energy is computed during inference, not init)
        eta_infer = 0.1
        state_after_1_step = run_inference(
            params, initial_state, clamps, structure, infer_steps=1, eta_infer=eta_infer
        )

        # Run more steps for final state
        final_state = run_inference(
            params,
            initial_state,
            clamps,
            structure,
            infer_steps=20,
            eta_infer=eta_infer,
        )

        # Verify that latent gradients were computed
        assert "hidden1" in final_state.nodes, "Should have node hidden1"
        assert (
            "latent_grad" in final_state.nodes["hidden1"]._fields
        ), "Should have latent gradients"
        assert final_state.nodes["hidden1"].latent_grad.shape == (
            batch_size,
            20,
        ), "Gradient shape mismatch"

        # Check that energy decreased (comparing 1 step vs 20 steps)
        initial_energy = sum(
            jnp.sum(state_after_1_step.nodes[name].energy)
            for name in structure.nodes
            if structure.nodes[name].in_degree > 0
        )
        final_energy = sum(
            jnp.sum(final_state.nodes[name].energy)
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
        initial_state = initialize_graph_state(
            structure,
            batch_size,
            state_key,
            clamps=clamps,
            state_init_config=structure.config["graph_state_initializer"],
            params=params,
        )
        final_state = run_inference(
            params, initial_state, clamps, structure, infer_steps=10, eta_infer=0.1
        )

        # Compute local gradients
        grads = compute_local_weight_gradients(params, final_state, structure)

        # Verify gradient structure matches params
        assert grads.nodes.keys() == params.nodes.keys(), "Gradient structure mismatch"
        assert (
            grads.nodes.keys() == structure.nodes.keys()
        ), "Gradient structure mismatch"

        # Check that gradients are computed for non-source nodes
        for node_name, node_info in structure.nodes.items():
            if node_info.in_degree > 0:
                node_grads = grads.nodes[node_name]

                # Check if node_grads is a NodeParams
                assert isinstance(
                    node_grads, NodeParams
                ), f"Expected NodeParams for {node_name}"

                # Each key in params should also exist in gradients
                for edge_key in params.nodes[node_name].weights.keys():
                    assert (
                        edge_key in node_grads.weights.keys()
                    ), f"Missing gradient for {edge_key} in {node_name}"
                    weight_grad = node_grads.weights[edge_key]

                    # Verify gradient shape
                    w = params.nodes[node_name].weights[edge_key]
                    assert (
                        weight_grad.shape == w.shape
                    ), f"Gradient shape mismatch for {node_name} edge {edge_key}"


class TestTraining:
    """Test suite for training functionality."""

    def test_complete_training_step(self, graph, rng_key):
        """Test a complete training step with local learning."""
        params, structure = graph
        batch_size = 8
        input_shape = structure.nodes["input"].shape
        output_shape = structure.nodes["output"].shape

        # Split rng_key for data generation
        rng_key, x_key, y_key = jax.random.split(rng_key, 3)

        # Create dummy batch with full shape (batch, *shape)
        batch = {
            "x": jax.random.normal(x_key, (batch_size, *input_shape)),
            "y": jax.random.normal(y_key, (batch_size, *output_shape)),
        }

        # Create optimizer
        optimizer = create_optimizer({"type": "adam", "lr": 0.01})
        opt_state = optimizer.init(params)

        # Run training step
        infer_steps = 10
        eta_infer = 0.1
        new_params, new_opt_state, energy, final_state = train_step(
            params,
            opt_state,
            batch,
            structure,
            optimizer,
            rng_key,
            infer_steps,
            eta_infer,
        )

        # Verify parameters were updated
        for node_name in ["hidden1", "hidden2", "output"]:
            edge_key = next(
                iter(structure.nodes[node_name].in_edges)
            )  # Get one incoming edge
            w_old = params.nodes[node_name].weights[edge_key]
            w_new = new_params.nodes[node_name].weights[edge_key]
            diff = jnp.max(jnp.abs(w_new - w_old))
            assert diff > 0, f"Weights not updated for {node_name}"

        # Verify energy is a valid number
        assert not jnp.isnan(energy), "Energy should not be NaN"
        assert energy > 0, "Energy should be positive"


class TestForwardMethods:
    """Test suite for node forward methods."""

    @pytest.fixture
    def forward_setup(self, graph, rng_key):
        """Setup for forward method tests."""
        params, structure = graph
        batch_size = 4

        # Split rng_key for each node
        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))

        # Create dummy latent states with full shapes (batch, *shape)
        nodes = {}
        for i, (node_name, node_info) in enumerate(structure.nodes.items()):
            full_shape = (batch_size, *node_info.shape)
            nodes[node_name] = NodeState(
                z_latent=jax.random.normal(node_keys[i], full_shape),
                latent_grad=jnp.zeros(full_shape),
                z_mu=jnp.zeros(full_shape),
                error=jnp.zeros(full_shape),
                energy=jnp.zeros((batch_size,)),
                pre_activation=jnp.zeros(full_shape),
                substructure={},
            )

        # Create dummy GraphState
        state = GraphState(nodes=nodes, batch_size=batch_size)

        return params, structure, state

    def test_forward_inference_shapes(self, forward_setup):
        """Test that forward_inference returns correct shapes."""
        params, structure, state = forward_setup

        for node_name, node_info in structure.nodes.items():
            if node_info.in_degree > 0:  # Skip source nodes
                node_class = node_info.node.__class__
                node_state = state.nodes[node_name]

                # Collect edge inputs
                edge_inputs = {}
                for edge_key in node_info.in_edges:
                    in_edge_info = structure.edges[edge_key]
                    edge_inputs[edge_key] = state.nodes[in_edge_info.source].z_latent

                # Run forward_inference
                new_state, input_grads = node_class.forward_inference(
                    params.nodes[node_name],
                    edge_inputs,
                    node_state,
                    node_info,
                    is_clamped=False,
                )

                # Verify state shapes
                assert (
                    new_state.z_mu.shape == node_state.z_latent.shape
                ), f"z_mu shape mismatch for {node_name}"
                assert (
                    new_state.error.shape == node_state.z_latent.shape
                ), f"error shape mismatch for {node_name}"

                # Verify input gradient shapes
                for edge_key in node_info.in_edges:
                    edge_info = structure.edges[edge_key]
                    source_shape = structure.nodes[edge_info.source].shape
                    expected_shape = (state.batch_size, *source_shape)
                    assert (
                        input_grads[edge_key].shape == expected_shape
                    ), f"Input gradient shape mismatch for {edge_key}"

    def test_forward_learning_shapes(self, forward_setup):
        """Test that forward_learning returns correct shapes."""
        params, structure, state = forward_setup

        for node_name, node_info in structure.nodes.items():
            if node_info.in_degree > 0:  # Skip source nodes
                node_class = node_info.node.__class__
                node_state = state.nodes[node_name]
                node_params = params.nodes[node_name]

                # Collect edge inputs
                edge_inputs = {}
                for edge_key in node_info.in_edges:
                    in_edge_info = structure.edges[edge_key]
                    edge_inputs[edge_key] = state.nodes[in_edge_info.source].z_latent

                # Run forward_learning
                new_state, param_grads = node_class.forward_learning(
                    node_params, edge_inputs, node_state, node_info
                )

                # Verify param gradient shapes match original params
                for edge_key in node_params.weights:
                    assert (
                        param_grads.weights[edge_key].shape
                        == node_params.weights[edge_key].shape
                    ), f"Weight gradient shape mismatch for {edge_key}"

                for bias_key in node_params.biases:
                    assert (
                        param_grads.biases[bias_key].shape
                        == node_params.biases[bias_key].shape
                    ), f"Bias gradient shape mismatch for {bias_key}"


@pytest.mark.parametrize("activation_type", ["identity", "relu", "tanh", "sigmoid"])
def test_different_activations(activation_type, rng_key):
    """Test graph construction with different activation functions."""
    config = {
        "node_list": [
            {
                "name": "input",
                "shape": (10,),
                "type": "linear",
                "activation": {"type": "identity"},
            },
            {
                "name": "hidden",
                "shape": (20,),
                "type": "linear",
                "activation": {"type": activation_type},
            },
            {
                "name": "output",
                "shape": (5,),
                "type": "linear",
                "activation": {"type": "identity"},
            },
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }

    params, structure = _create_pc_graph(
        rng_key=rng_key, **graph_kwargs_from_legacy_config(config)
    )
    expected_class_names = {
        "identity": "IdentityActivation",
        "relu": "ReLUActivation",
        "tanh": "TanhActivation",
        "sigmoid": "SigmoidActivation",
    }
    assert (
        structure.nodes["hidden"].activation.__class__.__name__
        == expected_class_names[activation_type]
    )


@pytest.mark.parametrize("weight_init_type", ["uniform", "normal", "xavier", "kaiming"])
def test_different_weight_initializations(weight_init_type, rng_key):
    """Test graph construction with different weight initialization methods."""
    config = {
        "node_list": [
            {"name": "input", "shape": (10,), "type": "linear"},
            {
                "name": "hidden",
                "shape": (20,),
                "type": "linear",
                "weight_init": {"type": weight_init_type},
            },
            {"name": "output", "shape": (5,), "type": "linear"},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }

    params, structure = _create_pc_graph(
        rng_key=rng_key, **graph_kwargs_from_legacy_config(config)
    )

    # Check that weights are initialized (not zero or NaN)
    for edge_key, weight in params.nodes["hidden"].weights.items():
        assert not jnp.allclose(
            weight, 0
        ), f"Weights should not be all zeros for {weight_init_type}"
        assert not jnp.any(
            jnp.isnan(weight)
        ), f"Weights should not contain NaN for {weight_init_type}"
