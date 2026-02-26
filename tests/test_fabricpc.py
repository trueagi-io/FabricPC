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
from fabricpc.graph.graph_net import compute_local_weight_gradients
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference
from fabricpc.training import train_step
from fabricpc.training.optimizers import create_optimizer
from fabricpc.nodes import Linear
from fabricpc.nodes.base import _get_node_class_from_info
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
)
from fabricpc.core.initializers import (
    XavierInitializer,
    UniformInitializer,
    NormalInitializer,
    KaimingInitializer,
)

# Set up JAX
jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_graph_structure(rng_key):
    """Fixture providing a sample graph structure with initialized parameters."""
    # Create nodes
    input_node = Linear(shape=(10,), name="input")
    hidden1 = Linear(
        shape=(20,),
        activation=ReLUActivation(),
        weight_init=XavierInitializer(),
        name="hidden1",
    )
    hidden2 = Linear(shape=(15,), activation=TanhActivation(), name="hidden2")
    output_node = Linear(shape=(5,), activation=SigmoidActivation(), name="output")

    # Build graph structure
    structure = graph(
        nodes=[input_node, hidden1, hidden2, output_node],
        edges=[
            Edge(source=input_node, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output_node.slot("in")),
            Edge(source=hidden1, target=output_node.slot("in")),  # skip connection
        ],
        task_map=TaskMap(x=input_node, y=output_node),
    )

    # Initialize params
    params = initialize_params(structure, rng_key)

    return params, structure


@pytest.fixture
def graph_fixture(sample_graph_structure):
    """Fixture providing a constructed graph with parameters and structure."""
    return sample_graph_structure


class TestGraphConstruction:
    """Test suite for graph construction and validation."""

    def test_graph_construction_with_slots(self, sample_graph_structure):
        """Test building a graph with slot validation and node classes."""
        params, structure = sample_graph_structure

        # Verify structure
        assert len(structure.nodes) == 4, "Should have 4 nodes"
        assert len(structure.edges) == 4, "Should have 4 edges"

        # Check slots
        hidden1_node = structure.nodes["hidden1"]
        assert "in" in hidden1_node.node_info.slots, "hidden1 should have 'in' slot"
        assert hidden1_node.node_info.slots[
            "in"
        ].is_multi_input, "Linear node slots should be multi-input"

        # Check that output node has two incoming connections
        output_node = structure.nodes["output"]
        assert (
            output_node.node_info.in_degree == 2
        ), "Output should have 2 incoming edges"

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
        a_node = Linear(shape=(10,), name="a")
        b_node = Linear(shape=(5,), name="b")

        with pytest.raises(KeyError, match="no slot"):
            b_node.slot("invalid_slot")


class TestInference:
    """Test suite for inference and gradient computation."""

    @pytest.fixture
    def inference_data(self, graph_fixture, rng_key):
        """Prepare data for inference tests."""
        params, structure = graph_fixture
        batch_size = 32
        input_shape = structure.nodes["input"].node_info.shape
        output_shape = structure.nodes["output"].node_info.shape

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
            if structure.nodes[name].node_info.in_degree > 0
        )
        final_energy = sum(
            jnp.sum(final_state.nodes[name].energy)
            for name in structure.nodes
            if structure.nodes[name].node_info.in_degree > 0
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
        for node_name, node in structure.nodes.items():
            if node.node_info.in_degree > 0:
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

    def test_complete_training_step(self, graph_fixture, rng_key):
        """Test a complete training step with local learning."""
        params, structure = graph_fixture
        batch_size = 8
        input_shape = structure.nodes["input"].node_info.shape
        output_shape = structure.nodes["output"].node_info.shape

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
                iter(structure.nodes[node_name].node_info.in_edges)
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
    def forward_setup(self, graph_fixture, rng_key):
        """Setup for forward method tests."""
        params, structure = graph_fixture
        batch_size = 4

        # Split rng_key for each node
        node_names = list(structure.nodes.keys())
        node_keys = jax.random.split(rng_key, len(node_names))

        # Create dummy latent states with full shapes (batch, *shape)
        nodes = {}
        for i, (node_name, node) in enumerate(structure.nodes.items()):
            node_info = node.node_info
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

        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            if node_info.in_degree > 0:  # Skip source nodes
                node_class = _get_node_class_from_info(node_info)
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
                    source_shape = structure.nodes[edge_info.source].node_info.shape
                    expected_shape = (state.batch_size, *source_shape)
                    assert (
                        input_grads[edge_key].shape == expected_shape
                    ), f"Input gradient shape mismatch for {edge_key}"

    def test_forward_learning_shapes(self, forward_setup):
        """Test that forward_learning returns correct shapes."""
        params, structure, state = forward_setup

        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            if node_info.in_degree > 0:  # Skip source nodes
                node_class = _get_node_class_from_info(node_info)
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
    # Map activation type to activation instance
    activation_map = {
        "identity": IdentityActivation(),
        "relu": ReLUActivation(),
        "tanh": TanhActivation(),
        "sigmoid": SigmoidActivation(),
    }

    # Create nodes
    input_node = Linear(shape=(10,), name="input")
    hidden = Linear(
        shape=(20,), activation=activation_map[activation_type], name="hidden"
    )
    output_node = Linear(shape=(5,), name="output")

    # Build graph
    structure = graph(
        nodes=[input_node, hidden, output_node],
        edges=[
            Edge(source=input_node, target=hidden.slot("in")),
            Edge(source=hidden, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
    )

    # Initialize params
    params = initialize_params(structure, rng_key)

    # Verify the activation type by checking the class instance
    hidden_activation = structure.nodes["hidden"].node_info.activation
    assert isinstance(
        hidden_activation, type(activation_map[activation_type])
    ), f"Expected {type(activation_map[activation_type])}, got {type(hidden_activation)}"


@pytest.mark.parametrize("weight_init_type", ["uniform", "normal", "xavier", "kaiming"])
def test_different_weight_initializations(weight_init_type, rng_key):
    """Test graph construction with different weight initialization methods."""
    # Map weight init type to initializer instance
    initializer_map = {
        "uniform": UniformInitializer(),
        "normal": NormalInitializer(),
        "xavier": XavierInitializer(),
        "kaiming": KaimingInitializer(),
    }

    # Create nodes
    input_node = Linear(shape=(10,), name="input")
    hidden = Linear(
        shape=(20,), weight_init=initializer_map[weight_init_type], name="hidden"
    )
    output_node = Linear(shape=(5,), name="output")

    # Build graph
    structure = graph(
        nodes=[input_node, hidden, output_node],
        edges=[
            Edge(source=input_node, target=hidden.slot("in")),
            Edge(source=hidden, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
    )

    # Initialize params
    params = initialize_params(structure, rng_key)

    # Check that weights are initialized (not zero or NaN)
    for edge_key, weight in params.nodes["hidden"].weights.items():
        assert not jnp.allclose(
            weight, 0
        ), f"Weights should not be all zeros for {weight_init_type}"
        assert not jnp.any(
            jnp.isnan(weight)
        ), f"Weights should not contain NaN for {weight_init_type}"
