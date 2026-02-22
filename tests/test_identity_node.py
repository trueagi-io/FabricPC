"""
Test suite for the IdentityNode implementation.

The IdentityNode passes input through unchanged with no transformation and no
learnable parameters. This is useful for input nodes or auxiliary nodes that
serve as conduits for data without learning.

This test suite verifies:
1. Node registration and discovery
2. Passthrough behavior (input == output)
3. No learnable parameters
4. Graph construction and integration
5. Inference with identity nodes
6. Multi-dimensional input shapes
"""

"""
adding enough so as to invoke pre-commit checks which got skipped
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.registry import (
    get_node_class,
    list_node_types,
)
from fabricpc.core.types import NodeParams, NodeState, NodeInfo, GraphState
from fabricpc.graph.graph_net import create_pc_graph
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


class TestIdentityNodeRegistration:
    """Test that IdentityNode is properly registered."""

    def test_identity_node_registered(self):
        """Test that identity node type is registered."""
        types = list_node_types()
        assert "identity" in types, "identity node type should be registered"

    def test_get_identity_node_class(self):
        """Test that get_node_class returns the IdentityNode class."""
        from fabricpc.nodes.identity import IdentityNode

        assert get_node_class("identity") is IdentityNode

    def test_identity_node_case_insensitive(self):
        """Test that identity node lookup is case-insensitive."""
        from fabricpc.nodes.identity import IdentityNode

        assert get_node_class("IDENTITY") is IdentityNode
        assert get_node_class("Identity") is IdentityNode


class TestIdentityNodeInterface:
    """Test that IdentityNode implements the required interface."""

    def test_has_config_schema(self):
        """Test that IdentityNode has CONFIG_SCHEMA defined."""
        from fabricpc.nodes.identity import IdentityNode

        assert hasattr(IdentityNode, "CONFIG_SCHEMA")
        assert isinstance(IdentityNode.CONFIG_SCHEMA, dict)

    def test_has_slots(self):
        """Test that IdentityNode defines slots."""
        from fabricpc.nodes.identity import IdentityNode

        slots = IdentityNode.get_slots()
        assert isinstance(slots, dict)
        assert "in" in slots, "IdentityNode should have an 'in' slot"
        assert isinstance(slots["in"], SlotSpec)

    def test_has_initialize_params(self):
        """Test that IdentityNode has initialize_params method."""
        from fabricpc.nodes.identity import IdentityNode

        assert hasattr(IdentityNode, "initialize_params")
        assert callable(IdentityNode.initialize_params)

    def test_has_forward(self):
        """Test that IdentityNode has forward method."""
        from fabricpc.nodes.identity import IdentityNode

        assert hasattr(IdentityNode, "forward")
        assert callable(IdentityNode.forward)


class TestIdentityNodeNoParameters:
    """Test that IdentityNode has no learnable parameters."""

    def test_initialize_params_empty_weights(self, rng_key):
        """Test that initialize_params returns empty weights."""
        from fabricpc.nodes.identity import IdentityNode

        node_shape = (10,)
        input_shapes = {"source->target:in": (8,)}
        config = {}

        params = IdentityNode.initialize_params(
            rng_key, node_shape, input_shapes, config
        )

        assert isinstance(params, NodeParams)
        assert len(params.weights) == 0, "IdentityNode should have no weights"
        assert len(params.biases) == 0, "IdentityNode should have no biases"

    def test_initialize_params_multiple_inputs_empty(self, rng_key):
        """Test that params are empty even with multiple inputs."""
        from fabricpc.nodes.identity import IdentityNode

        node_shape = (20,)
        input_shapes = {
            "a->target:in": (10,),
            "b->target:in": (15,),
        }
        config = {}

        params = IdentityNode.initialize_params(
            rng_key, node_shape, input_shapes, config
        )

        assert len(params.weights) == 0
        assert len(params.biases) == 0


class TestIdentityNodeForward:
    """Test the forward pass of IdentityNode."""

    @pytest.fixture
    def identity_node_setup(self, rng_key):
        """Setup for identity node forward tests."""
        from fabricpc.nodes.identity import IdentityNode

        batch_size = 4
        node_shape = (10,)
        full_shape = (batch_size, *node_shape)

        # Create a mock NodeInfo
        node_info = NodeInfo(
            name="test_identity",
            shape=node_shape,
            node_type="identity",
            node_config={
                "name": "test_identity",
                "shape": node_shape,
                "type": "identity",
                "energy": {"type": "gaussian"},
                "activation": {"type": "identity"},
            },
            slots={"in": None},  # Simplified for testing
            in_degree=1,
            out_degree=0,
            in_edges=("source->test_identity:in",),
            out_edges=(),
        )

        # Create initial state
        k1, k2 = jax.random.split(rng_key)
        state = NodeState(
            z_latent=jax.random.normal(k1, full_shape),
            z_mu=jnp.zeros(full_shape),
            error=jnp.zeros(full_shape),
            energy=jnp.zeros((batch_size,)),
            pre_activation=jnp.zeros(full_shape),
            latent_grad=jnp.zeros(full_shape),
            substructure={},
        )

        # Create input data (what the identity node receives)
        input_data = jax.random.normal(k2, full_shape)
        inputs = {"source->test_identity:in": input_data}

        # Empty params for identity node
        params = NodeParams(weights={}, biases={})

        return {
            "node_info": node_info,
            "state": state,
            "inputs": inputs,
            "params": params,
            "input_data": input_data,
            "batch_size": batch_size,
        }

    def test_forward_returns_energy_and_state(self, identity_node_setup):
        """Test that forward returns (energy, state) tuple."""
        from fabricpc.nodes.identity import IdentityNode

        setup = identity_node_setup
        energy, new_state = IdentityNode.forward(
            setup["params"],
            setup["inputs"],
            setup["state"],
            setup["node_info"],
        )

        assert isinstance(energy, jnp.ndarray), "Energy should be a JAX array"
        assert energy.shape == (), "Energy should be a scalar"
        assert isinstance(new_state, NodeState), "Should return NodeState"

    def test_forward_passthrough_single_input(self, identity_node_setup):
        """Test that single input passes through unchanged as z_mu."""
        from fabricpc.nodes.identity import IdentityNode

        setup = identity_node_setup
        _, new_state = IdentityNode.forward(
            setup["params"],
            setup["inputs"],
            setup["state"],
            setup["node_info"],
        )

        # z_mu should equal the input
        np.testing.assert_allclose(
            new_state.z_mu,
            setup["input_data"],
            rtol=1e-5,
            err_msg="z_mu should equal input data for identity node",
        )

    def test_forward_computes_error(self, identity_node_setup):
        """Test that error is computed correctly (z_latent - z_mu)."""
        from fabricpc.nodes.identity import IdentityNode

        setup = identity_node_setup
        _, new_state = IdentityNode.forward(
            setup["params"],
            setup["inputs"],
            setup["state"],
            setup["node_info"],
        )

        expected_error = setup["state"].z_latent - setup["input_data"]
        np.testing.assert_allclose(
            new_state.error,
            expected_error,
            rtol=1e-5,
            err_msg="Error should be z_latent - z_mu",
        )

    def test_forward_computes_energy(self, identity_node_setup):
        """Test that energy is computed (non-NaN, finite)."""
        from fabricpc.nodes.identity import IdentityNode

        setup = identity_node_setup
        energy, new_state = IdentityNode.forward(
            setup["params"],
            setup["inputs"],
            setup["state"],
            setup["node_info"],
        )

        assert not jnp.isnan(energy), "Energy should not be NaN"
        assert jnp.isfinite(energy), "Energy should be finite"
        # Energy per sample should also be valid
        assert not jnp.any(
            jnp.isnan(new_state.energy)
        ), "Per-sample energy should not be NaN"


class TestIdentityNodeMultipleInputs:
    """Test IdentityNode behavior with multiple inputs (sum)."""

    def test_forward_sums_multiple_inputs(self, rng_key):
        """Test that multiple inputs are summed."""
        from fabricpc.nodes.identity import IdentityNode

        batch_size = 4
        node_shape = (10,)
        full_shape = (batch_size, *node_shape)

        node_info = NodeInfo(
            name="test_identity",
            shape=node_shape,
            node_type="identity",
            node_config={
                "name": "test_identity",
                "shape": node_shape,
                "type": "identity",
                "energy": {"type": "gaussian"},
                "activation": {"type": "identity"},
            },
            slots={"in": None},
            in_degree=2,
            out_degree=0,
            in_edges=("a->test_identity:in", "b->test_identity:in"),
            out_edges=(),
        )

        k1, k2, k3 = jax.random.split(rng_key, 3)
        state = NodeState(
            z_latent=jax.random.normal(k1, full_shape),
            z_mu=jnp.zeros(full_shape),
            error=jnp.zeros(full_shape),
            energy=jnp.zeros((batch_size,)),
            pre_activation=jnp.zeros(full_shape),
            latent_grad=jnp.zeros(full_shape),
            substructure={},
        )

        input_a = jax.random.normal(k2, full_shape)
        input_b = jax.random.normal(k3, full_shape)
        inputs = {
            "a->test_identity:in": input_a,
            "b->test_identity:in": input_b,
        }

        params = NodeParams(weights={}, biases={})

        _, new_state = IdentityNode.forward(params, inputs, state, node_info)

        # z_mu should be the sum of inputs
        expected_z_mu = input_a + input_b
        np.testing.assert_allclose(
            new_state.z_mu,
            expected_z_mu,
            rtol=1e-5,
            err_msg="z_mu should be sum of inputs for identity node",
        )


class TestIdentityNodeInGraph:
    """Integration tests for IdentityNode in graph context."""

    def test_graph_construction_with_identity_input(self, rng_key):
        """Test graph construction using identity node as input."""
        config = {
            "node_list": [
                {"name": "input", "shape": (8,), "type": "identity"},
                {"name": "hidden", "shape": (16,), "type": "linear"},
                {"name": "output", "shape": (4,), "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "hidden", "slot": "in"},
                {"source_name": "hidden", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        params, structure = create_pc_graph(config, rng_key)

        assert "input" in structure.nodes
        assert structure.nodes["input"].node_type == "identity"
        # Identity node should have no params
        assert len(params.nodes["input"].weights) == 0
        assert len(params.nodes["input"].biases) == 0

    def test_graph_construction_identity_between_layers(self, rng_key):
        """Test graph with identity node between linear layers."""
        config = {
            "node_list": [
                {"name": "input", "shape": (8,), "type": "linear"},
                {"name": "passthrough", "shape": (8,), "type": "identity"},
                {"name": "output", "shape": (4,), "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "passthrough", "slot": "in"},
                {"source_name": "passthrough", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        params, structure = create_pc_graph(config, rng_key)

        assert structure.nodes["passthrough"].node_type == "identity"
        assert structure.nodes["passthrough"].in_degree == 1
        assert structure.nodes["passthrough"].out_degree == 1

    def test_inference_with_identity_node(self, rng_key):
        """Test inference with identity node in the graph."""
        config = {
            "node_list": [
                {"name": "input", "shape": (8,), "type": "identity"},
                {"name": "output", "shape": (4,), "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

        params, structure = create_pc_graph(config, rng_key)

        batch_size = 4
        k1, k2, state_key = jax.random.split(rng_key, 3)

        x_data = jax.random.normal(k1, (batch_size, 8))
        y_data = jax.random.normal(k2, (batch_size, 4))

        clamps = {"input": x_data, "output": y_data}

        initial_state = initialize_graph_state(
            structure,
            batch_size,
            state_key,
            clamps=clamps,
            state_init_config=structure.config["graph_state_initializer"],
            params=params,
        )

        # Run inference
        final_state = run_inference(
            params,
            initial_state,
            clamps,
            structure,
            infer_steps=5,
            eta_infer=0.1,
        )

        # Verify inference completed without errors
        assert final_state is not None
        assert "input" in final_state.nodes
        assert "output" in final_state.nodes

        # Input node should have zero error (clamped)
        # Note: For clamped input nodes with in_degree=0, z_mu = z_latent
        assert not jnp.any(jnp.isnan(final_state.nodes["input"].z_latent))
        assert not jnp.any(jnp.isnan(final_state.nodes["output"].z_latent))


class TestIdentityNodeShapes:
    """Test IdentityNode with various input shapes."""

    @pytest.mark.parametrize(
        "node_shape",
        [
            (10,),  # 1D
            (8, 8),  # 2D
            (4, 4, 3),  # 3D (image-like)
            (16, 32),  # 2D sequence-like
        ],
    )
    def test_various_shapes(self, rng_key, node_shape):
        """Test IdentityNode with different output shapes."""
        from fabricpc.nodes.identity import IdentityNode

        batch_size = 2
        full_shape = (batch_size, *node_shape)

        node_info = NodeInfo(
            name="test",
            shape=node_shape,
            node_type="identity",
            node_config={
                "name": "test",
                "shape": node_shape,
                "type": "identity",
                "energy": {"type": "gaussian"},
                "activation": {"type": "identity"},
            },
            slots={"in": None},
            in_degree=1,
            out_degree=0,
            in_edges=("source->test:in",),
            out_edges=(),
        )

        k1, k2 = jax.random.split(rng_key)
        state = NodeState(
            z_latent=jax.random.normal(k1, full_shape),
            z_mu=jnp.zeros(full_shape),
            error=jnp.zeros(full_shape),
            energy=jnp.zeros((batch_size,)),
            pre_activation=jnp.zeros(full_shape),
            latent_grad=jnp.zeros(full_shape),
            substructure={},
        )

        input_data = jax.random.normal(k2, full_shape)
        inputs = {"source->test:in": input_data}
        params = NodeParams(weights={}, biases={})

        energy, new_state = IdentityNode.forward(params, inputs, state, node_info)

        # Verify output shapes
        assert new_state.z_mu.shape == full_shape
        assert new_state.error.shape == full_shape
        assert energy.shape == ()

        # Verify passthrough
        np.testing.assert_allclose(new_state.z_mu, input_data, rtol=1e-5)


class TestIdentityNodeSourceNode:
    """Test IdentityNode behavior when used as source node (no inputs)."""

    def test_source_node_behavior(self, rng_key):
        """Test that identity node as source has z_mu = z_latent."""
        from fabricpc.nodes.identity import IdentityNode

        batch_size = 4
        node_shape = (10,)
        full_shape = (batch_size, *node_shape)

        # Source node has in_degree=0
        node_info = NodeInfo(
            name="source",
            shape=node_shape,
            node_type="identity",
            node_config={
                "name": "source",
                "shape": node_shape,
                "type": "identity",
                "energy": {"type": "gaussian"},
                "activation": {"type": "identity"},
            },
            slots={"in": None},
            in_degree=0,  # Source node
            out_degree=1,
            in_edges=(),
            out_edges=("source->target:in",),
        )

        z_latent = jax.random.normal(rng_key, full_shape)
        state = NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros(full_shape),
            error=jnp.zeros(full_shape),
            energy=jnp.zeros((batch_size,)),
            pre_activation=jnp.zeros(full_shape),
            latent_grad=jnp.zeros(full_shape),
            substructure={},
        )

        inputs = {}  # No inputs for source node
        params = NodeParams(weights={}, biases={})

        energy, new_state = IdentityNode.forward(params, inputs, state, node_info)

        # For source nodes, z_mu should equal z_latent
        np.testing.assert_allclose(
            new_state.z_mu,
            z_latent,
            rtol=1e-5,
            err_msg="Source identity node should have z_mu = z_latent",
        )

        # Error should be zero
        np.testing.assert_allclose(
            new_state.error,
            jnp.zeros_like(z_latent),
            atol=1e-7,
            err_msg="Source identity node should have zero error",
        )


class TestIdentityNodeGradients:
    """Test gradient computation through IdentityNode."""

    def test_forward_inference_gradients(self, rng_key):
        """Test that forward_inference computes gradients correctly."""
        from fabricpc.nodes.identity import IdentityNode

        batch_size = 4
        node_shape = (10,)
        full_shape = (batch_size, *node_shape)

        node_info = NodeInfo(
            name="test",
            shape=node_shape,
            node_type="identity",
            node_config={
                "name": "test",
                "shape": node_shape,
                "type": "identity",
                "energy": {"type": "gaussian"},
                "activation": {"type": "identity"},
            },
            slots={"in": None},
            in_degree=1,
            out_degree=1,
            in_edges=("source->test:in",),
            out_edges=("test->target:in",),
        )

        k1, k2 = jax.random.split(rng_key)
        state = NodeState(
            z_latent=jax.random.normal(k1, full_shape),
            z_mu=jnp.zeros(full_shape),
            error=jnp.zeros(full_shape),
            energy=jnp.zeros((batch_size,)),
            pre_activation=jnp.zeros(full_shape),
            latent_grad=jnp.zeros(full_shape),
            substructure={},
        )

        input_data = jax.random.normal(k2, full_shape)
        inputs = {"source->test:in": input_data}
        params = NodeParams(weights={}, biases={})

        new_state, input_grads = IdentityNode.forward_inference(
            params, inputs, state, node_info, is_clamped=False
        )

        # Check that input gradients are returned
        assert "source->test:in" in input_grads
        assert input_grads["source->test:in"].shape == full_shape

        # Gradients should be finite
        assert jnp.all(jnp.isfinite(input_grads["source->test:in"]))

    def test_forward_learning_no_weight_gradients(self, rng_key):
        """Test that forward_learning returns empty gradients (no weights)."""
        from fabricpc.nodes.identity import IdentityNode

        batch_size = 4
        node_shape = (10,)
        full_shape = (batch_size, *node_shape)

        node_info = NodeInfo(
            name="test",
            shape=node_shape,
            node_type="identity",
            node_config={
                "name": "test",
                "shape": node_shape,
                "type": "identity",
                "energy": {"type": "gaussian"},
                "activation": {"type": "identity"},
            },
            slots={"in": None},
            in_degree=1,
            out_degree=0,
            in_edges=("source->test:in",),
            out_edges=(),
        )

        k1, k2 = jax.random.split(rng_key)
        state = NodeState(
            z_latent=jax.random.normal(k1, full_shape),
            z_mu=jnp.zeros(full_shape),
            error=jnp.zeros(full_shape),
            energy=jnp.zeros((batch_size,)),
            pre_activation=jnp.zeros(full_shape),
            latent_grad=jnp.zeros(full_shape),
            substructure={},
        )

        input_data = jax.random.normal(k2, full_shape)
        inputs = {"source->test:in": input_data}
        params = NodeParams(weights={}, biases={})

        new_state, params_grad = IdentityNode.forward_learning(
            params, inputs, state, node_info
        )

        # Gradients should be empty since there are no parameters
        assert len(params_grad.weights) == 0
        assert len(params_grad.biases) == 0
