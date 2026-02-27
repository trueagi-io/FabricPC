"""
Test suite for the IdentityNode implementation.

Tests behavior unique to IdentityNode:
- No learnable parameters
- Passthrough behavior (input == z_mu)
- Multiple inputs are summed
- Source node behavior (z_mu = z_latent when no inputs)
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.nodes.identity import IdentityNode
from fabricpc.core.types import NodeParams, NodeState, NodeInfo

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


class TestIdentityNodeNoParameters:
    """Test that IdentityNode has no learnable parameters."""

    def test_initialize_params_empty(self, rng_key):
        """Test that initialize_params returns empty weights and biases."""
        node_shape = (10,)
        input_shapes = {"source->target:in": (8,)}
        config = {}

        params = IdentityNode.initialize_params(
            rng_key, node_shape, input_shapes, config
        )

        assert isinstance(params, NodeParams)
        assert len(params.weights) == 0, "IdentityNode should have no weights"
        assert len(params.biases) == 0, "IdentityNode should have no biases"


class TestIdentityNodePassthrough:
    """Test passthrough behavior."""

    @pytest.fixture
    def node_setup(self, rng_key):
        """Setup for identity node tests."""
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
            in_degree=1,
            out_degree=0,
            in_edges=("source->test_identity:in",),
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
        inputs = {"source->test_identity:in": input_data}
        params = NodeParams(weights={}, biases={})

        return {
            "node_info": node_info,
            "state": state,
            "inputs": inputs,
            "params": params,
            "input_data": input_data,
        }

    def test_single_input_passthrough(self, node_setup):
        """Test that single input passes through unchanged as z_mu."""
        setup = node_setup
        _, new_state = IdentityNode.forward(
            setup["params"],
            setup["inputs"],
            setup["state"],
            setup["node_info"],
        )

        np.testing.assert_allclose(
            new_state.z_mu,
            setup["input_data"],
            rtol=1e-5,
            err_msg="z_mu should equal input data for identity node",
        )


class TestIdentityNodeMultipleInputs:
    """Test IdentityNode behavior with multiple inputs."""

    def test_multiple_inputs_summed(self, rng_key):
        """Test that multiple inputs are summed."""
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

        expected_z_mu = input_a + input_b
        np.testing.assert_allclose(
            new_state.z_mu,
            expected_z_mu,
            rtol=1e-5,
            err_msg="z_mu should be sum of inputs for identity node",
        )


class TestIdentityNodeSourceNode:
    """Test IdentityNode behavior when used as source node (no inputs)."""

    def test_source_node_z_mu_equals_z_latent(self, rng_key):
        """Test that identity node as source has z_mu = z_latent."""
        batch_size = 4
        node_shape = (10,)
        full_shape = (batch_size, *node_shape)

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
            in_degree=0,
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

        inputs = {}
        params = NodeParams(weights={}, biases={})

        _, new_state = IdentityNode.forward(params, inputs, state, node_info)

        np.testing.assert_allclose(
            new_state.z_mu,
            z_latent,
            rtol=1e-5,
            err_msg="Source identity node should have z_mu = z_latent",
        )

        np.testing.assert_allclose(
            new_state.error,
            jnp.zeros_like(z_latent),
            atol=1e-7,
            err_msg="Source identity node should have zero error",
        )


class TestIdentityNodeLearning:
    """Test that IdentityNode produces no weight gradients."""

    def test_forward_learning_empty_gradients(self, rng_key):
        """Test that forward_learning returns empty gradients."""
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

        _, params_grad = IdentityNode.forward_learning(
            params, inputs, state, node_info
        )

        assert len(params_grad.weights) == 0
        assert len(params_grad.biases) == 0
