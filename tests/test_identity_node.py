"""
Test suite for the IdentityNode implementation.

Tests behavior unique to IdentityNode:
- No learnable parameters
- Output (z_mu) equals sum of inputs
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.nodes.identity import IdentityNode
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


def make_node_info(name, node_shape, in_edges, scale=1.0):
    """Helper to create NodeInfo for tests."""
    return NodeInfo(
        name=name,
        shape=node_shape,
        node_type="identity",
        node_class=IdentityNode,
        node_config={
            "name": name,
            "shape": node_shape,
            "type": "identity",
            "scale": scale,
        },
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=None,
        slots={"in": None},
        in_degree=len(in_edges),
        out_degree=0,
        in_edges=tuple(in_edges),
        out_edges=(),
    )


def make_state(rng_key, batch_size, node_shape):
    """Helper to create NodeState for tests."""
    full_shape = (batch_size,) + node_shape
    return NodeState(
        z_latent=jax.random.normal(rng_key, full_shape),
        z_mu=jnp.zeros(full_shape),
        error=jnp.zeros(full_shape),
        energy=jnp.zeros((batch_size,)),
        pre_activation=jnp.zeros(full_shape),
        latent_grad=jnp.zeros(full_shape),
    )


class TestIdentityNode:
    """Test IdentityNode behavior."""

    def test_no_learnable_parameters(self, rng_key):
        """IdentityNode should have no weights or biases."""
        params = IdentityNode.initialize_params(
            rng_key,
            node_shape=(10,),
            input_shapes={"a->b:in": (8,)},
            weight_init=None,
            config={},
        )
        assert len(params.weights) == 0
        assert len(params.biases) == 0

    @pytest.mark.parametrize("num_inputs", [1, 3])
    def test_output_equals_sum_of_inputs(self, rng_key, num_inputs):
        """z_mu should equal sum of inputs.

        Note: source nodes (zero inputs) are never forwarded during inference;
        their state is clamped directly. We only test with >= 1 input.
        """
        batch_size, node_shape = 4, (10,)
        full_shape = (batch_size,) + node_shape

        # Create inputs
        keys = jax.random.split(rng_key, num_inputs + 1)
        edge_keys = [f"src{i}->node:in" for i in range(num_inputs)]
        inputs = {
            k: jax.random.normal(keys[i], full_shape) for i, k in enumerate(edge_keys)
        }

        state = make_state(keys[-1], batch_size, node_shape)
        node_info = make_node_info("node", node_shape, edge_keys)
        params = NodeParams(weights={}, biases={})

        _, new_state = IdentityNode.forward(params, inputs, state, node_info)

        expected = sum(inputs.values())

        np.testing.assert_allclose(new_state.z_mu, expected, rtol=1e-5)
