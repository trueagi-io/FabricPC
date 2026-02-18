"""
Test suite for LinearExplicitGrad gradient computation.

Verifies that LinearExplicitGrad (using JAX autodiff) produces
numerically equivalent gradients to LinearNode (using manual formulas).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeState, NodeParams, NodeInfo
from fabricpc.graph.graph_net import create_pc_graph
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference, gather_inputs
from fabricpc.nodes import (
    get_node_class,
    LinearNode,
    LinearExplicitGrad,
    validate_node_config,
)

jax.config.update(
    "jax_platform_name", "cpu"
)  # using cuda causes larger numerical differences because of TF32 precision


def make_node_config(node_type: str, activation: str) -> dict:
    """
    Create a properly validated node config with all defaults from the schema.

    Uses validate_node_config to ensure all defaults (e.g., flatten_input, use_bias)
    are populated from the node class's CONFIG_SCHEMA. Also resolves energy and
    activation configs using the node class's default resolvers.
    """
    node_class = get_node_class(node_type)
    raw_config = {
        "name": "test_node",
        "shape": (1,),  # placeholder, will be overridden
        "type": node_type,
        "activation": {"type": activation},
    }
    validated_config = validate_node_config(node_class, raw_config)
    # Resolve energy and activation configs (normally done by from_config)
    validated_config["energy"] = node_class._resolve_energy_config(raw_config)
    validated_config["activation"] = node_class._resolve_activation_config(raw_config)
    return validated_config


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def grad_tolerance():
    """Fixture to provide gradient comparison tolerance."""
    return 1e-5


def create_config(node_type: str):
    """Create a small network config with specified node type."""
    return {
        "node_list": [
            {
                "name": "input",
                "shape": (8,),
                "type": node_type,
                "activation": {"type": "identity"},
            },
            {
                "name": "hidden",
                "shape": (12,),
                "type": node_type,
                "activation": {"type": "tanh"},
            },
            {
                "name": "output",
                "shape": (4,),
                "type": node_type,
                "activation": {"type": "sigmoid"},
            },
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }


class TestLinearAutoGradNode:
    """Test that LinearExplicitGrad produces identical gradients to LinearNode."""

    @pytest.mark.parametrize("activation", ["identity", "relu", "tanh", "sigmoid"])
    def test_forward_inference_equivalence(self, rng_key, activation, grad_tolerance):
        """Test that forward_inference produces equivalent input gradients for different activations."""
        batch_size = 4
        input_dim = 6
        output_dim = 8

        rngkey_weights, rngkey_inputs, rngkey_latent = jax.random.split(rng_key, 3)

        edge_key = "src->dst:in"
        params = NodeParams(
            weights={
                edge_key: jax.random.normal(rngkey_weights, (input_dim, output_dim))
                * 0.1
            },
            biases={"b": jnp.zeros((1, output_dim))},
        )
        inputs = {edge_key: jax.random.normal(rngkey_inputs, (batch_size, input_dim))}

        # Create validated node_config with all defaults from schema
        validated_config = make_node_config("linear", activation)
        node_info = NodeInfo(
            name="dst",
            shape=(output_dim,),
            node_type="linear",
            node_config=validated_config,
            slots={},
            in_degree=1,
            out_degree=0,
            in_edges=(edge_key,),
            out_edges=(),
        )
        # Override node_type for autograd version
        validated_config_explicit = make_node_config("linear_explicit_grad", activation)
        node_info_explicit = NodeInfo(
            name="dst",
            shape=(output_dim,),
            node_type="linear_explicit_grad",
            node_config=validated_config_explicit,
            slots={},
            in_degree=1,
            out_degree=0,
            in_edges=(edge_key,),
            out_edges=(),
        )

        # Create initial node state with random latent
        z_latent = jax.random.normal(rngkey_latent, (batch_size, output_dim))
        node_state = NodeState(
            z_latent=z_latent,
            latent_grad=jnp.zeros((batch_size, output_dim)),
            z_mu=jnp.zeros((batch_size, output_dim)),
            error=jnp.zeros((batch_size, output_dim)),
            energy=jnp.zeros((batch_size,)),
            pre_activation=jnp.zeros((batch_size, output_dim)),
            substructure={},
        )

        # Compare forward_inference results
        state_linear, grads_linear = LinearNode.forward_inference(
            params, inputs, node_state, node_info, is_clamped=True
        )
        state_autograd, grads_autograd = LinearExplicitGrad.forward_inference(
            params, inputs, node_state, node_info_explicit, is_clamped=True
        )

        # Compare input gradients
        for edge_key in grads_linear:
            max_diff = jnp.max(
                jnp.abs(grads_linear[edge_key] - grads_autograd[edge_key])
            )
            assert (
                max_diff < grad_tolerance
            ), f"Input gradient mismatch for activation={activation}, edge={edge_key}: max diff = {max_diff}"

        # Compare state values
        assert jnp.allclose(
            state_linear.z_mu, state_autograd.z_mu, atol=grad_tolerance
        ), f"z_mu mismatch for activation={activation}"
        assert jnp.allclose(
            state_linear.error, state_autograd.error, atol=grad_tolerance
        ), f"error mismatch for activation={activation}"

    @pytest.mark.parametrize("activation", ["identity", "relu", "tanh", "sigmoid"])
    def test_forward_learning_equivalence(self, rng_key, activation, grad_tolerance):
        """Test that forward_learning produces equivalent param gradients for different activations."""
        batch_size = 4
        input_dim = 6
        output_dim = 8

        rngkey_weights, rngkey_inputs, rngkey_latent = jax.random.split(rng_key, 3)

        edge_key = "src->dst:in"
        params = NodeParams(
            weights={
                edge_key: jax.random.normal(rngkey_weights, (input_dim, output_dim))
                * 0.1
            },
            biases={"b": jnp.zeros((1, output_dim))},
        )
        inputs = {edge_key: jax.random.normal(rngkey_inputs, (batch_size, input_dim))}

        # Create validated node_config with all defaults from schema
        validated_config = make_node_config("linear", activation)
        node_info = NodeInfo(
            name="dst",
            shape=(output_dim,),
            node_type="linear",
            node_config=validated_config,
            slots={},
            in_degree=1,
            out_degree=0,
            in_edges=(edge_key,),
            out_edges=(),
        )
        # Override node_type for autograd version
        validated_config_explicit = make_node_config("linear_explicit_grad", activation)
        node_info_explicit = NodeInfo(
            name="dst",
            shape=(output_dim,),
            node_type="linear_explicit_grad",
            node_config=validated_config_explicit,
            slots={},
            in_degree=1,
            out_degree=0,
            in_edges=(edge_key,),
            out_edges=(),
        )

        # Create initial node state with random latent
        z_latent = jax.random.normal(rngkey_latent, (batch_size, output_dim))
        node_state = NodeState(
            z_latent=z_latent,
            latent_grad=jnp.zeros((batch_size, output_dim)),
            z_mu=jnp.zeros((batch_size, output_dim)),
            error=jnp.zeros((batch_size, output_dim)),
            energy=jnp.zeros((batch_size,)),
            pre_activation=jnp.zeros((batch_size, output_dim)),
            substructure={},
        )

        # Compare forward_learning results
        state_linear, grads_linear = LinearNode.forward_learning(
            params, inputs, node_state, node_info
        )
        state_autograd, grads_autograd = LinearExplicitGrad.forward_learning(
            params, inputs, node_state, node_info_explicit
        )

        # Compare weight gradients
        for edge_key in grads_linear.weights:
            max_diff = jnp.max(
                jnp.abs(
                    grads_linear.weights[edge_key] - grads_autograd.weights[edge_key]
                )
            )
            assert (
                max_diff < grad_tolerance
            ), f"Weight gradient mismatch for activation={activation}, edge={edge_key}: max diff = {max_diff}"

        # Compare bias gradients
        for bias_key in grads_linear.biases:
            max_diff = jnp.max(
                jnp.abs(grads_linear.biases[bias_key] - grads_autograd.biases[bias_key])
            )
            assert (
                max_diff < grad_tolerance
            ), f"Bias gradient mismatch for activation={activation}, bias={bias_key}: max diff = {max_diff}"

    def test_gradient_equivalence_full_network(self, rng_key, grad_tolerance):
        """Test gradient equivalence across a full network with inference."""
        batch_size = 8

        # Create two identical networks with different node types
        config_linear = create_config("linear")
        config_autograd = create_config("linear_explicit_grad")

        # Use same key for identical initialization
        params_linear, structure_linear = create_pc_graph(config_linear, rng_key)
        params_autograd, structure_autograd = create_pc_graph(config_autograd, rng_key)

        # Verify params are identical
        for node_name in params_linear.nodes:
            for edge_key in params_linear.nodes[node_name].weights:
                w_linear = params_linear.nodes[node_name].weights[edge_key]
                w_autograd = params_autograd.nodes[node_name].weights[edge_key]
                assert jnp.allclose(
                    w_linear, w_autograd
                ), f"Params differ for {node_name}/{edge_key}"

        # Create identical input/output data
        rngkey_x, rngkey_y, rngkey_state = jax.random.split(rng_key, 3)
        x_data = jax.random.normal(rngkey_x, (batch_size, 8))
        y_data = jax.random.normal(rngkey_y, (batch_size, 4))
        clamps = {"input": x_data, "output": y_data}

        # Initialize states identically
        state_linear = initialize_graph_state(
            structure_linear,
            batch_size,
            rngkey_state,
            clamps=clamps,
            params=params_linear,
        )
        state_autograd = initialize_graph_state(
            structure_autograd,
            batch_size,
            rngkey_state,
            clamps=clamps,
            params=params_autograd,
        )

        # Run inference
        state_linear = run_inference(
            params_linear,
            state_linear,
            clamps,
            structure_linear,
            infer_steps=5,
            eta_infer=0.1,
        )
        state_autograd = run_inference(
            params_autograd,
            state_autograd,
            clamps,
            structure_autograd,
            infer_steps=5,
            eta_infer=0.1,
        )

        # Compare gradients for each non-input node using forward_inference
        for node_name in ["hidden", "output"]:
            node_info = structure_linear.nodes[node_name]
            # Override node_type for autograd version
            info_fields = node_info.__dict__.copy()
            node_info_explicit = NodeInfo(
                **{**info_fields, "node_type": "linear_explicit_grad"}
            )

            # Gather inputs for gradient computation
            inputs = gather_inputs(node_info, structure_linear, state_linear)

            # Compute input gradients using forward_inference
            _, grads_linear = LinearNode.forward_inference(
                params_linear.nodes[node_name],
                inputs,
                state_linear.nodes[node_name],
                node_info,
                is_clamped=(node_name == "output"),  # Clamp output node
            )
            _, grads_autograd = LinearExplicitGrad.forward_inference(
                params_autograd.nodes[node_name],
                inputs,
                state_autograd.nodes[node_name],
                node_info_explicit,
                is_clamped=(node_name == "output"),  # Clamp output node
            )

            # Compare
            for edge_key in grads_linear:
                max_diff = jnp.max(
                    jnp.abs(grads_linear[edge_key] - grads_autograd[edge_key])
                )
                assert (
                    max_diff < grad_tolerance
                ), f"Input gradient mismatch at {node_name} for {edge_key}: max diff = {max_diff}"


class TestLinearAutoGradNodeRegistration:
    """Test that LinearExplicitGrad is properly registered."""

    def test_node_type_registered(self):
        """Test that linear_explicit_grad node type is registered."""
        node_class = get_node_class("linear_explicit_grad")
        assert node_class is LinearExplicitGrad

    def test_network_creation_with_autograd_nodes(self, rng_key):
        """Test that a network can be created using linear_explicit_grad nodes."""
        config = create_config("linear_explicit_grad")
        params, structure = create_pc_graph(config, rng_key)

        assert len(structure.nodes) == 3
        assert all(
            info.node_type == "linear_explicit_grad"
            for info in structure.nodes.values()
        )
