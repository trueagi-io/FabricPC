"""
Test suite for LinearExplicitGrad gradient computation.

Verifies that LinearExplicitGrad (using JAX autodiff) produces
numerically equivalent gradients to Linear (using manual formulas).
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeState, NodeParams, NodeInfo
from fabricpc.core.inference import run_inference, gather_inputs
from fabricpc.nodes import (
    Linear,
    LinearExplicitGrad,
)
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
)
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.graph.state_initializer import initialize_graph_state

jax.config.update(
    "jax_platform_name", "cpu"
)  # using cuda causes larger numerical differences because of TF32 precision


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def grad_tolerance():
    """Fixture to provide gradient comparison tolerance."""
    return 1e-5


def create_graph(node_class, rng_key):
    """Create a small network using specified node class."""
    input_node = node_class(shape=(8,), name="input")
    hidden = node_class(shape=(12,), activation=TanhActivation(), name="hidden")
    output_node = node_class(shape=(4,), activation=SigmoidActivation(), name="output")

    structure = graph(
        nodes=[input_node, hidden, output_node],
        edges=[
            Edge(source=input_node, target=hidden.slot("in")),
            Edge(source=hidden, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


class TestLinearAutoGradNode:
    """Test that LinearExplicitGrad produces identical gradients to Linear."""

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

        # Map activation strings to instances
        activation_map = {
            "identity": IdentityActivation(),
            "relu": ReLUActivation(),
            "tanh": TanhActivation(),
            "sigmoid": SigmoidActivation(),
        }
        activation_inst = activation_map[activation]

        # Create NodeInfo for Linear
        node_info = NodeInfo(
            name="dst",
            shape=(output_dim,),
            node_type="Linear",
            node_class=Linear,
            node_config={"use_bias": True, "flatten_input": False},
            activation=activation_inst,
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=NormalInitializer(),
            slots={},
            in_degree=1,
            out_degree=0,
            in_edges=(edge_key,),
            out_edges=(),
        )

        # Create NodeInfo for LinearExplicitGrad
        node_info_explicit = NodeInfo(
            name="dst",
            shape=(output_dim,),
            node_type="LinearExplicitGrad",
            node_class=LinearExplicitGrad,
            node_config={"use_bias": True, "flatten_input": False},
            activation=activation_inst,
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=NormalInitializer(),
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
        state_linear, grads_linear = Linear.forward_inference(
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

        # Map activation strings to instances
        activation_map = {
            "identity": IdentityActivation(),
            "relu": ReLUActivation(),
            "tanh": TanhActivation(),
            "sigmoid": SigmoidActivation(),
        }
        activation_inst = activation_map[activation]

        # Create NodeInfo for Linear
        node_info = NodeInfo(
            name="dst",
            shape=(output_dim,),
            node_type="Linear",
            node_class=Linear,
            node_config={"use_bias": True, "flatten_input": False},
            activation=activation_inst,
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=NormalInitializer(),
            slots={},
            in_degree=1,
            out_degree=0,
            in_edges=(edge_key,),
            out_edges=(),
        )

        # Create NodeInfo for LinearExplicitGrad
        node_info_explicit = NodeInfo(
            name="dst",
            shape=(output_dim,),
            node_type="LinearExplicitGrad",
            node_class=LinearExplicitGrad,
            node_config={"use_bias": True, "flatten_input": False},
            activation=activation_inst,
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=NormalInitializer(),
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
        state_linear, grads_linear = Linear.forward_learning(
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
        params_linear, structure_linear = create_graph(Linear, rng_key)
        params_autograd, structure_autograd = create_graph(LinearExplicitGrad, rng_key)

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
            node = structure_linear.nodes[node_name]
            node_info = node.node_info

            # Create NodeInfo for autograd version with same config
            node_info_explicit = NodeInfo(
                name=node_info.name,
                shape=node_info.shape,
                node_type="LinearExplicitGrad",
                node_class=LinearExplicitGrad,
                node_config=node_info.node_config,
                activation=node_info.activation,
                energy=node_info.energy,
                latent_init=node_info.latent_init,
                weight_init=node_info.weight_init,
                slots=node_info.slots,
                in_degree=node_info.in_degree,
                out_degree=node_info.out_degree,
                in_edges=node_info.in_edges,
                out_edges=node_info.out_edges,
            )

            # Gather inputs for gradient computation
            inputs = gather_inputs(node_info, structure_linear, state_linear)

            # Compute input gradients using forward_inference
            _, grads_linear = Linear.forward_inference(
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
    """Test that LinearExplicitGrad is a subclass of Linear."""

    def test_node_class_on_node_info(self):
        """Test that node_class field on NodeInfo works for dispatch."""
        node_info = NodeInfo(
            name="test",
            shape=(4,),
            node_type="LinearExplicitGrad",
            node_class=LinearExplicitGrad,
            node_config={"use_bias": True, "flatten_input": False},
            activation=TanhActivation(),
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=NormalInitializer(),
            slots={},
            in_degree=0,
            out_degree=0,
            in_edges=(),
            out_edges=(),
        )
        assert node_info.node_class is LinearExplicitGrad
        assert issubclass(LinearExplicitGrad, Linear)

    def test_network_creation_with_autograd_nodes(self, rng_key):
        """Test that a network can be created using LinearExplicitGrad nodes."""
        params, structure = create_graph(LinearExplicitGrad, rng_key)

        assert len(structure.nodes) == 3
        for node_name in structure.nodes:
            node = structure.nodes[node_name]
            assert node.node_info.node_type == "LinearExplicitGrad"
