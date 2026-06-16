#!/usr/bin/env python3
"""
Test that custom nodes defined outside the fabricpc package can be used
in FabricPC graphs.

This test verifies the external custom-node extension contract: an external
package can subclass NodeBase, implement get_slots/initialize_params/forward,
and participate in graph construction, parameter initialization, state
initialization, and forward execution.

The custom node defined here (ScaledSumNode) is intentionally minimal: it
sums its inputs, scales by a learned scalar, and adds a bias. This is enough
to exercise the contract without pulling in dataset dependencies.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import InitializerBase, NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear


class ScaledSumNode(NodeBase):
    """
    A minimal custom node defined outside fabricpc to test the extension contract.

    Computes: output = scale * sum(inputs) + bias
    where scale is a learned scalar and bias is a learned vector.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(std=0.01),
        weight_init=NormalInitializer(std=0.01),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single multi-input slot that accepts arbitrary number of edges."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase],
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Initialize scale (scalar) and bias (vector).

        No per-edge weights since this node sums all inputs directly.
        """
        out_numel = int(np.prod(node_shape))

        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Single scalar scale parameter
        scale = jax.random.normal(subkey1, ()) * 0.1 + 1.0

        # Bias vector
        bias = jax.random.normal(subkey2, (out_numel,)) * 0.01

        # NodeParams expects weights and biases as dicts keyed by edge
        # For a node with no per-edge weights, we use a special key
        weights = {"_scale": scale.reshape((1, 1))}  # Shape (1, 1) for consistency
        biases = {"_bias": bias}

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: scale * sum(inputs) + bias, then compute energy.

        Follows the six required steps from NodeBase.forward docstring.
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        out_numel = int(np.prod(out_shape))

        # Sum all inputs (flatten each to match output shape)
        pre_activation_flat = jnp.zeros((batch_size, out_numel))
        for edge_key, x in inputs.items():
            x_flat = x.reshape(batch_size, -1)
            # Truncate or pad to match output size
            if x_flat.shape[1] >= out_numel:
                x_flat = x_flat[:, :out_numel]
            else:
                x_flat = jnp.pad(
                    x_flat, ((0, 0), (0, out_numel - x_flat.shape[1]))
                )
            pre_activation_flat = pre_activation_flat + x_flat

        # Apply scale and bias
        scale = params.weights["_scale"].squeeze()
        bias = params.biases["_bias"]
        pre_activation_flat = scale * pre_activation_flat + bias

        # Reshape to output shape
        pre_activation = pre_activation_flat.reshape((batch_size,) + out_shape)

        # Step 1: Compute z_mu (prediction) via activation
        activation = node_info.activation
        z_mu = activation.forward(pre_activation, activation.config)

        # Step 2: Record pre_activation
        # Step 3: Compute error
        error = state.z_latent - z_mu

        # Step 4: Update state fields
        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
        )

        # Step 5: Populate energy via energy_functional
        state = ScaledSumNode.energy_functional(state, node_info)

        # Step 6: Return (total_energy, state)
        return jnp.sum(state.energy), state


class TestExternalCustomNode:
    """Test suite for external custom node integration."""

    @pytest.fixture
    def rng_key(self):
        """JAX random key fixture."""
        return jax.random.PRNGKey(42)

    def test_custom_node_instantiation(self):
        """Test that a custom node can be instantiated."""
        node = ScaledSumNode(shape=(10,), name="custom")
        assert node.name == "custom"
        assert node.shape == (10,)

    def test_custom_node_slots(self):
        """Test that get_slots returns the expected slot specification."""
        slots = ScaledSumNode.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True

    def test_custom_node_in_graph(self, rng_key):
        """Test that a custom node can be placed in a FabricPC graph."""
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(shape=(8,), name="custom")
        output_node = Linear(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, custom_node, output_node],
            edges=[
                Edge(source=input_node, target=custom_node.slot("in")),
                Edge(source=custom_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        assert len(structure.nodes) == 3
        assert "custom" in structure.nodes
        assert structure.nodes["custom"].node_info.in_degree == 1
        assert structure.nodes["custom"].node_info.out_degree == 1

    def test_custom_node_params_initialize(self, rng_key):
        """Test that parameters initialize correctly for a custom node."""
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(shape=(8,), name="custom")

        structure = graph(
            nodes=[input_node, custom_node],
            edges=[
                Edge(source=input_node, target=custom_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        assert "custom" in params.nodes
        custom_params = params.nodes["custom"]
        assert "_scale" in custom_params.weights
        assert "_bias" in custom_params.biases
        assert custom_params.weights["_scale"].shape == (1, 1)
        assert custom_params.biases["_bias"].shape == (8,)

    def test_custom_node_state_initialize(self, rng_key):
        """Test that state initializes correctly for a custom node."""
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(shape=(8,), name="custom")

        structure = graph(
            nodes=[input_node, custom_node],
            edges=[
                Edge(source=input_node, target=custom_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        x_data = jax.random.normal(rng_key, (batch_size, 8))
        clamps = {"input": x_data}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )

        assert "custom" in state.nodes
        custom_state = state.nodes["custom"]
        assert custom_state.z_latent.shape == (batch_size, 8)
        assert custom_state.z_mu.shape == (batch_size, 8)

    def test_custom_node_forward_pass(self, rng_key):
        """Test that forward pass works through a custom node."""
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(shape=(8,), name="custom")
        output_node = Linear(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, custom_node, output_node],
            edges=[
                Edge(source=input_node, target=custom_node.slot("in")),
                Edge(source=custom_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2, key3 = jax.random.split(rng_key, 3)
        x_data = jax.random.normal(key1, (batch_size, 8))
        y_data = jax.random.normal(key2, (batch_size, 4))
        clamps = {"input": x_data, "output": y_data}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key3,
            clamps=clamps,
            params=params,
        )

        # Run forward on the custom node directly
        custom_node_final = structure.nodes["custom"]
        custom_params = params.nodes["custom"]
        custom_state = state.nodes["custom"]
        node_info = custom_node_final.node_info

        # Get input from the input node
        input_state = state.nodes["input"]
        edge_key = "input->custom:in"
        inputs = {edge_key: input_state.z_latent}

        # Run forward
        total_energy, new_state = ScaledSumNode.forward(
            custom_params, inputs, custom_state, node_info
        )

        # Verify outputs
        assert jnp.isfinite(total_energy)
        assert new_state.z_mu.shape == (batch_size, 8)
        assert new_state.error.shape == (batch_size, 8)
        # Energy is summed over the last axis by GaussianEnergy, so shape is (batch_size,)
        assert new_state.energy.shape == (batch_size,)

    def test_custom_node_multiple_inputs(self, rng_key):
        """Test that a custom node correctly handles multiple input edges."""
        input1 = Linear(shape=(8,), name="input1")
        input2 = Linear(shape=(8,), name="input2")
        custom_node = ScaledSumNode(shape=(8,), name="custom")

        structure = graph(
            nodes=[input1, input2, custom_node],
            edges=[
                Edge(source=input1, target=custom_node.slot("in")),
                Edge(source=input2, target=custom_node.slot("in")),
            ],
            task_map=TaskMap(x=input1),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        assert "custom" in params.nodes
        custom_node_final = structure.nodes["custom"]
        assert custom_node_final.node_info.in_degree == 2

    def test_custom_node_with_activation(self, rng_key):
        """Test that custom nodes work with non-identity activations."""
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(
            shape=(8,),
            name="custom",
            activation=ReLUActivation(),
        )

        structure = graph(
            nodes=[input_node, custom_node],
            edges=[
                Edge(source=input_node, target=custom_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)
        x_data = jax.random.normal(key1, (batch_size, 8))
        clamps = {"input": x_data}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        # Verify activation is stored in node_info
        custom_node_final = structure.nodes["custom"]
        assert isinstance(custom_node_final.node_info.activation, ReLUActivation)

    def test_custom_node_is_jit_compatible(self, rng_key):
        """Test that custom node forward pass can be JIT compiled."""
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(shape=(8,), name="custom")

        structure = graph(
            nodes=[input_node, custom_node],
            edges=[
                Edge(source=input_node, target=custom_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)
        x_data = jax.random.normal(key1, (batch_size, 8))
        clamps = {"input": x_data}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        custom_node_final = structure.nodes["custom"]
        custom_params = params.nodes["custom"]
        custom_state = state.nodes["custom"]
        node_info = custom_node_final.node_info

        input_state = state.nodes["input"]
        edge_key = "input->custom:in"
        inputs = {edge_key: input_state.z_latent}

        # JIT compile the forward pass
        @jax.jit
        def jitted_forward(p, i, s):
            return ScaledSumNode.forward(p, i, s, node_info)

        total_energy, new_state = jitted_forward(custom_params, inputs, custom_state)

        assert jnp.isfinite(total_energy)
        assert new_state.z_mu.shape == (batch_size, 8)

    def test_custom_node_gradients_compute(self, rng_key):
        """Test that gradients can be computed through a custom node."""
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(shape=(8,), name="custom")

        structure = graph(
            nodes=[input_node, custom_node],
            edges=[
                Edge(source=input_node, target=custom_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        key1, key2 = jax.random.split(rng_key)
        x_data = jax.random.normal(key1, (batch_size, 8))
        clamps = {"input": x_data}

        state = initialize_graph_state(
            structure=structure,
            batch_size=batch_size,
            rng_key=key2,
            clamps=clamps,
            params=params,
        )

        custom_node_final = structure.nodes["custom"]
        custom_params = params.nodes["custom"]
        custom_state = state.nodes["custom"]
        node_info = custom_node_final.node_info

        input_state = state.nodes["input"]
        edge_key = "input->custom:in"
        inputs = {edge_key: input_state.z_latent}

        # Compute gradients via autodiff
        def loss_fn(p):
            energy, _ = ScaledSumNode.forward(p, inputs, custom_state, node_info)
            return energy

        grads = jax.grad(loss_fn)(custom_params)

        assert "_scale" in grads.weights
        assert "_bias" in grads.biases
        assert jnp.isfinite(grads.weights["_scale"]).all()
        assert jnp.isfinite(grads.biases["_bias"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
