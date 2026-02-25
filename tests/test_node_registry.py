"""
Test suite for node dispatch and custom node creation.

Tests node class dispatch via _node_class_map, custom node creation
by subclassing NodeBase, and integration with graph construction.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

from fabricpc.nodes.base import (
    NodeBase,
    SlotSpec,
    _register_node_class,
    _get_node_class_from_info,
)
from fabricpc.nodes import (
    LinearNode,
    LinearExplicitGrad,
)
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params

jax.config.update("jax_platform_name", "cpu")


class TestNodeDispatch:
    """Test node class dispatch via _node_class_map."""

    def test_builtin_nodes_registered(self):
        """Test that built-in nodes are registered for dispatch."""
        node_info_linear = NodeInfo(
            name="test",
            shape=(4,),
            node_type="LinearNode",
            node_config={},
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            slots={},
            in_degree=0,
            out_degree=0,
            in_edges=(),
            out_edges=(),
        )
        assert _get_node_class_from_info(node_info_linear) is LinearNode

    def test_explicit_grad_node_registered(self):
        """Test that LinearExplicitGrad is registered for dispatch."""
        node_info = NodeInfo(
            name="test",
            shape=(4,),
            node_type="LinearExplicitGrad",
            node_config={},
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            slots={},
            in_degree=0,
            out_degree=0,
            in_edges=(),
            out_edges=(),
        )
        assert _get_node_class_from_info(node_info) is LinearExplicitGrad

    def test_unknown_node_type_raises(self):
        """Test that unknown node type raises ValueError."""
        node_info = NodeInfo(
            name="test",
            shape=(4,),
            node_type="NonexistentNode",
            node_config={},
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            slots={},
            in_degree=0,
            out_degree=0,
            in_edges=(),
            out_edges=(),
        )
        with pytest.raises(ValueError, match="Unknown node type"):
            _get_node_class_from_info(node_info)


class TestCustomNodeCreation:
    """Test creating custom node types by subclassing NodeBase."""

    def test_custom_node_subclass(self):
        """Test creating a custom node type via subclassing."""

        class ConstantNode(NodeBase):
            DEFAULT_ACTIVATION = IdentityActivation
            DEFAULT_ENERGY = GaussianEnergy
            DEFAULT_LATENT_INIT = NormalInitializer

            def __init__(self, shape, name, value=1.0, **kwargs):
                super().__init__(shape=shape, name=name, value=value, **kwargs)

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                value = node_info.node_config.get("value", 1.0)
                z_mu = jnp.full_like(state.z_latent, value)
                error = state.z_latent - z_mu
                state = state._replace(
                    z_mu=z_mu,
                    pre_activation=z_mu,
                    error=error,
                )
                state = node_class.energy_functional(state, node_info)
                total_energy = jnp.sum(state.energy)
                return total_energy, state

        # Register for dispatch
        _register_node_class(ConstantNode)

        # Verify dispatch works
        node_info = NodeInfo(
            name="test",
            shape=(4,),
            node_type="ConstantNode",
            node_config={"value": 2.0},
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            slots={},
            in_degree=0,
            out_degree=0,
            in_edges=(),
            out_edges=(),
        )
        assert _get_node_class_from_info(node_info) is ConstantNode

    def test_custom_node_in_graph(self):
        """Test using a custom node in a graph."""

        class PassthroughNode(NodeBase):
            DEFAULT_ACTIVATION = IdentityActivation
            DEFAULT_ENERGY = GaussianEnergy
            DEFAULT_LATENT_INIT = NormalInitializer

            def __init__(self, shape, name, **kwargs):
                super().__init__(shape=shape, name=name, **kwargs)

            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, config):
                return NodeParams(weights={}, biases={})

            @staticmethod
            def forward(params, inputs, state, node_info):
                # Sum all inputs as z_mu
                z_mu = sum(inputs.values())
                error = state.z_latent - z_mu
                state = state._replace(
                    z_mu=z_mu,
                    pre_activation=z_mu,
                    error=error,
                )
                state = node_class.energy_functional(state, node_info)
                total_energy = jnp.sum(state.energy)
                return total_energy, state

        # Register for dispatch
        _register_node_class(PassthroughNode)

        # Use in a graph
        from fabricpc.nodes import Linear

        input_node = Linear(shape=(8,), name="input")
        passthrough = PassthroughNode(shape=(8,), name="passthrough")
        output_node = Linear(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, passthrough, output_node],
            edges=[
                Edge(source=input_node, target=passthrough.slot("in")),
                Edge(source=passthrough, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
        )

        key = jax.random.PRNGKey(0)
        params = initialize_params(structure, key)

        assert len(structure.nodes) == 3
        assert structure.nodes["passthrough"].node_info.node_type == "PassthroughNode"


class TestNodeProperties:
    """Test node object properties."""

    def test_node_shape_property(self):
        """Test that node shape property works."""
        from fabricpc.nodes import Linear

        node = Linear(shape=(128,), name="test")
        assert node.shape == (128,)

    def test_node_name_property(self):
        """Test that node name property works."""
        from fabricpc.nodes import Linear

        node = Linear(shape=(10,), name="my_node")
        assert node.name == "my_node"

    def test_node_slot_method(self):
        """Test that node.slot() returns SlotRef."""
        from fabricpc.nodes import Linear

        node = Linear(shape=(10,), name="test")
        slot_ref = node.slot("in")
        assert slot_ref.node is node
        assert slot_ref.slot == "in"

    def test_node_invalid_slot_raises(self):
        """Test that accessing nonexistent slot raises KeyError."""
        from fabricpc.nodes import Linear

        node = Linear(shape=(10,), name="test")
        with pytest.raises(KeyError, match="no slot"):
            node.slot("nonexistent")

    def test_node_info_none_before_graph(self):
        """Test that node_info is None before graph() is called."""
        from fabricpc.nodes import Linear

        node = Linear(shape=(10,), name="test")
        assert node.node_info is None

    def test_node_info_set_after_graph(self):
        """Test that node_info is set after graph() is called."""
        from fabricpc.nodes import Linear

        input_node = Linear(shape=(8,), name="input")
        output_node = Linear(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
        )

        # Finalized nodes in structure should have node_info
        assert structure.nodes["input"].node_info is not None
        assert structure.nodes["output"].node_info is not None
        assert structure.nodes["output"].node_info.in_degree == 1


class TestIntegration:
    """Integration tests with graph construction."""

    def test_graph_creation_with_linear_nodes(self):
        """Test that graphs can be created with Linear nodes."""
        from fabricpc.nodes import Linear

        input_node = Linear(shape=(8,), name="input")
        output_node = Linear(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
        )

        key = jax.random.PRNGKey(0)
        params = initialize_params(structure, key)

        assert len(structure.nodes) == 2
        assert structure.nodes["input"].node_info.node_type == "LinearNode"
        assert structure.nodes["output"].node_info.node_type == "LinearNode"

    def test_graph_creation_with_explicit_grad_nodes(self):
        """Test graph creation with LinearExplicitGrad nodes."""
        input_node = LinearExplicitGrad(shape=(8,), name="input")
        output_node = LinearExplicitGrad(shape=(4,), name="output")

        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
        )

        key = jax.random.PRNGKey(0)
        params = initialize_params(structure, key)

        assert structure.nodes["input"].node_info.node_type == "LinearExplicitGrad"
        assert structure.nodes["output"].node_info.node_type == "LinearExplicitGrad"
