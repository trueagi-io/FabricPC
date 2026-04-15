"""
Unit tests for native FabricPC continual learning nodes.

Tests the core components:
- CausalGradientRegistry
- TransWeaveRegistry
- CausalLinear
- TransWeaveLinear
- CausalTransWeaveLinear
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ReLUActivation, IdentityActivation
from fabricpc.core.energy import GaussianEnergy

from fabricpc.continual.native_nodes import (
    CausalGradientRegistry,
    TransWeaveRegistry,
    CausalLinear,
    TransWeaveLinear,
    CausalTransWeaveLinear,
    apply_causal_to_gradients,
    register_task_end_for_nodes,
    get_transferred_params,
)
from fabricpc.continual.weight_causal import PerWeightCausalConfig


class TestCausalGradientRegistry:
    """Test CausalGradientRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        CausalGradientRegistry.reset_instance(
            PerWeightCausalConfig(
                gradient_history_size=10,
                min_history_for_detection=5,
            )
        )

    def test_singleton(self):
        """Test singleton pattern."""
        reg1 = CausalGradientRegistry.get_instance()
        reg2 = CausalGradientRegistry.get_instance()
        assert reg1 is reg2

    def test_register_gradient(self):
        """Test gradient registration."""
        registry = CausalGradientRegistry.get_instance()

        gradient = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        registry.register_gradient("node1", "edge1", gradient)

        assert "node1" in registry._histories
        assert "edge1" in registry._histories["node1"]

    def test_get_non_gaussian_mask_insufficient_history(self):
        """Test mask returns None with insufficient history."""
        registry = CausalGradientRegistry.get_instance()

        # Only register 2 gradients (less than min_history_for_detection=5)
        for i in range(2):
            gradient = jnp.array([[float(i), float(i + 1)]])
            registry.register_gradient("node1", "edge1", gradient)

        mask = registry.get_non_gaussian_mask("node1", "edge1")
        assert mask is None

    def test_get_non_gaussian_mask_with_history(self):
        """Test mask computation with sufficient history."""
        registry = CausalGradientRegistry.get_instance()

        # Register 6 gradients (more than min_history_for_detection=5)
        np.random.seed(42)
        for i in range(6):
            gradient = jnp.array(np.random.randn(2, 3))
            registry.register_gradient("node1", "edge1", gradient)

        mask = registry.get_non_gaussian_mask("node1", "edge1")
        assert mask is not None
        assert mask.shape == (2, 3)

    def test_apply_causal_correction(self):
        """Test causal correction application."""
        registry = CausalGradientRegistry.get_instance()

        # Register enough gradients for detection
        np.random.seed(42)
        for i in range(10):
            gradient = jnp.array(np.random.laplace(0, 1, size=(5,)))
            registry.register_gradient("node1", "edge1", gradient)

        # Apply correction
        new_gradient = jnp.array(np.random.laplace(0, 1, size=(5,)))
        corrected = registry.apply_causal_correction("node1", "edge1", new_gradient)

        assert corrected.shape == new_gradient.shape
        # Corrected should potentially be different from original
        # (not a strict test since correction depends on distribution)

    def test_clear(self):
        """Test clearing history."""
        registry = CausalGradientRegistry.get_instance()

        gradient = jnp.array([[1.0, 2.0]])
        registry.register_gradient("node1", "edge1", gradient)
        registry.register_gradient("node2", "edge1", gradient)

        # Clear specific node
        registry.clear("node1")
        assert "node1" not in registry._histories
        assert "node2" in registry._histories

        # Clear all
        registry.clear()
        assert len(registry._histories) == 0

    def test_disabled_config(self):
        """Test that disabled config bypasses everything."""
        CausalGradientRegistry.reset_instance(PerWeightCausalConfig(enable=False))
        registry = CausalGradientRegistry.get_instance()

        gradient = jnp.array([[1.0, 2.0]])
        registry.register_gradient("node1", "edge1", gradient)

        # Should not track when disabled
        assert "node1" not in registry._histories

        # Correction should return original
        corrected = registry.apply_causal_correction("node1", "edge1", gradient)
        assert jnp.allclose(corrected, gradient)


class TestTransWeaveRegistry:
    """Test TransWeaveRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        TransWeaveRegistry.reset_instance()

    def test_singleton(self):
        """Test singleton pattern."""
        reg1 = TransWeaveRegistry.get_instance()
        reg2 = TransWeaveRegistry.get_instance()
        assert reg1 is reg2

    def test_register_task_end(self):
        """Test registering task end state."""
        registry = TransWeaveRegistry.get_instance()
        registry.set_current_task(0)

        params = NodeParams(
            weights={"edge1": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
            biases={"b": jnp.array([0.1, 0.2])},
        )

        registry.register_task_end("node1", params)

        assert "node1" in registry._states
        assert 0 in registry._states["node1"].task_representations

    def test_get_transfer_init_no_history(self):
        """Test transfer init with no history."""
        registry = TransWeaveRegistry.get_instance()

        params = NodeParams(
            weights={"edge1": jnp.array([[1.0, 2.0]])},
            biases={},
        )

        result = registry.get_transfer_init("node1", params)

        # Should return original params when no history
        assert jnp.allclose(result.weights["edge1"], params.weights["edge1"])

    def test_get_transfer_init_with_history(self):
        """Test transfer init with previous task history."""
        registry = TransWeaveRegistry.get_instance()

        # Register task 0
        registry.set_current_task(0)
        params0 = NodeParams(
            weights={"edge1": jnp.array([[1.0, 0.0], [0.0, 1.0]])},
            biases={},
        )
        registry.register_task_end("node1", params0)

        # Register task 1
        registry.set_current_task(1)
        params1 = NodeParams(
            weights={"edge1": jnp.array([[2.0, 0.0], [0.0, 2.0]])},
            biases={},
        )
        registry.register_task_end("node1", params1)

        # Get transfer for task 2
        registry.set_current_task(2)
        new_params = NodeParams(
            weights={"edge1": jnp.array([[0.0, 0.0], [0.0, 0.0]])},
            biases={},
        )

        result = registry.get_transfer_init("node1", new_params, transfer_strength=0.5)

        # Should be blend of history and new params
        # Not all zeros (due to transfer)
        assert not jnp.allclose(result.weights["edge1"], new_params.weights["edge1"])

    def test_clear(self):
        """Test clearing state."""
        registry = TransWeaveRegistry.get_instance()

        params = NodeParams(weights={"edge1": jnp.array([[1.0]])}, biases={})
        registry.register_task_end("node1", params)
        registry.register_task_end("node2", params)

        registry.clear("node1")
        assert "node1" not in registry._states
        assert "node2" in registry._states

        registry.clear()
        assert len(registry._states) == 0


class TestCausalLinear:
    """Test CausalLinear node."""

    def setup_method(self):
        """Reset registries before each test."""
        CausalGradientRegistry.reset_instance(
            PerWeightCausalConfig(
                gradient_history_size=10,
                min_history_for_detection=5,
            )
        )

    def test_init(self):
        """Test node initialization."""
        node = CausalLinear(
            shape=(64,),
            name="test_causal",
            activation=ReLUActivation(),
        )

        assert node.name == "test_causal"
        assert node.shape == (64,)
        assert "causal_enable" in node._extra_config

    def test_get_slots(self):
        """Test slot definition."""
        slots = CausalLinear.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True

    def test_initialize_params(self):
        """Test parameter initialization."""
        key = jax.random.PRNGKey(42)

        params = CausalLinear.initialize_params(
            key=key,
            node_shape=(64,),
            input_shapes={"source->test:in": (128,)},
            weight_init=None,
            config={"use_bias": True, "flatten_input": False},
        )

        assert "source->test:in" in params.weights
        assert params.weights["source->test:in"].shape == (128, 64)
        assert "b" in params.biases


class TestTransWeaveLinear:
    """Test TransWeaveLinear node."""

    def setup_method(self):
        """Reset registries before each test."""
        TransWeaveRegistry.reset_instance()

    def test_init(self):
        """Test node initialization."""
        node = TransWeaveLinear(
            shape=(64,),
            name="test_transweave",
            transfer_strength=0.4,
            use_last_k_tasks=2,
        )

        assert node.name == "test_transweave"
        assert node._extra_config["transfer_strength"] == 0.4
        assert node._extra_config["use_last_k_tasks"] == 2


class TestCausalTransWeaveLinear:
    """Test CausalTransWeaveLinear node."""

    def setup_method(self):
        """Reset registries before each test."""
        CausalGradientRegistry.reset_instance()
        TransWeaveRegistry.reset_instance()

    def test_init(self):
        """Test node initialization."""
        node = CausalTransWeaveLinear(
            shape=(64,),
            name="test_combined",
            causal_config=PerWeightCausalConfig(enable=True),
            transfer_strength=0.3,
        )

        assert node.name == "test_combined"
        assert "causal_enable" in node._extra_config
        assert node._extra_config["transfer_strength"] == 0.3


class TestHelperFunctions:
    """Test helper functions."""

    def setup_method(self):
        """Reset registries before each test."""
        CausalGradientRegistry.reset_instance()
        TransWeaveRegistry.reset_instance()

    def test_apply_causal_to_gradients(self):
        """Test apply_causal_to_gradients helper."""
        gradients = {
            "edge1": jnp.array([[1.0, 2.0]]),
            "edge2": jnp.array([[3.0, 4.0]]),
        }

        result = apply_causal_to_gradients(gradients, "test_node")

        assert "edge1" in result
        assert "edge2" in result
        assert result["edge1"].shape == gradients["edge1"].shape

    def test_register_task_end_for_nodes(self):
        """Test register_task_end_for_nodes helper."""
        params = {
            "node1": NodeParams(
                weights={"edge1": jnp.array([[1.0]])},
                biases={},
            ),
            "node2": NodeParams(
                weights={"edge1": jnp.array([[2.0]])},
                biases={},
            ),
        }

        register_task_end_for_nodes(["node1", "node2"], params)

        registry = TransWeaveRegistry.get_instance()
        assert "node1" in registry._states
        assert "node2" in registry._states

    def test_get_transferred_params(self):
        """Test get_transferred_params helper."""
        registry = TransWeaveRegistry.get_instance()

        # Register some history
        registry.set_current_task(0)
        params0 = {
            "node1": NodeParams(
                weights={"edge1": jnp.array([[1.0, 2.0]])},
                biases={},
            ),
        }
        register_task_end_for_nodes(["node1"], params0)

        # Get transfer for new params
        new_params = {
            "node1": NodeParams(
                weights={"edge1": jnp.array([[0.0, 0.0]])},
                biases={},
            ),
        }

        result = get_transferred_params(["node1"], new_params, transfer_strength=0.5)

        assert "node1" in result
        # Should have some transfer from history
        assert not jnp.allclose(
            result["node1"].weights["edge1"],
            new_params["node1"].weights["edge1"],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
