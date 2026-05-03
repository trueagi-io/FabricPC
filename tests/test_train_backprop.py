#!/usr/bin/env python3
"""
Test suite for backprop training functions: train_backprop and evaluate_backprop.
"""

import pytest
import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization.state_initializer import GlobalStateInit
from fabricpc.training.train_backprop import (
    train_backprop,
    evaluate_backprop,
    compute_forward_pass,
    validate_feedforward_init,
)


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self, data):
        self.data = data
        self._index = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.data):
            raise StopIteration
        item = self.data[self._index]
        self._index += 1
        return item


@pytest.fixture
def graph_with_feedforward(rng_key):
    """Graph with feedforward state initializer for backprop training."""
    input_node = Linear(shape=(10,), name="input")
    hidden = Linear(shape=(20,), activation=ReLUActivation(), name="hidden")
    output = Linear(shape=(5,), activation=SoftmaxActivation(), name="output")

    structure = graph(
        nodes=[input_node, hidden, output],
        edges=[
            Edge(source=input_node, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def make_mock_loader(rng_key, batch_size=8, num_batches=4, input_dim=10, num_classes=5):
    """Create mock data loader with classification data."""
    batches = []
    for i in range(num_batches):
        key1, key2, rng_key = jax.random.split(rng_key, 3)
        x = jax.random.normal(key1, (batch_size, input_dim))
        y = jax.nn.one_hot(
            jax.random.randint(key2, (batch_size,), 0, num_classes), num_classes
        )
        batches.append({"x": x, "y": y})
    return MockDataLoader(batches)


class TestTrainBackprop:
    """Tests for train_backprop function."""

    def test_train_backprop_basic(self, graph_with_feedforward, rng_key):
        """Test train_backprop returns correct outputs and updates parameters."""
        params, structure = graph_with_feedforward
        train_loader = make_mock_loader(rng_key, num_batches=4)
        optimizer = optax.adam(0.01)
        config = {
            "num_epochs": 2,
            "loss_type": "cross_entropy",
        }

        trained_params, iter_results, epoch_results = train_backprop(
            params, structure, train_loader, optimizer, config, rng_key, verbose=False
        )

        # Verify return structure
        assert trained_params is not None
        assert len(iter_results) == 2  # 2 epochs
        assert len(iter_results[0]) == 4  # 4 batches per epoch
        assert len(epoch_results) == 2

        # Verify parameters were updated
        params_changed = False
        for node_name in params.nodes:
            for edge_key in params.nodes[node_name].weights:
                old_w = params.nodes[node_name].weights[edge_key]
                new_w = trained_params.nodes[node_name].weights[edge_key]
                if not jnp.allclose(old_w, new_w):
                    params_changed = True
                    break
        assert params_changed, "Parameters should be updated during training"

        # Verify losses are valid
        for loss in iter_results[0]:
            assert not jnp.isnan(loss)
            assert loss >= 0

    def test_train_backprop_with_callbacks(self, graph_with_feedforward, rng_key):
        """Test train_backprop with epoch and iteration callbacks."""
        params, structure = graph_with_feedforward
        train_loader = make_mock_loader(rng_key, num_batches=2)
        optimizer = optax.adam(0.01)
        config = {"num_epochs": 2}

        epoch_calls = []
        iter_calls = []

        def epoch_callback(epoch_idx, params, structure, config, rng):
            epoch_calls.append(epoch_idx)
            return {"epoch": epoch_idx}

        def iter_callback(epoch_idx, batch_idx, loss):
            iter_calls.append((epoch_idx, batch_idx))
            return float(loss)

        _, iter_results, epoch_results = train_backprop(
            params,
            structure,
            train_loader,
            optimizer,
            config,
            rng_key,
            verbose=False,
            epoch_callback=epoch_callback,
            iter_callback=iter_callback,
        )

        assert epoch_calls == [0, 1]
        assert len(iter_calls) == 4  # 2 epochs * 2 batches
        assert epoch_results[0]["epoch"] == 0

    def test_train_backprop_invalid_initializer_raises(self, rng_key):
        """Test that non-feedforward initializer raises error."""
        input_node = Linear(shape=(10,), name="input")
        output = Linear(shape=(5,), name="output")

        structure = graph(
            nodes=[input_node, output],
            edges=[
                Edge(source=input_node, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(),
            graph_state_initializer=GlobalStateInit(),  # Not feedforward
        )

        with pytest.raises(ValueError, match="FeedforwardStateInit"):
            validate_feedforward_init(structure)


class TestEvaluateBackprop:
    """Tests for evaluate_backprop function."""

    def test_evaluate_backprop_basic(self, graph_with_feedforward, rng_key):
        """Test evaluate_backprop returns valid metrics."""
        params, structure = graph_with_feedforward
        test_loader = make_mock_loader(rng_key, batch_size=4, num_batches=2)
        config = {"loss_type": "cross_entropy"}

        results = evaluate_backprop(params, structure, test_loader, config, rng_key)

        assert "loss" in results
        assert "accuracy" in results
        assert "perplexity" in results
        assert results["loss"] >= 0
        assert not jnp.isnan(results["loss"])
        assert 0 <= results["accuracy"] <= 1
        assert results["perplexity"] > 0

    def test_evaluate_backprop_no_rng_key(self, graph_with_feedforward, rng_key):
        """Test evaluate_backprop works without explicit rng_key."""
        params, structure = graph_with_feedforward
        test_loader = make_mock_loader(rng_key, batch_size=4, num_batches=2)
        config = {"loss_type": "cross_entropy"}

        results = evaluate_backprop(
            params, structure, test_loader, config, rng_key=None
        )
        assert "loss" in results
        assert "accuracy" in results


class TestComputeForwardPass:
    """Tests for compute_forward_pass function."""

    def test_forward_pass_shapes_and_clamps(self, graph_with_feedforward, rng_key):
        """Test forward pass produces correct shapes and respects input clamps."""
        params, structure = graph_with_feedforward
        batch_size = 8
        x = jnp.ones((batch_size, 10)) * 5.0
        y = jax.nn.one_hot(jnp.arange(batch_size) % 5, 5)
        batch = {"x": x, "y": y}

        state = compute_forward_pass(params, structure, batch, rng_key)

        # Check structure
        assert state.batch_size == batch_size
        assert "input" in state.nodes
        assert "hidden" in state.nodes
        assert "output" in state.nodes

        # Check shapes
        assert state.nodes["output"].z_mu.shape == (batch_size, 5)

        # Check input clamping
        assert jnp.allclose(state.nodes["input"].z_latent, x)


class TestIntegration:
    """Integration test for full training and evaluation pipeline."""

    def test_train_then_evaluate(self, rng_key):
        """Test complete pipeline: train then evaluate."""
        input_node = Linear(shape=(10,), name="input")
        hidden = Linear(shape=(20,), activation=ReLUActivation(), name="hidden")
        output = Linear(shape=(5,), activation=SoftmaxActivation(), name="output")

        structure = graph(
            nodes=[input_node, hidden, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("in")),
                Edge(source=hidden, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)

        train_loader = make_mock_loader(rng_key, batch_size=16, num_batches=10)
        test_loader = make_mock_loader(
            jax.random.PRNGKey(999), batch_size=8, num_batches=3
        )

        optimizer = optax.adam(0.01)
        train_config = {
            "num_epochs": 5,
            "loss_type": "cross_entropy",
        }

        trained_params, iter_results, _ = train_backprop(
            params,
            structure,
            train_loader,
            optimizer,
            train_config,
            rng_key,
            verbose=False,
        )

        eval_config = {"loss_type": "cross_entropy"}
        results = evaluate_backprop(
            trained_params, structure, test_loader, eval_config, rng_key
        )

        # Verify pipeline completed successfully
        assert trained_params is not None
        assert len(iter_results) == 5
        assert results["loss"] >= 0
        assert 0 <= results["accuracy"] <= 1
