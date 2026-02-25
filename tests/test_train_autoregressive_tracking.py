#!/usr/bin/env python3
"""
Tests for autoregressive training debug callback and sampled history collection.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import jax
import jax.numpy as jnp
import pytest

from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.graph import initialize_params
from fabricpc.nodes import Linear
from fabricpc.training.train_autoregressive import train_autoregressive

jax.config.update("jax_platform_name", "cpu")


class MockDataLoader:
    """Simple deterministic data loader for tests."""

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
def rng_key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def graph_for_autoregressive(rng_key):
    seq_len = 4
    vocab_size = 6

    input_node = Linear(
        shape=(seq_len, vocab_size),
        activation=IdentityActivation(),
        name="input",
    )
    output_node = Linear(
        shape=(seq_len, vocab_size),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="output",
    )

    structure = graph(
        nodes=[input_node, output_node],
        edges=[Edge(source=input_node, target=output_node.slot("in"))],
        task_map=TaskMap(x=input_node, y=output_node),
        graph_state_initializer={"type": "feedforward"},
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def make_mock_loader(
    rng_key: jax.Array,
    batch_size: int = 2,
    num_batches: int = 3,
    seq_len: int = 4,
    vocab_size: int = 6,
) -> MockDataLoader:
    """Create one-hot autoregressive batches."""
    batches = []
    key = rng_key

    for _ in range(num_batches):
        key, x_key, y_key = jax.random.split(key, 3)
        x_idx = jax.random.randint(x_key, (batch_size, seq_len), 0, vocab_size)
        y_idx = jax.random.randint(y_key, (batch_size, seq_len), 0, vocab_size)
        x = jax.nn.one_hot(x_idx, vocab_size)
        y = jax.nn.one_hot(y_idx, vocab_size)
        batches.append({"x": x, "y": y})

    return MockDataLoader(batches)


def test_debug_iter_callback_receives_sampled_history(
    graph_for_autoregressive, rng_key
):
    """History should be available only at configured sampling frequency."""
    params, structure = graph_for_autoregressive
    train_loader = make_mock_loader(rng_key, num_batches=3)
    config = {
        "num_epochs": 1,
        "infer_steps": 2,
        "eta_infer": 0.05,
        "use_causal_mask": False,
        "optimizer": {"type": "adam", "lr": 1e-3},
    }

    callback_records = []

    def debug_iter_callback(
        epoch_idx, batch_idx, params, energy, ce_loss, final_state, inference_history
    ):
        callback_records.append(
            {
                "epoch": epoch_idx,
                "batch": batch_idx,
                "energy": float(energy),
                "ce_loss": float(ce_loss),
                "has_history": inference_history is not None,
            }
        )
        assert params is not None
        assert "output" in final_state.nodes
        if inference_history is not None:
            assert len(inference_history) > 0
            assert "output" in inference_history[0]
            assert "energy" in inference_history[0]["output"]

    trained_params, iter_results, epoch_results = train_autoregressive(
        params,
        structure,
        train_loader,
        config,
        rng_key,
        verbose=False,
        debug_iter_callback=debug_iter_callback,
        debug_collect_inference_history=True,
        debug_collect_every=1,
        debug_history_every_n_batches=2,
    )

    assert trained_params is not None
    assert len(iter_results) == 1
    assert len(iter_results[0]) == len(train_loader)
    assert len(epoch_results) == 1

    assert len(callback_records) == len(train_loader)
    assert callback_records[0]["has_history"] is True
    assert callback_records[1]["has_history"] is False
    assert callback_records[2]["has_history"] is True


def test_invalid_debug_sampling_args_raise(graph_for_autoregressive, rng_key):
    """Invalid debug sampling arguments should fail fast."""
    params, structure = graph_for_autoregressive
    train_loader = make_mock_loader(rng_key, num_batches=1)
    config = {
        "num_epochs": 1,
        "infer_steps": 1,
        "eta_infer": 0.05,
        "use_causal_mask": False,
        "optimizer": {"type": "adam", "lr": 1e-3},
    }

    with pytest.raises(ValueError, match="debug_collect_every"):
        train_autoregressive(
            params,
            structure,
            train_loader,
            config,
            rng_key,
            verbose=False,
            debug_iter_callback=lambda *args, **kwargs: None,
            debug_collect_inference_history=True,
            debug_collect_every=0,
        )
