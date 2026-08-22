"""Parity tests for ``fabricpc.training.unified_trainer``.

The unification PR's load-bearing claim is that ``unified_trainer.train`` and
``unified_trainer.evaluate`` produce **bitwise-identical** results to the
legacy ``train.train_pcn`` / ``train_backprop.train_backprop`` (and their eval
counterparts) on a single device with the same seed and data. This file is
the committed evidence — the plan calls it "the key review evidence"
(``fabricpc/training/trainers_plan.md`` §Verification).

Each test runs the legacy trainer and the unified trainer side by side from
the same initial params + RNG key + data loader, then asserts the final
param trees and per-batch metrics match to within `1e-12`. Failure means a
behavior drift was introduced between the two paths.
"""

from typing import Iterator, List

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import Linear
from fabricpc.training.train import train_pcn, evaluate_pcn
from fabricpc.training.train_backprop import train_backprop, evaluate_backprop
from fabricpc.training.unified_trainer import train as unified_train
from fabricpc.training.unified_trainer import evaluate as unified_evaluate

PARITY_TOL = 1e-12


# ---------------------------------------------------------------------------
# Tiny deterministic fixtures — small enough to run in a few seconds, large
# enough to exercise the scaffolding (matmul, activation, optimizer step).
# ---------------------------------------------------------------------------


class _ListLoader:
    """Deterministic loader: same batches every time it is iterated."""

    def __init__(self, batches):
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator:
        return iter(self._batches)


def _make_batches(rng_key, *, batch_size=4, n_batches=2, in_dim=6, n_classes=3):
    batches: List[dict] = []
    key = rng_key
    for _ in range(n_batches):
        kx, ky, key = jax.random.split(key, 3)
        x = jax.random.normal(kx, (batch_size, in_dim))
        y = jax.nn.one_hot(
            jax.random.randint(ky, (batch_size,), 0, n_classes), n_classes
        )
        batches.append({"x": x, "y": y})
    return _ListLoader(batches)


def _pc_structure():
    """3-node Linear graph wired for PC training (InferenceSGD)."""
    x = Linear(shape=(6,), name="x")
    h = Linear(shape=(8,), activation=ReLUActivation(), name="h")
    y = Linear(shape=(3,), activation=SoftmaxActivation(), name="y")
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(),
    )


def _backprop_structure():
    """3-node Linear graph wired for backprop.

    ``graph()`` requires an ``inference`` argument unconditionally and defaults
    ``graph_state_initializer`` to ``FeedforwardStateInit()`` — same setup
    ``tests/test_train_backprop.py`` uses. The inference object is inert for
    backprop (never invoked).
    """
    x = Linear(shape=(6,), name="x")
    h = Linear(shape=(8,), activation=ReLUActivation(), name="h")
    y = Linear(shape=(3,), activation=SoftmaxActivation(), name="y")
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(),
    )


def _max_param_diff(a, b) -> float:
    diffs = jax.tree_util.tree_map(lambda p, q: jnp.max(jnp.abs(p - q)), a, b)
    return float(jax.tree_util.tree_reduce(jnp.maximum, diffs, jnp.array(0.0)))


# ---------------------------------------------------------------------------
# Train parity
# ---------------------------------------------------------------------------


def test_train_pc_parity(rng_key):
    """unified_train(algo='pc') must match train_pcn bitwise."""
    structure = _pc_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = _make_batches(rng_key)
    optimizer = optax.adam(1e-2)
    config_legacy = {"num_epochs": 1}
    config_unified = {"num_epochs": 1}

    legacy_params, legacy_iter, _ = train_pcn(
        params,
        structure,
        loader,
        optimizer,
        config_legacy,
        train_key,
        verbose=False,
        use_tqdm=False,
    )
    unified_params, unified_iter, _ = unified_train(
        params,
        structure,
        loader,
        optimizer,
        config_unified,
        train_key,
        algorithm="pc",
        verbose=False,
        use_tqdm=False,
    )

    assert _max_param_diff(legacy_params, unified_params) < PARITY_TOL
    assert jnp.allclose(
        jnp.array(legacy_iter[0]), jnp.array(unified_iter[0]), atol=PARITY_TOL
    )


def test_train_backprop_parity(rng_key):
    """unified_train(algorithm='backprop') must match train_backprop bitwise."""
    structure = _backprop_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = _make_batches(rng_key)
    optimizer = optax.adam(1e-2)
    config_legacy = {"num_epochs": 1, "loss_type": "cross_entropy"}
    config_unified = {"num_epochs": 1, "loss_type": "cross_entropy"}

    legacy_params, legacy_iter, _ = train_backprop(
        params,
        structure,
        loader,
        optimizer,
        config_legacy,
        train_key,
        verbose=False,
    )
    unified_params, unified_iter, _ = unified_train(
        params,
        structure,
        loader,
        optimizer,
        config_unified,
        train_key,
        algorithm="backprop",
        verbose=False,
        use_tqdm=False,
    )

    assert _max_param_diff(legacy_params, unified_params) < PARITY_TOL
    assert jnp.allclose(
        jnp.array(legacy_iter[0]), jnp.array(unified_iter[0]), atol=PARITY_TOL
    )


# ---------------------------------------------------------------------------
# Evaluate parity
# ---------------------------------------------------------------------------


def test_evaluate_pc_parity(rng_key):
    """unified_evaluate(algo='pc') must match evaluate_pcn bitwise."""
    structure = _pc_structure()
    params_key, eval_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = _make_batches(rng_key)

    legacy = evaluate_pcn(params, structure, loader, {}, eval_key)
    unified = unified_evaluate(params, structure, loader, {}, eval_key, algorithm="pc")

    for key in ("energy", "accuracy"):
        assert (
            abs(legacy[key] - unified[key]) < PARITY_TOL
        ), f"{key} drift: legacy={legacy[key]} unified={unified[key]}"


def test_evaluate_backprop_parity(rng_key):
    """unified_evaluate(algo='backprop') must match evaluate_backprop bitwise."""
    structure = _backprop_structure()
    params_key, eval_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = _make_batches(rng_key)

    legacy = evaluate_backprop(
        params, structure, loader, {"loss_type": "cross_entropy"}, eval_key
    )
    unified = unified_evaluate(
        params,
        structure,
        loader,
        {"loss_type": "cross_entropy"},
        eval_key,
        algorithm="backprop",
    )

    for key in ("loss", "accuracy"):
        assert (
            abs(legacy[key] - unified[key]) < PARITY_TOL
        ), f"{key} drift: legacy={legacy[key]} unified={unified[key]}"


# ---------------------------------------------------------------------------
# Contract guards
# ---------------------------------------------------------------------------


def test_unified_train_rejects_unknown_algorithm(rng_key):
    structure = _pc_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = _make_batches(rng_key, n_batches=1)
    optimizer = optax.adam(1e-2)

    with pytest.raises(ValueError):
        unified_train(
            params,
            structure,
            loader,
            optimizer,
            {"num_epochs": 1},
            train_key,
            algorithm="unknown",
            verbose=False,
            use_tqdm=False,
        )


def test_unified_train_reserves_auto(rng_key):
    """'auto' is reserved for the future autoregressive merge; raises a clear error today."""
    structure = _pc_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = _make_batches(rng_key, n_batches=1)
    optimizer = optax.adam(1e-2)

    with pytest.raises(NotImplementedError):
        unified_train(
            params,
            structure,
            loader,
            optimizer,
            {"num_epochs": 1},
            train_key,
            algorithm="auto",
            verbose=False,
            use_tqdm=False,
        )
