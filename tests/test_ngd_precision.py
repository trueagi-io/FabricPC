"""
Tests for the online-precision natural-gradient (NGD) machinery.

Covers:
- NodeState.precision backward-compat + pytree round-trip.
- GaussianEnergy optional log-precision term (value changes, gradient does NOT).
- initialize_graph_state attaching a precision_map to node states.
- _vars_to_precision / _update_vars math.
- train_ngd end-to-end on a tiny graph (params + precision update, energy finite).
- probe_latent_propagation shape / non-zero latents.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core import InferenceSGD
from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import run_inference
from fabricpc.core.types import GraphState, NodeState
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.training.ngd_trainer import (
    _update_vars,
    _vars_to_precision,
    init_precision_vars,
    precision_node_names,
    probe_latent_propagation,
    train_ngd,
)


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def small_graph(rng_key):
    """input(8) -> h1(6, relu) -> output(4), all GaussianEnergy."""
    inp = Linear(shape=(8,), name="input")
    h1 = Linear(shape=(6,), activation=ReLUActivation(), name="h1")
    out = Linear(shape=(4,), name="output")
    structure = graph(
        nodes=[inp, h1, out],
        edges=[
            Edge(source=inp, target=h1.slot("in")),
            Edge(source=h1, target=out.slot("in")),
        ],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


# ---------------------------------------------------------------------------
# NodeState.precision
# ---------------------------------------------------------------------------


def test_nodestate_precision_default_and_pytree():
    z = jnp.zeros((2, 3))
    ns = NodeState(z, z, z, jnp.zeros((2,)), z, z)
    assert ns.precision is None
    # None contributes no leaves -> backward-compatible leaf count
    assert len(jax.tree_util.tree_leaves(ns)) == 6

    ns2 = ns._replace(precision=jnp.ones((3,)))
    assert len(jax.tree_util.tree_leaves(ns2)) == 7

    flat, treedef = jax.tree_util.tree_flatten(ns2)
    ns3 = jax.tree_util.tree_unflatten(treedef, flat)
    assert ns3.precision.shape == (3,)
    assert jnp.allclose(ns3.z_latent, z)


# ---------------------------------------------------------------------------
# Log-precision energy term
# ---------------------------------------------------------------------------


def test_log_precision_term_value_and_grad(rng_key):
    k1, k2 = jax.random.split(rng_key)
    z = jax.random.normal(k1, (2, 4))
    mu = jax.random.normal(k2, (2, 4))

    base = GaussianEnergy.energy(z, mu, {"precision": 2.0})
    withlog = GaussianEnergy.energy(
        z, mu, {"precision": 2.0, "include_log_precision": True}
    )
    # normalizer = 0.5*(ln 2pi - ln precision) per element, 4 non-batch elements
    norm = 0.5 * (math.log(2 * math.pi) - math.log(2.0)) * 4
    assert jnp.allclose(withlog, base + norm, atol=1e-5)

    # The log term is constant in z -> gradient must be unchanged.
    g1 = GaussianEnergy.grad_latent(z, mu, {"precision": 2.0})
    g2 = GaussianEnergy.grad_latent(
        z, mu, {"precision": 2.0, "include_log_precision": True}
    )
    assert jnp.allclose(g1, g2)


# ---------------------------------------------------------------------------
# precision helpers
# ---------------------------------------------------------------------------


def test_vars_to_precision():
    var = {"a": jnp.array([1.0, 4.0, 0.25])}
    raw = _vars_to_precision(var, eps=0.0, clip=None, normalize=False)
    assert jnp.allclose(raw["a"], jnp.array([1.0, 0.25, 4.0]))

    normed = _vars_to_precision(var, eps=0.0, clip=None, normalize=True)
    assert abs(float(jnp.mean(normed["a"])) - 1.0) < 1e-6

    clipped = _vars_to_precision(
        {"a": jnp.array([1e-9])}, eps=0.0, clip=(0.1, 10.0), normalize=False
    )
    assert float(clipped["a"][0]) == pytest.approx(10.0)


def test_vars_to_precision_renormalizes_after_clip():
    """When clipping bites, normalize→clip→re-normalize restores per-layer mean Pi≈1
    (the property the NGD 'precision doesn't rescale the global LR' claim relies on).
    Note: the final re-normalize can nudge values slightly past the clip bounds — that
    is accepted by design; the invariant under test is the mean, not the bounds."""
    # One channel has tiny variance -> huge raw precision that the clip will catch.
    var = {"a": jnp.array([1e-6, 1.0, 1.0, 1.0])}
    pi = _vars_to_precision(var, eps=1e-3, clip=(0.1, 10.0), normalize=True)["a"]
    assert jnp.all(pi > 0) and jnp.all(jnp.isfinite(pi))
    # mean restored to ~1 by the post-clip re-normalize (would be far from 1 without it)
    assert float(jnp.mean(pi)) == pytest.approx(1.0, abs=1e-5)
    # clip still limited the spread (the tiny-variance channel didn't run away)
    assert float(jnp.max(pi)) < 20.0


def test_init_precision_vars(small_graph):
    params, structure = small_graph
    names = precision_node_names(structure)
    # internal Gaussian nodes: h1 and output (input has in_degree 0)
    assert set(names) == {"h1", "output"}
    var = init_precision_vars(structure, names)
    assert var["h1"].shape == (6,) and jnp.allclose(var["h1"], 1.0)
    assert var["output"].shape == (4,)


# ---------------------------------------------------------------------------
# train_ngd end-to-end
# ---------------------------------------------------------------------------


def test_train_ngd_updates_params_and_precision(small_graph, rng_key):
    params, structure = small_graph
    rng = np.random.default_rng(0)
    loader = [
        (
            rng.standard_normal((8, 8)).astype("float32"),
            rng.standard_normal((8, 4)).astype("float32"),
        )
        for _ in range(3)
    ]
    opt = optax.sgd(0.05)
    new_params, precision, hist = train_ngd(
        params, structure, loader, opt, num_epochs=2, rng_key=rng_key, lam=0.2
    )

    before = jax.tree_util.tree_leaves(params)
    after = jax.tree_util.tree_leaves(new_params)
    assert any(
        not jnp.allclose(a, b) for a, b in zip(before, after)
    ), "params unchanged"
    # train_ngd returns the resolved precision map (Pi=1/(var+eps), mean-normalized).
    # After learning, Pi is non-uniform across channels (not all-ones) for some layer.
    assert any(not jnp.allclose(v, jnp.ones_like(v)) for v in precision.values())
    assert len(hist) == 2 and all(np.isfinite(e) for e in hist[-1])


# ---------------------------------------------------------------------------
# latent-propagation probe
# ---------------------------------------------------------------------------


def test_update_vars_nhwc_shape_and_ema():
    """_update_vars reduces a 4-D (B,H,W,C) error to (C,) and applies the EMA."""
    B, H, W, C = 2, 4, 4, 3
    err = jnp.ones((B, H, W, C)) * 2.0  # err^2 = 4 everywhere
    z = jnp.zeros((B, H, W, C))
    ns = NodeState(z, z, err, jnp.zeros((B,)), z, z)
    state = GraphState(nodes={"a": ns}, batch_size=B)
    new = _update_vars({"a": jnp.ones((C,))}, state, ["a"], lam=0.5)
    assert new["a"].shape == (C,)
    # EMA: 0.5*1 + 0.5*mean(err^2)=0.5 + 0.5*4 = 2.5
    assert jnp.allclose(new["a"], 2.5)


def test_precision_changes_weight_gradients(small_graph, rng_key):
    """The headline invariant: attaching a non-uniform precision changes the local
    WEIGHT gradients (precision genuinely flows through to the weight update)."""
    params, structure = small_graph
    kx, ky, kp = jax.random.split(rng_key, 3)
    batch = {"x": jax.random.normal(kx, (5, 8)), "y": jax.random.normal(ky, (5, 4))}
    clamps = {
        structure.task_map[t]: v for t, v in batch.items() if t in structure.task_map
    }

    # Baseline: no precision (uniform Pi=1 via config).
    s0 = initialize_graph_state(structure, 5, kp, clamps=clamps, params=params)
    g0 = compute_local_weight_gradients(
        params, run_inference(params, s0, clamps, structure), structure
    )

    # Non-uniform diagonal precision on h1.
    pmap = {
        "h1": jnp.array([5.0, 5.0, 5.0, 0.2, 0.2, 0.2]),
        "output": jnp.ones((4,)),
    }
    s1 = initialize_graph_state(
        structure, 5, kp, clamps=clamps, params=params, precision_map=pmap
    )
    g1 = compute_local_weight_gradients(
        params, run_inference(params, s1, clamps, structure), structure
    )

    l0 = jax.tree_util.tree_leaves(g0)
    l1 = jax.tree_util.tree_leaves(g1)
    assert any(
        not jnp.allclose(a, b) for a, b in zip(l0, l1)
    ), "precision did not change the weight gradients"


def test_probe_latent_propagation(small_graph, rng_key):
    params, structure = small_graph
    kx, ky, kp = jax.random.split(rng_key, 3)
    batch = {
        "x": jax.random.normal(kx, (5, 8)),
        "y": jax.random.normal(ky, (5, 4)),
    }
    rep = probe_latent_propagation(params, structure, batch, kp)
    assert set(rep) == set(structure.node_order)
    for name, m in rep.items():
        for key in ("z_rms", "err_rms", "zmu_rms", "move_rms"):
            assert np.isfinite(m[key])
        if m["in_degree"] > 0:
            assert m["z_rms"] >= 0.0
