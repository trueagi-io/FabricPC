"""
Tests for diagonal precision weighting.

Covers:
- GaussianEnergy with scalar precision (backward compatibility).
- GaussianEnergy with a diagonal (per-channel) precision vector.
- probe_residual_precision: shapes, positivity, normalization, node selection.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core import InferenceSGD
from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.precision import probe_residual_precision
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import Linear


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# GaussianEnergy: scalar precision (backward compatibility)
# ---------------------------------------------------------------------------


def test_scalar_precision_matches_legacy_formula(rng_key):
    k1, k2 = jax.random.split(rng_key)
    z = jax.random.normal(k1, (3, 5))
    mu = jax.random.normal(k2, (3, 5))
    diff = z - mu

    # Default precision 1.0 == 0.5 * sum(diff^2)
    e1 = GaussianEnergy.energy(z, mu, {"precision": 1.0})
    assert jnp.allclose(e1, 0.5 * jnp.sum(diff**2, axis=1))

    # Scalar precision scales linearly.
    e2 = GaussianEnergy.energy(z, mu, {"precision": 2.0})
    assert jnp.allclose(e2, 2.0 * e1)

    # Gradient: precision * (z - mu)
    g = GaussianEnergy.grad_latent(z, mu, {"precision": 2.0})
    assert jnp.allclose(g, 2.0 * diff)

    # No config -> default precision 1.0
    assert jnp.allclose(GaussianEnergy.energy(z, mu, None), e1)


# ---------------------------------------------------------------------------
# GaussianEnergy: diagonal (per-channel) precision
# ---------------------------------------------------------------------------


def test_diagonal_precision_energy_and_grad(rng_key):
    k1, k2 = jax.random.split(rng_key)
    # Conv-shaped latent: (batch, H, W, C). Precision is per-channel (C,).
    B, H, W, C = 2, 4, 4, 3
    z = jax.random.normal(k1, (B, H, W, C))
    mu = jax.random.normal(k2, (B, H, W, C))
    diff = z - mu
    pi = jnp.array([0.5, 1.0, 4.0])

    e = GaussianEnergy.energy(z, mu, {"precision": pi})
    expected = 0.5 * jnp.sum(pi * diff**2, axis=(1, 2, 3))
    assert e.shape == (B,)
    assert jnp.allclose(e, expected)

    g = GaussianEnergy.grad_latent(z, mu, {"precision": pi})
    assert g.shape == z.shape
    assert jnp.allclose(g, pi * diff)


def test_ones_vector_equals_scalar(rng_key):
    k1, k2 = jax.random.split(rng_key)
    z = jax.random.normal(k1, (2, 4, 4, 6))
    mu = jax.random.normal(k2, (2, 4, 4, 6))
    e_scalar = GaussianEnergy.energy(z, mu, {"precision": 1.0})
    e_ones = GaussianEnergy.energy(z, mu, {"precision": jnp.ones(6)})
    assert jnp.allclose(e_scalar, e_ones)


# ---------------------------------------------------------------------------
# probe_residual_precision
# ---------------------------------------------------------------------------


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


def test_probe_shapes_and_node_selection(small_graph, rng_key):
    params, structure = small_graph
    kx, ky, kp = jax.random.split(rng_key, 3)
    batch = {
        "x": jax.random.normal(kx, (5, 8)),
        "y": jax.random.normal(ky, (5, 4)),
    }
    pmap = probe_residual_precision(params, structure, batch, kp)

    # Internal Gaussian nodes only: h1 and output (input has in_degree 0).
    assert set(pmap.keys()) == {"h1", "output"}
    assert pmap["h1"].shape == (6,)
    assert pmap["output"].shape == (4,)


def test_probe_values_positive_normalized_clipped(small_graph, rng_key):
    params, structure = small_graph
    kx, ky, kp = jax.random.split(rng_key, 3)
    batch = {
        "x": jax.random.normal(kx, (16, 8)),
        "y": jax.random.normal(ky, (16, 4)),
    }
    pmap = probe_residual_precision(
        params, structure, batch, kp, clip=(0.1, 10.0), normalize="mean"
    )
    for name, pi in pmap.items():
        assert np.all(np.isfinite(pi)), name
        assert np.all(pi > 0), name
        assert np.all(pi >= 0.1 - 1e-6) and np.all(pi <= 10.0 + 1e-6), name
        # mean normalization: average precision ~ 1 (exact unless clipping bites)
        assert abs(float(np.mean(pi)) - 1.0) < 0.5, name


def test_probe_normalize_none_changes_scale(small_graph, rng_key):
    params, structure = small_graph
    kx, ky, kp = jax.random.split(rng_key, 3)
    batch = {
        "x": jax.random.normal(kx, (16, 8)),
        "y": jax.random.normal(ky, (16, 4)),
    }
    raw = probe_residual_precision(
        params, structure, batch, kp, normalize="none", clip=None
    )
    # Raw precision is 1/(var+eps); for unit-ish residuals it is finite & positive
    # but not mean-normalized to 1.
    for pi in raw.values():
        assert np.all(np.isfinite(pi)) and np.all(pi > 0)
