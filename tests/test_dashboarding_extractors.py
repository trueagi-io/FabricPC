"""Tests for dashboarding extractors, focused on distribution hardening.

A tracker must record pathological training states without crashing the
run: diverging PC training drives float32 weights toward the dtype limit,
where np.histogram (inside aim.Distribution) raises ValueError because
max - min overflows float32, and inf/NaN make the autodetected range
non-finite. flatten_for_distribution is the shared funnel for every
distribution payload, so it must yield histogram-safe samples.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from fabricpc.utils.dashboarding.extractors import flatten_for_distribution


def test_flattens_to_1d_float64():
    out = flatten_for_distribution(jnp.ones((4, 8)))
    assert out.shape == (32,)
    assert out.dtype == np.float64


def test_diverged_float32_range_is_histogrammable():
    # Range 6e38 overflows float32; in float64 the histogram succeeds.
    arr = jnp.array([-3e38, 3e38, 1.0], dtype=jnp.float32)
    out = flatten_for_distribution(arr)
    hist, edges = np.histogram(out, bins=64)
    assert hist.sum() == 3
    assert np.isfinite(edges).all()


def test_nonfinite_values_are_dropped():
    arr = jnp.array([1.0, jnp.inf, -jnp.inf, jnp.nan, 2.0], dtype=jnp.float32)
    out = flatten_for_distribution(arr)
    assert out.tolist() == [1.0, 2.0]


def test_all_nonfinite_yields_empty():
    arr = jnp.array([jnp.inf, jnp.nan], dtype=jnp.float32)
    out = flatten_for_distribution(arr)
    assert out.size == 0


@pytest.mark.parametrize(
    "arr",
    [
        jnp.zeros((16,)),
        jnp.array([np.finfo(np.float32).min, np.finfo(np.float32).max]),
    ],
)
def test_edge_distributions_are_histogrammable(arr):
    out = flatten_for_distribution(arr)
    hist, edges = np.histogram(out, bins=64)
    assert hist.sum() == out.size
    assert np.isfinite(edges).all()
