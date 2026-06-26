"""muPC residual-depth accounting for the decomposed v2 transformer.

Each block contributes two skip-connection merge points (MHA skip + MLP2
residual), so muPC's residual depth L == 2 * depth, and the forward scale on
variance-scalable edges shrinks as 1/sqrt(L)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
import jax
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.mupc import (
    MuPCConfig,
    compute_mupc_scalings,
    _count_skip_connections_depth,
)
from fabricpc.nodes.transformer_v2 import create_deep_transformer


def _build(depth):
    return create_deep_transformer(
        depth=depth,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=8,
        vocab_size=10,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_residual_depth_is_two_per_block(depth):
    s = _build(depth)
    L = _count_skip_connections_depth(s.nodes, s.edges, s.node_order)
    assert L == 2 * depth


def test_skip_slots_are_marked_skip_connections():
    s = _build(2)
    for i in range(2):
        mha = s.nodes[f"L{i}_mha"].node_info.slots
        assert mha["skip"].is_skip_connection
        assert not mha["skip"].is_variance_scalable
        assert mha["in"].is_variance_scalable
        mlp2 = s.nodes[f"L{i}_mlp2"].node_info.slots
        assert mlp2["residual"].is_skip_connection
        assert not mlp2["residual"].is_variance_scalable


def _scalable_in_edge_key(structure, node, slot):
    for k, e in structure.edges.items():
        if e.target == node and e.slot == slot:
            return k
    raise KeyError(f"no edge into {node}:{slot}")


def test_forward_scale_shrinks_with_residual_depth():
    s1, s2 = _build(1), _build(2)  # L = 2 and 4
    cfg = MuPCConfig()
    sc1 = compute_mupc_scalings(s1.nodes, s1.edges, cfg, s1.node_order)
    sc2 = compute_mupc_scalings(s2.nodes, s2.edges, cfg, s2.node_order)

    a1 = sc1["L0_mha"].forward_scale[_scalable_in_edge_key(s1, "L0_mha", "in")]
    a2 = sc2["L0_mha"].forward_scale[_scalable_in_edge_key(s2, "L0_mha", "in")]

    assert a2 < a1
    assert math.isclose(a2 / a1, math.sqrt(2 / 4), rel_tol=1e-6)
