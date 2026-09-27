"""Dropout: the helper, the v2 transformer nodes, and the train/eval contract.

The PC-specific requirement is that one batch uses ONE mask everywhere:
feedforward init, every inference sweep, the weight-gradient pass, and the
backprop objective. Otherwise PC would relax against a moving target and PC
vs backprop would compare different thinned networks.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fabricpc.core.dropout import (
    DROPOUT_STREAM,
    attach_dropout_keys,
    dropout,
    node_dropout_key,
)
from fabricpc.core.activations import (
    ReLUActivation,
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD, gather_inputs, run_inference
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import FeedforwardStateInit, initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.models import create_deep_transformer
from fabricpc.nodes import (
    ConvNode,
    EmbeddingNode,
    IdentityNode,
    Linear,
    LinearResidual,
    MhaResidualNode,
    Mlp2ResidualNode,
)
from fabricpc.training.trainer import build_clamps, evaluate, make_train_step

SEQ, VOCAB, B = 8, 11, 4
INTERNAL = ("embed", "L0_mha", "L0_mlp1", "L0_mlp2", "L1_mha", "L1_mlp1", "L1_mlp2")


def _build(rate=0.3, attn_rate=0.2, inference=None):
    return create_deep_transformer(
        depth=2,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=SEQ,
        vocab_size=VOCAB,
        inference=inference or InferenceSGD(eta_infer=0.05, infer_steps=3),
        dropout_rate=rate,
        attn_dropout_rate=attn_rate,
    )


def _batch(seed=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return {
        "x": jax.random.randint(k1, (B, SEQ), 0, VOCAB),
        "y": jax.random.randint(k2, (B, SEQ), 0, VOCAB),
    }


def _state(structure, params, batch, key, dropout_key=None, train=True):
    clamps = build_clamps(batch, structure, clamp_target=train)
    state = initialize_graph_state(
        structure, B, key, clamps=clamps, params=params, dropout_key=dropout_key
    )
    return state, clamps


def _assert_tree_equal(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _tree_differs(a, b):
    return any(
        not np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


class TestDropoutHelper:
    def test_identity_when_off(self):
        x = jnp.arange(12.0).reshape(3, 4)
        key = jax.random.PRNGKey(0)
        assert dropout(x, 0.0, key, site=0) is x
        assert dropout(x, 0.5, None, site=0) is x

    def test_inverted_scaling_and_rate(self):
        x = jnp.ones((200, 500))
        y = dropout(x, 0.3, jax.random.PRNGKey(1), site=0)
        vals = np.unique(np.asarray(y))
        np.testing.assert_allclose(vals, [0.0, 1.0 / 0.7], rtol=1e-6)
        assert abs(float(jnp.mean(y == 0)) - 0.3) < 0.01
        assert abs(float(jnp.mean(y)) - 1.0) < 0.01  # expectation preserved

    def test_deterministic_per_key_and_site(self):
        x = jnp.ones((64, 64))
        k = jax.random.PRNGKey(2)
        a = dropout(x, 0.5, k, site=0)
        np.testing.assert_array_equal(a, dropout(x, 0.5, k, site=0))
        assert not np.array_equal(a, dropout(x, 0.5, k, site=1))
        assert not np.array_equal(a, dropout(x, 0.5, jax.random.PRNGKey(3), site=0))

    def test_gradient_flows_only_through_kept_units(self):
        k = jax.random.PRNGKey(4)
        x = jnp.ones((32, 32))
        g = jax.grad(lambda v: jnp.sum(dropout(v, 0.5, k, site=0)))(x)
        y = dropout(x, 0.5, k, site=0)
        np.testing.assert_allclose(g, y)  # grad = mask / (1 - p)

    def test_node_key_depends_on_name_not_order(self):
        k = jax.random.PRNGKey(5)
        a = node_dropout_key(k, "L0_mha")
        np.testing.assert_array_equal(a, node_dropout_key(k, "L0_mha"))
        assert not np.array_equal(a, node_dropout_key(k, "L1_mha"))

    def test_attach_none_is_noop(self):
        s = _build()
        p = initialize_params(s, jax.random.PRNGKey(0))
        state, _ = _state(s, p, _batch(), jax.random.PRNGKey(0))
        assert attach_dropout_keys(state, s, None) is state


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
def test_invalid_rates_raise(rate):
    with pytest.raises(ValueError, match="dropout_rate"):
        EmbeddingNode(
            shape=(SEQ, 16), name="e", vocab_size=5, embed_dim=16, dropout_rate=rate
        )
    with pytest.raises(ValueError, match="dropout_rate"):
        Mlp2ResidualNode(
            shape=(SEQ, 16), name="m", embed_dim=16, ff_dim=32, dropout_rate=rate
        )
    with pytest.raises(ValueError, match="attn_dropout_rate"):
        MhaResidualNode(
            shape=(SEQ, 16), name="a", embed_dim=16, num_heads=2, attn_dropout_rate=rate
        )


def test_node_state_default_has_no_key():
    z = jnp.zeros((2, 3))
    ns = NodeState(z, z, z, jnp.zeros((2,)), z)  # old 5-field construction
    assert ns.dropout_key is None


def test_rates_reach_node_config():
    s = _build(rate=0.3, attn_rate=0.2)
    assert s.nodes["embed"].node_info.node_config["dropout_rate"] == 0.3
    assert s.nodes["L0_mha"].node_info.node_config["attn_dropout_rate"] == 0.2
    assert s.nodes["L1_mlp2"].node_info.node_config["dropout_rate"] == 0.3


# ---------------------------------------------------------------------------
# Graph-level behavior
# ---------------------------------------------------------------------------


def test_params_independent_of_dropout_rate():
    p_drop = initialize_params(_build(0.3, 0.2), jax.random.PRNGKey(0))
    p_none = initialize_params(_build(0.0, 0.0), jax.random.PRNGKey(0))
    _assert_tree_equal(p_drop, p_none)


def test_no_key_matches_zero_rate_model():
    """Eval semantics: without a dropout key the dropout graph is exactly the
    rate-0 graph."""
    s_drop, s_none = _build(0.3, 0.2), _build(0.0, 0.0)
    p = initialize_params(s_drop, jax.random.PRNGKey(0))
    batch, key = _batch(), jax.random.PRNGKey(1)
    a, _ = _state(s_drop, p, batch, key, train=False)
    b, _ = _state(s_none, p, batch, key, train=False)
    _assert_tree_equal(a, b)
    assert all(ns.dropout_key is None for ns in a.nodes.values())


def test_zero_rate_with_key_matches_no_key():
    """A rate-0 graph given a dropout key is numerically unchanged."""
    s = _build(0.0, 0.0)
    p = initialize_params(s, jax.random.PRNGKey(0))
    batch, key = _batch(), jax.random.PRNGKey(1)
    a, _ = _state(s, p, batch, key, dropout_key=jax.random.PRNGKey(9))
    b, _ = _state(s, p, batch, key)
    for name in s.nodes:
        np.testing.assert_array_equal(a.nodes[name].z_mu, b.nodes[name].z_mu)


def test_masks_reproducible_and_key_dependent():
    s = _build()
    p = initialize_params(s, jax.random.PRNGKey(0))
    batch, key = _batch(), jax.random.PRNGKey(1)
    a, _ = _state(s, p, batch, key, dropout_key=jax.random.PRNGKey(7))
    b, _ = _state(s, p, batch, key, dropout_key=jax.random.PRNGKey(7))
    c, _ = _state(s, p, batch, key, dropout_key=jax.random.PRNGKey(8))
    off, _ = _state(s, p, batch, key)
    _assert_tree_equal(a, b)
    assert _tree_differs(a.nodes["logits"].z_mu, c.nodes["logits"].z_mu)
    assert _tree_differs(a.nodes["logits"].z_mu, off.nodes["logits"].z_mu)


def test_every_dropout_site_is_active():
    """Each configured site changes its node's prediction when switched on."""
    batch, key, dk = _batch(), jax.random.PRNGKey(1), jax.random.PRNGKey(7)
    for rate, attn_rate, node in [
        (0.3, 0.0, "embed"),
        (0.3, 0.0, "L0_mha"),
        (0.0, 0.3, "L0_mha"),
        (0.3, 0.0, "L0_mlp2"),
    ]:
        s = _build(rate, attn_rate)
        p = initialize_params(s, jax.random.PRNGKey(0))
        on, _ = _state(s, p, batch, key, dropout_key=dk)
        off, _ = _state(s, p, batch, key)
        # Feed the node the same inputs (the no-dropout latents) and compare.

        info = s.nodes[node].node_info
        inputs = scale_inputs(gather_inputs(info, s, off), info.scaling_config)
        z_on, _ = info.node_class.predict(p.nodes[node], inputs, on.nodes[node], info)
        z_off, _ = info.node_class.predict(p.nodes[node], inputs, off.nodes[node], info)
        assert _tree_differs(z_on, z_off), (rate, attn_rate, node)
    # LnMlp1 and the vocab projection carry no dropout.
    s = _build(0.5, 0.5)
    for name in ("L0_mlp1", "logits"):
        assert "dropout_rate" not in s.nodes[name].node_info.node_config


# ---------------------------------------------------------------------------
# PC consistency: one mask per batch
# ---------------------------------------------------------------------------


def _internal_energy(state):
    return max(float(jnp.max(jnp.abs(state.nodes[n].energy))) for n in INTERNAL)


def test_feedforward_init_is_zero_energy_under_dropout():
    """z_latent = z_mu at every unclamped internal node: FF init and the
    inference forward used the same masks."""
    s = _build(inference=InferenceSGD(eta_infer=0.0, infer_steps=1))
    p = initialize_params(s, jax.random.PRNGKey(0))
    state, clamps = _state(
        s, p, _batch(), jax.random.PRNGKey(1), dropout_key=jax.random.PRNGKey(7)
    )
    state = run_inference(p, state, clamps, s)  # one forward, no latent move
    assert _internal_energy(state) < 1e-10
    assert float(jnp.sum(state.nodes["logits"].energy)) > 0.0


def test_masks_fixed_across_inference_sweeps():
    """With eta=0, T sweeps must leave the FF-init state untouched. A mask
    redrawn per sweep would give nonzero internal energy."""
    s = _build(inference=InferenceSGD(eta_infer=0.0, infer_steps=6))
    p = initialize_params(s, jax.random.PRNGKey(0))
    state0, clamps = _state(
        s, p, _batch(), jax.random.PRNGKey(1), dropout_key=jax.random.PRNGKey(7)
    )
    state = jax.jit(lambda pp, st: run_inference(pp, st, clamps, s))(p, state0)
    assert _internal_energy(state) < 1e-10
    for n in s.nodes:
        np.testing.assert_allclose(state.nodes[n].z_latent, state0.nodes[n].z_latent)
        np.testing.assert_array_equal(
            state.nodes[n].dropout_key, state0.nodes[n].dropout_key
        )


def test_pc_and_backprop_share_masks_in_train_step():
    """At T=0 from FF init, PC's output-layer gradient equals backprop's
    exactly — only if both steps used the same masks."""
    s = _build(inference=InferenceSGD(eta_infer=0.1, infer_steps=0))
    p = initialize_params(s, jax.random.PRNGKey(0))
    opt = optax.sgd(1.0)
    batch, key = _batch(), jax.random.PRNGKey(3)
    pc = make_train_step(s, opt, algorithm="pc")
    bp = make_train_step(s, opt, algorithm="backprop")
    p_pc, *_ = pc(p, opt.init(p), batch, key)
    p_bp, *_ = bp(p, opt.init(p), batch, key)
    for name in ("W_out",):
        np.testing.assert_allclose(
            p_pc.nodes["logits"].weights[name],
            p_bp.nodes["logits"].weights[name],
            rtol=1e-5,
            atol=1e-6,
        )
    np.testing.assert_allclose(
        p_pc.nodes["logits"].biases["b_out"],
        p_bp.nodes["logits"].biases["b_out"],
        rtol=1e-5,
        atol=1e-6,
    )
    # PC at T=0 leaves internal weights untouched (zero internal error, up to
    # float round-off between the init and gradient passes). With mismatched
    # masks the internal errors would be O(1) and these would move visibly.
    for name in INTERNAL:
        for x, y in zip(
            jax.tree_util.tree_leaves(p_pc.nodes[name]),
            jax.tree_util.tree_leaves(p.nodes[name]),
        ):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-7)


def test_train_step_masks_depend_on_batch_key_only():
    s = _build()
    p = initialize_params(s, jax.random.PRNGKey(0))
    opt = optax.sgd(0.1)
    step = make_train_step(s, opt, algorithm="backprop")
    batch = _batch()
    a = step(p, opt.init(p), batch, jax.random.PRNGKey(3))
    b = step(p, opt.init(p), batch, jax.random.PRNGKey(3))
    c = step(p, opt.init(p), batch, jax.random.PRNGKey(4))
    _assert_tree_equal(a[0], b[0])
    assert _tree_differs(a[0], c[0])
    # The trainer's dropout key is fold_in(batch_key, DROPOUT_STREAM).
    expect = node_dropout_key(
        jax.random.fold_in(jax.random.PRNGKey(3), DROPOUT_STREAM), "L0_mha"
    )
    np.testing.assert_array_equal(a[3].nodes["L0_mha"].dropout_key, expect)


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_train_step_finite(algorithm):
    s = _build(0.5, 0.5)
    p = initialize_params(s, jax.random.PRNGKey(0))
    opt = optax.adam(1e-3)
    step = make_train_step(s, opt, algorithm=algorithm)
    p2, _, m, _ = step(p, opt.init(p), _batch(), jax.random.PRNGKey(3))
    assert np.isfinite(float(m["energy"]))
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(p2))


def test_epc_runs_with_dropout():
    s = _build(inference=EPCInference(eta_infer=1e-3, infer_steps=2))
    p = initialize_params(s, jax.random.PRNGKey(0))
    opt = optax.sgd(0.1)
    p2, _, m, st = make_train_step(s, opt, algorithm="pc")(
        p, opt.init(p), _batch(), jax.random.PRNGKey(3)
    )
    assert np.isfinite(float(m["energy"]))
    assert st.nodes["L0_mha"].dropout_key is not None


# ---------------------------------------------------------------------------
# Evaluation stays deterministic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_evaluate_ignores_dropout(algorithm):
    s_drop, s_none = _build(0.5, 0.5), _build(0.0, 0.0)
    p = initialize_params(s_drop, jax.random.PRNGKey(0))
    loader = [_batch(0), _batch(1)]
    key = jax.random.PRNGKey(0)
    r1 = evaluate(p, s_drop, loader, {}, key, algorithm=algorithm)
    r2 = evaluate(p, s_drop, loader, {}, key, algorithm=algorithm)
    r3 = evaluate(p, s_none, loader, {}, key, algorithm=algorithm)
    assert r1 == r2 == r3


# ---------------------------------------------------------------------------
# General nodes: Linear, LinearResidual, ConvNode
# ---------------------------------------------------------------------------

GENERAL = ("conv", "hidden", "res")


def _build_general(rate=0.3, inference=None):
    """pixels(8,8,1) -> conv -> hidden (Linear) -> res (LinearResidual) -> out."""
    pixels = IdentityNode(shape=(8, 8, 1), name="pixels")
    conv = ConvNode(
        shape=(8, 8, 4),
        kernel_size=(3, 3),
        activation=ReLUActivation(),
        name="conv",
        dropout_rate=rate,
    )
    hidden = Linear(
        shape=(32,),
        activation=ReLUActivation(),
        flatten_input=True,
        name="hidden",
        dropout_rate=rate,
    )
    res = LinearResidual(
        shape=(32,), activation=TanhActivation(), name="res", dropout_rate=rate
    )
    out = Linear(
        shape=(3,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="out",
    )
    return graph(
        nodes=[pixels, conv, hidden, res, out],
        edges=[
            Edge(source=pixels, target=conv.slot("in")),
            Edge(source=conv, target=hidden.slot("in")),
            Edge(source=hidden, target=res.slot("in")),
            Edge(source=hidden, target=res.slot("skip")),
            Edge(source=res, target=out.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=out),
        inference=inference or InferenceSGD(eta_infer=0.05, infer_steps=3),
        graph_state_initializer=FeedforwardStateInit(),
    )


def _general_batch(seed=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return {
        "x": jax.random.normal(k1, (B, 8, 8, 1)),
        "y": jax.random.randint(k2, (B,), 0, 3),
    }


@pytest.mark.parametrize("rate", [-0.1, 1.0])
def test_general_nodes_reject_invalid_rates(rate):
    with pytest.raises(ValueError, match="dropout_rate"):
        Linear(shape=(4,), name="l", dropout_rate=rate)
    with pytest.raises(ValueError, match="dropout_rate"):
        LinearResidual(shape=(4,), name="r", dropout_rate=rate)
    with pytest.raises(ValueError, match="dropout_rate"):
        ConvNode(shape=(4, 4, 2), name="c", kernel_size=(3, 3), dropout_rate=rate)


def test_general_nodes_no_key_matches_zero_rate():
    s_drop, s_none = _build_general(0.3), _build_general(0.0)
    p = initialize_params(s_drop, jax.random.PRNGKey(0))
    _assert_tree_equal(p, initialize_params(s_none, jax.random.PRNGKey(0)))
    batch, key = _general_batch(), jax.random.PRNGKey(1)
    a, _ = _state(s_drop, p, batch, key, train=False)
    b, _ = _state(s_none, p, batch, key, train=False)
    _assert_tree_equal(a, b)


def test_general_nodes_each_site_active():
    s = _build_general(0.3)
    p = initialize_params(s, jax.random.PRNGKey(0))
    batch, key = _general_batch(), jax.random.PRNGKey(1)
    on, _ = _state(s, p, batch, key, dropout_key=jax.random.PRNGKey(7))
    off, _ = _state(s, p, batch, key)

    for node in GENERAL:
        info = s.nodes[node].node_info
        inputs = scale_inputs(gather_inputs(info, s, off), info.scaling_config)
        z_on, _ = info.node_class.predict(p.nodes[node], inputs, on.nodes[node], info)
        z_off, _ = info.node_class.predict(p.nodes[node], inputs, off.nodes[node], info)
        assert _tree_differs(z_on, z_off), node


def test_linear_residual_never_drops_skip():
    """Zero transform weights: z_mu is the skip input, with or without dropout."""
    s = _build_general(0.9)
    p = initialize_params(s, jax.random.PRNGKey(0))
    res_p = p.nodes["res"]._replace(
        weights={k: jnp.zeros_like(w) for k, w in p.nodes["res"].weights.items()},
        biases={k: jnp.zeros_like(b) for k, b in p.nodes["res"].biases.items()},
    )
    batch, key = _general_batch(), jax.random.PRNGKey(1)
    on, _ = _state(s, p, batch, key, dropout_key=jax.random.PRNGKey(7))

    info = s.nodes["res"].node_info
    inputs = gather_inputs(info, s, on)
    z, _ = info.node_class.predict(res_p, inputs, on.nodes["res"], info)
    np.testing.assert_allclose(z, on.nodes["hidden"].z_latent, rtol=1e-6)


def test_general_nodes_masks_fixed_across_sweeps():
    s = _build_general(0.3, inference=InferenceSGD(eta_infer=0.0, infer_steps=6))
    p = initialize_params(s, jax.random.PRNGKey(0))
    state0, clamps = _state(
        s, p, _general_batch(), jax.random.PRNGKey(1), dropout_key=jax.random.PRNGKey(7)
    )
    state = jax.jit(lambda pp, st: run_inference(pp, st, clamps, s))(p, state0)
    assert max(float(jnp.max(jnp.abs(state.nodes[n].energy))) for n in GENERAL) < 1e-10


def test_general_nodes_pc_and_backprop_share_masks():
    s = _build_general(0.3, inference=InferenceSGD(eta_infer=0.1, infer_steps=0))
    p = initialize_params(s, jax.random.PRNGKey(0))
    opt = optax.sgd(1.0)
    batch, key = _general_batch(), jax.random.PRNGKey(3)
    p_pc, *_ = make_train_step(s, opt, algorithm="pc")(p, opt.init(p), batch, key)
    p_bp, *_ = make_train_step(s, opt, algorithm="backprop")(p, opt.init(p), batch, key)
    for x, y in zip(
        jax.tree_util.tree_leaves(p_pc.nodes["out"]),
        jax.tree_util.tree_leaves(p_bp.nodes["out"]),
    ):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_general_nodes_train_and_eval(algorithm):
    s_drop, s_none = _build_general(0.3), _build_general(0.0)
    p = initialize_params(s_drop, jax.random.PRNGKey(0))
    opt = optax.adam(1e-3)
    p2, _, m, _ = make_train_step(s_drop, opt, algorithm=algorithm)(
        p, opt.init(p), _general_batch(), jax.random.PRNGKey(3)
    )
    assert np.isfinite(float(m["energy"]))
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(p2))
    loader = [_general_batch(0), _general_batch(1)]
    key = jax.random.PRNGKey(0)
    r1 = evaluate(p, s_drop, loader, {}, key, algorithm=algorithm)
    r2 = evaluate(p, s_none, loader, {}, key, algorithm=algorithm)
    assert r1 == r2
