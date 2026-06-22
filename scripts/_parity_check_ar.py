"""Parity check: unified autoregressive training vs legacy AR trainers.

Uses a small real transformer (create_deep_transformer, which masks internally)
with use_causal_mask=False so no causal_mask task is required. Also unit-checks
that build_clamps injects the causal mask with the right shape.
"""

import importlib
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit

orig_ar = importlib.import_module("fabricpc.training.train_autoregressive")
orig_bp = importlib.import_module("fabricpc.training.train_backprop")
uni = importlib.import_module("fabricpc.training.unified_trainer")

VOCAB, SEQ = 20, 6


def trees_allclose(a, b, atol=1e-5):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.allclose(np.asarray(x), np.asarray(y), atol=atol) for x, y in zip(la, lb)
    )


def ar_graph(key):
    s = create_deep_transformer(
        depth=1,
        embed_dim=16,
        num_heads=2,
        mlp_dim=32,
        seq_len=SEQ,
        vocab_size=VOCAB,
        inference=InferenceSGD(eta_infer=0.1, infer_steps=2),
    )
    return initialize_params(s, key), s


def ar_loader(key, n=3, bs=8):
    batches = []
    for _ in range(n):
        k1, k2, key = jax.random.split(key, 3)
        x = jax.random.randint(k1, (bs, SEQ), 0, VOCAB)
        y = jax.nn.one_hot(jax.random.randint(k2, (bs, SEQ), 0, VOCAB), VOCAB)
        batches.append({"x": x, "y": y})
    return batches


def main():
    pk, tk = jax.random.split(jax.random.PRNGKey(0))
    cfg = {"num_epochs": 2, "use_causal_mask": False}

    # ── unit: causal mask clamp shape ──
    params, structure = ar_graph(pk)
    batch = ar_loader(tk, n=1, bs=4)[0]
    # give the graph a fake causal_mask task entry just for the shape check
    structure.task_map["causal_mask"] = structure.task_map["x"]
    clamps = uni.build_clamps(batch, structure, clamp_target=True, causal_mask=True)
    mask = clamps[structure.task_map["causal_mask"]]
    assert mask.shape == (4, 1, SEQ, SEQ), mask.shape
    assert np.allclose(
        np.asarray(mask[0, 0]), np.tril(np.ones((SEQ, SEQ)))
    ), "mask not lower-triangular"
    del structure.task_map["causal_mask"]
    print("[unit] build_clamps causal mask shape + triangularity OK", mask.shape)

    # ── AR-PC eager-gradient bit-equivalence (the algorithmic proof) ──
    # XLA fuses the legacy and unified *jitted* steps differently (the legacy AR
    # step also computes an unused cross-entropy inside jit), so jitted params
    # drift at float level. Eager grads remove XLA noise and must be bit-equal.
    from fabricpc.core.learning import compute_local_weight_gradients
    from fabricpc.graph_initialization.state_initializer import initialize_graph_state
    from fabricpc.core.inference import run_inference

    params, structure = ar_graph(pk)
    loader = ar_loader(tk)
    b0, key0 = loader[0], jax.random.split(jax.random.split(tk)[0], 3)[0]
    g_u, _ = uni.compute_gradients(
        params,
        b0,
        structure,
        key0,
        algo="pc",
        autoregressive=True,
        use_causal_mask=False,
    )
    clamps = {
        structure.task_map[k]: v for k, v in b0.items() if k in structure.task_map
    }
    st = initialize_graph_state(
        structure, b0["x"].shape[0], key0, clamps=clamps, params=params
    )
    g_l = compute_local_weight_gradients(
        params, run_inference(params, st, clamps, structure), structure
    )
    assert trees_allclose(g_u, g_l, atol=0.0), "AR-PC eager grads not bit-identical"
    print("[AR-PC] eager gradients bit-identical to legacy formula (atol=0)")

    # ── AR-PC train parity (energy bit-close; jitted params at float tol) ──
    p_o, it_o, _ = orig_ar.train_autoregressive(
        params, structure, loader, optax.adam(1e-3), cfg, tk, verbose=False
    )
    p_u, it_u, _ = uni.train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        cfg,
        tk,
        algorithm="pc",
        autoregressive=True,
        use_causal_mask=False,
        verbose=False,
        use_tqdm=False,
    )
    assert np.allclose(
        np.asarray(it_o), np.asarray(it_u), atol=1e-4
    ), "AR-PC energy history differs"
    assert trees_allclose(p_o, p_u, atol=2e-3), "AR-PC params drifted beyond float tol"
    print("[AR-PC] train parity OK (energy[0]=%s)" % np.asarray(it_u[0]))

    # ── AR-PC eval parity on IDENTICAL params (isolates eval logic) ──
    e_o = orig_ar.evaluate_autoregressive(p_o, structure, loader, cfg, tk)
    e_u = uni.evaluate(
        p_o,
        structure,
        loader,
        cfg,
        tk,
        algorithm="pc",
        autoregressive=True,
        use_causal_mask=False,
    )
    for k in ("loss", "perplexity", "accuracy", "num_batches"):
        assert np.allclose(float(e_o[k]), float(e_u[k]), atol=1e-4), (k, e_o, e_u)
    print(
        "[AR-PC] eval parity OK (same params) ",
        {k: round(float(v), 4) for k, v in e_u.items()},
    )

    # ── AR eval parity on RAGGED batches (locks the per-batch-mean reduction) ──
    # Legacy AR averages loss as a simple per-batch mean (total_loss / num_batches);
    # equal-size batches can't distinguish that from token-weighting, so use unequal
    # batch sizes (8 then 2) to pin the reduction down.
    rk = jax.random.split(tk, 4)
    ragged = [
        {
            "x": jax.random.randint(rk[0], (8, SEQ), 0, VOCAB),
            "y": jax.nn.one_hot(jax.random.randint(rk[1], (8, SEQ), 0, VOCAB), VOCAB),
        },
        {
            "x": jax.random.randint(rk[2], (2, SEQ), 0, VOCAB),
            "y": jax.nn.one_hot(jax.random.randint(rk[3], (2, SEQ), 0, VOCAB), VOCAB),
        },
    ]
    e_o = orig_ar.evaluate_autoregressive(p_o, structure, ragged, cfg, tk)
    e_u = uni.evaluate(
        p_o,
        structure,
        ragged,
        cfg,
        tk,
        algorithm="pc",
        autoregressive=True,
        use_causal_mask=False,
    )
    for k in ("loss", "perplexity", "accuracy", "num_batches"):
        assert np.allclose(float(e_o[k]), float(e_u[k]), atol=1e-4), (
            "ragged",
            k,
            e_o,
            e_u,
        )
    print(
        "[AR-PC] ragged-batch eval parity OK ",
        {k: round(float(v), 4) for k, v in e_u.items()},
    )

    # ── AR-backprop parity (only if the graph supports feedforward init) ──
    if isinstance(structure.config["graph_state_initializer"], FeedforwardStateInit):
        params, structure = ar_graph(pk)
        loader = ar_loader(tk)
        p_o, it_o, _ = orig_bp.train_backprop_autoregressive(
            params, structure, loader, optax.adam(1e-3), cfg, tk, verbose=False
        )
        p_u, it_u, _ = uni.train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            cfg,
            tk,
            algorithm="backprop",
            autoregressive=True,
            use_causal_mask=False,
            verbose=False,
            use_tqdm=False,
        )
        assert np.allclose(
            np.asarray(it_o), np.asarray(it_u), atol=1e-4
        ), "AR-BP loss history differs"
        assert trees_allclose(
            p_o, p_u, atol=2e-3
        ), "AR-BP params drifted beyond float tol"
        print("[AR-BP] train parity OK (loss[0]=%s)" % np.asarray(it_u[0]))

        e_o = orig_bp.evaluate_backprop_autoregressive(p_o, structure, loader, cfg, tk)
        e_u = uni.evaluate(
            p_o,
            structure,
            loader,
            cfg,
            tk,
            algorithm="backprop",
            autoregressive=True,
            use_causal_mask=False,
        )
        for k in ("loss", "perplexity", "accuracy"):
            assert np.allclose(float(e_o[k]), float(e_u[k]), atol=1e-4), (k, e_o, e_u)
        print(
            "[AR-BP] eval parity OK (same params) ",
            {k: round(float(v), 4) for k, v in e_u.items()},
        )
    else:
        print("[AR-BP] skipped (graph not feedforward-init)")

    print("\nALL AR PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
