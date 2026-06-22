"""Parity check: unified_trainer vs the original train.py / train_backprop.py.

Same seed + same data must yield identical params, metric histories, and eval
metrics. This is the review evidence that the unified file reproduces both
algorithms exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation, TanhActivation
from fabricpc.core.inference import InferenceSGD

# Originals (load submodules via importlib; the package __init__ re-exports
# functions of the same name which shadow the module attributes)
import importlib

orig_pc = importlib.import_module("fabricpc.training.train")
orig_bp = importlib.import_module("fabricpc.training.train_backprop")
# Unified
uni = importlib.import_module("fabricpc.training.unified_trainer")


def trees_allclose(a, b, atol=1e-6):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    return all(
        np.allclose(np.asarray(x), np.asarray(y), atol=atol) for x, y in zip(la, lb)
    )


def pc_graph(rng_key):
    x = IdentityNode(shape=(4,), name="x")
    h = Linear(shape=(8,), activation=TanhActivation(), name="h")
    y = Linear(shape=(2,), name="y")
    structure = graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
    )
    return initialize_params(structure, rng_key), structure


def bp_graph(rng_key):
    inp = Linear(shape=(10,), name="input")
    h = Linear(shape=(20,), activation=ReLUActivation(), name="hidden")
    out = Linear(shape=(5,), activation=SoftmaxActivation(), name="output")
    structure = graph(
        nodes=[inp, h, out],
        edges=[
            Edge(source=inp, target=h.slot("in")),
            Edge(source=h, target=out.slot("in")),
        ],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(),
    )
    return initialize_params(structure, rng_key), structure


def loader_pc(rng_key, n=4, bs=8):
    x = jax.random.normal(rng_key, (n * bs, 4))
    y = jax.random.normal(rng_key, (n * bs, 2))
    return [(x[i * bs : (i + 1) * bs], y[i * bs : (i + 1) * bs]) for i in range(n)]


def loader_bp(rng_key, n=4, bs=8, cls=5):
    batches = []
    for _ in range(n):
        k1, k2, rng_key = jax.random.split(rng_key, 3)
        x = jax.random.normal(k1, (bs, 10))
        y = jax.nn.one_hot(jax.random.randint(k2, (bs,), 0, cls), cls)
        batches.append({"x": x, "y": y})
    return batches


def main():
    seed = jax.random.PRNGKey(0)
    pk, tk = jax.random.split(seed)

    # ── PC parity ──
    params, structure = pc_graph(pk)
    loader = loader_pc(tk)
    cfg = {"num_epochs": 2}

    p_o, it_o, ep_o = orig_pc.train_pcn(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        cfg,
        tk,
        verbose=False,
        use_tqdm=False,
    )
    p_u, it_u, ep_u = uni.train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        cfg,
        tk,
        algorithm="pc",
        verbose=False,
        use_tqdm=False,
    )
    assert trees_allclose(p_o, p_u), "PC params differ"
    assert np.allclose(
        np.asarray(it_o), np.asarray(it_u), atol=1e-6
    ), "PC energy history differs"
    print("[PC] train parity OK  (energy[0]=%s)" % np.asarray(it_u[0]))

    e_o = orig_pc.evaluate_pcn(p_o, structure, loader, cfg, tk)
    e_u = uni.evaluate(p_u, structure, loader, cfg, tk, algorithm="pc")
    assert np.allclose(e_o["energy"], e_u["energy"], atol=1e-6) and np.allclose(
        e_o["accuracy"], e_u["accuracy"], atol=1e-6
    ), (e_o, e_u)
    print("[PC] eval parity OK   ", e_o, "==", e_u)

    # ── Backprop parity ──
    params, structure = bp_graph(pk)
    loader = loader_bp(tk)
    cfg = {"num_epochs": 2, "loss_type": "cross_entropy"}

    p_o, it_o, ep_o = orig_bp.train_backprop(
        params, structure, loader, optax.adam(1e-2), cfg, tk, verbose=False
    )
    p_u, it_u, ep_u = uni.train(
        params,
        structure,
        loader,
        optax.adam(1e-2),
        cfg,
        tk,
        algorithm="backprop",
        verbose=False,
        use_tqdm=False,
    )
    assert trees_allclose(p_o, p_u), "BP params differ"
    assert np.allclose(
        np.asarray(it_o), np.asarray(it_u), atol=1e-6
    ), "BP loss history differs"
    print("[BP] train parity OK  (loss[0]=%s)" % np.asarray(it_u[0]))

    e_o = orig_bp.evaluate_backprop(p_o, structure, loader, cfg, tk)
    e_u = uni.evaluate(p_u, structure, loader, cfg, tk, algorithm="backprop")
    assert np.allclose(float(e_o["loss"]), float(e_u["loss"]), atol=1e-6), (e_o, e_u)
    assert np.allclose(float(e_o["accuracy"]), float(e_u["accuracy"]), atol=1e-6), (
        e_o,
        e_u,
    )
    assert np.allclose(e_o["perplexity"], e_u["perplexity"], atol=1e-5), (e_o, e_u)
    print(
        "[BP] eval parity OK   ",
        {k: float(v) for k, v in e_o.items()},
        "==",
        {k: float(v) for k, v in e_u.items()},
    )

    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
