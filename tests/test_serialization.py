#!/usr/bin/env python3
"""
Tests for model checkpointing: fabricpc.serialization.save_checkpoint / load_checkpoint.

Covers self-describing param round-trips (exact values, types, and dtypes — no
structure required), optional best-effort structure recovery (incl. the
class-rename degradation), optional GraphState, optimizer-state resume parity,
and the error / edge-case contract (checksums, format magic, version, overwrite
crash-safety, metadata).
"""

import json

import pytest
import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import Linear, IdentityNode, ConvNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import (
    GlobalStateInit,
    initialize_graph_state,
)
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import (
    XavierInitializer,
    UniformInitializer,
    NormalInitializer,
)
from fabricpc.core.mupc import MuPCConfig
from fabricpc import serialization
from fabricpc.serialization import (
    save_checkpoint,
    load_checkpoint,
    serialize_structure,
    deserialize_structure,
    FORMAT,
    FORMAT_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
def _leaves_equal(a, b):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(jnp.array_equal(x, y) for x, y in zip(la, lb))


def _mlp_structure():
    """Small MLP whose hidden layer uses UniformInitializer — a component whose
    config keys do NOT match its constructor params, exercising the
    reconstruct-object-state (not replay-constructor) code path."""
    x = IdentityNode(shape=(12,), name="pixels")
    h = Linear(
        shape=(8,),
        activation=SigmoidActivation(),
        name="h",
        weight_init=UniformInitializer(min_val=-0.2, max_val=0.2),
    )
    y = Linear(
        shape=(4,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )


@pytest.fixture
def mlp():
    structure = _mlp_structure()
    params = initialize_params(structure, jax.random.PRNGKey(0))
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(params)
    # Take one real step so opt_state holds non-trivial moments.
    grads = jax.tree_util.tree_map(lambda p: jnp.full_like(p, 0.1), params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return structure, params, optimizer, opt_state


# ---------------------------------------------------------------------------
# Parameter round-trips — self-describing (no structure needed)
# ---------------------------------------------------------------------------
def test_params_roundtrip_exact_and_typed_without_structure(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params)  # params only, no structure
    loaded = load_checkpoint(ckpt)
    assert _leaves_equal(params, loaded.params)
    assert isinstance(loaded.params, GraphParams)
    assert all(isinstance(np_, NodeParams) for np_ in loaded.params.nodes.values())
    assert loaded.structure is None  # none was saved
    assert loaded.opt_state is None


def test_params_dtype_fidelity_bf16_fp16(tmp_path):
    """Self-describing params must round-trip bf16/fp16 and 0-d scalars exactly."""
    params = GraphParams(
        nodes={
            "h": NodeParams(
                weights={"in->h": jnp.array([[1.5, -2.25]], dtype=jnp.bfloat16)},
                biases={"b": jnp.array([0.5, -0.5], dtype=jnp.float16)},
            ),
            "x": NodeParams(weights={}, biases={}),  # terminal node, no params
        }
    )
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params)
    loaded = load_checkpoint(ckpt).params
    assert _leaves_equal(params, loaded)
    assert loaded.nodes["h"].weights["in->h"].dtype == jnp.bfloat16
    assert loaded.nodes["h"].biases["b"].dtype == jnp.float16


def test_opt_state_roundtrip_exact_typed_and_usable(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, opt_state=opt_state, structure=structure)
    loaded = load_checkpoint(ckpt, optimizer=optimizer)
    assert _leaves_equal(opt_state, loaded.opt_state)
    assert type(loaded.opt_state) is type(opt_state)
    grads = jax.tree_util.tree_map(jnp.ones_like, loaded.params)
    optimizer.update(grads, loaded.opt_state, loaded.params)


def test_metadata_returned(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, metadata={"epoch": 7, "acc": 0.98})
    loaded = load_checkpoint(ckpt)
    assert loaded.metadata == {"epoch": 7, "acc": 0.98}


# ---------------------------------------------------------------------------
# Structure fidelity (optional payload)
# ---------------------------------------------------------------------------
def test_structure_full_fidelity(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure)
    loaded = load_checkpoint(ckpt).structure
    assert loaded is not None

    assert set(loaded.nodes) == set(structure.nodes)
    assert set(loaded.edges) == set(structure.edges)
    assert loaded.task_map == structure.task_map
    assert loaded.node_order == structure.node_order

    for name, node in structure.nodes.items():
        ln = loaded.nodes[name]
        assert type(ln) is type(node)
        assert ln.name == node.name
        assert ln.shape == node.shape
        oi, li = node.node_info, ln.node_info
        assert li.in_edges == oi.in_edges
        assert li.out_edges == oi.out_edges
        assert type(li.activation) is type(oi.activation)
        assert type(li.energy) is type(oi.energy)
        assert li.slots.keys() == oi.slots.keys()

    inf_o = structure.config["inference"]
    inf_l = loaded.config["inference"]
    assert type(inf_l) is type(inf_o)
    assert dict(inf_l.config) == dict(inf_o.config)


def test_violator_initializer_roundtrips(mlp, tmp_path):
    """UniformInitializer renames its config keys (min/max vs min_val/max_val),
    so cls(**config) would raise. The snapshot/__new__ path must restore it."""
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure)
    loaded = load_checkpoint(ckpt).structure
    wi = loaded.nodes["h"].node_info.weight_init
    assert isinstance(wi, UniformInitializer)
    assert dict(wi.config) == {"min": -0.2, "max": 0.2}


def test_reconstructed_structure_reinitializes_identically(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure)
    loaded = load_checkpoint(ckpt).structure
    p_orig = initialize_params(structure, jax.random.PRNGKey(7))
    p_recon = initialize_params(loaded, jax.random.PRNGKey(7))
    assert _leaves_equal(p_orig, p_recon)


def test_nested_component_state_init_roundtrips(tmp_path):
    """GlobalStateInit nests an InitializerBase inside its config — the hardest
    reconstruction case. It must round-trip."""
    x = IdentityNode(shape=(6,), name="x")
    y = Linear(shape=(3,), activation=SigmoidActivation(), name="y")
    structure = graph(
        nodes=[x, y],
        edges=[Edge(source=x, target=y.slot("in"))],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
        graph_state_initializer=GlobalStateInit(
            initializer=NormalInitializer(std=0.123)
        ),
    )
    params = initialize_params(structure, jax.random.PRNGKey(0))
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure)
    loaded = load_checkpoint(ckpt).structure
    si = loaded.config["graph_state_initializer"]
    assert isinstance(si, GlobalStateInit)
    nested = si.config["initializer"]
    assert isinstance(nested, NormalInitializer)
    assert dict(nested.config)["std"] == 0.123


def test_conv_node_config_and_mupc_roundtrip(tmp_path):
    """Conv2D stores tuple kernel_size/stride/padding in node_config; muPC
    attaches MuPCScalingFactors — both must round-trip exactly."""
    img = IdentityNode(shape=(8, 8, 1), name="img")
    conv = ConvNode(shape=(8, 8, 4), name="conv", kernel_size=(3, 3), padding="SAME")
    out = Linear(
        shape=(5,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="out",
        flatten_input=True,
    )
    structure = graph(
        nodes=[img, conv, out],
        edges=[
            Edge(source=img, target=conv.slot("in")),
            Edge(source=conv, target=out.slot("in")),
        ],
        task_map=TaskMap(x=img, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
        scaling=MuPCConfig(),
    )
    params = initialize_params(structure, jax.random.PRNGKey(1))
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure)
    loaded = load_checkpoint(ckpt).structure

    cfg = dict(loaded.nodes["conv"].node_info.node_config)
    assert cfg["kernel_size"] == (3, 3)  # tuple, not list
    assert cfg["stride"] == (1, 1)
    assert cfg["padding"] == "SAME"

    sc_o = structure.nodes["conv"].node_info.scaling_config
    sc_l = loaded.nodes["conv"].node_info.scaling_config
    assert sc_l is not None
    assert sc_l.forward_scale == sc_o.forward_scale
    assert sc_l.self_grad_scale == sc_o.self_grad_scale


def test_serialize_structure_is_json_roundtrippable(mlp):
    structure = mlp[0]
    payload = serialize_structure(structure)
    text = json.dumps(payload)
    restored = deserialize_structure(json.loads(text))
    assert restored.node_order == structure.node_order
    assert set(restored.nodes) == set(structure.nodes)


# ---------------------------------------------------------------------------
# R1: params load even when a structure class is gone (best-effort structure)
# ---------------------------------------------------------------------------
def test_renamed_class_degrades_to_params_only(mlp, tmp_path):
    """If a node class can no longer be imported, structure recovery must degrade
    to None (with a warning) WITHOUT blocking the (self-describing) params."""
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure)

    # Simulate a class rename/move: point a node at a non-existent class path.
    spath = ckpt / "structure.json"
    payload = json.loads(spath.read_text())
    payload["nodes"]["h"]["class"] = "fabricpc.nodes.linear.RenamedAwayLinear"
    spath.write_text(json.dumps(payload))
    # Keep the checksum consistent so we exercise the import-failure path, not
    # the corruption path.
    mpath = ckpt / "metadata.json"
    meta = json.loads(mpath.read_text())
    meta["checksums"]["structure.json"] = serialization._sha256(spath)
    mpath.write_text(json.dumps(meta))

    with pytest.warns(RuntimeWarning, match="could not"):
        loaded = load_checkpoint(ckpt)
    assert loaded.structure is None
    assert _leaves_equal(params, loaded.params)  # weights still recovered


def test_unrebuildable_structure_degrades_to_params_only(mlp, tmp_path):
    """A structure that fails to rebuild for a NON-import reason (evolved/garbled
    snapshot schema) must still degrade to structure=None — not crash the load
    and take the self-describing params down with it."""
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure)

    # Corrupt the snapshot in a way that breaks rebuild but isn't an import error:
    # drop a required NodeInfo field. Keep the checksum consistent so we exercise
    # the rebuild-failure path, not the corruption path.
    spath = ckpt / "structure.json"
    payload = json.loads(spath.read_text())
    del payload["nodes"]["h"]["shape"]  # -> KeyError deep in _decode_node
    spath.write_text(json.dumps(payload))
    mpath = ckpt / "metadata.json"
    meta = json.loads(mpath.read_text())
    meta["checksums"]["structure.json"] = serialization._sha256(spath)
    mpath.write_text(json.dumps(meta))

    with pytest.warns(RuntimeWarning, match="could not be rebuilt"):
        loaded = load_checkpoint(ckpt)
    assert loaded.structure is None
    assert _leaves_equal(params, loaded.params)


# ---------------------------------------------------------------------------
# GraphState (optional)
# ---------------------------------------------------------------------------
def test_graph_state_roundtrip(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    state = initialize_graph_state(
        structure, 3, jax.random.PRNGKey(5), clamps={}, params=params
    )
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure, state=state)
    loaded = load_checkpoint(ckpt)
    assert loaded.state is not None
    assert _leaves_equal(state, loaded.state)
    assert loaded.state.batch_size == state.batch_size


def test_graph_state_skipped_without_structure(mlp, tmp_path):
    """state restoration needs a structure template; if structure can't be
    rebuilt, state is skipped rather than erroring."""
    structure, params, optimizer, opt_state = mlp
    state = initialize_graph_state(
        structure, 2, jax.random.PRNGKey(5), clamps={}, params=params
    )
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure, state=state)
    # Drop the structure file so structure=None on load.
    (ckpt / "structure.json").unlink()
    meta = json.loads((ckpt / "metadata.json").read_text())
    meta["checksums"].pop("structure.json", None)
    (ckpt / "metadata.json").write_text(json.dumps(meta))
    loaded = load_checkpoint(ckpt)
    assert loaded.structure is None
    assert loaded.state is None
    assert _leaves_equal(params, loaded.params)


# ---------------------------------------------------------------------------
# Resume-training parity
# ---------------------------------------------------------------------------
def test_resume_training_parity(mlp, tmp_path):
    """Train straight through == train 1 step -> save -> load -> train 1 step,
    bitwise, for both params and opt_state."""
    from fabricpc.training import train_step

    structure, params0, optimizer, opt_state0 = mlp
    batch = {
        "x": jax.random.normal(jax.random.PRNGKey(11), (5, 12)),
        "y": jax.nn.one_hot(jnp.array([0, 1, 2, 3, 0]), 4),
    }
    k1, k2 = jax.random.split(jax.random.PRNGKey(99))

    p, o = params0, opt_state0
    p, o, _, _ = train_step(p, o, batch, structure, optimizer, k1)
    p_ref, o_ref, _, _ = train_step(p, o, batch, structure, optimizer, k2)

    p, o = params0, opt_state0
    p, o, _, _ = train_step(p, o, batch, structure, optimizer, k1)
    ckpt = tmp_path / "resume"
    save_checkpoint(ckpt, p, opt_state=o, structure=structure)
    loaded = load_checkpoint(ckpt, optimizer=optimizer)
    p_res, o_res, _, _ = train_step(
        loaded.params, loaded.opt_state, batch, loaded.structure, optimizer, k2
    )

    assert _leaves_equal(p_ref, p_res)
    assert _leaves_equal(o_ref, o_res)


# ---------------------------------------------------------------------------
# Error / edge-case contract
# ---------------------------------------------------------------------------
def test_strict_requires_optimizer_for_opt_state(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, opt_state=opt_state, structure=structure)
    with pytest.raises(ValueError, match="optimizer state"):
        load_checkpoint(ckpt)


def test_non_strict_skips_opt_state(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, opt_state=opt_state, structure=structure)
    loaded = load_checkpoint(ckpt, strict=False)
    assert loaded.opt_state is None
    assert _leaves_equal(params, loaded.params)


def test_wrong_optimizer_raises(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, opt_state=opt_state, structure=structure)
    # adamw state has fields plain sgd lacks -> flax field-name mismatch raises.
    with pytest.raises(ValueError):
        load_checkpoint(ckpt, optimizer=optax.sgd(1e-3))


def test_overwrite_guard_and_force(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, opt_state=opt_state, structure=structure)
    with pytest.raises(FileExistsError):
        save_checkpoint(ckpt, params, structure=structure)
    save_checkpoint(ckpt, params, structure=structure, overwrite=True)  # no opt_state
    assert load_checkpoint(ckpt).opt_state is None


def test_non_json_metadata_raises_actionable(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_checkpoint(ckpt, params, metadata={"bad": {1, 2, 3}})  # set is not JSON
    assert not ckpt.exists()  # nothing partial written


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist")


def test_missing_required_file_raises(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, structure=structure)
    (ckpt / "params.msgpack").unlink()
    with pytest.raises(FileNotFoundError, match="params.msgpack"):
        load_checkpoint(ckpt)


def test_not_a_checkpoint_rejected(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params)
    mpath = ckpt / "metadata.json"
    meta = json.loads(mpath.read_text())
    meta["format"] = "something-else"
    mpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not a FabricPC checkpoint"):
        load_checkpoint(ckpt)


def test_future_format_version_rejected(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params)
    mpath = ckpt / "metadata.json"
    meta = json.loads(mpath.read_text())
    meta["format_version"] = FORMAT_VERSION + 1
    mpath.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(ckpt)


def test_corrupt_payload_detected_by_checksum(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params)
    (ckpt / "params.msgpack").write_bytes(b"corrupted-not-msgpack")
    with pytest.raises(ValueError, match="corrupt"):
        load_checkpoint(ckpt)


def test_params_structure_shape_mismatch_rejected(tmp_path):
    """If structure.json describes a model whose param shapes disagree with the
    saved params, validation must raise a clear shape error."""
    s1 = _mlp_structure()
    p1 = initialize_params(s1, jax.random.PRNGKey(0))
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, p1, structure=s1)

    # Build a structurally different model and splice its structure.json in
    # (keeping the checksum consistent so we hit the shape check, not corruption).
    x = IdentityNode(shape=(12,), name="pixels")
    h = Linear(shape=(99,), activation=SigmoidActivation(), name="h")  # 8 -> 99
    y = Linear(
        shape=(4,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
    )
    s2 = graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    spath = ckpt / "structure.json"
    spath.write_text(json.dumps(serialize_structure(s2)))
    mpath = ckpt / "metadata.json"
    meta = json.loads(mpath.read_text())
    meta["checksums"]["structure.json"] = serialization._sha256(spath)
    mpath.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="[Ss]hape"):
        load_checkpoint(ckpt)


def test_no_scratch_dirs_left_on_success(mlp, tmp_path):
    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, opt_state=opt_state, structure=structure)
    leftovers = [
        p
        for p in tmp_path.iterdir()
        if p.name.startswith(".ckpt.tmp-") or p.name.startswith(".ckpt.bak-")
    ]
    assert leftovers == []


def test_overwrite_crash_preserves_old_checkpoint(mlp, tmp_path, monkeypatch):
    """If the process dies mid-swap during an overwrite, the previous checkpoint
    must survive intact (not be destroyed and replaced by nothing)."""
    import os as _os

    structure, params, optimizer, opt_state = mlp
    ckpt = tmp_path / "ckpt"
    save_checkpoint(ckpt, params, opt_state=opt_state, structure=structure)

    real_replace = _os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # fail the staging -> path move
            raise RuntimeError("simulated crash during swap")
        return real_replace(src, dst)

    monkeypatch.setattr(serialization.os, "replace", flaky_replace)
    with pytest.raises(RuntimeError, match="simulated crash"):
        save_checkpoint(ckpt, params, structure=structure, overwrite=True)

    monkeypatch.undo()
    loaded = load_checkpoint(ckpt, optimizer=optimizer)
    assert _leaves_equal(params, loaded.params)
    assert loaded.opt_state is not None  # surviving checkpoint is the OLD one
    leftovers = [
        p
        for p in tmp_path.iterdir()
        if p.name.startswith(".ckpt.tmp-") or p.name.startswith(".ckpt.bak-")
    ]
    assert leftovers == []
