"""
Model checkpointing for FabricPC: save/load of parameters, optimizer state,
graph structure, and (optionally) inference state.

A checkpoint is a **directory**:

    <ckpt>/
        metadata.json      # format magic + version, library versions, checksums
        params.msgpack     # GraphParams arrays (flax msgpack) — always present
        structure.json     # the GraphStructure snapshot          (optional)
        opt_state.msgpack  # optax optimizer state arrays          (optional)
        state.msgpack      # a settled GraphState                  (optional)

What is persisted, and how it round-trips:

1. ``params: GraphParams`` — the model; always saved. **Self-describing:** flax
   msgpack stores the pytree keyed by node / weight / bias names, so on load
   ``msgpack_restore`` rebuilds ``GraphParams``/``NodeParams`` *from the file
   alone* — no ``structure`` and **no model-definition classes required**. This
   is what lets a checkpoint's weights load even if a node class was renamed or
   moved since it was written.

2. ``opt_state: optax.OptState`` — optional; for *resuming training*. Its pytree
   *type* is defined by the optimizer, not the data on disk, so it is restored
   into a template ``optimizer.init(params)`` — pass the ``optimizer`` to
   :func:`load_checkpoint`. (flax matches by field name, so a wrong optimizer
   raises rather than silently misloads.)

3. ``structure: GraphStructure`` — optional, and reconstructed **best-effort** on
   load. It holds arbitrary Python class instances; we serialize a versioned JSON
   *snapshot* of ``NodeInfo`` (class import path + data) and restore each object
   via ``cls.__new__(cls)`` + direct attribute / ``.config`` assignment.
   ``__new__`` (not ``cls(**config)``) is required because several real
   constructors aren't round-trippable through ``__init__`` (``UniformInitializer``
   renames config keys, ``ZerosInitializer`` drops ``gain``, ``GlobalStateInit``
   nests a component). A *snapshot* (persisting ``node_order`` + muPC factors)
   rather than re-running the ``graph`` builder avoids the builder's best-effort
   topological sort on cyclic PC graphs. Saving the structure lets a model reload
   in a fresh process with no model code; **but any failure to rebuild it (a class
   that can no longer be imported, an evolved snapshot schema, a corrupt structure
   file) degrades to ``structure=None`` with a warning and still returns the
   params** — the snapshot never blocks weight recovery. (A structure that *does*
   rebuild but disagrees with the params on shape still raises — a real integrity
   error, not a graceful-degradation case.)

4. ``state: GraphState`` — optional; only for persistent-latent / Hopfield-style
   runs (the standard PC trainer re-initialises ``z`` each batch). Restored into
   an ``initialize_graph_state`` template, so it loads only when ``structure`` was
   rebuilt.

Robustness: a ``format`` magic string + ``format_version`` are checked before any
decode; each binary payload carries a sha256 ``checksum`` verified on load
(corruption → clear error); writes are crash-safe (staged in a temp dir, moved
with atomic ``os.replace``, and on overwrite the previous checkpoint is moved
aside and restored on failure so it is never lost); and when ``structure`` is
present the loaded params are validated against it (node-set + shapes).
No pickle anywhere.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import types
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import jax
import numpy as np
import flax.serialization as flax_ser

from fabricpc.core.types import (
    GraphParams,
    NodeParams,
    GraphStructure,
    NodeInfo,
    SlotInfo,
    EdgeInfo,
)
from fabricpc.core.activations import ActivationBase
from fabricpc.core.energy import EnergyFunctional
from fabricpc.core.initializers import InitializerBase
from fabricpc.core.inference import InferenceBase
from fabricpc.core.mupc import MuPCScalingFactors
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import (
    StateInitBase,
    initialize_graph_state,
)

# Magic string identifying our checkpoints — lets load reject an unrelated
# directory with a clear message before attempting to decode anything.
FORMAT = "fabricpc-checkpoint"

# Bump when the on-disk format changes incompatibly. load_checkpoint refuses any
# checkpoint whose format_version exceeds the version it understands.
FORMAT_VERSION = 1

# File names within the checkpoint directory.
_METADATA_FILE = "metadata.json"
_STRUCTURE_FILE = "structure.json"
_PARAMS_FILE = "params.msgpack"
_OPT_STATE_FILE = "opt_state.msgpack"
_STATE_FILE = "state.msgpack"

# Reserved key used to tag non-plain values in the JSON encoding. A config dict
# must not use this key (it is internal to the codec).
_TAG = "__fpc__"

# Component base classes whose instances are reconstructed from (class, config).
_COMPONENT_BASES = (
    ActivationBase,
    EnergyFunctional,
    InitializerBase,
    InferenceBase,
    StateInitBase,
)


PathLike = Union[str, os.PathLike]


class Checkpoint(NamedTuple):
    """Result of :func:`load_checkpoint`. Use named access (``ckpt.params``).

    - ``params`` always loads (self-describing — no model code required).
    - ``opt_state`` is ``None`` unless the file stored one *and* an ``optimizer``
      was passed to reconstruct its pytree types.
    - ``structure`` is the rebuilt graph, or ``None`` if the file stored none or
      a node/component class can no longer be imported (best-effort).
    - ``state`` is a restored ``GraphState``, or ``None`` unless the file stored
      one *and* ``structure`` was rebuilt to template it.
    - ``metadata`` is the user metadata dict passed at save time (``{}`` if none).
    """

    params: GraphParams
    opt_state: Optional[Any]
    structure: Optional[GraphStructure]
    state: Optional[Any]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Class path import / export
# ---------------------------------------------------------------------------
def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_class(path: str) -> type:
    module_name, _, qualname = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Malformed class path: {path!r}")
    module = importlib.import_module(module_name)
    obj: Any = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


# ---------------------------------------------------------------------------
# Recursive JSON codec for structure metadata
#
# Plain JSON scalars / lists / dicts pass through unchanged so the file stays
# human-readable; only values JSON cannot represent faithfully are tagged with
# the reserved _TAG key: tuples (to distinguish from lists) and component
# instances (class + config). Component configs are themselves encoded
# recursively, which transparently handles nested components (e.g. the
# InitializerBase stored inside GlobalStateInit's config).
# ---------------------------------------------------------------------------
def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    # numpy scalars -> python scalars (python float repr round-trips exactly)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, _COMPONENT_BASES):
        return {
            _TAG: "component",
            "class": _class_path(type(obj)),
            "config": _to_jsonable(dict(obj.config)),
        }
    if isinstance(obj, tuple):
        return {_TAG: "tuple", "items": [_to_jsonable(v) for v in obj]}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (dict, types.MappingProxyType)):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"Cannot serialize structure: dict key {k!r} is not a string."
                )
            if k == _TAG:
                raise ValueError(
                    f"Cannot serialize structure: config uses reserved key {_TAG!r}."
                )
            out[k] = _to_jsonable(v)
        return out
    raise ValueError(
        f"Cannot serialize value of type {type(obj).__name__!r} in graph "
        f"structure metadata. Supported: None, bool, int, float, str, tuple, "
        f"list, dict, and component instances (activation/energy/initializer/"
        f"inference/state-init). Arrays belong in params, not structure config."
    )


def _from_jsonable(spec: Any) -> Any:
    if isinstance(spec, list):
        return [_from_jsonable(v) for v in spec]
    if isinstance(spec, dict):
        tag = spec.get(_TAG)
        if tag is None:
            return {k: _from_jsonable(v) for k, v in spec.items()}
        if tag == "tuple":
            return tuple(_from_jsonable(v) for v in spec["items"])
        if tag == "component":
            cls = _import_class(spec["class"])
            config = _from_jsonable(spec["config"])
            return _reconstruct_component(cls, config)
        raise ValueError(f"Unknown encoding tag {tag!r} in structure file.")
    return spec  # None / bool / int / float / str


def _reconstruct_component(cls: type, config: Dict[str, Any]) -> Any:
    """Rebuild a component from its class and config, bypassing ``__init__``.

    All components only read from ``self.config``, so restoring that attribute
    yields a fully functional instance regardless of constructor quirks.
    ``StateInitBase`` stores a plain dict; the others use ``MappingProxyType`` —
    we match the original so immutability semantics are preserved.
    """
    obj = cls.__new__(cls)
    if issubclass(cls, StateInitBase):
        obj.config = config
    else:
        obj.config = types.MappingProxyType(config)
    return obj


# ---------------------------------------------------------------------------
# Structure encode / decode
# ---------------------------------------------------------------------------
def _encode_component_or_none(obj: Any) -> Any:
    return None if obj is None else _to_jsonable(obj)


def _encode_slot(slot: SlotInfo) -> Dict[str, Any]:
    return {
        "name": slot.name,
        "parent_node": slot.parent_node,
        "is_multi_input": slot.is_multi_input,
        "is_variance_scalable": slot.is_variance_scalable,
        "is_skip_connection": slot.is_skip_connection,
        "in_neighbors": list(slot.in_neighbors),
    }


def _decode_slot(d: Dict[str, Any]) -> SlotInfo:
    return SlotInfo(
        name=d["name"],
        parent_node=d["parent_node"],
        is_multi_input=d["is_multi_input"],
        is_variance_scalable=d["is_variance_scalable"],
        is_skip_connection=d["is_skip_connection"],
        in_neighbors=tuple(d["in_neighbors"]),
    )


def _encode_scaling(sc: Optional[MuPCScalingFactors]) -> Any:
    if sc is None:
        return None
    return {
        "forward_scale": {k: float(v) for k, v in sc.forward_scale.items()},
        "self_grad_scale": float(sc.self_grad_scale),
        "topdown_grad_scale": {k: float(v) for k, v in sc.topdown_grad_scale.items()},
        "weight_grad_scale": {k: float(v) for k, v in sc.weight_grad_scale.items()},
    }


def _decode_scaling(d: Optional[Dict[str, Any]]) -> Optional[MuPCScalingFactors]:
    if d is None:
        return None
    return MuPCScalingFactors(
        forward_scale=dict(d["forward_scale"]),
        self_grad_scale=d["self_grad_scale"],
        topdown_grad_scale=dict(d["topdown_grad_scale"]),
        weight_grad_scale=dict(d["weight_grad_scale"]),
    )


def _encode_node(node: Any) -> Dict[str, Any]:
    ni: NodeInfo = node.node_info
    return {
        "class": _class_path(type(node)),
        "name": ni.name,
        "shape": list(ni.shape),
        "node_type": ni.node_type,
        "node_config": _to_jsonable(dict(ni.node_config)),
        "activation": _encode_component_or_none(ni.activation),
        "energy": _encode_component_or_none(ni.energy),
        "latent_init": _encode_component_or_none(ni.latent_init),
        "weight_init": _encode_component_or_none(ni.weight_init),
        "slots": {name: _encode_slot(s) for name, s in ni.slots.items()},
        "in_degree": ni.in_degree,
        "out_degree": ni.out_degree,
        "in_edges": list(ni.in_edges),
        "out_edges": list(ni.out_edges),
        "scaling_config": _encode_scaling(ni.scaling_config),
    }


def _decode_node(d: Dict[str, Any]) -> Any:
    cls = _import_class(d["class"])
    node_config = types.MappingProxyType(_from_jsonable(d["node_config"]))
    node_info = NodeInfo(
        name=d["name"],
        shape=tuple(d["shape"]),
        node_type=d["node_type"],
        node_class=cls,
        node_config=node_config,
        activation=_from_jsonable(d["activation"]) if d["activation"] else None,
        energy=_from_jsonable(d["energy"]) if d["energy"] else None,
        latent_init=_from_jsonable(d["latent_init"]) if d["latent_init"] else None,
        weight_init=_from_jsonable(d["weight_init"]) if d["weight_init"] else None,
        slots={name: _decode_slot(s) for name, s in d["slots"].items()},
        in_degree=d["in_degree"],
        out_degree=d["out_degree"],
        in_edges=tuple(d["in_edges"]),
        out_edges=tuple(d["out_edges"]),
        scaling_config=_decode_scaling(d["scaling_config"]),
    )
    # Reconstruct the node instance by restoring its state directly (bypassing
    # __init__). Setting the final namespace-prefixed name verbatim avoids the
    # double-prefixing that re-running a constructor under a GraphNamespace would
    # cause.
    node = cls.__new__(cls)
    node._name = node_info.name
    node._shape = node_info.shape
    node._activation = node_info.activation
    node._energy = node_info.energy
    node._latent_init = node_info.latent_init
    node._weight_init = node_info.weight_init
    node._extra_config = node_config
    node._node_info = node_info
    return node


def _encode_edge(edge: EdgeInfo) -> Dict[str, Any]:
    return {
        "key": edge.key,
        "source": edge.source,
        "target": edge.target,
        "slot": edge.slot,
    }


def _decode_edge(d: Dict[str, Any]) -> EdgeInfo:
    return EdgeInfo(
        key=d["key"], source=d["source"], target=d["target"], slot=d["slot"]
    )


def serialize_structure(structure: GraphStructure) -> Dict[str, Any]:
    """Encode a :class:`GraphStructure` into a JSON-serializable dict."""
    return {
        "nodes": {name: _encode_node(node) for name, node in structure.nodes.items()},
        "edges": {key: _encode_edge(e) for key, e in structure.edges.items()},
        "task_map": dict(structure.task_map),
        "node_order": list(structure.node_order),
        "config": _to_jsonable(dict(structure.config)),
    }


def deserialize_structure(d: Dict[str, Any]) -> GraphStructure:
    """Decode a dict produced by :func:`serialize_structure` back into a
    :class:`GraphStructure`."""
    return GraphStructure(
        nodes={name: _decode_node(nd) for name, nd in d["nodes"].items()},
        edges={key: _decode_edge(ed) for key, ed in d["edges"].items()},
        task_map=dict(d["task_map"]),
        node_order=tuple(d["node_order"]),
        config=_from_jsonable(d["config"]),
    )


# ---------------------------------------------------------------------------
# Array I/O (isolated so the backend can be swapped for orbax later)
# ---------------------------------------------------------------------------
def _save_arrays(pytree: Any, file_path: Path) -> None:
    file_path.write_bytes(flax_ser.to_bytes(pytree))


def _load_arrays(target: Any, file_path: Path) -> Any:
    """Restore arrays into a *template* pytree (recovers exact NamedTuple types).

    Used for opt_state / GraphState, whose pytree types are defined by the
    optimizer / structure rather than by the data on disk.
    """
    return flax_ser.from_bytes(target, file_path.read_bytes())


def _restore_params(file_path: Path) -> GraphParams:
    """Rebuild ``GraphParams`` from the file alone — **no structure or model
    classes required**.

    ``flax`` msgpack stores the pytree as a nested dict keyed by the node /
    weight / bias names, so ``msgpack_restore`` recovers that dict (dtype
    preserved, incl. bfloat16/float16) and we wrap it back into the
    ``GraphParams``/``NodeParams`` types directly. This is what lets a checkpoint
    load even if a node class was renamed or moved since it was written.
    """
    restored = flax_ser.msgpack_restore(file_path.read_bytes())
    return GraphParams(
        nodes={
            name: NodeParams(
                weights={k: jax.numpy.asarray(v) for k, v in nd["weights"].items()},
                biases={k: jax.numpy.asarray(v) for k, v in nd["biases"].items()},
            )
            for name, nd in restored["nodes"].items()
        }
    )


def _sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Metadata & validation
# ---------------------------------------------------------------------------
def _keypath_str(path: Tuple[Any, ...]) -> str:
    return jax.tree_util.keystr(path)


def _leaf_shape_map(params: GraphParams) -> Dict[str, Tuple[int, ...]]:
    leaves = jax.tree_util.tree_flatten_with_path(params)[0]
    return {_keypath_str(p): tuple(np.shape(leaf)) for p, leaf in leaves}


def _validate_params_against_structure(
    params: GraphParams, structure: GraphStructure
) -> None:
    """Check the loaded params are compatible with a (re)built structure, raising
    a clear error on any node-set or shape mismatch — catches a structure whose
    node shapes drifted in code since the checkpoint was written."""
    got = _leaf_shape_map(params)
    want = _leaf_shape_map(initialize_params(structure, jax.random.PRNGKey(0)))

    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing or extra:
        raise ValueError(
            "Loaded params are incompatible with the structure. Missing leaves "
            f"(structure expects, file lacks): {missing}. Extra leaves (file has, "
            f"structure lacks): {extra}."
        )
    for path, want_shape in want.items():
        if got[path] != want_shape:
            raise ValueError(
                f"Shape mismatch at {path}: file has {got[path]}, structure "
                f"expects {want_shape}."
            )


def _warn_on_flax_drift(saved_flax: Optional[str]) -> None:
    """Soft-warn if the flax version that wrote the arrays differs (major.minor)
    from the one loading them — the msgpack array format is flax-owned."""
    if not saved_flax:
        return
    import flax

    current = flax.__version__
    if saved_flax.split(".")[:2] != current.split(".")[:2]:
        warnings.warn(
            f"Checkpoint arrays were written with flax {saved_flax}, but flax "
            f"{current} is loading them. The msgpack array format is usually "
            f"stable across versions, but verify the loaded parameters if you "
            f"see decode issues.",
            RuntimeWarning,
            stacklevel=2,
        )


def _library_versions() -> Dict[str, str]:
    import optax
    import flax

    versions = {
        "jax": jax.__version__,
        "optax": optax.__version__,
        "flax": flax.__version__,
    }
    try:  # fabricpc may not be pip-installed in a dev checkout
        from importlib.metadata import version as _pkg_version

        versions["fabricpc"] = _pkg_version("fabricpc")
    except Exception:
        versions["fabricpc"] = "unknown"
    return versions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: PathLike,
    params: GraphParams,
    *,
    opt_state: Optional[Any] = None,
    structure: Optional[GraphStructure] = None,
    state: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """Save a checkpoint to ``path``.

    Args:
        path: Destination directory for the checkpoint (created if absent).
        params: Network parameters (``GraphParams`` pytree) — always saved; this
            is the model.
        opt_state: Optimizer state from ``optimizer.init(params)``. Pass it to
            *resume training* later; omit for an inference-only model.
        structure: The graph structure. Optional: params load without it (they
            are self-describing), but saving it lets the model be reloaded in a
            fresh process with no model-definition code, and enables loading any
            saved ``state``.
        state: A settled ``GraphState`` to snapshot — only for persistent-latent /
            Hopfield-style runs; the standard PC trainer re-initialises ``z`` each
            batch, so it is not needed to resume normal training.
        metadata: Optional user metadata (any JSON-serializable dict) — e.g.
            epoch, accuracy, config. Returned by :func:`load_checkpoint`.
        overwrite: If False (default), raise ``FileExistsError`` when ``path``
            exists. If True, atomically replace it.

    Returns:
        The ``Path`` written.

    The write is crash-safe: files are staged in a sibling temp directory and
    moved into place only with atomic ``os.replace`` calls. On ``overwrite`` an
    existing checkpoint is moved aside (not deleted) and restored on failure, so a
    crash during the swap leaves the *previous* checkpoint intact, never nothing.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint path {str(path)!r} already exists. Pass overwrite=True "
            f"to replace it."
        )

    if metadata is not None:
        try:
            user_meta = json.loads(json.dumps(metadata))
        except TypeError as e:
            raise ValueError(
                f"metadata is not JSON-serializable: {e}. Use only "
                f"str/int/float/bool/list/dict (no arrays or custom objects)."
            ) from e
    else:
        user_meta = {}

    meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "library_versions": _library_versions(),
        "has_opt_state": opt_state is not None,
        "has_structure": structure is not None,
        "has_state": state is not None,
        "batch_size": int(state.batch_size) if state is not None else None,
        "user": user_meta,
        # checksums of the binary payloads, filled in after they are written.
        "checksums": {},
    }

    # Unique scratch names (pid + random) so concurrent saves to the same path
    # — even within one process — never share or clobber each other's staging.
    token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging = path.parent / f".{path.name}.tmp-{token}"
    backup = path.parent / f".{path.name}.bak-{token}"
    for scratch in (staging, backup):
        if scratch.exists():
            shutil.rmtree(scratch)
    staging.mkdir(parents=True)

    moved_to_backup = False
    try:
        _save_arrays(params, staging / _PARAMS_FILE)
        meta["checksums"][_PARAMS_FILE] = _sha256(staging / _PARAMS_FILE)
        if structure is not None:
            (staging / _STRUCTURE_FILE).write_text(
                json.dumps(serialize_structure(structure), indent=2)
            )
            meta["checksums"][_STRUCTURE_FILE] = _sha256(staging / _STRUCTURE_FILE)
        if opt_state is not None:
            _save_arrays(opt_state, staging / _OPT_STATE_FILE)
            meta["checksums"][_OPT_STATE_FILE] = _sha256(staging / _OPT_STATE_FILE)
        if state is not None:
            _save_arrays(state, staging / _STATE_FILE)
            meta["checksums"][_STATE_FILE] = _sha256(staging / _STATE_FILE)
        # metadata.json is written last (it carries the other files' checksums).
        (staging / _METADATA_FILE).write_text(json.dumps(meta, indent=2))

        # Crash-safe swap: each step is an atomic os.replace, and any existing
        # checkpoint is moved aside (not deleted) before the new one moves in, so
        # a crash between the two replaces leaves the previous checkpoint intact.
        if path.exists():
            os.replace(path, backup)
            moved_to_backup = True
        os.replace(staging, path)
    except BaseException:
        # The new checkpoint never landed — restore the previous one if we had
        # moved it aside, so `path` is never left empty.
        if moved_to_backup and not path.exists() and backup.exists():
            os.replace(backup, path)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        # Only discard the backup once `path` holds a valid checkpoint (the new
        # one landed, or the restore above succeeded). If `path` is somehow empty
        # — e.g. the restore itself was interrupted — keep the backup for manual
        # recovery rather than deleting the last copy.
        if backup.exists() and path.exists():
            shutil.rmtree(backup)
    return path


def _verify_checksum(path: Path, filename: str, meta: Dict[str, Any]) -> None:
    """Raise a clear error if a payload file's content doesn't match the hash
    recorded at save time (detects silent corruption that still decodes)."""
    expected = meta.get("checksums", {}).get(filename)
    if expected is None:
        return  # nothing recorded (e.g. older writer) — skip
    actual = _sha256(path / filename)
    if actual != expected:
        raise ValueError(
            f"Checkpoint file {filename!r} in {str(path)!r} is corrupt: its "
            f"checksum does not match the one recorded at save time."
        )


def load_checkpoint(
    path: PathLike,
    *,
    optimizer: Optional[Any] = None,
    strict: bool = True,
) -> Checkpoint:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    ``params`` always load — they are self-describing and need no ``structure``
    or model-definition code. ``structure`` is rebuilt best-effort (``None`` if
    the file stored none, or a node/component class can no longer be imported).

    Args:
        path: Checkpoint directory.
        optimizer: The optax optimizer used during training. Required to rebuild
            the *exact* optimizer-state pytree types for resuming (that type is
            defined by the optimizer, not the data on disk). If the file has
            optimizer state but none is given, behavior depends on ``strict``.
        strict: If True (default), raise when the file contains optimizer state
            but no ``optimizer`` was provided. If False, skip it (``opt_state``
            stays ``None``; params/structure still load).

    Returns:
        ``Checkpoint(params, opt_state, structure, state, metadata)`` — use named
        access. ``opt_state``/``structure``/``state`` are ``None`` when absent or
        not requestable (see the type's docstring).

    Raises:
        FileNotFoundError: If ``path`` or a required file is missing.
        ValueError: On a non-/future-format checkpoint, a corrupt payload, or a
            params/structure shape mismatch.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {str(path)!r}")
    for required in (_METADATA_FILE, _PARAMS_FILE):
        if not (path / required).is_file():
            raise FileNotFoundError(
                f"Checkpoint at {str(path)!r} is missing required file {required!r}."
            )

    # Identify + version-gate before decoding anything, so an unrelated directory
    # or a future-format file fails with a clear message, not an opaque error.
    try:
        meta = json.loads((path / _METADATA_FILE).read_text())
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError(
            f"Could not read {str(path)!r} as a FabricPC checkpoint: its "
            f"{_METADATA_FILE} is not valid JSON ({e})."
        ) from e
    if meta.get("format") != FORMAT:
        raise ValueError(
            f"{str(path)!r} is not a FabricPC checkpoint (expected format "
            f"{FORMAT!r}, found {meta.get('format')!r})."
        )
    file_version = meta.get("format_version")
    if file_version is None or file_version > FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version {file_version!r} is not supported by this "
            f"version of FabricPC (understands up to {FORMAT_VERSION}). The "
            f"checkpoint was likely written by a newer release."
        )

    # The msgpack array format is owned by flax; warn (don't fail) on drift.
    _warn_on_flax_drift(meta.get("library_versions", {}).get("flax"))

    # --- params: self-describing, no structure/classes required. ---
    _verify_checksum(path, _PARAMS_FILE, meta)
    try:
        params = _restore_params(path / _PARAMS_FILE)
    except Exception as e:
        raise ValueError(
            f"Could not read {_PARAMS_FILE} in {str(path)!r} as checkpoint "
            f"parameters: {e}"
        ) from e

    # --- structure: best-effort, and never blocks params. ANY failure to rebuild
    #     it — a renamed/moved class, an evolved snapshot schema, a corrupt
    #     structure file — degrades to structure=None with a warning rather than
    #     propagating, because the (self-describing) params are the durable
    #     artifact. A structure that *does* rebuild but is inconsistent with the
    #     params (shape/node-set mismatch) is a genuine integrity error and still
    #     raises, so tampering/corruption that survives rebuild is not silenced. ---
    structure = None
    if (path / _STRUCTURE_FILE).is_file():
        try:
            _verify_checksum(path, _STRUCTURE_FILE, meta)
            structure = deserialize_structure(
                json.loads((path / _STRUCTURE_FILE).read_text())
            )
        except Exception as e:
            structure = None
            warnings.warn(
                f"Checkpoint at {str(path)!r} stores a structure that could not "
                f"be rebuilt ({type(e).__name__}: {e}) — e.g. a node/component "
                f"class was renamed or moved, the snapshot schema changed, or the "
                f"structure file is corrupt. Returning params only (structure="
                f"None); any saved state cannot be restored.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            _validate_params_against_structure(params, structure)

    # --- opt_state: restored into optimizer.init(params); needs the optimizer. ---
    opt_state = None
    opt_file = path / _OPT_STATE_FILE
    if opt_file.is_file():
        if optimizer is None:
            if strict:
                raise ValueError(
                    f"Checkpoint at {str(path)!r} contains optimizer state, but no "
                    f"`optimizer` was provided. Pass the optax optimizer used for "
                    f"training so its state pytree can be reconstructed, or call "
                    f"with strict=False to load params only."
                )
            # Non-strict: skip opt_state.
        else:
            _verify_checksum(path, _OPT_STATE_FILE, meta)
            opt_state = _load_arrays(optimizer.init(params), opt_file)

    # --- state: restored into an initialize_graph_state template; needs structure. ---
    state = None
    state_file = path / _STATE_FILE
    if state_file.is_file() and structure is not None:
        _verify_checksum(path, _STATE_FILE, meta)
        template = initialize_graph_state(
            structure,
            meta["batch_size"],
            jax.random.PRNGKey(0),
            clamps={},
            params=params,
        )
        state = _load_arrays(template, state_file)

    return Checkpoint(
        params=params,
        opt_state=opt_state,
        structure=structure,
        state=state,
        metadata=meta.get("user", {}),
    )
