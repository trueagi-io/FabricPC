# Design culprits — `fabricpc/serialization.py` (Task 4: Model Checkpointing)

Rationale for the non-obvious decisions, so a reviewer can judge intent. This is the
**merged** design: the original implementation was compared against a reference design
(`HANDOFF_compare.md`); the comparison (`COMPARISON_checkpointing.md`) adopted the better
choice at each branch point. Scope: `save_checkpoint` / `load_checkpoint`
serializing params, optimizer state, graph structure, and (optionally) inference state.

## What a checkpoint is
A **directory**: `metadata.json` (format magic + version, library versions, per-file
sha256 checksums), `params.msgpack` (always), and optionally `structure.json`,
`opt_state.msgpack`, `state.msgpack`.

## Public API
```python
save_checkpoint(path, params, *, opt_state=None, structure=None, state=None,
                metadata=None, overwrite=False) -> Path
load_checkpoint(path, *, optimizer=None, strict=True)
    -> Checkpoint(params, opt_state, structure, state, metadata)  # NamedTuple, named access
```
`path`-first + everything-else-optional (adopted from the reference): a params-only
save is `save_checkpoint(path, params)`. The typed `Checkpoint` NamedTuple (kept from the
original) is typo-proof vs a string-keyed dict.

## The four payloads, and why each is handled the way it is

1. **`params` — always saved, and SELF-DESCRIBING (the key correctness fix).** flax
   msgpack stores the pytree keyed by node/weight/bias names, so on load
   `flax.serialization.msgpack_restore` rebuilds the nested dict (dtype preserved, incl.
   bf16/fp16) and we wrap it straight back into `GraphParams`/`NodeParams`
   (`_restore_params`). **No `structure`, and no model-definition classes, are needed to
   recover weights.** The original design decoded params into an
   `initialize_params(structure)` *target*, which meant a single renamed/moved node class
   made even the params unloadable — the comparison's highest-severity finding (R1). The
   reference's self-describing approach is strictly more robust here, so it was adopted.

2. **`opt_state` — optional; restored into `optimizer.init(params)`.** Its pytree *type*
   is defined by the optimizer, not the data on disk, so the only way to recover exact
   optax NamedTuple types is a template from the optimizer. Hence `load_checkpoint` takes
   the `optimizer`. flax `from_bytes` matches by **field name**, so a wrong optimizer
   raises rather than silently misloading (kept over the reference's length-only check).
   `strict` governs the has-opt_state-but-no-optimizer case (raise vs skip).

3. **`structure` — optional, reconstructed BEST-EFFORT.** Still serialized as a versioned
   JSON *snapshot* of `NodeInfo`, with each object restored via `cls.__new__(cls)` +
   direct attribute/`.config` assignment. `__new__` (not `cls(**config)`) is required
   because several real constructors aren't round-trippable through `__init__`
   (`UniformInitializer` renames config keys, `ZerosInitializer` drops `gain`,
   `GlobalStateInit` nests a component). A snapshot (persisting `node_order` + muPC
   factors) rather than re-running the `graph` builder avoids the builder's best-effort
   topological sort on cyclic PC graphs. **New:** *any* failure to rebuild the structure
   (a class that can no longer be imported, an evolved snapshot schema → `KeyError`, a
   malformed path / unknown tag → `ValueError`, or a corrupt structure file) is caught,
   **warns, and returns `structure=None`** — it never blocks the params (this is what
   makes #1's robustness real end-to-end). When the structure *does* rebuild, the loaded
   params are validated against it (node-set + shapes); a genuine mismatch there still
   raises (it is an integrity error, not a graceful-degradation case).

4. **`state: GraphState` — optional (adopted from the reference).** Only for
   persistent-latent / Hopfield-style runs; the standard PC trainer re-initialises `z`
   each batch, so it is not needed to resume normal training. Restored into an
   `initialize_graph_state` template, so it loads only when `structure` was rebuilt.

## Robustness decisions
- **Format magic + version.** `metadata.json` carries `format == "fabricpc-checkpoint"`
  and `format_version`; load checks both *before* decoding (unrelated dir → clear "not a
  FabricPC checkpoint"; future format → clear version error). `version > FORMAT_VERSION`
  rejects only *newer* files (forward-incompat, backward-compat) — kept over the
  reference's `!=`.
- **Per-file sha256 checksums** recorded at save, verified on load → a corrupt payload
  that still decodes is caught with a clear "is corrupt" message.
- **Error framing** (adopted from the reference): bad JSON, a non-checkpoint dir, a
  non-JSON `metadata`, and unreadable payloads all raise checkpoint-aware `ValueError`s.
- **Crash-safe writes.** Staged in a sibling `.tmp-<pid>` dir, moved with atomic
  `os.replace`. On `overwrite` the existing checkpoint is renamed aside to `.bak` and
  restored if the swap fails. The `finally` only discards the backup once `path` holds a
  valid checkpoint, so a failed *restore* can't delete the last copy (fixes comparison R3).
- **Array backend isolated** behind `_save_arrays`/`_load_arrays`/`_restore_params` (flax
  msgpack now; swappable for Orbax later). flax round-trips every dtype losslessly
  (verified incl. bf16/fp16), so no bespoke byte codec is needed.

## What is intentionally NOT done
- **No pickle** (version-brittle for arrays; breaks on class rename). The JSON snapshot is
  inspectable and degrades gracefully instead.
- **No Orbax dependency** (its async/sharding/CompositeHandler machinery buys little for a
  single-host research framework).
- **No content fingerprint of the *model*** beyond per-file checksums — in this bundled
  design structure+params come from the same file, so they are always mutually consistent.

## Constraints honored
- **Additive:** only new file `fabricpc/serialization.py`, three exports in
  `fabricpc/__init__.py`, and the opt-in `--checkpoint` branch in `examples/mnist_demo.py`.
- **No new demo:** an existing demo is extended, per the plan.
- **Tested:** `tests/test_serialization.py` (27 tests) covers self-describing param
  round-trip (incl. bf16/fp16 and no-structure), the class-rename → params-only
  degradation (R1), opt_state resume parity + wrong-optimizer, structure fidelity (the
  `UniformInitializer` violator, nested-component state-init, conv config, muPC),
  GraphState round-trip, and the full error contract (checksum corruption, format magic,
  version, non-JSON metadata, overwrite crash-safety, missing files).
