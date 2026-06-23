# Model-Checkpointing Comparison — `fabricpc/serialization.py` (YOURS) vs `HANDOFF_compare.md` reference

Comparison along two lenses (design/robustness and API/correctness/tests), grounded
against three empirical probes (flax dtype round-trip; load-path structure dependency;
class-rename failure). "YOURS" = the Task-4 implementation on branch
`worktree-task4-checkpointing`; "REFERENCE" = the design + full source in
`HANDOFF_compare.md`. File:line citations are to `fabricpc/serialization.py` unless
noted. **No code was modified** (per the handoff).

---

## 1. Summary verdict

**Materially different in design philosophy; convergent in mechanics for the common
case.** Both make the same core decisions: params always saved as the model; opt_state
optional and restored into an `optimizer.init(params)` template; JSON metadata; a
format-version gate; atomic writes; lossless dtypes. They split on the single hardest
question — **GraphStructure**:

- **YOURS serializes the structure** (versioned JSON snapshot, objects rebuilt via
  `cls.__new__` + `.config`/attr assignment) and therefore needs **no caller-supplied
  model code** to return a fully usable model. Cost: loading **anything** (params
  included) reconstructs the structure and imports every node/component class by its
  saved path — so a single class rename/move bricks the **whole** checkpoint.
- **REFERENCE never serializes structure.** Params are self-describing from the
  manifest and load with zero model imports; structure is an *optional* load input used
  only for validation and GraphState. Cost: you must rebuild the structure in code to
  get a usable model back.

Neither is a superset. YOURS optimizes for *self-contained reload*; REFERENCE optimizes
for *survival across code evolution*. The right answer is a merge (§5) whose linchpin is:
**decouple params decoding from structure reconstruction** (adopt REFERENCE here), while
**keeping YOURS' structure snapshot as an optional, best-effort bonus.**

---

## 2. Dimension table (13 dimensions from HANDOFF §4)

| # | Dimension | YOURS | REFERENCE | Diverge? | Preferable + why |
|---|-----------|-------|-----------|----------|------------------|
| 1 | What is persisted | params (always), opt_state (opt.), **structure (always)**, metadata under `user`, + provenance: format_version, library_versions, per-leaf param shapes+dtypes, node/edge counts (`save_checkpoint` L497–509; `_param_leaf_metadata` L389). No GraphState, step, rng. | params (always), opt_state (opt.), **GraphState (opt.)**, metadata, fabricpc_version, created_at, batch_size. No structure, step, rng. | **Yes** | Merge — each saves what the other drops (structure vs GraphState). |
| 2 | GraphState | **Not saved** (intentional; std trainer re-inits `z` per batch). | Optional `state=`, with the per-batch re-init rationale documented; restored into `initialize_graph_state` template. | **Yes** | **Reference** — covers persistent-latent/Hopfield + state inspection at ~zero cost when unused. YOURS has a real (if niche) capability gap. |
| 3 | GraphStructure | **Serialized** as versioned JSON snapshot; `cls.__new__` + restore private attrs/`.config`; embeds class import paths (L131, L211–224, L299–332); persists computed `node_order` + muPC factors. **No pickle.** | **Never serialized**; user rebuilds in code; optional `structure=` validates. | **Yes (central)** | Merge — YOURS wins self-containedness & deterministic `node_order`; REFERENCE wins refactor-robustness & simplicity. See §3.1. |
| 4 | Load-time inputs | **Loading params REQUIRES reconstructing structure** (`deserialize_structure` → `initialize_params(structure)` builds the decode target, L600→606→608), i.e. all node/component classes must be importable. `optimizer=` only for opt_state; `strict` gates the no-optimizer case. Returns `Checkpoint(params, opt_state, structure)`. | params self-describing — load from file alone, **no model imports**. `structure=`/`optimizer=` optional (validation / GraphState / opt_state). | **Yes** | **Reference** — self-describing params decouple weight recovery from code identity (the decisive point, §3.1/R1). |
| 5 | Format & deps | **Directory** of 4 files (2 JSON + 2 flax-msgpack); dep on flax (already transitive); no pickle (L5–11, L92–95, L374–379). | **Single `.npz`** (numpy only), JSON manifest as unicode array, explicit `allow_pickle=False`. | **Yes** | Slight edge **Reference** — one portable artifact + explicit pickle ban. YOURS' JSON is human-inspectable and the array backend is isolated (L372–379) for an easy Orbax swap. Toss-up. |
| 6 | Dtype fidelity | flax msgpack. **Verified lossless** for bf16/fp16/int32/0-d scalar (disk dtype wins over target). | Raw uint8 byte codec + `(dtype,shape)`; flattens to 1-D for 0-d; lossless by construction. | **No (equivalent)** | Equivalent outcome. Note: REFERENCE's stated motivation ("numpy can't store bf16") is **partly outdated** — `ml_dtypes` registers bf16 with numpy on this stack — but its codec remains a fair portability hedge. flax is fewer lines. |
| 7 | Atomicity / crash safety | Staging dir + `os.replace`; on overwrite, old ckpt renamed to `.bak` first, restored if swap fails (L511–543; tested). **No lost-checkpoint window.** | tmp file + single `os.replace` (atomic; single-file so replace *is* the swap). | **Yes (degree)** | **YOURS** for overwrite — genuinely no instant where both copies are absent. (Directory format *needs* this dance; single-file gets atomicity for free.) One minor cleanup caveat: R3. |
| 8 | Wrong-model safety | Per-leaf **keypath+shape** validation vs reconstructed structure **on every load, before decode** (L401–422, called L607); format-version gate; flax-drift warning. | name+shape **signature** validation **only when `structure=` passed**; treedef-length+shape check for opt_state/state. | **Yes** | **YOURS** — validation is automatic & unconditional (it always has the structure); REFERENCE leaves params unchecked unless you opt in. Neither catches a wrong-but-shape-identical model (R4). |
| 9 | opt_state / resume | flax `from_bytes` into `optimizer.init(params)`; recovers exact NamedTuple types; wrong optimizer → flax raises on **field-name** mismatch (verified); `strict` gates absent-optimizer (L610–624); bitwise resume parity tested. | `tree_unflatten` into `optimizer.init(params)`; checks treedef **length + leaf shape only, NOT field names**. | **Yes** | **YOURS** — field-name matching catches a same-arity/same-shape but different optimizer that REFERENCE would accept silently (R2). Both share the curated-message vs flax-phrased trade-off; REFERENCE's message is more bespoke. |
| 10 | API surface | `save_checkpoint(params, opt_state, structure, path, *, overwrite=False, metadata=None) -> None`; `load_checkpoint(path, *, optimizer=None, strict=True) -> Checkpoint` NamedTuple (L461, L546, L114). Module `fabricpc.serialization`; exported `__init__.py:51,62–65`. **`opt_state`/`structure` are required positionals; `overwrite` defaults False.** | `save(path, params, *, opt_state=None, state=None, metadata=None, overwrite=True) -> Path`; `load(path, *, structure=None, optimizer=None) -> dict`. Module `fabricpc/training/serialization.py`. | **Yes** | Mixed → merge. **YOURS**: typed NamedTuple return (tuple-unpack + named, typo-proof) and safe `overwrite=False`. **REFERENCE**: `path`-first + all-optional kwargs → trivial params-only save (YOURS forces `save_checkpoint(params, None, structure, path)`). Adopt REFERENCE's signature shape + YOURS' return type & default. |
| 11 | Versioning | `FORMAT_VERSION=1`, checked first; rejects `version > FORMAT_VERSION` or missing (L587–593); backward-compatible. | `format`+`version`; rejects `version != FORMAT_VERSION` (also rejects readable older/newer-minor). | **Yes** | **YOURS** — `>` is correct forward-incompat/backward-compat semantics. (REFERENCE's explicit `format` magic string is a nice touch worth borrowing — dim 12.) |
| 12 | Error handling | Missing dir/file → `FileNotFoundError` w/ name (L576–582); version & shape mismatch → clear `ValueError`. **But:** corrupt `params.msgpack`→raw `ExtraData`; corrupt `structure.json`→raw `JSONDecodeError`; class rename→raw `AttributeError`/`ModuleNotFoundError`; non-JSON metadata→bare `TypeError` (L509) — all **unwrapped** (verified). | All read failures wrapped by `_open` into "Could not read … as a FabricPC checkpoint"; missing manifest, bad format/version, non-JSON metadata → framed `ValueError`s naming allowed types. | **Yes** | **Reference** — uniformly checkpoint-aware messages. YOURS is loud (never silently misloads) but leaks raw library exceptions on corruption/rename. Concrete, cheap gap to close. |
| 13 | Tests | **21 passing** (verified): param/opt-state exact value+type, tuple-unpack, full structure fidelity incl. `UniformInitializer` ctor-quirk, nested-component state-init, conv tuple config, muPC scaling, re-init-identically, **resume parity**, strict/non-strict, params-only, overwrite guard+force, user metadata, missing files, future version, shape mismatch, staging cleanup, **overwrite-crash rollback**. | **16** (claimed): params no-structure, lossless bf16/fp16/int, metadata, structure validation + wrong-structure, opt_state resume + skip, **state round-trip**, atomic, overwrite, **no-pickle load**, future-version, **corrupt-file**. | **Yes** | Each tests its own design. **YOURS gaps:** no class-rename test (the risk its design *introduces*), no explicit bf16/fp16 dtype test, no corrupt-file/non-JSON-metadata test. **REFERENCE gaps:** no structure-snapshot round-trip (can't — no structure on disk), no overwrite-crash rollback. |

### Load-capability matrices

**YOURS** — always returns `Checkpoint(params, opt_state, structure)`:

| call | params | structure | opt_state |
|---|---|---|---|
| `load_checkpoint(path)`, no opt_state in file | ✅ | ✅ (from disk) | None |
| `load_checkpoint(path)`, file *has* opt_state, `strict=True` | — | — | **raises** (L614) |
| `load_checkpoint(path, strict=False)` | ✅ | ✅ | None (skipped) |
| `load_checkpoint(path, optimizer=opt)` | ✅ | ✅ | ✅ |

**REFERENCE** — returns `{params, opt_state, state, metadata}`:

| call | params | opt_state | state |
|---|---|---|---|
| `load(path)` | ✅ | None | None |
| `load(path, optimizer=opt)` | ✅ | ✅ | None |
| `load(path, structure=s)` | ✅ (validated) | None | ✅ |
| `load(path, structure=s, optimizer=opt)` | ✅ (validated) | ✅ | ✅ |

Asymmetry: YOURS *returns* structure with zero inputs and validates params
unconditionally, but **cannot decode params if any class is unimportable**; REFERENCE
never returns structure but decodes params from the file alone and validates only on
opt-in.

---

## 3. Key divergences (the branching points) — pros, cons, and the grounded call

### 3.1 Serialize structure (YOURS) vs rebuild-in-code (REFERENCE) — and the load-coupling that decides it

- **YOURS pros:** self-contained reload (hand someone a directory, they get a working
  model with no model-definition code); deterministic persisted `node_order` (avoids the
  builder's best-effort topological sort on cyclic PC graphs); faithfully round-trips
  constructor-quirk components (`UniformInitializer` key-rename, `GlobalStateInit`
  nesting) that `cls(**config)` would break; human-inspectable JSON; not pickle.
- **YOURS cons:** ~470 lines of bespoke codec coupled to **private** attribute names
  (`_decode_node` writes `_name`, `_node_info`, … L323–332) and to **class import
  paths** (L131). **Decisively:** `load_checkpoint` reconstructs the structure to build
  the params decode target (L600→606), so **params can't load unless every node and
  component class is still importable at its saved path** — verified: renaming `Linear`
  in `structure.json` raised a bare `AttributeError` and lost the weights too.
- **REFERENCE pros:** tiny, robust to refactors (no class identity persisted); params
  self-describing and decoded directly from manifest names; orthodox flax/orbax
  philosophy ("structure lives in code").
- **REFERENCE cons:** the checkpoint alone is not a complete model — you must re-run your
  model-definition code to get a usable structure back; can't snapshot structure for a
  fresh-process/deploy/inspect scenario.

**Better, grounded:** **Split the decision.** Adopt the REFERENCE's stance *for params*
(self-describing, target-free decode — params must never depend on class identity), and
keep YOURS' structure snapshot *as an optional, best-effort* artifact. YOURS' own design
doc rejects pickle because "pickle breaks on any class rename — fatal for a framework
that actively migrates node types"; the JSON snapshot **shares that exact
rename-fragility for class identity** and, worse, currently propagates it to params. The
fix preserves YOURS' headline capability while removing the high-severity risk.

### 3.2 Self-containedness vs robustness to evolution

This is the philosophical axis underneath 3.1. YOURS buys "reload with no code";
REFERENCE buys "survive node-type migration" (which this repo's git history shows is
active). **Better:** keep YOURS' self-containedness as a *convenience* layered on a
REFERENCE-style robust core — i.e. params + manifest are the durable artifact; the
structure snapshot is a bonus that degrades gracefully ("params loaded; structure
unavailable: class X moved") instead of aborting.

### 3.3 Directory + flax msgpack (YOURS) vs single `.npz` (REFERENCE)

- **YOURS pros:** human-readable JSON; array backend isolated for a future Orbax swap.
- **YOURS cons:** four files to move together; needs the `.bak` dance for atomic
  overwrite; couples the array format to flax's msgpack (warns, doesn't fail, on drift).
- **REFERENCE pros:** one portable file; `allow_pickle=False`; atomic overwrite for free;
  no flax dep; self-owned dtype codec.
- **REFERENCE cons:** binary blob (less inspectable); the raw-uint8 codec is extra code
  whose original motivation is partly obsolete here.

**Better, grounded:** roughly a wash for a research framework; **lean REFERENCE on
portability/packaging**. If YOURS keeps the directory, borrow the explicit `format` magic
string and the wrapped corrupt-file errors.

### 3.4 Dtype: flax msgpack vs raw codec — a non-divergence

Empirically both round-trip bf16/fp16/int/0-d losslessly. **Better:** either; flax is
less code and verified. Add an explicit dtype-fidelity test (YOURS lacks one).

### 3.5 opt_state matching & wrong-model safety

YOURS validates params on every load and matches opt_state by **field name** (catches a
wrong optimizer — verified); REFERENCE validates params only on opt-in and matches
opt_state by **length+shape only** (a same-shape different optimizer loads silently).
**Better:** **YOURS** on both. Neither catches a wrong model with coincidentally identical
shapes — close that in both with a content fingerprint (R4).

### 3.6 Atomicity on overwrite

YOURS' rename-aside `.bak` + restore (L511–543) is strictly more defensive than a single
`os.replace`. **Better:** **YOURS** (with the R3 cleanup fix).

---

## 4. Correctness risks

- **R1 (YOURS, HIGH) — a class rename/move bricks the entire checkpoint, with a cryptic
  error.** Params decode requires structure reconstruction (L600→606→`graph_net`
  initialize_params→`_import_class` L134–142). Verified: renaming a node class in
  `structure.json` → bare `AttributeError`, weights unrecoverable. REFERENCE has no
  equivalent risk for params. *This is the most important finding.*
- **R2 (REFERENCE, MEDIUM) — silent wrong-optimizer / unvalidated params.** `_unpack_tree`
  checks treedef length + shape only (not field names); params load unvalidated unless
  `structure=` is passed. YOURS is safer on both.
- **R3 (YOURS, LOW) — overwrite `finally` can delete the backup after a failed restore.**
  L539–543 `shutil.rmtree(backup)` runs unconditionally; if the `except`-path restore
  `os.replace(backup, path)` (L537) were itself interrupted, the only surviving copy
  could be removed. Narrow window; add a guard.
- **R4 (BOTH, LOW) — no content fingerprint.** A model with coincidentally identical leaf
  shapes loads with mismatched weights without error (YOURS L401–422; REFERENCE
  `_validate`). A params/structure hash in metadata closes it.
- **R5 (YOURS, INFO) — array format coupled to flax.** msgpack format is flax-owned;
  YOURS only warns on major.minor drift (L425–441). Acceptable; REFERENCE avoids it by
  owning its byte codec.

---

## 5. Recommendation — single merged design

Keep YOURS as the base (richer validation, stronger atomicity, typed result, correct
version semantics, field-name opt_state matching) and graft the REFERENCE's decoupling,
GraphState, and error framing:

1. **Params: self-describing, target-free decode (ADOPT REFERENCE).** Record
   node/weight/bias names in the manifest and decode params **without** building an
   `initialize_params` target — so `load_checkpoint(path)` recovers weights with **no
   model-class imports**. *Removes R1.* This is the single most important change.
2. **Structure: keep YOURS' JSON snapshot but make it optional & best-effort (MERGE).**
   Store `structure.json`; on load, wrap `_import_class` failures and degrade to
   "params loaded; structure unavailable: <class> renamed/moved" instead of aborting.
   *Fixes R1's blast radius + the dim-12 error quality.*
3. **GraphState: add optional `state=` (ADOPT REFERENCE)** + `batch_size` in metadata,
   restored into `initialize_graph_state` only when `structure` is available.
4. **Atomicity: keep YOURS' staging + `.bak` overwrite-rollback (KEEP)**, but guard the
   `finally` so a failed restore can't delete the backup (*fix R3*).
5. **opt_state: keep YOURS' flax field-name matching + strict/non-strict (KEEP)** over
   length-only (*addresses R2*); keep always-validating params.
6. **Versioning: keep YOURS' `> FORMAT_VERSION` (KEEP)**; borrow REFERENCE's explicit
   `format` magic string for "is this even ours" rejection.
7. **Error framing: adopt REFERENCE's wrapping (ADOPT)** — wrap array/JSON reads and the
   metadata `json.dumps` into checkpoint-aware `ValueError`s.
8. **API: adopt REFERENCE's optional-kwarg save signature (ADOPT)**
   `save(path, params, *, opt_state=None, structure=None, state=None, metadata=None,
   overwrite=False)`; keep YOURS' `Checkpoint` NamedTuple return & `overwrite=False`.
9. **Dtype: keep flax msgpack (KEEP)**; add an explicit bf16/fp16/0-d round-trip test.
10. **Add a params/structure fingerprint to metadata (NEW)** — closes R4 in both.

Net result: one ergonomic `save`; self-describing params that load regardless of code
drift; an optional, graceful structure snapshot; optional GraphState; unconditional
validation; best-in-class crash safety; and checkpoint-aware errors everywhere.
