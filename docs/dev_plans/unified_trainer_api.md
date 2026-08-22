# Unified Trainer API — energy-framed PC/backprop/autoregressive (fabricpc 0.5.0)

## Context

The repo carries four training harnesses with duplicated code and divergent contracts:
`fabricpc/training/train.py` (PC), `train_backprop.py`, `train_autoregressive.py` (PC-AR + backprop-AR),
and the in-progress `unified_trainer.py` (branch `feature/unified-trainer`, commit 772795b, purely additive).
Duplication is extensive: batch conversion inlined 8×, clamp assembly 9×, the energy sum open-coded 8×,
four structurally identical epoch loops, plus a fifth PC step in `utils/dashboarding/inference_tracking.py`
(whose line-159 TODO names this exact consolidation). Signatures diverge in argument order, step-return
arity (3/4/5-tuples), callback kwargs, config keys, and eval result keys.

This plan replaces all four with one API. Two semantic changes relative to the in-progress
`unified_trainer.py`:

1. **Backprop is framed in energy, not loss.** Both algorithms clamp identically during training
   (input and target). The backprop objective is the energy of the clamped nodes — the negative log
   probability the output node's energy functional assigns to the clamped target given the feedforward
   prediction — differentiated globally with `jax.value_and_grad`. PC differs only in that inference
   settles the latents first, the objective sums every internal node's energy, and gradients are local.
2. **The clamping mechanism is shared.** One `build_clamps` builds the same clamp dict for both
   algorithms; it also absorbs the integer-target one-hot and causal-mask injection that today live
   only in the legacy AR trainer.

## Confirmed decisions

| Decision | Choice |
|---|---|
| Legacy API | Clean break at 0.5.0. Delete all 17 legacy `fabricpc.training` exports + 3 aliases + top-level `train_pcn`/`evaluate_pcn`. Migrate every internal caller in this change. No shims. CHANGELOG migration table for PyPI users. |
| AR objective scale | Per-sample energy in all modes (sum over sequence positions ÷ batch size). AR-backprop effective learning rate shifts ×seq_len vs the legacy per-token mean — CHANGELOG flag. |
| Eval pmap | **Kept.** `evaluate` mirrors `train`'s device auto-detect, including the ragged-batch zero-padding path from `train.py:573-659`. |
| Callback contract | `epoch_callback(epoch_idx, params, structure, config, rng_key, metrics)` and `iter_callback(epoch_idx, batch_idx, metrics)`, one contract for all modes; `metrics` is a dict (see below). |
| Tuner pruning signal | Train perplexity `exp(metrics["target_energy"])` — free, since the clamped target node's energy under `CrossEntropyEnergy` is the teacher-forced cross-entropy. Energy is not a performance metric. |

## Design

### Symbol table

- `structure` — the `GraphStructure` (static topology: nodes, edges, `task_map`, `node_order`, config).
- `params` — the `GraphParams` pytree of per-node weights/biases; the only learnable state.
- `state` — a `GraphState`: per node, `z_latent` (the latent value), `z_mu` (the prediction from
  upstream latents), `error`, `energy` (per-sample, shape `(batch,)`).
- `clamps` — a dict `{node_name: array}`; a node is clamped iff its name is a key. Consumed by the
  state initializer (sets `z_latent` to the clamp) and the inference loop (skips latent updates).
- **target nodes** — the clamped nodes with `in_degree > 0`, i.e. the nodes `task_map` maps
  non-`x` batch keys to (the output node in every current graph).
- `E(z, μ)` — the node's energy functional applied to its latent `z` and prediction `μ`;
  `CrossEntropyEnergy` gives `-Σ z·log μ` (negative log probability), `GaussianEnergy` gives
  `0.5·precision·Σ(z-μ)²`, summed over all non-batch dimensions.

### The unified step (both axes: `algorithm` × `autoregressive`)

| sub-step | PC | backprop |
|---|---|---|
| 1 batch→dict | `convert_batch` | (shared) |
| 2 build clamps | `build_clamps(clamp_target=True)` — x AND targets | **(shared, identical)** |
| 3 produce state | `initialize_graph_state` + `run_inference` (settle) | `initialize_graph_state` with `FeedforwardStateInit` (single forward pass, no inference) |
| 4 objective | `graph_energy` over all `in_degree>0` nodes ÷ batch | `graph_energy` over **target nodes** ÷ batch (shared function, different node subset) |
| 5 gradients | `compute_local_weight_gradients` (local, per node) | `jax.value_and_grad(objective, has_aux=True)` over steps 3–4 (global) |
| 6 apply update | optax `update` + `apply_updates` | (shared) |

Verified mechanics (read in source, this branch):

- `FeedforwardStateInit.initialize_state` (`graph_initialization/state_initializer.py:294-301`)
  keeps a clamped node's `z_latent` = clamp and stores the freshly computed `z_mu`/`error`/`energy`
  from `node_class.forward`. After a feedforward pass with target `y` clamped, the target node's
  stored energy is exactly `E(y, μ(params))` — cross-entropy under `CrossEntropyEnergy` (modulo
  `clip(μ,1e-7,1)` vs legacy `μ+1e-10`), `0.5·precision·SSE` under `GaussianEnergy`. So `loss_type`
  is retired: **the output node's energy functional in the graph definition selects the loss.**
- Unclamped nodes keep `energy = 0` under feedforward init, and free/source nodes get `E(z,z)` only
  after PC inference — hence the explicit `in_degree>0` / target-node filters.
- No gradient blockers on the differentiated path (no `stop_gradient` in the package; one-hot runs
  outside the closure; clamp-dtype validation is trace-time Python). Differentiating through
  `FeedforwardStateInit` is already proven by the in-progress branch's 1e-12 backprop parity tests.
- Autoregressive changes nothing in steps 3–6: it only affects clamp construction (shifted int32
  token targets need one-hot; v1 transformer graphs need the causal-mask clamp).

### Metrics dict (step output, callbacks, results)

- `metrics["energy"]` — the per-sample objective from step 4 (what the gradients descend).
- `metrics["target_energy"]` — summed target-node energy ÷ number of predictions
  (`prod(target.shape[:-1])`: batch·seq for sequences, batch for classification). Under
  `CrossEntropyEnergy` this is the teacher-forced per-token cross-entropy; `exp` of it is train
  perplexity. For backprop it is the objective under a different normalization; for PC it isolates
  the output term from the hidden-layer residuals.

Both are computed inside the jitted step from the already-materialized state — no extra forward work.
`iter_results[epoch][batch]` and `epoch_results[epoch]` store these dicts (floats) unless the
corresponding callback returns a replacement value.

## Public API

```python
# fabricpc/training/trainer.py
ALGORITHMS = ("pc", "backprop")

def convert_batch(batch_data) -> Dict[str, jnp.ndarray]
def create_causal_mask(seq_len: int) -> jnp.ndarray

def build_clamps(batch, structure, *, clamp_target: bool,
                 autoregressive: bool = False) -> Dict[str, jnp.ndarray]

def make_train_step(structure, optimizer, *, algorithm="pc", autoregressive=False)
    # -> step(params, opt_state, batch, rng_key)
    #    -> (params, opt_state, metrics: dict, final_state: GraphState)
    # jitted; validation (algorithm prerequisites) runs once at build time.
    # Public replacement for the 9 legacy train_step call sites; callers hold the
    # returned step, so the jit cache persists across calls.

def train(params, structure, train_loader, optimizer, config, rng_key, *,
          algorithm="pc", autoregressive=False, verbose=True, use_tqdm=True,
          epoch_callback=None, iter_callback=None, pmap_single_device=False,
) -> tuple[GraphParams, list, list]   # (params, iter_results, epoch_results)

def evaluate(params, structure, test_loader, config, rng_key, *,
             algorithm="pc", autoregressive=False, pmap_single_device=False,
) -> Dict[str, float]

# fabricpc/training/generation.py
def generate(params, structure, prompt, max_new_tokens, rng_key,
             temperature=1.0, top_k=None, top_p=None) -> jnp.ndarray

# fabricpc/training/device_utils.py
replicate_params, replicate_opt_state, shard_batch, unshard_metrics  # moved from train.py

# fabricpc/core/energy.py
def graph_energy(state, structure, *, node_names=None) -> jnp.ndarray
    # Total energy summed over the selected nodes and the batch.
    # node_names=None selects all in_degree>0 nodes. Callers normalize explicitly
    # (÷ state.batch_size for per-sample, ÷ prediction count for per-token).
    # Replaces the 8 open-coded sums.
```

Defaults realize the requirement: `train(...)` with no mode arguments is PC, non-autoregressive.
Everything after `rng_key` is keyword-only, so `functools.partial(train, algorithm="backprop")`
satisfies the unchanged `ExperimentArm` positional contract.

`config` reads only `num_epochs` (fractional supported). `train`/`evaluate` raise `ValueError`
if `config` contains a retired key (`"loss_type"`, `"use_causal_mask"`) with one-line migration
text — fail-fast, not a fallback.

### `build_clamps` (absorbs legacy `build_train_clamps`)

1. For each batch key present in `structure.task_map`: clamp `task_map[key]` if `clamp_target`,
   else clamp only non-target keys (evaluation: inputs clamped, targets free).
2. Integer-target one-hot (training, both algorithms): if a clamped target array has integer dtype,
   `jax.nn.one_hot(y, num_classes)` with `num_classes` from the target node's `NodeInfo.shape[-1]`.
   Fixes the in-progress branch's gap where stock int32 token loaders
   (`utils/data/dataloader.py:263-268`) hit `TypeError` in `_validate_clamp_dtypes`.
3. Causal mask, graph-derived: if `autoregressive` and `"causal_mask" in structure.task_map`,
   inject `broadcast_to(tril[None,None], (batch, 1, seq, seq))` at `task_map["causal_mask"]`;
   otherwise no-op. The `use_causal_mask` flag disappears from signatures and config. v1 graphs
   declare the mask node in their `TaskMap`; v2 graphs mask internally
   (`MhaResidualNode(is_causal=True)`) and have no such task key, so the current
   `_require_causal_mask_node` raise on v2 graphs is deleted.

### Step pseudocode

```python
def _batch_grads(params, batch, structure, rng_key, *, algorithm, autoregressive):
    batch_size = next(iter(batch.values())).shape[0]
    clamps = build_clamps(batch, structure, clamp_target=True,
                          autoregressive=autoregressive)          # (2) identical for both
    target_nodes = tuple(n for n in clamps
                         if structure.nodes[n].node_info.in_degree > 0)  # static
    # backprop with no clamped target has an empty objective -> raise at trace time
    if algorithm == "pc":
        state = initialize_graph_state(structure, batch_size, rng_key,
                                       clamps=clamps, params=params)     # (3a)
        state = run_inference(params, state, clamps, structure)          # (3a) settle
        energy = graph_energy(state, structure) / batch_size             # (4) all internal
        grads = compute_local_weight_gradients(params, state, structure) # (5a) local
    else:  # backprop
        def objective(p):
            state = initialize_graph_state(structure, batch_size, rng_key,
                                           clamps=clamps, params=p)      # (3b) feedforward only
            return graph_energy(state, structure,
                                node_names=target_nodes) / batch_size, state  # (4) targets only
        (energy, state), grads = jax.value_and_grad(objective, has_aux=True)(params)  # (5b) global
    n_predictions = <static prod of target shape[:-1]> * ...             # batch*seq or batch
    metrics = {"energy": energy,
               "target_energy": graph_energy(state, structure,
                                             node_names=target_nodes) / n_predictions}
    return grads, metrics, state

def _make_step(structure, optimizer, *, algorithm, autoregressive, pmean):
    def step(params, opt_state, batch, rng_key):
        grads, metrics, state = _batch_grads(...)
        if pmean: grads = jax.lax.pmean(grads, axis_name="devices")
        params, opt_state = _apply_update(params, opt_state, grads, optimizer)  # (6) optax
        return params, opt_state, metrics, state
    return jax.pmap(step, axis_name="devices") if pmean else jax.jit(step)
```

Training loop (`_run_training_loop`, private): the in-progress `run_training_loop` body — device
auto-detect (jit for 1 device, pmap for N, `pmap_single_device` override), tqdm, fractional epochs,
ragged-shard skip-with-warning, per-epoch key splitting (preserves the legacy `train_pcn` RNG
stream) — minus the metric_fn/metric_name machinery. Backprop thereby gains tqdm and multi-device
data parallelism.

### `evaluate`

Shared eval clamping `build_clamps(clamp_target=False)` in both modes; PC settles via
`run_inference`, backprop takes the feedforward pass. Per-batch jitted step returns
`(ce_sum, correct, count, energy_sum, weights)` accumulated host-side; the pmap path shards batches
with the zero-padding ragged-batch handling ported from `train.py:573-659` and reduces with `psum`.
Cross-entropy is computed from the free output's `z_mu` with the `CrossEntropyEnergy` eps
(`clip(μ,1e-7,1)`), one-hotting integer targets on the fly.

Normalized result keys:

| key | when | definition |
|---|---|---|
| `accuracy` | always | total correct ÷ total predictions |
| `cross_entropy` | always | total CE ÷ total predictions (per-token; equals per-sample for classification) |
| `energy` | `algorithm="pc"` | per-sample `graph_energy` (internal nodes; free output contributes zero) |
| `perplexity` | `autoregressive=True` | `exp(cross_entropy)` |

`evaluate_transformer` folds in as `evaluate(..., algorithm="pc", autoregressive=True)` — the exact
key set `tests/test_transformer_nodes.py:443-466` already asserts. Two deliberate numerical fixes to
flag: the legacy transformer eval applied `softmax` to `z_latent`, which for a free output already
holds post-softmax probabilities (a double softmax), and added an external squared-error term to
energy; the unified evaluate reads `z_mu` directly and reports pure internal energy. `loss` is
renamed `cross_entropy`; `num_batches` and the `debug=` kwarg are dropped.

### Callbacks and downstream consumers

- One contract: `iter_callback(epoch_idx, batch_idx, metrics)`,
  `epoch_callback(epoch_idx, params, structure, config, rng_key, metrics)` (epoch-mean metrics).
  The legacy AR-only `energy=`/`ce_loss=` kwargs die with the in-jit unused cross-entropy.
- `ExperimentArm` (`experiments/ab_experiment.py`) is untouched: arms pass `train` /
  `functools.partial(train, algorithm="backprop")` as `train_fn`, same for `evaluate` as `eval_fn`.
- `BayesianTuner` (`tuning/bayesian_tuner.py`): calls `train(..., autoregressive=True)`; drops the
  `"use_causal_mask"` config injection; `epoch_callback` reports
  `exp(metrics["target_energy"])` (train perplexity) to Optuna and keeps the divergence guard on
  `metrics["energy"]`; final score from `evaluate(...)["cross_entropy"]`/`["perplexity"]`.
- `utils/dashboarding/callbacks.py`: `create_iter_callback`/`create_detailed_iter_callback` take the
  metrics dict (read `metrics["energy"]`); `eval_fn` 5-positional contract already matches `evaluate`.
- `utils/dashboarding/inference_tracking.py:train_step_with_history`: same signature, body rebuilt on
  `build_clamps` → `initialize_graph_state` → `run_inference_with_history` → `graph_energy` →
  `compute_local_weight_gradients` → optax. Its energy becomes internal-only and per-sample
  (was all-node, unnormalized) — CHANGELOG flag. Deletes the fifth duplicate step and the TODO.

## Alternatives considered

- **Trainer class vs module functions.** A `Trainer(structure, optimizer, algorithm=...)` object
  would cache jitted steps across calls. Rejected: the codebase is uniformly functional-JAX, the
  `ExperimentArm`/callback ecosystem passes trainers as function values, and the public
  `make_train_step` factory already gives callers a persistent jitted step.
- **Strategy objects / enum dispatch vs string flag.** Strategy classes per algorithm would add a
  layer for exactly two variants whose difference is three lines inside one function. String flag
  with build-time validation (`_validate_algorithm`) kept; branches resolve at trace time (all mode
  flags close over the jitted step, matching the codebase's no-static-argnums pattern).
- **Loss-based backprop objective (status quo of the in-progress branch)** — separate
  `_metric_cross_entropy(z_mu, y)` machinery with `metric_fn`/`metric_name`/`loss_type`. Rejected per
  the energy framing: the clamped target node's energy is the same quantity, the graph already
  defines it, and one objective function serves both algorithms with a node-subset argument.
- **Per-token AR normalization** — preserves legacy backprop gradient scale but makes the energy
  definition mode-dependent. Rejected (user decision): per-sample everywhere.
- **`use_causal_mask` caller flag (status quo)** — means different things on v1 vs v2 transformer
  graphs (v2 masks internally; the in-progress code raises on v2). Rejected for graph derivation:
  the `TaskMap` already declares whether a mask node exists.
- **Deprecation shims for one release** — rejected (user decision): clean break at 0.5.0.
- **`ExperimentArm.algorithm` field** — couples the experiment harness to the trainer signature;
  `functools.partial` keeps it trainer-agnostic. Rejected.
- **Tuner pruning on validation perplexity** — objective-aligned but costs one eval pass per epoch;
  train perplexity is free from `metrics["target_energy"]`. Rejected (user decision).

## File changes

**Created:** `fabricpc/training/trainer.py`, `fabricpc/training/generation.py`,
`fabricpc/training/device_utils.py`, `tests/test_trainer.py`.

**Deleted:** `fabricpc/training/{train,train_backprop,train_autoregressive,unified_trainer,multi_gpu}.py`,
`scripts/_parity_check_unified.py`, `scripts/_parity_check_ar.py`,
`tests/test_train_backprop.py`, `tests/test_unified_trainer.py`.

**Modified:** `fabricpc/core/energy.py` (+`graph_energy`), `fabricpc/training/__init__.py`
(exports: `train, evaluate, make_train_step, generate, build_clamps, convert_batch,
create_causal_mask, replicate_params, replicate_opt_state, shard_batch, unshard_metrics`),
`fabricpc/__init__.py` (`train, evaluate` replace `train_pcn, evaluate_pcn`; docstring example),
`pyproject.toml` (0.5.0), `fabricpc/tuning/bayesian_tuner.py`,
`fabricpc/utils/dashboarding/{inference_tracking,callbacks}.py`, plus the callers/tests/docs below.

## Implementation steps (one PR, ordered)

1. `core/energy.py`: add `graph_energy`; unit tests (subset selection, `in_degree>0` default).
2. Create `device_utils.py` (move the four pmap utilities from `train.py`; rename
   `unshard_energies` → `unshard_metrics`).
3. Create `trainer.py` per the design above (reuse the in-progress `run_training_loop`,
   `_validate_algo`, `_accuracy` bodies where they match).
4. Create `generation.py`: port `_generation_step`/`generate` from
   `train_autoregressive.py:377-585`, clamps via `build_clamps`.
5. Rebuild `inference_tracking.py:train_step_with_history` on the shared helpers.
6. Rewire `training/__init__.py` + `fabricpc/__init__.py`; delete the five legacy modules and both
   parity scripts; bump version. **Before deleting:** one-off spot-check that new PC `train` matches
   legacy `train_pcn` bitwise on the `mnist_demo` config (same RNG stream expected).
7. Migrate `bayesian_tuner.py` and `dashboarding/callbacks.py`.
8. Migrate examples/scripts (below).
9. Rewrite/upgrade tests (below).
10. Docs + README + CHANGELOG.
11. Verification (below).

## Caller migration

- Mechanical `train_pcn→train`, `evaluate_pcn→evaluate` (PC arms/all-kwargs sites):
  `examples/{mnist_demo,mnist_conv_demo,mupc_demo,jpc_fc_resnet_compare,mnist_multi_gpu,
  storkey_hopfield_recall,mnist_cyclic_graph,mnist_lateral_connections,storkey_hopfield_demo}.py`,
  `scripts/storkey_hopfield_diagnostic.py`. `resnet18_cifar10_demo.py` additionally adds the
  `metrics` parameter to its `epoch_callback`.
- Manual step loops → `step = make_train_step(structure, optimizer[, algorithm="backprop"])`,
  unpack `(params, opt_state, metrics, final_state)`: `examples/mnist_advanced.py`,
  `examples/scaling/mlp_scaling.py` (positional `loss_type` argument disappears),
  `scripts/storkey_hopfield_diagnostic.py` (two jit-lambda sites).
- `examples/PC_backprop_compare.py`: backprop arm via
  `functools.partial(train, algorithm="backprop")` / `partial(evaluate, algorithm="backprop")`;
  metric key `loss`→`cross_entropy` if used.
- `examples/transformer_demo.py` (v1): hand-rolled AR loops →
  `make_train_step(..., autoregressive=True)`; `build_train_clamps` → `build_clamps(...,
  clamp_target=True, autoregressive=True)`; evals → `evaluate(..., autoregressive=True)` dropping
  `debug=`; `generate_autoregressive→generate` (top_k/top_p unchanged); remove `"use_causal_mask"`
  from config.
- `examples/transformer_v2_demo.py`: four trainer/eval calls → `train`/`evaluate` with
  `autoregressive=True` (+`algorithm="backprop"` for that mode); drop `"use_causal_mask"`;
  `metrics["loss"]`→`["cross_entropy"]`; `generate`.
- `examples/mnist_aim_tracking.py`: `evaluate_pcn→evaluate`; `train_step_with_history` energy-scale
  label update.
- `examples/transformer_tuning.py`: check base_config for retired keys.

## Test plan

New `tests/test_trainer.py`:
- PC parity vs an in-test hand-rolled reference step (clamps → init → inference → energy → local
  grads → optax) at 1e-12 — permanent parity evidence independent of the deleted legacy files.
- Backprop gradient correctness: reference CE loss composed from raw jnp ops on a 2-layer
  softmax graph; `jax.grad` of it vs the step's applied update with `optax.sgd(1.0)`; plus a
  `GaussianEnergy`-output variant asserting the objective equals `0.5·precision·SSE/batch`.
- 4 mode-combo smoke trains (pc/backprop × AR/non-AR; AR via a tiny `create_deep_transformer`):
  finite metrics, params change, `evaluate` returns exactly the per-mode key set.
- Integer targets: int32 `(batch, seq)` token targets and `(batch,)` class labels train in both
  algorithms (regression test for the one-hot gap).
- `build_clamps` units: v1 graph yields the `(batch,1,seq,seq)` tril clamp; v2 graph yields no mask
  key; eval mode leaves targets free.
- Contract guards: unknown algorithm; backprop without `FeedforwardStateInit`; backprop with no
  clamped target; retired config keys raise; PC energy decreases over steps; fractional epochs;
  callback arities; `metrics` dict keys.
- `generate`: shape/dtype/prompt-prefix test (first AR pytest coverage; supersedes the deleted
  parity scripts).

Updates in place: `test_fabricpc.py`, `test_ndim_shapes.py`, `test_optimizers.py`,
`test_storkey_hopfield.py`, `test_conv_pool_integration.py`, `test_mupc.py`,
`test_transformer_nodes.py` (step sites → `make_train_step`; `:443` → `evaluate(...,
autoregressive=True)` with the same key-set assertion); `test_multi_gpu.py` (imports; keep both
train- and eval-pmap parity tests, eval path retained); `test_bayesian_tuner.py` (fakes drive the
metrics-dict callbacks); `test_experiments.py` stubs already match the contract.

## Docs / CHANGELOG

- `docs/user_guides/08_training_and_evaluation.md`: full rewrite — `train`/`evaluate` mode args,
  `make_train_step` custom loops, `generate`, the energy framing (backprop loss = clamped-target
  energy; the output node's energy functional selects the loss), eval key table, graph-derived
  masking, multi-GPU, `ExperimentArm` partial pattern.
- Mechanical renames: `02_quickstart.md`, `03_how_predictive_coding_works.md`, `07_optimizers.md`,
  `09_experiment_tracking.md`, `14_api_data.md`, `15_api_experiments.md`, `README.md`.
  All snippets must bind against the new signatures in the same change
  (`tests/test_doc_snippets.py` enforces this).
- `CHANGELOG.md` `[0.5.0]`: migration table (every removed name → replacement) + behavior notes:
  AR-backprop per-sample objective (÷ legacy lr by seq_len to reproduce), CE eps `clip(1e-7,1)` vs
  `+1e-10`, retired config keys raise, Gaussian-output backprop objective is `0.5·precision·SSE`
  per sample (not element-mean MSE), eval key normalization + transformer-eval double-softmax fix,
  `train_step_with_history` energy scale, backprop gains tqdm/pmap, multi-input eval batches now
  clamp all non-target task keys.

## Verification

```bash
.venv/bin/python -m pytest tests/ -x -q
grep -rn "train_pcn\|evaluate_pcn\|train_backprop\|train_autoregressive\|evaluate_autoregressive\|generate_autoregressive\|evaluate_transformer\|multi_gpu\|unified_trainer\|loss_type\|use_causal_mask\|build_train_clamps" \
    fabricpc examples scripts docs tests README.md | grep -v CHANGELOG   # must be empty
JAX_PLATFORMS=cpu .venv/bin/python examples/mnist_demo.py                # PC smoke
JAX_PLATFORMS=cpu .venv/bin/python examples/transformer_v2_demo.py --num_epochs 0.02          # AR-PC
JAX_PLATFORMS=cpu .venv/bin/python examples/transformer_v2_demo.py --num_epochs 0.02 --mode backprop
JAX_PLATFORMS=cpu .venv/bin/python examples/PC_backprop_compare.py       # ExperimentArm partials
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
    .venv/bin/python -m pytest tests/test_multi_gpu.py -q                # real 2-device pmap leg
.venv/bin/python -c "from fabricpc import train, evaluate; from fabricpc.training import make_train_step, generate"
```

## Risks

- PC bitwise continuity vs legacy `train_pcn` is expected (identical clamp/init/inference/gradient/
  RNG stream) but is spot-checked in step 6 before deletion; the permanent guarantee is the in-test
  reference at 1e-12.
- AR-backprop learning rates need retuning (×seq_len objective scale) — CHANGELOG.
- Transformer eval numbers shift (double-softmax and external-SSE fixes) — correct, but flag.
- `evaluate` now clamps all non-target task keys (legacy clamped only `x`); no current caller has
  multi-input eval batches — CHANGELOG line.
