# Review: `unified_trainer_api.md` — API design

Scope: the public API design in `docs/dev_plans/unified_trainer_api.md` (plan line numbers below refer
to that file), judged against JAX/Flax design practice and training-loop conventions. All source claims
verified on branch `feature/unified-trainer` (HEAD c521b4d). JAX floor is 0.7.0 (`pyproject.toml:34`),
installed 0.10.1.

## Summary

The consolidation is right and the plan's strongest idea — the output node's energy functional selects
the loss, so one objective function serves both algorithms — is correct and verified against
`FeedforwardStateInit` (`state_initializer.py:294-301`). Graph-derived masking and dtype-derived
one-hot are correct applications of the same principle. Four design-level objections remain, each
cheapest to fix inside the same 0.5.0 clean break:

| # | Finding | Severity | Plan section |
|---|---|---|---|
| 1 | `train` neither accepts nor returns optimizer state — non-resumable | major | Public API (plan:110-113) |
| 2 | Multi-device path built on `jax.pmap`, JAX's legacy parallelism API | major | Public API, device_utils (plan:123-124) |
| 3 | `autoregressive` parameter has no remaining training-side effect | major | Public API (plan:100-117) |
| 4 | `evaluate` hard-codes classification metrics, contradicting the energy framing | major | `evaluate` (plan:201-224) |
| 5 | Per-batch `float()` cast blocks JAX async dispatch | moderate | Metrics dict (plan:87-89) |
| 6 | RNG contract undocumented; per-epoch `split(key, max_batches)` couples the stream to loader length | moderate | Training loop (plan:195-199) |
| 7 | Callback contract: six positionals, return-value-rewrites-history, exception-only early stop | moderate | Callbacks (plan:32, 227-231) |
| 8 | `config` has one live key | minor | plan:138-140 |
| 9 | Step returns unused `final_state`; no buffer donation | moderate | make_train_step (plan:103-108) |

## 1. `train` is non-resumable (major)

`train` takes `optimizer`, creates `opt_state` internally, and returns
`(params, iter_results, epoch_results)` (plan:110-113). A second `train` call on the returned params
silently resets Adam moments and any optax schedule's step count. Nothing in the repo can resume
training today — every legacy loop does the same (`train.py:344`, `unified_trainer.py:579`), no caller
serializes anything, and `orbax-checkpoint` is a declared dependency (`pyproject.toml:36`) that is
never imported. The plan enshrines that limitation in a fresh contract: checkpoint/resume, staged
training, and schedule continuation all require abandoning `train` for a hand-rolled
`make_train_step` loop, which recreates the duplication this plan exists to delete.

The Flax convention is a train-state bundle (params, opt_state, step) threaded in and out.
The minimal functional version:

```python
class TrainResult(NamedTuple):
    params: GraphParams
    opt_state: optax.OptState
    iter_results: list
    epoch_results: list

def train(params, structure, train_loader, optimizer, config, rng_key, *,
          opt_state=None,   # created via optimizer.init(params) when None
          ...) -> TrainResult
```

One caller changes: `ab_experiment.py:344` unpacks `trained_params, _, _ =`; it becomes
`result.params`. The plan already migrates every caller and deletes every legacy module — keeping the
legacy 3-tuple while breaking everything else is the one place the clean break stops short.
A NamedTuple also ends the 3/4/5-tuple arity drift the plan's own Context section lists as a defect.

## 2. pmap is the legacy parallelism API (major)

Every multi-device element of the plan — `pmap_single_device`, `device_utils.py`
(`replicate_params`, `replicate_opt_state`, `shard_batch`, `unshard_metrics`, plan:123-124), the
`lax.pmean` + `axis_name` plumbing (plan:189-192), the per-device RNG reshaping, and the ported
zero-padding eval path (plan:31, train.py:573-659) — exists to serve `jax.pmap`. Since JAX 0.4 the
recommended data-parallel mechanism is `jax.jit` over a `Mesh` with `NamedSharding`: params replicated
(`P()`), batch sharded on the leading axis (`P('data')`), and XLA inserts the gradient all-reduce.
The plan targets jax>=0.7 with no upper bound, so a new public surface built on pmap is built on the
API JAX maintains only for compatibility.

What jit-with-sharding deletes from the proposed API:

- `pmap_single_device` — a test-only knob in a public signature. One jitted step runs identically on
  1 or N devices; the multi-device test leg stays `XLA_FLAGS=--xla_force_host_platform_device_count=2`
  with no API cooperation.
- All four `device_utils.py` exports, and with them the module.
- The dual jit/pmap branch in `_make_step` (plan:192) and the `pmean` plumbing.
- The callback ambiguity the plan never resolves: under pmap, params are replicated per device and the
  legacy loop de-replicates before `epoch_callback` (`train.py:455`); the plan does not specify which
  the callback sees. Under jit-sharding there is one params pytree.

What it does not delete: the ragged-final-batch policy (the sharded axis must divide the mesh size),
but that becomes one policy in one code path instead of the current three (skip-with-warning in
training, zero-pad in `evaluate_pcn`, silent drop in `evaluate_transformer` — `train.py:412-421`,
`:611-621`, `:761-762`).

Cost: `test_multi_gpu.py` is rewritten against sharding rather than migrated, and the eval pmap port
(plan:31) is replaced by batch sharding. If that scope is deferred, the fallback is to keep pmap
**private** — do not export the four device utilities (plan:280), so 0.5.0's public surface is not
bound to pmap semantics it will have to break again.

## 3. The `autoregressive` flag is dead by the plan's own logic (major)

The plan derives the causal mask from the graph (`"causal_mask" in structure.task_map`, plan:150-155)
and one-hot from the target dtype (plan:146-149), and states "autoregressive changes nothing in steps
3–6" (plan:75-76). What remains for the flag:

- In `build_clamps`/`make_train_step`/`train`: gating the mask injection — redundant, because the
  task-map key already gates it and no non-AR graph in the repo declares one (the only `causal_mask`
  task keys are `transformer_demo.py:216` and `test_state_initializer.py:253`, both v1 AR).
- In `evaluate`: selecting the `perplexity` result key (plan:217).

Delete the parameter everywhere. For `evaluate`, emit `perplexity` whenever `cross_entropy` is emitted
— `exp` of the per-prediction cross-entropy is defined identically for classification and sequences —
or key it on the target node's energy functional (finding 4). `generate` is inherently
autoregressive and never needed the flag.

Payoff: the mode matrix collapses from `algorithm × autoregressive` (2×2) to `algorithm` (2); the
smoke-test combinations halve (plan:338-340); every `functools.partial` site, the tuner call
(plan:233), and both transformer demo migrations (plan:318-325) shrink. The plan already applied
exactly this argument to kill `use_causal_mask`; it applies verbatim to the flag that replaced it.

Alternative considered — keep the flag as a user-intent assertion that validates the graph is
AR-shaped. Rejected: the plan deletes precisely such an assertion (`_require_causal_mask_node`,
`unified_trainer.py:89-99`) as a defect, and dtype/rank validation already fails fast downstream.

## 4. `evaluate` contradicts the energy framing (major)

The train side lets the graph define the objective; the eval side then returns `accuracy` and
`cross_entropy` "always" (plan:212-215), computed with the `CrossEntropyEnergy` eps on `z_mu`
regardless of the output node's functional. Concrete failures with in-repo graphs:

- `jpc_fc_resnet_compare.py:207,294,377` — `GaussianEnergy` outputs, linear `z_mu`.
  `-Σ y·log(clip(μ,1e-7,1))` on a linear output returns a finite but meaningless number (negative
  activations clip to 1e-7, activations ≥1 clip to 1).
- A Hopfield recall probe (`IdentityNode` output, `GaussianEnergy` default, `identity.py:45`) gets
  `accuracy = argmax(z_mu) == argmax(y)` over a ±1 pattern vector — the index of the largest element,
  not recall. Legacy `evaluate_pcn` has this defect (`train.py:527-534` branches on nothing); the plan
  copies it into a fresh contract and adds the meaningless `cross_entropy` key beside it.

The coherent contract mirrors training: always report `target_energy` (target-node energy under its
own functional ÷ prediction count — the eval twin of the train metric, and an MSE-scale number for
Gaussian outputs); report `cross_entropy`/`perplexity`/`accuracy` only when the target node's
functional is `CrossEntropyEnergy`. The check is free at trace time: `node_info.energy` is a live
instance on static pytree aux data, and `isinstance(..., CrossEntropyEnergy)` is already the tested
idiom (`test_energy.py:177-178`). Conditional keys are not new complexity — the plan's own table
already makes `energy` PC-only and `perplexity` AR-only. Also specify `argmax(..., axis=-1)`; the
legacy hard-coded `axis=1` (`train.py:532-533`) mis-reduces rank>2 outputs.

Every current caller reads only `accuracy` from classification graphs, so nothing breaks today;
the point is that the fresh contract should not guarantee keys it cannot define.

## 5. Per-batch host sync (moderate)

`iter_results[epoch][batch]` stores floats (plan:88-89). A `float()` cast per step blocks on the
device stream, so the host cannot dispatch step N+1 while step N executes — JAX's async dispatch is
forfeited every batch. Legacy `train_autoregressive` already demonstrates the fix: accumulate device
scalars, materialize at epoch end (`train_autoregressive.py:337-344,371`). Recommend: keep metrics as
device scalars in the loop; convert at epoch boundaries; materialize per batch only when an
`iter_callback` is present or tqdm postfix display demands it (and gate the postfix on `verbose`).

## 6. RNG contract (moderate)

The key is consumed only by latent initialization — `run_inference` takes no key and there is no
dropout or noise anywhere in the package. The plan should state this contract; it is the difference
between "reproducible given the loader" and users assuming stochastic training. Second, the preserved
legacy stream is `split(epoch_key, max_batches)` (`train.py:401`): every per-batch key shifts when the
loader length changes. The modern idiom `fold_in(epoch_key, batch_idx)` decouples keys from batch
count. Sequencing if adopted: run the step-6 bitwise spot-check against legacy first, then switch
streams before writing the permanent 1e-12 in-test reference — after the legacy files are deleted
there is never a cheaper moment. Optional; documenting the contract is not.

## 7. Callback contract (moderate)

Three separable points on `epoch_callback(epoch_idx, params, structure, config, rng_key, metrics)`
(plan:32):

- Six positional arguments, of which the tuner — the main consumer — uses one
  (`bayesian_tuner.py:114-121` binds and ignores params/structure/config/rng_key). The AR-only
  `energy=`/`ce_loss=` kwargs this plan is killing (plan:230) are the precedent for how a positional
  contract rots: it cannot grow without breaking every implementor. A single context argument
  (a NamedTuple with those six fields) grows by field addition instead.
- The return-value-replaces-stored-history convention is load-bearing (dashboards store mid-training
  eval dicts into `epoch_results` via it, `callbacks.py:70-72`) but appears only in the plan's metrics
  section (plan:88-89). It belongs in the callback contract itself.
- Early stopping is exception-only: tuner pruning works because `optuna.TrialPruned` escapes the epoch
  loop (`bayesian_tuner.py:134,144,178`). That is fine, but it is an implicit guarantee — one
  defensive `try/except` around callbacks silently breaks pruning. State the guarantee ("callback
  exceptions propagate") in the contract and add it to the test list (plan:344-346 covers arities but
  not propagation).

## 8. `config` has one live key (minor)

`config` reads only `num_epochs` (plan:138). The dict survives because the `ExperimentArm` positional
contract and callbacks carry it, which is legitimate — but the plan should say the trainer treats it
as otherwise opaque, and in particular that settling parameters (`infer_steps`, `eta_infer`) live in
the inference object inside `structure.config` (`inference.py:258-261`), not here; that confusion is
the likeliest user error with a parameter named `config`. Fractional epochs: define what
`epoch_results` contains for a partial epoch (mean over the batches run, presumably).

## 9. Step outputs and donation (moderate)

- The internal training-loop step should not return `final_state`. XLA cannot dead-code-eliminate a
  jitted output the caller discards, so every node's latents/errors/energies stay live across the
  optimizer update every batch. Keep `final_state` on the public `make_train_step` (the escape hatch
  the dashboards and `mnist_advanced`-style loops need); build the internal loop's step without it.
- Donate params and opt_state in the internal loop's step (`jax.jit(step, donate_argnums=(0, 1))`) —
  both are dead after the call, and donation removes a full extra copy of the model from peak memory.
  The repo currently uses donation nowhere. Leave the public factory non-donating: a donated input
  buffer raises on reuse, and external callers legitimately keep initial params (the parity tests
  deepcopy for exactly this reason).
- `metrics["energy"]` denotes different quantities per algorithm — all internal nodes for PC, target
  nodes only for backprop (plan:80) — and its normalization (÷batch) differs from `target_energy`'s
  (÷predictions) inside the same dict. Document both in the metrics section and CHANGELOG, and state
  that cross-algorithm energy comparison is invalid; `ab_experiment` users comparing PC vs backprop
  arms are one `metric="energy"` away from a nonsense comparison.

## Minor

- Annotate `algorithm: Literal["pc", "backprop"]` — free IDE/typecheck support for the string enum.
- `verbose=True, use_tqdm=True` overlap; collapse to one progress control or define precedence.
- `evaluate` on a graph with no target task key: define it (raise). Legacy returns `energy: 0.0`
  through the `total_samples` guard (`train.py:655`).
- The plan's keyword-only-after-`rng_key` rule (plan:134-136) is right; note the in-progress
  `unified_trainer.py:699-716` is positional-or-keyword, so the implementation must follow the plan,
  not the branch.

## Errata in the plan text

- plan:149 — the int32 target is materialized at `dataloader.py:336` and `:466`; `:268` is the yield.
- plan:233-236 — the tuner migration bullet misses `_log` (`bayesian_tuner.py:230-247`), which reads
  the `loss` key the plan renames to `cross_entropy`; unmigrated it logs 0.0 silently.
- plan:146 — the one-hot rule keyed on "integer dtype" leaves bool targets hitting the
  `_validate_clamp_dtypes` raise (`state_initializer.py:333` accepts floating only). One-hot every
  non-float target, or name bool in the fail-fast message.
- plan:31/:207 — the pmap eval path being ported sums `in_degree>0` energy (`train.py:631`) while the
  single-device `eval_step` sums all nodes (`train.py:519-523`). The plan's key table says internal
  nodes; the port must reconcile the two definitions, not copy either verbatim.

## Recommended amendments to `unified_trainer_api.md`

1. `train` accepts `opt_state=None` and returns `TrainResult(params, opt_state, iter_results,
   epoch_results)`; migrate the one arm unpack site (`ab_experiment.py:344`). [finding 1]
2. Replace pmap with jit + `NamedSharding` data parallelism; delete `pmap_single_device` and
   `device_utils.py` from the public API. If deferred: keep pmap machinery private, unexported. [2]
3. Delete `autoregressive` from all signatures; mask from `task_map`, one-hot from dtype,
   `perplexity` emitted alongside `cross_entropy`. [3]
4. `evaluate` keys metrics off the target node's energy functional: `target_energy` always;
   `cross_entropy`/`perplexity`/`accuracy` only for `CrossEntropyEnergy` targets; `argmax` on
   `axis=-1`. [4]
5. Metrics stay device scalars in-loop, materialized at epoch end or when a callback needs them. [5]
6. Document the RNG contract; optionally move to `fold_in` after the legacy parity spot-check. [6]
7. Callback contract: single context argument (or explicitly frozen positional list), the
   return-replaces-history rule, and the exceptions-propagate guarantee, each stated and tested. [7]
8. Fold in the four errata. [Errata]
