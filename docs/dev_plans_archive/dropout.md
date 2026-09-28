# Dropout with per-batch masks shared by PC and backprop

Line references are to the base commit of this branch.

## Context

FabricPC has no dropout. The trainer's RNG contract says so outright: "the training key affects only latent initialization ... the package has no dropout or noise" (`fabricpc/training/trainer.py:46`, repeated in the `train()` docstring at line 537 and in `docs/user_guides/08_training_and_evaluation.md` under "The RNG contract"). The v1 `TransformerBlock` accepts a `dropout_rate` that is documented as unused (`fabricpc/nodes/transformer.py:146`), and the v2 demo attributes part of its BPE overfitting to the missing regularization (`examples/transformer_v2_demo.py:63`). The unified trainer design already listed "RNG seed splitting into the inference path for future dropout" as follow-up work (`docs/dev_plans_archive/unified_trainer_api.md:806`).

Dropout in a PC graph has a constraint that feedforward libraries do not face. One training step evaluates each node's `predict()` many times on the same batch:

1. `FeedforwardStateInit` runs one forward pass to set `z_latent = z_mu` (`fabricpc/graph_initialization/state_initializer.py:278`).
2. `run_inference` calls every node's forward once per sweep inside `lax.fori_loop` (`fabricpc/core/inference.py:311`).
3. `compute_local_weight_gradients` runs the forward again per node (`fabricpc/core/learning.py:6`).
4. Under backprop, the objective differentiates one forward pass (`fabricpc/training/trainer.py:414`).

If each evaluation drew a fresh mask, the energy PC relaxes would change every sweep, the settled state would not be the equilibrium of any one network, and the feedforward-init invariant (zero error at unclamped nodes) would break at step 0. PC and backprop would also train different thinned networks for the same batch, which defeats PC-vs-backprop gradient comparisons at a shared checkpoint.

`predict(params, inputs, state, node_info)` (`fabricpc/nodes/base.py:333`) has no RNG argument, so the randomness source has to reach the node through one of these four arguments.

**Intended outcome.** Inverted dropout that is (a) fixed for a whole training step, (b) identical under PC and backprop for the same batch key, (c) reproduced by the tools that re-run a training step, (d) off in evaluation and generation, and (e) bitwise-neutral for graphs that configure no dropout.

## Design

### One key per node, carried in `NodeState`

`NodeState` gains a sixth field, `dropout_key: Optional[jnp.ndarray] = None` (`fabricpc/core/types.py:132`), registered as a pytree child. `None` means dropout is off. A node rebuilds its mask from this key on every `predict()`, so all evaluations on one state draw the same mask. The key is a pytree leaf that no solver updates: every solver writes states through `_replace`, which preserves it, and ePC differentiates only the error pytree.

`fabricpc/core/dropout.py` (new) holds:

- `dropout(x, rate, key, site)`: inverted dropout, `where(bernoulli(fold_in(key, site), 1 - rate), x / (1 - rate), 0)`. It returns `x` itself when `rate == 0.0` or `key is None`; both are static at trace time, so a graph without dropout computes exactly what it computed before. `site` separates independent dropout sites inside one node.
- `node_dropout_key(dropout_key, node_name)`: `fold_in(dropout_key, crc32(node_name))`. Hashing the name rather than the node's position keeps a node's masks stable when other nodes are added; `crc32` is stable across processes, unlike Python's salted `hash`.
- `attach_dropout_keys(state, dropout_key)`: writes each node's key; a no-op for `None`, idempotent otherwise.
- `validate_dropout_rate(node_name, param, rate)`: rejects rates outside `[0, 1)` at construction.
- `training_dropout_key(batch_key)`: `fold_in(batch_key, DROPOUT_STREAM)`, the one definition of a training step's dropout key, used by every caller that trains or reproduces a training step.

### Key plumbing through state initialization

`initialize_graph_state(..., dropout_key=None)` (`fabricpc/graph_initialization/state_initializer.py:362`):

- passes the key to the initializer through the existing `config` dict (`{**state_init.config, "dropout_key": key}`), so `StateInitBase.initialize_state`'s signature does not change;
- after the initializer returns, calls `attach_dropout_keys` so that every initializer, custom ones included, ends with keys attached.

`FeedforwardStateInit` reads `config.get("dropout_key")` and attaches the keys before its forward pass, so that `z_latent = z_mu` holds under the same masks that inference will use. Initializers without a forward pass need no change.

Cost of this channel: a custom initializer that runs its own forward pass and ignores `config["dropout_key"]` computes its latents without masks. The keys are still attached afterwards, so inference runs with dropout but starts from nonzero internal errors, and nothing raises. Guide 06 and the `initialize_graph_state` docstring state the contract; no custom initializer in the repository runs a forward pass.

### Trainer

`_batch_grads` (`fabricpc/training/trainer.py:391`) derives `dropout_key = training_dropout_key(rng_key)` from the batch key and passes it to both `initialize_graph_state` calls (lines 400 and 414). The batch key itself still drives latent initialization unchanged, so the latent-initialization stream is bit-for-bit the same as before. `evaluate` (line 799) passes no key.

### Nodes

Each node validates its rate in `__init__`, stores it in `node_config`, and applies `dropout(node_config.get("dropout_rate", 0.0), state.dropout_key, site)` in `predict()`. The read keeps a `0.0` default because `NodeInfo` is a public dataclass that tests and external code build directly, without the constructor; there a missing rate means no dropout. A strict `node_config["dropout_rate"]` read breaks the existing node-level tests (`tests/test_auto_node_grad.py`, `tests/test_convolutional.py`) with `KeyError`.

| Node | Site(s) |
|---|---|
| `EmbeddingNode(dropout_rate)` | embedding output |
| `MhaResidualNode(dropout_rate, attn_dropout_rate)` | attention probabilities (site 0); attention branch before the residual add (site 1) |
| `Mlp2ResidualNode(dropout_rate)` | MLP branch before the residual add |
| `Linear(dropout_rate)` | activation output |
| `LinearResidual(dropout_rate)` | transform path before the residual add; the skip path is never dropped |
| `ConvNode(dropout_rate)` | activation output, element-wise |

The transformer placement follows GPT-2. `LnMlp1Node` and `VocabProjectionNode` carry no dropout. `create_deep_transformer` gains `dropout_rate` (embedding and both residual branches) and `attn_dropout_rate`, both defaulting to `0.0`.

`LinearExplicitGrad` subclasses `Linear` but does not expose `dropout_rate`: its analytic gradients assume the undropped prediction. Its `Linear.__init__` call stores the default `0.0`, so it never drops.

The v1 `TransformerBlock` keeps its `dropout_rate` parameter (`fabricpc/nodes/transformer.py:162`) but now raises on a non-zero value instead of silently ignoring it, pointing to the v2 nodes.

## Migration inventory

Every library caller of `initialize_graph_state` and the decision for each. The rule: a caller that trains, or that re-runs a training step to observe it, passes `training_dropout_key(key)`; evaluation, generation, and general-purpose probes pass none.

| Caller | Purpose | Dropout key |
|---|---|---|
| `fabricpc/training/trainer.py:400` (`_batch_grads`, PC) | training step | **passed** |
| `fabricpc/training/trainer.py:414` (`_batch_grads`, backprop objective) | training step | **passed** |
| `fabricpc/utils/dashboarding/inference_tracking.py:316` (`train_step_with_history`) | training step, documented as the trainer's PC step with history | **passed** |
| `fabricpc/utils/dashboarding/callbacks.py:44` (`make_tracked_settle`) | re-settles a trained batch with its `batch_key` for the dashboard; it must show the trajectory the step took, and the same callback's energy and backprop state already come from the step | **passed** |
| `fabricpc/training/regime_probe.py:153` (`RegimeProbe`) | judges whether the training settle is stable (λ_max against 2/η). Inverted dropout scales the kept units, which raises λ_max, so a keyless probe understates it | **passed** (one fixed mask from the probe key) |
| `fabricpc/training/trainer.py:799` (`evaluate`) | evaluation | none: deterministic |
| `fabricpc/training/generation.py:56` (`generate`) | generation | none: deterministic |
| `fabricpc/utils/dashboarding/inference_tracking.py:140` (`make_tracked_probe`) | general probe of a batch the caller supplies | none; the caller passes a key through `initialize_graph_state` when it wants masks |

Other shared code that depends on node semantics: `validate_linear_gaussian` (`fabricpc/utils/linear_pc_oracle.py:273`) accepted any `Linear`; its closed-form equilibrium does not hold under a mask, so it now rejects nodes with dropout. The callers in `examples/` and `scripts/` build graphs that configure no dropout, so they are unaffected. Guide 15's advice to measure λ_max at init now says to initialize with the training dropout key on a graph with dropout.

`NodeState` construction sites: the three built-in initializers (`state_initializer.py`) and the tests construct five-field states and get the `None` default. No call site changes.

Tests that pin current behavior across the move: the full existing suite (every graph at the default rate `0.0`), plus `test_no_key_matches_zero_rate_model` and `test_zero_rate_with_key_matches_no_key` in `tests/test_dropout.py`.

## Tests

`tests/test_dropout.py` (new), on the v2 transformer and on a `ConvNode -> Linear -> LinearResidual -> Linear` graph:

- helper: identity when off, inverted scaling and rate, determinism per key and site, gradients only through kept units, name-based node keys;
- construction: invalid rates raise; rates reach `node_config`; five-field `NodeState` defaults to no key; parameter initialization is independent of the rate;
- rate `0` or no key equals the no-dropout graph; every dropout site changes its node's prediction when on; the `LinearResidual` skip path is never dropped;
- PC consistency: zero internal energy after feedforward init under dropout; with `eta_infer = 0`, several sweeps leave the state unchanged (a per-sweep mask would not);
- PC and backprop share masks: at `infer_steps = 0` the output-layer update is identical under both algorithms;
- masks stay fixed under ePC and under an ePC-then-sPC `InferenceSchedule`: with zero step sizes, `z_latent` and the internal energies stay at their feedforward-init values (a mask that differed in ePC's derived forward would move them);
- training-step masks depend only on the batch key and equal `node_dropout_key(training_dropout_key(batch_key), name)`;
- the dashboard re-settle with a batch key reaches the training step's settled state (and `tests/test_dashboarding_callbacks.py::test_resettle_states_match_make_inference_history_on_context` now builds its reference with the training dropout key);
- the v1 `TransformerBlock` rejects a non-zero rate; the linear oracle rejects nodes with dropout;
- finite training steps under PC, backprop, and ePC; `evaluate` ignores dropout.

`tests/test_inference_tracking.py`: the `train_step_with_history` parity test is parametrized over `dropout_rate` in `{0.0, 0.5}` (via `make_classification_structure(hidden_dropout_rate=...)` in `tests/conftest.py`), pinning its masks to the trainer's.

`tests/test_bayesian_tuner.py`: a dropout trial completes with a finite validation perplexity, logs its rate, and scores differently from the same trial without dropout.

## Alternatives considered

**An RNG argument on `predict()`.** Explicit, and familiar from Flax. Rejected: it changes the node contract for every built-in and custom node, the base-owned templates (`forward`, `forward_with_aux`, `forward_from_error`), and ePC's derived forward, all for a feature most nodes do not use. The key in `NodeState` reaches `predict()` through an argument it already has.

**A fresh mask per forward call** (the feedforward-library convention: split a key each call and thread it through the inference loop). Rejected: the mask would change between sweeps, so inference would relax a different energy each sweep, the feedforward-init invariant would fail, and PC and backprop would see different networks. This is the failure the design exists to prevent.

**Masks stored as arrays in `NodeState`** instead of a key. Saves regenerating masks on each call. Rejected: memory scales with activations per site (the attention-probability mask is `(batch, heads, seq, seq)` per block), mask shapes differ per site and node type so the state schema would become node-specific, and the key is two `uint32`s.

**A dedicated `DropoutNode`** between layers. Needs no node changes. Rejected: it adds a latent variable and an energy term per dropout site, which changes the PC dynamics being regularized; it cannot express dropout inside a node (attention probabilities, a residual branch before its add); and it doubles the graph for a standard layer stack.

**A graph-level `training` flag in the structure config.** Explicit and easy to read at the call site, the pattern PyTorch's `train()`/`eval()` users expect. Rejected: the structure is static under `jit`, so toggling the flag recompiles, and a randomness source is still needed. The key's presence (`None` or not) already acts as the flag with no recompilation.

**A new `dropout_key` parameter on `StateInitBase.initialize_state`.** Visible in the signature, so a custom initializer cannot overlook it. Rejected: it breaks every custom initializer's signature. The existing `config` dict carries the key to initializers that need it (only those with their own forward pass), and `initialize_graph_state` attaches keys afterwards for the rest; the cost is the silent case described under "Key plumbing".

## Out of scope, with reasons

- **Dropout inside the v1 `TransformerBlock`**: it raises on a non-zero rate instead. The block is monolithic, and the v1-vs-v2 fork is an open item; wiring it is a separate change.
- **Other nodes** (`IdentityNode`, `SkipConnection`, pooling, `StorkeyHopfield`): no learned transformation to regularize. Custom nodes opt in as described in `docs/user_guides/06_custom_nodes.md`.
- **Spatial or channel dropout for `ConvNode`**: element-wise only; a structured variant is a separate option.

## Verification

```bash
pytest tests/test_dropout.py tests/test_inference_tracking.py tests/test_bayesian_tuner.py -q
pytest -q
pre-commit run --all-files
```

The dropout, trainer, state-initializer, transformer-node, and tuner tests also pass with `jax==0.7.0`, the declared floor.

Default behavior is unchanged by construction: every rate defaults to `0.0`, `dropout()` returns its input unchanged at rate `0.0`, and the latent-initialization stream is untouched. The existing suite runs every graph at the default rate and passes unchanged. Checked directly: training the same graphs with the same keys for two epochs on the base commit and on this branch (sPC, backprop, ePC, an ePC-then-sPC `InferenceSchedule`, and the v2 transformer under PC and backprop, all at rate `0.0`) gives bitwise-equal final parameters. To reproduce, run `train(...)` on any rate-0 graph from both checkouts and compare the parameter leaves with `np.array_equal`.
