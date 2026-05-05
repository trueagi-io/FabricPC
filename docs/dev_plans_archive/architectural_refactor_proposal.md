# Architectural refactor proposal — FabricPC-JAX

Date: 2026-05-04
Branch at time of review: `clinfra`
Scope reviewed: `core/`, `nodes/`, `training/`, `graph_assembly/`, `graph_initialization/` — ~8.8k LOC across 51 modules.

## Executive summary

The codebase is sound at the level of individual abstractions. `NodeBase` has a minimal contract, `GraphStructure` is a clean static pytree, and the assembly/initialization split is principled. The architectural issues sit at **module boundaries** and **parallel-implementation** seams, where recent feature growth (muPC, transformers, backprop comparison) was added alongside existing code instead of through it.

The proposal below is organized in three tiers by leverage. Tier 1 items are project-shape changes that benefit everything downstream; Tier 2 items are targeted refactors; Tier 3 items are cheap cleanup. A suggested execution order is at the bottom.

---

## Tier 1 — High-leverage restructuring

### 1.1 Collapse the three trainers into one loop

`training/train.py` (845), `training/train_autoregressive.py` (677), and `training/train_backprop.py` (710) are ~60–70% mechanical duplication.

**Concrete duplication:**

- The fractional-epoch + batch-iteration scaffold appears **four times** — `train.py:357-466`, `train_autoregressive.py:206-296`, `train_backprop.py:171-254`, `train_backprop.py:389-475`.
- The 6-line `batch → jnp.array` conversion is reimplemented inline **six times** (`_convert_batch` exists at `train.py:100` and is then ignored by every other module).
- The `evaluate_autoregressive` / `evaluate_backprop_autoregressive` pair (`train_autoregressive.py:559-677` vs `train_backprop.py:594-710`) are line-for-line near-duplicates including a copy-pasted 25-line `[DEBUG]` block (`train_autoregressive.py:628-653` vs `train_backprop.py:659-684`).
- `train_backprop.py:29` already imports helpers from `train_autoregressive.py` — the precedent for de-duplication exists, it just stopped after two functions.

**Why consolidation is feasible.** The actual semantic axis (PC iterative inference vs single-pass backprop, with/without causal mask) is a tiny fraction of surface area and is already pluggable through `structure.config["graph_state_initializer"]`. Backprop is "PC + `FeedforwardStateInit` + skip the inference iterations".

**Asymmetric capability that should be lifted.** Pmap is wired only into `train.py`. Per the global rule "fix infrastructure for expanded scope, don't patch the first failure", the right move during consolidation is to lift pmap into the unified loop, not preserve a dual-mode flag.

**Target shape:**

- `training/loop.py` — unified `train(...)` / `evaluate(...)` (epoch scaffold, batch conversion, callbacks, tqdm, optional pmap dispatch)
- `training/steps.py` — `pc_step`, `backprop_step` (pure `(params, opt_state, batch, structure) → (params, opt_state, metrics)`); causal-mask clamp added by a small `clamp_builder` callable, not by branching inside the step
- `training/metrics.py` — accuracy / perplexity / CE-loss helpers (kills the duplicated debug block)
- `training/generation.py` — `generate_autoregressive` (`train_autoregressive.py:407`) is the one genuinely separate concern (inference-only, scan-based)
- `training/optimizers.py` — natural-gradient transforms (current `natural_gradients.py` content, after rename)

Estimated collapse: ~2200 → ~500–700 LOC, with strictly more capability (pmap available to AR/backprop too).

**Blocker to surface first.** The transformer evaluator at `train.py:770-794` adds an explicit external `(pred − target)²` term that no other evaluator uses. That's a real semantic divergence — needs a deliberate decision before merging, not a silent unification.

### 1.2 Decide the transformer v1 vs v2 fork

`nodes/transformer.py:121` (`TransformerBlock`, monolithic LN→MHA→+→LN→FFN→+) and `nodes/transformer_v2.py` (5-node PC decomposition: `EmbeddingNode`, `MhaResidualNode`, `LnMlp1Node`, `Mlp2ResidualNode`, `VocabProjectionNode`) **coexist; v2 is not a replacement.**

- v1 has the muPC variance machinery worked out: `1/√2` residual scaling (`transformer.py:319`), per-position `√eff_ctx` softmax compensation (`transformer.py:343-353`), and an LN-absorbs-scale weight-grad fix (`transformer.py:392-422`).
- v2 nodes do **none** of this — `MhaResidualNode.forward` (`transformer_v2.py:243`) is a plain `z_mu = x + mha`.
- `examples/transformer_demo.py:56` uses v1 `TransformerBlock` *and* v2 `EmbeddingNode` together — cross-version coupling already in the wild.

**The decision needed:** either port the muPC compensation into v2 nodes and retire v1, or move v2 under `experimental/` and stop adding callers to it. The current "both, partially" state is the worst option.

**Side benefit of the decision.** The `nodes ↔ graph_assembly` import cycle that `__init__.py:46-47` documents is a direct consequence — `create_deep_transformer()` is a 95-line graph-builder living inside a node file (`transformer_v2.py:437-531`). Moving it to `examples/` or `fabricpc/models/` removes the cycle for free.

**Related duplication.** `core/positional.py` (RoPE, used only by v2) and `nodes/transformer.py:28-118` (RoPE, used only by v1) define **two parallel implementations** with the same function names but **different return types and signatures** (real cos/sin tuple vs complex `freqs_cis`). Pick one, migrate the other caller.

### 1.3 Introduce a typed Node protocol in `core/`

`core/inference.py` and `core/learning.py` call into nodes polymorphically through `node_info.node_class.forward_and_latent_grads(...)` (`inference.py:171`) and `node_class.forward_and_weight_grads(...)` (`learning.py:44`). The contract is real (`nodes/base.py:347, 448`) but not declared in `core/`. Result: `GraphStructure.nodes: Dict[str, Any]` (`types.py:165`) — `Any`-typed because `NodeBase` lives downstream.

**Proposal.** Lift a thin `NodeProtocol` (or ABC) into `core/`. `GraphStructure.nodes` becomes properly typed, `core/` stops dispatching through stringly-named classes, and the contract becomes self-documenting. This is the single change that most clarifies the layering and unblocks future static analysis.

### 1.4 Decide muPC: first-class core concept, or external registry

muPC concerns are spread across "generic" types as optional fields:

- `SlotInfo.is_variance_scalable`, `SlotInfo.is_skip_connection` (`types.py:26-29`)
- `NodeInfo.scaling_config: Optional[MuPCScalingFactors]` (`types.py:61`)
- `ActivationBase.variance_gain`, `ActivationBase.jacobian_gain` (`activations.py`, only consulted by `mupc.py:286-287`)
- `core/inference.py:28` imports `core/scaling.py` and calls `scale_inputs` / `scale_input_grads` / `scale_self_grad` even when `scaling_config is None` (degenerate no-ops)

This is the worst of both worlds: muPC is structurally embedded in supposedly-generic types, but every callsite has to handle the `None` case. Two coherent endpoints:

- **Commit:** drop the `Optional`, every node carries muPC factors, scaling is part of the universal forward pass. Aligns with the "design for expanded scope" rule — muPC is now the production path.
- **Extract:** muPC consults a separate registry (`Dict[node_name, MuPCScalingFactors]`) at `inference.py` only when present. `SlotInfo` and `ActivationBase` lose the muPC-specific flags; `mupc.py` consults its own activation-gain table.

**Recommended endpoint: commit.** The codebase has been moving deeper into muPC for several recent commits (`2e6ae62`, `cd4a9f7`, `370b303`, `295f5fc`); the optionality is no longer pulling its weight.

---

## Tier 2 — Targeted refactors

### 2.1 Node duplication

- **`SkipConnection.forward` is a 25-line copy of `IdentityNode.forward`** (`skip_connection.py:99-124` vs `identity.py:113-164`). Only difference is a slot flag and a `scale` config that `SkipConnection` lacks. Make `SkipConnection` a one-line subclass overriding only `get_slots()`.
- **`LinearResidual.initialize_params` and `forward` are 90% copies of `Linear`'s** (`linear_residual.py:128-141, 160-202` vs `linear.py:131-148, 185-219`). Refactor to inherit from `Linear` and override only `get_slots()` plus a small forward post-step that adds the skip sum.
- **`LinearExplicitGrad`** (`linear_explicit_grad.py`) is a test-and-demo asset that raises `NotImplementedError` for any non-Gaussian energy. Keep it as a subclass-override demonstration, but consider relocating to `tests/` or labeling it explicitly as a reference implementation in `nodes/__init__.py`.

### 2.2 Module-boundary fixes

- **`core/positional.py` (RoPE) is not core** — it's a transformer building block (only consumer is `nodes/transformer_v2.py`). Move to `nodes/_transformer_helpers.py` or similar.
- **`graph_assembly/graph_construction.py:11` imports `FeedforwardStateInit` from `graph_initialization`** — wrong-direction dependency. Either move `FeedforwardStateInit` to `graph_assembly` (it's the assembly-time default), or pass the default in via the caller.
- **`learning.py:2` imports `gather_inputs` from `inference.py`** for a pure graph-traversal helper. Move to `core/state_ops.py` (or a new `core/graph_ops.py`) and let both import it.
- **`FeedforwardStateInit.initialize_state`** (`state_initializer.py:228-309`) replicates the muPC `scale_inputs` + `forward` sequence that the inference loop also runs. Two callsites that must stay in sync. Extract a shared `run_node_forward_with_scaling` helper.

### 2.3 `InferenceBase` re-shaping

`InferenceBase` (`inference.py:71`) uses every method as `@staticmethod` then dispatches via `cls = type(structure.config["inference"])` — a workaround for JAX tracing the `self` of a class instance. Only `compute_new_latent` is a genuine extension point; the rest is fixed PC mechanics.

**Reshape into an "optimizer-like" protocol:** a frozen-dataclass config object plus two pure functions (`update_latents`, optional `init_state`). Halves the abstraction surface and makes adding momentum/adaptive variants trivial — currently impossible without changing the `NodeState` pytree layout (`types.py:124-129`), which is a real extensibility gap.

### 2.4 LayerNorm-absorbs-scale compensation

`TransformerBlock.forward_and_weight_grads` LN-compensation logic (`transformer.py:392-422`) will be needed by every node containing internal LayerNorm. Lift to a reusable helper on `NodeBase` (or on a `LayerNormMixin`) when the second node needs it — by the rule "fix the infrastructure for the expanded scope," that moment is when v2's `MhaResidualNode` and `LnMlp1Node` are made muPC-correct (Tier 1.2).

---

## Tier 3 — Cleanup (cheap and safe)

- **`training/multi_gpu.py`** — fully deprecated module, every function emits a `DeprecationWarning` and forwards to `train.py`. No callers in `examples/`, `tests/`, or `fabricpc/`. Delete + drop the aliases at `training/__init__.py:32-35`.
- **`training/optimizers.py`** — 13-line re-export shim, two callers disagree on which import path to use (`tests/test_optimizers.py:11` uses the shim; `examples/mnist_advanced.py:37` skips it). Pick one path, delete the other.
- **`MuPCConfig.depth_metric`, `MuPCConfig.min_depth`** (`mupc.py:140-159`) — explicitly deprecated, no internal users.
- **Dead public exports** in `core/__init__.py`: `compute_energy`, `compute_energy_gradient`, `get_energy_and_gradient` (~65 lines of public surface, zero callers). `OnesInitializer`, `UniformInitializer` similarly unused internally.
- **Stale module references in docstrings** — `core/scaling.py:14` points at `graph_net.py` (renamed in `370b303`), `core/mupc.py:122` points at the `builder/` package (renamed to `graph_assembly/` in `2e6ae62`).
- **`ZerosInitializer.__init__` accepts `gain` and silently discards it** (`initializers.py:93-101`) — the constructor doesn't store it and `initialize` returns `jnp.zeros(shape)` regardless. Either honor it or drop it.
- **`UniformInitializer` stores config under `"min"`/`"max"` keys** (`initializers.py:162`) while taking constructor args `min_val`/`max_val` — surprising for callers reading `init.config`.
- **`DEFAULT_ENERGY` / `DEFAULT_ACTIVATION` class attributes** in `transformer_v2.py:45-46` — never read anywhere. Delete or wire them through `NodeBase.__init__`.
- **`SoftmaxActivation.derivative`** (`activations.py:345`) returns the diagonal `s*(1-s)` and admits in-comment that it's wrong for non-element-wise use. Footgun for any future non-autodiff caller — either rename or delete (the full Jacobian at `SoftmaxActivation.jacobian` is the correct form).
- **Repeated `MappingProxyType` immutability pattern** in `InferenceBase`, `EnergyFunctional`, `ActivationBase`, `InitializerBase` (4× verbatim) — extract a mixin.

---

## What to leave alone

- The `NodeBase` contract (`get_slots`, `initialize_params`, `forward`, plus `forward_and_latent_grads` / `forward_and_weight_grads` defaults) is genuinely well-designed. The terminal/dangling/internal handling at `base.py:347` is graph-shape concern, not node-type concern, and belongs there.
- The `graph_assembly` ↔ `graph_initialization` seam (topology vs tensor allocation) is principled; only the import direction needs fixing.
- The `GraphParams` / `GraphState` / `GraphStructure` static/dynamic pytree split is correct.
- `SlotSpec`'s cross-flag invariant in `__post_init__` (`base.py:67`) — small, the right place.
- `mupc.py` ↔ `scaling.py` ↔ `initializers.MuPCInitializer` separation — `mupc.py` produces factors, `scaling.py` consumes them, `MuPCInitializer` samples unit-variance weights. Cleanly factored; only `TYPE_CHECKING` dependency between modules.

---

## Suggested execution order

1. **Tier 1.4 (commit muPC)** + **Tier 1.3 (NodeProtocol)** — both are type-system clarifications that make subsequent work safer. `GraphStructure.nodes` becomes properly typed; muPC stops being conditional.
2. **Tier 1.1 (trainer consolidation)** — the largest LOC win, lower risk after #1 because the `step_fn` interface is stronger.
3. **Tier 1.2 (transformer fork decision)** — needs a research-direction call, not just a refactor. Defer until v1's muPC machinery is documented well enough to port.
4. **Tier 2 + 3** — cherry-pick as you encounter the relevant code; most are isolated, can be bundled with adjacent feature work.

## Open decisions for the project owner

- **muPC commit vs extract** (Tier 1.4) — recommendation is commit; confirm.
- **Transformer v1 vs v2** (Tier 1.2) — port v1's variance machinery into v2 and retire v1, or move v2 to `experimental/`?
- **Trainer consolidation timing** (Tier 1.1) — block on the CL_plan work landing first, or pursue in parallel on a separate branch?
- **Transformer evaluator's external `(pred − target)²` term** (`train.py:770-794`) — keep, drop, or align with the standard PC energy?

---

## Inventory of files referenced

Core:
- `fabricpc/core/types.py`
- `fabricpc/core/topology.py`
- `fabricpc/core/state_ops.py`
- `fabricpc/core/scaling.py`
- `fabricpc/core/positional.py`
- `fabricpc/core/learning.py`
- `fabricpc/core/__init__.py`
- `fabricpc/core/inference.py`
- `fabricpc/core/energy.py`
- `fabricpc/core/activations.py`
- `fabricpc/core/mupc.py`
- `fabricpc/core/initializers.py`

Nodes:
- `fabricpc/nodes/base.py`
- `fabricpc/nodes/__init__.py`
- `fabricpc/nodes/identity.py`
- `fabricpc/nodes/linear.py`
- `fabricpc/nodes/linear_residual.py`
- `fabricpc/nodes/linear_explicit_grad.py`
- `fabricpc/nodes/skip_connection.py`
- `fabricpc/nodes/storkey_hopfield.py`
- `fabricpc/nodes/transformer.py`
- `fabricpc/nodes/transformer_v2.py`

Training:
- `fabricpc/training/train.py`
- `fabricpc/training/train_backprop.py`
- `fabricpc/training/train_autoregressive.py`
- `fabricpc/training/__init__.py`
- `fabricpc/training/optimizers.py`
- `fabricpc/training/natural_gradients.py`
- `fabricpc/training/multi_gpu.py`

Assembly / initialization:
- `fabricpc/graph_assembly/graph_construction.py`
- `fabricpc/graph_assembly/__init__.py`
- `fabricpc/graph_initialization/state_initializer.py`
- `fabricpc/graph_initialization/params_initializer.py`

Top-level:
- `fabricpc/__init__.py`