# Plan: Deprecate and remove `pre_activation` from `NodeState`

## Context

`NodeState.pre_activation` is the linear-combination value `z = Σ inputs·W + b`
produced during a node's forward pass — the input to the activation function
(`z_mu = f(z)`). It is currently stored as a durable field in the `NodeState`
pytree and threaded through state init, every node forward, the clamp-to-latent
reset, JAX pytree registration, and state-distribution dashboarding.

In reality `pre_activation` is a **transient forward-pass intermediate**, not
inferred state. It is recomputed every forward pass and consumed exactly once:

- The autodiff path (`Linear` and all autodiff-based nodes) does not consume it
  at all — JAX's `value_and_grad` extracts the gradient through the live
  computation graph.
- The only production reader is `LinearExplicitGrad.compute_gain_mod_error`
  (`fabricpc/nodes/linear_explicit_grad.py:154-162`), which uses it to evaluate
  `f'(z) = activation.derivative(state.pre_activation, ...)`.
  `LinearExplicitGrad` is itself a verification/teaching node, exercised only by
  `tests/test_auto_node_grad.py` against autodiff for Identity/ReLU/Tanh/Sigmoid.
- Diagnostic readers: `fabricpc/utils/dashboarding/extractors.py` (per-node stats
  and distributions) and `examples/storkey_hopfield_diagnostic.py` (tanh
  saturation metric).

Storing it in `NodeState` is therefore (a) misclassification of transient data
as durable state, (b) a permanent memory cost across all nodes/timesteps, and
(c) a hidden constraint that every node implementer must remember to populate.

**Goal:** remove `pre_activation` from `NodeState`. Migrate the single explicit-
gradient consumer to thread `pre_activation` locally; drop the diagnostic
readers in favor of `z_mu` equivalents. The activation `derivative(x, config)`
API stays exactly as it is (still takes pre-activation `x`), so all activations
including GELU continue to work unchanged.

## Approach

### 1. Migrate `LinearExplicitGrad` via a shared `_forward_with_preact` helper

`pre_activation` becomes a value threaded locally through `LinearExplicitGrad`'s
gradient methods instead of a `NodeState` field. The activation API is untouched.

**File: `fabricpc/nodes/linear.py`**

`Linear` exposes a private forward variant that returns the updated state
together with the pre-activation; the public `forward` delegates to it and
discards `pre_activation`:

```python
class Linear(FlattenInputMixin, NodeBase):
    @staticmethod
    def _forward_with_preact(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[NodeState, jnp.ndarray]:
        """Internal forward returning (state, pre_activation).

        Shared between Linear.forward (which discards pre_activation) and
        LinearExplicitGrad's gradient methods (which thread pre_activation
        into f'(z) for the gain-modulated error). Not for use by other node
        types — LinearResidual, StorkeyHopfield, and Transformer compute
        pre_activation differently and override forward directly.
        """
        ...
        return state, pre_activation

    @staticmethod
    def forward(params, inputs, state, node_info) -> NodeState:
        state, _ = Linear._forward_with_preact(params, inputs, state, node_info)
        return state
```

**File: `fabricpc/nodes/linear_explicit_grad.py`**

`compute_gain_mod_error` receives `pre_activation` explicitly instead of reading
`state.pre_activation`:

```python
@staticmethod
def compute_gain_mod_error(
    pre_activation: jnp.ndarray,
    error: jnp.ndarray,
    node_info: NodeInfo,
) -> jnp.ndarray:
    activation = node_info.activation
    f_prime = type(activation).derivative(pre_activation, activation.config)
    return error * f_prime
```

Both `forward_and_latent_grads` and `forward_and_weight_grads` call
`Linear._forward_with_preact` exactly once, obtaining state and
`pre_activation` together:

```python
state, pre_activation = Linear._forward_with_preact(
    params, inputs, state, node_info
)
gain_mod_error = node_class.compute_gain_mod_error(
    pre_activation, state.error, node_info
)
```

History: the first implementation used a two-step design — a
`Linear._compute_pre_activation` helper computing only the linear combination,
plus a separate `node_class.forward()` call inside each gradient method. The
companion review (`deprecate_and_remove_pre_activation_review.md`) flagged two
problems: the helper carried a silent Linear-only contract (`LinearResidual`
sums `:skip` edges after activation, `StorkeyHopfield` blends `x` and `x·W` by
strength, and `Transformer` has no classical pre-activation, so wrapping any of
them would produce wrong `f'(z)`), and each gradient method recomputed the
matmul that the forward call had already done. Collapsing the helper into
`_forward_with_preact` removed the duplicate computation and moved the
Linear-only contract into the helper's docstring.

### 2. Remove `pre_activation` from `NodeState` and its pytree

**File: `fabricpc/core/types.py`**
- Lines 111-130: drop the field and its docstring entry from the `NamedTuple`.
- Lines 190-204: drop `ns.pre_activation` from the pytree flatten lambda.

### 3. Remove `pre_activation` from every constructor / `_replace` site

All writes catalogued in the inventory:

- `fabricpc/graph_initialization/state_initializer.py` lines ~153, ~204, ~264
  — remove `pre_activation=jnp.zeros(shape)` from all three NodeState constructors.
- `fabricpc/nodes/base.py:400-404` — drop `pre_activation=jnp.zeros_like(state.pre_activation)`
  from the terminal-node `_replace`.
- `fabricpc/nodes/linear.py:200-219` — handled in step 1.
- `fabricpc/nodes/linear_residual.py:170-197` — drop from `_replace`.
- `fabricpc/nodes/storkey_hopfield.py:319-335` — drop from `_replace`.
- `fabricpc/nodes/skip_connection.py:106-116` — drop from `_replace`.
- `fabricpc/nodes/identity.py:146-154` — drop from `_replace`. Identity also
  computes `pre_activation = z_mu` solely to populate the field; that line goes
  too.
- `fabricpc/nodes/transformer.py:376-383` — drop from `_replace` (and remove
  the dead local at lines 492-493 if it isn't used elsewhere).
- `examples/custom_node.py:163-190` — drop from `_replace`. Keep the local
  `pre_activation` variable since the example's forward still needs it to
  compute `z_mu`; just don't store it.
- `examples/resnet18_cifar10_demo.py:193` — drop from `_replace`.
- `examples/jpc_fc_resnet_compare.py:266, 361, 440` — drop from `_replace`.

### 4. Update tests

- `tests/test_fabricpc.py:308-311, 482-485` — drop `pre_activation=...` from
  NodeState construction. Tests don't read the field; construction is the only
  reference.
- `tests/test_auto_node_grad.py:137-140, 244-247` — drop `pre_activation=...`
  from NodeState construction. Test logic asserts gradient equality between
  `LinearExplicitGrad` and autodiff `Linear`; with step 1 in place,
  `LinearExplicitGrad` no longer needs `state.pre_activation`.

### 5. Drop dashboarding pre-activation diagnostics

**File: `fabricpc/utils/dashboarding/extractors.py`**

- Lines 76-99: delete `extract_preactivation_statistics` entirely.
  `extract_activation_statistics` (`z_mu` stats, lines 102-125) and
  `extract_latent_statistics` (`z_latent` stats) remain.
- Lines 252-307: in `extract_all_distributions`, remove `"pre_activation"` from
  the output dict (line 280) and the corresponding flatten call (lines 292-294).
  Update the docstring on line 269.

Verify with `grep -rn "extract_preactivation_statistics\|distributions\[.pre_activation.\]" .`
that no training script / Aim integration depends on either of these. If a
caller exists, remove that call site in the same change.

### 6. Update the Storkey Hopfield diagnostic

**File: `examples/storkey_hopfield_diagnostic.py:578-581`**

Replace
```python
"tanh_saturation_frac": float(jnp.mean(jnp.abs(hop_state.pre_activation) > 2.0)),
"pre_act_mean_abs": float(jnp.mean(jnp.abs(hop_state.pre_activation))),
```
with the post-activation equivalent:
```python
"tanh_saturation_frac": float(jnp.mean(jnp.abs(hop_state.z_mu) > jnp.tanh(2.0))),
"act_mean_abs": float(jnp.mean(jnp.abs(hop_state.z_mu))),
```
`|tanh(z)| > tanh(2) ≈ 0.964` is exactly equivalent to `|z| > 2.0` since `tanh`
is monotonic. The mean-abs metric loses some information (saturation compresses
magnitude), so the second key is renamed accordingly.

### 7. Update user-facing docs

**File: `docs/user_guides/06_custom_nodes.md`**
- Update the example custom node (around lines 195-227 and 311-314) to drop
  `pre_activation` from the `_replace` call. The local computation of
  `pre_activation` inside the forward function stays (it's how `z_mu` is
  built); only the storage step goes.
- Add a note after the example's `_replace` block stating that
  `pre_activation` is a transient local variable inside `forward()`, not a
  `NodeState` field, and enumerating the writable fields
  (`z_latent`, `z_mu`, `error`, `energy`, `latent_grad`). The surviving
  `pre_activation` references in the guide are local variables in the Conv2D
  and `MyDenseNode` examples — the correct post-refactor pattern (compute,
  apply the activation, discard) — and stay as-is.

**File: `docs/user_guides/09_experiment_tracking.md`** — remove
`extract_preactivation_statistics` from the extractor import list, matching
the exports in `fabricpc/utils/dashboarding/__init__.py`.

**File: `docs/user_guides/03_how_predictive_coding_works.md`** — add the
missing `latent_grad` row to the NodeState mapping table. A pre-existing gap,
fixed in the same pass so the table lists all five fields of the
post-refactor pytree.

Doc-pass verification: `grep -rn "pre_activation" docs/user_guides/` must
match only local-variable usages inside `forward()` examples — no
`NodeState.pre_activation`, `state.pre_activation`,
`_replace(pre_activation=...)`, or `extract_preactivation_statistics`
anywhere.

Archive docs under `docs/dev_plans_archive/` are historical and need not be
touched.

## Critical files (modified)

- `fabricpc/core/types.py` — remove field + pytree entry
- `fabricpc/nodes/linear.py` — factor `_forward_with_preact` out of `forward`,
  stop storing `pre_activation` in state
- `fabricpc/nodes/linear_explicit_grad.py` — `compute_gain_mod_error` takes
  `pre_activation` explicitly; both gradient methods obtain it from
  `_forward_with_preact`
- `fabricpc/nodes/{linear_residual, storkey_hopfield, skip_connection, identity, transformer}.py`
  — drop `pre_activation` from `_replace`
- `fabricpc/nodes/base.py` — drop from terminal-node reset
- `fabricpc/graph_initialization/state_initializer.py` — drop from 3 init sites
- `fabricpc/utils/dashboarding/extractors.py` — drop preact stats + distribution
- `tests/test_fabricpc.py`, `tests/test_auto_node_grad.py` — drop from constructions
- `examples/custom_node.py`, `examples/resnet18_cifar10_demo.py`,
  `examples/jpc_fc_resnet_compare.py`, `examples/storkey_hopfield_diagnostic.py`
- `examples/scaling/scaling_analysis_plots.py` — state-memory coefficient 5 → 4
  (found in review)
- `docs/user_guides/06_custom_nodes.md`

Activations (`fabricpc/core/activations.py`) are intentionally unchanged.

## Verification

End-to-end checks, in order:

1. **Type/import sanity:**
   `python -c "from fabricpc.core.types import NodeState; print(NodeState._fields)"`
   — output must not contain `pre_activation`.
2. **Unit tests:** `pytest tests/test_fabricpc.py tests/test_state_initializer.py -q` — all green.
3. **Explicit-grad equivalence (the key regression):**
   `pytest tests/test_auto_node_grad.py -v`. This is the canonical test for
   step 1: `LinearExplicitGrad.compute_gain_mod_error` must keep producing the
   same gradient as autodiff `Linear` after switching to a locally-computed
   `pre_activation`. Failure here means the local recomputation diverges from
   what `Linear.forward` computes (e.g. `flatten_input` flag mishandled, bias
   absent edge case).
4. **GELU autodiff path still works:** smoke-run a graph with `GeluActivation`
   via the standard `Linear` (e.g. start `examples/transformer_demo.py` for a
   few steps). Should be unaffected since autodiff doesn't call `derivative()`.
5. **End-to-end example smoke:** `python examples/resnet18_cifar10_demo.py` for
   a few iterations — exercises node forward, state init, and dashboarding
   together.
6. **Storkey Hopfield diagnostic:** `python examples/storkey_hopfield_diagnostic.py`
   — confirms the renamed/rewritten metrics still compute and produce the
   expected numerical equivalence on a matched run.
7. **Dashboard wiring:** `grep -rn "extract_preactivation_statistics\|\"pre_activation\""` in the
   repo after edits — no remaining call sites or key references.


## Results

  Step 1 — LinearExplicitGrad migration (fabricpc/nodes/linear.py, linear_explicit_grad.py):
  - Linear exposes `_forward_with_preact(params, inputs, state, node_info) -> (state, pre_activation)`; the public Linear.forward delegates to it and discards pre_activation.
  - LinearExplicitGrad.compute_gain_mod_error now takes pre_activation, error, node_info directly.
  - Both forward_and_latent_grads / forward_and_weight_grads call `_forward_with_preact` exactly once, obtaining state and pre_activation together — no recomputation, no separate forward call.

  Step 2 — Drop the field (fabricpc/core/types.py): removed pre_activation from the NodeState NamedTuple and from the pytree flatten lambda.

  Step 3 — Drop write sites: state_initializer.py (3 sites), base.py terminal-node reset, linear_residual.py, storkey_hopfield.py, skip_connection.py, identity.py, transformer.py (both _replace and the inlined _mha local), and examples/jpc_fc_resnet_compare.py (3 sites).
  Of the other examples listed in the plan, none needed migration on the final base: main had deleted examples/custom_node.py (PR #19) and examples/resnet18_cifar10_demo.py no longer references pre_activation. A rebase re-added custom_node.py as an empty file by accident; it has been removed from the branch.

  Step 4 — Tests: dropped pre_activation=... from NodeState constructions in test_fabricpc.py and test_auto_node_grad.py.

  Step 5 — Dashboarding: deleted extract_preactivation_statistics, removed the "pre_activation" key from extract_all_distributions, and updated __init__.py exports.

  Step 6 — Storkey Hopfield diagnostic: replaced pre_activation-based tanh_saturation_frac / pre_act_mean_abs with z_mu-based equivalents (renamed to act_mean_abs); updated the print line at l.696.

  Step 7 — Docs: dropped pre_activation from the _replace call in 06_custom_nodes.md and added the transient-pre_activation note enumerating the writable NodeState fields; removed extract_preactivation_statistics from the 09_experiment_tracking.md import block; added the latent_grad row to the NodeState mapping table in 03_how_predictive_coding_works.md and the latent_grad field to the node-state list in 04_building_models.md; updated the 5→4 tensor count in examples/scaling/scaling_analysis_plots.py (line 792 commentary block). Doc-pass grep: pre_activation appears in docs/user_guides/ only as a local variable inside forward() examples.

  Verification:
  - NodeState._fields → ('z_latent', 'z_mu', 'error', 'energy', 'latent_grad') ✓
  - pytest tests/test_auto_node_grad.py -v → 9/9 passed (gradient equivalence intact across Identity/ReLU/Tanh/Sigmoid).
  - pytest tests/ → 127/127 passed.
  - grep extract_preactivation_statistics|"pre_activation" → no remaining references.

## Review follow-ups applied

From `deprecate_and_remove_pre_activation_review.md`:

- **Coefficient bug fixed:** `examples/scaling/scaling_analysis_plots.py:905`
  still computed PC state memory as `5 * num_nodes * batch_size * w * 4` bytes,
  inconsistent with the file's own 4-tensor commentary (z_latent, z_mu, error,
  latent_grad). The leftover 5 overestimated PC state memory by 25% and
  distorted the PC/BP memory-ratio analysis. Corrected to 4.
- **`_compute_pre_activation` replaced by `_forward_with_preact`:** the review
  flagged the original helper's silent Linear-only contract and the duplicate
  matmul (helper + separate forward call in each gradient method). The helper
  was collapsed into the private forward variant described in step 1; its
  docstring states the contract (shared with LinearExplicitGrad; not for
  LinearResidual, StorkeyHopfield, or Transformer).
- Verification re-run after both follow-ups: pytest tests/test_auto_node_grad.py
  → 9/9, pytest tests/ → 127/127.

## Later update

The per-sample energy summation refactor
(`per_sample_energy_summation_refactor_plan.md`) removed the batch-summed
energy from the node `forward()` return contract. `_forward_with_preact`
accordingly returns `(state, pre_activation)` rather than
`(energy, state, pre_activation)`, and LinearExplicitGrad unpacks the 2-tuple.
The step 1 code above reflects this final signature.