# Complete the `forward()` contract change: NodeState-only return, batch summation owned by gradient methods

## Context

The node `forward()` contract changed: it no longer returns the batch-summed scalar energy, only the updated `NodeState`. The scalar was only ever needed as the differentiable output for `jax.value_and_grad`, so the batch summation moves to the autodiff callsites — `forward_and_latent_grads()` and, by the same necessity, `forward_and_weight_grads()`.

The working tree already migrated the `forward()` bodies of Linear, IdentityNode, LinearResidual, SkipConnection, StorkeyHopfield, and TransformerBlock (they `return state`). What remains: the abstract contract in `base.py` (annotation + docstring), the three internal callers that still unpack two values, the `transformer_v2.py` nodes that still return `(jnp.sum(state.energy), state)`, `LinearExplicitGrad` which unpacks a 3-tuple from the now-2-tuple `_forward_with_preact`, two tests, and the documentation. There is also a latent bug: the `energy_fn` closure in `forward_and_latent_grads` sums `new_state.energy` (undefined at that point) instead of `new_s.energy` (`fabricpc/nodes/base.py:440`).

Scope note: the request names `forward_and_latent_grads()` as the owner of batch summation, but `forward_and_weight_grads()` (base and the TransformerBlock override) also calls `jax.value_and_grad` directly on `forward` and therefore also needs a closure that sums `state.energy` to a scalar. Both gradient methods get the same treatment; there is no alternative that keeps autodiff working.

## New contract (to be stated in `base.py` docstrings and `06_custom_nodes.md`)

**`forward(params, inputs, state, node_info) -> NodeState`**
- Computes `z_mu`, `error = z_latent - z_mu`, and per-sample energy via `energy_functional` (shape `(batch,)` in `state.energy`).
- Returns only the updated `NodeState`. No scalar energy.

**`forward_and_latent_grads(params, inputs, state, node_info, is_clamped) -> (NodeState, input_grads, self_grad)`** — explicit responsibilities:
1. Handles in-degree-0 nodes specially: no `forward()` call; `z_mu <- z_latent` (cast to `z_mu` dtype), zero error/energy/gradients.
2. Calls `node_class.forward()` for every node with in-degree > 0. (For unclamped out-degree-0 nodes this yields `z_mu` only; gradients are zero.)
3. Sums `state.energy` over the batch dimension to produce the scalar differentiated by autodiff.
4. Calls `jax.value_and_grad` on that scalar w.r.t. the input tensors and `z_latent`.

**`forward_and_weight_grads(params, inputs, state, node_info) -> (NodeState, params_grad)`**
- Same summation ownership: wraps `forward()` in a closure returning `(jnp.sum(state.energy), state)` and differentiates w.r.t. `params`.

## Changes

### 1. `fabricpc/nodes/base.py`

- **Abstract `forward()` (lines 293–313):** annotation `-> tuple[jax.Array, NodeState]` → `-> NodeState`. Rewrite docstring: forward computes `z_mu`, `error`, and per-sample energy (`state.energy`, shape `(batch,)`) and returns the updated `NodeState`; batch summation to a scalar is owned by `forward_and_latent_grads()`/`forward_and_weight_grads()`.
- **`forward_and_latent_grads` (lines 348–449):**
  - Line 416 (unclamped out-degree-0 branch): `total_energy, new_state = node_class.forward(...)` → `new_state = node_class.forward(...)`.
  - `energy_fn` closure (lines 434–443): fix `total_energy = jnp.sum(new_state.energy)` → `jnp.sum(new_s.energy)`; delete the TODO comment (the change it requested is now done).
  - Docstring: add the four-point contract above (in-degree-0 special case, forward call for in-degree > 0, batch summation, `jax.value_and_grad`), keeping the existing muPC-scaling and override guidance.
- **`forward_and_weight_grads` (lines 451–486):** replace `jax.value_and_grad(node_class.forward, argnums=0, has_aux=True)(params, ...)` with a closure over `params`:
  ```python
  def energy_fn(p):
      new_s = node_class.forward(p, inputs, state, node_info)
      return jnp.sum(new_s.energy), new_s

  (total_energy, new_state), params_grad = jax.value_and_grad(
      energy_fn, has_aux=True
  )(params)
  ```
  Docstring: note that this method sums `state.energy` over the batch to obtain the autodiff scalar.

### 2. `fabricpc/nodes/linear.py`

- `_forward_with_preact` (line 95–96): annotation → `tuple[NodeState, jnp.ndarray]`; docstring "(energy, state, pre_activation)" → "(state, pre_activation)".
- `forward` (lines 207–213): annotation → `NodeState`; drop "returning energy scalar and updated state" and the "-> total energy" step from the docstring.

### 3. Stale annotations/docstrings in other node files

Change `-> Tuple[jax.Array, NodeState]` / `-> tuple[jax.Array, NodeState]` to `-> NodeState` and fix any docstring `Returns` mention of the tuple:
- `fabricpc/nodes/identity.py:118`
- `fabricpc/nodes/linear_residual.py:156`
- `fabricpc/nodes/skip_connection.py:104`
- `fabricpc/nodes/storkey_hopfield.py:284`
- `fabricpc/nodes/transformer.py:289`
- `examples/custom_node.py:149`

### 4. `fabricpc/nodes/transformer.py` — `forward_and_weight_grads` override (line 399)

Same closure fix as base (the LayerNorm-compensation post-processing stays unchanged).

### 5. `fabricpc/nodes/transformer_v2.py` — deferred at first, applied in the branch's final commit

Deliberately skipped in the initial refactor commits to avoid a merge conflict
with a concurrent branch that touched this file, then applied in the branch's
final commit once that conflict was resolved (see Outcome).

Edits:
- `EmbeddingNode.forward` (lines 97, 113): annotation → `NodeState`; `return jnp.sum(state.energy), state` → `return state`.
- `EmbeddingNode.forward_and_latent_grads` (line 117): `_, new_state = node_info.node_class.forward(...)` → `new_state = node_info.node_class.forward(...)`.
- `MhaResidualNode.forward` (line 248), `LnMlp1Node.forward` (line 312), `Mlp2ResidualNode.forward` (line 372), `VocabProjectionNode.forward` (line 429): `return jnp.sum(state.energy), state` → `return state`.

### 6. `fabricpc/nodes/linear_explicit_grad.py`

Lines 67 and 128: `_, state, pre_activation = Linear._forward_with_preact(...)` → `state, pre_activation = Linear._forward_with_preact(...)`.

### 7. `fabricpc/graph_initialization/state_initializer.py:282`

`_, projected = node_class.forward(...)` → `projected = node_class.forward(...)`.

### 8. Tests

- `tests/test_fabricpc.py:508`: `_, new_state = IdentityNode.forward(...)` → `new_state = IdentityNode.forward(...)`.
- `tests/test_transformer_nodes.py:217`: `_, new_state = EmbeddingNode.forward(...)` → `new_state = EmbeddingNode.forward(...)`.

### 9. Documentation — `docs/user_guides/06_custom_nodes.md`

- Line 186 (Step 5 example docstring): `Returns: (total_energy, updated_state)` → `Returns: updated NodeState`.
- Line 246 (Step 5 numbered list): `7. **Return**: \`(total_energy, updated_state)\`` → return the updated `NodeState`; add that `state.energy` stays per-sample (shape `(batch,)`) — batch summation is owned by `forward_and_latent_grads()`/`forward_and_weight_grads()`.
- "Explicit Gradients" section (lines 322–340): the example signatures and returns are wrong. Replace with the real contracts: `forward_and_latent_grads(params, inputs, state, node_info, is_clamped) -> (NodeState, input_grads, self_grad)` and `forward_and_weight_grads(params, inputs, state, node_info) -> (NodeState, params_grad)`; drop the nonexistent `scaling_factors` parameter and the `(total_energy, updated_state, custom_grads)` return. Keep the `LinearExplicitGrad` pointer (it exists at `fabricpc/nodes/linear_explicit_grad.py`).
- Add a short subsection stating the explicit `forward_and_latent_grads()` contract (the four responsibilities under "New contract" above), so custom-node authors overriding it know what the base implementation guarantees.

No changes to `docs/user_guides/10_api_nodes.md` (constructor-level reference only; no method contracts documented there) or to `docs/dev_plans_archive/` (historical records).

## Verification

1. `python -m pytest tests/` — full suite; `test_fabricpc.py`, `test_auto_node_grad.py` (exercises `LinearExplicitGrad` against autodiff `Linear`), and `test_transformer_nodes.py` cover both gradient paths and the migrated call sites.
2. `grep -rn "total_energy, \|tuple\[jax.Array, NodeState\]\|Tuple\[jax.Array, NodeState\]" fabricpc/ tests/ examples/` — must return nothing (no residual two-value unpacking of `forward()` or stale annotations).
3. Run one training example end-to-end to confirm the inference and learning loops work through `state_initializer`, `forward_and_latent_grads`, and `forward_and_weight_grads`.

## Outcome

All sections implemented on branch clinfra2. Sections 1–4 and 6–9 landed in
the initial refactor commits; section 5 (transformer_v2) was deferred to avoid
a merge conflict with a concurrent branch and applied in the branch's final
commit once that conflict was resolved.

While section 5 was pending, 9 of 11 tests in `tests/test_transformer_nodes.py`
failed (TestEmbeddingNode ×4, TestTransformerBlock ×4,
TestEvaluateTransformer::test_smoke) — every test that builds a graph
containing transformer_v2 nodes. Mechanism: the transformer_v2 `forward()`
methods still returned the old `(scalar_energy, NodeState)` tuple, so the
state initializer failed at `projected.z_mu` (`state_initializer.py:290`,
`AttributeError: 'tuple' object has no attribute 'z_mu'`) and the base
gradient closures failed at `new_s.energy`. The failures were left visible on
purpose — no xfail markers — as the signal that the section 5 migration was
pending.

Final verification with all sections applied:
- `pytest tests/` → 269/269 passed.
- Stale-pattern grep → clean; the only remaining `total_energy` occurrences are
  the intended `energy_fn` closures inside the two gradient methods.
- Pre-deferral smoke run on the plan's original base: 3 epochs conv MNIST on
  GPU via the (since-removed) `examples/custom_node.py`, energy 1.16 → ~0.001,
  test accuracy 98.74%.
