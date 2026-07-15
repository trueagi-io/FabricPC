# User Guides Update for `pre_activation` Removal from `NodeState`

## Context

The `deprecate_and_remove_pre_activation` refactor (archived in
`docs/dev_plans_archive/deprecate_and_remove_pre_activation_plan.md`) removed
`pre_activation` from `NodeState`, reducing the pytree from 6 to 5 fields:
`z_latent, z_mu, error, energy, latent_grad`. Code-side changes are complete and
tested; this plan finalizes the doc-side updates so the user guides match the
post-refactor working copy.

The working copy already has two minimal edits committed locally but unstaged:

- `docs/user_guides/06_custom_nodes.md` — removed `pre_activation=pre_activation`
  from the Conv2D `state._replace(...)` call.
- `docs/user_guides/09_experiment_tracking.md` — removed
  `extract_preactivation_statistics` from the extractor import list.

All other surviving `pre_activation` references in the user guides are
**transient local variables** inside node `forward()` examples, which is the
correct post-refactor pattern (compute the value, apply the activation, discard
it) and must be preserved.

The intended outcome is a fully consistent set of user guides where:

1. No doc refers to `pre_activation` as a `NodeState` field, attribute, or
   pytree leaf.
2. Custom-node authors migrating from the old API see an explicit migration
   note explaining the pattern change.
3. The `NodeState` mapping table in `03_how_predictive_coding_works.md` reflects
   the actual 5-field shape (currently missing `latent_grad`).

## Files to Modify

1. `docs/user_guides/06_custom_nodes.md`
2. `docs/user_guides/09_experiment_tracking.md` (verify only — no further edit
   beyond the working copy)
3. `docs/user_guides/03_how_predictive_coding_works.md`

## Changes

### 1. `06_custom_nodes.md` — add a migration note for custom-node authors

The Conv2D example (lines ~173–237) and the `MyDenseNode` `FlattenInputMixin`
example (lines ~300–316) both use `pre_activation` as a local variable. The
working-copy edit already removed it from `state._replace(...)`. Add a brief
inline clarification near the forward-method walkthrough so authors don't
mistake the local variable for a state field.

**Edit A — after the `state._replace(...)` block (around line 228), insert a
note:**

```markdown
> **Note (`pre_activation` is transient):** `pre_activation` is a local
> variable inside `forward()` — not a `NodeState` field. Earlier versions
> stored it on state; the current API does not. Compute it locally, pass it
> to the activation, and let it go out of scope. The `NodeState` fields you
> can write back via `_replace()` are `z_latent`, `z_mu`, `error`, `energy`,
> and `latent_grad`.
```

**Edit B — update the Step 5 summary list (around line 245), which says
"Update state: Replace relevant fields in the `NodeState` namedtuple", to
enumerate the actual fields:**

Change from:

```
5. **Update state**: Replace relevant fields in the `NodeState` namedtuple
```

to:

```
5. **Update state**: Replace relevant fields in the `NodeState` namedtuple
   (`z_latent`, `z_mu`, `error`, `energy`, `latent_grad` — `pre_activation`
   is *not* a state field; compute it locally and discard).
```

**No other changes to this file.** The Conv2D loop (`pre_activation = None`
accumulator), the activation call
(`z_mu = type(activation).forward(pre_activation, activation.config)`), and
the `MyDenseNode.compute_linear(...)` example are all correct post-refactor
usage and stay as-is.

### 2. `09_experiment_tracking.md` — verify only

The working-copy diff already removed `extract_preactivation_statistics` from
the imports in the "Metric Extractors" section (line ~207). Verify against
`fabricpc/utils/dashboarding/__init__.py` (lines 41–53) that no other
extractor name in the doc is stale. As of the current working copy, the doc's
import list matches the module's public exports exactly. **No further edit.**

### 3. `03_how_predictive_coding_works.md` — complete the NodeState table

The "Mapping to FabricPC" table (lines 138–149) lists four `NodeState` fields
(`z_latent`, `z_mu`, `error`, `energy`) but omits `latent_grad`. This is a
pre-existing gap, not introduced by the `pre_activation` refactor, but the
user requested a broader refresh so the 5-field `NodeState` shape is fully
documented.

**Edit — add one row to the table, between the `Energy` row and the
`Inference update` row:**

```
| Latent gradient | `NodeState.latent_grad` | Gradient accumulator `dE/dz_latent` consumed by the inference update |
```

The wording mirrors the existing rows in tone (PC concept → FabricPC type →
plain-English description) and matches the docstring on `NodeState.latent_grad`
in `fabricpc/core/types.py:127`.

## Reused Sources (for grounding the edits)

- `fabricpc/core/types.py:111–127` — authoritative `NodeState` definition
  (5 fields, with docstrings).
- `fabricpc/core/types.py:188–201` — pytree registration enumerates the
  5 leaves.
- `fabricpc/utils/dashboarding/__init__.py:41–53` — current exported extractor
  list (no `extract_preactivation_statistics`).
- `fabricpc/nodes/linear_residual.py`, `fabricpc/nodes/storkey_hopfield.py` —
  reference implementations of the "compute `pre_activation` locally, discard
  after applying the activation" pattern that the docs are describing.
- `docs/dev_plans_archive/deprecate_and_remove_pre_activation_plan.md` and
  `..._review.md` — design rationale; do not edit, just reference.

## Out of Scope

- Restoring or expanding `pre_activation` anywhere — it is permanently removed.
- The `state.custom_fields` block in `06_custom_nodes.md` (lines 367–388) is
  unrelated to this refactor and remains its own pre-existing inconsistency.
- Type-hint / API-reference doc files beyond `10_api_nodes.md` — that file
  does not document `NodeState` shape (verified via grep), so no edits needed.
- Code, examples, or tests — all already updated in the working copy and
  passing.

## Verification

After the three edits land:

1. `grep -rn "pre_activation" docs/user_guides/` — expected matches: only
   local-variable usages inside the Conv2D and `MyDenseNode` examples in
   `06_custom_nodes.md`, plus the conceptual "pre-activation values" phrase
   in `04_building_models.md:61`. **No `NodeState.pre_activation`,
   `state.pre_activation`, `_replace(pre_activation=...)`, or
   `extract_preactivation_statistics` matches anywhere.**
2. `grep -n "NodeState" docs/user_guides/03_how_predictive_coding_works.md`
   — the mapping table now contains five `NodeState.*` rows
   (`z_latent`, `z_mu`, `error`, `energy`, `latent_grad`), matching the
   pytree leaf count in `fabricpc/core/types.py:188–201`.
3. Manual read of the Conv2D example in `06_custom_nodes.md`: the migration
   note appears immediately after the `state._replace(...)` block, and Step 5
   in the summary list enumerates the five `NodeState` fields. The example
   code itself is unchanged from the working copy.
4. Open `docs/user_guides/06_custom_nodes.md` and `09_experiment_tracking.md`
   in a Markdown renderer (or `mkdocs serve` if configured) and confirm the
   note and table render cleanly.
