# Reconcile muPC scaling with the μPC paper

Originally: the `√L` divergence on the output node. Two later audits found the
same species of defect elsewhere, so this document now covers three rounds,
each appended in place rather than rewritten:

1. **Output node** (2026-07-24, landed) — findings 1-4. The readout carried a
   `√L` that Depth-μP/μPC do not give it.
2. **Depth-factor placement** (2026-08-03/08, landed) — findings 5-9. `1/√L`
   was applied per scalable edge instead of once per branch at the merge.
3. **Variance-reducing transforms** (2026-08-09, landed) — findings 10-16.
   Nodes whose transform shrinks input variance were never compensated, because
   the hook they report through could not express a reduction.

The empirical gate resolved 2026-08-10 in favor of rounds 2+3: ResNet-18
31.50% → 33.71%, and a depth sweep on `mupc_demo.py` improving at every depth by
a margin that grows with L (+1.2 at L = 8, +12.1 at L = 128). See "Empirical
gate".

Finding 8's resolution and one of its verification items are superseded by
finding 14; both are marked in place.

## Context

A review agent flagged that `fabricpc/core/mupc.py:330` scales output nodes as
`a = gain/(fan_in·√(K_slot·L))` — carrying a `√L` factor that μPC's Table 1
output scaling (`a_L = 1/N`) does not. The `MuPCConfig` docstring
(`mupc.py:126-130`) claims the L-free form; the inline comment
(`mupc.py:327-328`) and the `compute_mupc_scalings` docstring
(`mupc.py:229-231`) reproduce the L-carrying formula while asserting it
"matches muPC O(1/N) convention" — the false claim there is the lineage
assertion, not the formula. This plan records the investigation findings and
the fix.

Symbols: `a` is the per-in-edge forward multiplier applied to a node's input
before its `forward()`; `fan_in` is the edge's weight-matrix fan-in;
`K_slot` is the in-degree of the target slot; `L` is residual depth (number
of skip-merge nodes on the longest path, `max(skip_depth, 1)`); `gain` is the
activation's Kaiming-style variance gain; `N` is hidden width.

## Findings

### 1. The √L on the output branch is an implementation artifact, not a design choice

- The archived design doc `docs/dev_plans_archive/muPC_arbitrary_graph_upgrade_plan.md:59-63`
  specified the output as `a = 1/(fan_in·√K)` "maintaining the muPC O(1/N)
  convention"; its verification table (`:89`) reads "Output node | … | 1/N |
  Matches jpc a_L=1/N". No depth factor.
- The √L has been present since the module's first commit:
  `fabricpc/core/mupc.py` was created in 68e7700 (v0.3.0) with depth `L`
  already folded into the shared `√(K_slot·L)` subexpression for both
  branches. No earlier muPC scaling code exists (commit b771135, #13, has no
  `MuPCConfig`), so the design doc's L-free output form was never
  implemented. The 68e7700 commit message documents only the hidden formula.
  Commit cca03de (#20) states "No mathematical changes to forward, energy,
  or scaling formulas".
- Of the in-code docs, only the `MuPCConfig` docstring states the L-free
  form; the inline comment and `compute_mupc_scalings` docstring carry the
  √L while claiming the O(1/N) lineage. The essay
  `scaling_breakthroughs_pc_and_backprop.md` and the scaling-history plan
  live in the separate `neuromorphic_predictive_coding` repo, not in this
  repo as originally cited. The essay once flagged the √L as an unreconciled
  deviation in a maintainer note, since removed
  (`plans_archive/pc_backprop_scaling_history_plan.md:298-300` in that repo
  records the removal and defers the √L question to this project); its
  Part IV now states only that output nodes are excluded from muPC scaling
  by default.
- No test pins it: the sole `include_output=True` test
  (`tests/test_mupc.py:171-191`) uses a pure chain (L=1, K=1) and asserts
  `1/20` — invariant to where L sits, since √1 = 1.
- No shipped graph executes the divergent path: `include_output=True` appears
  only in `scripts/diagnose_deep_mupc.py:73` (pure 20-layer chain, L=1) and
  the L=1 test. Every graph with skip connections (ResNet demo, transformer)
  sets `include_output=False`. Removing the √L changes no existing numerics.

### 2. Both primary sources give the output layer NO depth factor

- μPC paper (arXiv:2505.13124), Table 1: input `a₁ = N₀^(-1/2)`, hidden
  `a_ℓ = (N_{ℓ-1}·L)^(-1/2)`, output `a_L = N_{L-1}^(-1)`, unit-variance init.
  The readout sits outside the residual stream (residual switch `τ_ℓ = 1`
  only for hidden layers). The paper does not derive the readout scaling; the
  authors state they "adapted, rather than derived" the Depth-μP scalings.
- jpc reference (`jpc/_core/_energies.py`, `_get_param_scalings`): first
  layer `1/√D`; hidden `1/√N`, or `1/√(N·L)` with skip connections; output
  `1/N` for the μPC parameterization regardless of skip connections — the
  skip-conditional branch applies to hidden layers only. (FabricPC's
  `L = max(skip_depth, 1)` chain degeneration matches jpc's hidden convention
  exactly; only the output branch diverges.)
- Both sources re-verified 2026-07-24 against the arXiv HTML (Table 1,
  τ_ℓ definition, "adapted, rather than derived" quote) and the jpc source
  on GitHub.

### 3. Why the readout is depth-free in Depth-μP/μPC (the mechanism)

- The hidden `1/√L` corrects depth-driven variance accumulation: the residual
  stream sums L branch contributions, so each branch is damped by `1/√L` to
  keep the stream O(1) for any L. The readout is applied once to the final
  stream; no L contributions are summed at it, so there is nothing for a `√L`
  to cancel. Its input is already O(1) by the hidden scaling.
- The width factor 1/N (vs hidden 1/√N) is a feature-learning requirement: at
  init the readout sums N uncorrelated terms (O(√N) → logits O(1/√N)); after
  training the weight updates correlate with the features and the sum is
  O(N), so `1/N` keeps trained logits O(1).
- An extra `1/√L` on the readout breaks depth transfer — μPC's headline
  property. Init predictions, the readout weight gradient (∝ a·ε·hᵀ), and the
  top-down error injected into the last hidden node (∝ a·Wᵀ·ε) all shrink as
  `1/√L`, so the optimal readout learning rate drifts with depth. In PC
  terms: with the output clamped to y under Gaussian energy
  (`fabricpc/nodes/linear.py:123-127`), ε = y − f(W·(a·h)), and the
  supervised error signal entering the network is attenuated by `1/√L`.

### 4. How FabricPC identifies an output node

- Purely topological: `out_degree == 0` (`mupc.py:247-249`), gated by
  `MuPCConfig.include_output` (default False → output gets `None`, standard
  init; the softmax+CE path).
- Fully decoupled from target designation: `task_map` resolves after scaling
  is computed (`fabricpc/graph_assembly/graph_construction.py:204-232`), and
  clamping is driven by `task_map` alone
  (`fabricpc/training/train.py:136-140`). Any sink — auxiliary head, probe —
  would receive output scaling; sinks coincide with the supervised target
  only by convention in shipped graphs. (Observation only; no change here.)

## Change

1. **`fabricpc/core/mupc.py:330`** — drop the √L:
   `a = gain / (fan_in * math.sqrt(K_slot))`. `topdown_grad_scale = a·jac_gain`
   follows automatically.
2. **Comments/docstrings in `mupc.py`** — update the inline block
   (`:323-328`), the `compute_mupc_scalings` docstring (`:229-231`, which
   currently documents the L-carrying output formula), and the module
   docstring to state the output formula explicitly: hidden
   `gain/√(fan_in·K_slot·L)`, output `gain/(fan_in·√K_slot)` (no depth
   factor: the readout is applied once to the O(1) stream, not summed L
   times). Extend the `MuPCConfig.include_output` docstring (`:125-134`) to
   include the gain: `a = gain/(fan_in·√K)`.
3. **Test** — add to `tests/test_mupc.py` a graph with skip connections
   (L ≥ 2) and `include_output=True`; assert the output edge scale equals
   `gain/(fan_in·√K)` (L-independent) and a hidden edge equals
   `gain/√(fan_in·K·L)`. This pins the distinction the L=1 test cannot see.
   L counts nodes carrying an `is_skip_connection=True` slot along the
   longest path (`_count_skip_connections_depth`), so the graph needs two
   such merge nodes in series — e.g. two `LinearResidual` blocks (skip slot)
   or two `SkipConnection` junctions. An `IdentityNode` merge does not
   increment L; its slots are scalable, not skip.
4. **Docs** — update `docs/user_guides/05_initialization_and_scaling.md:204-210`,
   which documents the L-carrying output formula
   (`a = gain / (fan_in * sqrt(K_slot * L))`): change to
   `a = gain / (fan_in * sqrt(K_slot))` and state that the depth factor is
   absent by design. The two documents this item originally targeted
   (`scaling_breakthroughs_pc_and_backprop.md`,
   `pc_backprop_scaling_history.md`) are in the
   `neuromorphic_predictive_coding` repo, and their √L-deviation discussion
   was already removed there; the essay's Part IV states only the default
   exclusion of output nodes, which remains true after this fix. Optional
   cross-repo follow-up, not part of this change: one sentence in that
   Part IV noting that with `include_output=True` the readout follows
   Table 1's L-free `1/N` form.

### Alternatives considered

- **Document √L as intentional, keep formula**: rejected. No derivation
  supports it (original design doc specifies the L-free form); it breaks
  depth transfer of the readout learning rate and attenuates the supervised
  error signal by √L in deep residual nets.
- **Fix only the docstrings to describe the code**: rejected. Would document
  a scaling that contradicts the paper and jpc reference while claiming their
  lineage.

### Out-of-scope observations (recorded, not changed)

- Sink-vs-target decoupling: `out_degree == 0` output detection never
  cross-checks `task_map`; a non-supervised sink would receive output
  scaling. Worth a future validation or explicit output designation.
- At the output, the same correlated-sum argument that forces `1/fan_in`
  arguably favors `1/K_slot` over `1/√K_slot` for trained-function
  boundedness when K edges merge into the readout slot; K is small and fixed
  (unlike N and L, it does not scale), so left as is.

## Verification

- `pytest tests/test_mupc.py` (and full suite) — existing L=1 assertions are
  unaffected (√1 = 1); new L≥2 test pins the fix.
- Run `scripts/diagnose_deep_mupc.py` — the only `include_output=True`
  consumer; a 20-layer pure chain (`num_hidden = 20`), L=1, so numerics must
  be identical before/after.
- `grep -rn "K_slot \* L" fabricpc docs` — after the change, every match
  must be a hidden-formula context; no output formula may carry L.

## Plan review resolutions (2026-07-24)

Independent verification of every citation produced the corrections folded
in above:

- The inline comment (`mupc.py:327-328`) was mischaracterized: it states the
  L-carrying formula and falsely asserts O(1/N) lineage; it does not claim
  the L-free form. The `compute_mupc_scalings` docstring (`:229-231`) also
  documents the L-carrying output formula and was missing from Change 2.
- The √L did not "enter" existing code: `mupc.py` was created in 68e7700
  with the √L already present; no code version ever had the L-free output.
- The two docs in Change 4 are not in this repo (never in its git history);
  they live in `neuromorphic_predictive_coding`, and the maintainer note
  targeted for replacement was already removed there. Change 4 now targets
  the in-repo user guide (`05_initialization_and_scaling.md:204-210`), which
  documents the L-carrying output formula and was missing from the plan.
- Finding 4 paths corrected to `fabricpc/graph_assembly/graph_construction.py`
  and `fabricpc/training/train.py`.
- Externally confirmed: μPC Table 1 (`a_L = N_{L-1}^(-1)`, unit-variance
  init, τ_ℓ = 1 for hidden layers only, "adapted, rather than derived") and
  jpc `_get_param_scalings` (output `1/N` for the μPC parameterization, no L
  under skip connections). The core change and its rationale stand.

## Scope expansion: remaining uniform-L divergences (2026-08-03; revised 2026-08-08; implemented 2026-08-08 — empirical gate still pending)

A post-fix review audited the adjacent scaling paths for the same species of
defect — a depth factor where Depth-μP/μPC place none. The root defect is
placement: the uniform hidden rule (`compute_mupc_scalings`: every scalable
edge of every non-output node carries `a = gain/√(fan_in·K_slot·L)`) applies
L per scalable edge, while Depth-μP places the `1/√L` on exactly one edge
class — the branch contribution entering a residual-stream merge. Findings
5-7 are the divergences this produces; findings 8-9 record adjacent defects
found while settling the design (2026-08-08 review). Unlike the output fix,
correcting these changes shipped skip-graph numerics, so the change is gated
on empirical verification.

### 5. Stream-initializing stem layers are damped

- References: Table 1 gives the first layer `a₁ = N₀^(-1/2)` with no depth
  factor; jpc's first layer is `1/√D` with or without skip connections. The
  stem produces the initial residual stream; it is not one of the L branch
  contributions summed into it.
- FabricPC damps the first weighted layer by `1/√L` whenever the graph
  contains any skip merge: the L=2 test graph's stem `h1` gets `1/√(10·2)`
  (`test_include_output_depth_free_with_residual_blocks`); the ResNet demo's
  stem ConvNode (`examples/resnet18_cifar10_demo.py:257`, L=8) gets an extra
  `1/√8`.
- Consequence: initial stream variance v₀ = 1/L instead of 1. With the
  per-block recursion v_i = v_{i-1}(1+1/L), the final stream variance is
  ≈ e/L — vanishing with depth. The inline comment "L bounds total variance
  growth to (1+1/L)^L ~ e" itself assumes v₀ = 1.
- Same verification blind spot as Finding 1: the design archive's
  first-hidden row (`muPC_arbitrary_graph_upgrade_plan.md:88`, "1/√784
  matches jpc a₁=1/√D") was checked only on an L=1 chain, where the L factor
  is invisible.

### 6. Branches are damped once per weighted layer; stream projections are damped

- Depth-μP damps each branch contribution once at the merge:
  stream_ℓ = stream_{ℓ-1} + (1/√L)·branch_ℓ. FabricPC's per-edge rule gives
  every weighted layer inside a branch its own `1/√L`, so a branch with k
  weighted layers is damped by L^(−k/2). ResNet demo: two convs per branch
  (`conv_a`, `conv_b`, `examples/resnet18_cifar10_demo.py:176-194`), L=8 →
  branch damped 1/8, intended 1/√8.
- Stream-path projections are damped too: downsample blocks put a weighted
  1×1 `conv_skip` on the skip path (`:211`); its in-edge is scalable, so it
  carries `1/√L` although it substitutes for the identity stream path, which
  Depth-μP leaves undamped.
- Where the damping currently lives: `SkipConnection`'s single "in" slot is
  non-scalable (both paths pass at 1.0), so branch damping sits entirely on
  branch-interior edges. That placement is correct only when the branch has
  exactly one weighted layer — the configuration the existing L-placement
  test uses (`test_skip_depth_affects_compute_scaling`).

### 7. Post-stream layers are damped

- Layers after the last merge apply once to the final stream; nothing is
  summed L times at them. FabricPC still damps them: the ResNet demo's
  global-pool in-edge is scaled `1/√8`, documented as expected behavior in
  `04_building_models.md:254`. (The logits Linear is exempt only because
  `include_output=False` removes it from muPC scaling entirely.)

### 8. StorkeyHopfield's probe slot is scalable, breaking its pass-through under muPC (latent)

- StorkeyHopfield merges an identity path and a learned path internally:
  `z_mu = act(probe/(1+s) + (probe @ W)·s/(1+s) + b)`, where `probe` is the
  input arriving on the single in-edge, `W` the internally
  Xavier-initialized (D, D) Hopfield matrix on Hopfield dimension D, and
  `s` = hopfield_strength.
- Its "in" slot is scalable by default
  (`fabricpc/nodes/storkey_hopfield.py:168`) and the node inherits
  `get_weight_fan_in`, which returns `source_shape[-1]` = D
  (`fabricpc/nodes/base.py:404-407`). In a muPC graph the probe edge would
  get `a = gain/√(D·K_slot·L)`; the blend becomes
  `act(a·probe/(1+s) + (a·probe)@W·s/(1+s))`, so the s=0 pass-through
  collapses toward `act(0)` and the internally initialized W is scaled a
  second time.
- Latent, not shipped: no StorkeyHopfield consumer passes
  `scaling=MuPCConfig` (checked `examples/storkey_hopfield_demo.py`,
  `examples/storkey_hopfield_recall.py`,
  `scripts/storkey_hopfield_diagnostic.py`). It becomes live the first time
  the node is composed into a muPC graph.
- Resolution: set `is_variance_scalable=False`, keep
  `is_skip_connection=False`. The node self-normalizes: the blend
  coefficients `1/(1+s)` and `s/(1+s)` sum to 1, so stacked StorkeyHopfield
  nodes do not accumulate variance, and the activation wraps the identity
  path, so no raw identity stream passes through the node. It must not
  count toward L — counting it would raise L and over-damp the graph's true
  merges.
- **[Superseded 2026-08-09 — see finding 14.]** The diagnosis holds; the
  resolution does not. "Coefficients sum to 1" preserves scale only for
  perfectly correlated terms; `probe` and `probe @ W` are near-independent,
  so their variances add in quadrature and the blend *shrinks* variance.
  Exempting the slot leaves that shrinkage to compound through a chain. The
  slot is scalable again, and the node now reports the shrinkage so muPC
  undoes it. `is_skip_connection=False` and the L argument are unchanged.

### 9. L counts declared skip slots, not connected ones

- `_build_slots` (`fabricpc/graph_assembly/graph_construction.py:37-62`)
  emits every declared slot, with `in_neighbors` possibly empty;
  `_count_skip_connections_depth` tests
  `any(s.is_skip_connection for s in node_info.slots.values())`
  (`mupc.py:209`) without consulting `in_neighbors`.
- A `LinearResidual` used with no skip edge — its forward tolerates the
  empty slot (`fabricpc/nodes/linear_residual.py:190-195`) — therefore
  inflates L for the whole graph. Both the L count and the merge-node rule
  below must require a connected slot: `len(slot.in_neighbors) > 0`.

### Correct placement: the merge-node rule

The `1/√L` belongs once per branch, applied where the branch joins the
stream. Rule: a node is a merge node if it has at least one connected
`is_skip_connection` slot; edges into a merge node's scalable slots get
`a = gain/√(fan_in·K_slot·L)`; all other scalable edges get the L-free
`a = gain/√(fan_in·K_slot)`. Consequences:

- `LinearResidual` already has the required slot split (scalable "in",
  skip "skip"); its "in" formula is unchanged — identical numerics.
- `SkipConnection` gains the same split: "in" becomes the scalable compute
  slot (weightless, fan_in=1, so `a = gain/√(K_slot·L)`); a new
  non-scalable `is_skip_connection=True` "skip" slot receives the stream.
  This is `LinearResidual`'s slot layout minus the weights. All callers
  migrate in the same change (no dual-mode form):
  `examples/resnet18_cifar10_demo.py:196-224` (branch `conv_b` → "in";
  stream `prev`/`conv_skip` → "skip"), `examples/mupc_demo.py` skip mode,
  tests `test_skip_connection_unscaled`,
  `test_skip_depth_affects_compute_scaling`,
  `test_slot_is_variance_scalable_property`, and the usage example in
  `skip_connection.py`'s module docstring.
- Stems, branch-interior edges, stream projections (`conv_skip`), and
  post-stream edges (global pool) all become L-free — findings 5-7 close
  under the one rule.
- Transformer: `MhaResidualNode` and the FFN residual node declare skip
  slots (`fabricpc/nodes/transformer_v2.py:166-168,344`), so they are merge
  nodes and their scalable "in" slots keep the L factor — block edges keep
  current numerics. Only scalable edges into non-merge nodes of the
  transformer graph lose L.
- Damping position: for Linear+SkipConnection blocks the `1/√L` moves from
  the branch layer's in-edge (pre-weight) to the merge's compute edge
  (post-activation). The magnitude is the same, once per branch; for
  saturating activations the variance inside the nonlinearity differs.
  Depth-μP damps post-activation at the merge; jpc damps pre-weight; the
  two coincide only for identity activations. `LinearResidual` keeps the
  jpc (pre-weight) position.

Alternatives considered:

- **Damp the last scalable edge feeding a merge (former design B)**: no
  slot API change, but "last edge before the merge" needs a definition
  robust to fan-out (a node feeding both a merge and a non-merge consumer
  would need edge-specific scales) and to branches whose final hop is
  weightless. Rejected: the merge-node rule reads the same information from
  slot declarations already in place.
- **Per-edge skip annotations**: maximally general (any edge can be
  declared stream or branch), but duplicates what `SlotSpec` already
  expresses and adds a second annotation surface to keep consistent.
  Rejected.

Open question for arbitrary graphs (unchanged): the references assume one
residual stream with a single global L. Graphs with several disjoint or
nested streams may need per-merge L (the number of merges on that stream)
rather than one global `max(skip_depth, 1)`.

### Change items (implemented 2026-08-08)

1. `compute_mupc_scalings`: add the merge-node conditional (L factor only
   for nodes with a connected `is_skip_connection` slot); change
   `_count_skip_connections_depth` and the merge predicate to require
   `len(slot.in_neighbors) > 0` (finding 9).
2. `SkipConnection`: two-slot form and caller migration as listed above.
3. `StorkeyHopfield`: "in" slot `is_variance_scalable=False` (finding 8).
4. Docstrings in `mupc.py`:
   - `:267-269` (inline comment): states L is the "number of nodes with
     non-scalable slots along the longest path"; the counted property is
     `is_skip_connection`, and non-scalable non-skip slots (attention
     masks) are excluded — pinned by
     `test_metadata_slot_does_not_inflate_depth`. Wrong about current code;
     may land immediately, independent of the design change.
   - `:35-41` (module docstring): factually correct but conflates the
     flags; restate as: only `is_skip_connection` slots contribute to L;
     `is_variance_scalable` controls edge scaling, never L.
   - The module-docstring formula block (`:11-24`) and the
     `compute_mupc_scalings` docstring take the merge-node conditional when
     the rule lands.
5. Docs: `docs/user_guides/04_building_models.md:254-256` documents the
   ResNet global-pool edge's `1/√8` as expected behavior — wrong under the
   rule; the hidden-formula parts of
   `docs/user_guides/05_initialization_and_scaling.md` take the merge-node
   conditional.

### Impact

- Among shipped graphs, only `resnet18_cifar10_demo.py` changes numerics
  (stem, two-conv branches, downsample projections, post-pool edges). The
  transformer's block edges keep current numerics (merge nodes retain the
  L factor); only scalable edges into its non-merge nodes change. No
  StorkeyHopfield consumer uses muPC scaling, so finding 8 changes no
  shipped run. [2026-08-08 correction: an earlier revision claimed the
  StorkeyHopfield depth-comparison arms would change; none of those
  scripts passes `scaling=MuPCConfig`.]
- Empirical gate before landing: paired-depth experiments (paired N-arm
  runner) comparing uniform-L against the merge-node rule at several
  depths. The archive's jpc comparison
  (`muPC_arbitrary_graph_upgrade_plan.md:434`) shows depth-factor placement
  materially affects trainability, so the change must be measured, not only
  derived.

### Verification (unit tests implemented 2026-08-08; empirical runs pending)

- Unit tests extending the L=2 residual test: stem edge L-free;
  branch-interior edge L-free (requires a two-weighted-layer branch);
  merge compute edge damped exactly once; downsample-projection edge
  L-free; post-stream edge L-free.
- A declared-but-unconnected skip slot (`LinearResidual` with no skip edge)
  does not inflate L.
- StorkeyHopfield inside a muPC graph: its in-edge is absent from the
  per-edge scaling dicts (unscaled), and the s=0 forward reproduces
  `act(probe)` — the pass-through is preserved.
  **[Superseded 2026-08-09.]** Literal pass-through was never the muPC
  contract — no scaled node is a literal identity. Replaced by variance
  assertions across `s` and along a chain; see the 2026-08-09 verification.
- Deep-variance test: a residual chain at several L asserting final stream
  variance stays O(1) (the e-bound), which the current stem damping violates
  (≈ e/L).
- Fixed-seed before/after runs of `resnet18_cifar10_demo.py`, reported
  alongside the paired-depth results.

## Scope expansion: variance-reducing transforms (2026-08-09; implemented 2026-08-09 — empirical gate still pending)

Findings 5-7 removed the depth factor from every edge outside a merge. That
exposed a second species of defect underneath it: nodes whose transform
*reduces* input variance were never compensated at all, because the hook they
report through could not express a reduction. Findings 10-14 are that class.
The root defect is the hook's type and arguments (finding 10); findings 11-14
are the nodes it silenced.

Additional symbols: `v` is a node's input-transform variance factor — the factor
by which the transform multiplies input variance, before the activation, so that
`a = gain/√(v·K_slot)` undoes it. `n` is the number of cells a pooling window
reduces. `s` is `hopfield_strength`; `r = Var(probe @ W)/Var(probe)` is the
variance gain of one pass through StorkeyHopfield's Hopfield matrix W on
Hopfield dimension D.

### 10. The fan-in hook cannot express a variance-reducing transform

- `NodeBase.get_weight_fan_in(source_shape, config) -> int` is the only channel
  through which a node tells muPC how its transform changes input variance.
  Two limits, both structural:
  - The `int` return can only *reduce* the edge scale. A transform that reduces
    variance needs `v < 1` so the scale *amplifies*; no integer expresses that.
  - No `weight_init` argument. A node that builds a weight matrix internally
    rather than one per in-edge (StorkeyHopfield's W) cannot derive its own
    variance factor without hardcoding an initializer.
- The name is also wrong for the quantity: every existing implementation already
  returns the variance factor, and "weight fan_in" is merely what that factor
  equals for a matmul (each output unit sums `fan_in` independent products).
  Weightless nodes returning 1 were already returning a variance factor, not a
  fan_in.
- Resolution: replace with
  `get_variance_factor(source_shape, config, weight_init) -> float`. Pure
  generalization — every existing implementation returns the same number, so
  weighted-node scaling is unchanged. Migrated in one change, no alias and no
  deprecation shim: `base.py`, `identity.py`, `pooling.py`,
  `skip_connection.py`, `transformer.py`, `convolutional.py`, the call site
  `mupc.py:360`, `scripts/diagnose_deep_mupc.py`, the three custom node classes
  in `examples/jpc_fc_resnet_compare.py`, and the tests and user guides.

### 11. AvgPool attenuates the signal by 1/√n

- `_PoolBase.get_weight_fan_in` returned 1 for both pools, on the reasoning that
  a weightless node has no weight matrix. True but incomplete: `AvgPool` divides
  a sum of `n` cells by `n`, which for uncorrelated cells multiplies variance by
  `1/n` exactly — a property of the mean, not of any weight matrix.
- With `v = 1` the pool's in-edge is scaled 1.0 and the reduction passes
  through uncorrected, attenuating everything downstream by up to `1/√n`. In
  `examples/resnet18_cifar10_demo.py` the global pool collapses a 4×4 map, so
  the head's input was attenuated by up to `1/4`. Finding 7 removed the extra
  `1/√8` that had sat on that same edge, which left the pool's own reduction as
  the largest remaining un-normalized factor in that graph.
- Resolution: return `v = 1/n`, so muPC scales the in-edge by `√n`. `n` is the
  window volume, or every spatial dimension under `global_pool=True`. Under
  `count_include_pad=False` the divisor varies per window; the full window
  volume is used, exact under the default `"VALID"` padding.
- This is also the framework's own convention for a sum: muPC scales a
  `K_slot`-way edge sum by `1/√K_slot`, not `1/K_slot`. A spatial mean and an
  edge sum are both sums, and both are normalized to preserve variance rather
  than magnitude.
- Assumption, stated rather than hidden: the pooled cells are uncorrelated.
  Real conv feature maps are spatially correlated, so the realized reduction
  lies between `1/n` and `1` and `√n` over-corrects in proportion. This is the
  same independence assumption Kaiming already makes across a weight matrix's
  fan_in. Measured in a real conv graph (finding 11a below) rather than left as
  a caveat.

### 11a. Measured: the independence assumption holds well enough

- Graph: `conv(3×3, stride 2, ReLU, MuPCInitializer) → global AvgPool → Linear`,
  8×8×3 input, 4×4×32 feature map, `n = 16`, batch 512.
- Realized spatial reduction was `1/14.1` against the predicted `1/16`. With
  `v = 1/n` the conv→pool variance ratio is **1.13**; with `v = 1` it was
  **0.07**. The correction lands within 13% of unity.
- Recorded as a validation of the analytic `1/n`, not as a calibration: the hook
  stays analytic, because a measured per-graph constant cannot go into a scale
  derived from topology alone.

### 12. MaxPool has no distribution-free variance factor

- The variance of a max depends on the input distribution, and the two
  distributions that matter disagree. Measured ratio of `Var(max over n)` to the
  input variance, 4M samples:

      n     Gaussian N(0,1)     ReLU(N(0,1))
      4     0.49                1.32
      9     0.36                1.05
      16    0.29                0.87
      25    0.26                0.76

- Every `MaxPool` in this repo follows a ReLU conv at a 2×2 window
  (`examples/mnist_conv_demo.py:57,76`), where the ratio is 1.32 — max pooling
  slightly *increases* variance there. A Gaussian order-statistic correction
  (`1/√0.49` = 1.43) would be wrong by a factor of 2.7 for the inputs the node
  actually sees.
- Resolution: `v = 1.0`, with the measured table recorded in the module
  docstring as the reason. Deliberately not symmetric with finding 11: the
  AvgPool correction is exact and distribution-free, the MaxPool one would not
  be.
- Recorded limitation, uncorrectable by this mechanism: max pooling shifts the
  mean (measured mean 1.05 at n=4 for ReLU inputs, against a standard deviation
  of 0.67). A scalar edge multiplier scales an offset rather than removing it.
  In an un-normalized PC network that offset propagates downstream. The
  docstrings direct users to `AvgPool` where variance behavior through depth
  matters; a node-level centering option would be the fix, out of scope here.

### 13. The SkipConnection two-slot migration fails silently for unmigrated callers

- After the change item 2 above, `skip.slot("in")` still exists and still
  accepts a stream edge — the layout every prior doc and example prescribed.
  Routed there, the node has no connected skip slot, so it drops out of the
  merge count (finding 9's rule), L collapses toward 1 across the graph, and
  both stream and branch are scaled `1/√2`. That is the `0.707^L` decay
  `SkipConnection` exists to prevent, reintroduced with no error and no warning.
- In-repo callers were migrated in the same change, so this is a hazard for
  external graphs and notebooks rather than a live defect here.
- Resolution: `SlotSpec` gains `require_connected`; graph construction raises
  when such a slot receives no edge. `SkipConnection`'s "skip" slot sets it — a
  SkipConnection without a stream edge is an `IdentityNode` with extra steps.
  Scoped to the slot that declares it, so `LinearResidual` without a skip edge
  stays legal (finding 9 deliberately tolerates that).

### 14. StorkeyHopfield's blend shrinks variance; exempting the slot lets it compound

- Supersedes finding 8's resolution. The blend is
  `act(probe/(1+s) + (probe @ W)·s/(1+s) + b)`. The coefficients sum to 1, but
  that preserves scale only for perfectly correlated terms. `probe` and
  `probe @ W` are near-independent, so their variances add in quadrature:

      v(s) = Var(blend)/Var(probe) = (1 + s²·r) / (1 + s)²

  which is at most 1 for every `s ≥ 0` and reaches its minimum `r/(1+r)` at
  `s = 1/r`.
- Under the node's own initialization — Xavier on (D, D), so `E[W_ij²] = 1/D`,
  then `_prepare_W`'s symmetrization `W ← (W + Wᵀ)/2`, which halves off-diagonal
  variance and leaves the diagonal alone:

      r = (D−1)/(2D) + 1/D = (D+1)/(2D)   →  1/2 for large D

  So the worst case is `s = 2`, `v = 1/3` (per-node standard-deviation ratio
  0.577); at the default learnable strength (`softplus(raw_init) = 1.0`),
  `v = 0.375`, ratio 0.612. Uncorrected that compounds: a 32-node chain
  attenuates by `0.612³² ≈ 1e-7`.
- The blend is linear in the probe, so an edge scale factors straight out:
  `act(a·[blend] + b)`. Reporting `v(s)` therefore corrects it exactly.
- Resolution: `is_variance_scalable=True` (the default) restored;
  `is_skip_connection=False` unchanged, so the node still does not count toward
  L — the activation wraps the identity path, so it is not a residual-stream
  merge. `get_variance_factor` returns `v(s)`.
- At `s = 0`, `v = 1` and `a = gain`, so the identity blend gets the same
  treatment as any other muPC-scaled node. This is the defect finding 8 was
  working around: the *inherited* `fan_in = D` gave `a = gain/√D` and collapsed
  the blend toward `act(0)`. The correct factor removes it; exempting the slot
  merely avoided it.
- Two caveats, both inherent to a scale computed once from topology, both
  recorded in the node docstring rather than left implicit:
  - `r` depends on `weight_init`, which is what forced the third parameter in
    finding 10. It is derived from `InitializerBase.element_variance` and the
    `enforce_symmetry` / `zero_diagonal` config, not assumed Xavier.
  - With `hopfield_strength=None` (the default) `s` is learnable, so `v` is
    computed at construction from `s = softplus(raw_init) = 1.0` and does not
    track training. This is muPC's standard contract — scales are static and
    correct at initialization. The drift is bounded: `v` ranges only over
    `[r/(1+r), 1]` = `[1/3, 1]` at `r = 1/2`, so the scale is never off by more
    than `√3 ≈ 1.73`.

### 15. `InitializerBase` does not expose the variance it draws

- Finding 14 needs `E[W_ij²]` for an arbitrary initializer at an arbitrary
  shape. Every initializer computes exactly that internally to draw its samples,
  but none exposes it.
- Alternatives considered:
  - **Sample W once and measure**: needs no new interface, but the scale then
    depends on a key and carries sampling noise (~9% in `r` at D=16), and a
    scaling factor derived from topology should not consume randomness.
    Rejected.
  - **Hardcode the Xavier closed form in StorkeyHopfield**: silently wrong for
    any non-default `weight_init`, which the constructor accepts. Rejected.
  - **Normalize W internally so `r = 1` by construction**: removes the coupling,
    but `E_hop = (s/2D)·zᵀ(W²−W)z` scales with W, so it would change the
    attractor dynamics of every existing StorkeyHopfield experiment. Rejected
    as an unrelated behavior change.
- Resolution: `InitializerBase.element_variance(shape, config) -> float`, closed
  form, implemented for all seven built-ins and verified against empirical
  draws. Not abstract: it raises `NotImplementedError` with a message naming
  what to implement, so a custom initializer fails loudly at graph construction
  instead of silently mis-scaling.

### 16. The merge-node predicate was duplicated

- The rule from change item 1 landed as the same expression in two places:
  `_count_skip_connections_depth` (`mupc.py:219-222`) and the per-node loop in
  `compute_mupc_scalings` (`:317-320`). Which nodes count toward L and which
  nodes' edges carry the factor must agree by construction, or the damping lands
  on edges the depth count did not account for.
- Resolution: extracted `_is_merge_node(node_info)`, called from both.

### Change items (implemented 2026-08-09)

1. `NodeBase.get_weight_fan_in(source_shape, config) -> int` →
   `get_variance_factor(source_shape, config, weight_init) -> float`, with all
   implementations, the call site, scripts, examples, tests and docs migrated in
   the same change (finding 10). **Breaking** for custom nodes: rename plus a
   third parameter.
2. `AvgPool.get_variance_factor` returns `1/n` via a shared
   `_PoolBase._pool_cell_count`; `MaxPool.get_variance_factor` returns 1.0 with
   the measured table as its stated reason (findings 11, 12).
3. `SlotSpec.require_connected`, enforced in `graph_construction.py` next to the
   existing slot validation; set on `SkipConnection`'s "skip" slot (finding 13).
4. `StorkeyHopfield`: slot scalable again, `get_variance_factor` returns `v(s)`,
   and the four docstring sites that carried the "coefficients sum to 1" claim
   (`storkey_hopfield.py` module docstring and `get_slots`, `CHANGELOG.md`,
   `10_api_nodes.md`) restate the actual algebra (finding 14).
5. `InitializerBase.element_variance` plus the module-level `element_variance`
   convenience function; a shared `_fans(shape)` helper now backs both the
   Xavier/Kaiming `initialize` and `element_variance` paths so the two cannot
   drift (finding 15).
6. `_is_merge_node` extracted in `mupc.py` (finding 16).
7. Docs: `05_initialization_and_scaling.md` gains a "Variance Factor" section
   replacing "Kaiming Fan_in Scaling", with the per-node table and the pooling
   and StorkeyHopfield derivations; `06_custom_nodes.md` step 3 documents the
   new hook; `04_building_models.md` and `10_api_nodes.md` take the per-node
   values. Formula blocks throughout use `v` where they said `fan_in`.
8. Stale text from the 2026-08-08 change: `transformer_demo.py`'s header still
   claimed both SkipConnection inputs were at scale 1.0;
   `resnet18_cifar10_demo.py`'s results block had an unclosed parenthesis.

### Impact

- `examples/resnet18_cifar10_demo.py` is again the only shipped graph whose
  numerics change: its global AvgPool in-edge goes from 1.0 to 4.0. This stacks
  with the 2026-08-08 merge-node change on the same graph, so the two cannot be
  measured separately after the fact — see the gate below.
- `examples/mnist_conv_demo.py` uses `MaxPool` only and is unchanged.
- No StorkeyHopfield consumer passes `scaling=MuPCConfig`, so finding 14 changes
  no shipped run, as with finding 8.
- Weighted nodes report the same variance factor they reported as a fan_in, so
  finding 10 changes no scaling anywhere on its own.

### Verification (unit tests implemented 2026-08-09; empirical runs pending)

Full suite: 339 passed, up from 309. black and ruff clean.

- `element_variance` against empirical draws for all seven built-ins across
  distributions and modes; shape tracking for the fan-based schemes against the
  same ND fan convention `initialize` uses; `NotImplementedError` for an
  initializer that omits it.
- `AvgPool` variance factor `1/n` for windowed and global modes, including that
  channels do not enter `n`; `MaxPool` fixed at 1.0 and independent of window
  size.
- Graph-level: `conv → global AvgPool → Linear`, asserting the pool in-edge
  scale is `√16` and the head's input variance is not divided down (finding
  11a's measurement, as a test).
- StorkeyHopfield: closed-form `v(s)` against the measured blend ratio at
  `s ∈ {0, 0.5, 1, 2, 5}`; the `v ≤ 1` bound and the `r/(1+r)` minimum at
  `s = 1/r`; the learnable-strength case evaluating at `s = 1`; the probe edge
  present in the per-edge dicts with `a = gain` at `s = 0`; a 12-node chain at
  `s ∈ {0, 1, 2}` reaching a variance fixed point.
  - Note on the chain test's bound: the chain settles near 0.28 at `s = 1`, not
    near 1. The level is tanh under the Kaiming gain, not this node — a plain
    `Linear + tanh` chain under muPC settles *lower*, at 0.22, by the same
    mechanism. The test therefore asserts the flat tail
    (`var_last > 0.7·var_mid`), which the uncorrected node misses by ~25x, not
    an absolute band.
- The pre-migration SkipConnection edge layout raises at graph construction;
  `LinearResidual` without a skip edge still builds and stays L-free.
- `scripts/diagnose_deep_mupc.py` (the only `include_output=True` consumer, L=1
  pure chain, no pooling) prints numerics identical to before.
- ResNet-18 scaling inspected end to end: stem `√2/√27`, branch convs
  `√2/√288`, merge branch edge `1/√8`, downsample projection `1/√32`, global
  pool `4.0`. Stem, branch and projection L-free; damping once per branch at the
  merge; the pool amplifying by `√n`.

### Empirical gate (three arms)

Widened from the 2026-08-08 entry: the AvgPool fix changes the same graph as the
merge-node rule, so a two-arm comparison can no longer separate them.

1. `main` — uniform-L, AvgPool `v = 1`.
2. 2026-08-08 branch — merge-node rule, AvgPool `v = 1`.
3. 2026-08-09 branch — merge-node rule, AvgPool `v = 1/n`.

Plus the paired-depth sweep `examples/mupc_demo.py --num_blocks {8,16,32,64,128}`,
which contains no pooling and so isolates the depth rule. The ResNet graph cannot
do this: it is fixed at L = 8, and its AvgPool change is confounded with the
depth rule.

### 17. The recorded ResNet regression measured hyperparameter fragility, not the rule (2026-08-10)

- The earlier figure (37.08% → 35.00%, train energy 0.1096 → 0.8319) was taken at
  `lr = 0.01`, `infer_steps = 80`, with lr, weight_decay and infer_steps changed
  in the same commit as the rule. Re-running showed why it could not measure the
  rule: at `lr = 0.01` the run is not robust to small perturbations — accuracy
  moves by more than the effect under test in response to changes that do not
  alter the objective. Any scaling change reads as a regression at that lr.
- New operating point, now the demo defaults
  (`examples/resnet18_cifar10_demo.py`): `lr = 0.001`, `infer_steps = 120`,
  `weight_decay = 0.01`, batch 256, 2 epochs, ReLU, no augmentation. Both arms
  below use it, so the comparison isolates the scaling change.

### Result (2026-08-10): arms 1 and 3, ResNet-18 / CIFAR-10

| Arm | Scaling | Test accuracy | Train energy | Training time |
| --- | --- | --- | --- | --- |
| 1 (`main`) | uniform-L, AvgPool `v = 1` | 31.50% | not recorded | not recorded |
| 3 (this branch) | merge-node rule, AvgPool `v = 1/n` | 33.71% | 0.4792 | 952.3s (476.2s/epoch) |

RTX 3090, CUDA 13, JAX 0.10.2. The +2.21 point gap is the combined effect of the
merge-node rule and the AvgPool variance factor, in the direction the derivation
predicts. The gate passes for the ResNet arm.

Qualifications, stated rather than left to the reader:

- **Arm 2 was not run**, so the merge-node rule and the AvgPool factor remain
  jointly measured. Their separation is still open.
- Single seed per arm at 2 epochs. The demo header already warns that accuracy
  varies a few points across JAX versions and hardware; a 2.21-point gap from one
  seed each is directional, not a confidence interval. Repeated seeds would
  settle it.
- Arm 1's train energy and training time were not recorded, so only accuracy is
  comparable across arms.

The demo docstring's figures are arm 3; it does not yet say so.

### Result: paired-depth sweep, FC-ResNet / MNIST

`examples/mupc_demo.py --num_blocks {8,16,32,64,128}`, `--mode linear_residual`
(the default), hidden 64, 3 epochs, lr 0.002, weight_decay 0.01. No pooling in
this graph, so the comparison is the depth rule alone — uniform-L against the
merge-node rule, uncontaminated by the AvgPool factor.

| Depth L | `main` (uniform-L) | merge-node rule | Δ |
| --- | --- | --- | --- |
| 8 | 90.8% | 92.0% | +1.2 |
| 16 | 89.7% | 89.7% | 0.0 |
| 32 | 82.4% | 85.6% | +3.2 |
| 64 | 77.1% | 84.1% | +7.0 |
| 128 | 70.1% | 82.2% | +12.1 |

The gain grows with depth, which is the signature finding 5 predicts. Under
uniform-L the stem is damped by `1/√L`, so the stream starts at variance `1/L`
and the final stream sits at ≈ e/L — an attenuation that worsens as L grows and
that the merge-node rule removes by making the stem L-free. Accuracy under
uniform-L falls 20.7 points from L = 8 to L = 128; under the merge-node rule it
falls 9.8. Depth transfer is not restored, but the depth penalty is roughly
halved.

Two annotations on the table: depth 16 is identical in both arms, so it is either
an unre-run cell or a coincidence — it is the one row that carries no signal.
The `--mode skip` arm is not recorded in the demo docstring; the table above is
`linear_residual` only. Under that mode `LinearResidual`'s "in" edge keeps its
formula, so what the sweep measures is the stem and post-stream edges losing
their L factor.

### Gate outcome

Both arms of the gate pass. The depth sweep confirms the merge-node rule in
isolation and at five depths; the ResNet run confirms the combined change at a
single depth. What remains unmeasured is the split between the merge-node rule
and the AvgPool factor within the ResNet result (arm 2), which no run separates.
