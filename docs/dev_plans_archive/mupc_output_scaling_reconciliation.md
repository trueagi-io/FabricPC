# Reconcile muPC output-node scaling with the μPC paper (√L divergence)

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
