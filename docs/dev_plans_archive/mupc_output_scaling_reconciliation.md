# Reconcile muPC output-node scaling with the μPC paper (√L divergence)

## Context

A review agent flagged that `fabricpc/core/mupc.py:330` scales output nodes as
`a = gain/(fan_in·√(K_slot·L))` — carrying a `√L` factor that μPC's Table 1
output scaling (`a_L = 1/N`) does not — while the inline comment
(`mupc.py:328`) and the `MuPCConfig` docstring (`mupc.py:126-130`) claim the
L-free form. This plan records the investigation findings and the fix.

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
- The √L entered when depth `L` was folded into the shared `√(K_slot·L)`
  subexpression for both branches (commit 68e7700, v0.3.0); that commit
  message documents only the hidden formula. Commit cca03de (#20) states "no
  mathematical changes to scaling formulas".
- Both in-code docstrings state the L-free form; the essay
  (`docs/background/scaling_breakthroughs_pc_and_backprop.md:330-335,435-443`)
  and `docs/dev_plans/pc_backprop_scaling_history.md:250-257` flag the √L as
  an unreconciled deviation, not an intent.
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
  `1/N` unconditionally — no L even in the skip-connection case. (FabricPC's
  `L = max(skip_depth, 1)` chain degeneration matches jpc's hidden convention
  exactly; only the output branch diverges.)

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
  is computed (`graph_construction.py:204-232`), and clamping is driven by
  `task_map` alone (`train.py:136-140`). Any sink — auxiliary head, probe —
  would receive output scaling; sinks coincide with the supervised target
  only by convention in shipped graphs. (Observation only; no change here.)

## Change

1. **`fabricpc/core/mupc.py:330`** — drop the √L:
   `a = gain / (fan_in * math.sqrt(K_slot))`. `topdown_grad_scale = a·jac_gain`
   follows automatically.
2. **Comments/docstrings in `mupc.py`** — update the inline block
   (`:323-328`) and the module docstring to state the output formula
   explicitly: hidden `gain/√(fan_in·K_slot·L)`, output `gain/(fan_in·√K_slot)`
   (no depth factor: the readout is applied once to the O(1) stream, not
   summed L times). Extend the `MuPCConfig.include_output` docstring
   (`:125-134`) to include the gain: `a = gain/(fan_in·√K)`.
3. **Test** — add to `tests/test_mupc.py` a graph with skip connections
   (L ≥ 2) and `include_output=True`; assert the output edge scale equals
   `gain/(fan_in·√K)` (L-independent) and a hidden edge equals
   `gain/√(fan_in·K·L)`. This pins the distinction the L=1 test cannot see.
4. **Docs** — update `docs/background/scaling_breakthroughs_pc_and_backprop.md`
   Part IV (`:330-335`, "departs from muPC's Table 1" paragraph) and replace
   the maintainer note (`:435-443`) with a note that the code now matches
   Table 1; update `docs/dev_plans/pc_backprop_scaling_history.md:250-257`
   likewise.

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
  consumer; L=1, so numerics must be identical before/after.
