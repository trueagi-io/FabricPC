# Fix FabricPC muPC Scaling for Deep Residual Networks

## Context

The `jpc_fc_resnet_compare.py` A/B test shows FabricPC's muPC scaling collapses at depth >= 32 while JPC's scaling remains stable to depth 128+. The root cause is **exponential signal decay through attenuated skip connections**.

## Root Cause: Skip Attenuation Kills Signal-to-Noise Ratio

FabricPC's in-degree formula treats the skip connection and linear path as K=2 independent, equally-important inputs, scaling both by 1/sqrt(K):

```
FabricPC:  z_mu = (1/sqrt(N)) * act(x) @ W  +  (1/sqrt(2)) * x
JPC:       z_mu = (1/sqrt(N*L)) * act(x) @ W  +  1.0 * x
```

### Forward variance per layer (both preserve O(1) — this is NOT the problem)

| Component | FabricPC | JPC |
|-----------|----------|-----|
| Linear path variance | scale^2 * N * Var(act)/1 = (1/N)*N*(1/2) = **0.5** | (1/(NL))*N*(1/2) = **0.5/L** |
| Skip path variance | (1/sqrt(2))^2 = **0.5** | 1.0^2 = **1.0** |
| Total output variance | **1.0** (preserved) | **1 + 0.5/L** (bounded growth) |

### The real problem: signal coherence through skip connections

The top-down gradient at layer l propagates to layer l-1 through the Jacobian:

```
J = scale * diag(act'(x)) @ W  +  skip_scale * I
```

The **coherent signal** (component that correlates with the original error) propagates through the `skip_scale * I` term. Over L layers:

| Metric | FabricPC | JPC |
|--------|----------|-----|
| Signal amplitude after L layers | **(1/sqrt(2))^L** | **1.0^L = 1** |
| Noise amplitude (maintained by variance preservation) | **O(1)** | **O(1)** |
| SNR at depth 8 | 0.707^8 = 0.06 | 1.0 |
| SNR at depth 16 | 0.707^16 = 0.004 | 1.0 |
| SNR at depth 32 | 0.707^32 = 1.5e-5 | 1.0 |
| SNR at depth 64 | 0.707^64 = 2.3e-10 | 1.0 |

This explains the results table exactly: depth 8 still works (~92.6%), depth 16 is degraded (~85.6%), depth 32+ collapses to chance (~10%).

### Why the K=2 formula is wrong for residual connections

The formula `a = gain/sqrt(fan_in * K)` assumes all K inputs are **independent and equally important**. In a residual block:
- The skip connection is the **identity signal carrier** (must be preserved)
- The linear path is the **learned perturbation** (should be small for deep nets)

These are fundamentally asymmetric. Scaling both by 1/sqrt(K) destroys the residual structure.

### Why JPC's depth factor works

With skip_scale=1.0, variance grows as (1 + linear_var)^L per layer. JPC sets linear_var = O(1/L) via the 1/sqrt(N*L) factor, giving total growth (1+1/L)^L -> e ~ 2.72 as L -> inf. Bounded.

### Note on the transformer plan

The `transformer_mupc_variance_plan.md` references the 1/sqrt(2) residual scaling from the comparison script as "validated". That plan is safe because **LayerNorm renormalizes variance at each residual connection**, preventing the exponential decay. The FC-ResNet has no LayerNorm, exposing the issue directly.

## Implementation Plan

### Step 1: Add variance diagnostic probes to comparison script

**File:** `examples/jpc_fc_resnet_compare.py`

Add a `--probe_variance` flag that instruments the trained model to measure per-layer statistics after inference converges:
- Forward z_mu variance at each layer
- Top-down gradient variance at each layer  
- Signal coherence: correlation between input-layer gradient and output error
- Skip vs linear path contribution ratio

This produces an empirical variance table confirming the theoretical analysis.

Implementation: add a function `probe_variance(params, structure, test_loader, ...)` that:
1. Takes a batch from the test loader
2. Runs inference to convergence
3. For each layer, logs `Var(z_mu)`, `Var(latent_grad)`, and the skip/linear decomposition
4. Prints a per-layer table

### Step 2: Add corrected `fabricpc_v2` scaling mode

**File:** `examples/jpc_fc_resnet_compare.py` — `compute_scaling_factors()`

Add a new mode `"fabricpc_v2"` that fixes the skip attenuation:

```python
elif mode == "fabricpc_v2":
    # Fixed residual scaling: unattenuated skip + depth-compensated linear
    relu_gain = math.sqrt(2.0)
    in_scale = 1.0 / math.sqrt(input_dim)
    hidden_scale = relu_gain / math.sqrt(width * depth)  # depth factor restores bounded variance
    skip_scale = 1.0                                      # identity mapping preserved
    out_scale = 1.0 / width
```

Key change: `skip_scale = 1.0` and `hidden_scale` includes depth factor `L` instead of in-degree `K=2`.

Also add the mode to the argparse choices and the `--scaling` flag.

### Step 3: Run comparison experiments

Run the comparison at depths 8, 16, 32, 64 for all three modes:
- `jpc`: reference (expected ~83-91%)
- `fabricpc`: current broken scaling (expected collapse at depth 32+)
- `fabricpc_v2`: fixed scaling (expected to match or approach jpc)

Also run with `--probe_variance` at depth 32 and 64 for `fabricpc` vs `fabricpc_v2` to confirm the variance analysis.

### Step 4: Update comparison script documentation

Update the docstring results table and comments to reflect the new mode and findings.

## Key Files

| File | Role |
|------|------|
| `examples/jpc_fc_resnet_compare.py` | Comparison script — add probes + new mode (lines 120-164 for scaling, 428-519 for graph builder) |
| `fabricpc/core/mupc.py` | Core muPC framework (lines 213-231 for the formula) — **read-only for now** |
| `fabricpc/nodes/base.py` | forward_and_latent_grads / forward_and_weight_grads (lines 432-464 for gradient scaling) — **read-only** |
| `fabricpc/core/inference.py` | InferenceSGDNormClip (lines 279-323) — **read-only** |

## Out of Scope (follow-up work)

Fixing `mupc.py` itself to handle residual connections in general graphs. This requires a design decision about how to distinguish skip edges from transform edges in the graph topology. Options include:
1. Compute effective depth from longest path and use it as L
2. Allow edge annotations (skip vs transform)  
3. Detect residual patterns automatically from graph structure

This is a larger architectural question that should be addressed after validating the fix in the comparison script.

## Verification

1. Run `fabricpc_v2` at depths 8/16/32/64 — accuracy should remain >80% at all depths
2. Variance probes at depth 64: `fabricpc` should show exponential gradient decay; `fabricpc_v2` should show stable gradients
3. Existing tests pass: `python -m pytest tests/test_mupc.py` (no changes to core mupc.py)
