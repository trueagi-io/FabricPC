# Unity Variance Through TransformerBlock for muPC

## Context

The `TransformerBlock` node encapsulates a complete transformer layer (MHA + FFN with residual connections) as a single predictive coding node. For muPC compatibility, we need `Var(z_mu) = 1` given `Var(input) = 1`. The current GPT-2-style initialization (1/sqrt(2*n_blocks) on output projections with n_blocks=1) produces `Var(z_mu) ≈ 1.5` — not unity. Additionally, the node lacks a `get_weight_fan_in()` override, and the softmax attention averaging contracts variance by ~1/S at init.

## Variance Accounting (Current Code)

Each step through `forward()` with current init, assuming `Var(input) = 1`:

| Step | Operation | Variance | Notes |
|------|-----------|----------|-------|
| 1 | `LN1(input)` | **1.0** | LayerNorm normalizes to unit variance |
| 2 | `Q = x_norm @ W_q`, W_q~N(0, 1/sqrt(d)) | **1.0** | d * 1 * (1/d) = 1 |
| 3 | `K = x_norm @ W_k` | **1.0** | Same as Q |
| 4 | `V = x_norm @ W_v` | **1.0** | Same as Q |
| 5 | `RoPE(Q)`, `RoPE(K)` | **1.0** | Rotation preserves norm: cos^2 + sin^2 = 1 |
| 6 | `scores = QK^T / sqrt(d_h)` | **1.0** | d_h * 1 * 1 / d_h = 1 |
| 7 | `attn = softmax(scores)` | — | Stochastic row vectors summing to 1 |
| 8 | `context = attn @ V` | **1/S** | At init, attn ≈ uniform (1/S each), so Var = S*(1/S)^2*1 = 1/S |
| 9 | `reshape + W_o`, W_o~N(0, 1/sqrt(2d)) | **1/(2S)** | d * (1/S) * 1/(2d) = 1/(2S) |
| 10 | `x_res1 = input + attn_output` | **≈ 1** | 1 + 1/(2S) ≈ 1 (residual dominates) |
| 11 | `LN2(x_res1)` | **1.0** | LayerNorm normalizes |
| 12 | `GELU(x_norm2 @ W_ff1)`, Kaiming init | **≈ 1.0** | d * 1 * (2/d) = 2 pre-act; GELU halves → ~1 |
| 13 | `ff_out = x @ W_ff2`, N(0, 1/sqrt(2*d_ff)) | **0.5** | d_ff * 1 * 1/(2*d_ff) = 1/2 |
| 14 | `z_mu = x_res1 + ff_output` | **≈ 1.5** | 1 + 0.5 = 1.5. **Not unity.** |

**Three problems identified:**
1. **Softmax averaging** contracts variance by 1/S (step 8) — attention output is near-zero at init
2. **W_o and W_ff2** use GPT-2 multi-block convention (1/sqrt(2N)) which is wrong for a single PC node
3. **Residual additions** (steps 10, 14) accumulate variance without compensation

## Key Insight: LayerNorm Absorbs External muPC Scaling

Since `LN(a*x) = LN(x)` for scalar a > 0, the muPC external `forward_scale` applied to the input is completely absorbed by the first LayerNorm. The topdown gradient also self-corrects: `dE/d(ax) = (1/a) * dE/dx`, so `topdown_grad = a * jac_gain * (1/a) * dE/dx = jac_gain * dE/dx`. The `a` factors cancel exactly.

**Implication:** All variance control must be handled **inside** `forward()`. The external muPC machinery is effectively a pass-through for this node.

## Proposed Variance Chain (After Changes)

| Step | Operation | Variance | Change |
|------|-----------|----------|--------|
| 1 | `LN1(input)` | **1.0** | — |
| 2-5 | Q, K, V projections + RoPE | **1.0** | — |
| 6 | `scores / sqrt(d_h)` | **1.0** | — |
| 7-8 | `context = attn @ V` | **1/S** | — |
| 9 | `_mha output * sqrt(S)` | **1.0** | **NEW: compensate softmax contraction** |
| 10 | `context @ W_o`, W_o~N(0, 1/sqrt(d)) | **1.0** | **CHANGED: Xavier init (was 1/sqrt(2d))** |
| 11 | `x_res1 = (1/sqrt(2)) * (input + attn_out)` | **1.0** | **NEW: balanced residual scaling** |
| 12 | `LN2(x_res1)` | **1.0** | — |
| 13 | `GELU(x @ W_ff1)` Kaiming | **1.0** | — |
| 14 | `ff_out = x @ W_ff2`, N(0, 1/sqrt(d_ff)) | **1.0** | **CHANGED: Xavier init (was 1/sqrt(2*d_ff))** |
| 15 | `z_mu = (1/sqrt(2)) * (x_res1 + ff_out)` | **1.0** | **NEW: balanced residual scaling** |

## Design Decisions

**sqrt(S) attention compensation:** Applied in `forward()` after `_mha()` returns, conditional on `scaling_config is not None`. At init with uniform attention, this is exact. During training, attention peaks and W_o adapts — muPC only guarantees init-time properties. The `_mha()` method itself stays unchanged (backward compatible).

**1/sqrt(2) residual scaling:** Matches the validated `PreActResBlock` pattern in `examples/jpc_fc_resnet_compare.py` (lines 320-333) which uses `skip_scale = 1/sqrt(K)` with K=2. Applied as `(1/sqrt(2)) * (a + b)` at each residual add.

**Init changes as new default:** The current 1/sqrt(2*n_blocks) with n_blocks=1 is a half-implemented GPT-2 convention. Xavier (1/sqrt(d) and 1/sqrt(d_ff)) is the correct single-block init regardless of muPC. The residual variance control is now handled by the forward() scaling, not the init.

## Implementation

### 1. Add `get_weight_fan_in()` to TransformerBlock
**File:** `fabricpc/nodes/transformer.py`, after `get_slots()` (after line 174)

```python
@staticmethod
def get_weight_fan_in(source_shape, config):
    """Return embed_dim as fan_in for muPC scaling.
    
    LayerNorm absorbs the external forward_scale (LN(ax) = LN(x)),
    so the specific fan_in value doesn't affect forward variance or
    gradients. We return embed_dim for consistency with the framework.
    """
    return source_shape[-1]
```

### 2. Update `initialize_params()` — W_o and W_ff2 to Xavier
**File:** `fabricpc/nodes/transformer.py`, lines 211-241

- Remove `n_blocks = 1` (line 211)
- W_o: `NormalInitializer(std=1.0 / jnp.sqrt(embed_dim))` (was `1.0 / jnp.sqrt(embed_dim * 2 * n_blocks)`)
- W_ff2: `NormalInitializer(std=1.0 / jnp.sqrt(ff_dim))` (was `1.0 / jnp.sqrt(ff_dim * 2 * n_blocks)`)
- Update the documentation table (lines 196-209) to reflect new init rationale

### 3. Add internal muPC scaling in `forward()`
**File:** `fabricpc/nodes/transformer.py`, lines 258-346

Conditional on `node_info.scaling_config is not None`:

a) After `_mha()` returns (line 310), scale by sqrt(seq_len):
```python
if node_info.scaling_config is not None:
    attn_output = attn_output * jnp.sqrt(jnp.float32(seq_len))
```

b) At residual connection 1 (line 313):
```python
if node_info.scaling_config is not None:
    inv_sqrt2 = jnp.float32(1.0 / jnp.sqrt(2.0))
    x_res1 = inv_sqrt2 * (input_tensor + attn_output)
else:
    x_res1 = input_tensor + attn_output
```

c) At residual connection 2 (line 330):
```python
if node_info.scaling_config is not None:
    z_mu = inv_sqrt2 * (x_res1 + ff_output)
else:
    z_mu = x_res1 + ff_output
```

### 4. Update init documentation table
**File:** `fabricpc/nodes/transformer.py`, lines 196-209

Replace the table to reflect Xavier W_o/W_ff2 and document the muPC forward() scaling factors.

## Verification

1. **Variance test:** Initialize a TransformerBlock with muPC scaling, feed random unit-variance input, verify `Var(z_mu) ≈ 1.0` (tolerance: 0.7 to 1.5 for finite-size effects)
2. **Regression test:** Without muPC (scaling_config=None), verify forward() output matches current behavior exactly (no residual scaling, no sqrt(S))
3. **Fan-in test:** Verify `TransformerBlock.get_weight_fan_in((128, 512), {})` returns 512
4. **Integration test:** Run existing `test_mupc.py` tests to ensure no regressions in the muPC framework

## Files to Modify
- `fabricpc/nodes/transformer.py` — all changes (get_weight_fan_in, init, forward scaling)
- `tests/test_mupc.py` — add transformer-specific variance tests

## Notes
- **JIT safety:** `node_info.scaling_config is not None` is a static condition (NodeInfo is frozen dataclass in static auxdata), safe for JAX JIT tracing
- **Causal masking caveat:** sqrt(S) is exact for non-causal attention. With causal masking, per-position contraction varies (1/t for position t). This is acceptable — muPC provides init-time guarantees and the model adapts during training
- **transformer_v2 follow-up:** `MhaResidualNode` and `Mlp2ResidualNode` have the same residual variance issue; should be addressed separately using the same pattern
