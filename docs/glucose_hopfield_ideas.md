# Hopfield Networks for Glucose Forecasting: Brainstorm

## Context

### Current glucose model

```
glucose (batch, seq_len, 1)
  → ContinuousEmbed (batch, seq_len, d_model)
  → [MultiScaleMha(+skip) → LnMlp1 → Mlp2Residual(+skip)] × depth
  → RegressionOutput (batch, horizon)
```

- **PC vs backprop**: PC achieved ~21 mg/dL MAE, backprop ~19 mg/dL — PC did not improve over backprop.
- **Best tuned PC**: Optuna trial 34 — context 64, depth 2, 1 head, val MAE 21.3 mg/dL.

### Existing StorkeyHopfield node

Classical Hopfield node with dual energy: `E_total = E_pc + s * E_hop` where `E_hop = (1/2D) z^T (W² - W) z`.

Key findings from MNIST experiments:
- **+2.6% to +10.6%** accuracy improvement on **noisy, few-shot** data.
- No benefit on clean, abundant data.
- Performance peaks at `hopfield_strength ≈ 2.0` (inverted-U response).
- PC inference naturally provides attractor dynamics via energy gradient descent — no explicit self-feedback recurrence needed.

### Why Hopfield could help glucose

CGM data has properties that align with Hopfield's proven strengths:

1. **Noisy inputs** — CGM sensors have ~10–15% measurement error, placing glucose squarely in the regime where Hopfield already helps on MNIST.
2. **Recurring temporal motifs** — post-meal spikes, dawn phenomenon, exercise dips are repeated patterns that could be stored as attractors.
3. **Distribution shift** — test data from different days/conditions requires generalization; attractor memory could regularize predictions toward plausible patterns.
4. **The PC gap** — PC underperformed backprop by ~2 mg/dL, suggesting the inference loop isn't adding value. Hopfield energy could give PC inference something productive to do during its inner loop.

---

## Ideas

### 1. Hopfield Denoising After Embedding

**Promise: High | Effort: Low**

Insert a `StorkeyHopfield` node between `ContinuousEmbeddingNode` and the first transformer block. The embedding maps scalar glucose → d_model vectors. The Hopfield layer learns prototypical embedding patterns and pulls noisy embeddings toward them during PC inference.

```
glucose → Embed → StorkeyHopfield → [MHA + MLP] × depth → Output
```

**Why it could work**: Directly exploits the proven noisy-data advantage. CGM noise is in the input; denoising early prevents error propagation downstream. The Hopfield node learns a "vocabulary" of typical glucose embedding states.

**Design choices**:
- Shape: `(seq_len, embed_dim)` — operates on last axis (embed_dim), so each timestep's embedding gets pulled toward learned prototypes independently.
- Strength: start with learnable (peaks around 2.0 in MNIST experiments).
- No bias, enforce symmetry (matching the working defaults).
- Activation: try both `TanhActivation` (Hopfield default) and `IdentityActivation` (to not compress the embedding range).

**Implementation**: Minimal — add one `StorkeyHopfield` node and one `Edge` in `create_glucose_transformer()`.

---

### 2. Temporal Hopfield Memory (New Node Type)

**Promise: High | Effort: Medium**

Create a `TemporalHopfieldNode` that operates across the **time axis** instead of the feature axis. The current `StorkeyHopfield` operates on the last dimension (D = embed_dim). For glucose, the temporal patterns (shape of a post-meal spike, dawn phenomenon trajectory) are what should be stored as attractors.

```
z_latent shape: (batch, seq_len, embed_dim)
Current Hopfield W: (embed_dim, embed_dim)  — per-feature attractors
Temporal Hopfield W: (seq_len, seq_len)     — per-timestep attractors
```

**Why it could work**: The fundamental glucose patterns are temporal shapes, not feature-space clusters. A post-meal glucose response always follows a similar trajectory — that's exactly an attractor pattern. During PC inference, noisy or partial input sequences get pulled toward the nearest learned temporal archetype.

**Design sketch**:
- Transpose input to `(batch, embed_dim, seq_len)`, apply Hopfield W on the last axis, transpose back.
- Or use `jnp.einsum('bse,tt->bte', ...)` to apply `W_temporal` on the time axis directly.
- Same energy formulation: `E_hop = (1/2T) z^T (W² - W) z` where T = seq_len.
- Optionally combine both: one W on time axis + one W on feature axis (separable Hopfield).

**Caveat**: seq_len can be 64–128, giving W_temporal shape (64, 64) to (128, 128). This is manageable but larger than the embed_dim=32 case.

---

### 3. Multi-Scale Hopfield

**Promise: Medium–High | Effort: Medium**

The glucose model already uses multi-scale attention (DS=1, DS=2, DS=4). Add Hopfield memory at each scale:

```
Embed → Hopfield_fine(seq_len)
      + Hopfield_medium(seq_len/2)
      + Hopfield_coarse(seq_len/4)
      → fuse → MHA...
```

**Why it could work**: Different glucose patterns operate at different timescales:
- Rapid spikes: 5–15 min (fine scale)
- Meal responses: 30–90 min (medium scale)
- Circadian rhythm: hours (coarse scale)

Each Hopfield layer at a different scale stores attractors at its natural resolution. This mirrors the multi-scale attention design already in the model.

**Design**: Use the existing `_avg_pool_1d` and `_upsample_nearest_1d` helpers from `glucose_model.py` for downsampling/upsampling. Each scale has its own `StorkeyHopfield` with W of shape `(embed_dim, embed_dim)`.

---

### 4. Hopfield as Output Prior

**Promise: Medium | Effort: Low**

Add a `StorkeyHopfield` node before the `RegressionOutputNode`. This constrains the learned representation to lie near prototypical "forecast shapes" before the final projection to the horizon.

```
[MHA + MLP] × depth → StorkeyHopfield → RegressionOutput
```

**Why it could work**: Prevents unrealistic forecasts. The Hopfield attractors learn that "glucose representations before projection should look like these patterns," preventing wild extrapolations the regression head might produce.

**Alternative placement**: After the regression head, operating directly on the `(horizon,)` output. Hopfield W would be `(horizon, horizon)` = `(12, 12)` — very small, learning 12-step forecast shape prototypes. This is more interpretable: the attractors are directly in glucose-value space.

---

### 5. Modern Hopfield Attention (Ramsauer et al. 2020)

**Promise: High (theoretical) | Effort: Significant**

Replace or augment the softmax attention in `MultiScaleMhaResidualNode` with Modern Hopfield Network attention. The key insight from Ramsauer et al.: standard attention `softmax(QK^T/√d)V` is a single iteration of a Modern Hopfield network. In a PC framework, the inference loop could iterate this energy minimization — effectively running multiple attention iterations per PC step.

**The math**:
- Standard attention: one Hopfield update step.
- Modern Hopfield energy: `E = -β·logsumexp(X·ξ/β) + ½||ξ||² + const`
- PC inference: z_latent converges toward the Hopfield fixed point over inference steps, performing iterated attention.

**Why it could work**: This is where PC could genuinely outperform backprop. Backprop does one attention step (one Hopfield iteration). PC inference could iterate toward the optimal attention pattern — multiple Hopfield iterations for free. This is the unique theoretical advantage of combining PC with Hopfield.

**Implementation sketch**: Create a `ModernHopfieldAttentionNode` where:
- `forward()`: compute standard attention (one Hopfield step) for z_mu.
- Energy: use Modern Hopfield energy `E = -logsumexp(...)` instead of/alongside Gaussian.
- PC inference: z_latent naturally converges toward the Hopfield fixed point over inference steps.

**Why it's hard**:
- Requires a new energy functional (logsumexp-based).
- Careful numerical stability (logsumexp overflow with large inner products).
- The theory is well-established for retrieval/classification but less proven for regression.
- Interaction between Modern Hopfield energy and Gaussian PC energy needs careful balancing.

**References**:
- Ramsauer et al. (2020). "Hopfield Networks is All You Need." arXiv:2008.02217.
- Widrich et al. (2020). "Modern Hopfield Networks and Attention for Immune Repertoire Classification."

---

### 6. Hopfield Cross-Attention with Learned Pattern Bank

**Promise: Medium | Effort: Medium**

Maintain a learned set of K prototype patterns (not from input — just learnable parameters) and use the Hopfield mechanism as cross-attention between the current sequence and the pattern bank.

```
Parameters: P ∈ R^{K × embed_dim}   (K learned glucose prototypes)
Retrieval:  output = softmax(X @ P^T / √d) @ P
```

**Why it could work**: Unlike self-attention (which looks at relationships within the current window), this retrieves from a global memory of glucose patterns. Similar to "memory-augmented" transformers, but with the Hopfield energy making retrieval a natural part of PC inference.

**Design choices**:
- K = 16–64 prototypes (hyperparameter).
- Prototypes are learnable parameters, updated via weight gradients.
- Can be placed after embedding or after the transformer blocks.
- The Hopfield energy on the prototype retrieval gives PC inference a role in sharpening the pattern match.

---

### 7. Dual-Energy Node: Gaussian + Hopfield in Existing MHA

**Promise: Medium | Effort: Low**

Instead of creating a new node type, add `accumulate_hopfield_energy()` to the existing `MultiScaleMhaResidualNode`. Add a `(embed_dim, embed_dim)` weight matrix W_hop and accumulate the Hopfield energy on top of the standard Gaussian PC energy.

```python
# In MultiScaleMhaResidualNode.forward():
state = node_class.energy_functional(state, node_info)       # E_pc (existing)
state = StorkeyHopfield.accumulate_hopfield_energy(state, W_hop, strength)  # E_hop (new)
```

**Why it could work**: Minimal code change. The attention node already operates in the right shape. The Hopfield energy acts as a regularizer on attention output representations, encouraging them to cluster near learned patterns.

**Trade-off**: Conceptually less clean than a separate Hopfield node — mixes two different concerns in one node. But fastest to try.

---

## Recommended Sequencing

| Phase | Idea | Goal |
|-------|------|------|
| 1 | **Idea 1** — Hopfield after embedding | Lowest-risk test: does Hopfield help glucose at all? |
| 2 | **Idea 4** — Hopfield as output prior | Complementary placement — if input denoising helps, does output regularization too? |
| 3 | **Idea 2** — Temporal Hopfield | Most theoretically motivated for time series; needs new node |
| 4 | **Idea 5** — Modern Hopfield attention | Most ambitious; where PC could genuinely beat backprop |

### Central hypothesis to test

**Does Hopfield energy give the PC inference loop something productive to do, closing the ~2 mg/dL gap with backprop?**

The current gap suggests PC inference is wasting effort — the latent updates during inference aren't improving the prediction. Hopfield energy redirects that effort toward denoising (idea 1), temporal pattern completion (idea 2), or iterated attention (idea 5).

### Experimental controls

For each idea, run three comparisons:
1. **PC + Hopfield** vs **PC baseline** — does Hopfield improve PC?
2. **PC + Hopfield** vs **Backprop baseline** — does it close the gap?
3. **Backprop + Hopfield** vs **Backprop baseline** — is the benefit PC-specific or general?

If Hopfield helps both PC and backprop equally, the benefit is from the extra parameters/regularization, not from the PC-Hopfield synergy. If it helps PC more (or only PC), that confirms the hypothesis.

### Metrics

- Primary: test MAE (mg/dL) — must beat backprop baseline of ~19 mg/dL.
- Secondary: convergence speed (epochs to best), training stability (variance across seeds).
- Diagnostic: per-node energy decomposition during inference (E_pc vs E_hop) to verify Hopfield is active.
