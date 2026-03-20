# Bayesian Predictive Coding Transformer — Technical Documentation

## 1. The Big Picture: What Problem Does This Solve?

Standard transformers have **no built-in notion of confidence**. Every token representation is treated with equal certainty. When a model encounters ambiguous input (a word that could mean two things, a noisy signal), it processes it with the same computational weight as a perfectly clear input. This leads to:

- Overconfident predictions on ambiguous data
- No mechanism to "flag" unreliable representations to downstream layers
- No way to suppress noise propagation through the network

This implementation adds **per-token uncertainty tracking** to a transformer built on **predictive coding** (PC) — a biologically-inspired alternative to backpropagation where learning happens through local prediction errors rather than a global backward pass.

## 2. The Three Pillars

The system stands on three distinct ideas, each independently motivated:

### Pillar 1: Predictive Coding (the learning framework)

Instead of backpropagation, each node in the graph:
1. **Predicts** what its output should be (`z_mu`) based on its inputs
2. **Compares** that prediction to its current latent belief (`z_latent`)
3. **Computes error**: `error = z_latent - z_mu`
4. **Minimizes energy**: `E = (1/2) * precision * error^2` (Gaussian energy)

During **inference**, latent states are iteratively updated via gradient descent on this energy:
```
z_latent -= eta * dE/dz_latent
```

During **learning**, weight gradients are computed locally at each node:
```
dE/dW = jax.grad(forward, argnums=params)
```

This means **no global backward pass** — each node learns from its own prediction error, using only locally available information. This is the Hebbian learning principle.

 How Variance Works in This System

  What It Represents

  Each token at each layer carries a single scalar: "how much is the model struggling to predict this token's
   representation?" It's not a learned parameter — it's a running measurement of prediction error magnitude. 

  How It's Calculated — Step by Step

  At the start of inference, every token at every layer gets the same initial variance:

  variance = 0.1  # for all tokens, all layers

  Then, on every inference step, after the PC forward pass runs through the graph:

  Step 1: Read the MLP node's prediction error for this layer. The error is the difference between what the  
  node "believes" its output should be (z_latent) and what the computation actually produced (z_mu):

  error = state.nodes["bay_mlp_0"].error   # shape: (batch, seq, embed_dim)

  This is a full vector per token — 64 dimensions in the demo config.

  Step 2: Collapse it to a single scalar per token by averaging the squared error across the embedding       
  dimension:

  error_sq = mean(error^2, axis=-1, keepdims=True)   # shape: (batch, seq, 1)

  Squaring ensures the sign doesn't matter (positive and negative errors contribute equally). Averaging over 
  embed_dim gives one number summarizing "how wrong was the prediction for this token overall."

  Step 3: Smooth it with an exponential moving average:

  new_var = 0.9 * old_var + 0.1 * error_sq

  This means:
  - The variance at step t is 90% determined by its history and 10% by the latest error
  - A single spike in error won't cause a dramatic jump
  - But if errors are consistently high at a position, variance will steadily climb
  - If errors drop, variance will gradually decay

  Concretely, after k steps with constant error_sq = e, the variance converges to:

  var_converged = e    (the geometric series 0.1 * e * sum(0.9^i) = e)

  So the EMA converges to the mean squared error itself — it's an online estimator.

  Step 4: Clip to safety bounds:

  new_var = clip(new_var, 1e-4, 10.0)

  This prevents two failure modes:
  - var = 0 would cause 1/var = inf in the certainty gate and 1/sqrt(var) = inf in attention
  - var = 1000 from a transient spike would effectively freeze the token

  How Variance Flows Back Into Computation

  The variance value is injected into the graph through clamped IdentityNodes, which feed into both the      
  attention and MLP nodes via their "var" slot. This creates a feedback loop:

  High prediction error at position t
          |
          v
  Variance at position t increases
          |
          +---> Attention: scores involving token t are dampened
          |     (other tokens pay less attention to t, t pays less attention to others)
          |
          +---> MLP: update to token t is scaled down
                (token stays closer to its residual input)
          |
          v
  On next inference step, token t has been processed more conservatively
          |
          v
  If the underlying signal is truly noisy: error stays high -> variance stays high (correct!)
  If the error was transient: error drops -> variance decays -> normal processing resumes

  What Value It Adds — Three Concrete Benefits

  1. Noise suppression in attention

  Without uncertainty: if token 7 has a noisy representation (because its target was random), it still       
  participates fully in attention. Other tokens attend to it, incorporating noise into their own
  representations. This noise propagates through layers.

  With uncertainty: token 7's high variance means sqrt(var_7 + var_k) is large for every pair involving token
   7. Its attention scores are dampened after softmax, so other tokens effectively ignore it. The noise is   
  contained.

  2. Conservative processing of ambiguous tokens

  Without uncertainty: the MLP applies the same transformation magnitude to every token. An uncertain token  
  gets transformed just as aggressively as a confident one, potentially amplifying errors.

  With uncertainty: the certainty gate sigmoid(1/var) scales down the MLP's contribution. For var=5.0, only  
  55% of the MLP update is applied. The residual connection dominates, keeping the token close to its        
  previous-layer representation rather than risking a large incorrect transformation.

  3. Interpretable confidence signal

  After inference, you can inspect the variance at each position and layer. This tells you:
  - Which tokens the model found hardest to predict
  - Which layers had the most difficulty
  - Whether the model's uncertainty correlates with actual ambiguity in the data

  In the synthetic task, this should manifest as noisy positions (3, 7, 11, 15, ...) having visibly higher   
  variance than clean positions — the model knows it can't predict those positions reliably.

  Why This Specific Approach vs Alternatives

  Why not learn variance as a parameter? If variance were a learnable weight, the optimizer could push it    
  toward whatever value minimizes loss fastest — likely collapsing it to near-zero (overconfidence) or using 
  it as a general-purpose scaling factor unrelated to actual uncertainty. The EMA approach forces variance to
   directly reflect prediction difficulty.

  Why not compute variance inside the forward pass? The PC framework's energy function is E = 0.5 * error^2. 
  If variance were computed inside forward(), it would become part of the energy landscape and affect how    
  latent states are updated. This would couple the uncertainty estimate to the inference dynamics in
  unpredictable ways. Keeping it external means the energy minimization operates on a clean landscape, and   
  variance is a pure diagnostic that modulates but doesn't distort the optimization.

  Why per-token and not per-dimension? A (batch, seq, embed_dim) variance would give richer information —    
  maybe dimension 17 is uncertain but dimension 42 is confident. But it would also be 64x more expensive to  
  track, harder to inject through the graph (would need full-dim variance nodes), and would complicate the   
  attention modulation (you'd need to decide how to combine 64 variance dimensions into a single attention   
  score modifier). The per-token scalar is the simplest thing that captures the core intuition: "is this     
  token's representation reliable?"


### Pillar 2: Transformer Architecture (the computation)

The model follows a standard transformer design:

```
Token IDs -> Embedding -> [Attention + MLP] x depth -> Vocab Projection
```

With modern architectural choices:
- **Pre-LayerNorm** (normalize before attention/MLP, not after — more stable training)
- **RoPE** (Rotary Position Embeddings — encodes position by rotating Q/K vectors, no learned position embeddings needed, generalizes to unseen sequence lengths)
- **Causal masking** (token at position t can only attend to positions <= t)
- **GELU activation** in the MLP (smoother than ReLU, standard in modern transformers)
- **Residual connections** (output = input + transformation)

### Pillar 3: Uncertainty Modulation (the novel addition)

Each layer gets a **variance node** — a scalar per token position that represents "how uncertain is this token's representation." This variance modulates computation in two specific places:

- **In attention** — dampens interactions between uncertain tokens
- **In the MLP** — gates how much the MLP is allowed to modify uncertain tokens

The variance itself is **not learned by gradient descent**. It's tracked externally via an exponential moving average of squared prediction errors — a running estimate of "how well is the model predicting at this position."

## 3. The Novel Components in Detail

### 3.1 BayesianAttentionNode — Uncertainty-Modulated Attention

This is the core innovation. A standard attention score is:

```
score(q, k) = (Q . K^T) / sqrt(d_head)
```

The Bayesian version adds a second scaling factor:

```
score(q, k) = (Q . K^T) / sqrt(d_head) / sqrt(var_q + var_k + eps)
```

**What this does mathematically:**

The `var_q + var_k` term is the **combined uncertainty** of the query and key tokens. Dividing by its square root has these effects:

| var_q | var_k | combined | sqrt | Effect on score |
|-------|-------|----------|------|-----------------|
| Low   | Low   | Small    | ~1   | Score unchanged (confident pair) |
| High  | Low   | Medium   | >1   | Score reduced (uncertain query) |
| Low   | High  | Medium   | >1   | Score reduced (uncertain key) |
| High  | High  | Large    | >>1  | Score strongly dampened |

After softmax, this means **uncertain tokens contribute less to the attention-weighted sum**. The model naturally "pays less attention" to tokens it's unsure about.

**Design choice: why additive variance, not multiplicative?**

Using `var_q + var_k` (additive) rather than `var_q * var_k` (multiplicative) means that **either** token being uncertain is enough to dampen the interaction. With multiplication, two slightly uncertain tokens would barely dampen each other (0.3 * 0.3 = 0.09), which doesn't match the desired behavior.

**Design choice: why token-level, not per-head or per-dimension?**

The variance is `(batch, seq, 1)` — one scalar per token, shared across all heads and embedding dimensions. This is a deliberate simplification:
- It keeps the variance tracking cheap (seq scalars per layer vs seq * heads * dim)
- It broadcasts cleanly across the attention score matrix
- It captures the intuition that uncertainty is about "this token's position" rather than "this attention head's view of this token"

**Implementation detail: the broadcast pattern**

```python
var_flat = var[..., 0]                    # (batch, seq)
var_q = var_flat[:, None, :, None]        # (batch, 1, seq_q, 1)
var_k = var_flat[:, None, None, :]        # (batch, 1, 1, seq_k)
combined_var = var_q + var_k              # (batch, 1, seq_q, seq_k)
```

The `None` dimensions allow broadcasting across heads (dim 1) and creating the full `(seq_q, seq_k)` pairwise interaction matrix. This is applied uniformly across all attention heads.

### 3.2 BayesianMLPNode — Certainty-Gated Feedforward

After attention, the MLP applies a two-layer feedforward transformation. The Bayesian version gates the output:

```python
delta = W2 @ GELU(W1 @ LayerNorm(x) + b1) + b2    # standard MLP output
certainty = sigmoid(1.0 / (var + eps))               # gate: [0, 1]
delta = delta * certainty                            # scale by certainty
z_mu = x + delta                                     # residual
```

**How the certainty gate works:**

The function `sigmoid(1/var)` maps variance to a gate value:

| var (uncertainty) | 1/var | sigmoid(1/var) | Meaning |
|-------------------|-------|----------------|---------|
| 0.001 (very confident) | 1000 | ~1.0 | Full MLP update applied |
| 0.1 (moderate) | 10 | ~0.99995 | Essentially full |
| 1.0 (uncertain) | 1 | 0.73 | 73% of MLP update |
| 5.0 (very uncertain) | 0.2 | 0.55 | About half |
| 10.0 (max) | 0.1 | 0.52 | Just over half |

**Why this specific gating function?**

1. **Sigmoid** ensures the gate is always in [0, 1] — you never reverse or amplify the update
2. **1/var** creates a monotonically decreasing relationship with uncertainty
3. The gate never fully closes (sigmoid is always > 0) — even very uncertain tokens still get *some* update, they're not frozen
4. The transition is smooth, so JAX's autodiff can compute gradients through it for the PC inference

**Why gate the MLP but modulate scores in attention?**

These are two different mechanisms solving two different problems:
- **Attention** is about *which tokens to combine* — modulating scores before softmax changes the attention distribution
- **MLP** is about *how much to transform* — gating the output controls the magnitude of change

You can't meaningfully "gate" attention scores the same way (they go through softmax and must sum to 1). And you can't "modulate" an MLP output with an additive factor the way you can with attention scores.

### 3.3 External Variance Tracking — The EMA System

The variance is NOT a learnable parameter. It's computed externally in the inference loop:

```python
for t in range(infer_steps):
    # Run one PC inference step
    state = inference_step(...)

    # Update variance from prediction errors
    error = state.nodes[mlp_node].error         # (batch, seq, embed_dim)
    error_sq = mean(error^2, axis=-1)            # (batch, seq, 1) - avg over embed_dim
    new_var = 0.9 * old_var + 0.1 * error_sq     # EMA
    new_var = clip(new_var, 1e-4, 10.0)           # safety bounds
```

**Why EMA and not just the raw squared error?**

Raw squared error is noisy — it can spike on a single inference step due to random initialization or transient dynamics. The EMA (Exponential Moving Average) with `decay=0.9` smooths this out:
- 90% weight on the previous estimate (stability)
- 10% weight on the new observation (responsiveness)
- After ~20 steps, the variance reflects a reliable average of recent errors

**Why track variance from the MLP node's error, not the attention node's?**

The MLP is the last computation in each layer (after attention). Its prediction error reflects the **cumulative difficulty** of processing that token through both the attention and MLP stages. If attention struggled with a token, that difficulty flows into the MLP error.

**Why clip to [1e-4, 10.0]?**

- **Lower bound (1e-4)**: Prevents division by zero in `1/var` (certainty gate) and `sqrt(var)` (attention modulation). Without this, a perfectly predicted token would get `var=0`, causing `1/0 = inf`.
- **Upper bound (10.0)**: Prevents variance from exploding due to large transient errors during early inference steps. Without this, a single large error could push variance so high that the token becomes effectively frozen.

### 3.4 Variance Injection via Clamped Source Nodes

The variance enters the computation graph through `IdentityNode` instances:

```python
var_node = IdentityNode(shape=(seq_len, 1), name=f"var_{i}")
```

These nodes:
- Have **no learnable parameters** (identity pass-through)
- Are **clamped** externally — the inference loop overwrites their `z_latent` with the current variance values each step
- Feed into both the attention and MLP nodes of their layer via the `"var"` slot

**Why clamped nodes instead of computing variance inside the forward pass?**

1. **Separation of concerns**: Variance tracking is a meta-process that sits outside the PC energy minimization. Putting it inside `forward()` would mix it with the energy computation and break the local learning rules.
2. **Flexibility**: The variance schedule (EMA decay, clipping, initialization) can be changed without modifying the node implementations.
3. **Compatibility with fabricPC**: The framework expects nodes to have clean `forward() -> (energy, state)` signatures. Injecting external state through clamped nodes is the idiomatic pattern.

### 3.5 Decaying Inference Learning Rate

```python
eta_t = eta_infer * (0.98 ** t)
```

At step 0: `eta_t = 0.02`. At step 50: `eta_t = 0.007`. At step 100: `eta_t = 0.003`.

**Why?** In PC inference, the latent states are converging toward an energy minimum. Early steps should take large moves (to make progress), later steps should take small moves (to settle precisely). Without decay, the system can oscillate around the minimum indefinitely.

This is analogous to learning rate scheduling in standard training, but applied to the **inference** dynamics rather than the weight updates.

## 4. The Complete Data Flow

Here's one complete iteration of the inference loop, traced through all components:

```
Step t of inference:

1. INJECT VARIANCE
   For each layer i:
     var_i.z_latent <- current_variance[i]    # (batch, seq, 1)

2. CLAMP INPUTS
   input_ids.z_latent <- token_indices         # (batch, seq)
   output.z_latent    <- target_one_hot        # (batch, seq, vocab)
   var_i.z_latent     <- current_variance[i]   # prevent inference from modifying

3. PC INFERENCE STEP (InferenceBase.inference_step)
   a. Zero all latent gradients
   b. Forward pass through graph in topological order:

      input_ids -> embed:
        z_mu = embedding_lookup(input_ids.z_latent)
        error = embed.z_latent - z_mu
        energy = 0.5 * error^2

      [embed, var_0] -> bay_attn_0:
        Q, K, V = project(LayerNorm(embed.z_latent))
        Q, K = apply_rope(Q, K)
        scores = (Q.K^T / sqrt(d)) / sqrt(var_q + var_k + eps)    <- UNCERTAINTY MODULATION
        attn = softmax(causal_mask(scores))
        z_mu = embed.z_latent + W_o @ (attn @ V)                  <- RESIDUAL
        error = bay_attn_0.z_latent - z_mu
        energy = 0.5 * error^2

      [bay_attn_0, var_0] -> bay_mlp_0:
        delta = W2 @ GELU(W1 @ LayerNorm(bay_attn_0.z_latent))
        delta = delta * sigmoid(1/var)                              <- CERTAINTY GATE
        z_mu = bay_attn_0.z_latent + delta                         <- RESIDUAL
        error = bay_mlp_0.z_latent - z_mu
        energy = 0.5 * error^2

      ... (repeat for layers 1, 2) ...

      bay_mlp_2 -> output:
        z_mu = softmax(W_proj @ bay_mlp_2.z_latent)
        error = output.z_latent - z_mu                              <- compare to target
        energy = KL_divergence(output.z_latent, z_mu)

   c. Accumulate gradients: each node sends dE/dx upstream to its sources
   d. Update latents: z_latent -= eta_t * accumulated_grad (except clamped nodes)

4. UPDATE VARIANCE (external, not part of PC)
   For each layer i:
     error = bay_mlp_i.error                   # (batch, seq, embed_dim)
     error_sq = mean(error^2, axis=embed_dim)  # (batch, seq, 1)
     variance[i] = 0.9 * variance[i] + 0.1 * error_sq
     variance[i] = clip(variance[i], 1e-4, 10.0)

5. RECORD DIAGNOSTICS
   total_energy = sum of all non-source node energies
   mean_variance per layer
```

## 5. Training: How Weights Are Updated

Training wraps the inference loop:

```
For each batch:
  1. Initialize graph state (feedforward propagation from input)
  2. Run full Bayesian inference (100 steps) -> converged state
  3. Compute local weight gradients at each node:
     dE/dW = jax.grad(node.forward, argnums=params)
     This is the Hebbian learning rule - each node's weight gradient
     depends only on its own prediction error and its inputs.
  4. Apply Adam optimizer to update weights
```

**Key difference from backpropagation**: There is no global backward pass through the entire graph. Each node independently computes `dE/dW` using only:
- Its own error signal (`z_latent - z_mu`)
- Its own inputs (from upstream nodes' `z_latent`)
- Its own parameters

JAX's `value_and_grad` is used to compute these local gradients efficiently, but the gradient graph only spans a single node, not the full network.

## 6. The Synthetic Evaluation Task

The task is designed to have a clear signal for testing uncertainty:

```
Input:  [s, s+1, s+2, s+3, s+4, s+5, ...]  (mod vocab_size)
Target: [s+1, s+2, s+3, RAND, s+5, s+6, ..., s+7, RAND, ...]
```

- Most positions have a **deterministic** target: `next = current + 1`
- Positions 3, 7, 11, 15, 19, 23, 27, 31 have **random** targets

**What we expect to see:**
- Clean positions (blue in the plot): low variance — the model learns the "+1" rule easily
- Noisy positions (red in the plot): high variance — the targets are random, so prediction error stays high, driving up variance

This is the "can the model know what it doesn't know?" test.

## 7. Architecture Diagram

```
                    input_ids (clamped)
                         |
                         v
                    EmbeddingNode
                         |
          +--------------+--------------+
          |              |              |
       var_0          var_1          var_2
      (clamped)      (clamped)      (clamped)
       |    |         |    |         |    |
       v    v         v    v         v    v
    BayAttn_0      BayAttn_1      BayAttn_2
       |              |              |
       v              v              v
    BayMLP_0  -->  BayMLP_1  -->  BayMLP_2
                                     |
                                     v
                             VocabProjectionNode
                                  (output)
```

Each `var_i` node feeds into **both** the attention and MLP of its layer.
Edges between layers carry the token representations `(batch, seq, embed_dim)`.
Variance nodes carry `(batch, seq, 1)` — one scalar per token.

## 8. Summary of All Design Decisions

| Tweak | Where | Why |
|-------|-------|-----|
| Additive variance in attention: `sqrt(var_q + var_k)` | `BayesianAttentionNode.forward` | Either token being uncertain dampens the interaction |
| Sigmoid certainty gate: `sigmoid(1/var)` | `BayesianMLPNode.forward` | Smooth, bounded [0,1] gate that never fully closes |
| External EMA variance tracking | `run_bayesian_inference` | Keeps variance estimation separate from energy minimization |
| Clamped IdentityNodes for variance | `create_bayesian_pc_transformer` | Injects external state without breaking PC graph semantics |
| Variance from MLP error (not attention) | `run_bayesian_inference` | MLP error captures cumulative per-layer difficulty |
| Decaying inference LR: `0.98^t` | `run_bayesian_inference` | Prevents oscillation, ensures smooth convergence |
| Structured noise at fixed positions | `generate_synthetic_data` | Enables clean visualization without batch-averaging washing out the signal |
| Per-sample (not batch-averaged) variance plots | `plot_uncertainty_heatmap` | Shows actual position-level uncertainty differentiation |
| Pre-norm + RoPE + causal masking | `BayesianAttentionNode` | Modern transformer best practices for stable training |
| Variance clipping [1e-4, 10.0] | `run_bayesian_inference` | Prevents numerical instability (div-by-zero, explosion) |

## 9. Honest Limitations

This implementation is labeled "heuristic" rather than "formally Bayesian" because:

1. **No variational objective**: The variance is tracked via EMA, not optimized through an ELBO. There's no KL divergence term regularizing the uncertainty estimates.
2. **Token-level only**: One scalar variance per token — no per-head, per-dimension, or weight-level uncertainty.
3. **Point-estimate weights**: The weights themselves are deterministic. A fully Bayesian approach would maintain distributions over weights.
4. **No calibration guarantee**: The variance tracks prediction error magnitude, but there's no mechanism ensuring the uncertainty is *calibrated* (i.e., that a reported 70% confidence actually corresponds to being correct 70% of the time).

The natural next step would be **precision-weighted predictive coding**, where precision (1/variance) becomes part of the energy function itself: `E = (1/2) * precision * error^2 + (1/2) * log(1/precision)`. This integrates variance into the PC framework's energy minimization rather than tracking it externally.
