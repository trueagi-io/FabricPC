# Glucose master Optuna report

Generated: `2026-07-24T09:43:16.911444+00:00`

One page covering every major study category (transformer phases, epoch Hyperband, Hopfield, confirms).

## Schema & theory

### Hopfield architecture variants

All variants share the same PC transformer backbone. Placement of associative memory differs:

- **baseline** — no Hopfield node (pure transformer control)
- **embed-storkey** — Storkey Hopfield after embedding (early recall)
- **forecast-storkey** — Storkey Hopfield before the forecast head (late recall)
- **projection** — lightweight linear memory before the forecast head

PC inference runs at every node: predict `z_mu`, error `z - z_mu`, inner SGD for `infer_steps` at step size `eta_infer`, outer Adam at `lr`.

### Hyperparameter glossary

| Parameter | One-line meaning |
|-----------|------------------|
| `seq_len` | How many recent 5-min CGM readings the model sees. |
| `depth` | How many transformer blocks are stacked (shallowness). |
| `num_heads` | Parallel attention viewpoints in each block. |
| `variant` | Architecture family Optuna chooses (baseline / embed-storkey / forecast-storkey / projection). |
| `hopfield_strength` | How strongly Hopfield memory mixes into activations. |
| `lr` | Outer Adam learning rate for weight updates. |
| `eta_infer` | Inner PC step size for refining latent beliefs. |
| `infer_steps` | Inner PC iterations per forward pass. |
| `max_infer_norm` | Clip on PC latent update size. |
| `grad_clip` | Clip on outer weight gradients. |
| `lr_decay_epochs` | When cosine LR decay begins. |
| `weight_init_std` | Scale of random initial weights. |
| `weight_decay` | L2 penalty that discourages huge weights. |
| `readout` | How the sequence is turned into one glucose forecast. |
| `seed_offset` | Seed nudge so similar configs can still differ. |
| `energy` | How PC nodes score prediction error (Gaussian vs Huber). |
| `huber_delta` | Threshold where Huber switches from quadratic to linear. |
| `ipc` | Update latents layer-by-layer (incremental PC) vs all at once. |
| `infer_optimizer` | Optimiser inside the PC loop: SGD or Adam. |

#### Deeper explanations

##### `seq_len`

- **What it is**: History window length fed into the network.
- **Why you care**: Too short misses delayed glucose effects; too long adds noise and cost.
- **How changes show up**: 64 ≈ 5.3 hours of context; longer is not automatically better.

##### `depth`

- **What it is**: Number of attention+MLP stages stacked.
- **Why you care**: Depth is capacity. Shallow nets are simpler and often better on small CGM sets.
- **How changes show up**: depth=1 is a short assembly line; deeper stacks can overfit or train slower.

##### `num_heads`

- **What it is**: Multi-head attention count.
- **Why you care**: More heads can specialise, but need enough width/data.
- **How changes show up**: 1 head is common and stable on small models.

##### `variant`

- **What it is**: Which graph wiring to train: where Hopfield memory sits, or baseline (no Hopfield). This is an Optuna categorical choice — each trial gets one architecture from the search space.
- **Why you care**: Placement changes when associative recall can influence features vs the final forecast. Optuna explores variants; it is not fixed by hand per trial.
- **How changes show up**: baseline = pure transformer control. embed-* recalls early; forecast-* recalls late; projection is a lighter linear memory.

##### `hopfield_strength`

- **What it is**: Fixed scale (e.g. 0.5–2.0) or 'learnable'.
- **Why you care**: Too strong can overwrite useful transformer features; too weak is a no-op.
- **How changes show up**: learnable lets training pick the mix; fixed values are easier to compare across trials.

##### `lr`

- **What it is**: Step size for updating model weights.
- **Why you care**: Too high diverges; too low never improves in budget.
- **How changes show up**: Mid-range ~1e-3–4e-3 often works with champion-like PC settings.

##### `eta_infer`

- **What it is**: Learning rate of the PC inference loop.
- **Why you care**: Separate from weight LR — controls how hard latents correct prediction error.
- **How changes show up**: Around 1e-5–2.5e-5 was a healthy band in transformer PC runs.

##### `infer_steps`

- **What it is**: How many times latents are refined before forecasting.
- **Why you care**: More steps → tighter energy, more compute.
- **How changes show up**: Low teens (12–18) are typical; doubling rarely helps if η is wrong.

##### `max_infer_norm`

- **What it is**: Max norm for inner-loop updates.
- **Why you care**: Prevents exploding activations on sharp glucose swings.
- **How changes show up**: Lower = safer/slower settle; higher = freer but riskier.

##### `grad_clip`

- **What it is**: Global grad clip for Adam.
- **Why you care**: Stops rare huge gradients from wrecking a run.
- **How changes show up**: 0.5–1.0 are common stable choices.

##### `lr_decay_epochs`

- **What it is**: Epoch index that starts annealing LR.
- **Why you care**: Balances exploration early vs fine-tuning later.
- **How changes show up**: Later decay keeps LR high longer.

##### `weight_init_std`

- **What it is**: Normal init standard deviation.
- **Why you care**: Interacts with PC dynamics and depth.
- **How changes show up**: Smaller often safer with PC; larger can help or explode.

##### `weight_decay`

- **What it is**: Weight decay regularisation strength.
- **Why you care**: On small datasets, unconstrained weights memorise noise.
- **How changes show up**: Higher → stronger regularisation (can underfit). Zero → freer fit.

##### `readout`

- **What it is**: Regression head mode: flatten / mean_pool / last.
- **Why you care**: Maps a sequence of vectors to a single 60-min-ahead number.
- **How changes show up**: flatten often best here; mean_pool / last are lighter heads.

##### `seed_offset`

- **What it is**: Added to the base random seed.
- **Why you care**: PC runs can be seed-sensitive.
- **How changes show up**: Document the winning seed for fair replay.

##### `energy`

- **What it is**: Energy functional used inside PC nodes.
- **Why you care**: Gaussian punishes large errors hard; Huber is more robust to spikes.
- **How changes show up**: Huber can help when a few wild glucose points would dominate.

##### `huber_delta`

- **What it is**: Delta parameter for Huber energy (only if energy=huber).
- **Why you care**: Controls when an error is treated as an outlier.
- **How changes show up**: Smaller delta → more robust; larger → closer to plain MSE.

##### `ipc`

- **What it is**: Incremental Predictive Coding flag.
- **Why you care**: Layerwise updates can improve convergence on deeper stacks.
- **How changes show up**: On shallow nets the difference may be small.

##### `infer_optimizer`

- **What it is**: Which optimiser nudges latent activations during inference.
- **Why you care**: SGD is simple/fast; Adam adapts per-coordinate.
- **How changes show up**: Try SGD first; Adam if inference looks under-converged.

## Cross-study overview

| Category | Study | Best MAE | Best trial | Trials |
|----------|-------|---------:|-----------:|-------:|
| 1. Transformer PC | Phase 1 — broad search | 20.3776 | 21 | 34 |
| 1. Transformer PC | Phase 2 — refined | 20.6780 | 21 | 32 |
| 1. Transformer PC | Phase 3 — local | 20.8670 | 21 | 24 |
| 1. Transformer PC | Phase 4 — breakthrough | 19.8760 | 7 | 24 |
| 2. Transformer PC | Epochs v1 (Hyperband) | 19.0963 | 28 | 35 |
| 3. Hopfield PC Optuna | Hopfield v1 (native / early) | 20.3022 | 0 | 8 |
| 3. Hopfield PC Optuna | Hopfield WSL v1 | 20.1605 | 20 | 24 |
| 3. Hopfield PC Optuna | Hopfield WSL v2 | 20.1219 | 12 | 32 |
| 4. Confirmation trains (single-config re | Phase-1 champion confirm | 20.6285 | — | — |
| 4. Confirmation trains (single-config re | Phase-4 breakthrough confirm | 18.7815 | — | — |

**Current master best:** Phase-4 breakthrough confirm — MAE 18.7815 (`runs/glucose_pc_breakthrough_confirm`)


## 1. Transformer PC — update-budget Optuna

Archived update-budget search (phases 1–4). Validation every N optimizer updates; Median/Hyperband-style pruning on update checks.

### Phase 1 — broad search

- Study: `glucose_transformer_pc`
- Run dir: `runs/glucose_tuning`
- Trials: 34 (complete 7, pruned 25, fail 0, running 2)
- Best: trial 21 · MAE 20.3776
- Best params: `seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14`

#### Search space (what Optuna sampled)

| Hyperparameter | Search range |
|----------------|--------------|
| `seq_len` | categorical {64, 128} |
| `depth` | int 1–3 |
| `num_heads` | categorical {1, 2, 4} |
| `lr` | float log 3e-4 – 5e-3 |
| `eta_infer` | float log 1e-5 – 5e-4 |
| `infer_steps` | int 8–24 |
| `max_infer_norm` | categorical {0.5, 1.0, 5.0} |
| `grad_clip` | categorical {0.5, 1.0, 2.0} |
| `weight_init_std` | float log 0.01 – 0.03 |
| `weight_decay` | fixed 0.0 |
| `readout` | fixed flatten |

| Trial | State | Best MAE | Params |
|------:|-------|---------:|--------|
| 21 | COMPLETE | 20.378 | seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14 |
| 20 | COMPLETE | 20.519 | seq_len=64, depth=1, num_heads=4, lr=0.002787, eta_infer=1.261e-05, infer_steps=22 |
| 22 | COMPLETE | 20.684 | seq_len=64, depth=1, num_heads=4, lr=0.002613, eta_infer=1.034e-05, infer_steps=22 |
| 9 | COMPLETE | 21.010 | seq_len=64, depth=1, num_heads=1, lr=0.001943, eta_infer=1.264e-05, infer_steps=16 |
| 33 | PRUNED | 21.364 | seq_len=128, depth=1, num_heads=1, lr=0.004517, eta_infer=1.788e-05, infer_steps=22 |
| 18 | COMPLETE | 21.616 | seq_len=64, depth=2, num_heads=1, lr=0.00494, eta_infer=1.532e-05, infer_steps=16 |
| 23 | PRUNED | 21.665 | seq_len=128, depth=1, num_heads=4, lr=0.004492, eta_infer=1.32e-05, infer_steps=17 |
| 2 | COMPLETE | 22.205 | seq_len=64, depth=1, num_heads=4, lr=0.001246, eta_infer=8.419e-05, infer_steps=19 |
| 7 | PRUNED | 22.702 | seq_len=64, depth=1, num_heads=2, lr=0.001399, eta_infer=2.383e-05, infer_steps=8 |
| 30 | PRUNED | 24.236 | seq_len=64, depth=1, num_heads=1, lr=0.002147, eta_infer=1.148e-05, infer_steps=9 |
| 24 | PRUNED | 24.631 | seq_len=64, depth=1, num_heads=4, lr=0.00269, eta_infer=1.868e-05, infer_steps=24 |
| 25 | PRUNED | 24.882 | seq_len=64, depth=1, num_heads=2, lr=0.002106, eta_infer=1.381e-05, infer_steps=19 |
| 13 | PRUNED | 25.909 | seq_len=64, depth=1, num_heads=4, lr=0.001532, eta_infer=4.016e-05, infer_steps=12 |
| 29 | PRUNED | 25.930 | seq_len=64, depth=1, num_heads=1, lr=0.003303, eta_infer=2.688e-05, infer_steps=8 |
| 17 | PRUNED | 27.090 | seq_len=64, depth=1, num_heads=1, lr=0.001114, eta_infer=2.082e-05, infer_steps=18 |
| 1 | COMPLETE | 27.132 | seq_len=64, depth=1, num_heads=4, lr=0.001316, eta_infer=0.0001633, infer_steps=14 |
| 19 | PRUNED | 30.451 | seq_len=64, depth=1, num_heads=4, lr=0.002285, eta_infer=0.0001349, infer_steps=22 |
| 31 | PRUNED | 34.191 | seq_len=64, depth=1, num_heads=2, lr=0.001424, eta_infer=1.177e-05, infer_steps=20 |
| 4 | PRUNED | 37.966 | seq_len=64, depth=1, num_heads=2, lr=0.001357, eta_infer=0.0001608, infer_steps=16 |
| 26 | PRUNED | 39.812 | seq_len=128, depth=1, num_heads=4, lr=0.001216, eta_infer=1.699e-05, infer_steps=17 |
| 32 | PRUNED | 40.938 | seq_len=64, depth=2, num_heads=4, lr=0.001279, eta_infer=1.284e-05, infer_steps=22 |
| 11 | PRUNED | 41.960 | seq_len=128, depth=1, num_heads=4, lr=0.0005062, eta_infer=0.0001464, infer_steps=19 |
| 16 | PRUNED | 41.968 | seq_len=128, depth=2, num_heads=2, lr=0.001591, eta_infer=0.0001275, infer_steps=20 |
| 3 | PRUNED | 42.683 | seq_len=128, depth=3, num_heads=2, lr=0.0009195, eta_infer=3.779e-05, infer_steps=9 |
| 6 | PRUNED | 43.204 | seq_len=128, depth=3, num_heads=1, lr=0.0006064, eta_infer=9.163e-05, infer_steps=14 |
| 14 | PRUNED | 43.274 | seq_len=64, depth=1, num_heads=1, lr=0.000317, eta_infer=6.905e-05, infer_steps=23 |
| 10 | PRUNED | 43.356 | seq_len=64, depth=2, num_heads=4, lr=0.000459, eta_infer=0.0001008, infer_steps=16 |
| 15 | PRUNED | 43.422 | seq_len=64, depth=1, num_heads=2, lr=0.0003483, eta_infer=0.0002437, infer_steps=24 |
| 5 | PRUNED | 43.654 | seq_len=64, depth=3, num_heads=1, lr=0.0005901, eta_infer=1.985e-05, infer_steps=16 |
| 0 | PRUNED | 43.706 | seq_len=128, depth=2, num_heads=1, lr=0.001481, eta_infer=0.0003616, infer_steps=16 |
| 12 | PRUNED | 44.866 | seq_len=64, depth=3, num_heads=1, lr=0.0004119, eta_infer=1.846e-05, infer_steps=9 |
| 8 | PRUNED | 46.198 | seq_len=128, depth=3, num_heads=2, lr=0.004215, eta_infer=0.0004085, infer_steps=9 |

### Phase 2 — refined

- Study: `glucose_transformer_pc_v2`
- Run dir: `runs/glucose_tuning_pc_v2`
- Trials: 32 (complete 4, pruned 28, fail 0, running 0)
- Best: trial 21 · MAE 20.6780
- Best params: `seq_len=64, depth=1, num_heads=4, lr=0.004202, eta_infer=1.358e-05, infer_steps=14`

#### Search space (what Optuna sampled)

| Hyperparameter | Search range |
|----------------|--------------|
| `seq_len` | categorical {64} |
| `depth` | int 1–2 |
| `num_heads` | categorical {1, 4} |
| `lr` | float log 1e-3 – 5e-3 |
| `eta_infer` | float log 3e-6 – 8e-5 |
| `infer_steps` | int 10–20 |
| `max_infer_norm` | categorical {0.5, 1.0} |
| `grad_clip` | categorical {0.5, 1.0} |
| `weight_init_std` | float log 0.012 – 0.025 |
| `weight_decay` | float log 1e-6 – 1e-3 |
| `readout` | fixed flatten |

| Trial | State | Best MAE | Params |
|------:|-------|---------:|--------|
| 21 | COMPLETE | 20.678 | seq_len=64, depth=1, num_heads=4, lr=0.004202, eta_infer=1.358e-05, infer_steps=14 |
| 19 | COMPLETE | 21.220 | seq_len=64, depth=1, num_heads=4, lr=0.002936, eta_infer=3.401e-05, infer_steps=17 |
| 23 | COMPLETE | 21.410 | seq_len=64, depth=1, num_heads=4, lr=0.0045, eta_infer=2.565e-05, infer_steps=11 |
| 0 | COMPLETE | 22.139 | seq_len=64, depth=1, num_heads=4, lr=0.004234, eta_infer=5.414e-05, infer_steps=12 |
| 31 | PRUNED | 23.393 | seq_len=64, depth=1, num_heads=4, lr=0.004448, eta_infer=1.382e-05, infer_steps=10 |
| 30 | PRUNED | 23.872 | seq_len=64, depth=1, num_heads=4, lr=0.003638, eta_infer=7.817e-05, infer_steps=20 |
| 4 | PRUNED | 23.888 | seq_len=64, depth=1, num_heads=4, lr=0.004141, eta_infer=5.52e-06, infer_steps=20 |
| 18 | PRUNED | 24.027 | seq_len=64, depth=1, num_heads=4, lr=0.003314, eta_infer=5.418e-05, infer_steps=14 |
| 17 | PRUNED | 24.338 | seq_len=64, depth=1, num_heads=4, lr=0.004346, eta_infer=9.021e-06, infer_steps=19 |
| 24 | PRUNED | 24.854 | seq_len=64, depth=1, num_heads=4, lr=0.004622, eta_infer=5.201e-06, infer_steps=12 |
| 16 | PRUNED | 26.816 | seq_len=64, depth=1, num_heads=1, lr=0.00428, eta_infer=6.264e-05, infer_steps=12 |
| 7 | PRUNED | 29.870 | seq_len=64, depth=1, num_heads=4, lr=0.004108, eta_infer=9.438e-06, infer_steps=12 |
| 11 | PRUNED | 30.473 | seq_len=64, depth=1, num_heads=1, lr=0.002533, eta_infer=7.574e-06, infer_steps=18 |
| 28 | PRUNED | 31.068 | seq_len=64, depth=1, num_heads=4, lr=0.003498, eta_infer=9.483e-06, infer_steps=15 |
| 10 | PRUNED | 31.154 | seq_len=64, depth=2, num_heads=4, lr=0.004484, eta_infer=5.232e-06, infer_steps=13 |
| 13 | PRUNED | 31.198 | seq_len=64, depth=2, num_heads=1, lr=0.003547, eta_infer=4.927e-06, infer_steps=12 |
| 20 | PRUNED | 31.282 | seq_len=64, depth=1, num_heads=4, lr=0.003487, eta_infer=6.554e-05, infer_steps=12 |
| 8 | PRUNED | 31.421 | seq_len=64, depth=1, num_heads=4, lr=0.002542, eta_infer=3.016e-06, infer_steps=20 |
| 29 | PRUNED | 31.771 | seq_len=64, depth=1, num_heads=4, lr=0.003143, eta_infer=1.412e-05, infer_steps=10 |
| 9 | PRUNED | 31.895 | seq_len=64, depth=2, num_heads=1, lr=0.004297, eta_infer=2.078e-05, infer_steps=15 |
| 25 | PRUNED | 32.447 | seq_len=64, depth=1, num_heads=4, lr=0.004441, eta_infer=7.25e-05, infer_steps=15 |
| 26 | PRUNED | 32.893 | seq_len=64, depth=1, num_heads=4, lr=0.002738, eta_infer=6.236e-05, infer_steps=12 |
| 22 | PRUNED | 33.257 | seq_len=64, depth=1, num_heads=4, lr=0.001917, eta_infer=3.073e-05, infer_steps=17 |
| 15 | PRUNED | 33.438 | seq_len=64, depth=1, num_heads=1, lr=0.001552, eta_infer=4.246e-05, infer_steps=19 |
| 1 | PRUNED | 33.928 | seq_len=64, depth=1, num_heads=4, lr=0.001795, eta_infer=3.214e-06, infer_steps=19 |
| 5 | PRUNED | 34.268 | seq_len=64, depth=2, num_heads=4, lr=0.002707, eta_infer=7.562e-06, infer_steps=13 |
| 2 | PRUNED | 35.682 | seq_len=64, depth=1, num_heads=4, lr=0.001464, eta_infer=4.64e-06, infer_steps=10 |
| 6 | PRUNED | 36.301 | seq_len=64, depth=2, num_heads=1, lr=0.001935, eta_infer=2.14e-05, infer_steps=11 |
| 3 | PRUNED | 36.305 | seq_len=64, depth=2, num_heads=1, lr=0.002113, eta_infer=5.973e-06, infer_steps=11 |
| 12 | PRUNED | 37.232 | seq_len=64, depth=1, num_heads=1, lr=0.001288, eta_infer=9.054e-06, infer_steps=17 |
| 27 | PRUNED | 37.308 | seq_len=64, depth=1, num_heads=4, lr=0.004096, eta_infer=3.308e-05, infer_steps=17 |
| 14 | PRUNED | 40.314 | seq_len=64, depth=2, num_heads=1, lr=0.001144, eta_infer=6.729e-06, infer_steps=11 |

### Phase 3 — local

- Study: `glucose_transformer_pc_local`
- Run dir: `runs/glucose_tuning_pc_local`
- Trials: 24 (complete 3, pruned 21, fail 0, running 0)
- Best: trial 21 · MAE 20.8670
- Best params: `seq_len=64, depth=1, num_heads=1, lr=0.003946, eta_infer=9.955e-06, infer_steps=15`

#### Search space (what Optuna sampled)

| Hyperparameter | Search range |
|----------------|--------------|
| `seq_len` | categorical {64} |
| `depth` | categorical {1} |
| `num_heads` | categorical {1} |
| `lr` | float log 1.5e-3 – 4.0e-3 |
| `eta_infer` | float log 8e-6 – 3e-5 |
| `infer_steps` | int 12–18 |
| `max_infer_norm` | categorical {0.5, 1.0} |
| `grad_clip` | categorical {0.5, 1.0} |
| `weight_init_std` | float log 0.014 – 0.022 |
| `weight_decay` | fixed 0.0 |
| `readout` | fixed flatten |

| Trial | State | Best MAE | Params |
|------:|-------|---------:|--------|
| 21 | COMPLETE | 20.867 | seq_len=64, depth=1, num_heads=1, lr=0.003946, eta_infer=9.955e-06, infer_steps=15 |
| 1 | COMPLETE | 21.450 | seq_len=64, depth=1, num_heads=1, lr=0.0034, eta_infer=1.1e-05, infer_steps=16 |
| 15 | PRUNED | 23.042 | seq_len=64, depth=1, num_heads=1, lr=0.002798, eta_infer=2.766e-05, infer_steps=17 |
| 0 | COMPLETE | 23.043 | seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14 |
| 23 | PRUNED | 23.259 | seq_len=64, depth=1, num_heads=1, lr=0.003689, eta_infer=1.395e-05, infer_steps=13 |
| 11 | PRUNED | 23.805 | seq_len=64, depth=1, num_heads=1, lr=0.003869, eta_infer=1.578e-05, infer_steps=16 |
| 22 | PRUNED | 23.880 | seq_len=64, depth=1, num_heads=1, lr=0.002972, eta_infer=2.611e-05, infer_steps=18 |
| 19 | PRUNED | 25.278 | seq_len=64, depth=1, num_heads=1, lr=0.003919, eta_infer=8.897e-06, infer_steps=12 |
| 3 | PRUNED | 30.321 | seq_len=64, depth=1, num_heads=1, lr=0.0032, eta_infer=9.5e-06, infer_steps=15 |
| 12 | PRUNED | 30.747 | seq_len=64, depth=1, num_heads=1, lr=0.003646, eta_infer=1.011e-05, infer_steps=17 |
| 18 | PRUNED | 30.911 | seq_len=64, depth=1, num_heads=1, lr=0.002791, eta_infer=1.519e-05, infer_steps=14 |
| 13 | PRUNED | 31.064 | seq_len=64, depth=1, num_heads=1, lr=0.003549, eta_infer=2.234e-05, infer_steps=16 |
| 20 | PRUNED | 31.077 | seq_len=64, depth=1, num_heads=1, lr=0.002735, eta_infer=8.955e-06, infer_steps=18 |
| 16 | PRUNED | 31.078 | seq_len=64, depth=1, num_heads=1, lr=0.003273, eta_infer=1.352e-05, infer_steps=18 |
| 6 | PRUNED | 31.198 | seq_len=64, depth=1, num_heads=1, lr=0.003913, eta_infer=2.197e-05, infer_steps=18 |
| 17 | PRUNED | 31.316 | seq_len=64, depth=1, num_heads=1, lr=0.003599, eta_infer=1.043e-05, infer_steps=17 |
| 4 | PRUNED | 31.453 | seq_len=64, depth=1, num_heads=1, lr=0.002474, eta_infer=1.179e-05, infer_steps=12 |
| 2 | PRUNED | 32.119 | seq_len=64, depth=1, num_heads=1, lr=0.0025, eta_infer=1.6e-05, infer_steps=12 |
| 9 | PRUNED | 32.259 | seq_len=64, depth=1, num_heads=1, lr=0.002128, eta_infer=2.611e-05, infer_steps=15 |
| 8 | PRUNED | 33.008 | seq_len=64, depth=1, num_heads=1, lr=0.001864, eta_infer=2.281e-05, infer_steps=13 |
| 7 | PRUNED | 33.168 | seq_len=64, depth=1, num_heads=1, lr=0.003295, eta_infer=2.36e-05, infer_steps=14 |
| 10 | PRUNED | 33.419 | seq_len=64, depth=1, num_heads=1, lr=0.001685, eta_infer=2.43e-05, infer_steps=14 |
| 14 | PRUNED | 34.239 | seq_len=64, depth=1, num_heads=1, lr=0.001744, eta_infer=1.111e-05, infer_steps=14 |
| 5 | PRUNED | 35.893 | seq_len=64, depth=1, num_heads=1, lr=0.0015, eta_infer=1.833e-05, infer_steps=16 |

### Phase 4 — breakthrough

- Study: `glucose_transformer_pc_breakthrough`
- Run dir: `runs/glucose_tuning_pc_breakthrough`
- Trials: 24 (complete 13, pruned 11, fail 0, running 0)
- Best: trial 7 · MAE 19.8760
- Best params: `seq_len=64, depth=1, num_heads=1, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, readout=flatten`

#### Search space (what Optuna sampled)

| Hyperparameter | Search range |
|----------------|--------------|
| `seq_len` | categorical {64} |
| `depth` | categorical {1} |
| `num_heads` | categorical {1} |
| `lr` | float log 1.8e-3 – 3.8e-3 |
| `eta_infer` | float log 9e-6 – 2.5e-5 |
| `infer_steps` | int 12–18 |
| `max_infer_norm` | categorical {0.5, 1.0} |
| `grad_clip` | categorical {0.5, 1.0} |
| `weight_init_std` | float log 0.014 – 0.021 |
| `weight_decay` | fixed 0.0 |
| `readout` | categorical {flatten, mean_pool, last} |
| `seed_offset` | int 0–40 |

| Trial | State | Best MAE | Params |
|------:|-------|---------:|--------|
| 7 | COMPLETE | 19.876 | seq_len=64, depth=1, num_heads=1, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, readout=flatten |
| 12 | COMPLETE | 20.011 | seq_len=64, depth=1, num_heads=1, lr=0.003533, eta_infer=2.207e-05, infer_steps=12, readout=flatten |
| 14 | COMPLETE | 20.361 | seq_len=64, depth=1, num_heads=1, lr=0.002667, eta_infer=9.998e-06, infer_steps=16, readout=flatten |
| 0 | COMPLETE | 20.388 | seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14, readout=flatten |
| 10 | PRUNED | 20.590 | seq_len=64, depth=1, num_heads=1, lr=0.003594, eta_infer=1.635e-05, infer_steps=12, readout=flatten |
| 15 | PRUNED | 20.862 | seq_len=64, depth=1, num_heads=1, lr=0.003773, eta_infer=2.39e-05, infer_steps=12, readout=flatten |
| 13 | COMPLETE | 20.866 | seq_len=64, depth=1, num_heads=1, lr=0.003681, eta_infer=1.787e-05, infer_steps=15, readout=flatten |
| 9 | PRUNED | 20.908 | seq_len=64, depth=1, num_heads=1, lr=0.003644, eta_infer=1.024e-05, infer_steps=13, readout=flatten |
| 17 | COMPLETE | 21.253 | seq_len=64, depth=1, num_heads=1, lr=0.003647, eta_infer=2.032e-05, infer_steps=13, readout=flatten |
| 6 | PRUNED | 21.414 | seq_len=64, depth=1, num_heads=1, lr=0.002416, eta_infer=1.59e-05, infer_steps=13, readout=flatten |
| 5 | COMPLETE | 21.934 | seq_len=64, depth=1, num_heads=1, lr=0.0035, eta_infer=1.05e-05, infer_steps=15, readout=last |
| 16 | COMPLETE | 21.986 | seq_len=64, depth=1, num_heads=1, lr=0.003424, eta_infer=1.604e-05, infer_steps=12, readout=last |
| 2 | COMPLETE | 22.034 | seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14, readout=last |
| 8 | COMPLETE | 22.629 | seq_len=64, depth=1, num_heads=1, lr=0.003447, eta_infer=1.452e-05, infer_steps=16, readout=flatten |
| 23 | PRUNED | 23.422 | seq_len=64, depth=1, num_heads=1, lr=0.003691, eta_infer=1.836e-05, infer_steps=12, readout=flatten |
| 21 | PRUNED | 23.747 | seq_len=64, depth=1, num_heads=1, lr=0.002834, eta_infer=2.457e-05, infer_steps=13, readout=flatten |
| 18 | PRUNED | 23.831 | seq_len=64, depth=1, num_heads=1, lr=0.002628, eta_infer=2.026e-05, infer_steps=12, readout=flatten |
| 11 | PRUNED | 25.080 | seq_len=64, depth=1, num_heads=1, lr=0.003093, eta_infer=1.044e-05, infer_steps=12, readout=flatten |
| 22 | PRUNED | 26.347 | seq_len=64, depth=1, num_heads=1, lr=0.002151, eta_infer=1.01e-05, infer_steps=18, readout=flatten |
| 4 | COMPLETE | 42.874 | seq_len=64, depth=1, num_heads=1, lr=0.0026, eta_infer=1.5e-05, infer_steps=14, readout=mean_pool |
| 3 | COMPLETE | 42.921 | seq_len=64, depth=1, num_heads=1, lr=0.0032, eta_infer=1.15e-05, infer_steps=16, readout=mean_pool |
| 1 | COMPLETE | 43.022 | seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14, readout=mean_pool |
| 20 | PRUNED | 43.148 | seq_len=64, depth=1, num_heads=1, lr=0.003696, eta_infer=1.744e-05, infer_steps=12, readout=mean_pool |
| 19 | PRUNED | 43.884 | seq_len=64, depth=1, num_heads=1, lr=0.002472, eta_infer=1.964e-05, infer_steps=12, readout=mean_pool |


## 2. Transformer PC — epoch Hyperband Optuna

Default epoch-based tuner (`glucose-transformer-tune`). Full epochs with Hyperband pruning; produced the current PC champion (~19.1 MAE).

### Epochs v1 (Hyperband)

- Study: `glucose_transformer_pc_epochs_v1`
- Run dir: `runs/glucose_tuning_epochs_v1`
- Trials: 35 (complete 13, pruned 18, fail 4, running 0)
- Best: trial 28 · MAE 19.0963
- Best params: `seq_len=64, depth=1, num_heads=4, lr=0.0003367, eta_infer=3.17e-05, infer_steps=21`

#### Search space (what Optuna sampled)

| Hyperparameter | Search range |
|----------------|--------------|
| `seq_len` | categorical {64, 128} |
| `depth` | int 1–3 |
| `num_heads` | categorical {1, 2, 4} |
| `lr` | float log 3e-4 – 5e-3 |
| `eta_infer` | float log 1e-5 – 5e-4 |
| `infer_steps` | int 8–24 |
| `max_infer_norm` | categorical {0.5, 1.0, 5.0} |
| `grad_clip` | categorical {0.5, 1.0, 2.0} |
| `lr_decay_epochs` | categorical {5, 10, 15} |
| `weight_init_std` | float log 0.01 – 0.03 |
| `energy` | categorical {gaussian, huber} |
| `ipc` | categorical {true, false} |
| `infer_optimizer` | categorical {sgd, adam} |
| `huber_delta` | float log 0.1 – 2.0 (if energy=huber) |

| Trial | State | Best MAE | Params |
|------:|-------|---------:|--------|
| 28 | COMPLETE | 19.096 | seq_len=64, depth=1, num_heads=4, lr=0.0003367, eta_infer=3.17e-05, infer_steps=21 |
| 21 | COMPLETE | 19.135 | seq_len=64, depth=1, num_heads=1, lr=0.0005683, eta_infer=4.022e-05, infer_steps=20 |
| 11 | FAIL | 19.187 | seq_len=128, depth=3, num_heads=4, lr=0.002241, eta_infer=2.645e-05, infer_steps=13 |
| 30 | COMPLETE | 19.398 | seq_len=64, depth=1, num_heads=1, lr=0.0005651, eta_infer=3.819e-05, infer_steps=20 |
| 29 | COMPLETE | 19.447 | seq_len=64, depth=2, num_heads=1, lr=0.0004468, eta_infer=2.889e-05, infer_steps=22 |
| 34 | COMPLETE | 19.575 | seq_len=64, depth=1, num_heads=1, lr=0.0005432, eta_infer=5.888e-05, infer_steps=15 |
| 31 | COMPLETE | 19.584 | seq_len=64, depth=1, num_heads=4, lr=0.0003646, eta_infer=5.101e-05, infer_steps=17 |
| 33 | FAIL | 19.598 | seq_len=64, depth=1, num_heads=2, lr=0.000346, eta_infer=2.163e-05, infer_steps=24 |
| 7 | FAIL | 19.633 | seq_len=64, depth=3, num_heads=1, lr=0.000383, eta_infer=1.194e-05, infer_steps=23 |
| 10 | FAIL | 19.824 | seq_len=128, depth=3, num_heads=1, lr=0.0007603, eta_infer=0.0004921, infer_steps=10 |
| 26 | PRUNED | 19.861 | seq_len=64, depth=1, num_heads=1, lr=0.0007214, eta_infer=6.548e-05, infer_steps=19 |
| 15 | COMPLETE | 19.893 | seq_len=64, depth=1, num_heads=1, lr=0.0005182, eta_infer=0.0001051, infer_steps=18 |
| 2 | COMPLETE | 19.926 | seq_len=128, depth=2, num_heads=2, lr=0.001844, eta_infer=3.614e-05, infer_steps=11 |
| 17 | COMPLETE | 19.983 | seq_len=64, depth=2, num_heads=1, lr=0.000752, eta_infer=0.0004862, infer_steps=9 |
| 18 | PRUNED | 20.129 | seq_len=64, depth=1, num_heads=1, lr=0.003081, eta_infer=0.0001338, infer_steps=20 |
| 13 | PRUNED | 20.430 | seq_len=64, depth=1, num_heads=1, lr=0.00309, eta_infer=5.115e-05, infer_steps=14 |
| 3 | PRUNED | 20.543 | seq_len=128, depth=2, num_heads=1, lr=0.0015, eta_infer=0.0002975, infer_steps=20 |
| 5 | PRUNED | 20.644 | seq_len=64, depth=2, num_heads=2, lr=0.0007457, eta_infer=1.462e-05, infer_steps=8 |
| 27 | PRUNED | 20.717 | seq_len=128, depth=2, num_heads=4, lr=0.002092, eta_infer=4.138e-05, infer_steps=14 |
| 25 | PRUNED | 20.786 | seq_len=128, depth=1, num_heads=2, lr=0.001392, eta_infer=1.018e-05, infer_steps=10 |
| 23 | PRUNED | 21.425 | seq_len=128, depth=2, num_heads=2, lr=0.001469, eta_infer=3.322e-05, infer_steps=16 |
| 22 | PRUNED | 22.831 | seq_len=128, depth=2, num_heads=2, lr=0.00218, eta_infer=0.0001265, infer_steps=10 |
| 4 | COMPLETE | 24.548 | seq_len=128, depth=1, num_heads=4, lr=0.0004313, eta_infer=0.0003144, infer_steps=14 |
| 24 | PRUNED | 25.112 | seq_len=64, depth=1, num_heads=1, lr=0.00103, eta_infer=0.0003752, infer_steps=18 |
| 12 | COMPLETE | 28.806 | seq_len=128, depth=1, num_heads=2, lr=0.001115, eta_infer=0.0003493, infer_steps=24 |
| 19 | PRUNED | 29.741 | seq_len=64, depth=1, num_heads=4, lr=0.001053, eta_infer=0.0002164, infer_steps=14 |
| 14 | PRUNED | 30.735 | seq_len=64, depth=3, num_heads=1, lr=0.003033, eta_infer=0.0001108, infer_steps=22 |
| 8 | PRUNED | 32.784 | seq_len=128, depth=1, num_heads=2, lr=0.0008918, eta_infer=0.0001831, infer_steps=17 |
| 32 | COMPLETE | 34.776 | seq_len=64, depth=1, num_heads=1, lr=0.0003216, eta_infer=1.429e-05, infer_steps=17 |
| 16 | COMPLETE | 38.907 | seq_len=64, depth=2, num_heads=2, lr=0.0006891, eta_infer=3.903e-05, infer_steps=24 |
| 9 | PRUNED | 42.036 | seq_len=128, depth=3, num_heads=4, lr=0.0004703, eta_infer=1.522e-05, infer_steps=11 |
| 0 | PRUNED | 45.340 | seq_len=64, depth=3, num_heads=2, lr=0.002956, eta_infer=1.585e-05, infer_steps=23 |
| 1 | PRUNED | 48.485 | seq_len=128, depth=1, num_heads=1, lr=0.002405, eta_infer=3.29e-05, infer_steps=19 |


## 3. Hopfield PC Optuna

Same PC backbone plus Hopfield associative-memory variants (baseline / embed-storkey / forecast-storkey / projection).

### Hopfield v1 (native / early)

- Study: `glucose_hopfield_pc_v1`
- Run dir: `runs/glucose_hopfield_tuning_v1`
- Trials: 8 (complete 0, pruned 3, fail 0, running 3)
- Best: trial 0 · MAE 20.3022
- Best params: `seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12`

#### Search space (what Optuna sampled)

| Hyperparameter | Search range |
|----------------|--------------|
| `seq_len` | categorical {64} |
| `depth` | categorical {1} |
| `num_heads` | categorical {1} |
| `variant` | categorical {baseline, embed-storkey, forecast-storkey, projection} |
| `hopfield_strength` | categorical {0.5, 1.0, 1.5, 2.0, learnable} |
| `lr` | float log ~1.8e-3 – 3.8e-3 (breakthrough band) |
| `eta_infer` | float log ~9e-6 – 2.5e-5 |
| `infer_steps` | int ~12–20 |
| `max_infer_norm` | categorical {0.5, 1.0} |
| `grad_clip` | categorical {0.5, 1.0} |
| `lr_decay_epochs` | categorical {5, 10, 15} |
| `weight_init_std` | float log ~0.013 – 0.022 |
| `seed_offset` | int 0–40 |

| Trial | State | Best MAE | Params |
|------:|-------|---------:|--------|
| 0 | PRUNED | 20.302 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |
| 1 | PRUNED | 35.422 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |

### Hopfield WSL v1

- Study: `glucose_hopfield_pc_wsl_v1`
- Run dir: `runs/glucose_hopfield_tuning_wsl_v1`
- Trials: 24 (complete 16, pruned 8, fail 0, running 0)
- Best: trial 20 · MAE 20.1605
- Best params: `seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002464, eta_infer=1.016e-05, infer_steps=17`

#### Search space (what Optuna sampled)

| Hyperparameter | Search range |
|----------------|--------------|
| `seq_len` | categorical {64} |
| `depth` | categorical {1} |
| `num_heads` | categorical {1} |
| `variant` | categorical {baseline, embed-storkey, forecast-storkey, projection} |
| `hopfield_strength` | categorical {0.5, 1.0, 1.5, 2.0, learnable} |
| `lr` | float log ~1.8e-3 – 3.8e-3 |
| `eta_infer` | float log ~9e-6 – 2.5e-5 |
| `infer_steps` | int ~12–20 |
| `max_infer_norm` | categorical {0.5, 1.0} |
| `grad_clip` | categorical {0.5, 1.0} |
| `lr_decay_epochs` | categorical {5, 10, 15} |
| `weight_init_std` | float log ~0.013 – 0.022 |
| `seed_offset` | int 0–40 |

| Trial | State | Best MAE | Params |
|------:|-------|---------:|--------|
| 20 | COMPLETE | 20.160 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002464, eta_infer=1.016e-05, infer_steps=17 |
| 16 | COMPLETE | 20.305 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003077, eta_infer=1.256e-05, infer_steps=16 |
| 14 | COMPLETE | 20.406 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003543, eta_infer=1.377e-05, infer_steps=16 |
| 23 | COMPLETE | 20.416 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.00368, eta_infer=1.58e-05, infer_steps=16 |
| 8 | COMPLETE | 20.487 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003525, eta_infer=1.466e-05, infer_steps=16 |
| 17 | COMPLETE | 20.925 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003783, eta_infer=2.286e-05, infer_steps=17 |
| 10 | COMPLETE | 21.049 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.003352, eta_infer=1.267e-05, infer_steps=13 |
| 18 | COMPLETE | 21.333 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003256, eta_infer=1.61e-05, infer_steps=14 |
| 0 | COMPLETE | 21.494 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |
| 19 | COMPLETE | 21.702 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=2.0, lr=0.003792, eta_infer=1.775e-05, infer_steps=16 |
| 12 | PRUNED | 22.243 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.5, lr=0.002201, eta_infer=1.879e-05, infer_steps=17 |
| 11 | PRUNED | 24.526 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=learnable, lr=0.002385, eta_infer=1.804e-05, infer_steps=18 |
| 1 | COMPLETE | 25.020 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |
| 9 | PRUNED | 25.633 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=2.0, lr=0.002269, eta_infer=1.516e-05, infer_steps=17 |
| 13 | COMPLETE | 27.651 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=1.0, lr=0.003306, eta_infer=2.121e-05, infer_steps=18 |
| 22 | PRUNED | 39.328 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.0037, eta_infer=1.015e-05, infer_steps=16 |
| 6 | COMPLETE | 43.528 | seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=1.5, lr=0.003533, eta_infer=2.207e-05, infer_steps=12 |
| 2 | PRUNED | 43.611 | seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |
| 7 | PRUNED | 44.566 | seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=2.0, lr=0.002667, eta_infer=9.998e-06, infer_steps=14 |
| 4 | PRUNED | 45.380 | seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=learnable, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |
| 3 | COMPLETE | 45.465 | seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=2.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |
| 5 | COMPLETE | 67.431 | seq_len=64, depth=1, num_heads=1, variant=forecast-storkey, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |
| 15 | COMPLETE | 624.772 | seq_len=64, depth=1, num_heads=1, variant=forecast-storkey, hopfield_strength=0.5, lr=0.003085, eta_infer=1.3e-05, infer_steps=12 |
| 21 | PRUNED | 985.832 | seq_len=64, depth=1, num_heads=1, variant=forecast-storkey, hopfield_strength=learnable, lr=0.003661, eta_infer=1.237e-05, infer_steps=17 |

### Hopfield WSL v2

- Study: `glucose_hopfield_pc_wsl_v2`
- Run dir: `runs/glucose_hopfield_tuning_wsl_v2`
- Trials: 32 (complete 21, pruned 8, fail 3, running 0)
- Best: trial 12 · MAE 20.1219
- Best params: `seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001955, eta_infer=1.081e-05, infer_steps=13`

#### Search space (what Optuna sampled)

| Hyperparameter | Search range |
|----------------|--------------|
| `seq_len` | categorical {64} |
| `depth` | categorical {1} |
| `num_heads` | categorical {1} |
| `variant` | categorical {baseline, projection} |
| `hopfield_strength` | categorical {0.5, 1.0, learnable} |
| `lr` | float log 1.8e-3 – 3.8e-3 |
| `eta_infer` | float log 9e-6 – 2.5e-5 |
| `infer_steps` | int 12–20 |
| `max_infer_norm` | categorical {0.5, 1.0} |
| `grad_clip` | categorical {0.5, 1.0} |
| `lr_decay_epochs` | categorical {5, 10, 15} |
| `weight_init_std` | float log 0.013 – 0.022 |
| `seed_offset` | int 0–40 |

| Trial | State | Best MAE | Params |
|------:|-------|---------:|--------|
| 12 | COMPLETE | 20.122 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001955, eta_infer=1.081e-05, infer_steps=13 |
| 20 | COMPLETE | 20.142 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001999, eta_infer=1.436e-05, infer_steps=14 |
| 13 | COMPLETE | 20.144 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002112, eta_infer=9.608e-06, infer_steps=14 |
| 4 | COMPLETE | 20.160 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002464, eta_infer=1.016e-05, infer_steps=17 |
| 27 | COMPLETE | 20.190 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002039, eta_infer=1.318e-05, infer_steps=14 |
| 5 | COMPLETE | 20.468 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003077, eta_infer=1.256e-05, infer_steps=16 |
| 21 | COMPLETE | 20.495 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002091, eta_infer=1.318e-05, infer_steps=18 |
| 24 | COMPLETE | 20.538 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.00207, eta_infer=1.559e-05, infer_steps=12 |
| 28 | COMPLETE | 20.683 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.001842, eta_infer=1.453e-05, infer_steps=15 |
| 15 | COMPLETE | 20.995 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002233, eta_infer=9.701e-06, infer_steps=16 |
| 30 | PRUNED | 21.142 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002386, eta_infer=1.008e-05, infer_steps=13 |
| 25 | PRUNED | 21.319 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001888, eta_infer=1.019e-05, infer_steps=12 |
| 26 | COMPLETE | 21.383 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.001882, eta_infer=1.314e-05, infer_steps=14 |
| 17 | PRUNED | 21.477 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002885, eta_infer=2.489e-05, infer_steps=12 |
| 6 | COMPLETE | 21.537 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002058, eta_infer=1.464e-05, infer_steps=17 |
| 10 | COMPLETE | 21.565 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.00276, eta_infer=1.358e-05, infer_steps=15 |
| 29 | PRUNED | 21.640 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002004, eta_infer=1.052e-05, infer_steps=12 |
| 16 | COMPLETE | 21.777 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003784, eta_infer=1.233e-05, infer_steps=15 |
| 22 | COMPLETE | 22.428 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002215, eta_infer=1.002e-05, infer_steps=12 |
| 8 | COMPLETE | 22.435 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003721, eta_infer=1.434e-05, infer_steps=13 |
| 11 | COMPLETE | 23.226 | seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003146, eta_infer=1.434e-05, infer_steps=18 |
| 18 | COMPLETE | 23.255 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.003497, eta_infer=1.577e-05, infer_steps=12 |
| 3 | PRUNED | 25.020 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=learnable, lr=0.003691, eta_infer=1.68e-05, infer_steps=12 |
| 31 | COMPLETE | 26.076 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.002424, eta_infer=1.322e-05, infer_steps=16 |
| 9 | COMPLETE | 26.890 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=learnable, lr=0.002109, eta_infer=2.16e-05, infer_steps=20 |
| 19 | COMPLETE | 27.819 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=learnable, lr=0.002399, eta_infer=1.994e-05, infer_steps=15 |
| 7 | PRUNED | 28.062 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.00375, eta_infer=1.56e-05, infer_steps=20 |
| 23 | PRUNED | 28.806 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.002127, eta_infer=1.185e-05, infer_steps=13 |
| 14 | PRUNED | 33.612 | seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.002885, eta_infer=2.419e-05, infer_steps=17 |


## 4. Confirmation trains (single-config replay)

Longer epoch loops that replay a winning Optuna config and report validation + held-out test metrics.

### Phase-1 champion confirm

- Dir: `runs/glucose_pc_best_confirm`
- Best val MAE: 20.6285
- Test MAE: 21.7399
- Test MARD %: 15.41

#### Replay hyperparameters

| Hyperparameter | Value |
|----------------|------:|
| `seq_len` | 64 |
| `depth` | 1 |
| `num_heads` | 1 |
| `lr` | 0.002975 |
| `eta_infer` | 1.281e-05 |
| `infer_steps` | 14 |
| `weight_init_std` | 0.01684 |
| `epochs` | 30 |
| `seed` | 42 |
| `batch_size` | 64 |
| `horizon` | 12 |

### Phase-4 breakthrough confirm

- Dir: `runs/glucose_pc_breakthrough_confirm`
- Best val MAE: 18.7815
- Test MAE: 18.1991
- Test MARD %: 14.66

#### Replay hyperparameters

| Hyperparameter | Value |
|----------------|------:|
| `seq_len` | 64 |
| `depth` | 1 |
| `num_heads` | 1 |
| `lr` | 0.003691 |
| `eta_infer` | 1.68e-05 |
| `infer_steps` | 12 |
| `weight_init_std` | 0.01531 |
| `epochs` | 30 |
| `seed` | 61 |
| `batch_size` | 64 |
| `horizon` | 12 |

## How to run

| Task | Command |
|------|---------|
| Install (CPU) | `uv sync --extra glucose` |
| Install (GPU / WSL) | `uv sync --extra glucose --extra cuda12` |
| Check JAX device | `uv run python -c "import jax; print(jax.devices())"` |
| Train PC transformer | `uv run glucose-transformer` |
| Epoch Optuna (default) | `uv run glucose-transformer-tune run` |
| Update-budget Optuna (archived) | `uv run glucose-transformer-tune-update-budget run` |
| Hopfield pilot train | `uv run glucose-hopfield` |
| Hopfield Optuna | `uv run glucose-hopfield-tune run` |
| Hopfield Optuna on WSL | `bash scripts/run_hopfield_optuna_wsl.sh` |
| Summarize one study | `uv run python scripts/summarize_glucose_tuning.py` |
| Per-study report | `uv run python scripts/generate_glucose_tuning_report.py --format all` |
| Hopfield study report | `uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all` |
| All studies + master | `uv run python scripts/generate_all_glucose_reports.py --format all` |
| This master report only | `uv run python scripts/generate_glucose_master_report.py` |

Per-study HTML copies land in `docs/reports/old/`. The live master report stays at `docs/reports/glucose_master_progress.*`.
