# Glucose PC Optuna progress report

Generated: `2026-07-24T08:13:24.174184+00:00`  
Study: `glucose_transformer_pc_local`  
Mode: predictive coding (PC) only

## Summary

| Metric | Value |
|--------|------:|
| Trials recorded | 24 |
| Complete | 3 |
| Pruned | 21 |
| Failed | 0 |
| Running | 0 |
| Best complete trial | 21 |
| Best val MAE (mg/dL) | 20.8670 |

## What helped (auto-generated from top trials)

- **seq_len=64** dominates top 5 (3/3)
- **depth=1** dominates top 5 (3/3)
- **num_heads=1** dominates top 5 (3/3)
- **lr**: range 0.002975–0.003946, median 0.0034
- **eta_infer**: range 9.955e-06–1.281e-05, median 1.1e-05
- **weight_init_std**: range 0.01684–0.01921, median 0.01684
- **infer_steps**: 15×1, 16×1, 14×1
- **grad_clip**: 0.5×3

## Top model architectures

### #1 — Trial 21 (MAE 20.867)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.003946, eta_infer=9.955e-06, infer_steps=15, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01921

### #2 — Trial 1 (MAE 21.450)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.0034, eta_infer=1.1e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684

### #3 — Trial 0 (MAE 23.043)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684

## Complete-trial leaderboard

| Trial | Best MAE | MARD% | Geometry | LR | η_infer | Infer steps | Grad clip |
|------:|---------:|------:|----------|---:|--------:|------------:|----------:|
| 21 | 20.867 | 16.47 | 64/d1/h1 | 0.003946 | 9.955e-06 | 15 | 0.5 |
| 1 | 21.450 | 16.81 | 64/d1/h1 | 0.0034 | 1.1e-05 | 16 | 0.5 |
| 0 | 23.043 | 17.53 | 64/d1/h1 | 0.002975 | 1.281e-05 | 14 | 0.5 |

## Best MAE by trial

| Trial | State | Best MAE (mg/dL) |
|------:|-------|-----------------:|
| 0 | COMPLETE | 23.043 |
| 1 | COMPLETE | 21.450 |
| 2 | PRUNED | 32.119 |
| 3 | PRUNED | 30.321 |
| 4 | PRUNED | 31.453 |
| 5 | PRUNED | 35.893 |
| 6 | PRUNED | 31.198 |
| 7 | PRUNED | 33.168 |
| 8 | PRUNED | 33.008 |
| 9 | PRUNED | 32.259 |
| 10 | PRUNED | 33.419 |
| 11 | PRUNED | 23.805 |
| 12 | PRUNED | 30.747 |
| 13 | PRUNED | 31.064 |
| 14 | PRUNED | 34.239 |
| 15 | PRUNED | 23.042 |
| 16 | PRUNED | 31.078 |
| 17 | PRUNED | 31.316 |
| 18 | PRUNED | 30.911 |
| 19 | PRUNED | 25.278 |
| 20 | PRUNED | 31.077 |
| 21 | COMPLETE | 20.867 |
| 22 | PRUNED | 23.880 |
| 23 | PRUNED | 23.259 |

## Top trial MAE traces (every 200 updates)

- **Trial 21** (best 20.867): 200:29.10 → 400:24.68 → 600:23.01 → 800:22.10 → 1000:25.08 → 1200:21.63 → 1400:21.08 → 1600:20.87 → 1800:22.09 → 2000:21.64 → 2200:24.62
- **Trial 1** (best 21.450): 200:30.00 → 400:25.09 → 600:23.00 → 800:22.70 → 1000:22.96 → 1200:24.87 → 1400:22.28 → 1600:21.45 → 1800:25.30 → 2000:21.80 → 2200:23.51 → 2400:28.03
- **Trial 0** (best 23.043): 200:30.02 → 400:25.29 → 600:23.04 → 800:26.27 → 1000:25.67

## PC confirmation train (epoch loop)

Directory: `runs/glucose_pc_best_confirm`

| Metric | Value |
|--------|------:|
| Best val MAE | 20.6285 |
| Test MAE | 21.7399 |
| Test RMSE | 32.0564 |
| Test MARD (%) | 15.41 |
| Epochs run | 7 |
| Wall time (s) | 164.4 |

### Epoch validation MAE

| Epoch | Val MAE | RMSE | MARD (%) |
|------:|--------:|-----:|---------:|
| 1 | 22.775 | 30.976 | 19.68 |
| 2 | 21.159 | 29.507 | 17.62 |
| 3 | 20.629 | 30.480 | 15.08 |
| 4 | 24.222 | 31.870 | 20.99 |
| 5 | 30.918 | 38.892 | 25.76 |
| 6 | 23.384 | 31.149 | 18.71 |
| 7 | 27.802 | 36.756 | 19.35 |

## How to read this report

- **Trial** = one hyperparameter recipe Optuna tried.
- **Epoch / update** = training progress *inside* that recipe (learning curve).
- **Pruned** = Optuna stopped a weak trial early to save compute (not a crash).
- **Early stop / patience** = validation MAE stopped improving, so training halted and kept the best checkpoint.
- Lower **MAE (mg/dL)** is better.

## Complete training report (all trials)

Each trial below keeps the same layout: summary → params → stage trace → stop reason.

### Trial 0 — COMPLETE (best MAE 23.043)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.002975, eta_infer=1.281e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.023 | 23.07 |
| 400 | 25.292 | 20.95 |
| 600 | 23.043 | 17.53 |
| 800 | 26.273 | 18.84 |
| 1000 | 25.669 | 18.76 |

### Trial 1 — COMPLETE (best MAE 21.450)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.0034, eta_infer=1.1e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.003 | 23.25 |
| 400 | 25.091 | 18.75 |
| 600 | 23.004 | 18.04 |
| 800 | 22.698 | 17.27 |
| 1000 | 22.959 | 16.79 |
| 1200 | 24.871 | 17.95 |
| 1400 | 22.285 | 17.80 |
| 1600 | 21.450 | 16.81 |
| 1800 | 25.303 | 21.90 |
| 2000 | 21.802 | 15.86 |
| 2200 | 23.511 | 19.81 |
| 2400 | 28.028 | 25.26 |

### Trial 2 — PRUNED (best MAE 32.119)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0025, eta_infer=1.6e-05, infer_steps=12, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01684

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 32.119 | 24.10 |

### Trial 3 — PRUNED (best MAE 30.321)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0032, eta_infer=9.5e-06, infer_steps=15, max_infer_norm=1, grad_clip=1, weight_init_std=0.018

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.321 | 24.06 |

### Trial 4 — PRUNED (best MAE 31.453)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002474, eta_infer=1.179e-05, infer_steps=12, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.0205

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.453 | 24.12 |

### Trial 5 — PRUNED (best MAE 35.893)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0015, eta_infer=1.833e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01662

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 35.893 | 28.35 |

### Trial 6 — PRUNED (best MAE 31.198)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003913, eta_infer=2.197e-05, infer_steps=18, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.01426

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.198 | 21.97 |

### Trial 7 — PRUNED (best MAE 33.168)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003295, eta_infer=2.36e-05, infer_steps=14, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01925

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.168 | 28.55 |

### Trial 8 — PRUNED (best MAE 33.008)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001864, eta_infer=2.281e-05, infer_steps=13, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01549

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.008 | 26.08 |

### Trial 9 — PRUNED (best MAE 32.259)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002128, eta_infer=2.611e-05, infer_steps=15, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01642

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 32.259 | 25.46 |

### Trial 10 — PRUNED (best MAE 33.419)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001685, eta_infer=2.43e-05, infer_steps=14, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01482

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.419 | 26.06 |

### Trial 11 — PRUNED (best MAE 23.805)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.003869, eta_infer=1.578e-05, infer_steps=16, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.01699

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.046 | 23.70 |
| 400 | 25.038 | 20.77 |
| 600 | 23.805 | 17.75 |

### Trial 12 — PRUNED (best MAE 30.747)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003646, eta_infer=1.011e-05, infer_steps=17, max_infer_norm=1, grad_clip=1, weight_init_std=0.02047

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.747 | 24.81 |

### Trial 13 — PRUNED (best MAE 31.064)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003549, eta_infer=2.234e-05, infer_steps=16, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01646

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.064 | 26.47 |

### Trial 14 — PRUNED (best MAE 34.239)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001744, eta_infer=1.111e-05, infer_steps=14, max_infer_norm=1, grad_clip=1, weight_init_std=0.01877

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 34.239 | 26.05 |

### Trial 15 — PRUNED (best MAE 23.042)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.002798, eta_infer=2.766e-05, infer_steps=17, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01637

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.107 | 22.22 |
| 400 | 24.377 | 19.39 |
| 600 | 23.042 | 18.39 |

### Trial 16 — PRUNED (best MAE 31.078)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003273, eta_infer=1.352e-05, infer_steps=18, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.0189

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.078 | 24.67 |

### Trial 17 — PRUNED (best MAE 31.316)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003599, eta_infer=1.043e-05, infer_steps=17, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01495

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.316 | 26.91 |

### Trial 18 — PRUNED (best MAE 30.911)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002791, eta_infer=1.519e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.0172

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.911 | 23.57 |

### Trial 19 — PRUNED (best MAE 25.278)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.003919, eta_infer=8.897e-06, infer_steps=12, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01706

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.470 | 23.69 |
| 400 | 25.278 | 21.18 |
| 600 | 25.480 | 21.73 |

### Trial 20 — PRUNED (best MAE 31.077)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002735, eta_infer=8.955e-06, infer_steps=18, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01828

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.077 | 24.06 |

### Trial 21 — COMPLETE (best MAE 20.867)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.003946, eta_infer=9.955e-06, infer_steps=15, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01921

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.104 | 23.60 |
| 400 | 24.682 | 19.93 |
| 600 | 23.011 | 18.43 |
| 800 | 22.103 | 16.88 |
| 1000 | 25.076 | 17.68 |
| 1200 | 21.631 | 16.24 |
| 1400 | 21.081 | 16.31 |
| 1600 | 20.867 | 16.47 |
| 1800 | 22.094 | 18.32 |
| 2000 | 21.643 | 16.04 |
| 2200 | 24.623 | 21.23 |

### Trial 22 — PRUNED (best MAE 23.880)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.002972, eta_infer=2.611e-05, infer_steps=18, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01646

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.931 | 22.91 |
| 400 | 24.601 | 18.80 |
| 600 | 23.880 | 17.88 |

### Trial 23 — PRUNED (best MAE 23.259)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.003689, eta_infer=1.395e-05, infer_steps=13, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01627

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.117 | 22.82 |
| 400 | 27.487 | 19.73 |
| 600 | 23.259 | 18.50 |

## Hyperparameter theory (for newbies)

Short glossary first, then practical intuition for each knob.

| Parameter | One-line meaning |
|-----------|------------------|
| `seq_len` | How many recent 5-min CGM readings the model sees (e.g. 64 ≈ 5.3 hours). |
| `depth` | How many transformer blocks are stacked (model “shallowness”). |
| `num_heads` | Parallel attention “viewpoints” inside each block. |
| `embed_dim` | Width of the internal vector for each timestep. |
| `lr` | Outer learning rate — how big each weight update is (Adam). |
| `eta_infer` | Inner PC step size for refining latent activations. |
| `infer_steps` | How many inner PC refinement iterations per forward pass. |
| `max_infer_norm` | Clip on the size of each PC latent update. |
| `grad_clip` | Clip on outer weight gradients (training stability). |
| `lr_decay_epochs` | When cosine LR decay begins (epoch-based runs). |
| `weight_init_std` | How large random initial weights are. |
| `weight_decay` | L2 penalty that discourages huge weights. |
| `readout` | How the sequence is turned into one glucose forecast. |
| `seed_offset` | Random-seed nudge so similar configs can still differ. |
| `energy` | How PC nodes score prediction error (Gaussian vs Huber). |
| `huber_delta` | Threshold where Huber switches from quadratic to linear. |
| `ipc` | Update latents layer-by-layer (incremental PC) vs all at once. |
| `infer_optimizer` | Optimiser inside the PC loop: SGD or Adam. |

### Deeper explanations

#### `seq_len`

- **What it is**: Sequence length — the history window length fed into the transformer.
- **Why you care**: Glucose has delayed effects (meals, activity). Too short and the model misses context; too long and it can drown in noise and cost more compute.
- **How changes show up**: 64 often wins here (recent day-part). 128 doubles history but did not reliably help under our budget.

#### `depth`

- **What it is**: Depth — number of identical transformer blocks stacked on top of each other.
- **Why you care**: Each block can refine representations. Depth is capacity: more layers can learn richer patterns, but also train slower and overfit more easily on small data.
- **How changes show up**: Shallow (depth=1) means one attention+MLP stage — like a short assembly line. Deeper (2–4) means more stages. On this single-person CGM set, shallow models often generalise better and finish trials sooner.

#### `num_heads`

- **What it is**: Multi-head attention splits attention into several heads that each look for different relationships in the sequence.
- **Why you care**: One head might track rising trends; another meal-like bumps. More heads = more specialised views, but also more parameters.
- **How changes show up**: With small embed dims, 1 head is common and stable. Extra heads help only if there is enough data and width to use them.

#### `embed_dim`

- **What it is**: Embedding dimension — size of the hidden vector representing each glucose reading inside the network.
- **Why you care**: Wider vectors can store richer features, but need more data and memory.
- **How changes show up**: Larger embed_dim → more expressive, heavier. Too large on tiny data can overfit (train looks good, val MAE worse).

#### `lr`

- **What it is**: Learning rate for the outer optimiser that updates model weights after each batch/update.
- **Why you care**: Too high → training jumps around or diverges (MAE explodes). Too low → crawls and never improves within the budget.
- **How changes show up**: Mid-range ~1e-3–4e-3 often worked for champion-like PC configs. Think of it as step size on a foggy hill: big steps miss the valley; tiny steps never arrive.

#### `eta_infer`

- **What it is**: PC inference learning rate (η). Inside each forward pass, latents are nudged to reduce prediction error.
- **Why you care**: PC has an inner loop separate from weight LR. η controls how aggressively beliefs are corrected before weights move.
- **How changes show up**: Too large → unstable energy / wild MAE. Too small → under-inferred latents (model never “settles”). Sweet spot here was roughly 1e-5–2.5e-5.

#### `infer_steps`

- **What it is**: Number of times the inner PC loop updates latents before producing a forecast.
- **Why you care**: More steps ≈ tighter energy minimum, but each step costs compute. Too few and PC barely runs.
- **How changes show up**: 12–16 steps were common in strong trials. Doubling steps rarely pays for itself if η is already well tuned.

#### `max_infer_norm`

- **What it is**: Maximum gradient/update norm allowed during the inner PC loop.
- **Why you care**: Stops latent activations from exploding when errors are large (e.g. after a sharp glucose swing).
- **How changes show up**: Lower clips = safer but slower settling. Higher clips = freer movement, more risk of blow-ups.

#### `grad_clip`

- **What it is**: Global gradient clipping for Adam weight updates.
- **Why you care**: Occasional huge gradients can wipe a good run. Clipping caps the damage.
- **How changes show up**: 0.5–1.0 were typical. Too tight can stall learning; too loose lets rare spikes destabilise training.

#### `lr_decay_epochs`

- **What it is**: Epoch index after which the outer learning rate anneals toward zero.
- **Why you care**: Early high LR explores; later lower LR fine-tunes. Decay timing changes that schedule.
- **How changes show up**: Later decay = longer aggressive phase. Earlier decay = settle sooner (good if you overshoot).

#### `weight_init_std`

- **What it is**: Standard deviation of Normal weight initialisation.
- **Why you care**: Starting scale interacts with depth and PC dynamics. Bad init can look like a “broken” hyperparameter set.
- **How changes show up**: Smaller std = gentler start (often better with PC). Larger std can help or explode depending on η/LR.

#### `weight_decay`

- **What it is**: Weight decay regularisation strength.
- **Why you care**: On small datasets, unconstrained weights memorise noise. Decay nudges them toward simpler solutions.
- **How changes show up**: Higher → stronger regularisation (can underfit). Zero → freer fit (can overfit).

#### `readout`

- **What it is**: Regression head mode: flatten / mean_pool / last.
- **Why you care**: The network outputs a sequence of vectors; readout decides how to map that to a single 60‑min-ahead number.
- **How changes show up**: flatten uses the full sequence (more parameters, often best here). mean_pool averages time; last uses only the newest step — lighter heads that may need their own LR/η retuning.

#### `seed_offset`

- **What it is**: Added to a base seed so trials explore different initialisations/data shuffles.
- **Why you care**: PC training can be seed-sensitive. Searching a small offset finds lucky (or unlucky) starts.
- **How changes show up**: Same architecture can move several MAE points just from seed — document the winning seed for replay.

#### `energy`

- **What it is**: Energy functional used inside PC nodes.
- **Why you care**: Gaussian (MSE) punishes large errors hard. Huber becomes linear for outliers — useful for glucose spikes.
- **How changes show up**: Gaussian is the classic default. Huber can be more robust when a few wild points would otherwise dominate.

#### `huber_delta`

- **What it is**: Delta parameter for Huber energy (only if energy=huber).
- **Why you care**: Controls when an error is treated as an “outlier”.
- **How changes show up**: Smaller delta → more robust / less spike-sensitive. Larger → closer to plain MSE.

#### `ipc`

- **What it is**: Incremental Predictive Coding flag.
- **Why you care**: Layerwise updates can improve convergence on deep stacks by letting lower layers settle first.
- **How changes show up**: On shallow nets the difference may be small; on deeper nets IPC can matter more.

#### `infer_optimizer`

- **What it is**: Which optimiser nudges latent activations during inference.
- **Why you care**: SGD is simple/fast; Adam adapts per-coordinate and may settle in fewer steps (more memory).
- **How changes show up**: Try SGD first for speed; Adam if inference looks under-converged at the same step count.

## Background

This work builds on our earlier results with conventional (non-PC) transformers for glucose
forecasting at [GlucoseDAO/glucose-forecasting](https://github.com/GlucoseDAO/glucose-forecasting).
Here we replace the standard forward pass with **predictive coding (PC)** — an inner
optimisation loop where each layer maintains its own "belief" about what the input should
look like, computes a prediction error, and iteratively refines its activations before the
outer weight update.

We also explore a **Hopfield extension** — adding a content-addressable associative memory
layer (Storkey Hopfield network) that can store and recall learned glucose dynamics such as
meal responses, exercise patterns, and dawn phenomenon. The Hopfield memory gives the model
an explicit pattern-recall mechanism beyond what attention alone provides.

## How the model works

### Standard PC Transformer

```
Glucose Input (batch, seq_len, 1)
       |
  Continuous Embedding  — linear projection to embed_dim
       |
  +--[ Transformer Block ] × depth --------+
  |    Multi-Scale Self-Attention (RoPE)    |
  |    at downsampling 1×, 2×, 4×           |
  |    LN → MLP expand (GELU)               |
  |    MLP contract + Residual skip          |
  |    PC Energy Node                        |
  +------------------------------------------+
       |
  Regression Output Head → Glucose Forecast (60 min)
```

### PC inference loop (runs at every node)

1. Predict `z_mu` from incoming activations
2. Compute `error = z_latent - z_mu`
3. Compute energy from error (Gaussian or Huber)
4. Update `z_latent` via SGD or Adam (step size = `eta_infer`, clip = `max_infer_norm`)
5. Repeat for `infer_steps` iterations

## Limitations

- **Single participant data** — Livia's personal CGM only for the deadline sprint.
- **Glucose-only input** — carbs / HR / steps not included yet.
- **Limited tuning budget** — finite Optuna trials and ranges.

## How to run

| Task | Command |
|------|---------|
| Install (CPU) | `uv sync --extra glucose` |
| Install (GPU / WSL) | `uv sync --extra glucose --extra cuda12` |
| Train PC transformer | `uv run glucose-transformer` |
| Start epoch Optuna | `uv run glucose-transformer-tune run` |
| Update-budget Optuna | `uv run glucose-transformer-tune-update-budget run` |
| Regenerate this report | `uv run python scripts/generate_glucose_tuning_report.py --format all` |
| All reports + master | `uv run python scripts/generate_all_glucose_reports.py --format all` |
| Master report only | `uv run python scripts/generate_glucose_master_report.py` |
