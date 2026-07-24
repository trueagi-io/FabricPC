# Glucose Hopfield Optuna progress report

Generated: `2026-07-24T09:43:16.567163+00:00`  
Study: `glucose_hopfield_pc_wsl_v2`  
Mode: **predictive coding (PC) only** + Hopfield memory variants

## Goal

Beat transformer phase-4 Optuna champion **19.876** validation MAE (same `prepare_data` split / comparable protocol).

## Settings used

| Setting | Value |
|---------|------:|
| Protocol | epoch-based Hyperband (mirrors glucose-transformer-tune) |
| max_workers | 6 |
| n_trials | 32 |
| max_epochs | 15 |
| min_pruning_epochs | 3 |
| patience | 4 |
| batch_size | 64 |
| embed_dim / mlp_dim | 32 / 128 |
| gpu_memory_budget_mib | 12000 |
| estimated_trial_memory_mib | 1900 |
| target_optuna_mae | 19.876 |

Search notes: Locked geometry 64/d1/h1; Hopfield variant+strength; PC knobs in breakthrough band; same prepare_data split

## Cross-run comparison (transformer phases vs Hopfield)

| Run | Best val MAE | Δ vs Hopfield best |
|-----|-------------:|-------------------:|
| Transformer 1 broad (`runs/glucose_tuning`) | 20.3776 | 0.2557 |
| Transformer 2 refined (`runs/glucose_tuning_pc_v2`) | 20.6780 | 0.5561 |
| Transformer 3 local (`runs/glucose_tuning_pc_local`) | 20.8670 | 0.7451 |
| Transformer 4 breakthrough (`runs/glucose_tuning_pc_breakthrough`) | 19.8760 | -0.2459 |
| **Hopfield this study** | **20.1219** | 0 |

Beat 19.876 target: **NO** (Δ 0.2459).

## Study summary (all trial states)

| Metric | Value |
|--------|------:|
| Trials recorded | 32 |
| Complete | 21 |
| Pruned | 8 |
| Failed | 3 |
| Running | 0 |
| Best complete trial | 12 |
| Best val MAE | 20.1219 |
| Best variant | baseline |
| Best hopfield_strength | 0.5 |

## Top model architectures

### #1 — Trial 12 (MAE 20.122)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: fixed = 0.5
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001955, eta_infer=1.081e-05, infer_steps=13, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01444, seed_offset=33

### #2 — Trial 20 (MAE 20.142)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: fixed = 0.5
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001999, eta_infer=1.436e-05, infer_steps=14, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01409, seed_offset=21

### #3 — Trial 13 (MAE 20.144)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: fixed = 0.5
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002112, eta_infer=9.608e-06, infer_steps=14, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01991, seed_offset=11

### #4 — Trial 4 (MAE 20.160)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: learnable (optimised during training)
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002464, eta_infer=1.016e-05, infer_steps=17, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01417, seed_offset=33

### #5 — Trial 27 (MAE 20.190)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: fixed = 0.5
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002039, eta_infer=1.318e-05, infer_steps=14, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01574, seed_offset=11

## Complete-trial leaderboard

| Trial | Best MAE | Δ vs 19.876 | variant | strength | LR | η_infer | steps | seed_off |
|------:|---------:|------------:|---------|----------|---:|--------:|------:|---------:|
| 12 | 20.122 | 0.246 | baseline | 0.5 | 0.001955 | 1.081e-05 | 13 | 33 |
| 20 | 20.142 | 0.266 | baseline | 0.5 | 0.001999 | 1.436e-05 | 14 | 21 |
| 13 | 20.144 | 0.268 | baseline | 0.5 | 0.002112 | 9.608e-06 | 14 | 11 |
| 4 | 20.160 | 0.284 | baseline | learnable | 0.002464 | 1.016e-05 | 17 | 33 |
| 27 | 20.190 | 0.314 | baseline | 0.5 | 0.002039 | 1.318e-05 | 14 | 11 |
| 5 | 20.468 | 0.592 | baseline | learnable | 0.003077 | 1.256e-05 | 16 | 23 |
| 21 | 20.495 | 0.619 | baseline | learnable | 0.002091 | 1.318e-05 | 18 | 37 |
| 24 | 20.538 | 0.662 | baseline | 0.5 | 0.00207 | 1.559e-05 | 12 | 29 |
| 28 | 20.683 | 0.807 | baseline | 1.0 | 0.001842 | 1.453e-05 | 15 | 27 |
| 15 | 20.995 | 1.119 | baseline | 0.5 | 0.002233 | 9.701e-06 | 16 | 39 |
| 26 | 21.383 | 1.507 | baseline | learnable | 0.001882 | 1.314e-05 | 14 | 15 |
| 6 | 21.537 | 1.661 | baseline | learnable | 0.002058 | 1.464e-05 | 17 | 27 |
| 10 | 21.565 | 1.689 | baseline | 0.5 | 0.00276 | 1.358e-05 | 15 | 13 |
| 16 | 21.777 | 1.901 | baseline | 1.0 | 0.003784 | 1.233e-05 | 15 | 18 |
| 22 | 22.428 | 2.552 | baseline | 0.5 | 0.002215 | 1.002e-05 | 12 | 29 |
| 8 | 22.435 | 2.559 | baseline | learnable | 0.003721 | 1.434e-05 | 13 | 8 |
| 11 | 23.226 | 3.350 | baseline | 1.0 | 0.003146 | 1.434e-05 | 18 | 0 |
| 18 | 23.255 | 3.379 | projection | 0.5 | 0.003497 | 1.577e-05 | 12 | 2 |
| 31 | 26.076 | 6.200 | projection | 0.5 | 0.002424 | 1.322e-05 | 16 | 20 |
| 9 | 26.890 | 7.014 | projection | learnable | 0.002109 | 2.16e-05 | 20 | 35 |
| 19 | 27.819 | 7.943 | projection | learnable | 0.002399 | 1.994e-05 | 15 | 6 |

## All trials (every state)

| Trial | State | Best MAE | variant | strength | stop / prune |
|------:|-------|---------:|---------|----------|--------------|
| 0 | FAIL | — | baseline | 1.0 | — |
| 1 | FAIL | — | baseline | learnable | — |
| 2 | FAIL | — | projection | 1.0 | — |
| 3 | PRUNED | 25.020 | projection | learnable | HyperbandPruner at epoch 3 |
| 4 | COMPLETE | 20.160 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 5 | COMPLETE | 20.468 | baseline | learnable | validation MAE regressed over 10% twice |
| 6 | COMPLETE | 21.537 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 7 | PRUNED | 28.062 | projection | 0.5 | HyperbandPruner at epoch 3 |
| 8 | COMPLETE | 22.435 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 9 | COMPLETE | 26.890 | projection | learnable | validation MAE regressed over 10% twice |
| 10 | COMPLETE | 21.565 | baseline | 0.5 | no validation MAE improvement for 4 epochs |
| 11 | COMPLETE | 23.226 | baseline | 1.0 | no validation MAE improvement for 4 epochs |
| 12 | COMPLETE | 20.122 | baseline | 0.5 | no validation MAE improvement for 4 epochs |
| 13 | COMPLETE | 20.144 | baseline | 0.5 | no validation MAE improvement for 4 epochs |
| 14 | PRUNED | 33.612 | projection | 0.5 | HyperbandPruner at epoch 3 |
| 15 | COMPLETE | 20.995 | baseline | 0.5 | no validation MAE improvement for 4 epochs |
| 16 | COMPLETE | 21.777 | baseline | 1.0 | no validation MAE improvement for 4 epochs |
| 17 | PRUNED | 21.477 | baseline | 0.5 | HyperbandPruner at epoch 3 |
| 18 | COMPLETE | 23.255 | projection | 0.5 | no validation MAE improvement for 4 epochs |
| 19 | COMPLETE | 27.819 | projection | learnable | validation MAE regressed over 10% twice |
| 20 | COMPLETE | 20.142 | baseline | 0.5 | validation MAE regressed over 10% twice |
| 21 | COMPLETE | 20.495 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 22 | COMPLETE | 22.428 | baseline | 0.5 | validation MAE regressed over 10% twice |
| 23 | PRUNED | 28.806 | projection | 0.5 | HyperbandPruner at epoch 3 |
| 24 | COMPLETE | 20.538 | baseline | 0.5 | no validation MAE improvement for 4 epochs |
| 25 | PRUNED | 21.319 | baseline | 0.5 | HyperbandPruner at epoch 3 |
| 26 | COMPLETE | 21.383 | baseline | learnable | validation MAE regressed over 10% twice |
| 27 | COMPLETE | 20.190 | baseline | 0.5 | no validation MAE improvement for 4 epochs |
| 28 | COMPLETE | 20.683 | baseline | 1.0 | no validation MAE improvement for 4 epochs |
| 29 | PRUNED | 21.640 | baseline | 0.5 | HyperbandPruner at epoch 3 |
| 30 | PRUNED | 21.142 | baseline | learnable | HyperbandPruner at epoch 3 |
| 31 | COMPLETE | 26.076 | projection | 0.5 | validation MAE regressed over 10% twice |

## Within-trial stage comparison (epoch MAE traces)

- **Trial 12** (baseline, str=0.5, lr=0.001955, η=1.08e-05, steps=13, best 20.122): e1:23.05 → e2:21.24 → e3:22.30 → e4:21.51 → e5:26.20 → e6:20.12 → e7:20.77 → e8:21.01 → e9:20.66 → e10:20.30
- **Trial 20** (baseline, str=0.5, lr=0.001999, η=1.44e-05, steps=14, best 20.142): e1:22.03 → e2:20.81 → e3:20.14 → e4:22.69 → e5:26.44
- **Trial 13** (baseline, str=0.5, lr=0.002112, η=9.61e-06, steps=14, best 20.144): e1:21.84 → e2:21.57 → e3:20.87 → e4:20.93 → e5:20.21 → e6:20.15 → e7:20.18 → e8:20.17 → e9:20.14 → e10:20.18 → e11:20.18 → e12:20.17 → e13:20.18
- **Trial 4** (baseline, str=learnable, lr=0.002464, η=1.02e-05, steps=17, best 20.160): e1:22.17 → e2:21.29 → e3:20.16 → e4:21.36 → e5:20.42 → e6:20.41 → e7:20.37
- **Trial 27** (baseline, str=0.5, lr=0.002039, η=1.32e-05, steps=14, best 20.190): e1:22.09 → e2:21.51 → e3:20.65 → e4:20.19 → e5:21.37 → e6:21.12 → e7:21.90 → e8:21.43
- **Trial 5** (baseline, str=learnable, lr=0.003077, η=1.26e-05, steps=16, best 20.468): e1:21.34 → e2:20.84 → e3:20.47 → e4:22.38 → e5:21.05 → e6:24.19 → e7:25.26

## How to read this report

- **Trial** = one hyperparameter recipe (including Hopfield variant/strength).
- **Epoch** = one pass over training data inside that trial (learning-curve x-axis).
- **Pruned** = Hyperband stopped a weak trial early (saves compute, not a crash).
- **Early stop / patience** = val MAE stopped improving → keep best checkpoint.
- Lower **MAE (mg/dL)** is better.

## Complete training report (all trials)

### Trial 0 — FAIL (best MAE —)

- **Variant**: `baseline` · strength `1.0`
- **Stop / prune reason**: —
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

### Trial 1 — FAIL (best MAE —)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: —
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

### Trial 2 — FAIL (best MAE —)

- **Variant**: `projection` · strength `1.0`
- **Stop / prune reason**: —
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

### Trial 3 — PRUNED (best MAE 25.020)

- **Variant**: `projection` · strength `learnable`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=learnable, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 25.020 |
| 2 | 25.758 |
| 3 | 51.912 |

### Trial 4 — COMPLETE (best MAE 20.160)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002464, eta_infer=1.016e-05, infer_steps=17, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01417, seed_offset=33

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.174 |
| 2 | 21.286 |
| 3 | 20.160 |
| 4 | 21.356 |
| 5 | 20.418 |
| 6 | 20.406 |
| 7 | 20.372 |

### Trial 5 — COMPLETE (best MAE 20.468)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003077, eta_infer=1.256e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=23

| Epoch | Val MAE |
|------:|--------:|
| 1 | 21.336 |
| 2 | 20.843 |
| 3 | 20.468 |
| 4 | 22.376 |
| 5 | 21.048 |
| 6 | 24.190 |
| 7 | 25.256 |

### Trial 6 — COMPLETE (best MAE 21.537)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002058, eta_infer=1.464e-05, infer_steps=17, max_infer_norm=1, grad_clip=1, lr_decay_epochs=15, weight_init_std=0.01543, seed_offset=27

| Epoch | Val MAE |
|------:|--------:|
| 1 | 21.537 |
| 2 | 22.081 |
| 3 | 22.686 |
| 4 | 23.324 |
| 5 | 34.628 |

### Trial 7 — PRUNED (best MAE 28.062)

- **Variant**: `projection` · strength `0.5`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.00375, eta_infer=1.56e-05, infer_steps=20, max_infer_norm=0.5, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01852, seed_offset=25

| Epoch | Val MAE |
|------:|--------:|
| 1 | 66.866 |
| 2 | 34.273 |
| 3 | 28.062 |

### Trial 8 — COMPLETE (best MAE 22.435)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003721, eta_infer=1.434e-05, infer_steps=13, max_infer_norm=0.5, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01689, seed_offset=8

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.435 |
| 2 | 28.720 |
| 3 | 22.931 |
| 4 | 27.838 |
| 5 | 22.607 |

### Trial 9 — COMPLETE (best MAE 26.890)

- **Variant**: `projection` · strength `learnable`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=learnable, lr=0.002109, eta_infer=2.16e-05, infer_steps=20, max_infer_norm=1, grad_clip=1, lr_decay_epochs=15, weight_init_std=0.01808, seed_offset=35

| Epoch | Val MAE |
|------:|--------:|
| 1 | 26.890 |
| 2 | 31.960 |
| 3 | 36.741 |

### Trial 10 — COMPLETE (best MAE 21.565)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.00276, eta_infer=1.358e-05, infer_steps=15, max_infer_norm=0.5, grad_clip=1, lr_decay_epochs=5, weight_init_std=0.02061, seed_offset=13

| Epoch | Val MAE |
|------:|--------:|
| 1 | 29.545 |
| 2 | 34.520 |
| 3 | 21.849 |
| 4 | 22.417 |
| 5 | 21.658 |
| 6 | 21.901 |
| 7 | 21.618 |
| 8 | 21.647 |
| 9 | 21.829 |
| 10 | 21.565 |
| 11 | 21.811 |
| 12 | 21.691 |
| 13 | 21.727 |
| 14 | 22.003 |

### Trial 11 — COMPLETE (best MAE 23.226)

- **Variant**: `baseline` · strength `1.0`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003146, eta_infer=1.434e-05, infer_steps=18, max_infer_norm=0.5, grad_clip=0.5, lr_decay_epochs=15, weight_init_std=0.02045, seed_offset=0

| Epoch | Val MAE |
|------:|--------:|
| 1 | 31.826 |
| 2 | 23.226 |
| 3 | 23.616 |
| 4 | 24.123 |
| 5 | 24.264 |
| 6 | 45.162 |

### Trial 12 — COMPLETE (best MAE 20.122)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001955, eta_infer=1.081e-05, infer_steps=13, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01444, seed_offset=33

| Epoch | Val MAE |
|------:|--------:|
| 1 | 23.053 |
| 2 | 21.242 |
| 3 | 22.299 |
| 4 | 21.508 |
| 5 | 26.201 |
| 6 | 20.122 |
| 7 | 20.769 |
| 8 | 21.009 |
| 9 | 20.665 |
| 10 | 20.296 |

### Trial 13 — COMPLETE (best MAE 20.144)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002112, eta_infer=9.608e-06, infer_steps=14, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01991, seed_offset=11

| Epoch | Val MAE |
|------:|--------:|
| 1 | 21.841 |
| 2 | 21.574 |
| 3 | 20.865 |
| 4 | 20.928 |
| 5 | 20.212 |
| 6 | 20.148 |
| 7 | 20.179 |
| 8 | 20.174 |
| 9 | 20.144 |
| 10 | 20.185 |
| 11 | 20.184 |
| 12 | 20.175 |
| 13 | 20.181 |

### Trial 14 — PRUNED (best MAE 33.612)

- **Variant**: `projection` · strength `0.5`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.002885, eta_infer=2.419e-05, infer_steps=17, max_infer_norm=0.5, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01447, seed_offset=40

| Epoch | Val MAE |
|------:|--------:|
| 1 | 33.612 |
| 2 | 50.127 |
| 3 | 34.689 |

### Trial 15 — COMPLETE (best MAE 20.995)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002233, eta_infer=9.701e-06, infer_steps=16, max_infer_norm=1, grad_clip=1, lr_decay_epochs=5, weight_init_std=0.01471, seed_offset=39

| Epoch | Val MAE |
|------:|--------:|
| 1 | 24.899 |
| 2 | 21.020 |
| 3 | 22.434 |
| 4 | 20.995 |
| 5 | 21.144 |
| 6 | 21.139 |
| 7 | 21.285 |
| 8 | 21.299 |

### Trial 16 — COMPLETE (best MAE 21.777)

- **Variant**: `baseline` · strength `1.0`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003784, eta_infer=1.233e-05, infer_steps=15, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=15, weight_init_std=0.01304, seed_offset=18

| Epoch | Val MAE |
|------:|--------:|
| 1 | 21.777 |
| 2 | 22.802 |
| 3 | 26.412 |
| 4 | 21.847 |
| 5 | 39.355 |

### Trial 17 — PRUNED (best MAE 21.477)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002885, eta_infer=2.489e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=15, weight_init_std=0.0165, seed_offset=12

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.425 |
| 2 | 21.477 |
| 3 | 32.991 |

### Trial 18 — COMPLETE (best MAE 23.255)

- **Variant**: `projection` · strength `0.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.003497, eta_infer=1.577e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01744, seed_offset=2

| Epoch | Val MAE |
|------:|--------:|
| 1 | 25.972 |
| 2 | 27.040 |
| 3 | 23.426 |
| 4 | 23.255 |
| 5 | 23.310 |
| 6 | 23.320 |
| 7 | 23.703 |
| 8 | 23.398 |

### Trial 19 — COMPLETE (best MAE 27.819)

- **Variant**: `projection` · strength `learnable`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=learnable, lr=0.002399, eta_infer=1.994e-05, infer_steps=15, max_infer_norm=0.5, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01607, seed_offset=6

| Epoch | Val MAE |
|------:|--------:|
| 1 | 35.066 |
| 2 | 27.926 |
| 3 | 27.819 |
| 4 | 28.439 |
| 5 | 35.948 |
| 6 | 52.769 |

### Trial 20 — COMPLETE (best MAE 20.142)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001999, eta_infer=1.436e-05, infer_steps=14, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01409, seed_offset=21

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.034 |
| 2 | 20.806 |
| 3 | 20.142 |
| 4 | 22.694 |
| 5 | 26.436 |

### Trial 21 — COMPLETE (best MAE 20.495)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002091, eta_infer=1.318e-05, infer_steps=18, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01355, seed_offset=37

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.382 |
| 2 | 21.429 |
| 3 | 20.710 |
| 4 | 21.635 |
| 5 | 20.560 |
| 6 | 20.495 |
| 7 | 20.547 |
| 8 | 20.528 |
| 9 | 20.508 |
| 10 | 20.515 |

### Trial 22 — COMPLETE (best MAE 22.428)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002215, eta_infer=1.002e-05, infer_steps=12, max_infer_norm=0.5, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01546, seed_offset=29

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.428 |
| 2 | 25.387 |
| 3 | 30.259 |

### Trial 23 — PRUNED (best MAE 28.806)

- **Variant**: `projection` · strength `0.5`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.002127, eta_infer=1.185e-05, infer_steps=13, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.0183, seed_offset=14

| Epoch | Val MAE |
|------:|--------:|
| 1 | 29.098 |
| 2 | 28.806 |
| 3 | 30.945 |

### Trial 24 — COMPLETE (best MAE 20.538)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.00207, eta_infer=1.559e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.0131, seed_offset=29

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.287 |
| 2 | 21.385 |
| 3 | 20.538 |
| 4 | 26.262 |
| 5 | 21.606 |
| 6 | 23.180 |
| 7 | 21.992 |

### Trial 25 — PRUNED (best MAE 21.319)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.001888, eta_infer=1.019e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, lr_decay_epochs=15, weight_init_std=0.01759, seed_offset=32

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.815 |
| 2 | 21.319 |
| 3 | 21.713 |

### Trial 26 — COMPLETE (best MAE 21.383)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.001882, eta_infer=1.314e-05, infer_steps=14, max_infer_norm=0.5, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01368, seed_offset=15

| Epoch | Val MAE |
|------:|--------:|
| 1 | 23.636 |
| 2 | 22.515 |
| 3 | 22.714 |
| 4 | 24.462 |
| 5 | 21.383 |
| 6 | 26.420 |
| 7 | 24.379 |

### Trial 27 — COMPLETE (best MAE 20.190)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002039, eta_infer=1.318e-05, infer_steps=14, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01574, seed_offset=11

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.091 |
| 2 | 21.507 |
| 3 | 20.655 |
| 4 | 20.190 |
| 5 | 21.369 |
| 6 | 21.115 |
| 7 | 21.898 |
| 8 | 21.432 |

### Trial 28 — COMPLETE (best MAE 20.683)

- **Variant**: `baseline` · strength `1.0`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.001842, eta_infer=1.453e-05, infer_steps=15, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01582, seed_offset=27

| Epoch | Val MAE |
|------:|--------:|
| 1 | 23.077 |
| 2 | 21.696 |
| 3 | 20.683 |
| 4 | 26.829 |
| 5 | 22.540 |
| 6 | 22.438 |
| 7 | 25.109 |

### Trial 29 — PRUNED (best MAE 21.640)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.002004, eta_infer=1.052e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.0206, seed_offset=0

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.505 |
| 2 | 21.640 |
| 3 | 21.802 |

### Trial 30 — PRUNED (best MAE 21.142)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002386, eta_infer=1.008e-05, infer_steps=13, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01554, seed_offset=27

| Epoch | Val MAE |
|------:|--------:|
| 1 | 23.526 |
| 2 | 21.142 |
| 3 | 21.236 |

### Trial 31 — COMPLETE (best MAE 26.076)

- **Variant**: `projection` · strength `0.5`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.002424, eta_infer=1.322e-05, infer_steps=16, max_infer_norm=1, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01357, seed_offset=20

| Epoch | Val MAE |
|------:|--------:|
| 1 | 26.076 |
| 2 | 34.468 |
| 3 | 29.655 |

## Hyperparameter theory (for newbies)

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
| `seed_offset` | Seed nudge so similar configs can still differ. |

### Deeper explanations

#### `seq_len`

- **What it is**: History window length fed into the network.
- **Why you care**: Too short misses delayed glucose effects; too long adds noise and cost.
- **How changes show up**: 64 ≈ 5.3 hours of context; longer is not automatically better.

#### `depth`

- **What it is**: Number of attention+MLP stages stacked.
- **Why you care**: Depth is capacity. Shallow nets are simpler and often better on small CGM sets.
- **How changes show up**: depth=1 is a short assembly line; deeper stacks can overfit or train slower.

#### `num_heads`

- **What it is**: Multi-head attention count.
- **Why you care**: More heads can specialise, but need enough width/data.
- **How changes show up**: 1 head is common and stable on small models.

#### `variant`

- **What it is**: Which graph wiring to train: where Hopfield memory sits, or baseline (no Hopfield). This is an Optuna categorical choice — each trial gets one architecture from the search space.
- **Why you care**: Placement changes when associative recall can influence features vs the final forecast. Optuna explores variants; it is not fixed by hand per trial.
- **How changes show up**: baseline = pure transformer control. embed-* recalls early; forecast-* recalls late; projection is a lighter linear memory.

#### `hopfield_strength`

- **What it is**: Fixed scale (e.g. 0.5–2.0) or 'learnable'.
- **Why you care**: Too strong can overwrite useful transformer features; too weak is a no-op.
- **How changes show up**: learnable lets training pick the mix; fixed values are easier to compare across trials.

#### `lr`

- **What it is**: Step size for updating model weights.
- **Why you care**: Too high diverges; too low never improves in budget.
- **How changes show up**: Mid-range ~1e-3–4e-3 often works with champion-like PC settings.

#### `eta_infer`

- **What it is**: Learning rate of the PC inference loop.
- **Why you care**: Separate from weight LR — controls how hard latents correct prediction error.
- **How changes show up**: Around 1e-5–2.5e-5 was a healthy band in transformer PC runs.

#### `infer_steps`

- **What it is**: How many times latents are refined before forecasting.
- **Why you care**: More steps → tighter energy, more compute.
- **How changes show up**: Low teens (12–18) are typical; doubling rarely helps if η is wrong.

#### `max_infer_norm`

- **What it is**: Max norm for inner-loop updates.
- **Why you care**: Prevents exploding activations on sharp glucose swings.
- **How changes show up**: Lower = safer/slower settle; higher = freer but riskier.

#### `grad_clip`

- **What it is**: Global grad clip for Adam.
- **Why you care**: Stops rare huge gradients from wrecking a run.
- **How changes show up**: 0.5–1.0 are common stable choices.

#### `lr_decay_epochs`

- **What it is**: Epoch index that starts annealing LR.
- **Why you care**: Balances exploration early vs fine-tuning later.
- **How changes show up**: Later decay keeps LR high longer.

#### `weight_init_std`

- **What it is**: Normal init standard deviation.
- **Why you care**: Interacts with PC dynamics and depth.
- **How changes show up**: Smaller often safer with PC; larger can help or explode.

#### `seed_offset`

- **What it is**: Added to the base random seed.
- **Why you care**: PC runs can be seed-sensitive.
- **How changes show up**: Document the winning seed for fair replay.

## Background

Builds on [GlucoseDAO/glucose-forecasting](https://github.com/GlucoseDAO/glucose-forecasting).
Adds **PC** inner loops and **Hopfield** associative memory for glucose pattern recall.

## How to run

| Task | Command |
|------|---------|
| Install (CPU) | `uv sync --extra glucose` |
| Install (GPU / WSL) | `uv sync --extra glucose --extra cuda12` |
| Hopfield pilot | `uv run glucose-hopfield` |
| Start Hopfield tuning | `uv run glucose-hopfield-tune run` |
| Hopfield Optuna on WSL | `bash scripts/run_hopfield_optuna_wsl.sh` |
| Regenerate this report | `uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all` |
| All reports + master | `uv run python scripts/generate_all_glucose_reports.py --format all` |
| Master report only | `uv run python scripts/generate_glucose_master_report.py` |
