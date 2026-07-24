# Glucose Hopfield Optuna progress report

Generated: `2026-07-24T08:13:20.673980+00:00`  
Study: `glucose_hopfield_pc_wsl_v1`  
Mode: **predictive coding (PC) only** + Hopfield memory variants

## Goal

Beat transformer phase-4 Optuna champion **19.876** validation MAE (same `prepare_data` split / comparable protocol).

## Settings used

| Setting | Value |
|---------|------:|
| Protocol | epoch-based Hyperband (mirrors glucose-transformer-tune) |
| max_workers | 3 |
| n_trials | 24 |
| max_epochs | 15 |
| min_pruning_epochs | 3 |
| patience | 4 |
| batch_size | 64 |
| embed_dim / mlp_dim | 32 / 128 |
| gpu_memory_budget_mib | 12000 |
| estimated_trial_memory_mib | 2500 |
| target_optuna_mae | 19.876 |

Search notes: Locked geometry 64/d1/h1; Hopfield variant+strength; PC knobs in breakthrough band; same prepare_data split

## Cross-run comparison (transformer phases vs Hopfield)

| Run | Best val MAE | Δ vs Hopfield best |
|-----|-------------:|-------------------:|
| Transformer 1 broad (`runs/glucose_tuning`) | 20.3776 | 0.2171 |
| Transformer 2 refined (`runs/glucose_tuning_pc_v2`) | 20.6780 | 0.5175 |
| Transformer 3 local (`runs/glucose_tuning_pc_local`) | 20.8670 | 0.7065 |
| Transformer 4 breakthrough (`runs/glucose_tuning_pc_breakthrough`) | 19.8760 | -0.2845 |
| **Hopfield this study** | **20.1605** | 0 |

Beat 19.876 target: **NO** (Δ 0.2845).

## Study summary (all trial states)

| Metric | Value |
|--------|------:|
| Trials recorded | 24 |
| Complete | 16 |
| Pruned | 8 |
| Failed | 0 |
| Running | 0 |
| Best complete trial | 20 |
| Best val MAE | 20.1605 |
| Best variant | baseline |
| Best hopfield_strength | learnable |

## Top model architectures

### #1 — Trial 20 (MAE 20.160)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: learnable (optimised during training)
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.002464, eta_infer=1.016e-05, infer_steps=17, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01417, seed_offset=33

### #2 — Trial 16 (MAE 20.305)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: learnable (optimised during training)
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003077, eta_infer=1.256e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01632, seed_offset=23

### #3 — Trial 14 (MAE 20.406)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: learnable (optimised during training)
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003543, eta_infer=1.377e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01717, seed_offset=29

### #4 — Trial 23 (MAE 20.416)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: learnable (optimised during training)
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.00368, eta_infer=1.58e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01573, seed_offset=13

### #5 — Trial 8 (MAE 20.487)

- **Architecture**: Pure transformer (no Hopfield node)
- **Hopfield strength**: learnable (optimised during training)
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003525, eta_infer=1.466e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01661, seed_offset=33

## Complete-trial leaderboard

| Trial | Best MAE | Δ vs 19.876 | variant | strength | LR | η_infer | steps | seed_off |
|------:|---------:|------------:|---------|----------|---:|--------:|------:|---------:|
| 20 | 20.160 | 0.284 | baseline | learnable | 0.002464 | 1.016e-05 | 17 | 33 |
| 16 | 20.305 | 0.429 | baseline | learnable | 0.003077 | 1.256e-05 | 16 | 23 |
| 14 | 20.406 | 0.530 | baseline | learnable | 0.003543 | 1.377e-05 | 16 | 29 |
| 23 | 20.416 | 0.540 | baseline | learnable | 0.00368 | 1.58e-05 | 16 | 13 |
| 8 | 20.487 | 0.611 | baseline | learnable | 0.003525 | 1.466e-05 | 16 | 33 |
| 17 | 20.925 | 1.049 | baseline | learnable | 0.003783 | 2.286e-05 | 17 | 33 |
| 10 | 21.049 | 1.173 | baseline | 0.5 | 0.003352 | 1.267e-05 | 13 | 37 |
| 18 | 21.333 | 1.457 | baseline | learnable | 0.003256 | 1.61e-05 | 14 | 33 |
| 0 | 21.494 | 1.618 | baseline | 1.0 | 0.003691 | 1.68e-05 | 12 | 19 |
| 19 | 21.702 | 1.826 | baseline | 2.0 | 0.003792 | 1.775e-05 | 16 | 37 |
| 1 | 25.020 | 5.144 | projection | 1.0 | 0.003691 | 1.68e-05 | 12 | 19 |
| 13 | 27.651 | 7.775 | projection | 1.0 | 0.003306 | 2.121e-05 | 18 | 35 |
| 6 | 43.528 | 23.652 | embed-storkey | 1.5 | 0.003533 | 2.207e-05 | 12 | 21 |
| 3 | 45.465 | 25.589 | embed-storkey | 2.0 | 0.003691 | 1.68e-05 | 12 | 19 |
| 5 | 67.431 | 47.555 | forecast-storkey | 1.0 | 0.003691 | 1.68e-05 | 12 | 19 |
| 15 | 624.772 | 604.896 | forecast-storkey | 0.5 | 0.003085 | 1.3e-05 | 12 | 29 |

## All trials (every state)

| Trial | State | Best MAE | variant | strength | stop / prune |
|------:|-------|---------:|---------|----------|--------------|
| 0 | COMPLETE | 21.494 | baseline | 1.0 | no validation MAE improvement for 4 epochs |
| 1 | COMPLETE | 25.020 | projection | 1.0 | validation MAE regressed over 10% twice |
| 2 | PRUNED | 43.611 | embed-storkey | 1.0 | HyperbandPruner at epoch 3 |
| 3 | COMPLETE | 45.465 | embed-storkey | 2.0 | no validation MAE improvement for 4 epochs |
| 4 | PRUNED | 45.380 | embed-storkey | learnable | HyperbandPruner at epoch 3 |
| 5 | COMPLETE | 67.431 | forecast-storkey | 1.0 | validation MAE regressed over 10% twice |
| 6 | COMPLETE | 43.528 | embed-storkey | 1.5 | no validation MAE improvement for 4 epochs |
| 7 | PRUNED | 44.566 | embed-storkey | 2.0 | HyperbandPruner at epoch 3 |
| 8 | COMPLETE | 20.487 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 9 | PRUNED | 25.633 | projection | 2.0 | HyperbandPruner at epoch 3 |
| 10 | COMPLETE | 21.049 | baseline | 0.5 | no validation MAE improvement for 4 epochs |
| 11 | PRUNED | 24.526 | projection | learnable | HyperbandPruner at epoch 3 |
| 12 | PRUNED | 22.243 | baseline | 1.5 | HyperbandPruner at epoch 3 |
| 13 | COMPLETE | 27.651 | projection | 1.0 | validation MAE regressed over 10% twice |
| 14 | COMPLETE | 20.406 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 15 | COMPLETE | 624.772 | forecast-storkey | 0.5 | validation MAE regressed over 10% twice |
| 16 | COMPLETE | 20.305 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 17 | COMPLETE | 20.925 | baseline | learnable | validation MAE regressed over 10% twice |
| 18 | COMPLETE | 21.333 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 19 | COMPLETE | 21.702 | baseline | 2.0 | no validation MAE improvement for 4 epochs |
| 20 | COMPLETE | 20.160 | baseline | learnable | no validation MAE improvement for 4 epochs |
| 21 | PRUNED | 985.832 | forecast-storkey | learnable | HyperbandPruner at epoch 3 |
| 22 | PRUNED | 39.328 | projection | 0.5 | HyperbandPruner at epoch 3 |
| 23 | COMPLETE | 20.416 | baseline | learnable | no validation MAE improvement for 4 epochs |

## Within-trial stage comparison (epoch MAE traces)

- **Trial 20** (baseline, str=learnable, lr=0.002464, η=1.02e-05, steps=17, best 20.160): e1:22.17 → e2:21.29 → e3:20.16 → e4:21.36 → e5:20.42 → e6:20.41 → e7:20.37
- **Trial 16** (baseline, str=learnable, lr=0.003077, η=1.26e-05, steps=16, best 20.305): e1:22.51 → e2:20.31 → e3:20.80 → e4:20.68 → e5:20.47 → e6:20.56
- **Trial 14** (baseline, str=learnable, lr=0.003543, η=1.38e-05, steps=16, best 20.406): e1:21.34 → e2:20.41 → e3:24.56 → e4:21.14 → e5:20.61 → e6:20.87
- **Trial 23** (baseline, str=learnable, lr=0.00368, η=1.58e-05, steps=16, best 20.416): e1:22.44 → e2:24.54 → e3:20.79 → e4:22.39 → e5:20.42 → e6:20.50 → e7:20.58 → e8:20.50 → e9:20.49
- **Trial 8** (baseline, str=learnable, lr=0.003525, η=1.47e-05, steps=16, best 20.487): e1:22.88 → e2:20.49 → e3:22.12 → e4:23.64 → e5:20.94 → e6:21.06
- **Trial 17** (baseline, str=learnable, lr=0.003783, η=2.29e-05, steps=17, best 20.925): e1:29.84 → e2:20.93 → e3:23.62 → e4:42.64

## How to read this report

- **Trial** = one hyperparameter recipe (including Hopfield variant/strength).
- **Epoch** = one pass over training data inside that trial (learning-curve x-axis).
- **Pruned** = Hyperband stopped a weak trial early (saves compute, not a crash).
- **Early stop / patience** = val MAE stopped improving → keep best checkpoint.
- Lower **MAE (mg/dL)** is better.

## Complete training report (all trials)

### Trial 0 — COMPLETE (best MAE 21.494)

- **Variant**: `baseline` · strength `1.0`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 21.494 |
| 2 | 22.610 |
| 3 | 22.746 |
| 4 | 21.857 |
| 5 | 30.045 |

### Trial 1 — COMPLETE (best MAE 25.020)

- **Variant**: `projection` · strength `1.0`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 25.020 |
| 2 | 25.758 |
| 3 | 51.912 |
| 4 | 32.787 |

### Trial 2 — PRUNED (best MAE 43.611)

- **Variant**: `embed-storkey` · strength `1.0`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 45.996 |
| 2 | 48.340 |
| 3 | 43.611 |

### Trial 3 — COMPLETE (best MAE 45.465)

- **Variant**: `embed-storkey` · strength `2.0`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=2.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 49.239 |
| 2 | 45.501 |
| 3 | 45.465 |
| 4 | 45.527 |
| 5 | 45.527 |
| 6 | 45.593 |
| 7 | 45.732 |

### Trial 4 — PRUNED (best MAE 45.380)

- **Variant**: `embed-storkey` · strength `learnable`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=learnable, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 45.589 |
| 2 | 46.555 |
| 3 | 45.380 |

### Trial 5 — COMPLETE (best MAE 67.431)

- **Variant**: `forecast-storkey` · strength `1.0`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=forecast-storkey, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 98.199 |
| 2 | 93.674 |
| 3 | 67.431 |
| 4 | 1240.155 |
| 5 | 1573.548 |

### Trial 6 — COMPLETE (best MAE 43.528)

- **Variant**: `embed-storkey` · strength `1.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=1.5, lr=0.003533, eta_infer=2.207e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=21

| Epoch | Val MAE |
|------:|--------:|
| 1 | 45.157 |
| 2 | 47.043 |
| 3 | 47.274 |
| 4 | 43.528 |
| 5 | 45.490 |
| 6 | 45.637 |
| 7 | 45.394 |
| 8 | 45.415 |

### Trial 7 — PRUNED (best MAE 44.566)

- **Variant**: `embed-storkey` · strength `2.0`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=2.0, lr=0.002667, eta_infer=9.998e-06, infer_steps=14, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=21

| Epoch | Val MAE |
|------:|--------:|
| 1 | 45.613 |
| 2 | 44.566 |
| 3 | 45.738 |

### Trial 8 — COMPLETE (best MAE 20.487)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003525, eta_infer=1.466e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01661, seed_offset=33

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.876 |
| 2 | 20.487 |
| 3 | 22.125 |
| 4 | 23.638 |
| 5 | 20.937 |
| 6 | 21.057 |

### Trial 9 — PRUNED (best MAE 25.633)

- **Variant**: `projection` · strength `2.0`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=2.0, lr=0.002269, eta_infer=1.516e-05, infer_steps=17, max_infer_norm=1, grad_clip=1, lr_decay_epochs=15, weight_init_std=0.01967, seed_offset=23

| Epoch | Val MAE |
|------:|--------:|
| 1 | 42.796 |
| 2 | 25.633 |
| 3 | 37.629 |

### Trial 10 — COMPLETE (best MAE 21.049)

- **Variant**: `baseline` · strength `0.5`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=0.5, lr=0.003352, eta_infer=1.267e-05, infer_steps=13, max_infer_norm=0.5, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01714, seed_offset=37

| Epoch | Val MAE |
|------:|--------:|
| 1 | 23.874 |
| 2 | 27.970 |
| 3 | 21.049 |
| 4 | 37.639 |
| 5 | 22.689 |
| 6 | 22.580 |
| 7 | 21.873 |

### Trial 11 — PRUNED (best MAE 24.526)

- **Variant**: `projection` · strength `learnable`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=learnable, lr=0.002385, eta_infer=1.804e-05, infer_steps=18, max_infer_norm=0.5, grad_clip=0.5, lr_decay_epochs=15, weight_init_std=0.01469, seed_offset=17

| Epoch | Val MAE |
|------:|--------:|
| 1 | 40.037 |
| 2 | 24.526 |
| 3 | 24.968 |

### Trial 12 — PRUNED (best MAE 22.243)

- **Variant**: `baseline` · strength `1.5`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.5, lr=0.002201, eta_infer=1.879e-05, infer_steps=17, max_infer_norm=0.5, grad_clip=1, lr_decay_epochs=5, weight_init_std=0.01939, seed_offset=40

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.243 |
| 2 | 24.378 |
| 3 | 26.102 |

### Trial 13 — COMPLETE (best MAE 27.651)

- **Variant**: `projection` · strength `1.0`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=1.0, lr=0.003306, eta_infer=2.121e-05, infer_steps=18, max_infer_norm=0.5, grad_clip=1, lr_decay_epochs=15, weight_init_std=0.01796, seed_offset=35

| Epoch | Val MAE |
|------:|--------:|
| 1 | 29.721 |
| 2 | 38.223 |
| 3 | 29.995 |
| 4 | 27.651 |
| 5 | 34.326 |
| 6 | 39.244 |

### Trial 14 — COMPLETE (best MAE 20.406)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003543, eta_infer=1.377e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01717, seed_offset=29

| Epoch | Val MAE |
|------:|--------:|
| 1 | 21.337 |
| 2 | 20.406 |
| 3 | 24.560 |
| 4 | 21.138 |
| 5 | 20.612 |
| 6 | 20.867 |

### Trial 15 — COMPLETE (best MAE 624.772)

- **Variant**: `forecast-storkey` · strength `0.5`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=forecast-storkey, hopfield_strength=0.5, lr=0.003085, eta_infer=1.3e-05, infer_steps=12, max_infer_norm=0.5, grad_clip=1, lr_decay_epochs=10, weight_init_std=0.01643, seed_offset=29

| Epoch | Val MAE |
|------:|--------:|
| 1 | 624.772 |
| 2 | 3195.408 |
| 3 | 4703.262 |

### Trial 16 — COMPLETE (best MAE 20.305)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003077, eta_infer=1.256e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01632, seed_offset=23

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.508 |
| 2 | 20.305 |
| 3 | 20.799 |
| 4 | 20.681 |
| 5 | 20.467 |
| 6 | 20.557 |

### Trial 17 — COMPLETE (best MAE 20.925)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003783, eta_infer=2.286e-05, infer_steps=17, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01646, seed_offset=33

| Epoch | Val MAE |
|------:|--------:|
| 1 | 29.843 |
| 2 | 20.925 |
| 3 | 23.619 |
| 4 | 42.643 |

### Trial 18 — COMPLETE (best MAE 21.333)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.003256, eta_infer=1.61e-05, infer_steps=14, max_infer_norm=1, grad_clip=1, lr_decay_epochs=5, weight_init_std=0.01906, seed_offset=33

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.419 |
| 2 | 22.502 |
| 3 | 24.280 |
| 4 | 22.448 |
| 5 | 21.333 |
| 6 | 21.361 |
| 7 | 21.656 |
| 8 | 21.411 |
| 9 | 21.499 |

### Trial 19 — COMPLETE (best MAE 21.702)

- **Variant**: `baseline` · strength `2.0`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=2.0, lr=0.003792, eta_infer=1.775e-05, infer_steps=16, max_infer_norm=0.5, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01727, seed_offset=37

| Epoch | Val MAE |
|------:|--------:|
| 1 | 41.152 |
| 2 | 21.702 |
| 3 | 21.935 |
| 4 | 22.089 |
| 5 | 21.742 |
| 6 | 21.741 |

### Trial 20 — COMPLETE (best MAE 20.160)

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

### Trial 21 — PRUNED (best MAE 985.832)

- **Variant**: `forecast-storkey` · strength `learnable`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=forecast-storkey, hopfield_strength=learnable, lr=0.003661, eta_infer=1.237e-05, infer_steps=17, max_infer_norm=0.5, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01982, seed_offset=18

| Epoch | Val MAE |
|------:|--------:|
| 1 | 1217.928 |
| 2 | 1075.810 |
| 3 | 985.832 |

### Trial 22 — PRUNED (best MAE 39.328)

- **Variant**: `projection` · strength `0.5`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=0.5, lr=0.0037, eta_infer=1.015e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01679, seed_offset=26

| Epoch | Val MAE |
|------:|--------:|
| 1 | 48.852 |
| 2 | 39.328 |
| 3 | 44.049 |

### Trial 23 — COMPLETE (best MAE 20.416)

- **Variant**: `baseline` · strength `learnable`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=learnable, lr=0.00368, eta_infer=1.58e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=5, weight_init_std=0.01573, seed_offset=13

| Epoch | Val MAE |
|------:|--------:|
| 1 | 22.438 |
| 2 | 24.541 |
| 3 | 20.788 |
| 4 | 22.393 |
| 5 | 20.416 |
| 6 | 20.503 |
| 7 | 20.582 |
| 8 | 20.495 |
| 9 | 20.495 |

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
