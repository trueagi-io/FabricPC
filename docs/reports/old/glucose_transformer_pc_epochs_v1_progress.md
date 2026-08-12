# Glucose PC Optuna progress report

Generated: `2026-07-24T08:13:22.795965+00:00`  
Study: `glucose_transformer_pc_epochs_v1`  
Mode: predictive coding (PC) only

## Summary

| Metric | Value |
|--------|------:|
| Trials recorded | 35 |
| Complete | 13 |
| Pruned | 18 |
| Failed | 4 |
| Running | 0 |
| Best complete trial | 28 |
| Best val MAE (mg/dL) | 19.0963 |

## What helped (auto-generated from top trials)

- **seq_len=64** dominates top 5 (5/5)
- **depth=1** dominates top 5 (4/5)
- **num_heads=1** dominates top 5 (4/5)
- **lr**: range 0.0003367–0.0005683, median 0.0005432
- **eta_infer**: range 2.889e-05–5.888e-05, median 3.819e-05
- **weight_init_std**: range 0.01171–0.02297, median 0.0194
- **infer_steps**: 20×2, 21×1, 22×1, 15×1
- **grad_clip**: 2.0×3, 0.5×2

## Top model architectures

### #1 — Trial 28 (MAE 19.096)

- **Geometry**: seq_len=64, depth=1, heads=4
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=4, lr=0.0003367, eta_infer=3.17e-05, infer_steps=21, max_infer_norm=5, grad_clip=2, weight_init_std=0.02297

### #2 — Trial 21 (MAE 19.135)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.0005683, eta_infer=4.022e-05, infer_steps=20, max_infer_norm=5, grad_clip=2, weight_init_std=0.01946

### #3 — Trial 30 (MAE 19.398)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.0005651, eta_infer=3.819e-05, infer_steps=20, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.0194

### #4 — Trial 29 (MAE 19.447)

- **Geometry**: seq_len=64, depth=2, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=2, num_heads=1, lr=0.0004468, eta_infer=2.889e-05, infer_steps=22, max_infer_norm=5, grad_clip=2, weight_init_std=0.01171

### #5 — Trial 34 (MAE 19.575)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.0005432, eta_infer=5.888e-05, infer_steps=15, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01836

## Complete-trial leaderboard

| Trial | Best MAE | MARD% | Geometry | LR | η_infer | Infer steps | Grad clip |
|------:|---------:|------:|----------|---:|--------:|------------:|----------:|
| 28 | 19.096 | 14.72 | 64/d1/h4 | 0.0003367 | 3.17e-05 | 21 | 2.0 |
| 21 | 19.135 | 14.95 | 64/d1/h1 | 0.0005683 | 4.022e-05 | 20 | 2.0 |
| 30 | 19.398 | 14.50 | 64/d1/h1 | 0.0005651 | 3.819e-05 | 20 | 0.5 |
| 29 | 19.447 | 15.19 | 64/d2/h1 | 0.0004468 | 2.889e-05 | 22 | 2.0 |
| 34 | 19.575 | 15.00 | 64/d1/h1 | 0.0005432 | 5.888e-05 | 15 | 0.5 |
| 31 | 19.584 | 14.96 | 64/d1/h4 | 0.0003646 | 5.101e-05 | 17 | 2.0 |
| 15 | 19.893 | 14.85 | 64/d1/h1 | 0.0005182 | 0.0001051 | 18 | 2.0 |
| 2 | 19.926 | 15.78 | 128/d2/h2 | 0.001844 | 3.614e-05 | 11 | 2.0 |
| 17 | 19.983 | 15.83 | 64/d2/h1 | 0.000752 | 0.0004862 | 9 | 0.5 |
| 4 | 24.548 | 19.38 | 128/d1/h4 | 0.0004313 | 0.0003144 | 14 | 0.5 |
| 12 | 28.806 | 24.57 | 128/d1/h2 | 0.001115 | 0.0003493 | 24 | 1.0 |
| 32 | 34.776 | 29.42 | 64/d1/h1 | 0.0003216 | 1.429e-05 | 17 | 0.5 |
| 16 | 38.907 | 31.30 | 64/d2/h2 | 0.0006891 | 3.903e-05 | 24 | 1.0 |

## Best MAE by trial

| Trial | State | Best MAE (mg/dL) |
|------:|-------|-----------------:|
| 0 | PRUNED | 45.340 |
| 1 | PRUNED | 48.485 |
| 2 | COMPLETE | 19.926 |
| 3 | PRUNED | 20.543 |
| 4 | COMPLETE | 24.548 |
| 5 | PRUNED | 20.644 |
| 7 | FAIL | 19.633 |
| 8 | PRUNED | 32.784 |
| 9 | PRUNED | 42.036 |
| 10 | FAIL | 19.824 |
| 11 | FAIL | 19.187 |
| 12 | COMPLETE | 28.806 |
| 13 | PRUNED | 20.430 |
| 14 | PRUNED | 30.735 |
| 15 | COMPLETE | 19.893 |
| 16 | COMPLETE | 38.907 |
| 17 | COMPLETE | 19.983 |
| 18 | PRUNED | 20.129 |
| 19 | PRUNED | 29.741 |
| 21 | COMPLETE | 19.135 |
| 22 | PRUNED | 22.831 |
| 23 | PRUNED | 21.425 |
| 24 | PRUNED | 25.112 |
| 25 | PRUNED | 20.786 |
| 26 | PRUNED | 19.861 |
| 27 | PRUNED | 20.717 |
| 28 | COMPLETE | 19.096 |
| 29 | COMPLETE | 19.447 |
| 30 | COMPLETE | 19.398 |
| 31 | COMPLETE | 19.584 |
| 32 | COMPLETE | 34.776 |
| 33 | FAIL | 19.598 |
| 34 | COMPLETE | 19.575 |

## Top trial MAE traces (every 200 updates)

- **Trial 28** (best 19.096): 1:21.90 → 2:20.17 → 3:20.52 → 4:20.17 → 5:21.63 → 6:20.51 → 7:20.61 → 8:19.20 → 9:19.10 → 10:19.12 → 11:19.42 → 12:19.21 → 13:19.12
- **Trial 21** (best 19.135): 1:22.86 → 2:20.68 → 3:19.37 → 4:19.33 → 5:19.48 → 6:19.36 → 7:19.36 → 8:19.18 → 9:19.13 → 10:19.23 → 11:19.92 → 12:19.35 → 13:19.17
- **Trial 30** (best 19.398): 1:23.21 → 2:21.83 → 3:19.40 → 4:19.63 → 5:26.03 → 6:78.80
- **Trial 29** (best 19.447): 1:22.52 → 2:20.21 → 3:21.07 → 4:19.63 → 5:20.12 → 6:20.07 → 7:19.77 → 8:19.47 → 9:19.45 → 10:19.73 → 11:21.12 → 12:19.61 → 13:19.45 → 14:19.67 → 15:20.44

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

### Trial 0 — PRUNED (best MAE 45.340)

- **Geometry**: `64/d3/h2` · readout `None`
- **Stop / prune reason**: energy explosion at epoch 2, step 2189: 1.46867
- **Params**: lr=0.002956, eta_infer=1.585e-05, infer_steps=23, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.02162

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 45.340 | 36.36 |

### Trial 1 — PRUNED (best MAE 48.485)

- **Geometry**: `128/d1/h1` · readout `None`
- **Stop / prune reason**: energy explosion at epoch 2, step 1915: 1.17159
- **Params**: lr=0.002405, eta_infer=3.29e-05, infer_steps=19, max_infer_norm=5, grad_clip=1, weight_init_std=0.02731

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 48.485 | 34.07 |

### Trial 2 — COMPLETE (best MAE 19.926)

- **Geometry**: `128/d2/h2` · readout `None`
- **Stop / prune reason**: —
- **Params**: lr=0.001844, eta_infer=3.614e-05, infer_steps=11, max_infer_norm=5, grad_clip=2, weight_init_std=0.01069

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 26.851 | 18.80 |
| 2 | 22.659 | 17.19 |
| 3 | 22.140 | 17.98 |
| 4 | 21.507 | 17.32 |
| 5 | 20.834 | 16.53 |
| 6 | 21.278 | 16.58 |
| 7 | 20.660 | 16.59 |
| 8 | 21.246 | 16.06 |
| 9 | 20.830 | 16.08 |
| 10 | 20.179 | 16.24 |
| 11 | 20.608 | 16.84 |
| 12 | 19.940 | 15.71 |
| 13 | 19.950 | 15.83 |
| 14 | 19.983 | 15.94 |
| 15 | 19.926 | 15.78 |

### Trial 3 — PRUNED (best MAE 20.543)

- **Geometry**: `128/d2/h1` · readout `None`
- **Stop / prune reason**: energy explosion at epoch 3, step 2973: 0.0363412
- **Params**: lr=0.0015, eta_infer=0.0002975, infer_steps=20, max_infer_norm=0.5, grad_clip=2, weight_init_std=0.02277

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 20.543 | 15.50 |
| 2 | 27.162 | 24.07 |

### Trial 4 — COMPLETE (best MAE 24.548)

- **Geometry**: `128/d1/h4` · readout `None`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: lr=0.0004313, eta_infer=0.0003144, infer_steps=14, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.02227

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 33.338 | 24.31 |
| 2 | 31.107 | 24.07 |
| 3 | 30.406 | 25.56 |
| 4 | 28.156 | 22.01 |
| 5 | 28.820 | 21.20 |
| 6 | 27.384 | 22.88 |
| 7 | 24.971 | 19.99 |
| 8 | 24.548 | 19.38 |
| 9 | 25.485 | 19.36 |
| 10 | 25.969 | 21.82 |
| 11 | 26.411 | 22.49 |
| 12 | 25.585 | 19.45 |

### Trial 5 — PRUNED (best MAE 20.644)

- **Geometry**: `64/d2/h2` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: lr=0.0007457, eta_infer=1.462e-05, infer_steps=8, max_infer_norm=1, grad_clip=2, weight_init_std=0.02313

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 26.655 | 23.32 |
| 2 | 20.644 | 16.19 |
| 3 | 22.145 | 15.95 |

### Trial 6 — PRUNED (best MAE —)

- **Geometry**: `128/d1/h1` · readout `None`
- **Stop / prune reason**: energy explosion at epoch 1, step 876: 3.69142
- **Params**: lr=0.004803, eta_infer=0.0002414, infer_steps=11, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01054

### Trial 7 — FAIL (best MAE 19.633)

- **Geometry**: `64/d3/h1` · readout `None`
- **Stop / prune reason**: TimeoutError('trial exceeded 7200 seconds')
- **Params**: lr=0.000383, eta_infer=1.194e-05, infer_steps=23, max_infer_norm=0.5, grad_clip=2, weight_init_std=0.01165

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 23.067 | 20.15 |
| 2 | 19.813 | 15.23 |
| 3 | 21.338 | 15.30 |
| 4 | 19.706 | 14.91 |
| 5 | 20.743 | 17.26 |
| 6 | 21.208 | 15.24 |
| 7 | 19.926 | 15.74 |
| 8 | 19.633 | 14.84 |
| 9 | 19.712 | 15.61 |
| 10 | 19.807 | 15.79 |

### Trial 8 — PRUNED (best MAE 32.784)

- **Geometry**: `128/d1/h2` · readout `None`
- **Stop / prune reason**: energy explosion at epoch 3, step 3118: 0.485937
- **Params**: lr=0.0008918, eta_infer=0.0001831, infer_steps=17, max_infer_norm=1, grad_clip=1, weight_init_std=0.01996

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 43.691 | 29.16 |
| 2 | 32.784 | 28.96 |

### Trial 9 — PRUNED (best MAE 42.036)

- **Geometry**: `128/d3/h4` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: lr=0.0004703, eta_infer=1.522e-05, infer_steps=11, max_infer_norm=5, grad_clip=1, weight_init_std=0.01668

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 42.036 | 32.29 |
| 2 | 46.864 | 39.36 |
| 3 | 45.825 | 38.71 |

### Trial 10 — FAIL (best MAE 19.824)

- **Geometry**: `128/d3/h1` · readout `None`
- **Stop / prune reason**: TimeoutError('trial exceeded 7200 seconds')
- **Params**: lr=0.0007603, eta_infer=0.0004921, infer_steps=10, max_infer_norm=1, grad_clip=2, weight_init_std=0.01698

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 21.840 | 16.47 |
| 2 | 20.700 | 16.28 |
| 3 | 20.657 | 15.36 |
| 4 | 20.772 | 16.95 |
| 5 | 20.195 | 15.60 |
| 6 | 19.970 | 15.89 |
| 7 | 19.868 | 15.11 |
| 8 | 19.824 | 15.23 |
| 9 | 20.867 | 15.57 |
| 10 | 19.865 | 15.75 |
| 11 | 21.143 | 17.51 |

### Trial 11 — FAIL (best MAE 19.187)

- **Geometry**: `128/d3/h4` · readout `None`
- **Stop / prune reason**: TimeoutError('trial exceeded 7200 seconds')
- **Params**: lr=0.002241, eta_infer=2.645e-05, infer_steps=13, max_infer_norm=5, grad_clip=2, weight_init_std=0.01421

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 20.933 | 16.83 |
| 2 | 22.210 | 17.58 |
| 3 | 21.000 | 15.49 |
| 4 | 20.796 | 17.40 |
| 5 | 19.249 | 15.03 |
| 6 | 19.538 | 15.60 |
| 7 | 19.187 | 14.95 |
| 8 | 19.327 | 15.32 |
| 9 | 19.331 | 15.34 |
| 10 | 19.532 | 15.71 |

### Trial 12 — COMPLETE (best MAE 28.806)

- **Geometry**: `128/d1/h2` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.001115, eta_infer=0.0003493, infer_steps=24, max_infer_norm=1, grad_clip=1, weight_init_std=0.01375

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 28.806 | 24.57 |
| 2 | 37.169 | 33.55 |
| 3 | 43.264 | 38.05 |

### Trial 13 — PRUNED (best MAE 20.430)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: lr=0.00309, eta_infer=5.115e-05, infer_steps=14, max_infer_norm=0.5, grad_clip=2, weight_init_std=0.01299

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 21.822 | 18.26 |
| 2 | 20.430 | 15.66 |
| 3 | 21.418 | 15.68 |

### Trial 14 — PRUNED (best MAE 30.735)

- **Geometry**: `64/d3/h1` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: lr=0.003033, eta_infer=0.0001108, infer_steps=22, max_infer_norm=5, grad_clip=2, weight_init_std=0.02634

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 32.056 | 27.16 |
| 2 | 34.542 | 24.33 |
| 3 | 30.735 | 21.91 |

### Trial 15 — COMPLETE (best MAE 19.893)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.0005182, eta_infer=0.0001051, infer_steps=18, max_infer_norm=5, grad_clip=2, weight_init_std=0.01764

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 22.599 | 19.32 |
| 2 | 19.893 | 14.85 |
| 3 | 78.020 | 60.07 |
| 4 | 71.013 | 55.61 |

### Trial 16 — COMPLETE (best MAE 38.907)

- **Geometry**: `64/d2/h2` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.0006891, eta_infer=3.903e-05, infer_steps=24, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.01405

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 38.907 | 31.30 |
| 2 | 45.903 | 40.86 |
| 3 | 45.438 | 35.09 |

### Trial 17 — COMPLETE (best MAE 19.983)

- **Geometry**: `64/d2/h1` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.000752, eta_infer=0.0004862, infer_steps=9, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.02789

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 28.194 | 24.95 |
| 2 | 20.863 | 15.30 |
| 3 | 20.460 | 15.02 |
| 4 | 19.983 | 15.83 |
| 5 | 64.940 | 50.04 |
| 6 | 38.519 | 32.20 |

### Trial 18 — PRUNED (best MAE 20.129)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 9
- **Params**: lr=0.003081, eta_infer=0.0001338, infer_steps=20, max_infer_norm=5, grad_clip=2, weight_init_std=0.0102

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 23.375 | 19.37 |
| 2 | 23.497 | 19.68 |
| 3 | 24.259 | 20.79 |
| 4 | 24.725 | 20.18 |
| 5 | 21.275 | 16.86 |
| 6 | 23.339 | 16.61 |
| 7 | 22.472 | 19.31 |
| 8 | 20.746 | 16.51 |
| 9 | 20.129 | 16.39 |

### Trial 19 — PRUNED (best MAE 29.741)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: energy explosion at epoch 2, step 2250: 0.997312
- **Params**: lr=0.001053, eta_infer=0.0002164, infer_steps=14, max_infer_norm=1, grad_clip=2, weight_init_std=0.01321

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 29.741 | 25.48 |

### Trial 20 — PRUNED (best MAE —)

- **Geometry**: `64/d2/h4` · readout `None`
- **Stop / prune reason**: energy explosion at epoch 1, step 661: 0.950826
- **Params**: lr=0.0006259, eta_infer=2.898e-05, infer_steps=19, max_infer_norm=5, grad_clip=1, weight_init_std=0.02449

### Trial 21 — COMPLETE (best MAE 19.135)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: lr=0.0005683, eta_infer=4.022e-05, infer_steps=20, max_infer_norm=5, grad_clip=2, weight_init_std=0.01946

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 22.859 | 19.59 |
| 2 | 20.679 | 15.36 |
| 3 | 19.367 | 14.51 |
| 4 | 19.326 | 14.80 |
| 5 | 19.479 | 15.68 |
| 6 | 19.363 | 14.56 |
| 7 | 19.356 | 15.40 |
| 8 | 19.176 | 14.83 |
| 9 | 19.135 | 14.95 |
| 10 | 19.227 | 15.15 |
| 11 | 19.917 | 16.34 |
| 12 | 19.347 | 15.43 |
| 13 | 19.166 | 14.76 |

### Trial 22 — PRUNED (best MAE 22.831)

- **Geometry**: `128/d2/h2` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 9
- **Params**: lr=0.00218, eta_infer=0.0001265, infer_steps=10, max_infer_norm=5, grad_clip=2, weight_init_std=0.01175

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 28.693 | 23.59 |
| 2 | 35.036 | 31.51 |
| 3 | 28.577 | 24.63 |
| 4 | 25.200 | 20.28 |
| 5 | 22.982 | 18.01 |
| 6 | 22.831 | 18.27 |
| 7 | 30.082 | 22.26 |
| 8 | 24.037 | 18.22 |
| 9 | 24.063 | 19.37 |

### Trial 23 — PRUNED (best MAE 21.425)

- **Geometry**: `128/d2/h2` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 3
- **Params**: lr=0.001469, eta_infer=3.322e-05, infer_steps=16, max_infer_norm=1, grad_clip=2, weight_init_std=0.01024

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 21.425 | 16.70 |
| 2 | 22.596 | 17.80 |
| 3 | 23.434 | 17.02 |

### Trial 24 — PRUNED (best MAE 25.112)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: energy explosion at epoch 3, step 3164: 0.0944476
- **Params**: lr=0.00103, eta_infer=0.0003752, infer_steps=18, max_infer_norm=5, grad_clip=2, weight_init_std=0.01872

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 144.442 | 105.69 |
| 2 | 25.112 | 19.07 |

### Trial 25 — PRUNED (best MAE 20.786)

- **Geometry**: `128/d1/h2` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 9
- **Params**: lr=0.001392, eta_infer=1.018e-05, infer_steps=10, max_infer_norm=5, grad_clip=2, weight_init_std=0.01411

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 23.496 | 18.02 |
| 2 | 23.870 | 19.74 |
| 3 | 21.221 | 16.29 |
| 4 | 20.890 | 16.33 |
| 5 | 20.820 | 16.45 |
| 6 | 20.793 | 16.32 |
| 7 | 20.824 | 16.50 |
| 8 | 20.802 | 16.43 |
| 9 | 20.786 | 16.40 |

### Trial 26 — PRUNED (best MAE 19.861)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 9
- **Params**: lr=0.0007214, eta_infer=6.548e-05, infer_steps=19, max_infer_norm=5, grad_clip=2, weight_init_std=0.02365

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 23.201 | 19.85 |
| 2 | 21.001 | 15.54 |
| 3 | 21.215 | 15.34 |
| 4 | 20.544 | 16.40 |
| 5 | 20.296 | 16.36 |
| 6 | 20.170 | 14.67 |
| 7 | 20.204 | 16.12 |
| 8 | 19.984 | 15.71 |
| 9 | 19.861 | 15.55 |

### Trial 27 — PRUNED (best MAE 20.717)

- **Geometry**: `128/d2/h4` · readout `None`
- **Stop / prune reason**: HyperbandPruner at epoch 9
- **Params**: lr=0.002092, eta_infer=4.138e-05, infer_steps=14, max_infer_norm=5, grad_clip=2, weight_init_std=0.01179

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 24.663 | 18.80 |
| 2 | 23.082 | 17.28 |
| 3 | 26.024 | 22.45 |
| 4 | 21.832 | 17.96 |
| 5 | 21.024 | 16.61 |
| 6 | 20.924 | 16.78 |
| 7 | 20.838 | 16.14 |
| 8 | 20.717 | 15.86 |
| 9 | 20.759 | 17.17 |

### Trial 28 — COMPLETE (best MAE 19.096)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: lr=0.0003367, eta_infer=3.17e-05, infer_steps=21, max_infer_norm=5, grad_clip=2, weight_init_std=0.02297

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 21.899 | 18.77 |
| 2 | 20.174 | 15.22 |
| 3 | 20.524 | 14.89 |
| 4 | 20.170 | 14.95 |
| 5 | 21.631 | 18.20 |
| 6 | 20.509 | 14.93 |
| 7 | 20.608 | 16.42 |
| 8 | 19.202 | 14.71 |
| 9 | 19.096 | 14.72 |
| 10 | 19.123 | 14.83 |
| 11 | 19.416 | 15.52 |
| 12 | 19.213 | 15.08 |
| 13 | 19.115 | 14.71 |

### Trial 29 — COMPLETE (best MAE 19.447)

- **Geometry**: `64/d2/h1` · readout `None`
- **Stop / prune reason**: —
- **Params**: lr=0.0004468, eta_infer=2.889e-05, infer_steps=22, max_infer_norm=5, grad_clip=2, weight_init_std=0.01171

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 22.521 | 19.50 |
| 2 | 20.211 | 15.25 |
| 3 | 21.070 | 15.10 |
| 4 | 19.628 | 14.85 |
| 5 | 20.119 | 16.51 |
| 6 | 20.069 | 14.73 |
| 7 | 19.766 | 15.93 |
| 8 | 19.466 | 14.90 |
| 9 | 19.455 | 14.85 |
| 10 | 19.726 | 15.83 |
| 11 | 21.124 | 17.79 |
| 12 | 19.613 | 15.70 |
| 13 | 19.447 | 15.19 |
| 14 | 19.665 | 15.82 |
| 15 | 20.441 | 16.96 |

### Trial 30 — COMPLETE (best MAE 19.398)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.0005651, eta_infer=3.819e-05, infer_steps=20, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.0194

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 23.213 | 19.99 |
| 2 | 21.830 | 16.24 |
| 3 | 19.398 | 14.50 |
| 4 | 19.627 | 14.85 |
| 5 | 26.028 | 19.88 |
| 6 | 78.805 | 57.59 |

### Trial 31 — COMPLETE (best MAE 19.584)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: no validation MAE improvement for 4 epochs
- **Params**: lr=0.0003646, eta_infer=5.101e-05, infer_steps=17, max_infer_norm=5, grad_clip=2, weight_init_std=0.01915

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 21.532 | 18.31 |
| 2 | 19.961 | 15.11 |
| 3 | 20.456 | 14.90 |
| 4 | 19.584 | 14.96 |
| 5 | 21.000 | 17.48 |
| 6 | 20.164 | 14.85 |
| 7 | 20.838 | 16.79 |
| 8 | 20.791 | 15.67 |

### Trial 32 — COMPLETE (best MAE 34.776)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.0003216, eta_infer=1.429e-05, infer_steps=17, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.02057

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 34.776 | 29.42 |
| 2 | 38.818 | 33.22 |
| 3 | 46.128 | 36.33 |

### Trial 33 — FAIL (best MAE 19.598)

- **Geometry**: `64/d1/h2` · readout `None`
- **Stop / prune reason**: —
- **Params**: lr=0.000346, eta_infer=2.163e-05, infer_steps=24, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.02132

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 22.950 | 19.82 |
| 2 | 20.512 | 15.61 |
| 3 | 22.092 | 15.80 |
| 4 | 19.943 | 14.94 |
| 5 | 21.901 | 18.52 |
| 6 | 20.545 | 15.13 |
| 7 | 20.290 | 16.32 |
| 8 | 19.598 | 14.90 |

### Trial 34 — COMPLETE (best MAE 19.575)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.0005432, eta_infer=5.888e-05, infer_steps=15, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01836

| Epoch | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 1 | 21.286 | 17.90 |
| 2 | 21.012 | 15.92 |
| 3 | 21.564 | 17.99 |
| 4 | 19.575 | 15.00 |
| 5 | 33.451 | 25.48 |
| 6 | 110.039 | 87.01 |

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
