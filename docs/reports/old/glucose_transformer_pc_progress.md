# Glucose PC Optuna progress report

Generated: `2026-07-24T08:13:36.570626+00:00`  
Study: `glucose_transformer_pc`  
Mode: predictive coding (PC) only

## Summary

| Metric | Value |
|--------|------:|
| Trials recorded | 34 |
| Complete | 7 |
| Pruned | 25 |
| Failed | 0 |
| Running | 2 |
| Best complete trial | 21 |
| Best val MAE (mg/dL) | 20.3776 |

## What helped (auto-generated from top trials)

- **seq_len=64** dominates top 5 (5/5)
- **depth=1** dominates top 5 (4/5)
- **num_heads=1** dominates top 5 (3/5)
- **lr**: range 0.001943–0.00494, median 0.002787
- **eta_infer**: range 1.034e-05–1.532e-05, median 1.264e-05
- **weight_init_std**: range 0.01582–0.02059, median 0.01727
- **infer_steps**: 22×2, 16×2, 14×1
- **grad_clip**: 1.0×3, 0.5×2

## Top model architectures

### #1 — Trial 21 (MAE 20.378)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684

### #2 — Trial 20 (MAE 20.519)

- **Geometry**: seq_len=64, depth=1, heads=4
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=4, lr=0.002787, eta_infer=1.261e-05, infer_steps=22, max_infer_norm=5, grad_clip=1, weight_init_std=0.01727

### #3 — Trial 22 (MAE 20.684)

- **Geometry**: seq_len=64, depth=1, heads=4
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=4, lr=0.002613, eta_infer=1.034e-05, infer_steps=22, max_infer_norm=5, grad_clip=1, weight_init_std=0.01582

### #4 — Trial 9 (MAE 21.010)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.001943, eta_infer=1.264e-05, infer_steps=16, max_infer_norm=1, grad_clip=1, weight_init_std=0.02059

### #5 — Trial 18 (MAE 21.616)

- **Geometry**: seq_len=64, depth=2, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=2, num_heads=1, lr=0.00494, eta_infer=1.532e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.0177

## Complete-trial leaderboard

| Trial | Best MAE | MARD% | Geometry | LR | η_infer | Infer steps | Grad clip |
|------:|---------:|------:|----------|---:|--------:|------------:|----------:|
| 21 | 20.378 | 15.48 | 64/d1/h1 | 0.002975 | 1.281e-05 | 14 | 0.5 |
| 20 | 20.519 | 15.63 | 64/d1/h4 | 0.002787 | 1.261e-05 | 22 | 1.0 |
| 22 | 20.684 | 16.51 | 64/d1/h4 | 0.002613 | 1.034e-05 | 22 | 1.0 |
| 9 | 21.010 | 16.56 | 64/d1/h1 | 0.001943 | 1.264e-05 | 16 | 1.0 |
| 18 | 21.616 | 16.80 | 64/d2/h1 | 0.00494 | 1.532e-05 | 16 | 0.5 |
| 2 | 22.205 | 17.12 | 64/d1/h4 | 0.001246 | 8.419e-05 | 19 | 1.0 |
| 1 | 27.132 | 20.69 | 64/d1/h4 | 0.001316 | 0.0001633 | 14 | 2.0 |

## Best MAE by trial

| Trial | State | Best MAE (mg/dL) |
|------:|-------|-----------------:|
| 0 | PRUNED | 43.706 |
| 1 | COMPLETE | 27.132 |
| 2 | COMPLETE | 22.205 |
| 3 | PRUNED | 42.683 |
| 4 | PRUNED | 37.966 |
| 5 | PRUNED | 43.654 |
| 6 | PRUNED | 43.204 |
| 7 | PRUNED | 22.702 |
| 8 | PRUNED | 46.198 |
| 9 | COMPLETE | 21.010 |
| 10 | PRUNED | 43.356 |
| 11 | PRUNED | 41.960 |
| 12 | PRUNED | 44.866 |
| 13 | PRUNED | 25.909 |
| 14 | PRUNED | 43.274 |
| 15 | PRUNED | 43.422 |
| 16 | PRUNED | 41.968 |
| 17 | PRUNED | 27.090 |
| 18 | COMPLETE | 21.616 |
| 19 | PRUNED | 30.451 |
| 20 | COMPLETE | 20.519 |
| 21 | COMPLETE | 20.378 |
| 22 | COMPLETE | 20.684 |
| 23 | PRUNED | 21.665 |
| 24 | PRUNED | 24.631 |
| 25 | PRUNED | 24.882 |
| 26 | PRUNED | 39.812 |
| 29 | PRUNED | 25.930 |
| 30 | PRUNED | 24.236 |
| 31 | PRUNED | 34.191 |
| 32 | PRUNED | 40.938 |
| 33 | PRUNED | 21.364 |

## Top trial MAE traces (every 200 updates)

- **Trial 21** (best 20.378): 200:29.59 → 400:25.94 → 600:23.30 → 800:24.37 → 1000:26.11 → 1200:21.35 → 1400:21.41 → 1600:20.78 → 1800:20.63 → 2000:21.76 → 2200:20.38 → 2400:20.41 → 2600:22.04 → 2800:20.62 → 3000:21.53
- **Trial 20** (best 20.519): 200:31.01 → 400:29.22 → 600:24.14 → 800:24.97 → 1000:22.54 → 1200:21.81 → 1400:21.59 → 1600:21.61 → 1800:21.18 → 2000:21.03 → 2200:20.71 → 2400:21.46 → 2600:20.52 → 2800:21.15 → 3000:20.82
- **Trial 22** (best 20.684): 200:32.25 → 400:26.27 → 600:23.57 → 800:22.90 → 1000:22.46 → 1200:22.72 → 1400:21.46 → 1600:21.67 → 1800:20.70 → 2000:20.77 → 2200:20.68 → 2400:20.76 → 2600:20.96
- **Trial 9** (best 21.010): 200:33.25 → 400:29.15 → 600:24.65 → 800:23.42 → 1000:23.00 → 1200:28.36 → 1400:21.86 → 1600:22.07 → 1800:22.70 → 2000:22.06 → 2200:21.04 → 2400:21.78 → 2600:21.56 → 2800:21.01 → 3000:21.61

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

### Trial 0 — PRUNED (best MAE 43.706)

- **Geometry**: `128/d2/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001481, eta_infer=0.0003616, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01273

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 43.706 | 37.44 |

### Trial 1 — COMPLETE (best MAE 27.132)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.001316, eta_infer=0.0001633, infer_steps=14, max_infer_norm=1, grad_clip=2, weight_init_std=0.01415

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 39.318 | 30.84 |
| 400 | 37.206 | 33.47 |
| 600 | 30.507 | 24.60 |
| 800 | 28.779 | 22.20 |
| 1000 | 28.080 | 21.99 |
| 1200 | 27.132 | 20.69 |
| 1400 | 31.318 | 27.58 |
| 1600 | 40.662 | 36.52 |

### Trial 2 — COMPLETE (best MAE 22.205)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.001246, eta_infer=8.419e-05, infer_steps=19, max_infer_norm=5, grad_clip=1, weight_init_std=0.01682

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 37.635 | 29.46 |
| 400 | 31.104 | 26.65 |
| 600 | 27.553 | 20.67 |
| 800 | 25.916 | 21.25 |
| 1000 | 25.017 | 19.48 |
| 1200 | 25.188 | 21.00 |
| 1400 | 23.959 | 19.17 |
| 1600 | 25.672 | 21.87 |
| 1800 | 23.089 | 17.48 |
| 2000 | 23.724 | 17.72 |
| 2200 | 22.982 | 17.37 |
| 2400 | 23.495 | 19.68 |
| 2600 | 22.407 | 17.25 |
| 2800 | 22.205 | 17.12 |
| 3000 | 22.641 | 17.02 |
| 3200 | 23.434 | 19.82 |
| 3400 | 23.537 | 17.16 |

### Trial 3 — PRUNED (best MAE 42.683)

- **Geometry**: `128/d3/h2` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0009195, eta_infer=3.779e-05, infer_steps=9, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01333

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 42.683 | 35.69 |

### Trial 4 — PRUNED (best MAE 37.966)

- **Geometry**: `64/d1/h2` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001357, eta_infer=0.0001608, infer_steps=16, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.02524

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 37.966 | 30.35 |

### Trial 5 — PRUNED (best MAE 43.654)

- **Geometry**: `64/d3/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0005901, eta_infer=1.985e-05, infer_steps=16, max_infer_norm=5, grad_clip=0.5, weight_init_std=0.01464

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 43.654 | 36.29 |

### Trial 6 — PRUNED (best MAE 43.204)

- **Geometry**: `128/d3/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0006064, eta_infer=9.163e-05, infer_steps=14, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01566

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 43.204 | 35.81 |

### Trial 7 — PRUNED (best MAE 22.702)

- **Geometry**: `64/d1/h2` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 9
- **Params**: lr=0.001399, eta_infer=2.383e-05, infer_steps=8, max_infer_norm=5, grad_clip=1, weight_init_std=0.02423

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 35.383 | 28.23 |
| 400 | 27.915 | 22.29 |
| 600 | 25.913 | 21.05 |
| 800 | 25.723 | 18.68 |
| 1000 | 23.813 | 18.62 |
| 1200 | 26.663 | 18.94 |
| 1400 | 23.515 | 19.11 |
| 1600 | 22.702 | 17.84 |
| 1800 | 26.174 | 18.56 |

### Trial 8 — PRUNED (best MAE 46.198)

- **Geometry**: `128/d3/h2` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.004215, eta_infer=0.0004085, infer_steps=9, max_infer_norm=5, grad_clip=2, weight_init_std=0.0212

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 46.198 | 33.93 |

### Trial 9 — COMPLETE (best MAE 21.010)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.001943, eta_infer=1.264e-05, infer_steps=16, max_infer_norm=1, grad_clip=1, weight_init_std=0.02059

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.252 | 25.10 |
| 400 | 29.153 | 24.81 |
| 600 | 24.655 | 19.64 |
| 800 | 23.424 | 17.89 |
| 1000 | 23.003 | 17.64 |
| 1200 | 28.355 | 19.54 |
| 1400 | 21.861 | 17.03 |
| 1600 | 22.066 | 17.62 |
| 1800 | 22.702 | 16.73 |
| 2000 | 22.061 | 17.61 |
| 2200 | 21.041 | 16.47 |
| 2400 | 21.777 | 17.69 |
| 2600 | 21.557 | 17.32 |
| 2800 | 21.010 | 16.56 |
| 3000 | 21.608 | 16.65 |

### Trial 10 — PRUNED (best MAE 43.356)

- **Geometry**: `64/d2/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.000459, eta_infer=0.0001008, infer_steps=16, max_infer_norm=0.5, grad_clip=2, weight_init_std=0.02011

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 43.356 | 36.22 |

### Trial 11 — PRUNED (best MAE 41.960)

- **Geometry**: `128/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0005062, eta_infer=0.0001464, infer_steps=19, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.01092

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 41.960 | 35.09 |

### Trial 12 — PRUNED (best MAE 44.866)

- **Geometry**: `64/d3/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0004119, eta_infer=1.846e-05, infer_steps=9, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.0137

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 44.866 | 36.54 |

### Trial 13 — PRUNED (best MAE 25.909)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.001532, eta_infer=4.016e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.01191

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 34.469 | 27.20 |
| 400 | 27.715 | 22.81 |
| 600 | 25.909 | 19.36 |

### Trial 14 — PRUNED (best MAE 43.274)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.000317, eta_infer=6.905e-05, infer_steps=23, max_infer_norm=5, grad_clip=1, weight_init_std=0.01322

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 43.274 | 35.68 |

### Trial 15 — PRUNED (best MAE 43.422)

- **Geometry**: `64/d1/h2` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.0003483, eta_infer=0.0002437, infer_steps=24, max_infer_norm=1, grad_clip=2, weight_init_std=0.02638

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 43.422 | 35.70 |

### Trial 16 — PRUNED (best MAE 41.968)

- **Geometry**: `128/d2/h2` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001591, eta_infer=0.0001275, infer_steps=20, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01329

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 41.968 | 36.24 |

### Trial 17 — PRUNED (best MAE 27.090)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.001114, eta_infer=2.082e-05, infer_steps=18, max_infer_norm=1, grad_clip=1, weight_init_std=0.02205

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 36.729 | 29.91 |
| 400 | 29.479 | 24.36 |
| 600 | 27.090 | 22.38 |

### Trial 18 — COMPLETE (best MAE 21.616)

- **Geometry**: `64/d2/h1` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.00494, eta_infer=1.532e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.0177

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.632 | 24.94 |
| 400 | 27.037 | 23.06 |
| 600 | 24.057 | 19.30 |
| 800 | 22.409 | 17.76 |
| 1000 | 24.780 | 17.87 |
| 1200 | 22.862 | 16.95 |
| 1400 | 21.616 | 16.80 |
| 1600 | 25.912 | 18.46 |
| 1800 | 24.926 | 21.49 |

### Trial 19 — PRUNED (best MAE 30.451)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.002285, eta_infer=0.0001349, infer_steps=22, max_infer_norm=1, grad_clip=1, weight_init_std=0.02731

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 35.954 | 26.51 |
| 400 | 30.654 | 23.51 |
| 600 | 30.451 | 26.05 |

### Trial 20 — COMPLETE (best MAE 20.519)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.002787, eta_infer=1.261e-05, infer_steps=22, max_infer_norm=5, grad_clip=1, weight_init_std=0.01727

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.014 | 23.24 |
| 400 | 29.220 | 25.43 |
| 600 | 24.136 | 19.53 |
| 800 | 24.969 | 18.49 |
| 1000 | 22.537 | 18.15 |
| 1200 | 21.807 | 16.76 |
| 1400 | 21.589 | 17.27 |
| 1600 | 21.615 | 16.36 |
| 1800 | 21.177 | 16.15 |
| 2000 | 21.034 | 16.68 |
| 2200 | 20.713 | 16.38 |
| 2400 | 21.462 | 17.48 |
| 2600 | 20.519 | 15.63 |
| 2800 | 21.148 | 16.95 |
| 3000 | 20.821 | 15.86 |

### Trial 21 — COMPLETE (best MAE 20.378)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.002975, eta_infer=1.281e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.588 | 22.25 |
| 400 | 25.937 | 21.80 |
| 600 | 23.299 | 17.88 |
| 800 | 24.369 | 17.97 |
| 1000 | 26.105 | 18.98 |
| 1200 | 21.346 | 16.61 |
| 1400 | 21.406 | 16.96 |
| 1600 | 20.781 | 16.04 |
| 1800 | 20.628 | 16.03 |
| 2000 | 21.756 | 16.69 |
| 2200 | 20.378 | 15.48 |
| 2400 | 20.409 | 15.95 |
| 2600 | 22.036 | 16.10 |
| 2800 | 20.620 | 16.27 |
| 3000 | 21.529 | 17.01 |

### Trial 22 — COMPLETE (best MAE 20.684)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.002613, eta_infer=1.034e-05, infer_steps=22, max_infer_norm=5, grad_clip=1, weight_init_std=0.01582

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 32.251 | 24.27 |
| 400 | 26.268 | 21.49 |
| 600 | 23.567 | 18.61 |
| 800 | 22.899 | 18.03 |
| 1000 | 22.459 | 17.16 |
| 1200 | 22.723 | 18.50 |
| 1400 | 21.461 | 16.48 |
| 1600 | 21.671 | 16.38 |
| 1800 | 20.704 | 16.20 |
| 2000 | 20.767 | 16.00 |
| 2200 | 20.684 | 16.51 |
| 2400 | 20.762 | 16.13 |
| 2600 | 20.960 | 15.74 |

### Trial 23 — PRUNED (best MAE 21.665)

- **Geometry**: `128/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 9
- **Params**: lr=0.004492, eta_infer=1.32e-05, infer_steps=17, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01231

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.261 | 22.61 |
| 400 | 25.321 | 18.99 |
| 600 | 23.850 | 18.72 |
| 800 | 22.994 | 18.09 |
| 1000 | 22.474 | 18.21 |
| 1200 | 22.777 | 18.88 |
| 1400 | 22.424 | 17.73 |
| 1600 | 21.665 | 17.47 |
| 1800 | 26.691 | 18.43 |

### Trial 24 — PRUNED (best MAE 24.631)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.00269, eta_infer=1.868e-05, infer_steps=24, max_infer_norm=5, grad_clip=1, weight_init_std=0.01568

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.914 | 24.76 |
| 400 | 29.082 | 24.90 |
| 600 | 24.631 | 19.78 |

### Trial 25 — PRUNED (best MAE 24.882)

- **Geometry**: `64/d1/h2` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.002106, eta_infer=1.381e-05, infer_steps=19, max_infer_norm=5, grad_clip=2, weight_init_std=0.01858

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.027 | 25.25 |
| 400 | 28.364 | 23.98 |
| 600 | 24.882 | 18.96 |

### Trial 26 — PRUNED (best MAE 39.812)

- **Geometry**: `128/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001216, eta_infer=1.699e-05, infer_steps=17, max_infer_norm=5, grad_clip=1, weight_init_std=0.01477

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 39.812 | 34.21 |

### Trial 27 — RUNNING (best MAE —)

- **Geometry**: `128/d1/h4` · readout `None`
- **Stop / prune reason**: —
- **Params**: lr=0.00256, eta_infer=2.586e-05, infer_steps=24, max_infer_norm=5, grad_clip=0.5, weight_init_std=0.02628

### Trial 28 — RUNNING (best MAE —)

- **Geometry**: `128/d1/h1` · readout `None`
- **Stop / prune reason**: —
- **Params**: lr=0.001651, eta_infer=2.915e-05, infer_steps=12, max_infer_norm=5, grad_clip=0.5, weight_init_std=0.01904

### Trial 29 — PRUNED (best MAE 25.930)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.003303, eta_infer=2.688e-05, infer_steps=8, max_infer_norm=1, grad_clip=2, weight_init_std=0.01961

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.621 | 22.56 |
| 400 | 25.930 | 21.52 |
| 600 | 33.292 | 28.67 |

### Trial 30 — PRUNED (best MAE 24.236)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.002147, eta_infer=1.148e-05, infer_steps=9, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01753

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.667 | 23.95 |
| 400 | 26.414 | 20.09 |
| 600 | 24.236 | 18.93 |

### Trial 31 — PRUNED (best MAE 34.191)

- **Geometry**: `64/d1/h2` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001424, eta_infer=1.177e-05, infer_steps=20, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.01381

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 34.191 | 25.93 |

### Trial 32 — PRUNED (best MAE 40.938)

- **Geometry**: `64/d2/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001279, eta_infer=1.284e-05, infer_steps=22, max_infer_norm=5, grad_clip=0.5, weight_init_std=0.01983

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 40.938 | 33.71 |

### Trial 33 — PRUNED (best MAE 21.364)

- **Geometry**: `128/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 9
- **Params**: lr=0.004517, eta_infer=1.788e-05, infer_steps=22, max_infer_norm=5, grad_clip=1, weight_init_std=0.01688

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.010 | 29.31 |
| 400 | 25.287 | 19.91 |
| 600 | 23.717 | 19.51 |
| 800 | 22.423 | 17.71 |
| 1000 | 22.394 | 17.82 |
| 1200 | 22.407 | 18.48 |
| 1400 | 21.672 | 16.87 |
| 1600 | 21.364 | 17.34 |
| 1800 | 22.473 | 18.37 |

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
