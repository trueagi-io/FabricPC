# Glucose PC Optuna progress report

Generated: `2026-07-24T08:13:25.021002+00:00`  
Study: `glucose_transformer_pc_v2`  
Mode: predictive coding (PC) only

## Summary

| Metric | Value |
|--------|------:|
| Trials recorded | 32 |
| Complete | 4 |
| Pruned | 28 |
| Failed | 0 |
| Running | 0 |
| Best complete trial | 21 |
| Best val MAE (mg/dL) | 20.6780 |

## What helped (auto-generated from top trials)

- **seq_len=64** dominates top 5 (4/4)
- **depth=1** dominates top 5 (4/4)
- **num_heads=4** dominates top 5 (4/4)
- **lr**: range 0.002936–0.0045, median 0.004234
- **eta_infer**: range 1.358e-05–5.414e-05, median 3.401e-05
- **weight_init_std**: range 0.01355–0.02449, median 0.0242
- **infer_steps**: 14×1, 17×1, 11×1, 12×1
- **grad_clip**: 1.0×3, 0.5×1

## Top model architectures

### #1 — Trial 21 (MAE 20.678)

- **Geometry**: seq_len=64, depth=1, heads=4
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=4, lr=0.004202, eta_infer=1.358e-05, infer_steps=14, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.0242, weight_decay=3.381e-05

### #2 — Trial 19 (MAE 21.220)

- **Geometry**: seq_len=64, depth=1, heads=4
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=4, lr=0.002936, eta_infer=3.401e-05, infer_steps=17, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01355, weight_decay=1.061e-05

### #3 — Trial 23 (MAE 21.410)

- **Geometry**: seq_len=64, depth=1, heads=4
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=4, lr=0.0045, eta_infer=2.565e-05, infer_steps=11, max_infer_norm=1, grad_clip=1, weight_init_std=0.02449, weight_decay=8.662e-05

### #4 — Trial 0 (MAE 22.139)

- **Geometry**: seq_len=64, depth=1, heads=4
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=4, lr=0.004234, eta_infer=5.414e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.02284, weight_decay=8.444e-05

## Complete-trial leaderboard

| Trial | Best MAE | MARD% | Geometry | LR | η_infer | Infer steps | Grad clip |
|------:|---------:|------:|----------|---:|--------:|------------:|----------:|
| 21 | 20.678 | 15.95 | 64/d1/h4 | 0.004202 | 1.358e-05 | 14 | 1.0 |
| 19 | 21.220 | 16.94 | 64/d1/h4 | 0.002936 | 3.401e-05 | 17 | 0.5 |
| 23 | 21.410 | 16.83 | 64/d1/h4 | 0.0045 | 2.565e-05 | 11 | 1.0 |
| 0 | 22.139 | 17.42 | 64/d1/h4 | 0.004234 | 5.414e-05 | 12 | 1.0 |

## Best MAE by trial

| Trial | State | Best MAE (mg/dL) |
|------:|-------|-----------------:|
| 0 | COMPLETE | 22.139 |
| 1 | PRUNED | 33.928 |
| 2 | PRUNED | 35.682 |
| 3 | PRUNED | 36.305 |
| 4 | PRUNED | 23.888 |
| 5 | PRUNED | 34.268 |
| 6 | PRUNED | 36.301 |
| 7 | PRUNED | 29.870 |
| 8 | PRUNED | 31.421 |
| 9 | PRUNED | 31.895 |
| 10 | PRUNED | 31.154 |
| 11 | PRUNED | 30.473 |
| 12 | PRUNED | 37.232 |
| 13 | PRUNED | 31.198 |
| 14 | PRUNED | 40.314 |
| 15 | PRUNED | 33.438 |
| 16 | PRUNED | 26.816 |
| 17 | PRUNED | 24.338 |
| 18 | PRUNED | 24.027 |
| 19 | COMPLETE | 21.220 |
| 20 | PRUNED | 31.282 |
| 21 | COMPLETE | 20.678 |
| 22 | PRUNED | 33.257 |
| 23 | COMPLETE | 21.410 |
| 24 | PRUNED | 24.854 |
| 25 | PRUNED | 32.447 |
| 26 | PRUNED | 32.893 |
| 27 | PRUNED | 37.308 |
| 28 | PRUNED | 31.068 |
| 29 | PRUNED | 31.771 |
| 30 | PRUNED | 23.872 |
| 31 | PRUNED | 23.393 |

## Top trial MAE traces (every 200 updates)

- **Trial 21** (best 20.678): 200:28.07 → 400:25.79 → 600:22.86 → 800:21.17 → 1000:20.98 → 1200:20.68 → 1400:21.07 → 1600:25.36 → 1800:21.82 → 2000:24.44
- **Trial 19** (best 21.220): 200:29.81 → 400:24.77 → 600:22.71 → 800:23.65 → 1000:21.62 → 1200:21.22 → 1400:22.01 → 1600:23.34 → 1800:27.98 → 2000:24.64
- **Trial 23** (best 21.410): 200:29.20 → 400:25.53 → 600:22.32 → 800:21.51 → 1000:23.30 → 1200:21.41 → 1400:22.72 → 1600:21.65
- **Trial 0** (best 22.139): 200:29.56 → 400:26.42 → 600:23.25 → 800:22.87 → 1000:22.14 → 1200:27.36 → 1400:26.71

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

### Trial 0 — COMPLETE (best MAE 22.139)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.004234, eta_infer=5.414e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.02284, weight_decay=8.444e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.562 | 22.01 |
| 400 | 26.415 | 22.45 |
| 600 | 23.250 | 18.35 |
| 800 | 22.870 | 18.49 |
| 1000 | 22.139 | 17.42 |
| 1200 | 27.363 | 22.60 |
| 1400 | 26.711 | 22.67 |

### Trial 1 — PRUNED (best MAE 33.928)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001795, eta_infer=3.214e-06, infer_steps=19, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01214, weight_decay=2.929e-06

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.928 | 25.95 |

### Trial 2 — PRUNED (best MAE 35.682)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001464, eta_infer=4.64e-06, infer_steps=10, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01755, weight_decay=0.0001065

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 35.682 | 28.47 |

### Trial 3 — PRUNED (best MAE 36.305)

- **Geometry**: `64/d2/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002113, eta_infer=5.973e-06, infer_steps=11, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.0132, weight_decay=3.073e-06

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 36.305 | 27.53 |

### Trial 4 — PRUNED (best MAE 23.888)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.004141, eta_infer=5.52e-06, infer_steps=20, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01731, weight_decay=1.416e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 28.937 | 22.89 |
| 400 | 23.888 | 19.55 |
| 600 | 29.179 | 20.93 |

### Trial 5 — PRUNED (best MAE 34.268)

- **Geometry**: `64/d2/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002707, eta_infer=7.562e-06, infer_steps=13, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01836, weight_decay=0.0003784

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 34.268 | 25.90 |

### Trial 6 — PRUNED (best MAE 36.301)

- **Geometry**: `64/d2/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001935, eta_infer=2.14e-05, infer_steps=11, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.02452, weight_decay=0.0002782

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 36.301 | 27.60 |

### Trial 7 — PRUNED (best MAE 29.870)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.004108, eta_infer=9.438e-06, infer_steps=12, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.0206, weight_decay=1.081e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.870 | 21.59 |

### Trial 8 — PRUNED (best MAE 31.421)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002542, eta_infer=3.016e-06, infer_steps=20, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.01889, weight_decay=0.0007656

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.421 | 23.89 |

### Trial 9 — PRUNED (best MAE 31.895)

- **Geometry**: `64/d2/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.004297, eta_infer=2.078e-05, infer_steps=15, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01736, weight_decay=1.239e-06

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.895 | 26.56 |

### Trial 10 — PRUNED (best MAE 31.154)

- **Geometry**: `64/d2/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.004484, eta_infer=5.232e-06, infer_steps=13, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01861, weight_decay=7.742e-06

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.154 | 25.48 |

### Trial 11 — PRUNED (best MAE 30.473)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002533, eta_infer=7.574e-06, infer_steps=18, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.02341, weight_decay=3.718e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.473 | 23.64 |

### Trial 12 — PRUNED (best MAE 37.232)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001288, eta_infer=9.054e-06, infer_steps=17, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.01294, weight_decay=0.0003293

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 37.232 | 29.64 |

### Trial 13 — PRUNED (best MAE 31.198)

- **Geometry**: `64/d2/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003547, eta_infer=4.927e-06, infer_steps=12, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01399, weight_decay=0.0001767

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.198 | 24.07 |

### Trial 14 — PRUNED (best MAE 40.314)

- **Geometry**: `64/d2/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001144, eta_infer=6.729e-06, infer_steps=11, max_infer_norm=1, grad_clip=1, weight_init_std=0.01987, weight_decay=6.925e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 40.314 | 33.15 |

### Trial 15 — PRUNED (best MAE 33.438)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001552, eta_infer=4.246e-05, infer_steps=19, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.02101, weight_decay=1.156e-06

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.438 | 25.68 |

### Trial 16 — PRUNED (best MAE 26.816)

- **Geometry**: `64/d1/h1` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.00428, eta_infer=6.264e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.02111, weight_decay=4.197e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.977 | 25.19 |
| 400 | 26.816 | 21.43 |
| 600 | 31.677 | 21.66 |

### Trial 17 — PRUNED (best MAE 24.338)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.004346, eta_infer=9.021e-06, infer_steps=19, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01423, weight_decay=1.355e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.287 | 21.47 |
| 400 | 25.998 | 22.50 |
| 600 | 24.338 | 17.87 |

### Trial 18 — PRUNED (best MAE 24.027)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.003314, eta_infer=5.418e-05, infer_steps=14, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.02442, weight_decay=8.262e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.599 | 23.95 |
| 400 | 27.258 | 23.20 |
| 600 | 24.027 | 19.28 |

### Trial 19 — COMPLETE (best MAE 21.220)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.002936, eta_infer=3.401e-05, infer_steps=17, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01355, weight_decay=1.061e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.813 | 22.52 |
| 400 | 24.773 | 20.44 |
| 600 | 22.714 | 17.70 |
| 800 | 23.651 | 17.45 |
| 1000 | 21.623 | 16.56 |
| 1200 | 21.220 | 16.94 |
| 1400 | 22.005 | 18.48 |
| 1600 | 23.337 | 20.10 |
| 1800 | 27.979 | 24.53 |
| 2000 | 24.643 | 17.21 |

### Trial 20 — PRUNED (best MAE 31.282)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003487, eta_infer=6.554e-05, infer_steps=12, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.02417, weight_decay=0.0005704

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.282 | 25.45 |

### Trial 21 — COMPLETE (best MAE 20.678)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.004202, eta_infer=1.358e-05, infer_steps=14, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.0242, weight_decay=3.381e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 28.070 | 22.10 |
| 400 | 25.789 | 22.26 |
| 600 | 22.855 | 18.44 |
| 800 | 21.173 | 16.80 |
| 1000 | 20.983 | 16.81 |
| 1200 | 20.678 | 15.95 |
| 1400 | 21.075 | 17.04 |
| 1600 | 25.357 | 21.65 |
| 1800 | 21.815 | 15.98 |
| 2000 | 24.442 | 17.61 |

### Trial 22 — PRUNED (best MAE 33.257)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.001917, eta_infer=3.073e-05, infer_steps=17, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01287, weight_decay=5.577e-06

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 33.257 | 25.94 |

### Trial 23 — COMPLETE (best MAE 21.410)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: no 0.25 mg/dL improvement over four checks
- **Params**: lr=0.0045, eta_infer=2.565e-05, infer_steps=11, max_infer_norm=1, grad_clip=1, weight_init_std=0.02449, weight_decay=8.662e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.196 | 22.08 |
| 400 | 25.532 | 18.65 |
| 600 | 22.320 | 17.63 |
| 800 | 21.508 | 16.80 |
| 1000 | 23.304 | 16.65 |
| 1200 | 21.410 | 16.83 |
| 1400 | 22.716 | 16.21 |
| 1600 | 21.649 | 17.14 |

### Trial 24 — PRUNED (best MAE 24.854)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.004622, eta_infer=5.201e-06, infer_steps=12, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.02359, weight_decay=1.391e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.737 | 22.93 |
| 400 | 24.854 | 20.59 |
| 600 | 29.082 | 20.46 |

### Trial 25 — PRUNED (best MAE 32.447)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.004441, eta_infer=7.25e-05, infer_steps=15, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01366, weight_decay=1.17e-06

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 32.447 | 22.66 |

### Trial 26 — PRUNED (best MAE 32.893)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.002738, eta_infer=6.236e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.02195, weight_decay=0.0002451

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 32.893 | 24.38 |

### Trial 27 — PRUNED (best MAE 37.308)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.004096, eta_infer=3.308e-05, infer_steps=17, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01421, weight_decay=2.826e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 37.308 | 33.80 |

### Trial 28 — PRUNED (best MAE 31.068)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003498, eta_infer=9.483e-06, infer_steps=15, max_infer_norm=1, grad_clip=1, weight_init_std=0.0248, weight_decay=6.518e-06

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.068 | 26.17 |

### Trial 29 — PRUNED (best MAE 31.771)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 1
- **Params**: lr=0.003143, eta_infer=1.412e-05, infer_steps=10, max_infer_norm=1, grad_clip=1, weight_init_std=0.01983, weight_decay=0.0006018

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.771 | 27.09 |

### Trial 30 — PRUNED (best MAE 23.872)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.003638, eta_infer=7.817e-05, infer_steps=20, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01526, weight_decay=1.81e-05

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 28.794 | 22.56 |
| 400 | 26.801 | 22.30 |
| 600 | 23.872 | 18.21 |

### Trial 31 — PRUNED (best MAE 23.393)

- **Geometry**: `64/d1/h4` · readout `None`
- **Stop / prune reason**: SuccessiveHalvingPruner at check 3
- **Params**: lr=0.004448, eta_infer=1.382e-05, infer_steps=10, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.02496, weight_decay=0.0001964

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 27.598 | 21.74 |
| 400 | 24.564 | 20.14 |
| 600 | 23.393 | 18.62 |

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
