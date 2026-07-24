# Glucose PC Optuna progress report

Generated: `2026-07-24T08:13:23.508108+00:00`  
Study: `glucose_transformer_pc_breakthrough`  
Mode: predictive coding (PC) only

## Summary

| Metric | Value |
|--------|------:|
| Trials recorded | 24 |
| Complete | 13 |
| Pruned | 11 |
| Failed | 0 |
| Running | 0 |
| Best complete trial | 7 |
| Best val MAE (mg/dL) | 19.8760 |

## What helped (auto-generated from top trials)

- **seq_len=64** dominates top 5 (5/5)
- **depth=1** dominates top 5 (5/5)
- **num_heads=1** dominates top 5 (5/5)
- **readout=flatten** dominates top 5 (5/5)
- **lr**: range 0.002667–0.003691, median 0.003533
- **eta_infer**: range 9.998e-06–2.207e-05, median 1.68e-05
- **weight_init_std**: range 0.01519–0.01879, median 0.01684
- **infer_steps**: 12×2, 16×1, 14×1, 15×1
- **grad_clip**: 0.5×4, 1.0×1

## Top model architectures

### #1 — Trial 7 (MAE 19.876)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: flatten
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01531, readout=flatten, seed_offset=19

### #2 — Trial 12 (MAE 20.011)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: flatten
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.003533, eta_infer=2.207e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.01519, readout=flatten, seed_offset=21

### #3 — Trial 14 (MAE 20.361)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: flatten
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.002667, eta_infer=9.998e-06, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01879, readout=flatten, seed_offset=21

### #4 — Trial 0 (MAE 20.388)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: flatten
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.002975, eta_infer=1.281e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684, readout=flatten, seed_offset=21

### #5 — Trial 13 (MAE 20.866)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: flatten
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.003681, eta_infer=1.787e-05, infer_steps=15, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01722, readout=flatten, seed_offset=11

## Complete-trial leaderboard

| Trial | Best MAE | MARD% | Geometry | LR | η_infer | Infer steps | Grad clip |
|------:|---------:|------:|----------|---:|--------:|------------:|----------:|
| 7 | 19.876 | 15.40 | 64/d1/h1 | 0.003691 | 1.68e-05 | 12 | 0.5 |
| 12 | 20.011 | 15.13 | 64/d1/h1 | 0.003533 | 2.207e-05 | 12 | 1.0 |
| 14 | 20.361 | 15.52 | 64/d1/h1 | 0.002667 | 9.998e-06 | 16 | 0.5 |
| 0 | 20.388 | 15.75 | 64/d1/h1 | 0.002975 | 1.281e-05 | 14 | 0.5 |
| 13 | 20.866 | 16.79 | 64/d1/h1 | 0.003681 | 1.787e-05 | 15 | 0.5 |
| 17 | 21.253 | 16.08 | 64/d1/h1 | 0.003647 | 2.032e-05 | 13 | 1.0 |
| 5 | 21.934 | 16.90 | 64/d1/h1 | 0.0035 | 1.05e-05 | 15 | 0.5 |
| 16 | 21.986 | 17.06 | 64/d1/h1 | 0.003424 | 1.604e-05 | 12 | 1.0 |
| 2 | 22.034 | 16.85 | 64/d1/h1 | 0.002975 | 1.281e-05 | 14 | 0.5 |
| 8 | 22.629 | 18.58 | 64/d1/h1 | 0.003447 | 1.452e-05 | 16 | 0.5 |
| 4 | 42.874 | 34.34 | 64/d1/h1 | 0.0026 | 1.5e-05 | 14 | 0.5 |
| 3 | 42.921 | 33.98 | 64/d1/h1 | 0.0032 | 1.15e-05 | 16 | 0.5 |
| 1 | 43.022 | 34.08 | 64/d1/h1 | 0.002975 | 1.281e-05 | 14 | 0.5 |

## Best MAE by trial

| Trial | State | Best MAE (mg/dL) |
|------:|-------|-----------------:|
| 0 | COMPLETE | 20.388 |
| 1 | COMPLETE | 43.022 |
| 2 | COMPLETE | 22.034 |
| 3 | COMPLETE | 42.921 |
| 4 | COMPLETE | 42.874 |
| 5 | COMPLETE | 21.934 |
| 6 | PRUNED | 21.414 |
| 7 | COMPLETE | 19.876 |
| 8 | COMPLETE | 22.629 |
| 9 | PRUNED | 20.908 |
| 10 | PRUNED | 20.590 |
| 11 | PRUNED | 25.080 |
| 12 | COMPLETE | 20.011 |
| 13 | COMPLETE | 20.866 |
| 14 | COMPLETE | 20.361 |
| 15 | PRUNED | 20.862 |
| 16 | COMPLETE | 21.986 |
| 17 | COMPLETE | 21.253 |
| 18 | PRUNED | 23.831 |
| 19 | PRUNED | 43.884 |
| 20 | PRUNED | 43.148 |
| 21 | PRUNED | 23.747 |
| 22 | PRUNED | 26.347 |
| 23 | PRUNED | 23.422 |

## Top trial MAE traces (every 200 updates)

- **Trial 7** (best 19.876): 200:29.18 → 400:24.88 → 600:27.58 → 800:22.14 → 1000:22.31 → 1200:22.91 → 1400:21.76 → 1600:21.04 → 1800:20.68 → 2000:21.86 → 2200:21.11 → 2400:21.45 → 2600:20.61 → 2800:20.37 → 3000:20.04 → 3200:20.93 → 3400:24.81 → 3600:21.94 → 3800:19.88 → 4000:21.39 → 4200:19.90
- **Trial 12** (best 20.011): 200:30.21 → 400:23.95 → 600:23.16 → 800:25.62 → 1000:25.04 → 1200:21.20 → 1400:20.70 → 1600:21.05 → 1800:21.13 → 2000:21.30 → 2200:20.33 → 2400:22.27 → 2600:20.01 → 2800:20.31 → 3000:20.69 → 3200:21.50 → 3400:20.95 → 3600:20.44 → 3800:20.01
- **Trial 14** (best 20.361): 200:30.63 → 400:25.64 → 600:23.54 → 800:23.05 → 1000:22.15 → 1200:25.44 → 1400:21.72 → 1600:21.63 → 1800:21.25 → 2000:22.34 → 2200:20.52 → 2400:21.21 → 2600:20.36 → 2800:20.78 → 3000:21.29 → 3200:20.75 → 3400:21.19
- **Trial 0** (best 20.388): 200:29.59 → 400:24.91 → 600:23.38 → 800:22.68 → 1000:24.02 → 1200:23.93 → 1400:21.86 → 1600:20.93 → 1800:21.36 → 2000:21.84 → 2200:20.83 → 2400:21.70 → 2600:22.27 → 2800:20.55 → 3000:20.90 → 3200:22.43 → 3400:20.39 → 3600:23.23 → 3800:20.50 → 4000:21.23

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

### Trial 0 — COMPLETE (best MAE 20.388)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.002975, eta_infer=1.281e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684, seed_offset=21

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.588 | 22.25 |
| 400 | 24.908 | 20.45 |
| 600 | 23.383 | 18.69 |
| 800 | 22.682 | 18.65 |
| 1000 | 24.015 | 17.64 |
| 1200 | 23.926 | 19.86 |
| 1400 | 21.855 | 17.60 |
| 1600 | 20.927 | 16.52 |
| 1800 | 21.359 | 17.50 |
| 2000 | 21.835 | 17.77 |
| 2200 | 20.827 | 16.68 |
| 2400 | 21.700 | 18.17 |
| 2600 | 22.273 | 18.17 |
| 2800 | 20.548 | 15.59 |
| 3000 | 20.896 | 15.50 |
| 3200 | 22.428 | 19.06 |
| 3400 | 20.388 | 15.75 |
| 3600 | 23.231 | 19.98 |
| 3800 | 20.503 | 15.21 |
| 4000 | 21.229 | 17.91 |

### Trial 1 — COMPLETE (best MAE 43.022)

- **Geometry**: `64/d1/h1` · readout `mean_pool`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.002975, eta_infer=1.281e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684, seed_offset=21

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 45.344 | 38.78 |
| 400 | 45.429 | 39.94 |
| 600 | 43.417 | 35.62 |
| 800 | 44.148 | 37.61 |
| 1000 | 43.419 | 32.38 |
| 1200 | 43.440 | 36.00 |
| 1400 | 43.164 | 35.05 |
| 1600 | 43.022 | 34.08 |
| 1800 | 43.517 | 36.35 |
| 2000 | 43.950 | 37.34 |
| 2200 | 43.466 | 32.05 |
| 2400 | 43.517 | 36.55 |
| 2600 | 43.117 | 35.45 |

### Trial 2 — COMPLETE (best MAE 22.034)

- **Geometry**: `64/d1/h1` · readout `last`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.002975, eta_infer=1.281e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684, seed_offset=21

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 32.814 | 26.10 |
| 400 | 23.694 | 18.79 |
| 600 | 25.541 | 18.07 |
| 800 | 24.920 | 18.59 |
| 1000 | 22.197 | 17.17 |
| 1200 | 22.040 | 17.12 |
| 1400 | 22.034 | 16.85 |
| 1600 | 23.709 | 19.58 |
| 1800 | 22.113 | 16.79 |
| 2000 | 22.808 | 18.06 |
| 2200 | 22.052 | 16.60 |

### Trial 3 — COMPLETE (best MAE 42.921)

- **Geometry**: `64/d1/h1` · readout `mean_pool`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.0032, eta_infer=1.15e-05, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684, seed_offset=21

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 45.347 | 38.91 |
| 400 | 45.403 | 39.98 |
| 600 | 43.080 | 34.47 |
| 800 | 42.921 | 33.98 |
| 1000 | 43.007 | 33.54 |
| 1200 | 43.164 | 35.41 |
| 1400 | 43.219 | 35.65 |
| 1600 | 42.977 | 33.01 |
| 1800 | 43.896 | 37.34 |

### Trial 4 — COMPLETE (best MAE 42.874)

- **Geometry**: `64/d1/h1` · readout `mean_pool`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.0026, eta_infer=1.5e-05, infer_steps=14, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684, seed_offset=7

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 45.289 | 38.31 |
| 400 | 45.189 | 39.38 |
| 600 | 43.383 | 35.67 |
| 800 | 43.223 | 35.49 |
| 1000 | 42.928 | 34.22 |
| 1200 | 42.874 | 34.34 |
| 1400 | 43.601 | 36.79 |
| 1600 | 42.890 | 33.74 |
| 1800 | 42.897 | 34.37 |
| 2000 | 43.832 | 37.11 |
| 2200 | 43.455 | 36.43 |

### Trial 5 — COMPLETE (best MAE 21.934)

- **Geometry**: `64/d1/h1` · readout `last`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.0035, eta_infer=1.05e-05, infer_steps=15, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01684, seed_offset=15

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.526 | 23.76 |
| 400 | 22.599 | 17.26 |
| 600 | 23.240 | 18.84 |
| 800 | 24.485 | 18.49 |
| 1000 | 24.158 | 17.73 |
| 1200 | 22.135 | 17.47 |
| 1400 | 24.262 | 20.09 |
| 1600 | 29.128 | 25.20 |
| 1800 | 21.994 | 16.81 |
| 2000 | 21.976 | 16.52 |
| 2200 | 23.014 | 17.22 |
| 2400 | 21.934 | 16.90 |

### Trial 6 — PRUNED (best MAE 21.414)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 14
- **Params**: lr=0.002416, eta_infer=1.59e-05, infer_steps=13, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01686, seed_offset=24

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 32.189 | 24.00 |
| 400 | 26.245 | 21.29 |
| 600 | 23.497 | 18.35 |
| 800 | 25.007 | 18.41 |
| 1000 | 23.113 | 17.43 |
| 1200 | 28.253 | 24.84 |
| 1400 | 24.859 | 21.49 |
| 1600 | 27.350 | 23.85 |
| 1800 | 22.837 | 19.34 |
| 2000 | 22.207 | 18.20 |
| 2200 | 21.414 | 16.59 |
| 2400 | 22.646 | 19.18 |
| 2600 | 21.999 | 18.04 |
| 2800 | 22.170 | 18.31 |

### Trial 7 — COMPLETE (best MAE 19.876)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01531, seed_offset=19

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.179 | 23.00 |
| 400 | 24.883 | 19.63 |
| 600 | 27.584 | 23.09 |
| 800 | 22.137 | 17.95 |
| 1000 | 22.312 | 16.47 |
| 1200 | 22.912 | 19.08 |
| 1400 | 21.756 | 17.77 |
| 1600 | 21.042 | 16.36 |
| 1800 | 20.685 | 16.33 |
| 2000 | 21.856 | 16.60 |
| 2200 | 21.114 | 16.93 |
| 2400 | 21.450 | 17.77 |
| 2600 | 20.614 | 16.35 |
| 2800 | 20.375 | 15.69 |
| 3000 | 20.038 | 15.47 |
| 3200 | 20.925 | 15.56 |
| 3400 | 24.809 | 18.13 |
| 3600 | 21.936 | 18.09 |
| 3800 | 19.876 | 15.40 |
| 4000 | 21.388 | 18.12 |
| 4200 | 19.897 | 15.42 |

### Trial 8 — COMPLETE (best MAE 22.629)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.003447, eta_infer=1.452e-05, infer_steps=16, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01914, seed_offset=29

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.769 | 23.51 |
| 400 | 25.143 | 20.71 |
| 600 | 23.169 | 18.57 |
| 800 | 24.793 | 20.98 |
| 1000 | 22.629 | 18.58 |
| 1200 | 24.067 | 19.83 |
| 1400 | 25.612 | 18.78 |
| 1600 | 26.221 | 17.93 |

### Trial 9 — PRUNED (best MAE 20.908)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 14
- **Params**: lr=0.003644, eta_infer=1.024e-05, infer_steps=13, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01891, seed_offset=34

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.765 | 23.98 |
| 400 | 30.964 | 26.90 |
| 600 | 24.002 | 19.57 |
| 800 | 23.103 | 17.95 |
| 1000 | 22.005 | 16.73 |
| 1200 | 21.583 | 16.42 |
| 1400 | 21.347 | 16.32 |
| 1600 | 21.195 | 16.72 |
| 1800 | 21.703 | 15.89 |
| 2000 | 21.114 | 16.34 |
| 2200 | 23.286 | 16.90 |
| 2400 | 21.142 | 16.22 |
| 2600 | 20.908 | 16.72 |
| 2800 | 34.833 | 31.13 |

### Trial 10 — PRUNED (best MAE 20.590)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 14
- **Params**: lr=0.003594, eta_infer=1.635e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01555, seed_offset=36

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 29.325 | 22.19 |
| 400 | 25.698 | 21.38 |
| 600 | 24.835 | 20.34 |
| 800 | 22.772 | 18.53 |
| 1000 | 25.163 | 18.32 |
| 1200 | 21.680 | 16.58 |
| 1400 | 21.288 | 16.20 |
| 1600 | 22.403 | 18.31 |
| 1800 | 21.928 | 17.96 |
| 2000 | 20.827 | 15.43 |
| 2200 | 20.936 | 17.10 |
| 2400 | 20.590 | 16.61 |
| 2600 | 25.368 | 21.73 |
| 2800 | 21.889 | 18.31 |

### Trial 11 — PRUNED (best MAE 25.080)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 4
- **Params**: lr=0.003093, eta_infer=1.044e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01403, seed_offset=16

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.776 | 23.33 |
| 400 | 25.189 | 20.17 |
| 600 | 26.914 | 22.29 |
| 800 | 25.080 | 18.23 |

### Trial 12 — COMPLETE (best MAE 20.011)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.003533, eta_infer=2.207e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.01519, seed_offset=21

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.214 | 25.24 |
| 400 | 23.949 | 18.20 |
| 600 | 23.159 | 17.78 |
| 800 | 25.615 | 21.52 |
| 1000 | 25.042 | 17.66 |
| 1200 | 21.196 | 16.14 |
| 1400 | 20.703 | 16.09 |
| 1600 | 21.046 | 15.84 |
| 1800 | 21.129 | 16.00 |
| 2000 | 21.301 | 16.19 |
| 2200 | 20.330 | 15.72 |
| 2400 | 22.270 | 18.41 |
| 2600 | 20.011 | 15.13 |
| 2800 | 20.307 | 15.61 |
| 3000 | 20.686 | 15.30 |
| 3200 | 21.500 | 16.00 |
| 3400 | 20.949 | 16.89 |
| 3600 | 20.436 | 16.32 |
| 3800 | 20.012 | 15.04 |

### Trial 13 — COMPLETE (best MAE 20.866)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.003681, eta_infer=1.787e-05, infer_steps=15, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01722, seed_offset=11

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 28.889 | 22.90 |
| 400 | 26.052 | 21.97 |
| 600 | 22.989 | 17.63 |
| 800 | 22.668 | 18.15 |
| 1000 | 21.510 | 17.08 |
| 1200 | 21.813 | 17.90 |
| 1400 | 20.977 | 16.33 |
| 1600 | 21.050 | 16.07 |
| 1800 | 20.866 | 16.79 |
| 2000 | 24.053 | 17.54 |
| 2200 | 22.315 | 16.33 |
| 2400 | 24.072 | 21.03 |
| 2600 | 22.572 | 19.32 |

### Trial 14 — COMPLETE (best MAE 20.361)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: no 0.25 mg/dL improvement over 6 checks
- **Params**: lr=0.002667, eta_infer=9.998e-06, infer_steps=16, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01879, seed_offset=21

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.633 | 23.59 |
| 400 | 25.644 | 20.30 |
| 600 | 23.536 | 18.37 |
| 800 | 23.054 | 18.53 |
| 1000 | 22.155 | 16.75 |
| 1200 | 25.438 | 21.29 |
| 1400 | 21.719 | 17.66 |
| 1600 | 21.633 | 16.57 |
| 1800 | 21.248 | 16.41 |
| 2000 | 22.343 | 18.16 |
| 2200 | 20.523 | 16.06 |
| 2400 | 21.210 | 16.97 |
| 2600 | 20.361 | 15.52 |
| 2800 | 20.781 | 16.17 |
| 3000 | 21.293 | 15.93 |
| 3200 | 20.745 | 17.01 |
| 3400 | 21.187 | 17.43 |

### Trial 15 — PRUNED (best MAE 20.862)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 14
- **Params**: lr=0.003773, eta_infer=2.39e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.0142, seed_offset=23

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.523 | 22.93 |
| 400 | 27.393 | 23.50 |
| 600 | 24.095 | 19.48 |
| 800 | 23.490 | 17.67 |
| 1000 | 21.918 | 16.81 |
| 1200 | 23.012 | 19.48 |
| 1400 | 24.594 | 17.84 |
| 1600 | 21.486 | 17.00 |
| 1800 | 22.370 | 18.37 |
| 2000 | 33.574 | 23.50 |
| 2200 | 20.937 | 16.33 |
| 2400 | 20.862 | 15.90 |
| 2600 | 21.184 | 15.60 |
| 2800 | 22.183 | 18.44 |

### Trial 16 — COMPLETE (best MAE 21.986)

- **Geometry**: `64/d1/h1` · readout `last`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.003424, eta_infer=1.604e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.0158, seed_offset=11

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 31.426 | 23.91 |
| 400 | 23.658 | 19.14 |
| 600 | 22.740 | 17.81 |
| 800 | 23.574 | 17.36 |
| 1000 | 21.986 | 17.06 |
| 1200 | 28.003 | 24.14 |
| 1400 | 24.393 | 20.40 |

### Trial 17 — COMPLETE (best MAE 21.253)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: validation MAE regressed over 10% twice
- **Params**: lr=0.003647, eta_infer=2.032e-05, infer_steps=13, max_infer_norm=0.5, grad_clip=1, weight_init_std=0.01624, seed_offset=15

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 28.838 | 23.31 |
| 400 | 25.915 | 21.81 |
| 600 | 23.153 | 18.75 |
| 800 | 21.928 | 16.92 |
| 1000 | 32.568 | 29.16 |
| 1200 | 21.733 | 16.47 |
| 1400 | 24.584 | 21.13 |
| 1600 | 21.253 | 16.08 |
| 1800 | 25.805 | 21.83 |
| 2000 | 27.258 | 19.26 |

### Trial 18 — PRUNED (best MAE 23.831)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 3
- **Params**: lr=0.002628, eta_infer=2.026e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.018, seed_offset=19

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.886 | 23.47 |
| 400 | 25.196 | 20.04 |
| 600 | 23.831 | 18.53 |

### Trial 19 — PRUNED (best MAE 43.884)

- **Geometry**: `64/d1/h1` · readout `mean_pool`
- **Stop / prune reason**: MedianPruner at check 3
- **Params**: lr=0.002472, eta_infer=1.964e-05, infer_steps=12, max_infer_norm=1, grad_clip=1, weight_init_std=0.01434, seed_offset=20

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 45.115 | 37.91 |
| 400 | 45.826 | 40.17 |
| 600 | 43.884 | 36.46 |

### Trial 20 — PRUNED (best MAE 43.148)

- **Geometry**: `64/d1/h1` · readout `mean_pool`
- **Stop / prune reason**: MedianPruner at check 3
- **Params**: lr=0.003696, eta_infer=1.744e-05, infer_steps=12, max_infer_norm=0.5, grad_clip=0.5, weight_init_std=0.01677, seed_offset=20

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 44.866 | 37.06 |
| 400 | 45.543 | 40.28 |
| 600 | 43.148 | 32.72 |

### Trial 21 — PRUNED (best MAE 23.747)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 3
- **Params**: lr=0.002834, eta_infer=2.457e-05, infer_steps=13, max_infer_norm=1, grad_clip=1, weight_init_std=0.01586, seed_offset=3

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.745 | 23.52 |
| 400 | 26.174 | 19.63 |
| 600 | 23.747 | 18.98 |

### Trial 22 — PRUNED (best MAE 26.347)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 3
- **Params**: lr=0.002151, eta_infer=1.01e-05, infer_steps=18, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01978, seed_offset=19

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 32.128 | 24.62 |
| 400 | 26.535 | 21.80 |
| 600 | 26.347 | 22.08 |

### Trial 23 — PRUNED (best MAE 23.422)

- **Geometry**: `64/d1/h1` · readout `flatten`
- **Stop / prune reason**: MedianPruner at check 3
- **Params**: lr=0.003691, eta_infer=1.836e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, weight_init_std=0.01485, seed_offset=16

| Update | Val MAE | MARD (%) |
|------:|--------:|---------:|
| 200 | 30.635 | 23.08 |
| 400 | 28.820 | 20.32 |
| 600 | 23.422 | 17.48 |

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
