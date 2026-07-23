# Glucose PC Optuna progress report

Generated: `2026-07-23T22:08:27.091736+00:00`  
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

## Hyperparameter glossary

| Parameter | Meaning |
|-----------|---------|
| `seq_len` | Input sequence length (number of 5-min glucose readings fed to the model). Longer = more history but heavier. |
| `depth` | Number of transformer blocks stacked. More depth = more capacity but slower inference and higher memory. |
| `num_heads` | Number of parallel attention heads in multi-scale self-attention. More heads = finer-grained attention patterns. |
| `embed_dim` | Dimensionality of token embeddings inside the transformer. Larger = more expressive but more parameters. |
| `lr` | Outer learning rate for weight updates (Adam/AdamW). Controls how fast weights move each step. |
| `eta_infer` | PC inference learning rate. Step size for the inner-loop SGD that updates latent activations to minimise prediction errors. |
| `infer_steps` | Number of PC inference iterations per forward pass. More steps = tighter energy minimisation but slower training. |
| `max_infer_norm` | Maximum gradient norm during PC inference. Clips the inner-loop update to prevent latent activations from exploding. |
| `grad_clip` | Global gradient clipping threshold for weight updates. Stabilises training by capping large gradients. |
| `lr_decay_epochs` | Epoch at which the learning rate starts cosine decay toward zero. Later = longer warm phase at full LR. |
| `weight_init_std` | Standard deviation for weight initialisation (Normal). Smaller = more conservative start; interacts with depth. |
| `weight_decay` | L2 regularisation coefficient on weights. Higher = stronger penalty on large weights, can prevent overfitting. |
| `readout` | Regression head pooling mode: 'flatten' (full seq*dim projection), 'mean_pool' (average over time), or 'last' (last timestep only). |
| `seed_offset` | Random seed offset for reproducibility and diversity across trials with otherwise similar configs. |
| `energy` | Energy functional for PC nodes: 'gaussian' (MSE-based) or 'huber' (robust to outliers). |
| `huber_delta` | Huber loss delta threshold. Only active when energy='huber'. Smaller = more robust to outliers. |
| `ipc` | Incremental Predictive Coding. When True, updates latents layer-by-layer instead of all-at-once (can improve convergence). |
| `infer_optimizer` | Optimiser for PC inference loop: 'sgd' (simple, fast) or 'adam' (adaptive, may converge faster but uses more memory). |

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

### Hopfield PC Transformer

```
Glucose Input (batch, seq_len, 1)
       |
  Continuous Embedding
       |
  [Storkey Hopfield Memory]  ← content-addressable pattern recall
       |                       stores learned glucose dynamics
  +--[ Transformer Block ] × depth --------+
  |    Multi-Scale MHA + Residual           |
  |    MLP + skip + PC Energy Node          |
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

### Energy functions (both searched during tuning)

- **Gaussian** (default): E = 0.5 ||error||^2 — standard MSE, penalises large errors quadratically
- **Huber**: quadratic for small errors, linear past `huber_delta` — robust to glucose spikes/outliers

## Limitations

- **Single participant data** — we started only 1.5 days before the deadline, so we used
  only Livia's personal CGM data rather than training across multiple participants.
- **Glucose-only input** — only continuous glucose values are fed to the model. Carbohydrate
  intake, heart rate, step count, and other covariates available in the full dataset are not included.
- **Limited tuning budget** — the tight timeline restricted the number of Optuna trials and
  hyperparameter ranges we could explore.

## How to run

### PC Transformer tuning (this report)

Searches both Gaussian and Huber energy, SGD and Adam inference,
IPC on/off, and all architecture params. Default: 32 trials, Hyperband pruning.

| Task | Command |
|------|---------|
| Start tuning | `uv run glucose-transformer-tune run` |
| Custom trial count | `uv run glucose-transformer-tune run --n-trials 64` |
| More parallel workers | `uv run glucose-transformer-tune run --n-trials 64 --max-workers 4` |
| Custom run directory | `uv run glucose-transformer-tune run --run-dir runs/my_experiment --study-name my_study` |
| Adjust epochs/patience | `uv run glucose-transformer-tune run --max-epochs 20 --patience 5` |
| Resume interrupted | `uv run glucose-transformer-tune run` (Optuna journal auto-resumes) |
| Regenerate this report | `uv run python scripts/generate_glucose_tuning_report.py --format all` |

### Hopfield variant tuning

Separate tuner that searches Hopfield memory placement (baseline / projection /
embed-storkey / forecast-storkey) and strength. Same PC dynamics search.

| Task | Command |
|------|---------|
| Start Hopfield tuning | `uv run glucose-hopfield-tune run` |
| Custom trial count | `uv run glucose-hopfield-tune run --n-trials 48 --max-workers 3` |
| Regenerate Hopfield report | `uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all` |

### All reports

| Task | Command |
|------|---------|
| Generate all reports | `uv run python scripts/generate_all_glucose_reports.py --format all` |
