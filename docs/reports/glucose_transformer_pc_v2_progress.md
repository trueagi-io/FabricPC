# Glucose PC Optuna progress report

Generated: `2026-07-23T22:08:27.437897+00:00`  
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
