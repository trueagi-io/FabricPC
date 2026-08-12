# Glucose PC Optuna progress report

Generated: `2026-07-23T22:08:26.811225+00:00`  
Study: `glucose_transformer_pc_epochs_v2`  
Mode: predictive coding (PC) only

## Summary

| Metric | Value |
|--------|------:|
| Trials recorded | 3 |
| Complete | 0 |
| Pruned | 0 |
| Failed | 0 |
| Running | 3 |

## What helped (auto-generated from top trials)

- **seq_len=128** dominates top 5 (2/3)
- **depth=1** dominates top 5 (2/3)
- **num_heads**: mixed (2×1, 1×1, 4×1)
- **lr**: range 0.0003751–0.002485, median 0.000378
- **eta_infer**: range 1.134e-05–0.0002705, median 1.31e-05
- **weight_init_std**: range 0.01307–0.02295, median 0.01499
- **infer_steps**: 19×2, 18×1
- **grad_clip**: 2.0×2, 1.0×1

## Top model architectures

### #1 — Trial 1 (MAE 20.673)

- **Geometry**: seq_len=128, depth=1, heads=2
- **Readout**: None
- **All params**: seq_len=128, depth=1, num_heads=2, lr=0.002485, eta_infer=1.134e-05, infer_steps=19, max_infer_norm=1, grad_clip=2, weight_init_std=0.02295

### #2 — Trial 0 (MAE 23.886)

- **Geometry**: seq_len=64, depth=1, heads=1
- **Readout**: None
- **All params**: seq_len=64, depth=1, num_heads=1, lr=0.0003751, eta_infer=0.0002705, infer_steps=19, max_infer_norm=5, grad_clip=1, weight_init_std=0.01499

### #3 — Trial 2 (MAE 27.937)

- **Geometry**: seq_len=128, depth=3, heads=4
- **Readout**: None
- **All params**: seq_len=128, depth=3, num_heads=4, lr=0.000378, eta_infer=1.31e-05, infer_steps=18, max_infer_norm=5, grad_clip=2, weight_init_std=0.01307

## Complete-trial leaderboard

| Trial | Best MAE | MARD% | Geometry | LR | η_infer | Infer steps | Grad clip |
|------:|---------:|------:|----------|---:|--------:|------------:|----------:|
| 1 | 20.673 | 16.84 | 128/d1/h2 | 0.002485 | 1.134e-05 | 19 | 2.0 |
| 0 | 23.886 | 18.60 | 64/d1/h1 | 0.0003751 | 0.0002705 | 19 | 1.0 |
| 2 | 27.937 | 22.61 | 128/d3/h4 | 0.000378 | 1.31e-05 | 18 | 2.0 |

## Best MAE by trial

| Trial | State | Best MAE (mg/dL) |
|------:|-------|-----------------:|
| 0 | RUNNING | 23.886 |
| 1 | RUNNING | 20.673 |
| 2 | RUNNING | 27.937 |

## Top trial MAE traces (every 200 updates)

- **Trial 1** (best 20.673): 1:22.46 → 2:22.05 → 3:26.55 → 4:20.67 → 5:23.60 → 6:22.22
- **Trial 0** (best 23.886): 1:33.65 → 2:26.78 → 3:25.37 → 4:24.03 → 5:23.97 → 6:23.89
- **Trial 2** (best 27.937): 1:32.14 → 2:27.94

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
