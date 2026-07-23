# Glucose Hopfield Optuna progress report

Generated: `2026-07-23T22:08:25.953905+00:00`  
Study: `glucose_hopfield_pc_v1`  
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
| Transformer 1 broad (`runs/glucose_tuning`) | 20.3776 | — |
| Transformer 2 refined (`runs/glucose_tuning_pc_v2`) | 20.6780 | — |
| Transformer 3 local (`runs/glucose_tuning_pc_local`) | 20.8670 | — |
| Transformer 4 breakthrough (`runs/glucose_tuning_pc_breakthrough`) | 19.8760 | — |

## Study summary (all trial states)

| Metric | Value |
|--------|------:|
| Trials recorded | 8 |
| Complete | 0 |
| Pruned | 3 |
| Failed | 0 |
| Running | 3 |

## Top model architectures

## Complete-trial leaderboard

| Trial | Best MAE | Δ vs 19.876 | variant | strength | LR | η_infer | steps | seed_off |
|------:|---------:|------------:|---------|----------|---:|--------:|------:|---------:|

## All trials (every state)

| Trial | State | Best MAE | variant | strength | stop / prune |
|------:|-------|---------:|---------|----------|--------------|
| 0 | PRUNED | 20.302 | baseline | 1.0 | energy explosion at epoch 3, step 3510: 1.01408 |
| 1 | PRUNED | 35.422 | projection | 1.0 | energy explosion at epoch 2, step 1863: 1.1069 |
| 2 | PRUNED | — | embed-storkey | 1.0 | energy explosion at epoch 1, step 1378: 9.49119 |
| 3 | RUNNING | — | embed-storkey | 2.0 | — |
| 4 | RUNNING | — | embed-storkey | learnable | — |
| 5 | RUNNING | — | forecast-storkey | 1.0 | — |
| 6 | WAITING | — | None | None | — |
| 7 | WAITING | — | None | None | — |

## Hyperparameter glossary

| Parameter | Meaning |
|-----------|---------|
| `seq_len` | Input sequence length (number of 5-min glucose readings fed to the model). Longer = more history but heavier. |
| `depth` | Number of transformer blocks stacked. More depth = more capacity but slower inference and higher memory. |
| `num_heads` | Number of parallel attention heads in multi-scale self-attention. |
| `variant` | Hopfield memory placement: 'baseline' (no Hopfield), 'embed-storkey' (after embedding), 'forecast-storkey' (before forecast head), 'projection' (linear memory). |
| `hopfield_strength` | Hopfield interaction strength. 'learnable' = optimised during training; a number = fixed scaling factor for the associative memory. |
| `lr` | Outer learning rate for weight updates (Adam/AdamW). Controls how fast weights move each step. |
| `eta_infer` | PC inference learning rate. Step size for the inner-loop SGD that updates latent activations to minimise prediction errors. |
| `infer_steps` | Number of PC inference iterations per forward pass. More steps = tighter energy minimisation but slower training. |
| `max_infer_norm` | Maximum gradient norm during PC inference. Clips the inner-loop update to prevent latent activations from exploding. |
| `grad_clip` | Global gradient clipping threshold for weight updates. Stabilises training by capping large gradients. |
| `lr_decay_epochs` | Epoch at which the learning rate starts cosine decay toward zero. Later = longer warm phase at full LR. |
| `weight_init_std` | Standard deviation for weight initialisation (Normal). Smaller = more conservative start; interacts with depth. |
| `seed_offset` | Random seed offset for reproducibility and diversity across trials with otherwise similar configs. |

## Background

This work builds on our earlier results with conventional transformers for glucose forecasting
at [GlucoseDAO/glucose-forecasting](https://github.com/GlucoseDAO/glucose-forecasting).
Here we add **predictive coding (PC)** inner loops and explore **Hopfield associative memory**
— a content-addressable memory layer that stores and recalls learned glucose dynamics (meal
responses, exercise patterns, dawn phenomenon). The Hopfield memory gives the model an explicit
pattern-recall mechanism beyond what attention alone provides.

## How the model works

### Hopfield variants searched

| Variant | Where the Hopfield memory sits | Intuition |
|---------|-------------------------------|-----------|
| `baseline` | No Hopfield node | Pure transformer (control group) |
| `embed-storkey` | After the embedding layer | Memory enriches token representations before attention |
| `forecast-storkey` | Before the forecast head | Memory pattern-matches right before making the prediction |
| `projection` | Linear projection memory | Lightweight associative recall with learned projections |

### Architecture (embed-storkey example)

```
Glucose Input (batch, seq_len, 1)
       |
  Continuous Embedding
       |
  [Storkey Hopfield Memory]  ← content-addressable pattern recall
       |                       stores learned glucose dynamics
  +--[ Transformer Block ] × depth --------+
  |    Multi-Scale Self-Attention (RoPE)    |
  |    LN → MLP expand (GELU)               |
  |    MLP contract + Residual skip          |
  +------------------------------------------+
       |
  Regression Output Head → Glucose Forecast (60 min)
```

### PC inference loop (runs at every node including Hopfield)

1. Predict `z_mu` from incoming activations
2. Compute `error = z_latent - z_mu`
3. Compute energy (Gaussian: E = 0.5 ||error||^2)
4. Update `z_latent` via SGD (step size = `eta_infer`, clip = `max_infer_norm`)
5. Repeat for `infer_steps` iterations

## Limitations

- **Single participant data** — we started only 1.5 days before the deadline, so we used
  only Livia's personal CGM data rather than training across multiple participants.
- **Glucose-only input** — only continuous glucose values are fed to the model. Carbohydrate
  intake, heart rate, step count, and other covariates available in the full dataset are not included.
- **Limited tuning budget** — the tight timeline restricted the number of Optuna trials and
  hyperparameter ranges we could explore.

## How to run

### Hopfield variant tuning (this report)

Searches over Hopfield variant placement (baseline / projection / embed-storkey /
forecast-storkey), strength (0.5–2.0 or learnable), and all PC/architecture params.
Default: 24 trials, Hyperband pruning.

| Task | Command |
|------|---------|
| Start tuning | `uv run glucose-hopfield-tune run` |
| Custom trial count | `uv run glucose-hopfield-tune run --n-trials 48` |
| More parallel workers | `uv run glucose-hopfield-tune run --n-trials 48 --max-workers 4` |
| Custom run directory | `uv run glucose-hopfield-tune run --run-dir runs/my_hopfield --study-name my_study` |
| Adjust epochs/patience | `uv run glucose-hopfield-tune run --max-epochs 20 --patience 5` |
| Resume interrupted | `uv run glucose-hopfield-tune run` (Optuna journal auto-resumes) |
| Regenerate this report | `uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all` |

### PC Transformer tuning (Gaussian vs Huber energy)

Separate tuner for the base PC transformer. Searches both Gaussian and Huber energy,
SGD and Adam inference, IPC on/off, and all architecture params.

| Task | Command |
|------|---------|
| Start transformer tuning | `uv run glucose-transformer-tune run` |
| Custom trial count | `uv run glucose-transformer-tune run --n-trials 64 --max-workers 4` |
| Regenerate transformer report | `uv run python scripts/generate_glucose_tuning_report.py --format all` |

### All reports

| Task | Command |
|------|---------|
| Generate all reports | `uv run python scripts/generate_all_glucose_reports.py --format all` |
