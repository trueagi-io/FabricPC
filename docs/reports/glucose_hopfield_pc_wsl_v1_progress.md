# Glucose Hopfield Optuna progress report

Generated: `2026-07-23T22:08:26.429028+00:00`  
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
