# Glucose Hopfield Optuna progress report

Generated: `2026-07-24T08:13:19.623946+00:00`  
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

## How to read this report

- **Trial** = one hyperparameter recipe (including Hopfield variant/strength).
- **Epoch** = one pass over training data inside that trial (learning-curve x-axis).
- **Pruned** = Hyperband stopped a weak trial early (saves compute, not a crash).
- **Early stop / patience** = val MAE stopped improving → keep best checkpoint.
- Lower **MAE (mg/dL)** is better.

## Complete training report (all trials)

### Trial 0 — PRUNED (best MAE 20.302)

- **Variant**: `baseline` · strength `1.0`
- **Stop / prune reason**: energy explosion at epoch 3, step 3510: 1.01408
- **Params**: seq_len=64, depth=1, num_heads=1, variant=baseline, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 21.964 |
| 2 | 20.302 |

### Trial 1 — PRUNED (best MAE 35.422)

- **Variant**: `projection` · strength `1.0`
- **Stop / prune reason**: energy explosion at epoch 2, step 1863: 1.1069
- **Params**: seq_len=64, depth=1, num_heads=1, variant=projection, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

| Epoch | Val MAE |
|------:|--------:|
| 1 | 35.422 |

### Trial 2 — PRUNED (best MAE —)

- **Variant**: `embed-storkey` · strength `1.0`
- **Stop / prune reason**: energy explosion at epoch 1, step 1378: 9.49119
- **Params**: seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

### Trial 3 — RUNNING (best MAE —)

- **Variant**: `embed-storkey` · strength `2.0`
- **Stop / prune reason**: —
- **Params**: seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=2.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

### Trial 4 — RUNNING (best MAE —)

- **Variant**: `embed-storkey` · strength `learnable`
- **Stop / prune reason**: —
- **Params**: seq_len=64, depth=1, num_heads=1, variant=embed-storkey, hopfield_strength=learnable, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

### Trial 5 — RUNNING (best MAE —)

- **Variant**: `forecast-storkey` · strength `1.0`
- **Stop / prune reason**: —
- **Params**: seq_len=64, depth=1, num_heads=1, variant=forecast-storkey, hopfield_strength=1.0, lr=0.003691, eta_infer=1.68e-05, infer_steps=12, max_infer_norm=1, grad_clip=0.5, lr_decay_epochs=10, weight_init_std=0.01531, seed_offset=19

### Trial 6 — WAITING (best MAE —)

- **Variant**: `None` · strength `None`
- **Stop / prune reason**: —

### Trial 7 — WAITING (best MAE —)

- **Variant**: `None` · strength `None`
- **Stop / prune reason**: —

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
