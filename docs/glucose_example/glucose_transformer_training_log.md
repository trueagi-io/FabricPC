# Glucose Transformer Training Log

## Problem: PC Training Instability at lr=0.02

### Observation

During the first full 30-epoch comparison run (PC vs backprop), the PC
training diverged mid-epoch 3:

| Epoch | Batch | MAE (mg/dL) | Energy | Status |
|-------|-------|-------------|--------|--------|
| 1 | 1 | 114.4 | 0.904 | Random init |
| 1 | 10 | 59.9 | 0.250 | Fast convergence |
| 1 | 1380 | ~46 | ~0.15 | Stabilized |
| 1 | val | 45.574 | — | Best so far |
| 2 | val | 45.267 | — | Marginal improvement (+0.3) |
| 3 | 1100 | 50.3 | 0.16 | Still OK |
| 3 | 1350 | 221.9 | 2.71 | Exploded |
| 4 | 110 | 132.6 | 1.08 | Unstable |

### Root cause

The learning rate `lr=0.02` was originally tuned for a step-based training
loop with ~6,000 total optimizer updates. With the switch to epoch-based
training, the total step count became `30 epochs × 1,380 batches = 41,400`
steps — 7x more updates.

The `warmup_cosine_decay_schedule` peaks at `lr=0.02` after 200 warmup steps
and decays over 41,400 steps. This means the LR stays near its peak value
(~0.019-0.02) for thousands of steps, much longer than in the 6K-step setup
where it peaked and immediately began meaningful decay.

With a 195K-parameter model on min-max normalized glucose data (range
40–399 mg/dL), `lr=0.02` is too aggressive for sustained training — the
model converges fast (epoch 1) but then overshoots and diverges.

### CUDA memory

The model is lightweight: **1,677 MiB / 16,376 MiB** (10% of GPU). Memory
is not a constraint.

## Attempt 2: lr=0.005

Lowered LR to 0.005 (4x reduction). Result:

| Epoch | Val MAE (mg/dL) |
|-------|-----------------|
| 1 | 26.1 |
| 6 | 45.2 |
| 7 | 45.1 |

The model achieved good val MAE at epoch 1 but then **regressed** — val MAE
nearly doubled by epoch 6. Same instability pattern as lr=0.02, just delayed.

### Root cause (refined)

The issue is not just LR magnitude. Without gradient clipping, accumulated
gradient norms grow over epochs and eventually push the model out of the
learned basin. The cosine schedule doesn't decay fast enough to compensate.

## Attempt 3: lr=1e-3 + gradient clipping + divergence guard

Three simultaneous fixes:

1. **LR 1e-3** for both PC and backprop (20x lower than original)
2. **Gradient clipping** at global norm 1.0 via
   `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))`
3. **Divergence guard**: auto-stop if val MAE exceeds 2x best val MAE

```bash
uv run glucose-transformer --mode compare --epochs 30 \
    --lr 1e-3 --lr_backprop 1e-3 --grad_clip 1.0 \
    --out_dir runs/glucose_compare
```

### Results (stable, both early-stopped)

**PC** (9 epochs, 12,420 steps, 1533s):

| Epoch | Val MAE (mg/dL) | Val RMSE | MARD (%) |
|-------|-----------------|----------|----------|
| 1 | 23.68 | 33.21 | 16.88 |
| 2 | 21.78 | 30.54 | 16.62 |
| 4 | 20.15 | 29.31 | 15.85 |
| 5 | **20.12** | 29.29 | 15.59 |
| Test | 20.85 | 30.79 | 15.78 |

**Backprop** (7 epochs, 9,660 steps, 268s):

| Epoch | Val MAE (mg/dL) | Val RMSE | MARD (%) |
|-------|-----------------|----------|----------|
| 1 | 19.98 | 28.87 | 15.55 |
| 2 | 19.57 | 28.62 | 15.42 |
| 3 | **18.82** | 27.94 | 14.60 |
| Test | 19.14 | 28.44 | 14.62 |

**Winner: Backprop** (Δ = 1.71 mg/dL on test MAE)

### Key observations

- Both modes are **stable** with gradient clipping at norm 1.0
- Backprop is **5.7x faster** (268s vs 1533s) due to PC's inference overhead
- PC early-stopped at epoch 9, backprop at epoch 7 — both peaked early
- PC may benefit from higher eta_infer or more infer_steps to close the gap
- The 20 mg/dL test MAE (60-min forecast at 5-min cadence) is a reasonable
  baseline for the GluMind-Uni architecture on Livia

## Architecture-aware Optuna tuning report

The resumable study `glucose_transformer_pc_architecture_v2` evaluated 40
process-isolated PC trials. Each trial used up to 6,000 optimizer updates, with
validation every 200 updates and successive-halving pruning. The search varied
context length (64/128), depth (1–3), attention heads (1/2/4), learning rate,
inference rate and steps, inference norm clipping, gradient clipping, and
weight initialization scale.

### Final winner

Trial 34 was the best completed trial. Optuna recorded an objective of
21.445 mg/dL, while the full validation history reached a lower observed
minimum of **21.332 mg/dL at update 2,800**. The discrepancy came from using
the 0.25 mg/dL early-stop significance threshold when updating the stored best;
best-checkpoint tracking now records every strict improvement independently of
that threshold.

| Hyperparameter | Final default |
|----------------|---------------|
| Context (`seq_len`) | 64 |
| Depth | 2 |
| Attention heads | 1 |
| Learning rate | 0.0032753171 |
| Inference rate (`eta_infer`) | 1.4435783e-5 |
| Inference steps | 19 |
| Inference norm clip | 1.0 |
| Gradient clip | 0.5 |
| Weight initialization std | 0.02186191 |

These values are now the defaults for `uv run glucose-transformer`.

### Leading trials by observed validation MAE

| Trial | State | Best MAE | Update | Context | Depth | Heads | LR | Grad clip |
|------:|-------|---------:|-------:|--------:|------:|------:|---:|----------:|
| 34 | Complete | **21.332** | 2,800 | 64 | 2 | 1 | 0.003275 | 0.5 |
| 33 | Complete | 21.902 | 1,000 | 64 | 2 | 1 | 0.004854 | 0.5 |
| 27 | Pruned | 22.267 | 1,200 | 64 | 2 | 1 | 0.003159 | 0.5 |
| 29 | Pruned | 22.303 | 1,400 | 64 | 1 | 2 | 0.001874 | 1.0 |
| 35 | Complete | 23.229 | 1,000 | 64 | 1 | 2 | 0.002694 | 1.0 |

The first 32 trials were run before stable early stops were distinguished from
true pruning, so some valid best checkpoints (including trials 27 and 29) have
historical state `Pruned`. The study was resumed for eight trials after the
fix, producing three selectable completed trials. The ranking above therefore
uses the minimum MAE in every trial history, not only Optuna's stored objective.

### Architecture results

| Factor | Trials | Median best MAE | Best MAE |
|--------|-------:|----------------:|---------:|
| Context 64 | 25 | 30.079 | **21.332** |
| Context 128 | 15 | 42.095 | 23.857 |
| Depth 1 | 22 | 31.068 | 22.303 |
| Depth 2 | 13 | 40.475 | **21.332** |
| Depth 3 | 5 | 44.171 | 30.079 |
| 1 head | 13 | 37.185 | **21.332** |
| 2 heads | 21 | 40.582 | 22.303 |
| 4 heads | 6 | 34.826 | 25.101 |
| Gradient clip 0.5 | 12 | 34.826 | **21.332** |
| Gradient clip 1.0 | 17 | 37.185 | 22.303 |
| Gradient clip 2.0 | 11 | 40.475 | 23.265 |

These groups are not controlled ablations because Optuna varied several
parameters simultaneously and pruned weak trials early. They nevertheless give
clear practical signals: context 64 dominated context 128; depth 3 was never
competitive; and all three leading trials used depth 2, one head, and gradient
clip 0.5.

### Stability findings

- Successive halving removed 31 trials early (24 at the first validation and
  7 at the third), focusing compute on plausible configurations.
- One trial was stopped by the energy-explosion guard. Trial 3 had reached
  23.265 MAE before energy jumped to 0.738 at update 1,443.
- Four of 40 histories had a post-best MAE spike above 10%; two exceeded 25%.
  Several isolated spikes recovered at the next validation, so requiring two
  consecutive regressions was more reliable than stopping after one spike.
- MAE explosions were not explained by parameter norm alone. For example,
  trial 0 jumped from 24.47 to 39.45 MAE while parameter norm changed only
  from 58.3 to 59.8; its energy rose from 0.050 to 0.120.
- The strongest region combined short context, a shallow one-head model,
  low inference rates around 1–2e-5, and tighter gradient clipping. This
  supports the hypothesis that excess architectural depth and aggressive PC
  dynamics contributed to instability.

## Original baseline architecture summary

| Parameter | Value |
|-----------|-------|
| Nodes | 12 |
| Edges | 17 |
| Parameters | 195,020 |
| Depth | 3 |
| Embed dim | 32 |
| Heads | 4 |
| MLP dim | 128 |
| Seq len | 128 (10.67h at 5-min cadence) |
| Horizon | 12 (60 min forecast) |

## Data summary

| Split | Windows |
|-------|---------|
| Train | 88,266 |
| Val | 19,207 |
| Test | 20,186 |
| Batches/epoch | 1,380 (batch_size=64) |
| Total steps (30 epochs) | 41,400 |

## Features implemented

- **Epoch-based training** with validation at the end of each epoch
- **Per-batch logging**: energy/MSE, MAE (mg/dL), MARD (%) on each training batch
- **Early stopping**: patience=4 epochs without val MAE improvement
- **Instability detection**: immediate stop on NaN/Inf loss
- **Checkpoint saving**: full training state (params, optimizer, RNG, epoch,
  best MAE) saved after each epoch for crash recovery
- **Resume support**: `--resume` flag restores from last checkpoint
- **Compare mode**: runs PC then backprop with CUDA memory cleanup between
- **Output files**: `config.json`, `history.csv`, `best_params.pkl`,
  `checkpoint.pkl`, `comparison.json`, `comparison.txt`

## Next steps

1. Confirm trial 34 with a full epoch-based PC run using the promoted defaults.
2. Repeat the winning configuration across multiple seeds to quantify variance.
3. Compare PC and backpropagation using the same context-64, depth-2,
   one-head geometry and report validation and held-out test metrics.
