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

## Architecture summary

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

1. Complete the architecture-aware Optuna study and compare instability rates
   by context length (64/128), depth (1–3), and attention heads (1/2/4).
2. Analyze validation MAE, energy, and parameter-norm trajectories jointly to
   distinguish inference explosions from optimizer drift.
3. Confirm the best PC configuration with a full epoch-based run and compare it
   against backpropagation using the same model geometry.
