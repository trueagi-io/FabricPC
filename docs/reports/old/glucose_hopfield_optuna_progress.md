# Glucose Hopfield Optuna progress report

Generated: `2026-07-23T18:32:58.446799+00:00`  
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

- **Trial 20** (baseline/learnable, best 20.160): e1:22.17 → e2:21.29 → e3:20.16 → e4:21.36 → e5:20.42 → e6:20.41 → e7:20.37
- **Trial 16** (baseline/learnable, best 20.305): e1:22.51 → e2:20.31 → e3:20.80 → e4:20.68 → e5:20.47 → e6:20.56
- **Trial 14** (baseline/learnable, best 20.406): e1:21.34 → e2:20.41 → e3:24.56 → e4:21.14 → e5:20.61 → e6:20.87
- **Trial 23** (baseline/learnable, best 20.416): e1:22.44 → e2:24.54 → e3:20.79 → e4:22.39 → e5:20.42 → e6:20.50 → e7:20.58 → e8:20.50 → e9:20.49
- **Trial 8** (baseline/learnable, best 20.487): e1:22.88 → e2:20.49 → e3:22.12 → e4:23.64 → e5:20.94 → e6:21.06
- **Trial 17** (baseline/learnable, best 20.925): e1:29.84 → e2:20.93 → e3:23.62 → e4:42.64

## Files

| Path | Role |
|------|------|
| `results_snapshot.json` | Full trial dump (all states) |
| `report_data.json` | Structured payload |
| `report.md` / `report.html` | Human-readable views |
| `best_trial.json` | Coordinator winner summary |
| `coordinator_config.json` | Exact settings for this run |
| `trials/trial_XXXX/` | Per-trial config, history, checkpoints |

Regenerate:

```bash
uv run python scripts/generate_glucose_hopfield_tuning_report.py --format all
```

## Final verdict (WSL GPU study complete)

Study finished **24/24** terminal trials. Best Optuna val MAE: **20.1605** (trial 20, `baseline`).

Beat transformer phase-4 target **19.876**? **No** (Δ **+0.284**).

### Cross-architecture comparison

| Run | Best val MAE | vs Hopfield best |
|-----|-------------:|-----------------:|
| Transformer phase-4 Optuna | 19.8760 | -0.284 |
| Transformer phase-1 Optuna | 20.3776 | +0.217 |
| Transformer phase-4 confirm (val/test) | 18.78 / 18.20 | (different protocol) |
| **Hopfield WSL Optuna (this study)** | **20.1605** | 0 |

Plain transformer PC (phase-4) remains the champion. This Hopfield study beat phase-1's 20.38 but did **not** beat phase-4's 19.876.

### Variant ranking (best history MAE)

| Variant | n with MAE | Best | Median |
|---------|----------:|-----:|-------:|
| baseline | 11 | 20.160 | 20.925 |
| projection | 5 | 24.526 | 25.633 |
| embed-storkey | 5 | 43.528 | 44.566 |
| forecast-storkey | 3 | 67.431 | 624.772 |

### What the data says

1. **`baseline` wins this search** — every competitive trial (<22 MAE) was the plain transformer graph. `hopfield_strength` is ignored for baseline; gains came from PC dynamics retune (`lr≈2.5e-3`, `η≈1.0e-5`, `infer_steps=17`, `seed=75`).
2. **`projection` control ~25–28 MAE** — adding a linear embed projection hurts vs matched baseline under the same PC band.
3. **`embed-storkey` stuck ~43–45 MAE** — Storkey-after-embed with champion-like PC knobs does not train; energy/param-norm often blow up. Not a small hyperparam miss.
4. **`forecast-storkey` fails hard** (67–600+ MAE) — memory on the horizon output is unstable under this PC loop.

### Recommended next steps

1. **Keep phase-4 transformer as production PC baseline** (19.876 Optuna / 18.78–18.20 confirm). Promote trial-20 baseline only if an epoch confirm beats that on the same split.
2. **Confirm trial 20 on the epoch trainer** (`glucose-transformer` or hopfield `--variant baseline` with trial-20 HPs, seed 75, 30 epochs) and compare test MAE to phase-4 confirm.
3. **Do not scale current Storkey placement** until redesign: try (a) much smaller fixed strength `0.1–0.5`, (b) freeze backbone then enable Hopfield, (c) **temporal** Hopfield on time axis (ideas doc phase 2), (d) settled-eval as the Optuna objective (feedforward MAE may hide attractor benefit).
4. **If chasing <19.876 next**: run a **baseline-only** Optuna phase (drop storkey/projection from search) with longer epoch budget / update-budget breakthrough protocol — this study already shows room in PC knobs (20.16 vs 19.88).
5. **Optional negative-result note in `glucose_hopfield_ideas.md`**: embed/forecast Storkey with identity activation + champion PC band failed under epoch Hyperband; next experiments need different energy coupling or training schedule.

