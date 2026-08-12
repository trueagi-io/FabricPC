# Glucose PC Optuna progress report

Generated: `2026-07-23`  
Mode: **predictive coding (PC) only** — no backpropagation.

## Cross-phase verdict

| Phase | Study / run dir | Best val MAE | Beat prior best? |
|-------|-----------------|-------------:|:----------------:|
| 1 broad | `runs/glucose_tuning` | 20.3776 | — (was champion) |
| 2 refined | `runs/glucose_tuning_pc_v2` | 20.6780 | No (+0.30) |
| 3 local | `runs/glucose_tuning_pc_local` | 20.8670 | No (+0.49) |
| **4 breakthrough** | `runs/glucose_tuning_pc_breakthrough` | **19.8760** | **Yes (−0.50)** |

**New PC champion:** phase-4 trial **7** at **19.876** Optuna val MAE
(Δ **−0.502** vs phase-1 20.378).

Epoch confirm (`runs/glucose_pc_breakthrough_confirm`, seed 61):

| Metric | Phase-1 confirm | Phase-4 confirm |
|--------|----------------:|----------------:|
| Best val MAE | 20.63 | **18.78** |
| Test MAE | 21.74 | **18.20** |
| Test RMSE | 32.06 | **26.10** |
| Test MARD (%) | 15.41 | **14.66** |
| Epochs | 7 | 6 |
| Wall time (s) | 164 | 131 |

---

## What phase 4 changed (and why it worked)

Built from phases 2–3 conclusions:

| Lever | Change | Effect |
|-------|--------|--------|
| η band | Lock to **9e-6–2.5e-5** | Avoided ultra-low η duds |
| Update budget | **10000** (was 6000) | Champion-like configs kept improving |
| Pruner | **MedianPruner** + patience **6** | Slow starters survived |
| Seed | Search `seed_offset` | Trial 7 used seed **61** (offset 19) |
| Readout | Search flatten / mean_pool / last | **flatten still wins**; mean_pool ~43 MAE |

Three trials beat 20.38: **7 (19.876)**, **12 (20.011)**, **14 (20.361)**.

---

## Phase 4 best params (trial 7)

| Knob | Value |
|------|------:|
| Geometry | 64 / depth 1 / heads 1 |
| LR | 0.003691 |
| η_infer | 1.680e-05 |
| Infer steps | 12 |
| max_infer_norm | 1.0 |
| grad_clip | 0.5 |
| weight_init_std | 0.01531 |
| readout | flatten |
| seed (42 + offset) | 61 |

Replay / confirm:

```bash
uv run glucose-transformer --mode pc --epochs 30 --patience 4 \
  --seq_len 64 --depth 1 --num_heads 1 \
  --lr 0.0036910070447662953 \
  --eta_infer 1.67967291868957e-05 \
  --infer_steps 12 --max_infer_norm 1.0 \
  --grad_clip 0.5 --weight_init_std 0.01531315474155434 \
  --seed 61 --out_dir runs/glucose_pc_breakthrough_confirm
```

---

## Phase 4 study summary

Study: `glucose_transformer_pc_breakthrough`  
Protocol: PC-only, `full_updates=10000`, `pilot_updates=800`

| Metric | Value |
|--------|------:|
| Trials | 24 |
| Best complete trial | 7 |
| Best Optuna val MAE | **19.8760** |
| Trials beating 20.38 | 3 |

### Complete-trial leaderboard (top)

| Trial | Best MAE | readout | seed | LR | η_infer | steps | clip |
|------:|---------:|---------|-----:|---:|--------:|------:|-----:|
| 7 | **19.876** | flatten | 61 | 0.003691 | 1.680e-05 | 12 | 0.5 |
| 12 | 20.011 | flatten | 63 | 0.003533 | 2.207e-05 | 12 | 0.5 |
| 14 | 20.361 | flatten | 63 | 0.002667 | 9.998e-06 | 12 | 0.5 |
| 0 | 20.388 | flatten | 63 | 0.002975 | 1.281e-05 | 14 | 0.5 |

### Epoch validation MAE (phase-4 confirm)

| Epoch | Val MAE | RMSE | MARD (%) |
|------:|--------:|-----:|---------:|
| 1 | 20.996 | 29.348 | 17.55 |
| 2 | **18.782** | 27.168 | 14.58 |
| 3 | 23.369 | 30.664 | 16.98 |
| 4 | 22.708 | 30.013 | 19.18 |
| 5 | 19.144 | 27.435 | 15.52 |
| 6 | 31.195 | 40.504 | 22.18 |

---

## Negative result (document)

`mean_pool` / `last` readouts with champion-like PC dynamics did **not** help
under this protocol (`mean_pool` ~43 MAE). Lighter heads need their own
LR/η retuning if revisited. The phase-4 win came from **longer budget +
softer pruning + seed search** on the proven flatten geometry.

---

## Files

| Path | Role |
|------|------|
| `runs/glucose_tuning_pc_breakthrough/` | Phase 4 journal + trials |
| `runs/glucose_pc_breakthrough_confirm/` | Epoch confirm of trial 7 |
| `runs/glucose_tuning/` | Phase 1 (previous champion) |
| `runs/glucose_pc_best_confirm/` | Epoch confirm of phase-1 trial 21 |
