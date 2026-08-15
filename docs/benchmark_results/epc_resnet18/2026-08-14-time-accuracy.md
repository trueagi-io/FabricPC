# ePC ResNet-18 time-to-accuracy follow-up — 2026-08-14

## Preregistered protocol

This protocol was written before running the follow-up grid or observing its
validation, internal-holdout, or new official-test results. It follows the
two-epoch benchmark in [2026-08-14.md](./2026-08-14.md), which found ePC 6.65x
faster per epoch but 5.39 percentage points less accurate at equal epochs.

### Question

Does validation-selected ePC recover sPC's accuracy within a conservative
proxy for the same training-time budget, or is the two-epoch accuracy gap still
present after spending ePC's speed advantage?

### Data isolation

- Training: CIFAR-10 `train[:80%]` (40,000 examples).
- Validation: `train[80%:90%]` (5,000 examples).
- Fresh primary holdout: `train[90%:]` (5,000 examples). The tuning command
  must not construct this loader.
- Secondary endpoint: the official CIFAR-10 `test` split (10,000 examples).
  This is not pristine globally because the earlier benchmark already observed
  it, but the follow-up does not use it for configuration selection.
- ReLU, no augmentation, batch size 256, AdamW, weight decay 0.01, and the same
  warm-up/cosine schedule shape as the earlier benchmark.

### Phase 1: validation-only selection

- Tuning seeds: 271828 and 314159; these are disjoint from final seeds.
- Three training epochs per candidate.
- Cartesian grid:
  - ePC inference steps: `{2, 5, 10}`;
  - peak learning rate: `{0.0003, 0.001, 0.003}`.
- Fixed ePC inference rate: 0.1.
- Primary selection metric: mean final-epoch validation accuracy across the two
  tuning seeds.
- Deterministic tie-break: fewer inference steps, then lower learning rate.
- Neither the primary holdout nor official test split may be evaluated during
  this phase.

Command:

```text
./ve/bin/python examples/epc_resnet18_time_accuracy.py --mode tune
```

### Phase 2: locked practical comparison

- Copy the selected ePC step count and learning rate from Phase 1 without
  modification.
- Final paired seeds: 0, 1000, and 2000.
- ePC trains for 13 epochs. This was fixed from the already-observed 6.65x
  two-epoch timing ratio (`2 * 6.65 = 13.3`) and rounds down, making it a
  conservative equal-time proxy.
- sPC remains fixed at 120 inference steps, inference rate 0.1, learning rate
  0.001, and two epochs.
- Both arms train on the same 40,000-example split and receive identical model
  seeds and initial minibatch streams within each paired seed.
- Record validation accuracy and training-only elapsed time after every epoch.
- After training, evaluate the 5,000-example primary holdout and official test
  split exactly once per model.

Command template (values must come from `SELECTED`):

```text
./ve/bin/python examples/epc_resnet18_time_accuracy.py --mode final --epc_steps SELECTED_STEPS --epc_lr SELECTED_LR
```

### Preregistered interpretation

The practical gate passes only if, on aggregate means:

1. selected ePC primary-holdout accuracy is at least sPC accuracy; and
2. selected ePC training time is no greater than sPC training time.

All values must be finite. Official-test accuracy, validation curves, paired
tests, and effect sizes are secondary evidence. If ePC misses the time budget,
the result is not an equal-time win even if accuracy is higher. If it remains
less accurate while within budget, the original accuracy gap is practically
relevant under this broader protocol.

### Known limitations

- Two tuning seeds and three final seeds are enough to expose a consistent
  effect but not to estimate a production-grade performance distribution.
- Thirteen epochs is a fixed proxy, not a hard real-time stop.
- The grid is intentionally small and does not claim a globally optimal ePC
  recipe.
- The internal holdout is the primary fresh endpoint because the official test
  split was observed in the preceding benchmark.

## Results

Pending execution of the preregistered commands.
