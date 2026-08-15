# ePC versus sPC independent-optimization study — 2026-08-15

## Status

Preregistered; results pending. This protocol follows
[the repository validation methodology](../../validation_methodology.md) and
replaces the invalidated 2026-08-14 recipe-transfer study.

## Question

Can ePC, using a training schedule selected for its own dynamics, reach the
same stable CIFAR-10 accuracy as tuned sPC and reach the tuned sPC accuracy
target in less cold-start GPU training time?

This is a capability and performance study. Equal learning rates, schedules,
checkpoint epochs, or update counts are not required.

## Data and endpoint seal

- Training: CIFAR-10 `train[:80%]` (40,000 examples).
- Validation: `train[80%:90%]` (5,000 examples).
- Held-out endpoint: `train[90%:]` (5,000 examples).
- The invalidated 2026-08-14 endpoint values are non-evidence and may not be
  used for any decision in this study. By project decision, the 5,000-example
  split is restored as this study's endpoint.
- Tuning mode must not construct the endpoint loader.
- Final mode evaluates the endpoint once per model, only at the
  validation-selected checkpoint after both recipes and the target are locked.
- The official CIFAR-10 test split is not used.

## Controlled factors

- ResNet-18 graph, ReLU, no augmentation, batch size 256.
- AdamW and weight decay 0.01. The optimizer family is fixed, while its peak
  learning rate and schedule are solver-specific.
- Warm-up lasts 0.25 epoch. Cosine decay ends after a candidate-specific 3 or
  5 epochs and then remains at 1% of peak.
- Each candidate is observed for eight epochs. Eight is an observation horizon,
  not a matched performance budget; checkpoint and time-to-target epochs may
  differ by solver.
- Cold-start training time includes JIT compilation. Validation time is
  excluded.

## Independent candidate spaces

ePC candidates are the Cartesian product of:

- inference steps: `{2, 5, 10}`;
- peak learning rate: `{0.00003, 0.0001, 0.0003}`;
- cosine-decay horizon: `{3, 5}` epochs;
- inference rate: fixed at 0.1.

sPC candidates are the Cartesian product of:

- inference steps: fixed at the established 120-step reference setting;
- peak learning rate: `{0.0003, 0.001, 0.003}`;
- cosine-decay horizon: `{3, 5}` epochs;
- inference rate 0.1 and norm clipping 1.0.

The different learning-rate ranges are intentional and follow each solver's
observed stability regime. No candidate is advantaged by requiring the other
solver's schedule.

## Tuning allocation and recipe selection

1. Screen every candidate for eight epochs with seed 271828.
2. Within each solver, shortlist the two highest-accuracy candidates that meet
   the stability rule. If fewer than two are stable, fill the shortlist by
   best validation accuracy so the failure is visible.
3. Run only those shortlisted candidates with confirmation seed 314159.
4. A result is stable when, over its final three epochs:
   - the validation-accuracy range is at most 1 percentage point; and
   - the best final-three accuracy is within 1 percentage point of the run's
     overall best accuracy.
5. A selectable recipe must be stable on both tuning seeds.
6. Select the recipe with the highest mean best-checkpoint validation accuracy
   across both seeds. Ties prefer lower mean time-to-best, then fewer inference
   steps, lower learning rate, and the shorter decay horizon.
7. If no shortlisted recipe is stable on both seeds, that solver is not
   empirically converged and final mode must not run.

Tuning command:

```text
./ve/bin/python examples/epc_resnet18_optimized.py --mode tune
```

Tuning cost is reported by solver but is not included in time-to-target.

## Common target and accuracy-time analysis

After selecting sPC, define the common target as the minimum validation
accuracy over the final three epochs of both selected-sPC tuning runs. This is
the observed floor of sPC's stable tuned plateau, not an epoch-matched or
post-endpoint target.

For each selected recipe, record:

- every epoch's cumulative training time and validation accuracy;
- the earliest time the locked target is reached;
- best-so-far accuracy at every epoch; and
- time and epoch of the selected best checkpoint.

The complete frontier is primary evidence alongside the target crossing.

## Locked final comparison

- Final paired seeds: 0, 1000, and 2000, disjoint from tuning seeds.
- Run both locked recipes for the same eight-epoch observation horizon, using
  each recipe's own inference and learning-rate schedule.
- Retain the earliest checkpoint attaining each run's best validation
  accuracy.
- Evaluate `train[90%:]` exactly once at that checkpoint.

Final mode requires the complete recipe and target printed by tuning:

```text
./ve/bin/python examples/epc_resnet18_optimized.py --mode final \
  --epc_steps LOCKED --epc_lr LOCKED --epc_decay_epochs LOCKED \
  --spc_steps LOCKED --spc_lr LOCKED --spc_decay_epochs LOCKED \
  --target_accuracy LOCKED
```

## Preregistered interpretation

The practical non-inferiority margin is 1 percentage point on mean paired
holdout accuracy. This is about 50 examples on the endpoint and is close to the
binomial sampling uncertainty at the expected accuracy.

The combined claim passes only if:

1. both recipes were stable on both tuning seeds;
2. mean ePC holdout accuracy is no more than 1 percentage point below mean sPC
   holdout accuracy;
3. every final ePC and sPC run reaches the locked validation target; and
4. mean ePC time-to-target is lower than mean sPC time-to-target.

Per-seed results, paired confidence intervals, target-crossing rates, total
training time, best-checkpoint time, full frontiers, and tuning cost are always
reported. With three final seeds, the operational margin is primary and the
confidence interval is uncertainty evidence rather than a powered equivalence
test.

## Invalidation conditions

The study is invalid if tuning constructs the endpoint loader, endpoint values
influence a recipe or target, the final command differs from the printed lock,
or non-finite values occur. Failure to find a stable recipe is a valid negative
result and blocks endpoint evaluation rather than changing the protocol.

## Results

Pending execution of the committed protocol and runner.
