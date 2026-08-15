# ePC ResNet-18 time-to-accuracy follow-up — 2026-08-14

> **Invalidated on 2026-08-15.** This run tested transfer of a short-horizon
> selected learning rate into a differently scaled long-horizon schedule. It
> does not answer whether independently optimized ePC reaches sPC's empirical
> limit faster, and none of its endpoint values may influence the replacement
> study. The retained report and CSVs are an audit record only.

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

### Execution integrity

The protocol and runner were committed as `cbe3b2a` before either command was
run. The tuning phase produced all 18 expected runs and 54 validation-curve
points. Every tuning `RUN_RESULT` recorded `holdout_accuracy=None` and
`official_test_accuracy=None`. Only after the runner printed
`SELECTED steps=10 lr=0.0003` was that exact pair supplied to final mode.

Final mode produced all six expected runs and 45 validation-curve points. All
reported values were finite. The complete machine-readable results are:

- [tuning observations](./2026-08-14-time-accuracy-tuning.csv);
- [final validation curves](./2026-08-14-time-accuracy-final-curves.csv);
- [final per-seed endpoints](./2026-08-14-time-accuracy-final-endpoints.csv).

### Validation-only selection

| ePC steps | Peak LR | Mean final validation | SE | Mean train time |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.0003 | 27.31% | 1.53% | 51.84s |
| 2 | 0.0010 | 10.50% | 0.32% | 51.56s |
| 2 | 0.0030 | 9.47% | 0.37% | 51.80s |
| 5 | 0.0003 | 27.58% | 1.04% | 70.05s |
| 5 | 0.0010 | 9.78% | 0.18% | 70.57s |
| 5 | 0.0030 | 10.16% | 0.10% | 71.17s |
| **10** | **0.0003** | **27.92%** | **1.26%** | **101.74s** |
| 10 | 0.0010 | 19.71% | 2.05% | 101.83s |
| 10 | 0.0030 | 9.35% | 0.25% | 103.85s |

The selected candidate was **10 steps at learning rate 0.0003**. It exceeded
the five-step runner-up by 0.34 percentage points. Both larger learning rates
were unstable or near chance by the third epoch, so the grid also shows that
ePC's training behavior is highly learning-rate-sensitive.

### Locked final comparison

| Seed | ePC holdout | sPC holdout | Difference | ePC train | sPC train |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.64% | 34.26% | -22.62 pp | 352.51s | 481.91s |
| 1000 | 9.60% | 32.52% | -22.92 pp | 351.28s | 487.97s |
| 2000 | 10.98% | 31.16% | -20.18 pp | 352.53s | 501.37s |

| Aggregate (mean +/- SE) | ePC | sPC | Difference |
| --- | ---: | ---: | ---: |
| Final validation accuracy | 10.45 +/- 0.16% | 32.84 +/- 0.92% | -22.39 pp |
| Primary holdout accuracy | 10.74 +/- 0.60% | 32.65 +/- 0.90% | -21.91 pp |
| Official test accuracy | 10.74 +/- 0.22% | 32.40 +/- 0.85% | -21.66 pp |
| Training time | 352.10 +/- 0.41s | 490.42 +/- 5.75s | -138.31s |

The paired primary-holdout comparison gave `t=-25.2478`, `p=0.001565`, and
paired Cohen's `d=-14.5768`. Its mean 95% confidence interval was -25.64 to
-18.17 percentage points. On the 5,000-example holdout, the mean gap
corresponds to about 1,095 fewer correct predictions per ePC model.

| Preregistered criterion | Result | Verdict |
| --- | --- | --- |
| Mean ePC holdout accuracy >= mean sPC accuracy | 10.74% < 32.65% | **FAIL** |
| Mean ePC train time <= mean sPC train time | 352.10s < 490.42s | PASS |
| Overall practical gate | Accuracy condition failed | **FAIL** |

ePC used 71.80% of sPC's training time, a 1.39x speedup, and saved 138.31
seconds on average. The result therefore is not a failure to meet the time
proxy: it is an accuracy failure under the locked training recipe.

### What the accuracy failure means

The final-epoch gap is practically real for this recipe, not a near miss
against an arbitrary threshold. All three ePC runs collapsed to approximately
chance accuracy, while all three sPC runs remained above 31% on the fresh
holdout. A system deployed with this unmodified ePC schedule would be unusable
at the selected endpoint.

It is not, however, evidence that ePC inherently tops out near 10% or cannot
approach sPC accuracy. The validation curves identify a repeatable training
collapse rather than a low early ceiling:

| Epoch | Mean ePC validation | Mean train time |
| ---: | ---: | ---: |
| 1 | 22.06% | 52.24s |
| 3 | **30.25%** | 101.87s |
| 4 | 29.05% | 126.93s |
| 5 | 17.86% | 151.77s |
| 7 | 11.01% | 201.89s |
| 13 | 10.45% | 351.88s |

At the common validation peak in epoch 3, ePC averaged 30.25 +/- 1.06%, only
2.59 percentage points below sPC's final 32.84%, after about 21% as much
training time. This is descriptive secondary evidence only: the protocol did
not checkpoint or evaluate the holdout at that epoch, so it cannot replace the
preregistered final endpoint.

The experiment also exposed a schedule-transfer mismatch. The cosine schedule
is scaled to each run's total epoch count. During three-epoch tuning, the
learning rate was about 1% of its peak by the end of epoch 3. During the
13-epoch final run, it was still about 91% of peak at epoch 3 and 73% at epoch
5. Thus the winning peak rate was selected under a rapidly decaying schedule
but deployed under a much longer high-rate schedule. The fact that all three
runs fell sharply between epochs 4 and 7 makes this the leading explanation,
although the current experiment does not isolate causality.

The practical conclusion is narrower and more useful than “ePC is inaccurate”:
ePC learns quickly and is substantially cheaper, but its current optimizer
schedule is not stable when its speed advantage is converted directly into
more epochs. A credible next study should tune the full long-horizon schedule
on validation data, save the best-validation checkpoint, and use a newly
declared endpoint under the replacement methodology. By project decision,
`train[90%:]` is restored as that study's held-out endpoint and this invalid
run is excluded from all decisions.

### Final verification

- `./ve/bin/pytest -q tests`: **402 passed** in 84.56 seconds. The only output
  was the two expected warnings from Optuna's experimental multivariate and
  grouped TPE options.
- The focused protocol/result suite passed all 19 tests.
- Black, Ruff, and `git diff --check` passed.
