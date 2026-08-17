# ePC versus sPC independent optimization: Runpod report — 2026-08-17

## Outcome

The preregistered combined capability-and-performance gate failed. Independent
tuning chose a materially different recipe for each solver, but ePC remained
well below sPC within the declared eight-epoch observation horizon:

- mean held-out endpoint accuracy was `0.30066667` for ePC and `0.39853333`
  for sPC;
- the paired ePC-minus-sPC endpoint difference was `-0.09786667`, versus a
  preregistered noninferiority margin of `0.01`;
- ePC reached the locked `0.3904` validation target in 0/3 final runs; and
- sPC reached it in 2/3 final runs, so the all-runs target condition also
  failed independently of the accuracy comparison.

This is a valid negative result for the [preregistered candidate spaces and
eight-epoch protocol](2026-08-15-independent-optimization.md). It is not a
claim that ePC cannot reach the same limiting result with a longer horizon,
different schedule family, or broader search. Such a claim requires a new
preregistered experiment and an untouched endpoint.

## Protocol lineage and execution

The study follows the repository's [validation
methodology](../../validation_methodology.md). It separates implementation
correctness from empirical capability and allows each solver its own learning
rate, decay schedule, inference-step count, and checkpoint.

The benchmark ran from source commit
`403aeaa89d4095c444820e652f37d808e10e79f2` on one secure Runpod Pod:

| Property | Value |
| --- | --- |
| Runpod Pod | `i86i6rvo58eyq9` |
| Data center | `EU-RO-1` |
| Accelerator | NVIDIA GeForce RTX 4090, one GPU |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Python environment | JAX/JAXlib 0.11.0, TensorFlow CPU 2.21.0, Optuna 4.9.0 |
| Train split | CIFAR-10 `train[:80%]`, 40,000 examples, 157 batches |
| Validation split | CIFAR-10 `train[80%:90%]`, 5,000 examples, 20 batches |
| Endpoint split | CIFAR-10 `train[90%:]`, 5,000 examples |

JAX was verified on the GPU and TensorFlow reported no GPU devices, preventing
the input pipeline from reserving accelerator memory. Tuning and final mode
used byte-identical resolved package manifests. Dataset files were checksummed
again immediately before final mode.

The earlier local 19-candidate prefix was not resumed. Timing participates in
selection and in the final claim, so mixing local accelerator timings with the
Runpod hardware would have made the evidence harder to interpret. The Runpod
tuning process therefore restarted at candidate 1 and produced a complete,
self-contained log.

Optuna was installed in the environment, but this run intentionally enumerated
the finite preregistered Cartesian grids. Adaptive search after observing
candidate results was outside the locked protocol.

## Tuning audit and selection

The tuning log contains:

- 224 `CURVE` records: eight epochs for each of 28 runs;
- 24 screen results with seed `271828` and four confirmation results with seed
  `314159`;
- 28/28 `RUN_RESULT` records with `endpoint_accuracy=None` and
  `endpoint_seconds=None`;
- two `SHORTLIST`, four `CANDIDATE_SUMMARY`, and one `LOCKED` record; and
- no traceback, CUDA error, non-finite value, or duplicate result key.

The confirmation summaries were:

| Solver | Candidate | Stable on both seeds | Mean best validation accuracy | SE | Mean time to best (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| ePC | `ePC-s10-lr0.0003-d5` | yes | 0.3030 | 0.0134 | 188.471 |
| ePC | `ePC-s10-lr0.0003-d3` | yes | 0.2825 | 0.0123 | 177.291 |
| sPC | `sPC-s120-lr0.001-d5` | yes | 0.3987 | 0.0025 | 1325.466 |
| sPC | `sPC-s120-lr0.003-d3` | yes | 0.4061 | 0.0089 | 1324.875 |

The resulting lock was:

| Solver | Inference steps | Peak learning rate | Decay horizon |
| --- | ---: | ---: | ---: |
| ePC | 10 | 0.0003 | 5 epochs |
| sPC | 120 | 0.003 | 3 epochs |

The common target was `0.3904`, the minimum validation accuracy over the final
three epochs of both selected-sPC tuning runs. The selected sPC recipe reached
it on both tuning seeds at about 628 seconds. The selected ePC recipe reached
it on neither tuning seed. Total candidate-training time was 3,048.354 seconds
for ePC and 12,997.796 seconds for sPC; search cost is separate from final
time-to-target.

## One-shot final results

Final mode was invoked once with the exact printed lock. The log contains 48
epoch curves, six run results, six final results, and exactly six non-null
endpoint values: one checkpoint per solver for each of three paired seeds.
There were no retries, duplicate result keys, non-finite values, CUDA errors,
or tracebacks.

| Seed | ePC best validation | sPC best validation | ePC endpoint | sPC endpoint | ePC time to target (s) | sPC time to target (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3192 | 0.4104 | 0.3142 | 0.4168 | not reached | 627.300 |
| 1000 | 0.3080 | 0.4032 | 0.3038 | 0.4006 | not reached | 627.556 |
| 2000 | 0.2898 | 0.3856 | 0.2840 | 0.3782 | not reached | not reached |

| Solver | Mean best validation ± SE | Mean endpoint ± SE | Mean time to best ± SE (s) | Total final training (s) |
| --- | ---: | ---: | ---: | ---: |
| ePC | 0.30567 ± 0.00857 | 0.30067 ± 0.00886 | 179.073 ± 7.398 | 625.981 |
| sPC | 0.39973 ± 0.00737 | 0.39853 ± 0.01119 | 959.410 ± 331.884 | 4868.588 |

The three paired endpoint differences were `-0.1026`, `-0.0968`, and
`-0.0942`. Their mean was `-0.09786667`; a two-sided 95% paired-t interval
computed from those three values is `[-0.10855, -0.08718]`. The runner reported
`t=-39.4174`, `p=0.00064299`, and Cohen's `d=-22.7576`. With only three seeds,
the preregistered one-point operational margin and the raw paired outcomes are
more informative than the nominal test statistics.

The performance gate was:

| Condition | Result |
| --- | --- |
| ePC noninferior within 0.01 endpoint accuracy | failed |
| Every final run reaches target 0.3904 | failed |
| ePC reaches target faster | failed / undefined because ePC never reached it |
| Combined gate | failed |

## Interpretation for debugging

The target choice is not what creates the substantive accuracy gap. The target
only determines the time-to-accuracy gate; the separately evaluated held-out
endpoint shows a 9.79 percentage-point mean deficit, nearly ten times the
preregistered practical margin. Validation and endpoint gaps are similar,
which argues against validation overfitting as the explanation.

ePC was much cheaper per epoch and reached its own best checkpoint sooner, but
that is not yet a practical speed advantage because it did so at substantially
lower accuracy. A useful speed claim requires comparable quality.

The most important limitation is search-boundary pressure. The selected ePC
candidate used the largest tested inference-step count, largest tested learning
rate, and longest tested decay horizon. Together with the short eight-epoch
horizon, that suggests the declared search may truncate ePC's attainable
regime. This is a hypothesis for a new experiment, not a post-hoc
reinterpretation of this endpoint result. The valid endpoint has now been
consumed and cannot be used to choose an extension.

External review should focus on whether the ePC update equations, optimizer
interaction, state initialization, and schedule parameterization explain the
low plateau, and on designing a longer-horizon, broader-boundary protocol with
a fresh endpoint. The current evidence does not justify changing this run's
gate or selectively extending failed seeds.

## Commands

Tuning:

```text
/opt/fabricpc-venv/bin/python examples/epc_resnet18_optimized.py --mode tune
```

Final:

```text
/opt/fabricpc-venv/bin/python examples/epc_resnet18_optimized.py --mode final \
  --epc_steps 10 --epc_lr 0.0003 --epc_decay_epochs 5 \
  --spc_steps 120 --spc_lr 0.003 --spc_decay_epochs 3 \
  --target_accuracy 0.39040000
```

## Evidence and reproducibility

The [evidence directory](2026-08-16-runpod-evidence/README.md) includes raw
tuning and final logs, launch and completion sidecars, exact dependency
manifests, environment installation output, dataset hashes, redacted Runpod
resource metadata, teardown verification, a sanitized structured summary, and
a SHA-256 manifest.

The raw log hashes are:

- tuning: `38884e84f15dd4b0539622b589a77cd09aaddfc59b22efcc1ad5a6b9c227381e`;
- final: `018764160c7da56a39e8dd16835b286be7eeba9f07d9fb0e80b57281a1545089`.

Runpod billing and account-balance captures are intentionally excluded from
this externally reviewable bundle because they are expense evidence, not
scientific or debugging evidence.
