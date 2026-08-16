# Validation methodology

This document defines how FabricPC distinguishes implementation correctness
from empirical solver capability and performance. A benchmark must state which
claim it supports; passing one layer does not imply the next.

## 1. Correctness validation

Correctness tests isolate implementation behavior at fixed parameters and
inputs. Appropriate checks include independent gradient oracles, fixed-point
identities, state invariants, finite-value checks, JIT coverage, topology
validation, and comparisons on analytically tractable graphs.

Matching hyperparameters, update counts, and initial states is useful here
because the purpose is causal diagnosis. These controlled ablations answer
whether two implementations behave differently under the same conditions.
They do not establish either solver's best attainable accuracy or practical
speed.

## 2. Capability validation

Capability studies ask whether each solver can reach the same empirical task
quality under a recipe appropriate to its own training dynamics. Solvers must
therefore be tuned independently. Unless the study explicitly concerns a
shared setting, learning rate, decay horizon, inference steps, inference rate,
training duration, and checkpoint epoch are not required to match.

The architecture, data, preprocessing, task loss, hardware, and paired random
initialization/minibatch seeds remain controlled. The optimizer family may be
held fixed to bound the search, but its hyperparameters and schedule are
solver-specific.

Attainable quality is measured at the best validation-selected checkpoint of a
recipe that remains stable near the end of its declared observation horizon.
Final-epoch accuracy is not substituted for best-checkpoint accuracy unless
that is the deployment rule being tested.

## 3. Performance validation

Performance studies compare best-so-far validation accuracy against cumulative
training wall time. Equal epochs are not a performance control because solver
epochs can have different costs and dynamics.

The primary timing measures are:

- time to a common accuracy target derived from the tuned reference solver;
- best accuracy under common wall-time budgets; and
- time to each solver's selected best checkpoint.

The full accuracy-time frontier is retained so a conclusion does not depend on
one convenient threshold. Hyperparameter-search cost is reported separately
from time-to-target.

Cold-start training time includes first-use JIT compilation. Validation and
endpoint evaluation time is excluded from training time and recorded
separately when material. Runs use the same hardware without a competing
compute workload.

On an active-display machine, ordinary graphics-only contexts and compositor
activity are allowed and recorded as host conditions. Any additional CUDA
compute or mixed compute/graphics process with nonzero accelerator utilization
is competing compute. The affected in-progress run is excluded in full;
timing samples are never trimmed after the fact.

## 4. Data isolation and selection

Every study declares disjoint training, validation, and endpoint splits before
tuning. Training and validation data may be used for recipe selection,
checkpoint selection, stability checks, target construction, and stopping
rules. Endpoint data may not influence any of those choices.

The endpoint loader must not be constructed in tuning mode. Its first use in a
valid study occurs only after both solver recipes, the common target, the
analysis rules, and final seeds are locked. Each final model is evaluated once
at its validation-selected checkpoint.

An invalidated exploratory run is excluded from all later decisions. By
project decision, the invalidated 2026-08-14 recipe-transfer run is treated as
non-evidence, and its endpoint values may not influence the replacement ePC
study. `train[90%:]` is designated as that replacement study's held-out
endpoint.

## 5. Preregistration and auditability

Before results are observed, a study records:

- its question and supported claim;
- data splits and loader-access rules;
- candidate spaces and tuning resource allocation;
- random seeds and pairing;
- candidate, checkpoint, and target selection rules;
- plateau or convergence criteria;
- final success criteria and statistical summaries;
- timing boundaries; and
- conditions that invalidate or extend the study.

The protocol and executable runner are committed before tuning. Raw per-epoch
curves, per-seed endpoints, selected recipes, environment details, and exact
commands are retained in machine-readable artifacts. Deviations are documented
before further endpoint evaluation.

Validation-only tuning may resume after an operational interruption only when
the runner can verify a complete, endpoint-free, contiguous prefix of the
preregistered candidate order. Complete means the full declared epoch curve
and internally consistent timing, checkpoint, stability, and result records.
The interrupted candidate is discarded in full and rerun from its first
epoch. Prefix retention is determined solely by order, record completeness,
and compute-isolation monitoring—not by observed validation values. A resumed
log replays the accepted prefix so every subsequent log is self-contained.

This recovery rule ends when endpoint evaluation begins. Final runs are
started only after an accelerator-isolation preflight; once any held-out
endpoint has been evaluated, the study is not retried or selectively resumed.
An interruption or competing compute after that point is reported as a study
failure requiring a newly preregistered endpoint protocol.

## 6. Claims and uncertainty

Final reports separate observed facts from inference. They report per-seed
values, means and standard errors, paired differences, confidence intervals,
and effect sizes where defined. Small-seed studies are labeled accordingly.

“Faster” requires reaching the common target and doing so in less training
time. “Same limiting result” requires a stable validation plateau and a locked
endpoint difference inside the preregistered practical margin. A failure to
transfer one solver's training recipe to another supports neither a capability
nor a limiting-accuracy claim.
