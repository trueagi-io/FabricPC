# Future Strategic Improvements For FabricPC

## Purpose

This document outlines the highest-value future improvements for `FabricPC`
based on the current repository state, including:

- the core predictive-coding library under `fabricpc/`
- the continual-learning port under `fabricpc/continual/`
- the Split-MNIST examples and benchmark harness
- the existing summary and parity documents
- the archived development plans and roadmap

The goal is not to restate everything already present. The goal is to identify
where new code is most likely to improve actual results, not just code quality.

## Current Position

`FabricPC` already has several strong foundations:

- a reusable JAX graph-based predictive-coding engine
- a modular continual-learning stack with replay, support selection, causal
  guidance, typing/demotion, and TransWeave components
- examples that run end-to-end on Split-MNIST
- a notebook-parity regression harness with CI coverage
- a healthy automated test suite

At the same time, the repository still shows three clear gaps between
infrastructure quality and result quality:

1. The continual-learning control stack is richer than the model architecture
   used in the shipped examples.
2. The port captures representative V18/V20.2b behavior, but not the full
   architectural flavor of the original notebook lineage.
3. Predictive coding is competitive on small MLP-style examples, but still
   underperforms badly on harder settings such as the transformer examples.

That means the next improvements should not all be "more heuristics." The most
useful work is the work that improves measurable learning behavior under a
stronger experimental standard.

## Strategic Priorities

## 1. Close The Architecture Gap Between The Port And The Notebook Lineage

### Why this matters

The current continual-learning examples use a simple feedforward graph while the
notebooks were built around a more explicitly columnar and hierarchical system.
As a result, some of the ported control logic is being exercised on a weaker
host architecture than the source ideas assumed.

### High-value work

- Add a canonical "notebook-style" continual-learning architecture in code.
- Represent columns, shells, and composer pathways as first-class graph
  construction helpers rather than example-local conventions.
- Build one reference architecture that is explicitly meant to mirror the V18 /
  V20.2b setting more closely than the current 4-node examples.
- Re-run parity and Split-MNIST experiments on that architecture.

### Expected upside

- better alignment between the control regime and the model it is steering
- less ambiguity about whether weak results come from the control logic or the
  host network
- a stronger target for future research and benchmarking

## 2. Make Exact Support Evaluation Smarter And Cheaper

### Why this matters

The best idea in the notebook-to-port lineage is still the conservative
"proposal plus exact verification" pattern. That is likely where further result
gains will come from. The current code has improved batching, but the exact
audit logic is still relatively limited in scope and still expensive enough to
discourage broader search.

### High-value work

- Extend support search beyond single-swap or tiny local neighborhoods when
  confidence warrants it.
- Add staged exact evaluation:
  - cheap coarse screen
  - medium-fidelity local audit
  - exact final verification
- Build support-search caching keyed by task context and support overlap so near
  duplicates are not re-evaluated repeatedly.
- Track audit win rate by proposal source:
  - replay bank
  - selector policy
  - causal challenger
  - TransWeave proposal
- Learn when exact audits are worth spending budget on.

### Expected upside

- more useful support changes
- less wasted audit budget
- better long-horizon retention without making the system overly aggressive

## 3. Improve The Quality Of The Replay Signal

### Why this matters

The replay system is now more efficient, but the sampling policy itself is
still fairly generic. Better results will likely require smarter replay, not
just faster replay.

### High-value work

- Replace uniform replay with priority-aware replay.
- Prioritize samples by:
  - forgetting risk
  - audit disagreement
  - support sensitivity
  - causal training value
- Add class-balanced and task-balanced replay modes and compare them directly.
- Store lightweight metadata per replay sample so the trainer can ask for
  "hard old-task examples" rather than undifferentiated past data.
- Add support-conditioned replay:
  samples that were historically diagnostic for selecting between near-tied
  support sets should be replayed more often during later support decisions.

### Expected upside

- better retention at fixed replay budget
- more informative old-task gradients
- stronger causal/audit supervision from fewer examples

## 4. Turn Causal Guidance Into A Better Model, Not Just A Better Feature Pipe

### Why this matters

The feature construction and scoring path has been batched, but the predictive
model is still a relatively simple ridge-style mechanism. Better results may
depend more on model quality and target design than on further engineering
speedups.

### High-value work

- Improve the target used for causal training:
  predict exact gain, rank gain, and "safe to try" separately rather than
  collapsing everything into one scalar.
- Split the predictor into multiple heads:
  - expected current-task gain
  - expected old-task loss
  - confidence / uncertainty
  - swap safety
- Add pairwise or listwise ranking losses for candidate ordering.
- Use uncertainty-aware gating so low-confidence causal scores do not drive
  support changes even when the raw prediction is large.
- Add ablations comparing:
  - no causal predictor
  - scalar predictor
  - ranking predictor
  - uncertainty-aware predictor

### Expected upside

- fewer harmful support swaps
- better use of audit rows
- clearer explanation of when causal guidance is genuinely helping

## 5. Make TransWeave And Composer Logic Measurably Earn Their Complexity

### Why this matters

The code includes meaningful TransWeave and composer infrastructure, but the
examples still make it hard to tell whether those mechanisms are consistently
producing net value. This is a classic research-system risk: real complexity
without enough proof of payoff.

### High-value work

- Add direct benchmarks for TransWeave contribution:
  with and without composer transfer on the same seeds and tasks.
- Track exact acceptance rate and post-acceptance gain for composer-sourced
  proposals.
- Separate "bookkeeping active" from "causally responsible for gains" in the
  reporting.
- Add stronger regularization for transfer proposals that are novel but not yet
  proven safe.
- If needed, narrow TransWeave to the cases where it clearly outperforms simple
  replay plus exact verification.

### Expected upside

- less ambiguity about whether TransWeave is helping or just present
- better research discipline around a complex subsystem

## 6. Expand Evaluation Beyond Split-MNIST

### Why this matters

Split-MNIST is useful, but it is too small and too forgiving to be the only
serious target. If the code is meant to improve continual learning, results
should generalize beyond this one benchmark.

### High-value work

- Add at least one harder image continual-learning benchmark.
- Reasonable next candidates:
  - Permuted MNIST
  - Split Fashion-MNIST
  - Split CIFAR-10 or CIFAR-100
- Add one non-vision sequential benchmark if the graph abstraction is intended
  to generalize broadly.
- Extend the parity system with "performance tiers":
  - notebook-parity tier
  - harder continual-learning tier
  - architecture stress tier

### Expected upside

- results that mean more than "works on Split-MNIST"
- earlier detection of mechanisms that only succeed on toy settings

## 7. Raise The Experimental Standard For Result Claims

### Why this matters

The repository already has tests and parity checks, but stronger result quality
will require stronger experiment discipline.

### High-value work

- Add multi-seed benchmark runners for the key continual-learning profiles.
- Store seed distributions, not just one-run summaries.
- Track more than mean accuracy:
  - backward transfer
  - per-task retention curves
  - support stability
  - swap acceptance rate
  - audit precision
  - replay usefulness
- Create a benchmark report template so every major change records:
  - config
  - seeds
  - baseline
  - deltas
  - failure modes
- Add CI smoke checks for benchmark scripts and scheduled runs for heavier
  multi-seed jobs.

### Expected upside

- fewer false positives from lucky runs
- faster identification of genuinely useful new code

## 8. Improve The Core Predictive-Coding Engine On Harder Models

### Why this matters

The example documentation already shows a serious gap: predictive coding does
reasonably well on standard MNIST MLPs but performs poorly on the transformer
example relative to backprop. That is a strategic issue for the whole library.

### High-value work

- Improve initialization and inference schedules for deep PC models.
- Add adaptive inference step control instead of fixed `infer_steps`.
- Explore layerwise or node-groupwise inference rates.
- Add better support for normalization layers and stabilizing nodes.
- Add core profiling and node-group parallelization, following the archived
  parallelization plan, for graphs with repeated homogeneous blocks.
- Build "known hard case" regression benchmarks for transformer PC training.

### Expected upside

- stronger PC performance outside small MLP settings
- clearer understanding of whether the bottleneck is optimization, architecture,
  or implementation

## 9. Introduce Better Data And Task Curricula

### Why this matters

The current sequential tasks are fixed and simple. Better results may come not
only from better mechanisms, but also from better curricula that let the system
learn transferable structure earlier.

### High-value work

- Try curriculum ordering rather than a fixed digit-pair order.
- Cluster tasks by representational similarity and compare against arbitrary
  orderings.
- Add adaptive replay/curriculum scheduling based on forgetting trends.
- Introduce task descriptors or context embeddings into the support-selection
  pipeline.

### Expected upside

- stronger reuse of earlier supports
- less destructive interference between poorly ordered tasks

## 10. Convert More Heuristic State Into Learnable Or Measured State

### Why this matters

A lot of current behavior is controlled by manually tuned thresholds and scales.
That is acceptable in a research port, but it limits confidence and portability.

### High-value work

- Learn or calibrate trust/gating parameters from held-out audit history.
- Add automatic threshold calibration for support swap acceptance.
- Track uncertainty bands for the selector, causal model, and replay-bank
  proposals.
- Build a small meta-controller that allocates trust between:
  - teacher
  - selector policy
  - replay bank
  - causal predictor
  - exact audit

### Expected upside

- less brittle behavior across seeds and tasks
- fewer hand-tuned settings that only work on one benchmark

## Recommended Execution Order

If the goal is "better results soonest," the best sequence is:

1. Build a stronger notebook-style reference architecture.
2. Improve exact support evaluation and replay policy together.
3. Upgrade the causal model from scalar regression to ranked / uncertainty-aware
   guidance.
4. Expand benchmarking beyond Split-MNIST with multi-seed reporting.
5. Reassess TransWeave under stronger experimental controls.
6. Push deeper into core predictive-coding stabilization for harder models.

This ordering matters. It avoids spending a lot of time optimizing mechanisms
whose value is still being judged on a weak host architecture or a weak
benchmark.

## Concrete Candidate Projects For Us To Build Next

The most practical next code projects are:

- `notebook_style_split_mnist.py`
  - a stronger reference architecture and benchmark entry point
- priority-aware replay metadata and samplers
- a richer exact audit engine with cached staged evaluation
- a multi-head causal support predictor with uncertainty output
- a multi-seed benchmark runner and report writer
- a harder continual-learning benchmark package beyond Split-MNIST

## Decision Rule For Future Work

New work should be favored if it satisfies at least one of these:

- improves retention or final mean accuracy on the parity and continual
  benchmarks
- reduces harmful support changes at similar compute cost
- makes a complex subsystem prove its value with cleaner evidence
- improves PC stability on architectures where it currently fails

New work should be deprioritized if it mostly adds mechanism count without
producing a stronger benchmark result or a clearer experimental explanation.

## Bottom Line

`FabricPC` is already a good engineering port of a complicated notebook family.
The biggest remaining opportunity is no longer "port more logic." It is to make
the existing ideas compete on stronger architectures, stronger benchmarks, and
stronger experimental standards. The most promising result-oriented areas are:

- architecture parity with the notebook setting
- smarter exact support evaluation
- smarter replay
- a more capable causal predictor
- harder and broader benchmarks
- deeper stabilization of predictive coding on nontrivial models
