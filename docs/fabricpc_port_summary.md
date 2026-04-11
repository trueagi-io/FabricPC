# FabricPC Port Summary

## Scope

This document summarizes the `FabricPC` repository after reviewing:

- `mnist_audit_guided_generality_v18_3_colab_v1.0.ipynb`
- `mnist_audit_guided_generality_v20_2b_colab.ipynb`
- `v_20_b_vs_v_18.pdf`
- the code under `FabricPC/`

The two notebooks were readable directly and show a Colab-centric, monolithic Split-MNIST continual-learning stack with generated module layers, replay-bank support search, causal guidance, typing/demotion logic, and later V20.2b conservative calibration of replay-bank, route, SB, and TransWeave mechanisms. The PDF could not be machine-extracted with the local shell tools available in this session, so the comparison details below are grounded primarily in the notebooks themselves and the corresponding ported module docstrings and config comments inside `FabricPC`.

## Executive Summary

`FabricPC` is not a literal notebook-to-notebook translation. It is a refactoring of the notebook ideas into two layers:

1. A reusable JAX predictive-coding engine for arbitrary graph-structured models.
2. A Split-MNIST continual-learning package that ports the notebook-specific support selection, causal guidance, replay, and TransWeave ideas into separate modules.

This is the right direction architecturally. The repository is substantially more maintainable than the source notebooks because the core graph mechanics, node abstractions, state initialization, training loops, continual-learning heuristics, and experiments are separated cleanly.

The implementation is also in good working order from a regression perspective: `154` tests passed locally.

## What The Source Notebooks Contained

The source notebooks are large Colab runtimes that:

- mount Google Drive and manage checkpoint / snapshot plumbing
- generate versioned Python modules inline as strings
- define a Split-MNIST continual-learning experiment stack
- train hierarchical / columnar models task-by-task
- maintain replay banks for support and demotion
- run exact or near-exact support audits at task boundaries
- add causal guidance from audit outcomes
- introduce conservative V20.2b calibration so new mechanisms remain helpful without overwhelming long-horizon continual learning

The `V20.2b additions` markdown in the newer notebook makes the main intent explicit: keep the runtime scaffold, but apply replay-bank support, routing, SB, and TransWeave more conservatively and gate them by evidence.

## How FabricPC Is Organized

### 1. Core predictive-coding engine

The reusable engine lives under `fabricpc/core`, `fabricpc/graph`, `fabricpc/builder`, `fabricpc/nodes`, and `fabricpc/training`.

Key pieces:

- `fabricpc/core/types.py`
  - immutable JAX pytrees for `GraphStructure`, `GraphParams`, `GraphState`, `NodeInfo`, and `NodeState`
- `fabricpc/builder/graph_builder.py`
  - validates nodes and edges, resolves slots, builds `GraphStructure`, and installs graph/state-initializer config
- `fabricpc/core/inference.py`
  - generic predictive-coding inference loop: zero latent grads, run local forward/error propagation, update latents
- `fabricpc/graph/state_initializer.py`
  - graph-level initialization strategies: global random, node-level distribution, or feedforward initialization
- `fabricpc/graph/graph_net.py`
  - parameter initialization and local weight-gradient computation after inference converges
- `fabricpc/training/train.py`
  - predictive-coding training loop using local gradients and Optax updates
- `fabricpc/training/train_backprop.py`
  - a standard backprop baseline using the same graph description
- `fabricpc/training/train_autoregressive.py`
  - autoregressive / causal-mask path for transformer-style models

This layer is the biggest improvement over the notebooks. The notebook code generated versioned modules around one experiment family; `FabricPC` turns the common mechanics into a library.

### 2. Node abstraction

The node interface is centered on `fabricpc/nodes/base.py`.

Each node defines:

- input slots
- parameter initialization
- forward computation
- local inference gradients
- local learning gradients

Current concrete nodes include:

- `Linear`
- `IdentityNode`
- transformer-related nodes in `transformer.py` and `transformer_v2.py`
- continual-learning-specific nodes in `fabricpc/continual/nodes.py`

This is one of the clearest examples of the port improving the original design. The notebooks bundled many behaviors into evolving model classes; the repo exposes node-level extension points instead.

### 3. Continual-learning layer ported from the notebooks

The notebook-specific logic lives under `fabricpc/continual`.

Main modules:

- `config.py`
  - flattened dataclass configuration replacing notebook-scattered flags
- `data.py`
  - Split-MNIST task loaders and task wrappers
- `trainer.py`
  - `SequentialTrainer`, the main task-by-task orchestration entry point
- `support.py`
  - support bank, demotion bank, replay buffer, selector policy, trust controller
- `causal.py`
  - causal fingerprint bank, contribution predictor, trust gates, feature builder, routing and SB helpers
- `weight_causal.py`
  - per-weight non-Gaussianity detection and adaptive correction
- `transweave.py`
  - composer-level and shell-demotion transfer logic
- `native_nodes.py`
  - FabricPC-native linear nodes with embedded causal / TransWeave hooks
- `optimal_transport.py`
  - Sinkhorn and related transport utilities

This package is the clearest port of the V18/V20 notebook family into maintainable code. Several docstrings explicitly call out that they were ported from the `mnist_audit_guided_generality` notebooks.

## Mapping From Notebook Ideas To Repository Modules

### Predictive-coding runtime

Notebook concept:
- iterative latent inference plus local learning

FabricPC port:
- `fabricpc/core/inference.py`
- `fabricpc/graph/graph_net.py`
- `fabricpc/training/train.py`

### Graph/model construction

Notebook concept:
- monolithic model classes and generated module snapshots

FabricPC port:
- `fabricpc/builder/graph_builder.py`
- `fabricpc/nodes/*`
- `fabricpc/core/types.py`

### Split-MNIST sequential training

Notebook concept:
- task-by-task continual-learning loop with resume/checkpoint support

FabricPC port:
- `fabricpc/continual/data.py`
- `fabricpc/continual/trainer.py`

### Support-bank selection and demotion

Notebook concept:
- support replay, exact audits, demotion tracking, hybrid selector policy

FabricPC port:
- `fabricpc/continual/support.py`
- `fabricpc/continual/config.py`

### Causal guidance from support-swap audit data

Notebook concept:
- learn from boundary audits which columns help future tasks

FabricPC port:
- `fabricpc/continual/causal.py`
- `fabricpc/continual/trainer.py`

### SB clarity / routing / conservative gating

Notebook concept:
- do not allow auxiliary selectors to dominate without evidence

FabricPC port:
- trust and gating logic in `support.py` and `causal.py`
- config parameters in `continual/config.py`

### TransWeave and transport-based transfer

Notebook concept:
- transfer between composer/shell representations with transport-style alignment

FabricPC port:
- `fabricpc/continual/transweave.py`
- `fabricpc/continual/augmentation.py`
- `fabricpc/continual/optimal_transport.py`

### Backprop baseline

Notebook concept:
- compare local-learning PC against more standard training

FabricPC port:
- `fabricpc/training/train_backprop.py`

## Actual Execution Flow In FabricPC

For a standard predictive-coding run:

1. Build a graph from node objects and edges.
2. Initialize node parameters with `initialize_params`.
3. Initialize graph state with one of the state initializers.
4. Clamp task inputs, and optionally labels during PC training.
5. Run iterative inference to reduce node energies.
6. Compute local node-level weight gradients from the converged state.
7. Apply Optax updates.

For the continual-learning path:

1. `SequentialTrainer.train_task()` asks `SupportManager` for active support columns.
2. If replay is enabled, an `InterleavedLoader` mixes current-task and replay samples.
3. Training runs in PC mode or backprop mode using the same graph structure.
4. The trainer evaluates the task.
5. Support-swap audits generate training signals for causal components.
6. The support manager updates trust, fingerprint, and predictor state.
7. Per-weight causal stats and TransWeave task-end registration are updated.
8. Samples are stored in the replay buffer for future tasks.

This is essentially the notebook workflow, but broken into composable modules.

## What Looks Well-Ported

- The predictive-coding engine is cleanly separated from the Split-MNIST experiment logic.
- The graph abstraction is substantially more reusable than the notebook-generated module stack.
- Predictive coding and backprop use the same model description, which makes baseline comparison easy.
- The continual-learning code preserves the notebook vocabulary instead of losing the research intent during refactoring.
- The repository includes broad test coverage across core mechanics, continual features, transformers, multi-GPU, initializers, and optimizers.

## Important Limitations In The Current Port

These are not fatal problems, but they matter when interpreting the code.

- Much of the continual-learning layer is still heuristic and Python-driven rather than JAX-native.
- The examples that exercise continual learning mostly use simple feedforward `Linear` graphs, while the richer notebook architecture is represented more as support infrastructure and custom nodes than as one canonical end-to-end model.
- The port favors maintainability over exact notebook equivalence, so it should be treated as a research implementation inspired by the notebook family rather than a byte-for-byte reproduction.

## Improvement Opportunities

### 1. Move more continual-learning hot paths out of Python/NumPy and into JAX

Highest-value performance improvement.

The core PC engine is JAX-based, but much of the continual layer uses Python lists, NumPy arrays, and per-column loops:

- `fabricpc/continual/support.py`
- `fabricpc/continual/causal.py`
- parts of `fabricpc/continual/trainer.py`

That is fine for orchestration, but it limits scaling and makes boundary-time selection logic CPU-bound. The best improvement would be to convert scoring, similarity, and candidate evaluation paths into batched JAX functions.

Expected benefit:
- faster task-boundary audits
- less Python overhead
- easier profiling and accelerator use

### 2. Vectorize causal feature scoring instead of building features column-by-column

`SupportManager._apply_causal_guidance()` builds features and predictions in small Python loops. The same pattern appears in several support-selection helpers. These paths should be batched so all candidate columns are scored in one array program.

Expected benefit:
- lower task-transition latency
- simpler code for ranking and swapping
- easier future extension to larger column counts

### 3. Separate orchestration from algorithms inside `SequentialTrainer`

`fabricpc/continual/trainer.py` has become the integration center for:

- replay
- support selection
- audits
- causal updates
- per-weight causal logic
- TransWeave registration
- checkpointing
- metrics

It works, but it is doing too much. Splitting task-boundary audit logic, replay integration, and transfer registration into smaller collaborators would make performance tuning and correctness testing easier.

Expected benefit:
- lower maintenance risk
- easier profiling
- smaller, more targeted unit tests

### 4. Make notebook-to-port parity measurable

The repo has strong unit tests, but there is no obvious parity harness that says:

- given a fixed Split-MNIST seed and config
- this port matches or closely tracks notebook behavior

Adding a small benchmark/regression suite for representative V18 and V20.2b settings would reduce uncertainty when changing support-selection or causal logic.

Expected benefit:
- safer refactors
- clearer scientific reproducibility
- easier diagnosis of LLM-port drift

### 5. Use `node_order` consistently, and specialize inference for acyclic graphs

Several core loops iterate over `structure.nodes` directly instead of using `structure.node_order`. Because dict insertion order is stable in modern Python, this is usually fine, but the explicit topological order already exists and should be the default traversal path for acyclic graphs.

A useful extension would be:

- a fast path for DAGs using topological order
- a separate path for cyclic graphs or SCC groups

Expected benefit:
- clearer semantics
- better opportunities for compile-time specialization
- easier reasoning about feedforward initialization vs recurrent inference

### 6. Add end-to-end profiling and benchmark artifacts

The repo has correctness coverage, but performance claims are not yet strongly documented. A benchmark folder with:

- inference throughput
- train-step time
- task-boundary audit time
- replay overhead
- multi-GPU scaling

would make optimization work much more disciplined.

### 7. Strengthen documentation around the “canonical” continual architecture

The continual package contains rich components, but the clearest examples still center on simpler MLP-style graphs. A dedicated example that uses the full column/composer/TransWeave stack would make the port easier to understand and validate against the notebook lineage.

Expected benefit:
- easier onboarding
- clearer scientific story
- less ambiguity about which features are experimental vs production-ready

## Recommended Priority Order

If the goal is better runtime performance and higher confidence in the port, the best order is:

1. Vectorize and JAX-ify support-selection and causal-scoring hot paths.
2. Add notebook-parity regression benchmarks for representative Split-MNIST settings.
3. Split `SequentialTrainer` into smaller boundary-management components.
4. Add a canonical full-stack continual-learning example and benchmark script.

## Validation Status

Local test result:

- `154 passed in 76.18s`

That gives good confidence that the repository is internally consistent, even though it does not by itself prove exact parity with the original notebooks.

## Bottom Line

`FabricPC` is a credible and useful refactoring of the notebook research code into a maintainable package. The strongest part of the port is the general JAX predictive-coding engine; the most notebook-specific part is the `fabricpc.continual` package, which preserves the support-bank, causal, and TransWeave ideas in a cleaner form.

The biggest remaining opportunity is performance: the core model mechanics are JAX-native, but a meaningful portion of the continual-learning logic is still Python/NumPy orchestration inherited in spirit from the notebook workflow. Moving those hot paths into batched JAX code, and then measuring parity against representative notebook settings, would improve both speed and confidence substantially.
