# FabricPC Port Summary

## Scope

This document summarizes the `FabricPC` repository after reviewing:

- `mnist_audit_guided_generality_v18_3_colab_v1.0.ipynb`
- `mnist_audit_guided_generality_v20_2b_colab.ipynb`
- `v_20_b_vs_v_18.pdf`
- the code under `FabricPC/`

The two notebooks were readable directly and show a Colab-centric, monolithic Split-MNIST continual-learning stack with generated module layers, replay-bank support search, causal guidance, typing/demotion logic, and later V20.2b conservative calibration of replay-bank, route, SB, and TransWeave mechanisms. The PDF was also extracted and read, and it confirms the high-level interpretation of the later branch as a replay-backed, exact-verified, anti-overwrite-calibrated control layer built around the V18 backbone.

## Executive Summary

`FabricPC` is not a literal notebook-to-notebook translation. It is a refactoring of the notebook ideas into two layers:

1. A reusable JAX predictive-coding engine for arbitrary graph-structured models.
2. A Split-MNIST continual-learning package that ports the notebook-specific support selection, causal guidance, replay, and TransWeave ideas into separate modules.

This is the right direction architecturally. The repository is substantially more maintainable than the source notebooks because the core graph mechanics, node abstractions, state initialization, training loops, continual-learning heuristics, and experiments are separated cleanly.

The implementation is also in good working order from a regression perspective: `154` tests passed locally. In addition, the shipped `split_mnist_continual.py` and `split_mnist_causal.py` examples both ran successfully in quick-smoke mode once MNIST was available locally.

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

## What The PDF Adds

The PDF is implementation-oriented and makes the V18 to V20.2b transition much clearer.

Its central claim is:

- V18 is an exact-teacher continual-learning system with exact boundary-time support search, controller rollout search, and one-swap maintenance.
- V20.2b keeps that backbone, but adds replay-backed online support proposals wrapped in conservative exact verification and anti-overwrite safeguards.

The most important changes highlighted in the PDF are:

- persistent selector state rather than mostly self-contained per-run state
- replay bank as a live support proposer rather than only an offline artifact
- conservative reselection against both the original chosen support and a local refinement baseline
- strong preference for high-overlap, local support repairs over radical support jumps
- task-history-aware anti-overwrite control
- muted but nonzero use of internal certificates near safe candidate regions

This PDF framing matches the structure of the port well. The key ported idea is not merely “more heuristics.” It is a new control regime around the V18 scaffold: replay may propose, but exact continual-learning-aware checks still decide.

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

## What The Shipped Examples Actually Exercise

I ran:

- `examples/split_mnist_continual.py --quick-smoke`
- `examples/split_mnist_causal.py --quick-smoke --num-tasks 3`

These runs are useful because they show what parts of the codebase are actually wired into a working end-to-end path today.

### Common runtime observations

- Both scripts build a simple 4-node feedforward graph: input -> hidden1 -> hidden2 -> output.
- They do not instantiate the richer column/composer notebook-style architecture directly.
- They do exercise the continual-learning infrastructure around that graph:
  - support selection
  - replay
  - audit row generation
  - causal predictor / trust logic
  - TransWeave bookkeeping
- Both scripts request `jax_platforms="cuda"`, but in this environment JAX could not initialize CUDA and fell back to CPU.
- Both required MNIST to be downloaded once through Keras before training could start.

### `split_mnist_continual.py` smoke run

Observed behavior:

- 5 tasks completed successfully
- mean test accuracy was about `0.9879`
- average forgetting was about `0.0748`
- replay activated after task 0 and grew from 500 to 2000 buffered samples
- the selected support columns remained `(0, 1, 6, 7)` throughout this smoke run
- causal examples accumulated, but `mix_gate` remained `0.0`, so causal guidance was effectively not steering support choice in that run
- TransWeave bookkeeping was active:
  - nonzero composer sources appeared from task 1 onward
  - shell demotions were recorded on later tasks

Artifacts were written under:

- `results/split_mnist/split_mnist_seed42_20260411_155447`

### `split_mnist_causal.py` smoke run

Observed behavior:

- 3 tasks completed successfully
- final mean accuracy across seen tasks was about `0.9556`
- average forgetting was about `0.0574`
- causal logic became visibly active by tasks 1 and 2
- `mix_gate` rose from `0.0` on task 0 to `0.2711` and then `0.3500`
- the causal predictor trained on audit rows and reported nonzero correlation by task 2
- the support columns in this run stayed `(0, 1, 2, 3)`

The causal example is therefore a better proof than the plain continual example that the V20-style support-guidance layer is actually live in the current codebase.

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

The PDF clarifies that this should be thought of as a support-proposal and conservative reselection pipeline, not just a selector utility.

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

The example runs confirm that this flow is not just theoretical. In the current repository, the support-selection, replay, audit, causal-update, and TransWeave bookkeeping path is all active in end-to-end execution.

## What Looks Well-Ported

- The predictive-coding engine is cleanly separated from the Split-MNIST experiment logic.
- The graph abstraction is substantially more reusable than the notebook-generated module stack.
- Predictive coding and backprop use the same model description, which makes baseline comparison easy.
- The continual-learning code preserves the notebook vocabulary instead of losing the research intent during refactoring.
- The repository includes broad test coverage across core mechanics, continual features, transformers, multi-GPU, initializers, and optimizers.
- The PDF’s framing of V20.2b as “replay proposes, exact audit judges” matches the actual shape of the `continual` package.
- The quick-smoke examples confirm that the continual infrastructure is operational rather than just partially ported scaffolding.

## Important Limitations In The Current Port

These are not fatal problems, but they matter when interpreting the code.

- Much of the continual-learning layer is still heuristic and Python-driven rather than JAX-native.
- The examples that exercise continual learning mostly use simple feedforward `Linear` graphs, while the richer notebook architecture described in the notebooks and PDF is represented more as support infrastructure and custom nodes than as one canonical end-to-end model.
- The port favors maintainability over exact notebook equivalence, so it should be treated as a research implementation inspired by the notebook family rather than a byte-for-byte reproduction.
- The smoke runs suggest that some mechanisms are present but muted under default quick-smoke settings. For example, the plain continual example accumulated causal data but never raised `mix_gate` above zero, so the causal layer did not materially influence support choice there.

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

This is now more important, not less, after running the examples. The shipped examples work, but they validate the control stack around a simple feedforward graph rather than validating notebook-style architectural parity.

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

Local example result:

- `examples/split_mnist_continual.py --quick-smoke` completed successfully on CPU after MNIST download
- `examples/split_mnist_causal.py --quick-smoke --num-tasks 3` completed successfully on CPU after MNIST download

Runtime caveats observed:

- the examples currently hard-request CUDA and then fall back to CPU if CUDA init fails
- plot export failed because Kaleido requires Chrome
- MNIST loading depends on a successful initial download unless the dataset is already cached locally

That gives good confidence that the repository is internally consistent, even though it does not by itself prove exact parity with the original notebooks.

## Bottom Line

`FabricPC` is a credible and useful refactoring of the notebook research code into a maintainable package. The strongest part of the port is the general JAX predictive-coding engine; the most notebook-specific part is the `fabricpc.continual` package, which preserves the support-bank, causal, and TransWeave ideas in a cleaner form.

The PDF and example runs make the current status more precise:

- the repository does implement the V20.2b-style control story
- the shipped examples do exercise that control stack successfully
- but the most visible end-to-end examples still sit on top of a relatively simple feedforward graph rather than a canonical notebook-style columnar architecture

The biggest remaining opportunity is therefore twofold:

- performance: move more continual-learning hot paths out of Python/NumPy and into batched JAX code
- parity: create a canonical full-stack continual-learning example and benchmark that more directly expresses the architecture described in the notebooks and PDF
