# AGI-Oriented Continual Learning + Transfer Learning Implementation Plan

## Based on: "Toward AGI-friendly Benchmarks for Continual + Transfer Learning" (Goertzel, April 2026)

This document provides comprehensive recommendations for implementing the AGI-oriented CL+TL benchmark framework from Ben Goertzel's paper into the FabricPC codebase.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Paper Key Concepts](#2-paper-key-concepts)
3. [FabricPC Current State Analysis](#3-fabricpc-current-state-analysis)
4. [Gap Analysis](#4-gap-analysis)
5. [Implementation Architecture](#5-implementation-architecture)
6. [Detailed Component Specifications](#6-detailed-component-specifications)
7. [Implementation Phases](#7-implementation-phases)
8. [Integration Points](#8-integration-points)
9. [Testing Strategy](#9-testing-strategy)
10. [Future Extensions](#10-future-extensions)

---

## 1. Executive Summary

### Paper's Core Thesis

The paper argues that AGI-oriented continual learning evaluation should focus on **CL+TL together**, not just anti-forgetting. The key insight is that a system should:

1. **Acquire reusable concepts** under non-stationarity
2. **Retain them** under explicit resource budgets
3. **Redeploy them selectively** on related but novel tasks
4. **Transfer across modalities, interfaces, and task categories**

### FabricPC's Strengths

FabricPC provides an excellent foundation with:

- **Predictive Coding Engine**: Local Hebbian learning (biologically plausible)
- **Sophisticated CL Infrastructure**: Support columns, causal fingerprints, TransWeave transfer
- **JAX-Based Architecture**: Immutable pytrees, JIT compilation, functional patterns
- **Existing Benchmark Framework**: `cl-benchmark` with classical CL metrics

### Implementation Strategy

Create a new `fabricpc/benchmarks/` module that:
- Implements the 4-axis benchmark matrix
- Extends metrics beyond classical CL
- Provides resource-constrained evaluation tracks
- Enables concept graph-based transfer analysis

---

## 2. Paper Key Concepts

### 2.1 The 4-Axis Benchmark Matrix

The paper proposes benchmarks as a cross-product: **B ⊆ F × S × P × R**

#### Axis 1: Task Families (F)

| Family | Description | Priority for FabricPC |
|--------|-------------|----------------------|
| T0: Control streams | Split-MNIST, Permuted-MNIST | **Already implemented** |
| T1: Realistic drift | CLEAR-style temporal evolution | High |
| T2: Compositional abstraction | ARC-style concept formation | High |
| T3: Multimodal transfer | Vision-language streams | Medium |
| T4: Sequential control | Continual RL | Medium |
| T5: Language/code streams | LLM domain adaptation | Low |
| T6: Minecraft Concept Ecology | New embodied benchmark | Future |
| T7: Object-Affordance Ecology | New household benchmark | Future |

#### Axis 2: Surface Types (S)

| Surface | Description | FabricPC Mapping |
|---------|-------------|------------------|
| Raw | Pixels, audio, tokens | Standard input (images) |
| Frozen-feature | Pre-extracted embeddings | Could use pretrained encoders |
| Structured | Object lists, scene graphs, predicates | **Maps to support columns** |
| Query/explanation | Reuse predictions, plans | Could expose causal fingerprints |

#### Axis 3: Stream Protocols (P)

| Protocol | Description | Current Status |
|----------|-------------|----------------|
| Abrupt switch | Sharp task boundaries | **Implemented** |
| Gradual drift | Slow distribution shift | Needs implementation |
| Recurring tasks | Same concept returns after gap | Needs implementation |
| Interleaved | Multiple concurrent tasks | Partial (replay) |
| Compositional recombination | Novel combinations of learned concepts | Needs implementation |
| Hidden boundaries | Task ID not revealed | Needs implementation |

#### Axis 4: Resource Tracks (R)

| Track | Description | Current Status |
|-------|-------------|----------------|
| Fixed-capacity | Parameters/memory capped | Not implemented |
| Bounded-growth | Growth charged against budget | Not implemented |
| Open-growth | Unlimited but metered | Partial (checkpoint size) |

### 2.2 Extended Metrics

#### Classical CL Panel (Already in FabricPC)

```
ACC_T = (1/T) Σ a_{T,i}           # Average accuracy
BWT = (1/(T-1)) Σ (a_{T,i} - a_{i,i})   # Backward transfer
FWT = (1/(T-1)) Σ (a_{i-1,i} - b_i)     # Forward transfer
Forgetting_i = max_{t} a_{t,i} - a_{T,i}
```

#### Transfer-Centred Additions (NEW - Need Implementation)

| Metric | Formula/Description | Priority |
|--------|---------------------|----------|
| **Few-shot transfer gain** | Gain@k(j) = p_CL(k) - p_scratch(k) | High |
| **Threshold gain** | ThresholdGain(τ) = n_scratch(τ) - n_CL(τ) | High |
| **Transfer matrix** | T_ij(k) = p_j(k \| trained on i) - p_scratch(k) | High |
| **Transfer selectivity** | Mean transfer on related pairs vs unrelated | High |
| **Negative transfer rate** | Fraction where T_ij < -ε | High |
| **Recurrence gain** | Speedup when concept recurs after gap | High |
| **Cross-surface transfer** | Transfer from raw→structured, etc. | Medium |
| **Task-ID dependence gap** | Performance with vs without task IDs | Medium |
| **Order robustness** | Variance across task orderings | Medium |

#### Resource & Explanation Metrics (NEW)

| Metric | Description | Priority |
|--------|-------------|----------|
| **Growth efficiency** | Performance gain / parameter growth | High |
| **Query efficiency** | Performance / external queries | Medium |
| **Reuse attribution** | System predicts what it will reuse | Medium |
| **Relation-model accuracy** | Induced graph vs gold concept graph | Medium |
| **Low-vs-high disruption** | Forgetting correlation with feature level | Low |

### 2.3 CL Strategy Comparison

The paper emphasizes comparing different **loci of continual learning**:

| Strategy | Description | FabricPC Analog |
|----------|-------------|-----------------|
| Global weight updates | Update entire network | Standard PC training |
| Modular growth | Frozen base + adapters | **Shell demotion system** |
| Domain-conditioned | Shared body with context signal | **Causal fingerprint routing** |
| Symbolic CL | Stable perception + explicit knowledge | **Support column selection** |

---

## 3. FabricPC Current State Analysis

### 3.1 Core Architecture

```
FabricPC/
├── fabricpc/
│   ├── core/           # Immutable types, activations, energy, inference
│   ├── nodes/          # NodeBase abstraction, Linear, Transformer, etc.
│   ├── graph/          # Graph construction, parameter management
│   ├── training/       # train_pcn(), train_backprop(), optimizers
│   ├── continual/      # CL-specific modules
│   │   ├── trainer.py       # SequentialTrainer (1793 lines)
│   │   ├── support.py       # SupportManager, ReplayBuffer (1383 lines)
│   │   ├── causal.py        # CausalFingerprintBank (1446 lines)
│   │   ├── transweave.py    # TransWeaveManager (1030 lines)
│   │   ├── weight_causal.py # Per-weight adaptation
│   │   └── config.py        # ExperimentConfig dataclasses
│   └── utils/          # Dashboarding, callbacks
└── cl-benchmark/       # Framework-agnostic evaluation
    ├── metrics/        # forgetting.py, averaging.py
    ├── evaluation/     # BenchmarkRunner, protocol
    └── baselines/      # naive, replay, EWC
```

### 3.2 Existing CL Mechanisms

#### Support Column System (`support.py`)
- **SupportState**: Tracks active columns per task
- **SupportBank**: Bank of successful support patterns
- **ReplayBuffer**: Per-task sample storage with interleaved loading
- **HybridSelectorPolicy**: Multiple scoring strategies

**Paper mapping**: This is a form of **modular growth** where non-shared columns act as task-specific adapters.

#### Causal Guidance System (`causal.py`)
- **CausalFingerprintBank**: Per-column, per-task activation statistics
- **CausalContributionPredictor**: Ridge regression for column value
- **CausalSelectorTrustController**: Multi-gate confidence blending
- **AgreementTracker**: Prediction vs outcome tracking

**Paper mapping**: This enables **domain-conditioned routing** based on learned task signatures.

#### TransWeave Transfer (`transweave.py`)
- **ComposerTransWeave**: Transfer composition patterns via Sinkhorn
- **ShellDemotionTransWeave**: Transfer neuron shell membership
- **TransWeaveManager**: Orchestrates multi-level transfer

**Paper mapping**: This is explicit **transfer learning** using optimal transport for knowledge alignment.

### 3.3 Existing Metrics (`cl-benchmark/metrics/`)

```python
# Already implemented:
compute_accuracy_matrix(model, tasks)
compute_forgetting(accuracy_matrix)
compute_backward_transfer(accuracy_matrix)
compute_forward_transfer(accuracy_matrix)
compute_average_accuracy(accuracy_matrix)
compute_intransigence(accuracy_matrix)
```

---

## 4. Gap Analysis

### 4.1 What Exists vs What's Needed

| Paper Component | Current Status | Gap Level |
|-----------------|---------------|-----------|
| Classical CL metrics | ✅ Complete | None |
| Abrupt-switch protocol | ✅ Complete | None |
| Replay buffer | ✅ Complete | None |
| Modular architecture (columns) | ✅ Complete | None |
| Transfer via optimal transport | ✅ Complete | None |
| Causal routing | ✅ Complete | None |
| **Few-shot transfer gain** | ❌ Missing | **High** |
| **Transfer matrix T_ij** | ❌ Missing | **High** |
| **Recurrence gain** | ❌ Missing | **High** |
| **Resource tracking** | ❌ Missing | **High** |
| **Gradual drift stream** | ❌ Missing | **Medium** |
| **Recurring tasks stream** | ❌ Missing | **Medium** |
| **Hidden boundary stream** | ❌ Missing | **Medium** |
| **Concept graph** | ❌ Missing | **Medium** |
| **Cross-surface transfer** | ❌ Missing | **Low** |
| **Embodied benchmarks** | ❌ Missing | **Future** |

### 4.2 Priority Matrix

```
                    Value to Paper Goals
                    Low         High
                ┌───────────┬───────────┐
Implementation  │ Defer     │ Phase 2   │
Effort: High    │ (Embodied)│ (Streams) │
                ├───────────┼───────────┤
Implementation  │ Optional  │ Phase 1   │
Effort: Low     │ (Viz)     │ (Metrics) │
                └───────────┴───────────┘
```

---

## 5. Implementation Architecture

### 5.1 New Module Structure

```
FabricPC/fabricpc/benchmarks/           # NEW MODULE
├── __init__.py
├── config.py                           # BenchmarkMatrixConfig
│
├── task_families/                      # Axis 1
│   ├── __init__.py
│   ├── base.py                         # TaskFamilyBase protocol
│   ├── control_streams.py              # Split-MNIST, etc.
│   ├── compositional.py                # ARC-style tasks
│   └── drift_datasets.py               # CLEAR-style drift
│
├── surfaces/                           # Axis 2
│   ├── __init__.py
│   ├── base.py                         # SurfaceType protocol
│   ├── raw.py                          # Raw sensory
│   ├── frozen_feature.py               # Pre-extracted
│   ├── structured.py                   # Symbolic/graph
│   └── query.py                        # Explanation interface
│
├── streams/                            # Axis 3
│   ├── __init__.py
│   ├── base.py                         # StreamProtocol base
│   ├── abrupt_switch.py                # Existing behavior
│   ├── gradual_drift.py                # NEW
│   ├── recurring.py                    # NEW
│   ├── interleaved.py                  # NEW
│   ├── compositional_recomb.py         # NEW
│   └── hidden_boundary.py              # NEW
│
├── resources/                          # Axis 4
│   ├── __init__.py
│   ├── tracker.py                      # ResourceTracker
│   ├── fixed_capacity.py               # Hard budget
│   ├── bounded_growth.py               # Controlled growth
│   └── open_growth.py                  # Unlimited
│
├── metrics/                            # Extended metrics
│   ├── __init__.py
│   ├── classical.py                    # Re-export existing
│   ├── transfer_matrix.py              # T_ij(k) pairwise
│   ├── few_shot.py                     # Gain@k, threshold
│   ├── recurrence.py                   # Recurrence gain
│   ├── selectivity.py                  # Transfer selectivity
│   ├── resource_metrics.py             # Growth/query efficiency
│   └── reuse_attribution.py            # Reuse scoring
│
├── ecology/                            # Concept graphs
│   ├── __init__.py
│   ├── concept_graph.py                # ConceptGraph class
│   ├── minecraft.py                    # Stub for future
│   └── object_affordance.py            # Stub for future
│
├── strategies/                         # CL strategy comparison
│   ├── __init__.py
│   ├── base.py                         # CLStrategy protocol
│   ├── global_update.py                # End-to-end updates
│   ├── modular_growth.py               # Frozen base + adapters
│   └── domain_conditioned.py           # Shared with context
│
├── runner.py                           # AGIBenchmarkRunner
├── results.py                          # AGIBenchmarkResults
└── visualization.py                    # Extended plotting
```

### 5.2 Configuration System

```python
# fabricpc/benchmarks/config.py

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional

@dataclass(frozen=True)
class StreamProtocolConfig:
    """Stream protocol parameters."""
    num_tasks: int = 20
    task_duration_steps: int = 1000
    drift_rate: float = 0.01
    recurrence_pattern: Tuple[int, ...] = ()
    recurrence_gaps: Tuple[str, ...] = ("short", "medium", "long")
    interleave_probability: float = 0.0
    reveal_task_ids: bool = True

@dataclass(frozen=True)
class ResourceTrackConfig:
    """Resource budget parameters."""
    track_type: str = "bounded_growth"  # fixed, bounded, open
    max_parameters: Optional[int] = None
    max_memory_mb: Optional[float] = None
    parameter_growth_budget_per_task: float = 0.1
    query_budget_per_task: int = 1000
    track_flops: bool = True

@dataclass(frozen=True)
class MetricsConfig:
    """Which metrics to compute."""
    # Classical (from existing cl-benchmark)
    compute_accuracy_matrix: bool = True
    compute_forgetting: bool = True
    compute_bwt: bool = True
    compute_fwt: bool = True
    compute_intransigence: bool = True

    # Transfer-centred (NEW)
    compute_transfer_matrix: bool = True
    compute_few_shot_gain: bool = True
    few_shot_k_values: Tuple[int, ...] = (1, 5, 10, 20)
    compute_transfer_selectivity: bool = True
    compute_negative_transfer_rate: bool = True
    compute_recurrence_gain: bool = True

    # Resource (NEW)
    compute_growth_efficiency: bool = True
    compute_query_efficiency: bool = False
    compute_reuse_attribution: bool = False

@dataclass(frozen=True)
class BenchmarkMatrixConfig:
    """
    4-Axis benchmark configuration from Goertzel paper.

    Defines a single cell in the benchmark matrix:
    B ⊆ F × S × P × R
    """
    # Axis 1: Task Family
    task_family: str = "control_streams"
    task_family_config: Dict[str, Any] = field(default_factory=dict)

    # Axis 2: Surface Type
    surface_type: str = "raw"
    surface_config: Dict[str, Any] = field(default_factory=dict)

    # Axis 3: Stream Protocol
    stream_protocol: str = "abrupt_switch"
    stream_config: StreamProtocolConfig = field(
        default_factory=StreamProtocolConfig
    )

    # Axis 4: Resource Track
    resource_track: str = "bounded_growth"
    resource_config: ResourceTrackConfig = field(
        default_factory=ResourceTrackConfig
    )

    # Metrics selection
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # CL strategy being evaluated
    cl_strategy: str = "modular_growth"

    # Run configuration
    num_runs: int = 3
    random_seed: int = 42
```

---

## 6. Detailed Component Specifications

### 6.1 Stream Protocol Base

```python
# fabricpc/benchmarks/streams/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Dict, Any, Optional, Tuple
import jax.numpy as jnp

@dataclass
class StreamStep:
    """Single step in a task stream."""
    task_id: int
    batch_data: Tuple[jnp.ndarray, jnp.ndarray]  # (x, y)
    task_boundary: bool  # True if this is first step of new task
    boundary_visible: bool  # True if learner knows about boundary
    is_recurrence: bool = False  # True if task appeared before
    recurrence_gap: Optional[int] = None  # Tasks since last occurrence
    metadata: Dict[str, Any] = field(default_factory=dict)

class StreamProtocol(ABC):
    """
    Base class for stream protocols (Axis 3).

    Generates a sequence of StreamSteps representing
    the non-stationary learning experience.
    """

    def __init__(self, config: StreamProtocolConfig, tasks: List[TaskData]):
        self.config = config
        self.tasks = tasks
        self._task_history: List[int] = []

    @abstractmethod
    def __iter__(self) -> Iterator[StreamStep]:
        """Iterate through stream steps."""
        pass

    @property
    @abstractmethod
    def total_steps(self) -> int:
        """Total number of steps in stream."""
        pass

    @property
    def provides_task_ids(self) -> bool:
        """Whether stream reveals task IDs to learner."""
        return self.config.reveal_task_ids

    def get_recurrence_info(self, task_id: int) -> Tuple[bool, Optional[int]]:
        """Check if task is recurring and compute gap."""
        if task_id in self._task_history:
            last_idx = len(self._task_history) - 1 - \
                       self._task_history[::-1].index(task_id)
            gap = len(self._task_history) - last_idx - 1
            return True, gap
        return False, None
```

### 6.2 Recurring Tasks Stream

```python
# fabricpc/benchmarks/streams/recurring.py

class RecurringStream(StreamProtocol):
    """
    Stream where tasks recur after gaps of varying length.

    From paper: "the recurrence gap should be parametrized in terms
    of both intervening task count and total intervening experience
    volume, since these can come apart."

    Implements short (2-3 tasks), medium (5-8 tasks), and
    long (10+ tasks) recurrence gaps.
    """

    def __init__(self, config: StreamProtocolConfig, tasks: List[TaskData]):
        super().__init__(config, tasks)
        self.recurrence_schedule = self._build_recurrence_schedule()

    def _build_recurrence_schedule(self) -> List[Tuple[int, str]]:
        """
        Build schedule of (task_id, gap_type) pairs.

        Returns list like:
        [(0, 'first'), (1, 'first'), (2, 'first'),
         (0, 'short'),  # recurs after 2 tasks
         (3, 'first'), (4, 'first'), ...,
         (1, 'medium'), # recurs after 6 tasks
         ...]
        """
        schedule = []
        # Initial presentation of each task
        for i in range(len(self.tasks)):
            schedule.append((i, 'first'))

        # Add recurrences based on config
        if 'short' in self.config.recurrence_gaps:
            # Recur task 0 after 2-3 intervening tasks
            schedule.insert(3, (0, 'short'))

        if 'medium' in self.config.recurrence_gaps:
            # Recur task 1 after 5-8 intervening tasks
            schedule.insert(9, (1, 'medium'))

        if 'long' in self.config.recurrence_gaps:
            # Recur task 0 after 10+ intervening tasks
            schedule.append((0, 'long'))

        return schedule

    def __iter__(self) -> Iterator[StreamStep]:
        for task_id, gap_type in self.recurrence_schedule:
            task = self.tasks[task_id]
            is_recurrence = gap_type != 'first'

            for batch_idx, (x, y) in enumerate(task.train_batches()):
                is_first = batch_idx == 0
                yield StreamStep(
                    task_id=task_id,
                    batch_data=(x, y),
                    task_boundary=is_first,
                    boundary_visible=self.config.reveal_task_ids,
                    is_recurrence=is_recurrence,
                    recurrence_gap=self._compute_gap(task_id) if is_recurrence else None,
                    metadata={'gap_type': gap_type},
                )
            self._task_history.append(task_id)
```

### 6.3 Transfer Matrix Computation

```python
# fabricpc/benchmarks/metrics/transfer_matrix.py

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

@dataclass
class TransferMatrixResult:
    """
    Pairwise transfer matrix T_ij(k).

    From paper: "The transfer matrix should be reported as a heat map
    or table. This is much more informative than a single
    forward-transfer scalar."
    """
    matrix: np.ndarray  # Shape: (num_tasks, num_tasks, len(k_values))
    k_values: Tuple[int, ...]
    selectivity: float  # Related vs unrelated transfer
    negative_transfer_rate: float  # Fraction T_ij < 0

    def to_heatmap(self, k: int) -> np.ndarray:
        """Extract matrix for specific k value."""
        k_idx = self.k_values.index(k)
        return self.matrix[:, :, k_idx]

def compute_transfer_matrix(
    model_factory: Callable,
    tasks: List,
    k_values: Tuple[int, ...] = (1, 5, 10, 20),
    relation_graph: Optional['ConceptGraph'] = None,
) -> TransferMatrixResult:
    """
    Compute pairwise transfer matrix T_ij(k).

    T_ij(k) = accuracy on task j after k examples,
              having been pre-trained on task i,
              minus baseline accuracy (scratch) on j with k examples.

    Args:
        model_factory: Function that creates fresh model instance
        tasks: List of TaskData objects
        k_values: Few-shot sample counts to evaluate
        relation_graph: Optional concept graph for selectivity

    Returns:
        TransferMatrixResult with full transfer analysis

    Note from paper: "For a 20-task stream, full pairwise ablation
    means (20 choose 2) = 190 ablation runs--which is expensive but
    not absurd for the structured tracks."
    """
    n_tasks = len(tasks)
    matrix = np.zeros((n_tasks, n_tasks, len(k_values)))

    # Compute baseline few-shot accuracy (no pre-training)
    baselines = {}
    for j, target_task in enumerate(tasks):
        baselines[j] = {}
        for k_idx, k in enumerate(k_values):
            model = model_factory()
            acc = _train_few_shot_and_eval(model, target_task, k)
            baselines[j][k] = acc

    # Compute transfer from each source to each target
    for i, source_task in enumerate(tasks):
        for j, target_task in enumerate(tasks):
            if i == j:
                continue

            for k_idx, k in enumerate(k_values):
                # Train fully on source, then few-shot on target
                model = model_factory()
                model.train_on_task(source_task)
                transfer_acc = _train_few_shot_and_eval(model, target_task, k)

                # T_ij(k) = transfer_acc - baseline_acc
                matrix[i, j, k_idx] = transfer_acc - baselines[j][k]

    # Compute selectivity if relation graph provided
    selectivity = 0.0
    if relation_graph:
        related_pairs = relation_graph.get_related_pairs()
        unrelated_pairs = relation_graph.get_unrelated_pairs()

        related_transfer = np.mean([
            matrix[i, j, :].mean()
            for i, j in related_pairs
        ])
        unrelated_transfer = np.mean([
            matrix[i, j, :].mean()
            for i, j in unrelated_pairs
        ])
        selectivity = related_transfer - unrelated_transfer

    # Compute negative transfer rate
    negative_transfer_rate = np.mean(matrix < -0.01)  # ε = 0.01

    return TransferMatrixResult(
        matrix=matrix,
        k_values=k_values,
        selectivity=selectivity,
        negative_transfer_rate=negative_transfer_rate,
    )
```

### 6.4 Few-Shot Transfer Gain

```python
# fabricpc/benchmarks/metrics/few_shot.py

@dataclass
class FewShotGainResult:
    """
    Few-shot transfer gain metrics.

    From paper: "This asks not 'how well do you eventually do?'
    but 'how much faster do you learn because of history?'"
    """
    gain_at_k: Dict[int, float]  # k -> gain percentage
    threshold_gain: Optional[float]  # Steps saved to reach baseline
    learning_curve_with_transfer: np.ndarray
    learning_curve_scratch: np.ndarray

def compute_few_shot_gain(
    model_with_history,
    model_factory: Callable,
    target_task,
    k_values: Tuple[int, ...] = (1, 2, 5, 10, 20, 50),
    target_threshold: Optional[float] = None,
) -> FewShotGainResult:
    """
    Compute few-shot transfer gain (Gain@k).

    Gain@k(j) = p_CL(k) - p_scratch(k)

    Where:
    - p_CL(k) = accuracy after k examples WITH transfer history
    - p_scratch(k) = accuracy after k examples from scratch
    """
    gains = {}
    curve_transfer = []
    curve_scratch = []

    for k in k_values:
        # With transfer
        acc_transfer = _eval_at_k(model_with_history, target_task, k)
        curve_transfer.append(acc_transfer)

        # From scratch
        fresh_model = model_factory()
        acc_scratch = _eval_at_k(fresh_model, target_task, k)
        curve_scratch.append(acc_scratch)

        gains[k] = acc_transfer - acc_scratch

    # Threshold gain: how many fewer examples to reach target?
    threshold_gain = None
    if target_threshold:
        k_transfer = _find_k_for_threshold(curve_transfer, k_values, target_threshold)
        k_scratch = _find_k_for_threshold(curve_scratch, k_values, target_threshold)
        if k_transfer and k_scratch:
            threshold_gain = k_scratch - k_transfer

    return FewShotGainResult(
        gain_at_k=gains,
        threshold_gain=threshold_gain,
        learning_curve_with_transfer=np.array(curve_transfer),
        learning_curve_scratch=np.array(curve_scratch),
    )
```

### 6.5 Resource Tracker

```python
# fabricpc/benchmarks/resources/tracker.py

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

@dataclass
class ResourceSnapshot:
    """Resource usage at a point in time."""
    task_id: int
    step: int
    parameter_count: int
    memory_bytes: int
    cumulative_flops: float
    query_count: int
    accuracy: Optional[float] = None

class ResourceTracker:
    """
    Track resource usage for bounded-capacity evaluation.

    From paper: "resource constraints are part of the science.
    It is not enough to say that growing architectures, replay
    buffers, or external memories are either allowed or disallowed
    in principle--they should instead be measured under explicit
    budget tracks."
    """

    def __init__(self, config: ResourceTrackConfig):
        self.config = config
        self.snapshots: List[ResourceSnapshot] = []
        self.initial_params: Optional[int] = None

    def count_parameters(self, params) -> int:
        """Count total parameters in pytree."""
        return sum(
            np.prod(leaf.shape)
            for leaf in jax.tree_util.tree_leaves(params)
            if hasattr(leaf, 'shape')
        )

    def estimate_memory(self, params) -> int:
        """Estimate memory in bytes."""
        return sum(
            leaf.nbytes
            for leaf in jax.tree_util.tree_leaves(params)
            if hasattr(leaf, 'nbytes')
        )

    def record(self, task_id: int, step: int, params, accuracy: float = None):
        """Record resource snapshot."""
        param_count = self.count_parameters(params)

        if self.initial_params is None:
            self.initial_params = param_count

        self.snapshots.append(ResourceSnapshot(
            task_id=task_id,
            step=step,
            parameter_count=param_count,
            memory_bytes=self.estimate_memory(params),
            cumulative_flops=0.0,  # TODO: Track via JAX profiling
            query_count=0,
            accuracy=accuracy,
        ))

    def check_budget(self, params) -> Tuple[bool, Dict[str, float]]:
        """
        Check if resource budget is satisfied.

        Returns:
            (within_budget, utilization_dict)
        """
        utilization = {}
        within_budget = True

        param_count = self.count_parameters(params)

        if self.config.track_type == "fixed_capacity":
            if self.config.max_parameters:
                util = param_count / self.config.max_parameters
                utilization['parameters'] = util
                within_budget &= util <= 1.0

        elif self.config.track_type == "bounded_growth":
            if self.initial_params:
                growth = (param_count - self.initial_params) / self.initial_params
                allowed = self.config.parameter_growth_budget_per_task * \
                          len(set(s.task_id for s in self.snapshots))
                utilization['growth'] = growth / allowed if allowed > 0 else 0
                within_budget &= growth <= allowed

        return within_budget, utilization

    def compute_growth_efficiency(self) -> float:
        """
        Compute growth efficiency metric.

        Growth efficiency = accuracy gain / parameter growth
        """
        if len(self.snapshots) < 2:
            return 0.0

        first = self.snapshots[0]
        last = self.snapshots[-1]

        if first.accuracy is None or last.accuracy is None:
            return 0.0

        acc_gain = last.accuracy - first.accuracy
        param_growth = (last.parameter_count - first.parameter_count) / \
                       first.parameter_count

        if param_growth <= 0:
            return float('inf') if acc_gain > 0 else 0.0

        return acc_gain / param_growth
```

### 6.6 Concept Graph

```python
# fabricpc/benchmarks/ecology/concept_graph.py

from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
import networkx as nx

@dataclass
class ConceptNode:
    """
    Node in the latent concept graph.

    From paper: "The graph is hidden from the learner but known
    to the evaluator. It enables relation-conditioned transfer
    analysis and reuse attribution scoring."
    """
    id: str
    level: str  # "primitive", "world_concept", "skill", "meta_transfer"
    dependencies: Set[str] = field(default_factory=set)
    tasks_requiring: Set[int] = field(default_factory=set)

    # Tracking
    acquired: bool = False
    acquisition_task: Optional[int] = None
    acquisition_step: Optional[int] = None

class ConceptGraph:
    """
    Latent concept graph for tracking concept ecology.

    Mirrors the structure from paper's Minecraft and
    Object-Affordance examples.
    """

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self._graph = nx.DiGraph()

    def add_concept(self, node: ConceptNode):
        """Add concept to graph."""
        self.nodes[node.id] = node
        self._graph.add_node(node.id, **vars(node))
        for dep in node.dependencies:
            self._graph.add_edge(dep, node.id)

    def get_related_pairs(self) -> List[Tuple[int, int]]:
        """
        Get task pairs that share concepts.

        Used for computing transfer selectivity.
        """
        pairs = []
        for node in self.nodes.values():
            tasks = list(node.tasks_requiring)
            for i in range(len(tasks)):
                for j in range(i+1, len(tasks)):
                    pairs.append((tasks[i], tasks[j]))
        return list(set(pairs))

    def get_unrelated_pairs(self) -> List[Tuple[int, int]]:
        """Get task pairs that don't share concepts."""
        all_tasks = set()
        for node in self.nodes.values():
            all_tasks.update(node.tasks_requiring)

        related = set(self.get_related_pairs())
        unrelated = []

        for i in all_tasks:
            for j in all_tasks:
                if i < j and (i, j) not in related:
                    unrelated.append((i, j))

        return unrelated

    def compute_reuse_attribution(self, task_id: int) -> float:
        """
        Compute reuse attribution score for a task.

        From paper: "Before starting each task, ask the system
        which prior tasks, modules, or concepts it expects to reuse.
        Score this prediction against the hidden prerequisite set."
        """
        required_concepts = [
            node for node in self.nodes.values()
            if task_id in node.tasks_requiring
        ]

        acquired_prereqs = sum(
            1 for node in required_concepts
            if all(
                self.nodes[dep].acquired
                for dep in node.dependencies
            )
        )

        total_prereqs = len(required_concepts)
        return acquired_prereqs / total_prereqs if total_prereqs > 0 else 1.0
```

### 6.7 AGI Benchmark Runner

```python
# fabricpc/benchmarks/runner.py

from fabricpc.benchmarks.config import BenchmarkMatrixConfig
from fabricpc.benchmarks.streams import create_stream
from fabricpc.benchmarks.resources import ResourceTracker
from fabricpc.benchmarks.metrics import (
    compute_transfer_matrix,
    compute_few_shot_gain,
    compute_recurrence_gain,
)
from cl_benchmark.metrics import (
    compute_forgetting,
    compute_backward_transfer,
    compute_forward_transfer,
)

@dataclass
class AGIBenchmarkResults:
    """
    Extended benchmark results for AGI-oriented CL+TL evaluation.

    From paper: "The primary report should include:
    1. a metric vector,
    2. a transfer matrix,
    3. relation-conditioned transfer curves,
    4. performance-versus-budget plots,
    5. order-robustness statistics,
    6. CL strategy type and ablation results,
    7. a short catalogue of characteristic failure cases."
    """
    config: BenchmarkMatrixConfig

    # Classical CL
    accuracy_matrix: np.ndarray
    acc_t: float
    bwt: float
    fwt: float
    forgetting: float
    intransigence: float

    # Transfer-centred
    transfer_matrix: Optional[TransferMatrixResult] = None
    few_shot_gains: Optional[Dict[int, FewShotGainResult]] = None
    recurrence_gains: Optional[Dict[str, float]] = None  # gap_type -> gain
    transfer_selectivity: float = 0.0
    negative_transfer_rate: float = 0.0

    # Resource
    resource_snapshots: List[ResourceSnapshot] = field(default_factory=list)
    growth_efficiency: float = 0.0

    # Order robustness
    order_variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for serialization."""
        return {
            'classical': {
                'acc_t': self.acc_t,
                'bwt': self.bwt,
                'fwt': self.fwt,
                'forgetting': self.forgetting,
                'intransigence': self.intransigence,
            },
            'transfer': {
                'selectivity': self.transfer_selectivity,
                'negative_transfer_rate': self.negative_transfer_rate,
            },
            'resource': {
                'growth_efficiency': self.growth_efficiency,
            },
            'config': {
                'task_family': self.config.task_family,
                'stream_protocol': self.config.stream_protocol,
                'resource_track': self.config.resource_track,
                'cl_strategy': self.config.cl_strategy,
            },
        }

class AGIBenchmarkRunner:
    """
    Extended benchmark runner for AGI-oriented CL+TL evaluation.

    Implements the full 4-axis framework from the paper.
    """

    def __init__(
        self,
        config: BenchmarkMatrixConfig,
        model_factory: Callable,
        tasks: List,
        concept_graph: Optional[ConceptGraph] = None,
    ):
        self.config = config
        self.model_factory = model_factory
        self.tasks = tasks
        self.concept_graph = concept_graph

        self.stream = create_stream(config.stream_protocol, config.stream_config, tasks)
        self.resource_tracker = ResourceTracker(config.resource_config)

    def evaluate(self, verbose: bool = True) -> AGIBenchmarkResults:
        """
        Run full AGI-oriented CL+TL evaluation.
        """
        model = self.model_factory()
        accuracy_matrix = []

        # Phase 1: Sequential training
        current_task = -1
        for step in self.stream:
            if step.task_boundary:
                if current_task >= 0:
                    # Evaluate on all seen tasks
                    accs = self._evaluate_all_tasks(model, current_task + 1)
                    accuracy_matrix.append(accs)

                    # Record resources
                    self.resource_tracker.record(
                        current_task,
                        len(accuracy_matrix),
                        model.params,
                        accuracy=np.mean(accs),
                    )

                current_task = step.task_id

            # Train step
            model.train_step(step.batch_data)

        # Final evaluation
        accs = self._evaluate_all_tasks(model, current_task + 1)
        accuracy_matrix.append(accs)
        accuracy_matrix = np.array(accuracy_matrix)

        # Phase 2: Compute classical metrics
        results = AGIBenchmarkResults(
            config=self.config,
            accuracy_matrix=accuracy_matrix,
            acc_t=np.mean(accuracy_matrix[-1]),
            bwt=compute_backward_transfer(accuracy_matrix),
            fwt=compute_forward_transfer(accuracy_matrix),
            forgetting=compute_forgetting(accuracy_matrix),
            intransigence=0.0,  # TODO
        )

        # Phase 3: Compute transfer metrics
        if self.config.metrics.compute_transfer_matrix:
            results.transfer_matrix = compute_transfer_matrix(
                self.model_factory,
                self.tasks,
                self.config.metrics.few_shot_k_values,
                self.concept_graph,
            )
            results.transfer_selectivity = results.transfer_matrix.selectivity
            results.negative_transfer_rate = results.transfer_matrix.negative_transfer_rate

        if self.config.metrics.compute_recurrence_gain:
            results.recurrence_gains = self._compute_recurrence_gains(model)

        # Phase 4: Compute resource metrics
        results.resource_snapshots = self.resource_tracker.snapshots
        results.growth_efficiency = self.resource_tracker.compute_growth_efficiency()

        return results
```

---

## 7. Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)

**Priority: Critical**

| Task | Files | Effort |
|------|-------|--------|
| Create benchmarks module structure | `fabricpc/benchmarks/__init__.py` | 1 day |
| Configuration dataclasses | `fabricpc/benchmarks/config.py` | 2 days |
| Stream protocol base + abrupt switch | `fabricpc/benchmarks/streams/` | 2 days |
| Resource tracker | `fabricpc/benchmarks/resources/tracker.py` | 2 days |
| Import classical metrics | `fabricpc/benchmarks/metrics/classical.py` | 1 day |

**Deliverable**: Basic benchmark framework that matches existing functionality.

### Phase 2: Extended Metrics (Weeks 3-4)

**Priority: High**

| Task | Files | Effort |
|------|-------|--------|
| Transfer matrix computation | `metrics/transfer_matrix.py` | 3 days |
| Few-shot gain metrics | `metrics/few_shot.py` | 2 days |
| Transfer selectivity & NTR | `metrics/selectivity.py` | 1 day |
| Recurrence gain | `metrics/recurrence.py` | 2 days |
| Resource efficiency metrics | `metrics/resource_metrics.py` | 1 day |

**Deliverable**: All transfer-centred metrics from the paper.

### Phase 3: Extended Stream Protocols (Weeks 5-6)

**Priority: High**

| Task | Files | Effort |
|------|-------|--------|
| Gradual drift stream | `streams/gradual_drift.py` | 2 days |
| Recurring tasks stream | `streams/recurring.py` | 3 days |
| Interleaved stream | `streams/interleaved.py` | 2 days |
| Hidden boundary stream | `streams/hidden_boundary.py` | 2 days |
| Compositional recombination | `streams/compositional_recomb.py` | 3 days |

**Deliverable**: Full coverage of stream protocols from paper.

### Phase 4: Concept Ecology (Weeks 7-8)

**Priority: Medium**

| Task | Files | Effort |
|------|-------|--------|
| Concept graph data structure | `ecology/concept_graph.py` | 2 days |
| Reuse attribution scoring | `metrics/reuse_attribution.py` | 2 days |
| Relation-model accuracy | `metrics/relation_model.py` | 2 days |
| Minecraft ecology stub | `ecology/minecraft.py` | 1 day |
| Object-Affordance stub | `ecology/object_affordance.py` | 1 day |

**Deliverable**: Concept graph support for transfer analysis.

### Phase 5: CL Strategy Comparison (Weeks 9-10)

**Priority: Medium**

| Task | Files | Effort |
|------|-------|--------|
| Strategy base protocol | `strategies/base.py` | 1 day |
| Global update strategy | `strategies/global_update.py` | 2 days |
| Modular growth strategy | `strategies/modular_growth.py` | 2 days |
| Domain-conditioned strategy | `strategies/domain_conditioned.py` | 2 days |
| Same-backbone ablation framework | `strategies/ablation.py` | 3 days |

**Deliverable**: Framework for comparing CL strategies per paper's recommendations.

### Phase 6: Integration & Testing (Weeks 11-12)

**Priority: High**

| Task | Files | Effort |
|------|-------|--------|
| SequentialTrainer integration | Extend `continual/trainer.py` | 3 days |
| AGIBenchmarkRunner | `benchmarks/runner.py` | 2 days |
| Visualization extensions | `benchmarks/visualization.py` | 2 days |
| Unit tests | `tests/benchmarks/` | 3 days |
| Integration tests | `tests/benchmarks/test_integration.py` | 2 days |

**Deliverable**: Complete, tested benchmark framework.

---

## 8. Integration Points

### 8.1 SequentialTrainer Integration

Extend `fabricpc/continual/trainer.py`:

```python
class SequentialTrainer:
    def __init__(
        self,
        ...,
        benchmark_config: Optional[BenchmarkMatrixConfig] = None,
    ):
        # Existing init...

        # AGI benchmark integration
        if benchmark_config:
            from fabricpc.benchmarks.runner import AGIBenchmarkRunner
            from fabricpc.benchmarks.resources import ResourceTracker

            self.benchmark_config = benchmark_config
            self.resource_tracker = ResourceTracker(benchmark_config.resource_config)
        else:
            self.benchmark_config = None
            self.resource_tracker = None

    def train_task(self, task_data: TaskData) -> TaskRunSummary:
        # Existing training...

        # Track resources if enabled
        if self.resource_tracker:
            within_budget, util = self.resource_tracker.check_budget(self.params)
            if not within_budget:
                self._handle_budget_violation(util)
            self.resource_tracker.record(
                task_data.task_id,
                self.global_step,
                self.params,
            )

        return summary
```

### 8.2 TransWeave Integration for CL Strategies

Map existing TransWeave to paper's strategy framework:

```python
# fabricpc/benchmarks/strategies/modular_growth.py

class ModularGrowthStrategy(CLStrategy):
    """
    Modular growth CL strategy.

    Maps to FabricPC's column-based architecture with shell demotion.
    Uses TransWeave for knowledge transfer between modules.
    """

    def __init__(self, transweave_manager: TransWeaveManager):
        self.transweave = transweave_manager

    def on_task_start(self, task_id: int, params: GraphParams):
        """Prepare modules for new task."""
        # Use shell demotion to identify which neurons to update
        # Freeze center shells, allow outer shells to adapt
        pass

    def on_task_end(self, task_id: int, params: GraphParams):
        """Register learned modules for potential transfer."""
        self.transweave.register_task_end(task_id, params)
```

### 8.3 Causal Fingerprints for Domain Conditioning

Map existing causal system to paper's domain-conditioned strategy:

```python
# fabricpc/benchmarks/strategies/domain_conditioned.py

class DomainConditionedStrategy(CLStrategy):
    """
    Domain-conditioned CL strategy.

    Uses FabricPC's causal fingerprints as domain context signals.
    """

    def __init__(self, causal_bank: CausalFingerprintBank):
        self.causal_bank = causal_bank

    def get_context_signal(self, task_id: int) -> np.ndarray:
        """Get domain context for conditioning."""
        return self.causal_bank.get_fingerprint(task_id)

    def select_columns(self, context: np.ndarray) -> List[int]:
        """Select columns based on context similarity."""
        # Use causal similarity for column selection
        return self.causal_bank.get_similar_columns(context)
```

### 8.4 Callback Integration

Extend callback system for benchmark tracking:

```python
# fabricpc/utils/dashboarding/callbacks.py

def create_agi_benchmark_callback(
    resource_tracker: ResourceTracker,
    concept_graph: Optional[ConceptGraph] = None,
) -> Callable:
    """Create callback for AGI benchmark tracking."""

    def callback(epoch_idx, params, structure, config, rng_key):
        # Track resources
        resource_tracker.record(
            task_id=config.current_task_id,
            step=epoch_idx,
            params=params,
        )

        # Track concept acquisition if graph provided
        if concept_graph:
            # Check which concepts are now accessible
            pass

    return callback
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/benchmarks/test_transfer_matrix.py

def test_transfer_matrix_shape():
    """Transfer matrix has correct dimensions."""
    tasks = [mock_task(i) for i in range(5)]
    result = compute_transfer_matrix(mock_model_factory, tasks, k_values=(1, 5))
    assert result.matrix.shape == (5, 5, 2)

def test_transfer_selectivity_range():
    """Selectivity is in valid range."""
    # With concept graph, selectivity should be meaningful
    result = compute_transfer_matrix(..., relation_graph=mock_graph)
    assert -1.0 <= result.selectivity <= 1.0

def test_negative_transfer_rate():
    """NTR correctly identifies harmful transfer."""
    matrix = np.array([[0, -0.1], [0.1, 0]])
    ntr = np.mean(matrix < 0)
    assert 0 <= ntr <= 1
```

### 9.2 Integration Tests

```python
# tests/benchmarks/test_runner.py

def test_full_benchmark_run():
    """End-to-end benchmark with all metrics."""
    config = BenchmarkMatrixConfig(
        task_family="control_streams",
        stream_protocol="recurring",
        resource_track="bounded_growth",
    )
    runner = AGIBenchmarkRunner(config, mock_model_factory, mock_tasks)
    results = runner.evaluate()

    # Classical metrics
    assert 0 <= results.acc_t <= 1
    assert results.accuracy_matrix.shape[0] == len(mock_tasks)

    # Transfer metrics
    assert results.transfer_matrix is not None
    assert 0 <= results.negative_transfer_rate <= 1

    # Resource metrics
    assert len(results.resource_snapshots) > 0

def test_resource_budget_enforcement():
    """Fixed capacity budget is enforced."""
    config = BenchmarkMatrixConfig(
        resource_track="fixed_capacity",
        resource_config=ResourceTrackConfig(
            track_type="fixed_capacity",
            max_parameters=1000,
        ),
    )
    # Model that grows beyond budget should fail or adapt
```

### 9.3 Regression Tests

```python
# tests/benchmarks/test_regression.py

def test_classical_metrics_parity():
    """New implementation matches existing cl-benchmark."""
    # Run same evaluation with both systems
    old_results = run_with_cl_benchmark(...)
    new_results = run_with_agi_benchmark(...)

    assert np.isclose(old_results['acc'], new_results.acc_t)
    assert np.isclose(old_results['bwt'], new_results.bwt)
    assert np.isclose(old_results['forgetting'], new_results.forgetting)
```

---

## 10. Future Extensions

### 10.1 Minecraft Concept Ecology (T6)

From the paper's 20-task pilot stream:

| ID | Task | Surface | Latent Tags |
|----|------|---------|-------------|
| 1 | Wood bootstrap | Embodied | C1, C2, S1 |
| 2 | Ravine bridge | Embodied | C3, S3 |
| 3 | One-night shelter | Embodied | C4, C5, C6, S2 |
| ... | ... | ... | ... |
| 20 | Reskinned sheep-pen | Cross-surface | C4, C5, C8, S4, M2, M3, M4 |

**Implementation requires:**
- MineRL/Malmo integration
- Structured state extraction (blocks, entities, inventory)
- Success predicate validators (bridge_valid, shelter_valid, etc.)
- Anti-leakage via texture reskins and novel mechanics

### 10.2 Embodied Object-Affordance Ecology (T7)

From the paper's 15-task pilot stream:

| ID | Task | Surface | Latent Tags |
|----|------|---------|-------------|
| 1 | Clean a spill | Embodied | A2, A11, S4, R3 |
| 6 | Use unfamiliar brewer | Knowledge + embodied | A3, A10, Q6 |
| 7 | Delayed brewer recurrence | Embodied (no corpus) | A3, A10, Q6, M3 |
| ... | ... | ... | ... |

**Implementation requires:**
- AI2-THOR/ProcTHOR/OmniGibson integration
- Object-affordance graph with states (hot/cold, wet/dry, etc.)
- Knowledge corpus with graded access (full, degraded, none)
- Fictional object generation for anti-leakage

### 10.3 Language/Code Streams (T5)

Future extension for LLM-based continual learning:
- Domain adaptation streams
- Code maintenance/repair tasks
- Tool use evolution

### 10.4 Same-Backbone Ablation Protocol

From the paper:

> "Choose a common backbone. Run the same benchmark stream under three CL strategies:
> (a) Full update: all weights updated continually
> (b) Frozen base + modular growth
> (c) Domain-conditioned: shared body with context signal"

This enables fair comparison of CL strategies on the same architecture.

---

## Appendix A: Paper Metrics Summary

### Classical CL Panel

```
ACC_T = (1/T) Σ_{i=1}^T a_{T,i}

BWT = (1/(T-1)) Σ_{i=1}^{T-1} (a_{T,i} - a_{i,i})

FWT = (1/(T-1)) Σ_{i=2}^T (a_{i-1,i} - b_i)

Forgetting_i = max_{t∈{i,...,T}} a_{t,i} - a_{T,i}
```

### Transfer-Centred Additions

```
Gain@k(j) = p_CL_j(k) - p_scratch_j(k)

ThresholdGain_j(τ) = n_scratch_j(τ) - n_CL_j(τ)

T_ij(k) = p_j(k | history includes task i) - p_scratch_j(k)

Selectivity(k) = (1/|R|) Σ_{(i,j)∈R} T_ij(k) - (1/|U|) Σ_{(i,j)∈U} T_ij(k)

NTR(k) = (1/|P|) Σ_{(i,j)∈P} 1[T_ij(k) < -ε]
```

---

## Appendix B: Recommended Leaderboard Record

From the paper's Appendix B:

```yaml
run_id: ...
system_name: ...
cl_strategy_type: modular_growth  # global_update | modular_growth | domain_conditioned | symbolic_cl | hybrid

track_name: ...
resource_track: bounded_growth
stream_ordering: mixed
with_task_ids: true
with_corpus_access: none
same_backbone_ablation: false

metrics:
  online_auc: ...
  final_retention: ...
  bwt: ...
  fwt: ...
  forgetting: ...
  intransigence: ...
  gain_at_k: ...
  threshold_gain: ...
  recurrence_gain_short: ...
  recurrence_gain_medium: ...
  recurrence_gain_long: ...
  cross_surface_transfer: ...
  selectivity: ...
  negative_transfer_rate: ...
  relation_model_accuracy: ...
  reuse_prediction_f1: ...
  reuse_ablation_auc: ...

resources:
  params_count: ...
  model_memory_mb: ...
  external_memory_mb: ...
  stored_samples: ...
  train_flops: ...
  infer_flops: ...
  queries_used: ...

artifacts:
  transfer_matrix_path: ...
  relation_curve_path: ...
  failure_case_sheet_path: ...
```

---

## Appendix C: FabricPC File Reference

### Critical Files to Modify

| File | Modification |
|------|--------------|
| `fabricpc/continual/trainer.py` | Add `benchmark_config` parameter, resource tracking hooks |
| `fabricpc/continual/config.py` | Add `BenchmarkMatrixConfig` import |
| `cl-benchmark/evaluation/runner.py` | Extend with AGI metrics |

### Critical Files to Create

| File | Purpose |
|------|---------|
| `fabricpc/benchmarks/__init__.py` | Module exports |
| `fabricpc/benchmarks/config.py` | 4-axis configuration |
| `fabricpc/benchmarks/runner.py` | AGIBenchmarkRunner |
| `fabricpc/benchmarks/streams/base.py` | StreamProtocol base |
| `fabricpc/benchmarks/metrics/transfer_matrix.py` | Pairwise transfer |
| `fabricpc/benchmarks/resources/tracker.py` | ResourceTracker |
| `fabricpc/benchmarks/ecology/concept_graph.py` | ConceptGraph |

---

*Document generated based on "Toward AGI-friendly Benchmarks for Continual + Transfer Learning" (Goertzel, April 2026) and analysis of FabricPC codebase.*
