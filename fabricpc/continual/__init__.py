"""
FabricPC Continual Learning Module

This module provides tools for continual/sequential learning on Split-MNIST
and similar tasks, ported from the mnist_audit_guided_generality notebook.

Key components:
- SplitMnistLoader: Data loading for sequential digit pair tasks
- SequentialTrainer: Task-by-task training with support selection
- Custom nodes for hierarchical composition architecture
- Support and demotion banks for replay-based continual learning
"""

from fabricpc.continual.config import (
    ExperimentConfig,
    TrainingConfig,
    PatchConfig,
    ColumnConfig,
    ShellConfig,
    SupportConfig,
    TypingConfig,
    ComposerConfig,
    HierarchyConfig,
    CloudConfig,
    TransWeaveConfig,
    ComposerTransWeaveConfig,
    ShellDemotionTransWeaveConfig,
    CheckpointConfig,
    AuditConfig,
    PredictorConfig,
    PerWeightCausalConfig,
    make_config,
)

from fabricpc.continual.data import (
    SplitMnistLoader,
    TaskData,
    build_split_mnist_loaders,
)

from fabricpc.continual.trainer import SequentialTrainer, TaskRunSummary

from fabricpc.continual.causal import (
    CausalFingerprintBank,
    CausalContributionPredictor,
    CausalSelectorTrustController,
    CausalSupportFeatureBuilder,
    SBClarityPenaltyResult,
    RoutingCertificates,
    RoutingBonusResult,
    compute_sb_clarity_penalty,
    compute_routing_bonus,
    weighted_corr,
    weighted_mae,
)

from fabricpc.continual.weight_causal import (
    PerWeightCausalLearner,
    WeightGradientTracker,
    PerWeightNonGaussianityDetector,
    AdaptiveWeightUpdater,
    PerWeightNonGaussianityResult,
    AdaptiveWeightUpdateResult,
    compute_weight_excess_kurtosis,
    compute_weight_multimodal_gap,
    compute_non_gaussianity_score,
    compute_sinkhorn_weight_correction,
)

from fabricpc.continual.transweave import (
    ComposerTransWeave,
    ShellDemotionTransWeave,
    TransWeaveManager,
    ComposerRepresentation,
    ComposerTransferResult,
    ShellState,
    ShellDemotionResult,
)

from fabricpc.continual.optimal_transport import (
    sinkhorn_transport,
    sinkhorn_1d_correction,
    cosine_cost_matrix,
    euclidean_cost_matrix,
    erfinv_approx,
)

from fabricpc.continual.native_nodes import (
    CausalLinear,
    TransWeaveLinear,
    CausalTransWeaveLinear,
    CausalGradientRegistry,
    TransWeaveRegistry,
    apply_causal_to_gradients,
    register_task_end_for_nodes,
    get_transferred_params,
)

from fabricpc.continual.parity import (
    ParityProfile,
    ParityMetrics,
    MetricTolerance,
    MetricCheck,
    ParityComparison,
    PROFILES,
    DEFAULT_TOLERANCES,
    make_parity_config,
    create_parity_network_structure,
    compute_parity_metrics,
    run_parity_profile,
    load_parity_baselines,
    save_parity_baselines,
    compare_against_baseline,
)

__all__ = [
    # Config
    "ExperimentConfig",
    "TrainingConfig",
    "PatchConfig",
    "ColumnConfig",
    "ShellConfig",
    "SupportConfig",
    "TypingConfig",
    "ComposerConfig",
    "HierarchyConfig",
    "CloudConfig",
    "TransWeaveConfig",
    "ComposerTransWeaveConfig",
    "ShellDemotionTransWeaveConfig",
    "CheckpointConfig",
    "AuditConfig",
    "PredictorConfig",
    "PerWeightCausalConfig",
    "make_config",
    # Data
    "SplitMnistLoader",
    "TaskData",
    "build_split_mnist_loaders",
    # Training
    "SequentialTrainer",
    "TaskRunSummary",
    # Causal (column-level)
    "CausalFingerprintBank",
    "CausalContributionPredictor",
    "CausalSelectorTrustController",
    "CausalSupportFeatureBuilder",
    "SBClarityPenaltyResult",
    "RoutingCertificates",
    "RoutingBonusResult",
    "compute_sb_clarity_penalty",
    "compute_routing_bonus",
    "weighted_corr",
    "weighted_mae",
    # Per-weight causal
    "PerWeightCausalLearner",
    "WeightGradientTracker",
    "PerWeightNonGaussianityDetector",
    "AdaptiveWeightUpdater",
    "PerWeightNonGaussianityResult",
    "AdaptiveWeightUpdateResult",
    "compute_weight_excess_kurtosis",
    "compute_weight_multimodal_gap",
    "compute_non_gaussianity_score",
    "compute_sinkhorn_weight_correction",
    # TransWeave (multi-level transfer learning)
    "ComposerTransWeave",
    "ShellDemotionTransWeave",
    "TransWeaveManager",
    "ComposerRepresentation",
    "ComposerTransferResult",
    "ShellState",
    "ShellDemotionResult",
    # Optimal Transport (shared utilities)
    "sinkhorn_transport",
    "sinkhorn_1d_correction",
    "cosine_cost_matrix",
    "euclidean_cost_matrix",
    "erfinv_approx",
    # Native FabricPC Nodes (embedded CL)
    "CausalLinear",
    "TransWeaveLinear",
    "CausalTransWeaveLinear",
    "CausalGradientRegistry",
    "TransWeaveRegistry",
    "apply_causal_to_gradients",
    "register_task_end_for_nodes",
    "get_transferred_params",
    # Notebook-parity regression harness
    "ParityProfile",
    "ParityMetrics",
    "MetricTolerance",
    "MetricCheck",
    "ParityComparison",
    "PROFILES",
    "DEFAULT_TOLERANCES",
    "make_parity_config",
    "create_parity_network_structure",
    "compute_parity_metrics",
    "run_parity_profile",
    "load_parity_baselines",
    "save_parity_baselines",
    "compare_against_baseline",
]
