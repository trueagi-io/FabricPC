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
    sinkhorn_transport,
    cosine_cost_matrix,
    euclidean_cost_matrix,
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
    "sinkhorn_transport",
    "cosine_cost_matrix",
    "euclidean_cost_matrix",
]
