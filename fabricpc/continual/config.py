"""
Configuration dataclasses for FabricPC Continual Learning.

Ported from mnist_audit_guided_generality_v18_3 notebook with clean,
flattened structure optimized for FabricPC's JAX-based architecture.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any, Dict


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs_per_task: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.1
    grad_clip_norm: float = (
        1.0  # Gradient clipping for stability (especially attention)
    )

    # Device selection
    device: str = "cuda"  # "cuda", "cpu", or specific like "cuda:0"

    # Data loading
    num_workers: int = 0
    pin_memory: bool = True

    # Fast development mode
    fast_dev_max_train_batches: Optional[int] = None
    fast_dev_max_test_batches: Optional[int] = None

    # Training mode: "pc" (predictive coding), "backprop", or "hybrid"
    training_mode: str = "hybrid"


@dataclass
class PatchConfig:
    """Patch embedding configuration."""

    patch_size: int = 4  # Patch size for image tokenization
    patch_embed_dim: int = 64  # Embedding dimension per patch
    use_positional_encoding: bool = True


@dataclass
class ColumnConfig:
    """Memory column configuration."""

    num_columns: int = 22  # Total number of memory columns (2 shared + 5 tasks × 4)
    memory_dim: int = 64  # Dimension of each column's memory
    shared_columns: int = 2  # Number of always-active shared columns
    topk_nonshared: int = 4  # Number of non-shared columns to activate per task
    aggregator_dim: int = (
        128  # Output dimension of aggregator layer (partitioned by task)
    )
    partition_aggregator: bool = (
        False  # Disabled: partitioning prevents later tasks from learning
    )
    use_attention_aggregator: bool = (
        False  # Use ComposerNode (attention) instead of Linear for aggregation
    )
    use_partitioned_aggregator: bool = (
        False  # Use PartitionedAggregator with true architectural isolation
    )
    partitioned_shared_dim: int = 32  # Output neurons for shared pathway
    partitioned_task_dim: int = 64  # Output neurons per task pathway
    attention_num_heads: int = 4  # Number of attention heads if using ComposerNode
    attention_num_layers: int = 1  # Number of attention layers if using ComposerNode


@dataclass
class ShellConfig:
    """Shell (hierarchical layer) configuration."""

    shell_sizes: Tuple[int, ...] = (8, 16, 8)  # Columns per shell level
    shell_activation: str = "relu"  # Activation function


@dataclass
class SupportConfig:
    """Support selection and replay bank configuration."""

    # Basic support selection
    topk_nonshared: int = 4
    target_nonshared_overlap: float = 0.50

    # Selector policy
    selector_policy_min_examples: int = 72
    selector_policy_target_examples: int = 360
    selector_policy_scale_max: float = 0.85
    selector_policy_floor_scale: float = 0.03
    selector_policy_refine_threshold: float = 0.008
    selector_policy_refine_topk: int = 8

    # Teacher model
    teacher_scale_max: float = 1.0
    teacher_min_examples: int = 4
    teacher_target_examples: int = 36
    selector_teacher_step: float = 0.18
    selector_teacher_override_margin: float = 0.015

    # Replay bank
    replay_bank_support_enable: bool = True
    replay_bank_support_oldtask_only: bool = False
    replay_bank_support_context_neighbors: int = 6
    replay_bank_support_topk_per_context: int = 4
    replay_bank_support_row_neighbors: int = 15
    replay_bank_support_refine_threshold: float = 0.0
    replay_bank_support_min_confidence: float = 0.0
    replay_bank_support_candidate_limit: int = 128
    replay_bank_support_force_top_global: int = 12
    replay_bank_support_old_bonus_scale: float = 0.30
    replay_bank_support_runtime_bonus_scale: float = 0.05
    replay_bank_support_exact_audit_topk: int = 8
    replay_bank_support_apply_if_exact_gain: float = 0.010

    # Causal mixing
    causal_max_effective_scale: float = 0.0
    causal_mix_max: float = 0.30
    causal_fallback_weight: float = 0.20

    # Causal Fingerprint Bank
    causal_similarity_conf_target: float = 8.0
    causal_similarity_blend_max: float = 0.75
    causal_similarity_floor: float = 0.05

    # Causal Contribution Predictor
    causal_min_examples: int = 36
    causal_target_examples: int = 90
    causal_ridge_lambda: float = 0.50
    causal_max_abs_target: float = 0.75
    causal_target_scale: float = 0.10
    causal_feature_dim: int = 21

    # Causal Trust Controller
    causal_agreement_target: float = 0.35
    causal_trend_tau: float = 0.16
    structural_trust_floor: float = 0.10
    structural_trust_target: float = 0.35

    # SB Distribution Clarity
    sb_enable: bool = True
    sb_kurtosis_threshold: float = 0.65
    sb_multimodal_threshold: float = 0.18
    sb_alpha_scale: float = 0.35
    sb_alpha_max: float = 0.25
    sb_sinkhorn_eps: float = 0.35
    sb_sinkhorn_iters: int = 8
    sb_clarity_weight: float = 0.0025
    sb_trust_cap: float = 0.35
    sb_gaussian_ref_scale: float = 1.0
    sb_cost_mode: str = "huber"

    # Routing Bonus
    route_enable: bool = True
    route_cert_mix_scale: float = 0.080
    route_diversity_scale: float = 0.060
    route_novelty_scale: float = 0.030
    route_demotion_scale: float = 0.050
    route_stability_scale: float = 0.040
    route_near_tie_gain_margin: float = 0.015
    route_exact_gain_floor: float = 0.004
    route_trust_cap: float = 0.35

    # Similarity/redundancy
    recent_support_penalty: float = 0.10
    similarity_redundancy_scale: float = 0.65

    # Hybrid selector
    hybrid_positive_margin: float = 0.003
    hybrid_knn_min_neighbors: int = 5
    hybrid_knn_max_neighbors: int = 15
    hybrid_distance_quantile: float = 0.60
    hybrid_distance_temp_floor: float = 0.15
    hybrid_knn_gate_floor: float = 0.05
    hybrid_rank_c: float = 1.25
    hybrid_rank_scale_floor: float = 0.05

    # Persistence
    selector_persistent_filename: str = "selector_persistent_state.pt"
    support_replay_filename: str = "support_selector_replay.pt"
    selector_policy_filename: str = "selector_policy.pt"
    selector_teacher_filename: str = "selector_teacher.pt"

    # Experience replay - DISABLED (removed as it undermines continual learning goals)
    use_replay: bool = False  # Experience replay disabled
    replay_ratio: float = 0.0  # Unused
    replay_buffer_size_per_task: int = 0  # Unused
    replay_buffer_total_size: int = 0  # Unused


@dataclass
class TypingConfig:
    """Typing/demotion configuration."""

    enable_demotion: bool = True
    start_after_steps: int = 100
    outer_every_steps: int = 10

    # Demotion parameters
    demotion_swap_audit_topk: int = 8
    demotion_max_accept_per_boundary: int = 1
    demotion_max_accept_per_column: int = 1

    # Replay bank demotion
    replay_bank_demotion_enable: bool = True
    replay_bank_demotion_neighbors: int = 15
    replay_bank_demotion_prefilter_topk: int = 16
    replay_bank_demotion_min_confidence: float = 0.0
    replay_bank_demotion_min_safe_prob: float = 0.15
    replay_bank_demotion_priority_mix: float = 0.20
    replay_bank_demotion_old_bonus_scale: float = 0.30

    # Posterior
    logbf_scale: float = 1.0
    posterior_momentum: float = 0.9
    target_tier_occupancy: Tuple[float, ...] = (0.25, 0.50, 0.25)

    # Audit
    support_swap_audit_topk: int = 24


@dataclass
class ComposerConfig:
    """Hierarchical composer attention configuration."""

    enable: bool = True
    hidden_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    dropout: float = 0.1

    # Scaling
    scale: float = 0.42
    scale_max: float = 0.90

    # Gating
    gate_temp: float = 0.52
    gate_entropy_ceiling_frac: float = 0.52
    gate_entropy_ceiling_weight: float = 0.009
    gate_dev_floor: float = 0.14
    gate_dev_weight: float = 0.005

    # Prior
    prior_logit_scale: float = 0.68
    prior_mix_scale: float = 0.16
    prior_kl_weight: float = 0.0008

    # Residual
    residual_gate_scale: float = 2.35
    query_score_scale: float = 1.0


@dataclass
class HierarchyConfig:
    """Hierarchical structure configuration."""

    enable: bool = True
    hidden_dim: int = 64

    # Loss weights
    mid_loss_weight: float = 0.06
    global_loss_weight: float = 0.03
    parent_child_loss_weight: float = 0.04


@dataclass
class CloudConfig:
    """Cloud augmentation configuration."""

    enable: bool = True
    max_examples: int = 16

    # Augmentation parameters
    shift_pixels: int = 1
    noise_std: float = 0.06
    mask_patch: int = 7
    contrast_scale: float = 0.90

    # Support cloud
    support_topk: int = 6
    support_min_beneficial_frac: float = 0.60
    support_variants: Tuple[str, ...] = (
        "identity",
        "shift_left",
        "shift_right",
        "noise",
        "mask",
        "contrast",
    )
    composer_variants: Tuple[str, ...] = (
        "identity",
        "shift_left",
        "shift_right",
        "noise",
        "mask",
        "contrast",
    )


@dataclass
class TransWeaveConfig:
    """TransWeave transport configuration."""

    enable: bool = True
    apply_on_task_ge: int = 1  # Only apply after this many tasks
    use_last_k_tasks: int = 3

    # Source weights
    source_local_weight: float = 0.50
    source_block_weight: float = 0.30
    source_global_weight: float = 0.20

    # Sinkhorn
    sinkhorn_eps: float = 0.30
    sinkhorn_iters: int = 20

    # Mixing
    identity_bonus: float = 0.12
    transport_diag_mix: float = 0.35
    prior_mix_scale: float = 0.55


@dataclass
class ComposerTransWeaveConfig:
    """
    Configuration for Composer-level TransWeave transfer learning.

    Transfers learned composition patterns (attention, projections) across tasks
    via Sinkhorn optimal transport. Reference: V20.2b Section 4.6, 7.7.
    """

    enable: bool = True

    # Transport parameters
    sinkhorn_eps: float = 0.25
    sinkhorn_iters: int = 15
    identity_bonus: float = 0.15
    transport_diag_mix: float = 0.30

    # Source weighting
    use_last_k_tasks: int = 3
    recency_decay: float = 0.7  # Exponential decay for older tasks

    # Transfer blending
    transfer_strength: float = 0.3  # How much to blend transferred vs original
    warmup_tasks: int = 1  # Don't apply transfer until this many tasks seen

    # Attention transfer
    transfer_attention_weights: bool = True
    transfer_query_keys: bool = True
    transfer_value_projections: bool = True

    # Regularization
    orthogonality_weight: float = 0.01  # Encourage diverse attention patterns
    sparsity_weight: float = 0.005  # Encourage sparse transport


@dataclass
class ShellDemotionTransWeaveConfig:
    """
    Configuration for Within-column Shell Demotion TransWeave.

    Implements radial shell semantics from V18/V20 (Section 5.2):
    - Protected center (shell 0): Most stable, rarely demoted
    - Stable inner tiers (shell 1): Moderately stable
    - Disposable outer tiers (shell 2): Task-local, frequently recycled

    Uses Sinkhorn transport to guide neuron transitions between shells.
    """

    enable: bool = True

    # Transport parameters
    sinkhorn_eps: float = 0.25  # Higher for more diffuse transport
    sinkhorn_iters: int = 25  # More iterations for convergence
    identity_bonus: float = 0.02  # Low to allow off-diagonal transport

    # Shell structure
    num_shells: int = 3  # [protected_center, stable_inner, disposable_outer]
    shell_sizes: Tuple[int, ...] = (8, 16, 8)  # Neurons per shell

    # Demotion thresholds
    # Lower thresholds to detect transport with larger shell configs
    demotion_threshold: float = 0.10  # Transport mass threshold for demotion
    promotion_threshold: float = 0.25  # Transport mass threshold for promotion
    stability_bonus: float = 0.08  # Bonus for keeping neurons in place

    # Cross-task patterns
    use_last_k_tasks: int = 2
    activity_ema_decay: float = 0.9  # EMA for tracking neuron activity

    # Safety constraints
    max_demotions_per_step: int = 2  # Max neurons demoted per shell per step
    min_shell_occupancy: float = 0.25  # Minimum fraction of shell filled
    protected_center_fraction: float = 0.5  # Fraction of center never demoted


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""

    enable_periodic: bool = True
    every_train_forwards: int = 160
    min_seconds: float = 180.0
    copy_persistent_on_partial: bool = True


@dataclass
class AuditConfig:
    """Audit and evaluation configuration."""

    min_seen_tasks: int = 1
    audit_batches_per_task: int = 4

    # Tier quotas
    per_micro_tier_quotas: Tuple[int, ...] = (2, 4, 4)
    smoke_per_micro_tier_quotas: Tuple[int, ...] = (0, 2, 2)

    # Topk settings
    support_swap_audit_topk: int = 24
    support_swap_exact_topk: int = 24
    support_bank_exact_audit_topk: int = 24
    support_cloud_audit_topk: int = 12

    # Support swap audit for causal learning
    support_swap_audit_enable: bool = True
    support_swap_audit_max_swaps: int = 45  # Max column swaps per audit
    support_audit_current_weight: float = 1.0  # Weight for current task loss
    support_audit_old_weight: float = 1.0  # Weight for old task loss
    support_audit_max_batches: int = 4  # Max batches to evaluate per swap


@dataclass
class TransitionAutotuneConfig:
    """Task-start autotuning for shell promotion/demotion thresholds."""

    enable: bool = False
    rollout_batches: int = 2
    max_columns_to_check: int = 10

    demotion_threshold_candidates: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    promotion_threshold_candidates: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.45)
    max_demotions_per_step_candidates: Tuple[int, ...] = (1, 2)

    target_demotions_per_column: float = 0.35
    target_promotions_per_column: float = 0.20
    max_demotions_per_column: float = 0.80
    max_promotions_per_column: float = 0.60

    target_weight: float = 1.0
    overfire_penalty: float = 2.0
    transport_cost_weight: float = 0.05
    entropy_weight: float = 0.02
    config_distance_weight: float = 0.10


@dataclass
class PredictorConfig:
    """Predictor model configuration."""

    min_examples: int = 16
    target_examples: int = 128
    train_steps: int = 64


@dataclass
class PerWeightCausalConfig:
    """
    Configuration for per-weight causal coding.

    Enables fine-grained causal control where individual weight updates
    are modulated based on their gradient distribution characteristics.
    Standard learning is used for weights with Gaussian-like gradients,
    while Sinkhorn-based (SB) learning is used for non-Gaussian weights.
    """

    # Enable per-weight causal coding
    enable: bool = True

    # Gradient tracking
    gradient_history_size: int = 32  # Number of recent gradients to track per weight
    min_history_for_detection: int = 8  # Minimum history before computing kurtosis

    # Non-Gaussianity thresholds
    kurtosis_threshold: float = 2.0  # Excess kurtosis threshold for non-Gaussianity
    multimodal_threshold: float = 0.5  # Multimodal gap threshold
    combined_threshold: float = 1.5  # Combined non-Gaussianity score threshold

    # Sinkhorn parameters for SB update
    sb_sinkhorn_eps: float = 0.1  # Regularization for Sinkhorn
    sb_sinkhorn_iters: int = 5  # Sinkhorn iterations
    sb_correction_strength: float = 0.3  # How much to blend SB correction

    # Adaptive blending
    blend_mode: str = "soft"  # "hard" (binary) or "soft" (smooth transition)
    soft_blend_scale: float = 1.0  # Scale for soft sigmoid transition

    # Per-layer control
    skip_bias_weights: bool = True  # Skip bias terms (typically more Gaussian)
    skip_small_weights: bool = True  # Skip weights below threshold
    small_weight_threshold: float = 1e-6  # Threshold for small weights

    # Statistics tracking
    track_statistics: bool = True  # Track per-weight statistics for debugging
    stats_update_every: int = 10  # Update summary stats every N steps


@dataclass
class EWCConfig:
    """
    Elastic Weight Consolidation (EWC) configuration.

    EWC prevents catastrophic forgetting by penalizing changes to weights
    that are important for previous tasks, measured via Fisher Information.
    """

    enable: bool = False  # Enable EWC regularization
    lambda_ewc: float = 5000.0  # EWC regularization strength
    fisher_samples: int = 200  # Number of samples for Fisher estimation
    online: bool = True  # Use online EWC (running Fisher) vs offline (per-task Fisher)
    gamma: float = 0.95  # Decay for online Fisher (importance of older tasks)
    normalize_fisher: bool = True  # Normalize Fisher by max value for stability


@dataclass
class ExperimentConfig:
    """
    Master configuration for Split-MNIST continual learning experiment.

    Aggregates all sub-configurations with sensible defaults.
    Use make_config() factory for quick_smoke vs full training setups.
    """

    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    shells: ShellConfig = field(default_factory=ShellConfig)
    support: SupportConfig = field(default_factory=SupportConfig)
    typing: TypingConfig = field(default_factory=TypingConfig)
    composer: ComposerConfig = field(default_factory=ComposerConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    transweave: TransWeaveConfig = field(default_factory=TransWeaveConfig)
    composer_transweave: ComposerTransWeaveConfig = field(
        default_factory=ComposerTransWeaveConfig
    )
    shell_demotion_transweave: ShellDemotionTransWeaveConfig = field(
        default_factory=ShellDemotionTransWeaveConfig
    )
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    transition_autotune: TransitionAutotuneConfig = field(
        default_factory=TransitionAutotuneConfig
    )
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    per_weight_causal: PerWeightCausalConfig = field(
        default_factory=PerWeightCausalConfig
    )
    ewc: EWCConfig = field(default_factory=EWCConfig)

    # Task configuration
    num_tasks: int = 5
    task_pairs: Tuple[Tuple[int, ...], ...] = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    num_output_classes: int = 10  # 10 for MNIST/CIFAR-10, 100 for CIFAR-100

    # Random seed
    seed: int = 0


def make_config(quick_smoke: bool = False) -> ExperimentConfig:
    """
    Factory function to create experiment configuration.

    Args:
        quick_smoke: If True, create minimal config for fast testing.
                    If False, create full training config.

    Returns:
        ExperimentConfig with appropriate settings.
    """
    cfg = ExperimentConfig()

    if quick_smoke:
        # Minimal configuration for smoke testing
        cfg.training.epochs_per_task = 1
        cfg.training.batch_size = 32
        cfg.training.fast_dev_max_train_batches = 2
        cfg.training.fast_dev_max_test_batches = 2

        cfg.patch.patch_embed_dim = 8
        cfg.columns.memory_dim = 4
        # 5 tasks x 2 non-shared = 10 needed + 2 shared = 12 total for no column reuse
        cfg.columns.num_columns = 12
        cfg.columns.shared_columns = 2
        cfg.columns.topk_nonshared = 2
        cfg.columns.aggregator_dim = 128
        cfg.shells.shell_sizes = (2, 4, 2)

        cfg.typing.start_after_steps = 0
        cfg.typing.outer_every_steps = 1

        cfg.predictor.min_examples = 4
        cfg.predictor.target_examples = 12
        cfg.predictor.train_steps = 12

        cfg.audit.min_seen_tasks = 1
        cfg.audit.audit_batches_per_task = 1
        cfg.audit.smoke_per_micro_tier_quotas = (0, 1, 1)
        cfg.audit.per_micro_tier_quotas = (1, 2, 2)
        cfg.audit.support_swap_audit_max_swaps = 12
        cfg.audit.support_audit_max_batches = 2

        cfg.composer.hidden_dim = 16
        cfg.composer.num_heads = 2
        cfg.composer.num_layers = 1
        cfg.composer.scale = 0.10

        cfg.hierarchy.hidden_dim = 16
        cfg.hierarchy.mid_loss_weight = 0.02
        cfg.hierarchy.global_loss_weight = 0.01
        cfg.hierarchy.parent_child_loss_weight = 0.02

        cfg.cloud.max_examples = 4
        cfg.cloud.support_topk = 4
        cfg.cloud.support_variants = ("identity", "shift_left", "noise")
        cfg.cloud.composer_variants = ("identity", "shift_left", "noise")

        cfg.transweave.sinkhorn_iters = 6

        # Composer TransWeave for smoke tests
        cfg.composer_transweave.enable = True
        cfg.composer_transweave.sinkhorn_iters = 5
        cfg.composer_transweave.use_last_k_tasks = 2
        cfg.composer_transweave.transfer_strength = 0.2
        cfg.composer_transweave.warmup_tasks = 1

        # Shell Demotion TransWeave for smoke tests
        cfg.shell_demotion_transweave.enable = True
        cfg.shell_demotion_transweave.sinkhorn_iters = 4
        cfg.shell_demotion_transweave.shell_sizes = (2, 4, 2)
        cfg.shell_demotion_transweave.max_demotions_per_step = 1
        cfg.shell_demotion_transweave.use_last_k_tasks = 1

        # Task-start transition autotune preset for smoke tests
        cfg.transition_autotune.enable = False
        cfg.transition_autotune.rollout_batches = 1
        cfg.transition_autotune.max_columns_to_check = 4
        cfg.transition_autotune.demotion_threshold_candidates = (
            0.05,
            0.10,
            0.15,
        )
        cfg.transition_autotune.promotion_threshold_candidates = (
            0.20,
            0.30,
            0.40,
        )
        cfg.transition_autotune.max_demotions_per_step_candidates = (1,)
        cfg.transition_autotune.target_demotions_per_column = 0.30
        cfg.transition_autotune.target_promotions_per_column = 0.15

        cfg.checkpoint.every_train_forwards = 8
        cfg.checkpoint.min_seconds = 0.0

        # Causal config for smoke tests
        cfg.support.causal_min_examples = 8
        cfg.support.causal_target_examples = 24
        cfg.support.sb_sinkhorn_iters = 4

        # Per-weight causal config for smoke tests
        cfg.per_weight_causal.enable = True
        cfg.per_weight_causal.gradient_history_size = 8
        cfg.per_weight_causal.min_history_for_detection = 4
        cfg.per_weight_causal.sb_sinkhorn_iters = 3
        cfg.per_weight_causal.stats_update_every = 2
    else:
        # Full training configuration
        cfg.training.epochs_per_task = 5
        cfg.training.batch_size = 256

        cfg.support.replay_bank_support_context_neighbors = 6
        cfg.support.replay_bank_support_topk_per_context = 4
        cfg.support.replay_bank_support_row_neighbors = 15
        cfg.support.replay_bank_support_candidate_limit = 128
        cfg.support.replay_bank_support_force_top_global = 12
        cfg.support.replay_bank_support_old_bonus_scale = 0.30
        cfg.support.replay_bank_support_runtime_bonus_scale = 0.05
        cfg.support.replay_bank_support_exact_audit_topk = 8
        cfg.support.replay_bank_support_apply_if_exact_gain = 0.010

        cfg.typing.replay_bank_demotion_neighbors = 15
        cfg.typing.replay_bank_demotion_prefilter_topk = 16
        cfg.typing.replay_bank_demotion_min_safe_prob = 0.15
        cfg.typing.replay_bank_demotion_priority_mix = 0.20
        cfg.typing.replay_bank_demotion_old_bonus_scale = 0.30

        cfg.hierarchy.mid_loss_weight = 0.06
        cfg.hierarchy.global_loss_weight = 0.03
        cfg.hierarchy.parent_child_loss_weight = 0.04

        cfg.composer.hidden_dim = 64
        cfg.composer.scale = 0.42
        cfg.composer.prior_kl_weight = 0.0008

        cfg.cloud.max_examples = 16
        cfg.cloud.support_topk = 6

        cfg.transweave.sinkhorn_iters = 20
        cfg.transweave.source_local_weight = 0.50
        cfg.transweave.source_block_weight = 0.30
        cfg.transweave.source_global_weight = 0.20

        # Composer TransWeave for full training
        cfg.composer_transweave.enable = True
        cfg.composer_transweave.sinkhorn_iters = 15
        cfg.composer_transweave.sinkhorn_eps = 0.25
        cfg.composer_transweave.use_last_k_tasks = 3
        cfg.composer_transweave.transfer_strength = 0.3
        cfg.composer_transweave.warmup_tasks = 1
        cfg.composer_transweave.identity_bonus = 0.15
        cfg.composer_transweave.transport_diag_mix = 0.30
        cfg.composer_transweave.recency_decay = 0.7

        # Shell Demotion TransWeave for full training
        cfg.shell_demotion_transweave.enable = True
        cfg.shell_demotion_transweave.sinkhorn_iters = 12
        cfg.shell_demotion_transweave.sinkhorn_eps = 0.20
        cfg.shell_demotion_transweave.shell_sizes = (8, 16, 8)
        cfg.shell_demotion_transweave.num_shells = 3
        cfg.shell_demotion_transweave.max_demotions_per_step = 2
        cfg.shell_demotion_transweave.use_last_k_tasks = 2
        # Thresholds tuned for larger shell configs (32 neurons total)
        # Lower thresholds detect transport with more diffuse distributions
        cfg.shell_demotion_transweave.demotion_threshold = 0.10
        cfg.shell_demotion_transweave.promotion_threshold = 0.25
        cfg.shell_demotion_transweave.protected_center_fraction = 0.5

        # Task-start transition autotune preset for full training
        cfg.transition_autotune.enable = False
        cfg.transition_autotune.rollout_batches = 2
        cfg.transition_autotune.max_columns_to_check = 10
        cfg.transition_autotune.demotion_threshold_candidates = (
            0.05,
            0.10,
            0.15,
            0.20,
        )
        cfg.transition_autotune.promotion_threshold_candidates = (
            0.15,
            0.25,
            0.35,
            0.45,
        )
        cfg.transition_autotune.max_demotions_per_step_candidates = (1, 2)
        cfg.transition_autotune.target_demotions_per_column = 0.35
        cfg.transition_autotune.target_promotions_per_column = 0.20
        cfg.transition_autotune.max_demotions_per_column = 0.80
        cfg.transition_autotune.max_promotions_per_column = 0.60

        cfg.checkpoint.every_train_forwards = 160
        cfg.checkpoint.min_seconds = 180.0

        # Selector tuning
        cfg.support.selector_teacher_step = 0.18
        cfg.support.teacher_scale_max = 1.00
        cfg.support.teacher_min_examples = 4
        cfg.support.teacher_target_examples = 36
        cfg.support.selector_teacher_override_margin = 0.015
        cfg.support.selector_policy_floor_scale = 0.08
        cfg.support.selector_policy_scale_max = 1.00
        cfg.support.selector_policy_refine_threshold = 0.0025
        cfg.support.selector_policy_refine_topk = 12
        cfg.support.recent_support_penalty = 0.10
        cfg.support.similarity_redundancy_scale = 0.65
        cfg.support.causal_fallback_weight = 0.20
        cfg.support.causal_mix_max = 0.30

        # Per-weight causal config for full training
        cfg.per_weight_causal.enable = True
        cfg.per_weight_causal.gradient_history_size = 32
        cfg.per_weight_causal.min_history_for_detection = 8
        cfg.per_weight_causal.kurtosis_threshold = 2.0
        cfg.per_weight_causal.combined_threshold = 1.5
        cfg.per_weight_causal.sb_correction_strength = 0.3
        cfg.per_weight_causal.blend_mode = "soft"
        cfg.per_weight_causal.track_statistics = True

    return cfg
