"""
Configuration dataclasses for FabricPC Continual Learning.

Ported from mnist_audit_guided_generality_v18_3 notebook with clean,
flattened structure optimized for FabricPC's JAX-based architecture.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs_per_task: int = 5
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.1

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

    num_columns: int = 32  # Total number of memory columns
    memory_dim: int = 64  # Dimension of each column's memory
    shared_columns: int = 8  # Number of always-active shared columns
    topk_nonshared: int = 4  # Number of non-shared columns to activate per task


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

    # Experience replay for preventing catastrophic forgetting
    use_replay: bool = True  # Enable experience replay during training
    replay_ratio: float = 0.5  # Ratio of replay samples to current task samples
    replay_buffer_size_per_task: int = 500  # Max samples stored per task
    replay_buffer_total_size: int = 5000  # Max total samples in buffer


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


@dataclass
class PredictorConfig:
    """Predictor model configuration."""

    min_examples: int = 16
    target_examples: int = 128
    train_steps: int = 64


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
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)

    # Task configuration
    num_tasks: int = 5
    task_pairs: Tuple[Tuple[int, int], ...] = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))

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
        cfg.columns.num_columns = 8
        cfg.columns.shared_columns = 2
        cfg.columns.topk_nonshared = 2
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

        cfg.checkpoint.every_train_forwards = 8
        cfg.checkpoint.min_seconds = 0.0

        # Causal config for smoke tests
        cfg.support.causal_min_examples = 8
        cfg.support.causal_target_examples = 24
        cfg.support.sb_sinkhorn_iters = 4
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

    return cfg
