"""
Sequential Trainer for Continual Learning.

Provides task-by-task training with support selection, checkpointing,
and evaluation across all seen tasks.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.graph import initialize_params
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.training.train_backprop import train_backprop, evaluate_backprop

from fabricpc.continual.config import ExperimentConfig
from fabricpc.continual.data import TaskData
from fabricpc.continual.support import SupportManager, SupportState
from fabricpc.continual.causal import (
    CausalSupportFeatureBuilder,
    weighted_corr,
)
from fabricpc.continual.weight_causal import (
    PerWeightCausalLearner,
    PerWeightCausalConfig,
)
from fabricpc.continual.transweave import (
    TransWeaveManager,
    ComposerTransWeave,
    ShellDemotionTransWeave,
)
from fabricpc.continual.gradient_protection import (
    GradientProtector,
    GradientProtectionConfig,
    gradient_protection_transform,
)
from fabricpc.continual.ewc import EWCManager, ewc_gradient_transform


@dataclass
class TaskRunSummary:
    """Summary of a single task training run."""

    task_id: int
    classes: Tuple[int, int]
    train_accuracy: float
    test_accuracy: float
    train_loss: float
    test_loss: float
    train_energy: float
    test_energy: float
    epochs_trained: int
    training_time: float
    support_cols: Tuple[int, ...]

    # Optional detailed metrics
    accuracy_per_class: Optional[Dict[int, float]] = None
    epoch_accuracies: List[float] = field(default_factory=list)
    epoch_losses: List[float] = field(default_factory=list)

    # Selector metrics
    selector_policy_used: bool = False
    selector_trust: float = 0.5

    # Causal selector metrics
    causal_selector_examples: float = 0.0
    causal_selector_corr: float = 0.0
    causal_selector_mae: float = 0.0
    causal_selector_effective_scale: float = 0.0
    causal_selector_coverage_gate: float = 0.0
    causal_selector_agreement_gate: float = 0.0
    causal_selector_trend_gate: float = 0.0
    causal_selector_mix_gate: float = 0.0
    causal_similarity_mean: float = 0.0
    fingerprint_coverage_mean: float = 0.0

    # SB clarity metrics
    sb_mean_kurtosis: float = 0.0
    sb_mean_transport: float = 0.0

    # Per-weight causal metrics
    per_weight_standard_fraction: float = 1.0
    per_weight_sb_fraction: float = 0.0
    per_weight_mean_kurtosis: float = 0.0
    per_weight_max_kurtosis: float = 0.0
    per_weight_fraction_non_gaussian: float = 0.0

    # TransWeave metrics
    transweave_composer_sources: int = 0
    transweave_composer_strength: float = 0.0
    transweave_composer_cost: float = 0.0
    transweave_shell_demotions: int = 0
    transweave_shell_promotions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "classes": list(self.classes),
            "train_accuracy": float(self.train_accuracy),
            "test_accuracy": float(self.test_accuracy),
            "train_loss": float(self.train_loss),
            "test_loss": float(self.test_loss),
            "train_energy": float(self.train_energy),
            "test_energy": float(self.test_energy),
            "epochs_trained": self.epochs_trained,
            "training_time": float(self.training_time),
            "support_cols": list(self.support_cols),
            "epoch_accuracies": self.epoch_accuracies,
            "epoch_losses": self.epoch_losses,
            "selector_policy_used": self.selector_policy_used,
            "selector_trust": float(self.selector_trust),
            # Causal metrics
            "causal_selector_examples": float(self.causal_selector_examples),
            "causal_selector_corr": float(self.causal_selector_corr),
            "causal_selector_mae": float(self.causal_selector_mae),
            "causal_selector_effective_scale": float(
                self.causal_selector_effective_scale
            ),
            "causal_selector_coverage_gate": float(self.causal_selector_coverage_gate),
            "causal_selector_agreement_gate": float(
                self.causal_selector_agreement_gate
            ),
            "causal_selector_trend_gate": float(self.causal_selector_trend_gate),
            "causal_selector_mix_gate": float(self.causal_selector_mix_gate),
            "causal_similarity_mean": float(self.causal_similarity_mean),
            "fingerprint_coverage_mean": float(self.fingerprint_coverage_mean),
            "sb_mean_kurtosis": float(self.sb_mean_kurtosis),
            "sb_mean_transport": float(self.sb_mean_transport),
            # Per-weight causal metrics
            "per_weight_standard_fraction": float(self.per_weight_standard_fraction),
            "per_weight_sb_fraction": float(self.per_weight_sb_fraction),
            "per_weight_mean_kurtosis": float(self.per_weight_mean_kurtosis),
            "per_weight_max_kurtosis": float(self.per_weight_max_kurtosis),
            "per_weight_fraction_non_gaussian": float(
                self.per_weight_fraction_non_gaussian
            ),
            # TransWeave metrics
            "transweave_composer_sources": self.transweave_composer_sources,
            "transweave_composer_strength": float(self.transweave_composer_strength),
            "transweave_composer_cost": float(self.transweave_composer_cost),
            "transweave_shell_demotions": self.transweave_shell_demotions,
            "transweave_shell_promotions": self.transweave_shell_promotions,
        }


class SequentialTrainer:
    """
    Trainer for sequential/continual learning on multiple tasks.

    Supports:
    - Task-by-task training with support selection
    - Both PC and backprop training modes
    - Accuracy matrix tracking across tasks
    - Checkpointing and resumption
    - Callbacks for monitoring

    Example:
        >>> config = make_config(quick_smoke=False)
        >>> trainer = SequentialTrainer(structure, config)
        >>> for task_data in tasks:
        ...     summary = trainer.train_task(task_data)
        ...     print(f"Task {task_data.task_id}: {summary.test_accuracy:.2%}")
        >>> print(trainer.accuracy_matrix())
    """

    def __init__(
        self,
        structure: GraphStructure,
        config: ExperimentConfig,
        params: Optional[GraphParams] = None,
        optimizer: Optional[optax.GradientTransformation] = None,
        rng_key: Optional[jax.Array] = None,
    ):
        """
        Initialize the sequential trainer.

        Args:
            structure: FabricPC graph structure
            config: Experiment configuration
            params: Initial parameters (if None, will initialize)
            optimizer: Optax optimizer (if None, uses adamw)
            rng_key: JAX random key (if None, uses default seed)
        """
        self.structure = structure
        self.config = config

        # Initialize random key
        if rng_key is None:
            rng_key = jax.random.PRNGKey(config.seed)
        self.rng_key = rng_key

        # Initialize parameters
        if params is None:
            init_key, self.rng_key = jax.random.split(self.rng_key)
            params = initialize_params(structure, init_key)
        self.params = params

        # Gradient protection for continual learning (initialize early for optimizer chaining)
        # Uses shell-based gradient masking to prevent catastrophic forgetting
        gradient_protection_enable = getattr(config, "gradient_protection_enable", True)
        gradient_protection_config = GradientProtectionConfig(
            enable=gradient_protection_enable,
            shell_scales=(
                0.0,
                0.1,
                1.0,
            ),  # Protected center, stable inner, disposable outer
            protect_inactive_columns=True,
            inactive_column_scale=0.0,  # Freeze inactive columns completely
            shared_column_scale=0.2,  # Moderate protection for shared columns
            max_gradient_norm=1.0,
        )
        self.gradient_protector = GradientProtector(gradient_protection_config)

        # Initialize optimizer (chain with gradient protection and EWC if enabled)
        if optimizer is None:
            base_optimizer = optax.adamw(
                config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            base_optimizer = optimizer

        # EWC (Elastic Weight Consolidation) for preventing catastrophic forgetting
        # Initialize here so it's available for optimizer chain
        self.ewc_manager = EWCManager(
            lambda_ewc=config.ewc.lambda_ewc,
            online=config.ewc.online,
            gamma=config.ewc.gamma,
            normalize_fisher=config.ewc.normalize_fisher,
        )
        self._ewc_enabled = config.ewc.enable

        # Build optimizer chain with continual learning components
        # Order: EWC gradients -> gradient protection -> base optimizer
        transforms = []

        # Add EWC penalty gradients (pulls weights toward important values)
        if self._ewc_enabled:
            transforms.append(ewc_gradient_transform(self.ewc_manager))

        # Add gradient protection (masks frozen columns/shells)
        if gradient_protection_enable:
            transforms.append(gradient_protection_transform(self.gradient_protector))

        # Add base optimizer
        transforms.append(base_optimizer)

        if len(transforms) > 1:
            self.optimizer = optax.chain(*transforms)
        else:
            self.optimizer = base_optimizer

        self.opt_state = self.optimizer.init(params)
        self._base_optimizer = base_optimizer  # Keep reference for diagnostics

        # Support manager
        num_columns = config.columns.num_columns
        num_shared = config.columns.shared_columns
        topk_nonshared = config.columns.topk_nonshared

        self.support_manager = SupportManager(
            num_columns=num_columns,
            num_shared=num_shared,
            topk_nonshared=topk_nonshared,
            config=config.support,
            num_tasks=config.num_tasks,
        )

        # Training state
        self.tasks: List[TaskData] = []
        self.summaries: List[TaskRunSummary] = []
        self.global_step = 0
        self.current_task_id = -1

        # Accuracy matrix: accuracy[trained_up_to_task][evaluated_on_task]
        self._accuracy_matrix: Dict[int, Dict[int, float]] = {}
        self._task_supports: Dict[int, Tuple[int, ...]] = {}

        # Per-weight causal learning
        self.per_weight_causal = PerWeightCausalLearner(
            config=PerWeightCausalConfig(
                enable=config.per_weight_causal.enable,
                gradient_history_size=config.per_weight_causal.gradient_history_size,
                min_history_for_detection=config.per_weight_causal.min_history_for_detection,
                kurtosis_threshold=config.per_weight_causal.kurtosis_threshold,
                multimodal_threshold=config.per_weight_causal.multimodal_threshold,
                combined_threshold=config.per_weight_causal.combined_threshold,
                sb_sinkhorn_eps=config.per_weight_causal.sb_sinkhorn_eps,
                sb_sinkhorn_iters=config.per_weight_causal.sb_sinkhorn_iters,
                sb_correction_strength=config.per_weight_causal.sb_correction_strength,
                blend_mode=config.per_weight_causal.blend_mode,
                soft_blend_scale=config.per_weight_causal.soft_blend_scale,
                skip_bias_weights=config.per_weight_causal.skip_bias_weights,
                skip_small_weights=config.per_weight_causal.skip_small_weights,
                small_weight_threshold=config.per_weight_causal.small_weight_threshold,
                track_statistics=config.per_weight_causal.track_statistics,
                stats_update_every=config.per_weight_causal.stats_update_every,
            )
        )
        self._last_per_weight_stats: Dict[str, float] = {}

        # TransWeave multi-level transfer learning
        self.transweave_manager = TransWeaveManager(
            num_columns=num_columns,
            composer_config=config.composer_transweave,
            shell_config=config.shell_demotion_transweave,
        )
        self._last_transweave_stats: Dict[str, Any] = {}
        self._last_transition_autotune: Dict[str, Any] = {}
        self._transition_autotune_history: List[Dict[str, Any]] = []

        # Track column usage history for TransWeave shell dynamics
        self._column_usage_history: Dict[int, List[int]] = {}  # col_id -> [task_ids]

        # Callbacks
        self.epoch_callbacks: List[Callable] = []
        self.task_callbacks: List[Callable] = []

    @property
    def training_mode(self) -> str:
        """Get current training mode."""
        return self.config.training.training_mode

    def _estimate_rollout_signal(self, loader, max_batches: int) -> float:
        """
        Estimate a task-start rollout signal without mutating loader state when possible.

        The signal is a lightweight proxy for task intensity/novelty used to
        calibrate shell promotion/demotion thresholds before full training.
        """
        if max_batches <= 0:
            return 0.5

        batch_means = []
        batch_stds = []

        if hasattr(loader, "images") and hasattr(loader, "batch_size"):
            images = np.asarray(loader.images)
            batch_size = int(loader.batch_size)
            for batch_idx in range(max_batches):
                start = batch_idx * batch_size
                if start >= len(images):
                    break
                batch = np.asarray(images[start : start + batch_size], dtype=np.float32)
                if batch.size == 0:
                    continue
                batch_means.append(float(np.mean(np.abs(batch))))
                batch_stds.append(float(np.std(batch)))
        else:
            for batch_idx, (images, _) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                batch = np.asarray(images, dtype=np.float32)
                if batch.size == 0:
                    continue
                batch_means.append(float(np.mean(np.abs(batch))))
                batch_stds.append(float(np.std(batch)))

        if not batch_means:
            return 0.5

        mean_abs = float(np.mean(batch_means))
        mean_std = float(np.mean(batch_stds))
        return float(np.clip(0.65 * mean_abs + 0.35 * mean_std, 0.05, 1.5))

    def _build_shell_rollout_state(
        self,
        task_id: int,
        active_cols: Tuple[int, ...],
        rollout_signal: float,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Build deterministic shell assignments and activity proxies for autotuning."""
        shell_sizes = self.config.shell_demotion_transweave.shell_sizes
        num_columns = self.config.columns.num_columns
        num_shells = len(shell_sizes)
        num_neurons = sum(shell_sizes)

        support_mean_acc = (
            self.support_manager.support_bank.get_mean_accuracy_by_column(num_columns)
        )
        active_set = set(active_cols)
        historical_active = {
            col_id
            for col_id, task_ids in self._column_usage_history.items()
            if any(prev_task < task_id for prev_task in task_ids)
        }

        shell_assignments = np.repeat(
            np.arange(num_shells, dtype=np.int32),
            np.asarray(shell_sizes, dtype=np.int32),
        )
        column_activities: Dict[int, np.ndarray] = {}
        column_assignments: Dict[int, np.ndarray] = {}

        for col_id in range(num_columns):
            is_active = col_id in active_set
            was_active = col_id in historical_active
            history_depth = len(
                [t for t in self._column_usage_history.get(col_id, []) if t < task_id]
            )
            support_prior = (
                float(support_mean_acc[col_id]) if support_mean_acc[col_id] > 0 else 0.5
            )

            activities = np.zeros(num_neurons, dtype=np.float64)
            for neuron_id, shell_id in enumerate(shell_assignments):
                shell_base = 0.78 - 0.22 * shell_id
                activity = shell_base
                if is_active:
                    activity += 0.10 + 0.08 * rollout_signal - 0.05 * shell_id
                elif was_active:
                    activity -= 0.03 + 0.04 * shell_id
                    activity += min(history_depth, 4) * 0.01
                else:
                    activity -= 0.15 + 0.05 * shell_id

                activity += 0.10 * (support_prior - 0.5)
                phase = (
                    task_id * 0.37 + col_id * 0.19 + shell_id * 0.23 + neuron_id * 0.11
                )
                activity += 0.04 * np.sin(phase)
                activities[neuron_id] = np.clip(activity, 0.05, 0.99)

            column_activities[col_id] = activities
            column_assignments[col_id] = shell_assignments.copy()

        return column_activities, column_assignments

    def _set_composer_task_id(self, task_id: int) -> None:
        """
        Update the current task_id for ComposerNode attention routing.

        This allows ComposerNode to use the correct task-specific query vectors
        during inference without requiring explicit task_id input connections.
        """
        from fabricpc.continual.nodes import set_current_task_id

        set_current_task_id(task_id)

    def _set_support_context(self, support_cols: Optional[Tuple[int, ...]]) -> None:
        """Update the active support columns for continual routing."""
        from fabricpc.continual.nodes import set_current_support_cols

        set_current_support_cols(support_cols)

    def _set_task_context(
        self,
        task_id: int,
        support_cols: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Update task-level routing context used by continual nodes."""
        self._set_composer_task_id(task_id)
        if support_cols is None:
            support_cols = self._task_supports.get(task_id)
        self._set_support_context(support_cols)

    def _autotune_transition_thresholds(
        self,
        task_data: TaskData,
        active_cols: Tuple[int, ...],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Autotune shell promotion/demotion thresholds from limited task-start rollouts.
        """
        autotune_cfg = self.config.transition_autotune
        shell_cfg = self.config.shell_demotion_transweave
        shell_transweave = self.transweave_manager.shell_transweave

        if not autotune_cfg.enable or not shell_cfg.enable:
            return {}

        available_history_cols = [
            col_id
            for col_id, history in shell_transweave.column_histories.items()
            if len(history) >= 1
        ]
        if not available_history_cols:
            return {}

        rollout_signal = self._estimate_rollout_signal(
            task_data.train_loader, autotune_cfg.rollout_batches
        )
        column_activities, column_assignments = self._build_shell_rollout_state(
            task_id=task_data.task_id,
            active_cols=active_cols,
            rollout_signal=rollout_signal,
        )

        max_cols = autotune_cfg.max_columns_to_check
        columns_to_check = (
            available_history_cols[:max_cols]
            if max_cols > 0
            else available_history_cols
        )
        if not columns_to_check:
            return {}

        original_cfg = shell_transweave.config
        best: Optional[Dict[str, Any]] = None

        target_demotions = autotune_cfg.target_demotions_per_column * len(
            columns_to_check
        )
        target_promotions = autotune_cfg.target_promotions_per_column * len(
            columns_to_check
        )
        max_demotions = autotune_cfg.max_demotions_per_column * len(columns_to_check)
        max_promotions = autotune_cfg.max_promotions_per_column * len(columns_to_check)

        try:
            for demotion_threshold in autotune_cfg.demotion_threshold_candidates:
                for promotion_threshold in autotune_cfg.promotion_threshold_candidates:
                    for max_step in autotune_cfg.max_demotions_per_step_candidates:
                        candidate_cfg = replace(
                            original_cfg,
                            demotion_threshold=float(demotion_threshold),
                            promotion_threshold=float(promotion_threshold),
                            max_demotions_per_step=int(max_step),
                        )
                        shell_transweave.config = candidate_cfg

                        total_demotions = 0
                        total_promotions = 0
                        total_transport_cost = 0.0
                        total_entropy = 0.0

                        for col_id in columns_to_check:
                            result = shell_transweave.compute_demotion_transport(
                                column_id=col_id,
                                current_activities=column_activities[col_id],
                                current_assignments=column_assignments[col_id],
                            )
                            total_demotions += len(result.demotion_candidates)
                            total_promotions += len(result.promotion_candidates)
                            total_transport_cost += float(
                                result.diagnostics.get("transport_cost", 0.0)
                            )
                            total_entropy += float(
                                result.diagnostics.get("shell_transition_entropy", 0.0)
                            )

                        score = autotune_cfg.target_weight * (
                            abs(total_demotions - target_demotions)
                            + abs(total_promotions - target_promotions)
                        )
                        score += autotune_cfg.overfire_penalty * max(
                            0.0, total_demotions - max_demotions
                        )
                        score += autotune_cfg.overfire_penalty * max(
                            0.0, total_promotions - max_promotions
                        )
                        score += autotune_cfg.transport_cost_weight * (
                            total_transport_cost / max(len(columns_to_check), 1)
                        )
                        score += autotune_cfg.entropy_weight * (
                            total_entropy / max(len(columns_to_check), 1)
                        )
                        score += autotune_cfg.config_distance_weight * (
                            abs(demotion_threshold - original_cfg.demotion_threshold)
                            + abs(
                                promotion_threshold - original_cfg.promotion_threshold
                            )
                            + 0.25 * abs(max_step - original_cfg.max_demotions_per_step)
                        )

                        candidate_result = {
                            "task_id": task_data.task_id,
                            "rollout_signal": rollout_signal,
                            "checked_columns": tuple(columns_to_check),
                            "demotion_threshold": float(demotion_threshold),
                            "promotion_threshold": float(promotion_threshold),
                            "max_demotions_per_step": int(max_step),
                            "estimated_demotions": int(total_demotions),
                            "estimated_promotions": int(total_promotions),
                            "score": float(score),
                        }
                        if best is None or score < best["score"]:
                            best = candidate_result
        finally:
            shell_transweave.config = original_cfg

        if best is None:
            return {}

        tuned_cfg = replace(
            original_cfg,
            demotion_threshold=best["demotion_threshold"],
            promotion_threshold=best["promotion_threshold"],
            max_demotions_per_step=best["max_demotions_per_step"],
        )
        self.config.shell_demotion_transweave = tuned_cfg
        shell_transweave.config = tuned_cfg

        self._last_transition_autotune = best
        self._transition_autotune_history.append(dict(best))

        if verbose:
            print("Transition autotune:")
            print(
                "  thresholds="
                f"({best['demotion_threshold']:.3f}, {best['promotion_threshold']:.3f}) "
                f"max_step={best['max_demotions_per_step']}"
            )
            print(
                f"  rollout estimate: demotions={best['estimated_demotions']}, "
                f"promotions={best['estimated_promotions']}, "
                f"signal={best['rollout_signal']:.3f}"
            )

        return best

    def train_task(
        self,
        task_data: TaskData,
        verbose: bool = True,
    ) -> TaskRunSummary:
        """
        Train on a single task.

        Args:
            task_data: Task data containing loaders and metadata
            verbose: Whether to print progress

        Returns:
            TaskRunSummary with training results
        """
        task_id = task_data.task_id
        self.current_task_id = task_id

        if verbose:
            print(f"\n{'='*50}")
            print(f"Training Task {task_id}: classes {task_data.classes}")
            print(f"{'='*50}")

        # Select support columns
        support_state = self.support_manager.select_support_for_task(task_id)
        support_cols = support_state.active_all
        self._task_supports[task_id] = support_cols
        self._set_task_context(task_id, support_cols)
        self._autotune_transition_thresholds(
            task_data=task_data,
            active_cols=support_cols,
            verbose=verbose,
        )

        if verbose:
            print(f"Active columns: {support_cols}")

        # Update gradient protection for this task
        # This sets up column activity and shell assignments for gradient masking
        shared_cols = tuple(range(self.config.columns.shared_columns))
        self.gradient_protector.set_task(task_id)
        self.gradient_protector.set_active_columns(
            active_columns=support_cols,
            shared_columns=shared_cols,
            num_columns=self.config.columns.num_columns,
        )
        # Set memory_dim for input masking on downstream layers (aggregator)
        self.gradient_protector.state.memory_dim = self.config.columns.memory_dim
        # Set active classes for output layer gradient masking
        self.gradient_protector.set_active_classes(
            active_classes=task_data.classes,
            num_output_classes=getattr(self.config, "num_output_classes", 10),
        )
        # Set aggregator partitioning for task-specific pathways (if enabled)
        if self.config.columns.partition_aggregator:
            self.gradient_protector.set_aggregator_config(
                aggregator_dim=self.config.columns.aggregator_dim,
                num_tasks=self.config.num_tasks,
            )
        else:
            # Disable aggregator partitioning
            self.gradient_protector.set_aggregator_config(
                aggregator_dim=self.config.columns.aggregator_dim,
                num_tasks=0,  # num_tasks=0 disables partitioning
            )

        # Get shell assignments from transweave manager if available
        shell_assignments = {}
        shell_sizes = self.config.shell_demotion_transweave.shell_sizes
        for col_id in range(self.config.columns.num_columns):
            history = self.transweave_manager.shell_transweave.column_histories.get(
                col_id, []
            )
            if history:
                # Use most recent shell state
                shell_assignments[col_id] = history[-1].shell_assignments
            else:
                # Default to all neurons in outer shell (shell 2 = trainable)
                num_neurons = sum(shell_sizes)
                shell_assignments[col_id] = np.full(
                    num_neurons, len(shell_sizes) - 1, dtype=np.int32
                )

        self.gradient_protector.set_shell_assignments(shell_assignments, shell_sizes)

        if verbose:
            protection_stats = self.gradient_protector.get_protection_stats()
            print(
                f"Gradient protection: {protection_stats['columns_fully_protected']} frozen, "
                f"{protection_stats['columns_partially_protected']} partial, "
                f"{protection_stats['columns_unprotected']} open"
            )

        # Training configuration
        train_config = {
            "num_epochs": self.config.training.epochs_per_task,
            "loss_type": "cross_entropy",
        }

        # Apply fast dev limits if set
        max_train_batches = self.config.training.fast_dev_max_train_batches
        max_test_batches = self.config.training.fast_dev_max_test_batches

        # Create limited loaders if needed
        train_loader = task_data.train_loader
        test_loader = task_data.test_loader

        # Track epoch metrics
        epoch_accuracies = []
        epoch_losses = []

        def epoch_callback(epoch_idx, params, structure, config, rng):
            """Callback to track per-epoch metrics."""
            # Evaluate on test set
            eval_key, _ = jax.random.split(rng)
            if self.training_mode == "backprop":
                metrics = evaluate_backprop(
                    params, structure, test_loader, config, eval_key
                )
            else:
                metrics = evaluate_pcn(params, structure, test_loader, config, eval_key)

            epoch_accuracies.append(float(metrics.get("accuracy", 0.0)))
            epoch_losses.append(float(metrics.get("loss", 0.0)))

            if verbose:
                print(
                    f"  Epoch {epoch_idx + 1}: "
                    f"acc={metrics.get('accuracy', 0):.4f}, "
                    f"loss={metrics.get('loss', 0):.4f}"
                )

            # Call user callbacks
            for cb in self.epoch_callbacks:
                cb(epoch_idx, params, task_id, metrics)

            return metrics

        # Train
        start_time = time.time()
        train_key, self.rng_key = jax.random.split(self.rng_key)

        if self.training_mode == "backprop":
            self.params, loss_history, _, self.opt_state = train_backprop(
                params=self.params,
                structure=self.structure,
                train_loader=train_loader,
                optimizer=self.optimizer,
                config=train_config,
                rng_key=train_key,
                verbose=verbose,
                epoch_callback=epoch_callback,
                opt_state=self.opt_state,
                return_opt_state=True,
            )
        else:  # PC mode or hybrid (default to PC)
            self.params, energy_history, _, self.opt_state = train_pcn(
                params=self.params,
                structure=self.structure,
                train_loader=train_loader,
                optimizer=self.optimizer,
                config=train_config,
                rng_key=train_key,
                verbose=verbose,
                epoch_callback=epoch_callback,
                opt_state=self.opt_state,
                return_opt_state=True,
            )

        training_time = time.time() - start_time

        # Final evaluation
        eval_key, self.rng_key = jax.random.split(self.rng_key)
        if self.training_mode == "backprop":
            train_metrics = evaluate_backprop(
                self.params, self.structure, train_loader, train_config, eval_key
            )
            test_metrics = evaluate_backprop(
                self.params, self.structure, test_loader, train_config, eval_key
            )
        else:
            train_metrics = evaluate_pcn(
                self.params, self.structure, train_loader, train_config, eval_key
            )
            test_metrics = evaluate_pcn(
                self.params, self.structure, test_loader, train_config, eval_key
            )

        # Run support swap audit for causal learning
        old_task_data = self.tasks[-1] if len(self.tasks) > 0 else None
        audit_rows = self._run_support_swap_audit(
            current_task_data=task_data,
            old_task_data=old_task_data,
            verbose=verbose,
        )

        # Update causal components with audit results
        causal_metrics = {"causal_selector_corr": 0.0, "causal_selector_mae": 0.0}
        if audit_rows:
            old_task_id = old_task_data.task_id if old_task_data else None
            causal_metrics = self.support_manager.update_causal_from_audit(
                audit_rows=audit_rows,
                current_task_id=task_id,
                old_task=old_task_id,
            )

            # Build causal training examples for predictor
            self._add_causal_training_examples(audit_rows, task_id)

        # Update causal trust with recent agreement and get diagnostics
        causal_diag = {}
        if self.support_manager.causal_trust is not None:
            # Get recent agreement from tracker
            recent_agreement, recent_rows = (
                self.support_manager.causal_trust.get_recent_agreement()
            )

            # Compute trust gates with actual agreement
            if self.support_manager.causal_predictor is not None:
                causal_diag = self.support_manager.causal_trust.compute(
                    predictor=self.support_manager.causal_predictor,
                    effective_internal_trust=self.support_manager.trust_controller.get_trust(),
                    recent_agreement=recent_agreement,
                    recent_rows=recent_rows,
                )
            else:
                causal_diag = self.support_manager.causal_trust.last_diag

        # Get per-weight causal stats
        per_weight_stats = self.per_weight_causal.get_stats()

        # Register end-of-task state with TransWeave for transfer learning
        transweave_stats = self._register_transweave_task_end(task_data)

        # Create summary
        summary = TaskRunSummary(
            task_id=task_id,
            classes=task_data.classes,
            train_accuracy=float(train_metrics.get("accuracy", 0.0)),
            test_accuracy=float(test_metrics.get("accuracy", 0.0)),
            train_loss=float(train_metrics.get("loss", 0.0)),
            test_loss=float(test_metrics.get("loss", 0.0)),
            train_energy=float(train_metrics.get("energy", 0.0)),
            test_energy=float(test_metrics.get("energy", 0.0)),
            epochs_trained=self.config.training.epochs_per_task,
            training_time=training_time,
            support_cols=support_cols,
            epoch_accuracies=epoch_accuracies,
            epoch_losses=epoch_losses,
            selector_policy_used=self.support_manager.trust_controller.should_use_policy(),
            selector_trust=self.support_manager.trust_controller.get_trust(),
            # Causal metrics
            causal_selector_examples=float(
                self.support_manager.causal_predictor.num_examples()
                if self.support_manager.causal_predictor
                else 0
            ),
            causal_selector_corr=float(causal_metrics.get("causal_selector_corr", 0.0)),
            causal_selector_mae=float(causal_metrics.get("causal_selector_mae", 0.0)),
            causal_selector_effective_scale=float(
                causal_diag.get("effective_scale", 0.0)
            ),
            causal_selector_coverage_gate=float(causal_diag.get("coverage_gate", 0.0)),
            causal_selector_agreement_gate=float(
                causal_diag.get("agreement_gate", 0.0)
            ),
            causal_selector_trend_gate=float(causal_diag.get("trend_gate", 0.0)),
            causal_selector_mix_gate=float(causal_diag.get("mix_gate", 0.0)),
            # Per-weight causal metrics
            per_weight_mean_kurtosis=float(per_weight_stats.get("mean_kurtosis", 0.0)),
            per_weight_max_kurtosis=float(per_weight_stats.get("max_kurtosis", 0.0)),
            per_weight_fraction_non_gaussian=float(
                per_weight_stats.get("mean_fraction_non_gaussian", 0.0)
            ),
            # TransWeave metrics
            transweave_composer_sources=transweave_stats.get("composer_sources", 0),
            transweave_composer_strength=float(
                transweave_stats.get("composer_strength", 0.0)
            ),
            transweave_composer_cost=float(transweave_stats.get("composer_cost", 0.0)),
            transweave_shell_demotions=transweave_stats.get("shell_demotions", 0),
            transweave_shell_promotions=transweave_stats.get("shell_promotions", 0),
        )

        # Record outcome in support manager
        self.support_manager.record_outcome(
            task_id=task_id,
            accuracy=summary.test_accuracy,
            loss=summary.test_loss,
        )

        # Update accuracy matrix
        self._update_accuracy_matrix(task_id, task_data)

        # Store summary
        self.summaries.append(summary)
        self.tasks.append(task_data)

        # Call task callbacks
        for cb in self.task_callbacks:
            cb(task_id, summary)

        self.global_step += 1

        # EWC: Consolidate knowledge after training this task
        if self._ewc_enabled:
            consolidate_key, self.rng_key = jax.random.split(self.rng_key)
            self._consolidate_ewc(task_data, consolidate_key, verbose=verbose)

        if verbose:
            print(f"\nTask {task_id} complete:")
            print(f"  Train accuracy: {summary.train_accuracy:.4f}")
            print(f"  Test accuracy:  {summary.test_accuracy:.4f}")
            print(f"  Training time:  {training_time:.1f}s")
            if summary.causal_selector_examples > 0:
                print(f"  Causal examples: {summary.causal_selector_examples:.0f}")
                print(f"  Causal corr:     {summary.causal_selector_corr:.4f}")
                print(f"  Causal mae:      {summary.causal_selector_mae:.4f}")
                print(
                    f"  Agreement gate:  {summary.causal_selector_agreement_gate:.4f}"
                )
                print(f"  Mix gate:        {summary.causal_selector_mix_gate:.4f}")
            if self.config.per_weight_causal.enable:
                print(f"  Per-weight causal:")
                print(
                    f"    Mean kurtosis:       {summary.per_weight_mean_kurtosis:.4f}"
                )
                print(f"    Max kurtosis:        {summary.per_weight_max_kurtosis:.4f}")
                print(
                    f"    Non-Gaussian frac:   {summary.per_weight_fraction_non_gaussian:.4f}"
                )
            if (
                self.config.composer_transweave.enable
                or self.config.shell_demotion_transweave.enable
            ):
                print(f"  TransWeave:")
                if self.config.composer_transweave.enable:
                    print(
                        f"    Composer sources:    {summary.transweave_composer_sources}"
                    )
                    print(
                        f"    Composer strength:   {summary.transweave_composer_strength:.4f}"
                    )
                if self.config.shell_demotion_transweave.enable:
                    print(
                        f"    Shell demotions:     {summary.transweave_shell_demotions}"
                    )
                    print(
                        f"    Shell promotions:    {summary.transweave_shell_promotions}"
                    )

        return summary

    def _evaluate_task_masked(
        self,
        task_data: TaskData,
        rng_key: jax.Array,
    ) -> float:
        """
        Evaluate on a task using task-masked accuracy.

        Only considers the classes belonging to this task when computing argmax.
        This prevents output interference from other tasks' class logits.

        Args:
            task_data: Task to evaluate
            rng_key: JAX random key

        Returns:
            Task-masked accuracy
        """
        from fabricpc.graph.state_initializer import initialize_graph_state
        from fabricpc.core.inference import run_inference

        task_classes = task_data.classes  # e.g., (0, 1) for task 0

        total_correct = 0
        total_samples = 0

        for batch_idx, batch_data in enumerate(task_data.test_loader):
            # Convert batch
            if isinstance(batch_data, (list, tuple)):
                batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
            elif isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            else:
                continue

            batch_key, rng_key = jax.random.split(rng_key)
            batch_size = batch["x"].shape[0]

            # Set up clamps (only input during eval)
            clamps = {}
            if "x" in self.structure.task_map:
                x_node = self.structure.task_map["x"]
                clamps[x_node] = batch["x"]

            # Initialize graph state
            state = initialize_graph_state(
                self.structure,
                batch_size,
                batch_key,
                clamps=clamps,
                params=self.params,
            )

            # Run inference
            final_state = run_inference(self.params, state, clamps, self.structure)

            # Get output logits
            y_node = self.structure.task_map["y"]
            predictions = final_state.nodes[y_node].z_mu  # (batch, num_classes)

            # Task-masked argmax: only consider this task's classes
            # Create mask: -inf for non-task classes, 0 for task classes
            num_classes = predictions.shape[-1]
            mask = jnp.full(num_classes, -jnp.inf)
            for cls in task_classes:
                mask = mask.at[cls].set(0.0)

            # Apply mask and argmax
            masked_predictions = predictions + mask[None, :]
            pred_labels = jnp.argmax(masked_predictions, axis=1)

            # Get true labels
            true_labels = jnp.argmax(batch["y"], axis=1)

            # Count correct
            correct = jnp.sum(pred_labels == true_labels)
            total_correct += int(correct)
            total_samples += batch_size

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return accuracy

    def _update_accuracy_matrix(
        self, current_task_id: int, current_task_data: TaskData
    ):
        """Update accuracy matrix after training on a task."""
        if current_task_id not in self._accuracy_matrix:
            self._accuracy_matrix[current_task_id] = {}

        # Build list of all tasks to evaluate (previously seen + current)
        tasks_to_eval = list(self.tasks)  # Previously seen tasks
        # Add current task if not already in the list
        if not any(t.task_id == current_task_data.task_id for t in tasks_to_eval):
            tasks_to_eval.append(current_task_data)

        # Evaluate on all seen tasks including current using TASK-MASKED evaluation
        for task_data in tasks_to_eval:
            eval_task_id = task_data.task_id
            eval_key, self.rng_key = jax.random.split(self.rng_key)

            # Set task_id in ComposerNode for proper attention routing
            support_cols = self._task_supports.get(eval_task_id)
            self._set_task_context(eval_task_id, support_cols)

            # Use task-masked evaluation to avoid output interference
            accuracy = self._evaluate_task_masked(task_data, eval_key)

            self._accuracy_matrix[current_task_id][eval_task_id] = accuracy

    def _collect_batches(
        self,
        loader,
        max_batches: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Collect up to `max_batches` batches from a loader."""
        batches = []
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= max_batches:
                break
            if isinstance(batch_data, (list, tuple)):
                images, labels = batch_data
            elif isinstance(batch_data, dict):
                images = batch_data.get("x")
                labels = batch_data.get("y")
            else:
                continue
            batches.append((np.asarray(images), np.asarray(labels)))
        return batches

    def _coalesce_batches(
        self,
        batches: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Collapse sampled batches into one large batch to bound evaluation cost."""
        if not batches:
            return []
        if len(batches) == 1:
            return list(batches)
        images = np.concatenate([images for images, _ in batches], axis=0)
        labels = np.concatenate([labels for _, labels in batches], axis=0)
        return [(images, labels)]

    def _compute_batch_state(
        self,
        batch: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
    ):
        """Compute a model state for a single batch under the current task context."""
        if self.training_mode == "backprop":
            from fabricpc.training.train_backprop import compute_forward_pass

            return compute_forward_pass(self.params, self.structure, batch, rng_key)

        from fabricpc.graph.state_initializer import initialize_graph_state
        from fabricpc.core.inference import run_inference

        batch_size = batch["x"].shape[0]
        clamps = {}
        if "x" in self.structure.task_map:
            clamps[self.structure.task_map["x"]] = batch["x"]

        state = initialize_graph_state(
            self.structure,
            batch_size,
            rng_key,
            clamps=clamps,
            params=self.params,
        )
        return run_inference(self.params, state, clamps, self.structure)

    def _evaluate_loss_on_batches(
        self,
        batches: List[Tuple[np.ndarray, np.ndarray]],
        task_id: int,
        support_cols: Tuple[int, ...],
    ) -> float:
        """Evaluate loss on sampled batches under a specific support selection."""
        if not batches:
            return 0.0

        eval_loader = self._coalesce_batches(batches)
        eval_key, self.rng_key = jax.random.split(self.rng_key)
        self._set_task_context(task_id, support_cols)
        eval_config = {"loss_type": "cross_entropy"}

        if self.training_mode == "backprop":
            metrics = evaluate_backprop(
                self.params,
                self.structure,
                eval_loader,
                eval_config,
                eval_key,
            )
        else:
            metrics = evaluate_pcn(
                self.params,
                self.structure,
                eval_loader,
                eval_config,
                eval_key,
            )

        return float(metrics.get("loss", 0.0))

    def _align_shell_assignments(
        self,
        assignments: np.ndarray,
        num_neurons: int,
    ) -> np.ndarray:
        """Align shell assignments to the actual column width."""
        assignments = np.asarray(assignments, dtype=np.int32)
        if assignments.shape[0] == num_neurons:
            return assignments
        if assignments.shape[0] > num_neurons:
            return assignments[:num_neurons]

        if assignments.size == 0:
            assignments = np.zeros((1,), dtype=np.int32)
        pad_value = int(assignments[-1])
        pad = np.full((num_neurons - assignments.shape[0],), pad_value, dtype=np.int32)
        return np.concatenate([assignments, pad], axis=0)

    def _default_shell_assignments(
        self, num_neurons: Optional[int] = None
    ) -> np.ndarray:
        """Create default contiguous shell assignments for a fresh column."""
        if num_neurons is None:
            num_neurons = self.config.columns.memory_dim
        shell_sizes = self.config.shell_demotion_transweave.shell_sizes
        assignments = np.empty(sum(shell_sizes), dtype=np.int32)
        start = 0
        for shell_id, shell_size in enumerate(shell_sizes):
            assignments[start : start + shell_size] = shell_id
            start += shell_size
        return self._align_shell_assignments(assignments, num_neurons)

    def _measure_column_activities(
        self,
        task_id: int,
        support_cols: Tuple[int, ...],
        batches: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        """Measure per-column neuron activity from real forward passes."""
        num_columns = self.config.columns.num_columns
        num_neurons = self.config.columns.memory_dim
        activity_sum = np.zeros((num_columns, num_neurons), dtype=np.float64)
        batch_count = 0

        for images, labels in self._coalesce_batches(batches):
            batch = {"x": jnp.array(images), "y": jnp.array(labels)}
            batch_key, self.rng_key = jax.random.split(self.rng_key)
            self._set_task_context(task_id, support_cols)
            state = self._compute_batch_state(batch, batch_key)
            if "columns" not in state.nodes:
                continue
            column_z = np.asarray(state.nodes["columns"].z_mu)
            activity_sum += np.mean(np.abs(column_z), axis=0)
            batch_count += 1

        if batch_count == 0:
            return {
                col_id: np.zeros((num_neurons,), dtype=np.float64)
                for col_id in range(num_columns)
            }

        activity_mean = activity_sum / batch_count
        return {col_id: activity_mean[col_id] for col_id in range(num_columns)}

    def _extract_composer_summary(
        self,
        task_id: int,
        support_cols: Tuple[int, ...],
        batches: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Dict[str, Any]]:
        """Extract measured composer state from the trained aggregator."""
        aggregator_node = self.structure.nodes.get("aggregator")
        if aggregator_node is None:
            return None

        node_class = aggregator_node.node_info.node_class
        if getattr(node_class, "__name__", "") != "ComposerNode":
            return None

        params = self.params.nodes["aggregator"]
        config = aggregator_node.node_info.node_config
        hidden_dim = aggregator_node.node_info.shape[0]
        num_heads = config.get("num_heads", 2)
        num_layers = config.get("num_layers", 1)
        head_dim = hidden_dim // max(num_heads, 1)
        num_columns = self.config.columns.num_columns

        attention_accum = []
        gate_accum = []

        support_mask = np.zeros((num_columns,), dtype=np.float32)
        for col_idx in support_cols:
            if 0 <= col_idx < num_columns:
                support_mask[col_idx] = 1.0

        for images, labels in self._coalesce_batches(batches):
            batch = {"x": jnp.array(images), "y": jnp.array(labels)}
            batch_key, self.rng_key = jax.random.split(self.rng_key)
            self._set_task_context(task_id, support_cols)
            state = self._compute_batch_state(batch, batch_key)
            if "columns" not in state.nodes:
                continue

            x = np.asarray(state.nodes["columns"].z_mu)
            mask = np.broadcast_to(support_mask[None, :], (x.shape[0], num_columns))

            x_proj = np.matmul(
                x, np.asarray(params.weights["input_proj"])
            ) + np.asarray(params.biases["input_proj_bias"])
            x_proj = np.asarray(
                node_class._layer_norm(
                    jnp.array(x_proj),
                    params.weights["ln_0_scale"],
                    params.biases["ln_0_shift"],
                )
            )

            last_attn = None
            for layer in range(num_layers):
                q = np.matmul(x_proj, np.asarray(params.weights[f"layer_{layer}_q"]))
                k = np.matmul(x_proj, np.asarray(params.weights[f"layer_{layer}_k"]))
                v = np.matmul(x_proj, np.asarray(params.weights[f"layer_{layer}_v"]))

                q = q.reshape(x.shape[0], num_columns, num_heads, head_dim).transpose(
                    0, 2, 1, 3
                )
                k = k.reshape(x.shape[0], num_columns, num_heads, head_dim).transpose(
                    0, 2, 1, 3
                )
                v = v.reshape(x.shape[0], num_columns, num_heads, head_dim).transpose(
                    0, 2, 1, 3
                )

                scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(
                    max(head_dim, 1)
                )
                scores = np.where(mask[:, None, None, :] > 0, scores, -1e9)
                attn = jax.nn.softmax(jnp.array(scores), axis=-1)
                last_attn = np.asarray(attn)

                out = np.matmul(last_attn, v)
                out = out.transpose(0, 2, 1, 3).reshape(
                    x.shape[0], num_columns, hidden_dim
                )
                out = np.matmul(
                    out, np.asarray(params.weights[f"layer_{layer}_out"])
                ) + np.asarray(params.biases[f"layer_{layer}_out_bias"])
                x_proj = np.asarray(
                    node_class._layer_norm(
                        jnp.array(x_proj + out),
                        params.weights[f"ln_{layer + 1}_scale"],
                        params.biases[f"ln_{layer + 1}_shift"],
                    )
                )

            if last_attn is None:
                continue

            gate_logits = np.matmul(
                x_proj, np.asarray(params.weights["gate_proj"])
            ).squeeze(-1)
            gate_logits = np.where(mask > 0, gate_logits, -1e9)
            attention_accum.append(np.mean(last_attn, axis=0))
            gate_accum.append(np.mean(gate_logits, axis=0))

        if not attention_accum:
            return None

        final_layer = num_layers - 1
        query_proj = (
            np.asarray(params.weights[f"layer_{final_layer}_q"])
            .reshape(hidden_dim, num_heads, head_dim)
            .transpose(1, 0, 2)
        )
        key_proj = (
            np.asarray(params.weights[f"layer_{final_layer}_k"])
            .reshape(hidden_dim, num_heads, head_dim)
            .transpose(1, 0, 2)
        )
        value_proj = (
            np.asarray(params.weights[f"layer_{final_layer}_v"])
            .reshape(hidden_dim, num_heads, head_dim)
            .transpose(1, 0, 2)
        )

        return {
            "attention_weights": np.mean(np.stack(attention_accum, axis=0), axis=0),
            "query_projections": query_proj,
            "key_projections": key_proj,
            "value_projections": value_proj,
            "output_projection": np.asarray(params.weights["output_proj"]),
            "gate_logits": np.mean(np.stack(gate_accum, axis=0), axis=0),
        }

    def _run_support_swap_audit(
        self,
        current_task_data: TaskData,
        old_task_data: Optional[TaskData] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run support swap audit to generate causal training data.

        For each chosen column, swaps it with unchosen columns and measures
        loss differences. These results feed into the CausalFingerprintBank
        and CausalContributionPredictor.

        Args:
            current_task_data: Current task's data
            old_task_data: Optional previous task data for old-task loss
            verbose: Whether to print progress

        Returns:
            List of audit row dictionaries
        """
        if not self.config.audit.support_swap_audit_enable:
            return []

        audit_rows = []
        current_task_id = current_task_data.task_id

        # Get current support columns
        chosen = list(self.current_state.active_nonshared)
        # Get unchosen columns (non-shared, not in chosen)
        all_nonshared = list(
            range(self.config.columns.shared_columns, self.config.columns.num_columns)
        )
        unchosen = [c for c in all_nonshared if c not in chosen]

        if not unchosen or not chosen:
            return []

        # Limit number of swaps
        max_swaps = self.config.audit.support_swap_audit_max_swaps
        max_batches = self.config.audit.support_audit_max_batches

        # Collect batches for evaluation
        current_batches = self._collect_batches(
            current_task_data.test_loader, max_batches
        )
        old_batches = (
            self._collect_batches(old_task_data.test_loader, max_batches)
            if old_task_data is not None
            else []
        )

        # Baseline loss with current support
        baseline_support = self.current_state.active_all
        baseline_current_loss = self._evaluate_loss_on_batches(
            current_batches, current_task_id, baseline_support
        )
        baseline_old_loss = (
            self._evaluate_loss_on_batches(
                old_batches, current_task_id, baseline_support
            )
            if old_batches
            else 0.0
        )

        if verbose:
            print(
                f"  Audit baseline: current_loss={baseline_current_loss:.4f}, old_loss={baseline_old_loss:.4f}"
            )

        chosen_arr = np.asarray(chosen, dtype=np.int32)
        unchosen_arr = np.asarray(unchosen, dtype=np.int32)
        chosen_idx_grid, swap_in_grid = np.meshgrid(
            np.arange(chosen_arr.size, dtype=np.int32),
            unchosen_arr,
            indexing="ij",
        )
        swap_out_grid = chosen_arr[chosen_idx_grid]

        flat_swap_out = swap_out_grid.reshape(-1)[:max_swaps]
        flat_swap_in = swap_in_grid.reshape(-1)[:max_swaps]
        if flat_swap_out.size == 0:
            return []

        current_weight = self.config.audit.support_audit_current_weight
        old_weight = self.config.audit.support_audit_old_weight

        old_task_id = old_task_data.task_id if old_task_data else None
        chosen_tuple = tuple(chosen)

        for i in range(flat_swap_out.shape[0]):
            alt_nonshared = list(chosen)
            swap_out = int(flat_swap_out[i])
            swap_in = int(flat_swap_in[i])
            swap_out_idx = alt_nonshared.index(swap_out)
            alt_nonshared[swap_out_idx] = swap_in
            alt_nonshared = tuple(sorted(alt_nonshared))
            alt_support = (
                tuple(range(self.config.columns.shared_columns)) + alt_nonshared
            )

            alt_current_loss = self._evaluate_loss_on_batches(
                current_batches, current_task_id, alt_support
            )
            alt_old_loss = (
                self._evaluate_loss_on_batches(
                    old_batches, current_task_id, alt_support
                )
                if old_batches
                else 0.0
            )

            current_gain = baseline_current_loss - alt_current_loss
            old_gain = baseline_old_loss - alt_old_loss
            combined_gain = current_weight * current_gain + old_weight * old_gain

            audit_rows.append(
                {
                    "swap_in": swap_in,
                    "swap_out": swap_out,
                    "chosen_current_loss": float(baseline_current_loss),
                    "alt_current_loss": float(alt_current_loss),
                    "chosen_old_loss": float(baseline_old_loss),
                    "alt_old_loss": float(alt_old_loss),
                    "current_gain": float(current_gain),
                    "old_gain": float(old_gain),
                    "combined_gain": float(combined_gain),
                    "current_task_id": current_task_id,
                    "old_task_id": old_task_id,
                    "chosen_support": chosen_tuple,
                    "alt_support": alt_support,
                }
            )

        self._set_task_context(current_task_id, baseline_support)

        if verbose:
            print(f"  Generated {len(audit_rows)} audit rows")

        return audit_rows

    def _add_causal_training_examples(
        self,
        audit_rows: List[Dict[str, Any]],
        current_task_id: int,
    ) -> None:
        """
        Build and add causal training examples from audit rows.

        Converts audit rows into feature vectors for the CausalContributionPredictor.
        """
        if self.support_manager.causal_predictor is None:
            return

        if not audit_rows:
            return

        feature_builder = self.support_manager.causal_feature_builder
        num_columns = self.config.columns.num_columns
        num_shared = self.config.columns.shared_columns

        # Get fingerprint data
        fp_mean = None
        fp_conf = None
        if self.support_manager.causal_bank is not None:
            fp_mean = self.support_manager.causal_bank.mean_gain()
            fp_conf = self.support_manager.causal_bank.column_confidence(
                self.config.support.causal_similarity_conf_target
            )

        # Build placeholder certificate arrays (in a full implementation,
        # these would come from the support manager's tracking)
        cert_general = np.zeros(num_columns)
        cert_specific = np.zeros(num_columns)
        cert_demotion = np.zeros(num_columns)
        cert_saturation = np.zeros(num_columns)
        novelty = np.zeros(num_columns)
        saturation = np.zeros(num_columns)
        recent_penalty = np.zeros(num_columns)
        reserve_bonus = np.zeros(num_columns)
        base_z = np.zeros(num_columns)

        X_list = []
        y_list = []
        w_list = []
        meta_list = []
        chosen_sets = []
        valid_rows = []

        for row in audit_rows:
            swap_in = row.get("swap_in", -1)
            if swap_in < 0 or swap_in >= num_columns:
                continue

            # Target is the combined gain (positive = swap_in is better)
            target = row.get("combined_gain", 0.0)
            # Clamp target
            max_abs = self.config.support.causal_max_abs_target
            target = float(np.clip(target, -max_abs, max_abs))
            # Scale target
            target = target * self.config.support.causal_target_scale

            # Weight based on magnitude of current loss
            weight = 1.0 + abs(row.get("chosen_current_loss", 0.0))

            X_list.append(swap_in)
            y_list.append(target)
            w_list.append(weight)
            chosen_sets.append(tuple(row.get("chosen_support", ())))
            valid_rows.append(row)
            meta_list.append(
                {
                    "swap_in": swap_in,
                    "swap_out": row.get("swap_out"),
                    "task_id": current_task_id,
                }
            )

        if X_list:
            X = feature_builder.build_features_batch(
                indices=X_list,
                roles="challenger",
                chosen_sets=chosen_sets,
                base_z=base_z,
                cert_general=cert_general,
                cert_specific=cert_specific,
                cert_demotion=cert_demotion,
                cert_saturation=cert_saturation,
                novelty=novelty,
                saturation=saturation,
                recent_penalty=recent_penalty,
                reserve_bonus=reserve_bonus,
                fingerprint_mean=fp_mean,
                fingerprint_confidence=fp_conf,
                current_task_id=current_task_id,
            )
            y = np.array(y_list)
            w = np.array(w_list)
            self.support_manager.add_causal_examples(X, y, w, meta_list)

            # Record predictions and outcomes for agreement tracking
            if (
                self.support_manager.causal_predictor is not None
                and self.support_manager.causal_predictor.trained
            ):
                predictions = self.support_manager.causal_predictor.predict(X)
                trust_ctrl = self.support_manager.causal_trust
                if trust_ctrl is not None:
                    for i, row in enumerate(valid_rows):
                        swap_in = row.get("swap_in", -1)
                        if swap_in >= 0:
                            # Record prediction
                            trust_ctrl.record_prediction(
                                task_id=current_task_id,
                                column_idx=swap_in,
                                predicted_score=float(predictions[i]),
                                role="challenger",
                            )
                            # Record actual outcome
                            trust_ctrl.record_outcome(
                                task_id=current_task_id,
                                column_idx=swap_in,
                                actual_gain=float(row.get("combined_gain", 0.0)),
                            )

    def _register_transweave_task_end(self, task_data: TaskData) -> Dict[str, Any]:
        """
        Register end-of-task state with TransWeave for transfer learning.

        Captures composer state and shell assignments to enable transfer
        to future tasks via Sinkhorn optimal transport.

        Args:
            task_data: Completed task metadata and loaders

        Returns:
            Dict with TransWeave statistics for this task
        """
        stats = {
            "composer_sources": 0,
            "composer_strength": 0.0,
            "composer_cost": 0.0,
            "shell_demotions": 0,
            "shell_promotions": 0,
        }

        task_id = task_data.task_id
        num_columns = self.config.columns.num_columns
        active_cols = self.current_state.active_all
        measure_batches = max(1, self.config.audit.support_audit_max_batches)
        measurement_batches = self._collect_batches(
            task_data.test_loader, measure_batches
        )

        # Register with composer TransWeave
        if self.config.composer_transweave.enable:
            composer_summary = self._extract_composer_summary(
                task_id=task_id,
                support_cols=active_cols,
                batches=measurement_batches,
            )
            if composer_summary is not None:
                self.transweave_manager.composer_transweave.register_task(
                    task_id=task_id,
                    attention_weights=composer_summary["attention_weights"],
                    query_projections=composer_summary["query_projections"],
                    key_projections=composer_summary["key_projections"],
                    value_projections=composer_summary["value_projections"],
                    output_projection=composer_summary["output_projection"],
                    gate_logits=composer_summary["gate_logits"],
                    metadata={
                        "source": "measured",
                        "num_batches": len(measurement_batches),
                    },
                )

                if task_id > 0:
                    transfer_result = (
                        self.transweave_manager.composer_transweave.compute_transfer(
                            target_task_id=task_id,
                            current_attention=composer_summary["attention_weights"],
                            current_queries=composer_summary["query_projections"],
                            current_keys=composer_summary["key_projections"],
                            current_values=composer_summary["value_projections"],
                        )
                    )
                    stats["composer_sources"] = len(transfer_result.source_tasks)
                    stats["composer_strength"] = transfer_result.transfer_strength
                    stats["composer_cost"] = transfer_result.diagnostics.get(
                        "mean_transport_cost", 0.0
                    )

        # Register with shell demotion TransWeave
        if self.config.shell_demotion_transweave.enable:
            # Update column usage history
            for col_id in active_cols:
                if col_id not in self._column_usage_history:
                    self._column_usage_history[col_id] = []
                self._column_usage_history[col_id].append(task_id)

            # First, compute demotions BEFORE registering new states
            # This compares new activities against historical patterns
            total_demotions = 0
            total_promotions = 0

            column_activities = self._measure_column_activities(
                task_id=task_id,
                support_cols=active_cols,
                batches=measurement_batches,
            )
            column_assignments = {}
            for col_id in range(num_columns):
                history = self.transweave_manager.shell_transweave.column_histories.get(
                    col_id, []
                )
                if history:
                    column_assignments[col_id] = self._align_shell_assignments(
                        history[-1].shell_assignments,
                        column_activities[col_id].shape[0],
                    )
                else:
                    column_assignments[col_id] = self._default_shell_assignments(
                        column_activities[col_id].shape[0]
                    )

            for col_id in range(num_columns):
                history = self.transweave_manager.shell_transweave.column_histories.get(
                    col_id, []
                )
                if history:
                    result = self.transweave_manager.shell_transweave.compute_demotion_transport(
                        column_id=col_id,
                        current_activities=column_activities[col_id],
                        current_assignments=column_assignments[col_id],
                    )
                    if result.demotion_candidates or result.promotion_candidates:
                        new_assignments, counts = (
                            self.transweave_manager.shell_transweave.apply_transitions(
                                column_assignments[col_id],
                                result,
                            )
                        )
                        column_assignments[col_id] = new_assignments
                        total_demotions += counts["demotions_applied"]
                        total_promotions += counts["promotions_applied"]

            for col_id in range(num_columns):
                self.transweave_manager.shell_transweave.register_shell_state(
                    column_id=col_id,
                    task_id=task_id,
                    shell_assignments=column_assignments[col_id],
                    neuron_activities=column_activities[col_id],
                )

            stats["shell_demotions"] = total_demotions
            stats["shell_promotions"] = total_promotions

        self._last_transweave_stats = stats
        return stats

    @property
    def current_state(self) -> SupportState:
        """Get current support state."""
        return self.support_manager.current_state

    def accuracy_matrix(self) -> np.ndarray:
        """
        Get the accuracy matrix.

        Returns:
            2D array where matrix[i][j] is accuracy on task j
            after training up to task i.
        """
        if not self._accuracy_matrix:
            return np.array([])

        # Collect all task IDs that have accuracy entries
        all_task_ids = set()
        for row in self._accuracy_matrix.values():
            if row:  # Only consider non-empty rows
                all_task_ids.update(row.keys())

        if not all_task_ids:
            return np.array([])

        max_task = max(all_task_ids)
        num_tasks = max_task + 1

        matrix = np.zeros((len(self._accuracy_matrix), num_tasks))
        for train_idx, (train_task, eval_dict) in enumerate(
            sorted(self._accuracy_matrix.items())
        ):
            for eval_task, acc in eval_dict.items():
                if eval_task < num_tasks:
                    matrix[train_idx, eval_task] = acc

        return matrix

    def evaluate_task(self, task_id: int) -> float:
        """
        Evaluate current model on a specific task.

        Args:
            task_id: Task ID to evaluate on

        Returns:
            Test accuracy on the task
        """
        if task_id >= len(self.tasks):
            raise ValueError(f"Task {task_id} not yet trained")

        # Set task_id in ComposerNode for proper attention routing
        self._set_task_context(task_id)

        task_data = self.tasks[task_id]
        eval_config = {"loss_type": "cross_entropy"}
        eval_key, self.rng_key = jax.random.split(self.rng_key)

        if self.training_mode == "backprop":
            metrics = evaluate_backprop(
                self.params,
                self.structure,
                task_data.test_loader,
                eval_config,
                eval_key,
            )
        else:
            metrics = evaluate_pcn(
                self.params,
                self.structure,
                task_data.test_loader,
                eval_config,
                eval_key,
            )

        return float(metrics.get("accuracy", 0.0))

    def save_checkpoint(self, path: str):
        """
        Save trainer state to checkpoint.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert params to numpy for serialization
        params_np = jax.tree_util.tree_map(np.array, self.params)

        def pack_object(value: Any) -> np.ndarray:
            payload = np.empty((), dtype=object)
            payload[()] = value
            return payload

        opt_state_np = pack_object(jax.tree_util.tree_map(np.array, self.opt_state))

        checkpoint = {
            "params": params_np,
            "opt_state": opt_state_np,
            "global_step": self.global_step,
            "current_task_id": self.current_task_id,
            "summaries": pack_object([s.to_dict() for s in self.summaries]),
            "accuracy_matrix": pack_object(
                {str(k): v for k, v in self._accuracy_matrix.items()}
            ),
            "task_supports": pack_object(
                {str(k): v for k, v in self._task_supports.items()}
            ),
            "support_manager": pack_object(self.support_manager.save_state()),
            "rng_key": np.array(self.rng_key),
            "transweave": pack_object(self.transweave_manager.save_state()),
        }

        np.savez_compressed(str(path), **checkpoint)

        # Also save summaries as JSON for easy inspection
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(
                {
                    "global_step": self.global_step,
                    "summaries": [s.to_dict() for s in self.summaries],
                },
                f,
                indent=2,
            )

    def load_checkpoint(self, path: str):
        """
        Load trainer state from checkpoint.

        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        checkpoint = np.load(str(path), allow_pickle=True)

        # Restore params
        self.params = jax.tree_util.tree_map(jnp.array, checkpoint["params"].item())
        if "opt_state" in checkpoint:
            self.opt_state = jax.tree_util.tree_map(
                jnp.array, checkpoint["opt_state"][()]
            )
        else:
            self.opt_state = self.optimizer.init(self.params)

        # Restore state
        self.global_step = int(checkpoint["global_step"])
        self.current_task_id = int(checkpoint["current_task_id"])
        self.rng_key = jnp.array(checkpoint["rng_key"])

        # Restore accuracy matrix
        self._accuracy_matrix = {
            int(k): v for k, v in checkpoint["accuracy_matrix"][()].items()
        }
        if "task_supports" in checkpoint:
            self._task_supports = {
                int(k): tuple(v) for k, v in checkpoint["task_supports"][()].items()
            }
        else:
            self._task_supports = {}

        # Restore support manager
        self.support_manager.load_state(checkpoint["support_manager"][()])

        # Restore TransWeave state if present
        if "transweave" in checkpoint:
            self.transweave_manager.load_state(checkpoint["transweave"][()])

        if self.current_task_id >= 0:
            self._set_task_context(self.current_task_id)

        # Restore summaries
        self.summaries = []
        for s in checkpoint["summaries"][()]:
            self.summaries.append(
                TaskRunSummary(
                    task_id=s["task_id"],
                    classes=tuple(s["classes"]),
                    train_accuracy=s["train_accuracy"],
                    test_accuracy=s["test_accuracy"],
                    train_loss=s["train_loss"],
                    test_loss=s["test_loss"],
                    train_energy=s.get("train_energy", 0.0),
                    test_energy=s.get("test_energy", 0.0),
                    epochs_trained=s["epochs_trained"],
                    training_time=s["training_time"],
                    support_cols=tuple(s["support_cols"]),
                    epoch_accuracies=s.get("epoch_accuracies", []),
                    epoch_losses=s.get("epoch_losses", []),
                )
            )

    def get_forgetting_metric(self) -> float:
        """
        Compute average forgetting across tasks.

        Forgetting is measured as the drop in accuracy on a task
        from when it was first learned to the current accuracy.

        Returns:
            Average forgetting (0 = no forgetting, 1 = complete forgetting)
        """
        if len(self._accuracy_matrix) < 2:
            return 0.0

        forgetting_sum = 0.0
        count = 0

        for task_id in range(len(self._accuracy_matrix) - 1):
            # Accuracy when first trained
            initial_acc = self._accuracy_matrix.get(task_id, {}).get(task_id, 0.0)
            # Current accuracy
            current_train_id = max(self._accuracy_matrix.keys())
            current_acc = self._accuracy_matrix.get(current_train_id, {}).get(
                task_id, 0.0
            )

            if initial_acc > 0:
                forgetting = max(0, initial_acc - current_acc)
                forgetting_sum += forgetting
                count += 1

        return forgetting_sum / count if count > 0 else 0.0

    def get_forward_transfer_metric(self) -> float:
        """
        Compute average forward transfer.

        Forward transfer measures how much learning previous tasks
        helps with learning new tasks.

        Returns:
            Average forward transfer (positive = beneficial)
        """
        # This would require comparing to a baseline without transfer
        # For now, return 0 as placeholder
        return 0.0

    def summary_dataframe(self):
        """
        Get training summaries as a pandas DataFrame.

        Returns:
            DataFrame with one row per task
        """
        try:
            import pandas as pd

            return pd.DataFrame([s.to_dict() for s in self.summaries])
        except ImportError:
            return [s.to_dict() for s in self.summaries]

    def process_gradients_causal(
        self,
        gradients: Any,
        is_bias_fn: Optional[Callable[[str], bool]] = None,
    ) -> Any:
        """
        Process gradients through per-weight causal system.

        This method can be used in custom training loops to apply
        per-weight causal coding (standard vs SB updates based on
        non-Gaussianity detection).

        Args:
            gradients: JAX pytree of gradients
            is_bias_fn: Optional function to determine if param is bias

        Returns:
            Corrected gradients (same structure as input)

        Example:
            # In custom training loop:
            gradients = jax.grad(loss_fn)(params, batch)
            corrected_grads = trainer.process_gradients_causal(gradients)
            updates, opt_state = optimizer.update(corrected_grads, opt_state)
            params = optax.apply_updates(params, updates)
        """
        if not self.config.per_weight_causal.enable:
            return gradients

        corrected, result = self.per_weight_causal.process_jax_gradients(
            gradients, is_bias_fn
        )
        self._last_per_weight_stats = result.diagnostics

        return corrected

    def get_per_weight_causal_stats(self) -> Dict[str, float]:
        """
        Get current per-weight causal statistics.

        Returns:
            Dict with statistics:
            - mean_kurtosis: Average excess kurtosis across tracked weights
            - max_kurtosis: Maximum excess kurtosis
            - mean_fraction_non_gaussian: Average fraction of non-Gaussian weights
            - num_params_tracked: Number of parameters being tracked
        """
        return self.per_weight_causal.get_stats()

    def get_per_weight_causal_history(self) -> List[Dict[str, float]]:
        """
        Get history of per-weight causal statistics over time.

        Returns:
            List of stats dictionaries over training steps
        """
        return self.per_weight_causal.updater.get_stats_history()

    def get_transweave_stats(self) -> Dict[str, Any]:
        """
        Get TransWeave transfer learning statistics.

        Returns:
            Dict with:
            - composer_tasks_registered: Number of tasks registered for composer transfer
            - total_demotion_candidates: Total neurons flagged for demotion
            - total_promotion_candidates: Total neurons flagged for promotion
            - last_composer_cost: Cost of last composer transfer
        """
        return self.transweave_manager.get_summary_stats()

    def _consolidate_ewc(
        self,
        task_data: TaskData,
        rng_key: jax.Array,
        verbose: bool = False,
    ) -> None:
        """
        Consolidate EWC knowledge after training a task.

        Computes Fisher Information over the task's training data and
        stores the optimal parameters to penalize future deviations.

        Args:
            task_data: The task that was just trained
            rng_key: JAX random key for sampling
            verbose: Whether to print progress
        """
        from fabricpc.training.train_backprop import compute_forward_pass, compute_loss

        if verbose:
            print(f"  Consolidating EWC for task {task_data.task_id}...")

        # Create gradient function for Fisher computation
        # This computes d(loss)/d(params) for each sample
        structure = self.structure

        def gradient_fn(params, batch, subkey):
            """Compute gradients for EWC Fisher estimation."""

            def loss_fn(p):
                state = compute_forward_pass(p, structure, batch, subkey)
                return compute_loss(
                    state, batch["y"], structure.task_map["y"], "cross_entropy"
                )

            loss, grads = jax.value_and_grad(loss_fn)(params)
            # Return (grads, energy, final_state) as expected by EWC
            return grads, float(loss), None

        # Consolidate with EWC manager
        self.ewc_manager.consolidate_task(
            params=self.params,
            gradient_fn=gradient_fn,
            data_loader=task_data.train_loader,
            num_samples=self.config.ewc.fisher_samples,
            rng_key=rng_key,
        )

        if verbose:
            print(f"    Tasks consolidated: {self.ewc_manager.state.num_tasks}")
            print(f"    Fisher max: {self.ewc_manager.state.fisher_max:.6f}")
            print(f"    Fisher mean: {self.ewc_manager.state.fisher_mean:.6f}")
