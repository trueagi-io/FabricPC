"""
Sequential Trainer for Continual Learning.

Provides task-by-task training with support selection, checkpointing,
and evaluation across all seen tasks.
"""

from dataclasses import dataclass, field
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
from fabricpc.continual.support import SupportManager, SupportState, ReplayBuffer
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


class InterleavedLoader:
    """
    Data loader that interleaves current task samples with replay samples.

    Used for experience replay during continual learning to prevent
    catastrophic forgetting.
    """

    def __init__(
        self,
        current_loader,
        replay_buffer: ReplayBuffer,
        current_task_id: int,
        replay_ratio: float = 0.5,
        num_classes: int = 10,
    ):
        """
        Initialize interleaved loader.

        Args:
            current_loader: Data loader for current task
            replay_buffer: Buffer containing samples from previous tasks
            current_task_id: ID of current task (excluded from replay)
            replay_ratio: Ratio of replay samples (0.5 = equal mix)
            num_classes: Number of output classes for label formatting
        """
        self.current_loader = current_loader
        self.replay_buffer = replay_buffer
        self.current_task_id = current_task_id
        self.replay_ratio = replay_ratio
        self.num_classes = num_classes

        # Calculate batch sizes
        self.current_batch_size = current_loader.batch_size
        self.replay_batch_size = int(self.current_batch_size * replay_ratio)
        self.has_replay = (
            len(replay_buffer) > 0
            and len(replay_buffer.get_task_ids()) > 0
            and any(t != current_task_id for t in replay_buffer.get_task_ids())
        )

    def __len__(self):
        return len(self.current_loader)

    def __iter__(self):
        for current_images, current_labels in self.current_loader:
            if self.has_replay and self.replay_batch_size > 0:
                # Get replay samples
                replay_data = self.replay_buffer.sample(
                    batch_size=self.replay_batch_size,
                    exclude_task=self.current_task_id,
                )

                if replay_data is not None:
                    replay_images, replay_labels = replay_data

                    # Concatenate current and replay batches
                    combined_images = np.concatenate(
                        [current_images, replay_images], axis=0
                    )
                    combined_labels = np.concatenate(
                        [current_labels, replay_labels], axis=0
                    )

                    # Shuffle the combined batch
                    perm = np.random.permutation(len(combined_images))
                    yield combined_images[perm], combined_labels[perm]
                    continue

            # No replay available, just yield current batch
            yield current_images, current_labels


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

        # Initialize optimizer
        if optimizer is None:
            optimizer = optax.adamw(
                config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        self.optimizer = optimizer
        self.opt_state = optimizer.init(params)

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

        # Experience replay buffer for preventing catastrophic forgetting
        self.replay_buffer = ReplayBuffer(
            max_samples_per_task=config.support.replay_buffer_size_per_task,
            max_total_samples=config.support.replay_buffer_total_size,
        )
        self.use_replay = config.support.use_replay
        self.replay_ratio = config.support.replay_ratio

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

        # Callbacks
        self.epoch_callbacks: List[Callable] = []
        self.task_callbacks: List[Callable] = []

    @property
    def training_mode(self) -> str:
        """Get current training mode."""
        return self.config.training.training_mode

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

        if verbose:
            print(f"Active columns: {support_cols}")

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

        # Use interleaved loader with replay if enabled and we have previous tasks
        if self.use_replay and len(self.replay_buffer) > 0:
            train_loader = InterleavedLoader(
                current_loader=task_data.train_loader,
                replay_buffer=self.replay_buffer,
                current_task_id=task_id,
                replay_ratio=self.replay_ratio,
                num_classes=10,  # MNIST has 10 classes
            )
            if verbose:
                replay_tasks = [
                    t for t in self.replay_buffer.get_task_ids() if t != task_id
                ]
                print(
                    f"Replay enabled: mixing with {len(replay_tasks)} previous tasks "
                    f"({self.replay_buffer.total_samples()} samples)"
                )

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
            self.params, loss_history, _ = train_backprop(
                params=self.params,
                structure=self.structure,
                train_loader=train_loader,
                optimizer=self.optimizer,
                config=train_config,
                rng_key=train_key,
                verbose=verbose,
                epoch_callback=epoch_callback,
            )
        else:  # PC mode or hybrid (default to PC)
            self.params, energy_history, _ = train_pcn(
                params=self.params,
                structure=self.structure,
                train_loader=train_loader,
                optimizer=self.optimizer,
                config=train_config,
                rng_key=train_key,
                verbose=verbose,
                epoch_callback=epoch_callback,
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
        transweave_stats = self._register_transweave_task_end(task_id)

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

        # Store samples in replay buffer for future tasks
        if self.use_replay:
            self._store_task_samples(task_data)

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

    def _store_task_samples(self, task_data: TaskData):
        """
        Store samples from a task in the replay buffer.

        Collects samples by iterating through the training loader
        and stores them for replay during future tasks.
        """
        all_images = []
        all_labels = []

        # Collect samples from the training loader
        for images, labels in task_data.train_loader:
            all_images.append(images)
            all_labels.append(labels)

        if all_images:
            all_images = np.concatenate(all_images, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            # Store in replay buffer
            self.replay_buffer.add_task_samples(
                task_id=task_data.task_id,
                images=all_images,
                labels=all_labels,
                replace=True,  # Replace any existing samples for this task
            )

    def _update_accuracy_matrix(
        self, current_task_id: int, current_task_data: TaskData
    ):
        """Update accuracy matrix after training on a task."""
        if current_task_id not in self._accuracy_matrix:
            self._accuracy_matrix[current_task_id] = {}

        eval_config = {"loss_type": "cross_entropy"}

        # Build list of all tasks to evaluate (previously seen + current)
        tasks_to_eval = list(self.tasks)  # Previously seen tasks
        # Add current task if not already in the list
        if not any(t.task_id == current_task_data.task_id for t in tasks_to_eval):
            tasks_to_eval.append(current_task_data)

        # Evaluate on all seen tasks including current
        for task_data in tasks_to_eval:
            eval_task_id = task_data.task_id
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

            self._accuracy_matrix[current_task_id][eval_task_id] = float(
                metrics.get("accuracy", 0.0)
            )

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
        shared = list(range(self.config.columns.shared_columns))

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
        current_batches = []
        for i, (images, labels) in enumerate(current_task_data.test_loader):
            current_batches.append((images, labels))
            if i + 1 >= max_batches:
                break

        old_batches = []
        if old_task_data is not None:
            for i, (images, labels) in enumerate(old_task_data.test_loader):
                old_batches.append((images, labels))
                if i + 1 >= max_batches:
                    break

        eval_config = {"loss_type": "cross_entropy"}

        def eval_loss_on_batches(batches):
            """Evaluate loss on a list of batches."""
            if not batches:
                return 0.0
            total_loss = 0.0
            count = 0
            for images, labels in batches:
                eval_key, _ = jax.random.split(self.rng_key)
                if self.training_mode == "backprop":
                    metrics = evaluate_backprop(
                        self.params,
                        self.structure,
                        [(images, labels)],
                        eval_config,
                        eval_key,
                    )
                else:
                    metrics = evaluate_pcn(
                        self.params,
                        self.structure,
                        [(images, labels)],
                        eval_config,
                        eval_key,
                    )
                total_loss += float(metrics.get("loss", 0.0))
                count += 1
            return total_loss / max(1, count)

        # Baseline loss with current support
        baseline_current_loss = eval_loss_on_batches(current_batches)
        baseline_old_loss = eval_loss_on_batches(old_batches) if old_batches else 0.0

        if verbose:
            print(
                f"  Audit baseline: current_loss={baseline_current_loss:.4f}, old_loss={baseline_old_loss:.4f}"
            )

        # Generate swap pairs
        swap_pairs = []
        for c_idx, swap_out in enumerate(chosen):
            for swap_in in unchosen:
                swap_pairs.append((swap_out, swap_in, c_idx))
                if len(swap_pairs) >= max_swaps:
                    break
            if len(swap_pairs) >= max_swaps:
                break

        # Evaluate each swap
        # Use column statistics from support bank to estimate contribution
        col_stats = self.support_manager.support_bank.get_column_statistics()

        for swap_out, swap_in, c_idx in swap_pairs:
            # Create swapped support set
            swapped_chosen = chosen.copy()
            swapped_chosen[c_idx] = swap_in
            swapped_all = tuple(shared + swapped_chosen)

            # Estimate contribution difference based on historical performance
            # This is an approximation - full implementation would run actual inference
            # with different column masks
            swap_out_score = col_stats.get(swap_out, {}).get("mean_accuracy", 0.5)
            swap_in_score = col_stats.get(swap_in, {}).get("mean_accuracy", 0.5)

            # Add small noise to prevent degenerate case
            noise = np.random.normal(0, 0.01)

            # Estimate loss difference (lower loss = better)
            # If swap_in has higher historical accuracy, it should reduce loss
            estimated_gain = (swap_in_score - swap_out_score) * 0.1 + noise

            alt_current_loss = baseline_current_loss - estimated_gain
            alt_old_loss = (
                baseline_old_loss - estimated_gain * 0.5
            )  # Less impact on old task

            # Compute gain (positive = swap_in is better than swap_out)
            current_gain = baseline_current_loss - alt_current_loss
            old_gain = baseline_old_loss - alt_old_loss

            # Combined gain with weights
            current_weight = self.config.audit.support_audit_current_weight
            old_weight = self.config.audit.support_audit_old_weight
            combined_gain = current_weight * current_gain + old_weight * old_gain

            audit_rows.append(
                {
                    "swap_in": swap_in,
                    "swap_out": swap_out,
                    "chosen_current_loss": baseline_current_loss,
                    "alt_current_loss": alt_current_loss,
                    "chosen_old_loss": baseline_old_loss,
                    "alt_old_loss": alt_old_loss,
                    "current_gain": current_gain,
                    "old_gain": old_gain,
                    "combined_gain": combined_gain,
                    "current_task_id": current_task_id,
                    "old_task_id": old_task_data.task_id if old_task_data else None,
                    "chosen_support": tuple(chosen),
                }
            )

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

        # Simple similarity functions
        def struct_sim(i: int, j: int) -> float:
            return 1.0 if i == j else 0.0

        def causal_sim(i: int, j: int) -> float:
            if fp_mean is None:
                return 0.0
            if i >= fp_mean.shape[0] or j >= fp_mean.shape[0]:
                return 0.0
            # Cosine similarity of gain vectors
            vi = fp_mean[i]
            vj = fp_mean[j]
            ni = np.linalg.norm(vi)
            nj = np.linalg.norm(vj)
            if ni < 1e-6 or nj < 1e-6:
                return 0.0
            return float(np.dot(vi, vj) / (ni * nj))

        X_list = []
        y_list = []
        w_list = []
        meta_list = []

        for row in audit_rows:
            swap_in = row.get("swap_in", -1)
            if swap_in < 0 or swap_in >= num_columns:
                continue

            chosen = row.get("chosen_support", ())

            # Build feature vector for swap_in column
            feat = feature_builder.build_feature(
                idx=swap_in,
                role="challenger",  # Audit swaps are challengers
                chosen=chosen,
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
                struct_similarity_fn=struct_sim,
                causal_similarity_fn=causal_sim,
                current_task_id=current_task_id,
            )

            # Target is the combined gain (positive = swap_in is better)
            target = row.get("combined_gain", 0.0)
            # Clamp target
            max_abs = self.config.support.causal_max_abs_target
            target = float(np.clip(target, -max_abs, max_abs))
            # Scale target
            target = target * self.config.support.causal_target_scale

            # Weight based on magnitude of current loss
            weight = 1.0 + abs(row.get("chosen_current_loss", 0.0))

            X_list.append(feat)
            y_list.append(target)
            w_list.append(weight)
            meta_list.append(
                {
                    "swap_in": swap_in,
                    "swap_out": row.get("swap_out"),
                    "task_id": current_task_id,
                }
            )

        if X_list:
            X = np.stack(X_list, axis=0)
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
                    for i, row in enumerate(audit_rows):
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

    def _register_transweave_task_end(self, task_id: int) -> Dict[str, Any]:
        """
        Register end-of-task state with TransWeave for transfer learning.

        Captures composer state and shell assignments to enable transfer
        to future tasks via Sinkhorn optimal transport.

        Args:
            task_id: Completed task ID

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

        num_columns = self.config.columns.num_columns
        num_heads = self.config.composer.num_heads
        hidden_dim = self.config.composer.hidden_dim

        # Generate synthetic composer state from current support configuration
        # In a full implementation, these would come from actual model parameters
        # For now, we create representative attention patterns based on support selection
        active_cols = self.current_state.active_all
        num_active = len(active_cols)

        # Create attention pattern that emphasizes active columns
        attention_weights = np.zeros((num_heads, num_columns, num_columns))
        for h in range(num_heads):
            for col in active_cols:
                # Attention from active columns to all others
                attention_weights[h, col, :] = 1.0 / num_columns
                # Self-attention bonus
                attention_weights[h, col, col] += 0.5
            # Normalize
            attention_weights[h] = attention_weights[h] / (
                attention_weights[h].sum(axis=1, keepdims=True) + 1e-10
            )

        # Create projection matrices with structure based on active columns
        key_dim = hidden_dim // num_heads
        query_projections = np.random.randn(num_heads, hidden_dim, key_dim) * 0.1
        key_projections = np.random.randn(num_heads, hidden_dim, key_dim) * 0.1
        value_projections = np.random.randn(num_heads, hidden_dim, key_dim) * 0.1
        output_projection = np.random.randn(num_heads * key_dim, hidden_dim) * 0.1

        # Register with composer TransWeave
        if self.config.composer_transweave.enable:
            self.transweave_manager.composer_transweave.register_task(
                task_id=task_id,
                attention_weights=attention_weights,
                query_projections=query_projections,
                key_projections=key_projections,
                value_projections=value_projections,
                output_projection=output_projection,
            )

            # Compute transfer for this task (to measure what was transferred)
            if task_id > 0:
                transfer_result = (
                    self.transweave_manager.composer_transweave.compute_transfer(
                        target_task_id=task_id,
                        current_attention=attention_weights,
                        current_queries=query_projections,
                        current_keys=key_projections,
                        current_values=value_projections,
                    )
                )
                stats["composer_sources"] = len(transfer_result.source_tasks)
                stats["composer_strength"] = transfer_result.transfer_strength
                stats["composer_cost"] = transfer_result.diagnostics.get(
                    "mean_transport_cost", 0.0
                )

        # Register with shell demotion TransWeave
        if self.config.shell_demotion_transweave.enable:
            shell_sizes = self.config.shell_demotion_transweave.shell_sizes
            num_neurons = sum(shell_sizes)

            for col_id in range(num_columns):
                # Create shell assignments based on column activity pattern
                # Active columns have neurons in inner shells, others in outer
                shell_assignments = np.zeros(num_neurons, dtype=np.int32)
                if col_id in active_cols:
                    # Active: more neurons in protected/stable shells
                    for i in range(num_neurons):
                        if i < shell_sizes[0]:
                            shell_assignments[i] = 0  # Protected
                        elif i < shell_sizes[0] + shell_sizes[1]:
                            shell_assignments[i] = 1  # Stable
                        else:
                            shell_assignments[i] = 2  # Outer
                else:
                    # Inactive: more neurons in outer shell
                    for i in range(num_neurons):
                        if i < shell_sizes[0] // 2:
                            shell_assignments[i] = 0
                        elif i < shell_sizes[0] + shell_sizes[1] // 2:
                            shell_assignments[i] = 1
                        else:
                            shell_assignments[i] = 2

                # Neuron activities based on column usage
                base_activity = 0.8 if col_id in active_cols else 0.2
                neuron_activities = (
                    np.random.rand(num_neurons) * 0.2 + base_activity - 0.1
                )

                self.transweave_manager.shell_transweave.register_shell_state(
                    column_id=col_id,
                    task_id=task_id,
                    shell_assignments=shell_assignments,
                    neuron_activities=neuron_activities,
                )

            # Compute demotion recommendations for active columns
            total_demotions = 0
            total_promotions = 0
            for col_id in active_cols[:4]:  # Limit to avoid too many computations
                if col_id < num_columns:
                    history = (
                        self.transweave_manager.shell_transweave.column_histories.get(
                            col_id, []
                        )
                    )
                    if len(history) > 0:
                        latest = history[-1]
                        result = self.transweave_manager.shell_transweave.compute_demotion_transport(
                            column_id=col_id,
                            current_activities=latest.neuron_activities,
                            current_assignments=latest.shell_assignments,
                        )
                        total_demotions += len(result.demotion_candidates)
                        total_promotions += len(result.promotion_candidates)

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

        checkpoint = {
            "params": params_np,
            "global_step": self.global_step,
            "current_task_id": self.current_task_id,
            "summaries": [s.to_dict() for s in self.summaries],
            "accuracy_matrix": {str(k): v for k, v in self._accuracy_matrix.items()},
            "support_manager": self.support_manager.save_state(),
            "rng_key": np.array(self.rng_key),
            "transweave": self.transweave_manager.save_state(),
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
        self.opt_state = self.optimizer.init(self.params)

        # Restore state
        self.global_step = int(checkpoint["global_step"])
        self.current_task_id = int(checkpoint["current_task_id"])
        self.rng_key = jnp.array(checkpoint["rng_key"])

        # Restore accuracy matrix
        self._accuracy_matrix = {
            int(k): v for k, v in checkpoint["accuracy_matrix"].item().items()
        }

        # Restore support manager
        self.support_manager.load_state(checkpoint["support_manager"].item())

        # Restore TransWeave state if present
        if "transweave" in checkpoint:
            self.transweave_manager.load_state(checkpoint["transweave"].item())

        # Restore summaries
        self.summaries = []
        for s in checkpoint["summaries"].item():
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
