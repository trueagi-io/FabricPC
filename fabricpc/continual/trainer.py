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
