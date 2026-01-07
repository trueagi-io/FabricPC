"""Core tracker classes for Aim integration.

This module provides the main tracking interface for experiment monitoring
with Aim. All tracking is designed to be optional and lazy-loaded.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.utils.dashboarding._aim_available import get_aim, is_aim_available
from fabricpc.utils.dashboarding.extractors import (
    extract_node_energies,
    flatten_for_distribution,
)


@dataclass
class TrackingConfig:
    """Configuration for what to track and when.

    Attributes:
        track_batch_loss: Track loss at batch level.
        track_batch_energy_per_node: Track per-node energy at batch level.
        track_epoch_loss: Track average loss at epoch level.
        track_epoch_accuracy: Track accuracy at epoch level.
        track_weight_distributions: Track weight distribution histograms.
        track_latent_distributions: Track latent state distributions.
        track_preactivation_distributions: Track pre-activation distributions.
        track_activation_distributions: Track activation (z_mu) distributions.
        track_error_statistics: Track prediction error statistics.
        track_inference_dynamics: Track per-inference-step evolution.
        inference_nodes_to_track: Nodes to track for inference dynamics.
        weight_distribution_every_n_epochs: Frequency for weight tracking.
        latent_distribution_every_n_batches: Frequency for latent tracking.
        experiment_name: Name of the experiment in Aim.
        run_name: Name of this specific run.
    """

    # Batch-level tracking
    track_batch_loss: bool = True
    track_batch_energy_per_node: bool = False

    # Epoch-level tracking
    track_epoch_loss: bool = True
    track_epoch_accuracy: bool = True
    track_weight_distributions: bool = True
    track_latent_distributions: bool = False
    track_preactivation_distributions: bool = False
    track_activation_distributions: bool = False
    track_error_statistics: bool = False

    # Inference dynamics tracking
    track_inference_dynamics: bool = False
    inference_nodes_to_track: List[str] = field(default_factory=list)

    # Frequency controls
    weight_distribution_every_n_epochs: int = 1
    latent_distribution_every_n_batches: int = 100

    # Naming
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None


class AimExperimentTracker:
    """Main tracker class for experiment tracking with Aim.

    Wraps Aim's Run object and provides FabricPC-specific tracking methods.
    All tracking methods are no-ops if Aim is not available.

    Example:
        tracker = AimExperimentTracker(config=TrackingConfig(
            track_weight_distributions=True,
            experiment_name="mnist_pcn"
        ))
        tracker.log_hyperparams(train_config)

        # In training loop:
        tracker.track_batch_loss(loss, epoch=epoch, batch=batch_idx)
        tracker.track_epoch_metrics({"accuracy": acc}, epoch=epoch)
        tracker.track_weight_distributions(params, structure, epoch=epoch)

        tracker.close()
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        aim_run: Any = None,
        repo: Optional[str] = None,
    ):
        """Initialize tracker.

        Args:
            config: TrackingConfig for what/when to track.
            aim_run: Optional existing Aim Run object.
            repo: Optional path to Aim repo (passed to Run if aim_run not provided).
        """
        self.config = config or TrackingConfig()
        self._run = aim_run
        self._repo = repo
        self._initialized = False
        self._global_step = 0

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of Aim Run.

        Returns:
            True if Aim is available and initialized, False otherwise.
        """
        if self._initialized:
            return self._run is not None

        self._initialized = True
        if not is_aim_available():
            self._run = None
            return False

        try:
            aim = get_aim()
            if self._run is None:
                self._run = aim.Run(
                    repo=self._repo,
                    experiment=self.config.experiment_name,
                )
                if self.config.run_name:
                    self._run.name = self.config.run_name
            return True
        except Exception:
            self._run = None
            return False

    @property
    def run(self) -> Any:
        """Access the underlying Aim Run (may be None if not available)."""
        self._ensure_initialized()
        return self._run

    def log_hyperparams(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to Aim.

        Args:
            hparams: Dictionary of hyperparameters.
        """
        if not self._ensure_initialized():
            return
        self._run["hparams"] = hparams

    def log_graph_structure(self, structure: GraphStructure) -> None:
        """Log graph structure information.

        Args:
            structure: GraphStructure object.
        """
        if not self._ensure_initialized():
            return
        self._run["graph"] = {
            "num_nodes": len(structure.nodes),
            "num_edges": len(structure.edges),
            "node_order": list(structure.node_order),
            "task_map": structure.task_map,
            "nodes": {
                name: {
                    "shape": list(info.shape),
                    "type": info.node_type,
                    "in_degree": info.in_degree,
                    "out_degree": info.out_degree,
                }
                for name, info in structure.nodes.items()
            },
        }

    def track_batch_loss(
        self,
        loss: float,
        epoch: int,
        batch: int,
        context: Optional[Dict[str, str]] = None,
    ) -> None:
        """Track batch-level loss.

        Args:
            loss: Loss value.
            epoch: Current epoch.
            batch: Current batch index.
            context: Optional additional context.
        """
        if not self.config.track_batch_loss:
            return
        if not self._ensure_initialized():
            return

        ctx = {"subset": "train", "level": "batch"}
        if context:
            ctx.update(context)

        self._run.track(
            loss,
            name="loss",
            step=self._global_step,
            epoch=epoch,
            context=ctx,
        )
        self._global_step += 1

    def track_batch_energy_per_node(
        self,
        state: GraphState,
        structure: GraphStructure,
        epoch: int,
        batch: int,
    ) -> None:
        """Track per-node energy at batch level.

        Args:
            state: Final GraphState after inference.
            structure: GraphStructure.
            epoch: Current epoch.
            batch: Current batch index.
        """
        if not self.config.track_batch_energy_per_node:
            return
        if not self._ensure_initialized():
            return

        energies = extract_node_energies(state)
        for node_name, energy in energies.items():
            if structure.nodes[node_name].in_degree > 0:
                mean_energy = float(np.mean(energy))
                self._run.track(
                    mean_energy,
                    name="node_energy",
                    step=self._global_step,
                    epoch=epoch,
                    context={"node": node_name, "subset": "train"},
                )

    def track_epoch_metrics(
        self,
        metrics: Dict[str, float],
        epoch: int,
        subset: str = "train",
    ) -> None:
        """Track epoch-level metrics.

        Args:
            metrics: Dictionary of metric name -> value.
            epoch: Current epoch.
            subset: "train" or "val".
        """
        if not self._ensure_initialized():
            return

        for name, value in metrics.items():
            should_track = (
                (name == "loss" and self.config.track_epoch_loss)
                or (name == "accuracy" and self.config.track_epoch_accuracy)
                or (name not in ["loss", "accuracy"])  # Track other metrics always
            )
            if should_track:
                self._run.track(
                    value,
                    name=name,
                    epoch=epoch,
                    context={"subset": subset, "level": "epoch"},
                )

    def track_weight_distributions(
        self,
        params: GraphParams,
        structure: GraphStructure,
        epoch: int,
        nodes: Optional[List[str]] = None,
    ) -> None:
        """Track weight distributions using Aim Distribution.

        Args:
            params: Current GraphParams.
            structure: GraphStructure.
            epoch: Current epoch.
            nodes: Optional list of nodes to track (default: all).
        """
        if not self.config.track_weight_distributions:
            return
        if epoch % self.config.weight_distribution_every_n_epochs != 0:
            return
        if not self._ensure_initialized():
            return

        aim = get_aim()
        nodes = nodes or list(params.nodes.keys())

        for node_name in nodes:
            node_params = params.nodes[node_name]
            for edge_key, weight in node_params.weights.items():
                dist = aim.Distribution(flatten_for_distribution(weight))
                self._run.track(
                    dist,
                    name="weights",
                    epoch=epoch,
                    context={"node": node_name, "edge": edge_key},
                )

            for bias_key, bias in node_params.biases.items():
                dist = aim.Distribution(flatten_for_distribution(bias))
                self._run.track(
                    dist,
                    name="biases",
                    epoch=epoch,
                    context={"node": node_name, "key": bias_key},
                )

    def track_latent_distributions(
        self,
        state: GraphState,
        epoch: int,
        batch: int,
        nodes: Optional[List[str]] = None,
    ) -> None:
        """Track latent state distributions.

        Args:
            state: GraphState.
            epoch: Current epoch.
            batch: Current batch.
            nodes: Optional list of nodes to track.
        """
        if not self.config.track_latent_distributions:
            return
        if batch % self.config.latent_distribution_every_n_batches != 0:
            return
        if not self._ensure_initialized():
            return

        aim = get_aim()
        nodes = nodes or list(state.nodes.keys())

        for node_name in nodes:
            node_state = state.nodes[node_name]

            # z_latent distribution
            dist = aim.Distribution(flatten_for_distribution(node_state.z_latent))
            self._run.track(
                dist,
                name="z_latent",
                step=self._global_step,
                epoch=epoch,
                context={"node": node_name},
            )

            # z_mu distribution (activations)
            if self.config.track_activation_distributions:
                dist = aim.Distribution(flatten_for_distribution(node_state.z_mu))
                self._run.track(
                    dist,
                    name="z_mu",
                    step=self._global_step,
                    epoch=epoch,
                    context={"node": node_name},
                )

            # pre_activation distribution
            if self.config.track_preactivation_distributions:
                dist = aim.Distribution(
                    flatten_for_distribution(node_state.pre_activation)
                )
                self._run.track(
                    dist,
                    name="pre_activation",
                    step=self._global_step,
                    epoch=epoch,
                    context={"node": node_name},
                )

    def track_inference_dynamics(
        self,
        state_history: List[GraphState],
        epoch: int,
        batch: int,
        nodes: Optional[List[str]] = None,
    ) -> None:
        """Track how states evolve over inference steps.

        Args:
            state_history: List of GraphState, one per inference step.
            epoch: Current epoch.
            batch: Current batch.
            nodes: Optional list of nodes to track.
        """
        if not self.config.track_inference_dynamics:
            return
        if not self._ensure_initialized():
            return

        nodes = nodes or self.config.inference_nodes_to_track
        if not nodes and state_history:
            nodes = list(state_history[0].nodes.keys())

        for infer_step, state in enumerate(state_history):
            for node_name in nodes:
                if node_name not in state.nodes:
                    continue
                node_state = state.nodes[node_name]

                # Track energy evolution
                mean_energy = float(np.mean(np.asarray(node_state.energy)))
                self._run.track(
                    mean_energy,
                    name="inference_energy",
                    step=infer_step,
                    context={
                        "node": node_name,
                        "epoch": epoch,
                        "batch": batch,
                    },
                )

                # Track latent gradient norm (convergence indicator)
                grad_norm = float(np.linalg.norm(np.asarray(node_state.latent_grad)))
                self._run.track(
                    grad_norm,
                    name="inference_grad_norm",
                    step=infer_step,
                    context={
                        "node": node_name,
                        "epoch": epoch,
                        "batch": batch,
                    },
                )

                # Track error norm
                error_norm = float(np.linalg.norm(np.asarray(node_state.error)))
                self._run.track(
                    error_norm,
                    name="inference_error_norm",
                    step=infer_step,
                    context={
                        "node": node_name,
                        "epoch": epoch,
                        "batch": batch,
                    },
                )

    def close(self) -> None:
        """Close the Aim run."""
        if self._run is not None:
            self._run.close()
            self._run = None


class StateHistoryCollector:
    """Collects GraphState history during inference for later analysis.

    This is a lightweight container that can be used with the modified
    inference loop to capture per-step dynamics without side effects
    during JIT compilation.

    Example:
        collector = StateHistoryCollector()
        final_state, history = run_inference_with_history(
            params, init_state, clamps, structure, infer_steps, eta_infer
        )
        collector.add_history(history, {"epoch": epoch, "batch": batch})

        # Later, outside JIT:
        tracker.track_inference_dynamics(collector.latest, epoch, batch)
    """

    def __init__(self, max_histories: int = 10):
        """Initialize collector.

        Args:
            max_histories: Maximum number of histories to keep.
        """
        self.max_histories = max_histories
        self._histories: List[List[GraphState]] = []
        self._metadata: List[Dict[str, Any]] = []

    def add_history(
        self,
        history: List[GraphState],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a state history from one training step.

        Args:
            history: List of GraphState from inference loop.
            metadata: Optional metadata (epoch, batch, etc.).
        """
        self._histories.append(history)
        self._metadata.append(metadata or {})

        # Keep only recent histories
        if len(self._histories) > self.max_histories:
            self._histories.pop(0)
            self._metadata.pop(0)

    @property
    def history(self) -> List[List[GraphState]]:
        """Get all collected histories."""
        return self._histories

    @property
    def latest(self) -> Optional[List[GraphState]]:
        """Get the most recent history."""
        return self._histories[-1] if self._histories else None

    @property
    def latest_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata for the most recent history."""
        return self._metadata[-1] if self._metadata else None

    def clear(self) -> None:
        """Clear all collected histories."""
        self._histories.clear()
        self._metadata.clear()
