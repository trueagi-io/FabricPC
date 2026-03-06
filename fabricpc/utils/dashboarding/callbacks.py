"""Callback factories for integration with train_pcn.

These functions create callbacks compatible with the train_pcn function's
iter_callback and epoch_callback parameters.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.utils.dashboarding.trackers import AimExperimentTracker, TrackingConfig


def create_iter_callback(
    tracker: AimExperimentTracker,
    batch_size: Optional[int] = None,
) -> Callable[[int, int, float], float]:
    """Create an iter_callback for train_pcn that tracks batch energy.

    Args:
        tracker: AimExperimentTracker instance.
        batch_size: Optional batch size for normalization.

    Returns:
        Callback function: (epoch_idx, batch_idx, energy) -> normalized_energy
    """

    def iter_callback(epoch_idx: int, batch_idx: int, energy: float) -> float:
        # Normalize energy by batch size if provided
        normalized_energy = energy / batch_size if batch_size else energy
        tracker.track_batch_energy(normalized_energy, epoch=epoch_idx, batch=batch_idx)
        return normalized_energy

    return iter_callback


def create_epoch_callback(
    tracker: AimExperimentTracker,
    structure: GraphStructure,
    eval_fn: Optional[Callable] = None,
    eval_loader: Any = None,
    eval_config: Optional[dict] = None,
) -> Callable[[int, GraphParams, GraphStructure, dict, jax.Array], Optional[dict]]:
    """Create an epoch_callback for train_pcn that tracks epoch metrics and distributions.

    Args:
        tracker: AimExperimentTracker instance.
        structure: GraphStructure.
        eval_fn: Optional evaluation function (e.g., evaluate_pcn).
        eval_loader: Optional evaluation data loader.
        eval_config: Optional evaluation config.

    Returns:
        Callback function matching epoch_callback signature.
    """

    def epoch_callback(
        epoch_idx: int,
        params: GraphParams,
        structure: GraphStructure,
        config: dict,
        rng_key: jax.Array,
    ) -> Optional[dict]:
        # Track weight distributions
        tracker.track_weight_distributions(params, structure, epoch=epoch_idx, batch=0)

        # Optionally run evaluation
        eval_metrics = None
        if eval_fn is not None and eval_loader is not None:
            eval_metrics = eval_fn(
                params, structure, eval_loader, eval_config or config, rng_key
            )
            tracker.track_epoch_metrics(eval_metrics, epoch=epoch_idx, subset="val")

        return eval_metrics

    return epoch_callback


def create_tracking_callbacks(
    config: Optional[TrackingConfig] = None,
    structure: Optional[GraphStructure] = None,
    eval_fn: Optional[Callable] = None,
    eval_loader: Any = None,
    eval_config: Optional[dict] = None,
    hparams: Optional[dict] = None,
    batch_size: Optional[int] = None,
    repo: Optional[str] = None,
) -> Tuple[AimExperimentTracker, Callable, Optional[Callable]]:
    """Create both iter_callback and epoch_callback with a shared tracker.

    This is the recommended way to set up tracking for train_pcn.

    Args:
        config: TrackingConfig (optional, uses defaults if not provided).
        structure: GraphStructure (required for epoch callback).
        eval_fn: Optional evaluation function.
        eval_loader: Optional evaluation data loader.
        eval_config: Optional evaluation config.
        hparams: Optional hyperparameters to log.
        batch_size: Optional batch size for energy normalization.
        repo: Optional path to Aim repository.

    Returns:
        Tuple of (tracker, iter_callback, epoch_callback).

    Example:
        tracker, iter_cb, epoch_cb = create_tracking_callbacks(
            config=TrackingConfig(experiment_name="mnist"),
            structure=structure,
            eval_fn=evaluate_pcn,
            eval_loader=test_loader,
            hparams=train_config,
        )

        trained_params, _, _ = train_pcn(
            params, structure, train_loader, train_config, rng_key,
            iter_callback=iter_cb,
            epoch_callback=epoch_cb,
        )

        tracker.close()
    """
    tracker = AimExperimentTracker(config or TrackingConfig(), repo=repo)

    if hparams:
        tracker.log_hyperparams(hparams)

    if structure:
        tracker.log_graph_structure(structure)

    iter_callback = create_iter_callback(tracker, batch_size=batch_size)
    epoch_callback = (
        create_epoch_callback(tracker, structure, eval_fn, eval_loader, eval_config)
        if structure
        else None
    )

    return tracker, iter_callback, epoch_callback


def create_detailed_iter_callback(
    tracker: AimExperimentTracker,
    structure: GraphStructure,
    batch_size: Optional[int] = None,
) -> Callable[[int, int, float, "GraphState"], float]:
    """Create an iter_callback that also tracks state distributions.

    This callback requires access to the final GraphState, so it must be
    used with a custom training loop rather than train_pcn.

    Args:
        tracker: AimExperimentTracker instance.
        structure: GraphStructure.
        batch_size: Optional batch size for normalization.

    Returns:
        Callback function: (epoch_idx, batch_idx, energy, final_state) -> normalized_energy
    """
    from fabricpc.core.types import GraphState

    def detailed_iter_callback(
        epoch_idx: int,
        batch_idx: int,
        energy: float,
        final_state: GraphState,
    ) -> float:
        normalized_energy = energy / batch_size if batch_size else energy

        # Track batch energy
        tracker.track_batch_energy(normalized_energy, epoch=epoch_idx, batch=batch_idx)

        # Track per-node energy
        tracker.track_batch_energy_per_node(
            final_state, structure, epoch=epoch_idx, batch=batch_idx
        )

        # Track state stats/distributions (if at right batch + infer_step frequency)
        if batch_idx % tracker.config.state_tracking_every_n_batches == 0:
            tracker.track_state(
                final_state, epoch=epoch_idx, batch=batch_idx, infer_step=0
            )

        return normalized_energy

    return detailed_iter_callback
