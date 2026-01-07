"""Aim experiment tracking integration for FabricPC.

This package provides experiment tracking capabilities using Aim.
All components are designed for lazy loading - Aim is only imported
when actually used.

Example:
    from fabricpc.utils.dashboarding import (
        AimExperimentTracker,
        TrackingConfig,
        create_tracking_callbacks,
    )

    tracker, iter_cb, epoch_cb = create_tracking_callbacks(
        config=TrackingConfig(experiment_name="my_experiment"),
        structure=structure,
        hparams=train_config,
    )

    trained_params, _, _ = train_pcn(
        params, structure, train_loader, train_config, rng_key,
        iter_callback=iter_cb,
        epoch_callback=epoch_cb,
    )

    tracker.close()
"""

from fabricpc.utils.dashboarding._aim_available import (
    get_aim,
    is_aim_available,
    require_aim,
)
from fabricpc.utils.dashboarding.callbacks import (
    create_detailed_iter_callback,
    create_epoch_callback,
    create_iter_callback,
    create_tracking_callbacks,
)
from fabricpc.utils.dashboarding.extractors import (
    extract_activation_statistics,
    extract_all_distributions,
    extract_bias_statistics,
    extract_error_statistics,
    extract_latent_grad_statistics,
    extract_latent_statistics,
    extract_node_energies,
    extract_preactivation_statistics,
    extract_total_energy,
    extract_weight_statistics,
    flatten_for_distribution,
)
from fabricpc.utils.dashboarding.inference_tracking import (
    extract_history_for_plotting,
    run_inference_with_full_history,
    run_inference_with_history,
    summarize_inference_convergence,
    train_step_with_history,
)
from fabricpc.utils.dashboarding.trackers import (
    AimExperimentTracker,
    StateHistoryCollector,
    TrackingConfig,
)

__all__ = [
    # Availability check
    "is_aim_available",
    "get_aim",
    "require_aim",
    # Core trackers
    "AimExperimentTracker",
    "TrackingConfig",
    "StateHistoryCollector",
    # Callbacks
    "create_iter_callback",
    "create_epoch_callback",
    "create_tracking_callbacks",
    "create_detailed_iter_callback",
    # Extractors
    "extract_node_energies",
    "extract_total_energy",
    "extract_latent_statistics",
    "extract_preactivation_statistics",
    "extract_activation_statistics",
    "extract_weight_statistics",
    "extract_bias_statistics",
    "extract_error_statistics",
    "extract_latent_grad_statistics",
    "extract_all_distributions",
    "flatten_for_distribution",
    # Inference tracking
    "run_inference_with_history",
    "run_inference_with_full_history",
    "train_step_with_history",
    "extract_history_for_plotting",
    "summarize_inference_convergence",
]
