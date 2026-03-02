"""
Optimizer utilities using Optax.
"""

import optax

from fabricpc.training.natural_gradients import (
    scale_by_natural_gradient_diag,
    scale_by_natural_gradient_layerwise,
)


def create_optimizer(config: dict) -> optax.GradientTransformation:
    """
    Create Optax optimizer from configuration.

    Args:
        config: Optimizer configuration with keys:
            - type: "adam", "sgd", "adamw", "ngd_diag", "ngd_layerwise"
            - lr: learning rate
            - weight_decay: optional weight decay (L2 regularization)
            - Other optimizer-specific parameters

    Returns:
        Optax optimizer

    Example:
        >>> config = {"type": "adam", "lr": 1e-3, "weight_decay": 1e-4}
        >>> optimizer = create_optimizer(config)
    """
    opt_type = config.get("type", "adam").lower()
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)

    # Create base optimizer
    if opt_type == "adam":
        optimizer = optax.adam(lr, **_filter_kwargs(config, ["b1", "b2", "eps"]))
    elif opt_type == "adamw":
        optimizer = optax.adamw(
            lr, weight_decay=weight_decay, **_filter_kwargs(config, ["b1", "b2", "eps"])
        )
    elif opt_type == "sgd":
        momentum = config.get("momentum", 0.0)
        optimizer = optax.sgd(lr, momentum=momentum)
    elif opt_type == "ngd_diag":
        fisher_decay = config.get("fisher_decay", 0.95)
        damping = config.get("damping", 1e-3)
        optimizer = optax.chain(
            scale_by_natural_gradient_diag(fisher_decay=fisher_decay, damping=damping),
            optax.scale(-lr),
        )
    elif opt_type in ("ngd_layerwise", "ngd_layer"):
        fisher_decay = config.get("fisher_decay", 0.95)
        damping = config.get("damping", 1e-3)
        optimizer = optax.chain(
            scale_by_natural_gradient_layerwise(
                fisher_decay=fisher_decay, damping=damping
            ),
            optax.scale(-lr),
        )
    else:
        raise ValueError(f"unknown optimizer type: {opt_type}")

    # Add weight decay if not using AdamW (which has built-in weight decay)
    if opt_type != "adamw" and weight_decay > 0:
        optimizer = optax.chain(optax.add_decayed_weights(weight_decay), optimizer)

    return optimizer


def _filter_kwargs(config: dict, keys: list) -> dict:
    """Extract specific keys from config if they exist."""
    return {k: config[k] for k in keys if k in config}
