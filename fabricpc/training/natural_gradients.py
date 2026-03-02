"""
Natural-gradient-style optimizer transforms for predictive coding training.

These transforms use online Fisher information approximations and can be
composed with Optax chains.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class DiagonalNaturalGradientState(NamedTuple):
    """State for diagonal natural-gradient preconditioning."""

    fisher_diag: Any


class LayerwiseNaturalGradientState(NamedTuple):
    """State for layer-wise natural-gradient preconditioning."""

    fisher_scalar: Any


def scale_by_natural_gradient_diag(
    fisher_decay: float = 0.95,
    damping: float = 1e-3,
) -> optax.GradientTransformation:
    """
    Precondition updates with an EMA diagonal Fisher approximation.

    Args:
        fisher_decay: EMA decay for the Fisher estimate in [0, 1).
        damping: Positive damping added to the Fisher diagonal.

    Returns:
        Optax gradient transformation.
    """
    _validate_hparams(fisher_decay, damping)
    one_minus_decay = 1.0 - fisher_decay

    def init_fn(params):
        fisher_diag = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
        return DiagonalNaturalGradientState(fisher_diag=fisher_diag)

    def update_fn(updates, state, params=None):
        del params
        fisher_diag = jax.tree_util.tree_map(
            lambda f, g: fisher_decay * f + one_minus_decay * jnp.square(g),
            state.fisher_diag,
            updates,
        )
        preconditioned_updates = jax.tree_util.tree_map(
            lambda g, f: g / (f + damping),
            updates,
            fisher_diag,
        )
        return preconditioned_updates, DiagonalNaturalGradientState(
            fisher_diag=fisher_diag
        )

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_natural_gradient_layerwise(
    fisher_decay: float = 0.95,
    damping: float = 1e-3,
) -> optax.GradientTransformation:
    """
    Precondition each tensor by one scalar Fisher estimate per leaf.

    This is a cheap layer-wise approximation. It is less expressive than a full
    diagonal preconditioner but can be more stable for large tensors.

    Args:
        fisher_decay: EMA decay for the Fisher estimate in [0, 1).
        damping: Positive damping added to each layer Fisher scalar.

    Returns:
        Optax gradient transformation.
    """
    _validate_hparams(fisher_decay, damping)
    one_minus_decay = 1.0 - fisher_decay

    def init_fn(params):
        fisher_scalar = jax.tree_util.tree_map(
            lambda p: jnp.zeros((), dtype=p.dtype), params
        )
        return LayerwiseNaturalGradientState(fisher_scalar=fisher_scalar)

    def update_fn(updates, state, params=None):
        del params
        fisher_scalar = jax.tree_util.tree_map(
            lambda f, g: fisher_decay * f + one_minus_decay * jnp.mean(jnp.square(g)),
            state.fisher_scalar,
            updates,
        )
        preconditioned_updates = jax.tree_util.tree_map(
            lambda g, f: g / (f + damping),
            updates,
            fisher_scalar,
        )
        return preconditioned_updates, LayerwiseNaturalGradientState(
            fisher_scalar=fisher_scalar
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _validate_hparams(fisher_decay: float, damping: float) -> None:
    """Validate natural-gradient hyperparameters."""
    if not 0.0 <= fisher_decay < 1.0:
        raise ValueError(f"fisher_decay must be in [0, 1). got {fisher_decay}")
    if damping <= 0.0:
        raise ValueError(f"damping must be > 0. got {damping}")
