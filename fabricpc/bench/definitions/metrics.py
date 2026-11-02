"""Scores for rows that are not classifiers, as ``EvalMetric``s.

The autoencoder rows (the faculty advisor's user story) are judged on how
well they rebuild the input and on how many hidden units stay silent.

Sparsity is read from each unit's prediction z_mu, its activation. In a
settled PC state the latent is z_mu plus a small leftover error, so reading
the latent would make every PC unit look slightly active and PC look less
sparse than its activations are. On the forward pass the two are equal.
"""

from typing import Sequence

import jax.numpy as jnp

from fabricpc.training.metrics import EvalMetric

# A ReLU unit is silent when its activation is exactly zero; this leaves room
# for float rounding and nothing more.
SILENT_BELOW = 1e-6


def _reconstruction_mse_fn(state, batch, structure):
    """Per sample, the mean over pixels of (target - output)^2; weight 1."""
    z_mu = state.nodes[structure.task_map["y"]].z_mu
    y = jnp.asarray(batch["y"], dtype=z_mu.dtype).reshape(z_mu.shape)
    axes = tuple(range(1, z_mu.ndim))
    value = jnp.mean((y - z_mu) ** 2, axis=axes)
    return value, jnp.ones_like(value)


reconstruction_mse = EvalMetric(fn=_reconstruction_mse_fn)


def sparsity(node_names: Sequence[str], silent_below: float = SILENT_BELOW):
    """Fraction of the named nodes' units with |activation| < silent_below,
    per sample, pooled over the nodes."""
    names = tuple(node_names)

    def fn(state, batch, structure):
        silent = 0.0
        total = 0
        for name in names:
            z = state.nodes[name].z_mu
            axes = tuple(range(1, z.ndim))
            silent = silent + jnp.sum(jnp.abs(z) < silent_below, axis=axes)
            total += int(jnp.size(z) // z.shape[0])
        value = silent / total
        return value, jnp.ones_like(value)

    return EvalMetric(fn=fn)
