"""Inverted dropout for FabricPC nodes.

Masks are a pure function of a per-node key carried in ``NodeState.dropout_key``
and a per-site integer, so every call of a node's ``predict`` on one batch —
feedforward initialization, every inference sweep, the local weight-gradient
pass, and the backprop objective — sees the SAME mask. PC therefore settles a
single fixed thinned network per batch, and PC and backprop given the same
``dropout_key`` train the same thinned network.

``dropout_key=None`` (the default everywhere except the training step) turns
dropout off: evaluation, generation, and probes stay deterministic.
"""

import zlib
from typing import Optional

import jax
import jax.numpy as jnp

# Folded into the trainer's per-batch key to get the dropout stream, so the
# latent-initialization stream (which uses the batch key directly) is unchanged.
DROPOUT_STREAM = 0x44524F50  # "DROP"


def training_dropout_key(batch_key: jax.Array) -> jax.Array:
    """The dropout key a training step derives from its batch key.

    The single definition used by the trainer (both algorithms), the
    dashboarding training step and re-settle, and ``RegimeProbe``. Pass it to
    ``initialize_graph_state(..., dropout_key=...)`` to reproduce a training
    step's masks.
    """
    return jax.random.fold_in(batch_key, DROPOUT_STREAM)


def validate_dropout_rate(node_name: str, param: str, rate: float) -> float:
    """Return ``float(rate)``; raise unless ``0 <= rate < 1``."""
    rate = float(rate)
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"Node '{node_name}': {param} must be in [0, 1), got {rate}")
    return rate


def node_dropout_key(dropout_key: jax.Array, node_name: str) -> jax.Array:
    """Per-node key: fold a stable hash of the node name into ``dropout_key``.

    Hashing the name (not the node's index) keeps a node's masks unchanged
    when other nodes are added to the graph. ``zlib.crc32`` is stable across
    processes, unlike Python's salted ``hash``.
    """
    return jax.random.fold_in(dropout_key, zlib.crc32(node_name.encode()))


def dropout(
    x: jnp.ndarray, rate: float, key: Optional[jax.Array], site: int
) -> jnp.ndarray:
    """Inverted dropout: zero with probability ``rate``, scale survivors by
    ``1 / (1 - rate)``. Identity when ``rate == 0`` or ``key is None``.

    ``rate`` is a static Python float (read from node config) and ``key``'s
    presence is static pytree structure, so both branches resolve at trace
    time. ``site`` distinguishes independent dropout sites inside one node.
    """
    if rate == 0.0 or key is None:
        return x
    keep = jax.random.bernoulli(jax.random.fold_in(key, site), 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), jnp.zeros_like(x))


def attach_dropout_keys(state, dropout_key: Optional[jax.Array]):
    """Write each node's ``node_dropout_key`` into its ``NodeState``.

    No-op when ``dropout_key`` is None. Idempotent: the per-node key is a
    pure function of ``(dropout_key, node_name)``.
    """
    if dropout_key is None:
        return state
    nodes = {
        name: ns._replace(dropout_key=node_dropout_key(dropout_key, name))
        for name, ns in state.nodes.items()
    }
    return state._replace(nodes=nodes)
