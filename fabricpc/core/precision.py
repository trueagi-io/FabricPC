"""
Diagonal precision estimation for predictive-coding graphs.

In a Gaussian predictive-coding network the energy of layer ``l`` is

    E_l = (1/2) * (z_l - mu_l)^T Pi_l (z_l - mu_l)

where ``Pi_l = Sigma_l^{-1}`` is the precision (inverse covariance) of the
prediction error ``e_l = z_l - mu_l``. With ``Pi_l = I`` (the framework default)
every feature contributes to the energy in proportion to its raw error variance,
so in a deep residual network the channels/layers whose errors happen to be large
dominate the global energy and drown out the rest. This is one of the mechanisms
behind depth degradation in deep PC.

Setting ``Pi_l`` to (an estimate of) the inverse error covariance whitens the
errors: every feature is rescaled to unit precision-weighted variance, so each
channel keeps an equal voice. A **diagonal** ``Pi_l`` (this module) is the cheap,
per-feature version of that whitening — the predictive-coding analogue of a
diagonal natural-gradient (NGD) preconditioner.

``probe_residual_precision`` runs a single clamped inference pass over one batch,
measures the per-channel residual variance at every internal Gaussian-energy node,
and returns ``Pi_l = 1 / (Var(e_l) + eps)`` as a per-node vector. The estimate is
frozen (static config); it does not adapt during training. Feed the returned map
back into the graph builder so each node is constructed with
``GaussianEnergy(precision=Pi_l)``.
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import run_inference
from fabricpc.core.types import GraphParams, GraphStructure


def probe_residual_precision(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    *,
    eps: float = 1e-3,
    clip: Optional[Tuple[float, float]] = (0.1, 10.0),
    normalize: str = "mean",
    channel_axis: int = -1,
) -> Dict[str, np.ndarray]:
    """
    Estimate a per-node diagonal precision from one clamped inference pass.

    The pass clamps the same task nodes that PC *training* clamps (every entry
    of ``batch`` whose key is in ``structure.task_map`` — typically both the
    input ``x`` and the target ``y``), so the measured residuals are the ones
    the network actually learns from. For each internal node carrying a
    ``GaussianEnergy`` the per-channel error variance is measured and inverted.

    Args:
        params: Model parameters (precision is independent of these — any
            initialized params for ``structure`` are fine).
        structure: The graph to probe. Must already be finalized (built).
        batch: A batch of data with task-named keys (e.g. ``{"x": ..., "y": ...}``).
        rng_key: PRNG key for latent-state initialization.
        eps: Floor added to the variance before inversion (numerical stability
            and a cap on how large precision can get for near-dead channels).
        clip: Optional ``(lo, hi)`` clamp applied to the precision *after*
            normalization, to bound the effect of extreme channels. ``None``
            disables clipping.
        normalize: Per-node normalization of the precision vector:
            - ``"mean"`` (default): divide by the mean so the average precision
              is 1.0. Preserves the layer's overall energy scale (so the learning
              rate need not be retuned) while redistributing weight across
              channels — this isolates the *diagonal* (whitening) effect from any
              change in cross-layer balance, which remains muPC's job.
            - ``"none"``: return the raw ``1/(var+eps)`` (changes energy scale).
        channel_axis: Axis treated as the feature/channel axis (default ``-1``).
            Variance is reduced over every other axis.

    Returns:
        Dict mapping node name -> diagonal precision vector (``np.float32``,
        shape ``(C,)``) for every internal (``in_degree > 0``) Gaussian-energy
        node. Nodes with non-Gaussian energy (e.g. the softmax/cross-entropy
        output) and terminal input nodes are omitted; build those with their
        default energy.
    """
    if normalize not in ("mean", "none"):
        raise ValueError(f"normalize must be 'mean' or 'none', got {normalize!r}")

    batch_size = next(iter(batch.values())).shape[0]

    # Clamp every task node present in the batch (training-style: x and y).
    clamps = {
        structure.task_map[task_name]: value
        for task_name, value in batch.items()
        if task_name in structure.task_map
    }

    # Run inference to equilibrium. JIT-compiled (same calls train uses) so the
    # one-off probe over a full batch stays cheap; structure/rng_key are captured
    # as constants, params/clamps are traced.
    from fabricpc.graph_initialization.state_initializer import initialize_graph_state

    def _infer(p, clamp_arrays):
        init_state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamp_arrays, params=p
        )
        return run_inference(p, init_state, clamp_arrays, structure)

    final_state = jax.jit(_infer)(params, clamps)

    precision_map: Dict[str, np.ndarray] = {}
    for node_name, node in structure.nodes.items():
        info = node.node_info
        if info.in_degree == 0:
            continue
        if not isinstance(info.energy, GaussianEnergy):
            continue

        error = final_state.nodes[node_name].error  # (batch, *spatial, C)
        axis = channel_axis % error.ndim
        reduce_axes = tuple(i for i in range(error.ndim) if i != axis)

        var = jnp.var(error, axis=reduce_axes)  # (C,)
        precision = 1.0 / (var + eps)

        if normalize == "mean":
            precision = precision / jnp.mean(precision)
        if clip is not None:
            precision = jnp.clip(precision, clip[0], clip[1])

        precision_map[node_name] = np.asarray(precision, dtype=np.float32)

    return precision_map
