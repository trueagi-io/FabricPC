"""
Natural-gradient (NGD) trainer with ONLINE diagonal precision for predictive coding.

This implements the precision-as-natural-gradient scheme of Ofner et al. 2021
("Predictive Coding, Precision and Natural Gradients", arXiv:2111.06942) for the
FabricPC graph framework:

  * Per-node DIAGONAL precision Pi_l (inverse error variance) is learned ONLINE:
    after each batch's inference reaches equilibrium, the per-channel error variance
    is folded into a running EMA  var_l <- (1-lam)*var_l + lam*mean(e_l^2)  and the
    precision used by the next step is Pi_l = 1/(var_l + eps)  (eq. 8, diagonal form;
    the EMA is the closed-form fixed point of the precision sub-problem that the
    +ln 2*pi*Sigma term in the free energy enforces -- so no gradient on Sigma needed).

  * That precision is fed back into the ENERGY via NodeState.precision, so it scales
    BOTH the inference latent updates AND the local weight gradients (which both route
    through energy_functional). Because the energy is already precision-weighted, the
    weight gradient carries Pi_l implicitly (the Explore-mapped fact that muPC's
    weight_grad_scale is 1.0 means there is no double application).

  * With a plain SGD optimizer the precision therefore acts as the per-channel adaptive
    learning rate -- this is the natural-gradient update (eqs. 10-11), the paper's
    "PC-SGD". Adam would re-normalize each gradient by its own RMS and largely cancel
    the precision scaling, which is why NGD uses SGD here, not Adam.

Precision is carried as a traced per-node dict through the jitted step, so it updates
every iteration with NO recompilation. The verified `train_pcn` path is untouched;
this is a separate, opt-in trainer.

Only internal nodes (in_degree > 0) carrying a GaussianEnergy get a learned precision;
terminal inputs and non-Gaussian outputs (e.g. softmax/cross-entropy) are left alone.
"""

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax

from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import run_inference
from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.graph_initialization.state_initializer import initialize_graph_state

# ---------------------------------------------------------------------------
# Which nodes get a learned precision
# ---------------------------------------------------------------------------


def precision_node_names(structure: GraphStructure) -> List[str]:
    """Internal (in_degree>0) nodes whose energy is Gaussian -> get a learned Pi."""
    names = []
    for name, node in structure.nodes.items():
        info = node.node_info
        if info.in_degree > 0 and isinstance(info.energy, GaussianEnergy):
            names.append(name)
    return names


def init_precision_vars(
    structure: GraphStructure, names: List[str]
) -> Dict[str, jnp.ndarray]:
    """Initial running variance = 1.0 per channel (so Pi=1: warm-starts at baseline)."""
    var = {}
    for name in names:
        channels = structure.nodes[name].node_info.shape[-1]
        var[name] = jnp.ones((channels,), dtype=jnp.float32)
    return var


# ---------------------------------------------------------------------------
# var (running EMA) <-> precision map
# ---------------------------------------------------------------------------


def _vars_to_precision(
    var: Dict[str, jnp.ndarray], eps: float, clip: Tuple[float, float], normalize: bool
) -> Dict[str, jnp.ndarray]:
    """Pi = 1/(var+eps), per-layer mean-normalized (optional) and clipped."""
    pmap = {}
    for name, v in var.items():
        pi = 1.0 / (v + eps)
        if normalize:
            pi = pi / jnp.mean(pi)
        if clip is not None:
            pi = jnp.clip(pi, clip[0], clip[1])
            if normalize:
                # Re-normalize AFTER clipping so the per-layer mean precision stays 1.0.
                # Clipping dead/extreme channels would otherwise rescale the effective
                # per-layer SGD learning rate (the "mean Pi=1" NGD property). Any tiny
                # post-renorm excursion past the clip bounds is acceptable.
                pi = pi / jnp.mean(pi)
        pmap[name] = pi
    return pmap


def _update_vars(
    var: Dict[str, jnp.ndarray], final_state, names: List[str], lam: float
) -> Dict[str, jnp.ndarray]:
    """EMA update of per-channel error variance from the inference equilibrium."""
    new = {}
    for name in names:
        err = final_state.nodes[name].error  # (batch, *spatial, C)
        # Channel-last assumption: precision is per LAST-axis channel. Holds for all
        # resnet18 nodes (conv NHWC, avgpool (B,C), skip NHWC). A node whose feature
        # axis is not last would need the channel_axis plumbing that
        # probe_residual_precision exposes.
        reduce_axes = tuple(range(err.ndim - 1))  # all but channel (last) axis
        # mean(err^2), NOT jnp.var: Sigma is the error SECOND MOMENT (zero-mean error at
        # the PC equilibrium per Ofner et al. eq. 8). This is intentionally the same
        # quantity diag_probe's precision targets, estimated without mean subtraction.
        batch_var = jnp.mean(err**2, axis=reduce_axes)  # (C,)
        new[name] = (1.0 - lam) * var[name] + lam * batch_var
    return new


# ---------------------------------------------------------------------------
# One NGD training step (jitted by the caller's closure)
# ---------------------------------------------------------------------------


def ngd_train_step(
    params: GraphParams,
    opt_state: optax.OptState,
    var: Dict[str, jnp.ndarray],
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    names: List[str],
    rng_key: jax.Array,
    *,
    lam: float,
    eps: float,
    clip: Tuple[float, float],
    normalize: bool,
):
    """
    Inference (with current online precision) -> local grads -> SGD -> precision EMA.

    Returns (params, opt_state, new_var, energy_per_sample_sum).
    """
    batch_size = next(iter(batch.values())).shape[0]
    clamps = {
        structure.task_map[t]: v for t, v in batch.items() if t in structure.task_map
    }

    precision_map = _vars_to_precision(var, eps, clip, normalize)
    # Invariant (guards the fori_loop carry structure): precision covers EXACTLY the
    # learned-precision node set, so which nodes carry a precision array is stable
    # across inference steps and across training steps. Static keys -> trace-time check.
    assert set(precision_map) == set(names), "precision_map must cover exactly `names`"

    init_state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
        precision_map=precision_map,
    )
    final_state = run_inference(params, init_state, clamps, structure)

    # Local (precision-weighted) weight gradients; SGD preserves the precision scaling.
    grads = compute_local_weight_gradients(params, final_state, structure)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Energy over internal nodes (per-sample, summed) for monitoring.
    energy = (
        sum(
            jnp.sum(final_state.nodes[n].energy)
            for n in structure.nodes
            if structure.nodes[n].node_info.in_degree > 0
        )
        / batch_size
    )

    new_var = _update_vars(var, final_state, names, lam)
    return params, opt_state, new_var, energy


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_ngd(
    params: GraphParams,
    structure: GraphStructure,
    train_loader,
    optimizer: optax.GradientTransformation,
    num_epochs: int,
    rng_key: jax.Array,
    *,
    lam: float = 0.05,
    eps: float = 1e-3,
    clip: Tuple[float, float] = (0.1, 10.0),
    normalize: bool = True,
    epoch_callback=None,
    verbose: bool = False,
):
    """
    Train with online-precision natural-gradient descent.

    Args:
        optimizer: should be plain SGD (e.g. optax.sgd(schedule)) so the per-channel
            precision survives into the update as the natural-gradient learning rate.
        lam: EMA rate for the online precision (per-step). Smaller = slower/steadier.
        eps, clip, normalize: as in probe_residual_precision -- numerical floor,
            post-normalization clamp, and per-layer mean normalization.

    Returns (trained_params, final_precision, energy_history):
      - final_precision: the resolved per-node precision map (Pi, with this call's
        eps/clip/normalize already applied) — pass straight to probe_latent_propagation
        so the probe uses exactly what training used (no kwarg re-derivation).
      - energy_history: list (per epoch) of per-batch energies. SAME reduction as
        train_pcn (sum over in_degree>0 node energies / batch_size), so the shape and
        node set match. NOTE the energies are PRECISION-WEIGHTED here (as they are for
        diag_probe), so compare energies within a precision mode, not across modes.
    """
    from fabricpc.training.train import _convert_batch

    names = precision_node_names(structure)
    var = init_precision_vars(structure, names)
    opt_state = optimizer.init(params)

    step = jax.jit(
        lambda p, o, v, b, k: ngd_train_step(
            p,
            o,
            v,
            b,
            structure,
            optimizer,
            names,
            k,
            lam=lam,
            eps=eps,
            clip=clip,
            normalize=normalize,
        )
    )

    energy_history = []
    for epoch in range(num_epochs):
        batch_energies = []
        for batch_data in train_loader:
            batch = _convert_batch(batch_data)
            rng_key, step_key = jax.random.split(rng_key)
            params, opt_state, var, energy = step(
                params, opt_state, var, batch, step_key
            )
            batch_energies.append(float(energy))
        energy_history.append(batch_energies)
        if verbose:
            avg = sum(batch_energies) / max(len(batch_energies), 1)
            print(f"Epoch {epoch + 1}/{num_epochs}  energy={avg:.4f}")
        if epoch_callback is not None:
            epoch_callback(epoch, params, structure, {}, rng_key)

    final_precision = _vars_to_precision(var, eps, clip, normalize)
    return params, final_precision, energy_history


# ---------------------------------------------------------------------------
# Verification probe: does signal reach the early layers?
# ---------------------------------------------------------------------------


def probe_latent_propagation(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    precision_map: Dict[str, jnp.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run one clamped inference pass and report, per node, whether the latent states are
    non-zero and how far they moved from initialization -- i.e. whether the (precision-
    weighted) error signal actually propagates back to the EARLY layers, or whether the
    early latents stay stuck at their init.

    Returns {node_name: {z_rms, err_rms, zmu_rms, move_rms, in_degree}} where:
        z_rms    = RMS of the converged latent state z_latent
        err_rms  = RMS of the prediction error (z_latent - z_mu)
        zmu_rms  = RMS of the prediction z_mu
        move_rms = RMS of (z_converged - z_init): how much inference moved the latent
        in_degree= node in-degree (0 = clamped input / terminal)
    """
    batch_size = next(iter(batch.values())).shape[0]
    clamps = {
        structure.task_map[t]: v for t, v in batch.items() if t in structure.task_map
    }
    init_state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
        precision_map=precision_map,
    )
    final_state = run_inference(params, init_state, clamps, structure)

    def rms(x):
        return float(jnp.sqrt(jnp.mean(x**2)))

    report = {}
    for name in structure.node_order:
        info = structure.nodes[name].node_info
        z0 = init_state.nodes[name].z_latent
        zf = final_state.nodes[name].z_latent
        report[name] = {
            "z_rms": rms(zf),
            "err_rms": rms(final_state.nodes[name].error),
            "zmu_rms": rms(final_state.nodes[name].z_mu),
            "move_rms": rms(zf - z0),
            "in_degree": int(info.in_degree),
        }
    return report
