"""Two checks recorded at the start and end of every trial.

How backprop-like a PC run learns. For each layer with weights, the cosine
between the weight update PC computes on one batch and the one backprop
computes on the same batch, and the ratio of their sizes. A cosine near 1
means the layer learns in backprop's direction; near 0, in an unrelated
direction. Two things push it down. A settled PC run follows the gradient
of its settled energy, not the loss, and the two part ways as the run
relaxes (on VGG-5, sPC's cosine fell from 0.85 at 8 settling steps to 0.46
at 150). And a state-based settle carries the error one hop per step,
scaled by about the inference rate each hop, so a first layer's error can
fall below float32 resolution and turn into noise. The size ratio shows the
second: Adam rescales small gradients, but not below its epsilon.

How stiff the network is. lambda_max of the energy Hessian in error
coordinates (``epsilon_spectrum``), a property of the weights alone, so it
can be compared across the three methods: on VGG-5 it went from about 6.5
at init to 27-46 under backprop and 114-176 under sPC. Settled PC can lower
its energy by making the network stiffer instead of more accurate, and this
number is where that shows. For state-based rows it also records the
stiffness the settle itself sees (``latent_curvature``) times the rate: the
settle is stable only while that is below 2.
"""

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from fabricpc.core.epsilon_spectrum import make_epsilon_spectrum
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.latent_curvature import make_latent_curvature
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import _batch_grads, build_clamps, convert_batch

_ITERS = 30


def _flat_weights(grads) -> Dict[str, jnp.ndarray]:
    return {
        name: jnp.concatenate([w.ravel() for w in node.weights.values()])
        for name, node in grads.nodes.items()
        if node.weights
    }


def backprop_alignment(params, structure, batch, key, algorithm: str) -> Optional[dict]:
    """Per layer, PC's weight update against backprop's on one batch.

    None for backprop rows. Uses the trainer's own gradient function, so the
    updates compared are exactly the ones training applies.
    """
    if algorithm == "backprop":
        return None
    batch = convert_batch(batch)
    pc = _flat_weights(_batch_grads(params, batch, structure, key, algorithm="pc")[0])
    bp = _flat_weights(
        _batch_grads(params, batch, structure, key, algorithm="backprop")[0]
    )
    layers = {}
    for name in pc:
        a, b = pc[name], bp[name]
        na, nb = jnp.linalg.norm(a), jnp.linalg.norm(b)
        layers[name] = {
            "cos": float(a @ b / (na * nb + 1e-30)),
            "size_ratio": float(na / (nb + 1e-30)),
        }
    weakest = min(layers, key=lambda n: layers[n]["cos"])
    return {
        "layers": layers,
        "weakest": weakest,
        "min_cos": layers[weakest]["cos"],
        "mean_cos": sum(v["cos"] for v in layers.values()) / len(layers),
    }


def stiffness(params, structure, batch, key, algorithm: str) -> dict:
    """lambda_max in error coordinates; state-based rows add their settle's."""
    clamps = build_clamps(convert_batch(batch), structure, clamp_target=True)
    batch_size = next(iter(clamps.values())).shape[0]
    state = initialize_graph_state(
        structure, batch_size, key, clamps=clamps, params=params
    )
    spectrum = make_epsilon_spectrum(structure, _ITERS)(params, state, clamps, key)
    out = {"lambda_max": float(spectrum.lambda_max)}
    solver = structure.config.get("inference")
    if algorithm == "spc" and not isinstance(solver, EPCInference):
        rho = float(
            make_latent_curvature(structure, _ITERS)(params, state, clamps, key)
        )
        out["settle_stiffness"] = rho
        out["eta_times_settle_stiffness"] = float(solver.config["eta_infer"]) * rho
    return out


def diagnose(params, structure, batch, key, algorithm: str) -> dict:
    """Both checks, as one record for a trial's ``diagnostics``."""
    key_a, key_s = jax.random.split(key)
    return {
        "backprop_alignment": backprop_alignment(
            params, structure, batch, key_a, algorithm
        ),
        "stiffness": stiffness(params, structure, batch, key_s, algorithm),
    }
