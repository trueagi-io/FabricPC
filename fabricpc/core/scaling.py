"""
muPC scaling utilities for predictive coding gradient computation.

Composable scaling functions applied at callsites (inference loop, learning loop)
to separate variance-preserving scaling from node computation. Node methods
(forward_and_latent_grads, forward_and_weight_grads) are pure autodiff — they know nothing
about muPC scaling.

Usage in inference loop (inference.py):
    scaled_inputs = scale_inputs(inputs, node_info.scaling_config)
    node_state, grads = node_class.forward_and_latent_grads(params, scaled_inputs, ...)
    grads = scale_input_grads(grads, node_info.scaling_config)

Usage in learning loop (graph_net.py):
    scaled_inputs = scale_inputs(inputs, node_info.scaling_config)
    node_state, grad_params = node_class.forward_and_weight_grads(params, scaled_inputs, ...)
    grad_params = scale_weight_grads(grad_params, node_info.scaling_config)
"""

from typing import Dict, Optional

import jax.numpy as jnp

from fabricpc.core.types import NodeParams

# TYPE_CHECKING avoids circular import with mupc.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fabricpc.core.mupc import MuPCScalingFactors


def scale_inputs(
    inputs: Dict[str, jnp.ndarray],
    scaling_config: Optional["MuPCScalingFactors"],
) -> Dict[str, jnp.ndarray]:
    """Pre-scale inputs by muPC forward scaling factors.

    When scaling_config is present, each input tensor is multiplied by
    its per-edge forward_scale factor. Since W @ (a*x) = a * (W @ x),
    this is mathematically equivalent to scaling the node output, but
    keeps all scaling logic outside the node's forward() method.

    Edges arriving at non-variance-scalable slots are absent from
    forward_scale and pass through unscaled — multiplying by 1.0 would
    silently promote integer inputs (e.g. token indices feeding an
    EmbeddingNode) to float.

    When scaling_config is None, returns inputs unchanged.
    """
    if scaling_config is None:
        return inputs
    fs = scaling_config.forward_scale
    return {k: x * fs[k] if k in fs else x for k, x in inputs.items()}


def scale_input_grads(
    input_grads: Dict[str, jnp.ndarray],
    scaling_config: Optional["MuPCScalingFactors"],
) -> Dict[str, jnp.ndarray]:
    """Post-scale input gradients by topdown gradient scaling factors.

    topdown_grad_scale = a * jacobian_gain, combining:
    - Chain rule correction (a): restores dE/dx from dE/d(a*x)
    - Jacobian compensation (jacobian_gain): normalizes per-hop gradient
      propagation to ~1.0 for saturating activations

    Edges arriving at non-variance-scalable slots are absent from
    topdown_grad_scale and pass through unscaled, mirroring scale_inputs.
    """
    if scaling_config is None:
        return input_grads
    td = scaling_config.topdown_grad_scale
    return {k: g * td[k] if k in td else g for k, g in input_grads.items()}


def scale_self_grad(
    z_latent_grad: jnp.ndarray,
    scaling_config: Optional["MuPCScalingFactors"],
) -> jnp.ndarray:
    """Post-scale self-latent gradient by muPC self_grad_scale."""
    if scaling_config is None:
        return z_latent_grad
    return z_latent_grad * scaling_config.self_grad_scale


def scale_weight_grads(
    params_grad: NodeParams,
    scaling_config: Optional["MuPCScalingFactors"],
) -> NodeParams:
    """Post-scale weight gradients by muPC weight gradient scaling.

    Per-weight scaling is applied only when the weight's key matches an
    edge_key present in `weight_grad_scale`. Weights whose names do not
    match any edge_key (e.g., internal parameters of TransformerBlock
    such as W_q/W_k/W_v whose tensor shapes are tied to attention head
    geometry rather than input edges) pass through at scale 1.0; such
    nodes are expected to handle their own scaling inside forward().

    `weight_grad_scale` is populated only for edges whose target slot is
    variance-scalable (see compute_mupc_scalings). Non-scalable slots
    (mask, skip, residual) contribute no entries and therefore do not
    influence weight-gradient scaling.
    """
    if scaling_config is None:
        return params_grad
    wg_scale = scaling_config.weight_grad_scale
    scaled_weights = {
        k: grad * wg_scale.get(k, 1.0) for k, grad in params_grad.weights.items()
    }
    return NodeParams(weights=scaled_weights, biases=params_grad.biases)
