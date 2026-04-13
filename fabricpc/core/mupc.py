"""
muPC (Maximal Update Parameterization for Predictive Coding) scaling computation.

This module computes per-node scaling factors that maintain O(1) variance of
activations, prediction errors, and gradients across networks of arbitrary
width and topology. Based on:

  - Depth-muP (Yang et al.) — width+depth scaling for deep networks
  - muPC (Innocenti et al., arXiv:2505.13124) — adaptation to predictive coding

Forward scaling uses a unified per-edge formula:

    a_k = gain / sqrt(fan_in_k * K)

where fan_in_k is the weight-matrix fan_in (from get_weight_fan_in()),
K is the node's in-degree, and gain = activation.variance_gain() is the
Kaiming-style gain compensating for activation-induced variance contraction
(e.g. sqrt(5/3) for tanh, sqrt(2) for ReLU, 1.0 for identity).
Weightless nodes (e.g. IdentityNode) return fan_in=1, so the formula
reduces to a=gain/sqrt(K).

Top-down gradient scaling combines chain rule correction with Jacobian
compensation for deep gradient propagation:

    c_td = a_k * jacobian_gain

Chain rule correction (a_k): Because _apply_forward_scaling pre-scales
inputs (x → a*x) before the value_and_grad closure, autodiff yields
dE/d(a*x). Multiplying by a restores dE/dx.

Jacobian compensation (jacobian_gain): The per-hop Jacobian
diag(act'(z)) @ (a*W) has RMS singular value ≈ gain * rms(act'(z)),
which is < 1 for saturating activations (e.g. tanh: 0.79). Over L
hops, gradients shrink as (gain*rms(act'))^L. The jacobian_gain
factor = 1/(gain*rms(act'(z))) normalizes this to ~1.0 per hop.
For identity/ReLU/LeakyReLU, jacobian_gain = 1.0 (exact preservation).

Additional scaling factors:
  - Self-gradient scaling (c_self per node): 1.0 (self-gradient is
    already O(1) when forward scaling maintains O(1) activations)
  - Weight gradient scaling (c_w per edge): 1.0 (optimizer handles
    magnitude; placeholder for future exploration)

Usage:
    from fabricpc.core.mupc import MuPCConfig

    structure = graph(
        nodes=[...], edges=[...], task_map=...,
        inference=InferenceSGD(...),
        scaling=MuPCConfig(include_output=True),
    )
"""

import math
import warnings
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass(frozen=True)
class MuPCScalingFactors:
    """
    Per-node scaling factors for muPC parameterization.

    All dictionaries are keyed by edge_key (e.g., "h1->h2:in").

    Attributes:
        forward_scale: Per-edge forward scaling factor (a_l).
            Applied to inputs before the node's forward() call.
        self_grad_scale: Scalar scaling for the node's self-gradient (dE/dz_self).
            Applied in energy_functional().
        topdown_grad_scale: Per-edge scaling for the top-down gradient to
            presynaptic nodes. Equals a * jacobian_gain, combining chain rule
            correction (a) with Jacobian compensation (jacobian_gain) for
            deep gradient propagation. Applied after autodiff in
            forward_inference().
        weight_grad_scale: Per-edge scaling for weight gradients.
            Applied after autodiff in forward_learning().
    """

    forward_scale: Dict[str, float]
    self_grad_scale: float
    topdown_grad_scale: Dict[str, float]
    weight_grad_scale: Dict[str, float]


@dataclass(frozen=True)
class MuPCConfig:
    """
    Configuration for muPC scaling computation.

    Pass this as the `scaling` argument to the graph() builder.

    Args:
        include_output: Whether to include output nodes (out_degree=0) in muPC
            scaling. When True, output nodes get a = 1/(fan_in * sqrt(K))
            (matching jpc reference a_L = 1/N for K=1). When False (default),
            output nodes are excluded and should use standard initialization
            (e.g., Xavier). Use True with MSE/Gaussian energy; False with
            softmax+CE.
        terminal_input_variance: Assumed variance of terminal input node
            outputs. Default 1.0. Not currently used in the simplified formula
            (all scaled nodes produce O(1) output), but reserved for future
            extensions with non-unit input variance.
    """

    include_output: bool = False
    terminal_input_variance: float = 1.0
    # Deprecated: kept for backward compatibility
    depth_metric: Optional[Any] = None
    min_depth: Optional[int] = None

    def __post_init__(self):
        if self.depth_metric is not None:
            warnings.warn(
                "MuPCConfig.depth_metric is deprecated and ignored. "
                "muPC scaling now uses a=1/sqrt(fan_in*K) based on graph "
                "topology (in-degree K) instead of depth metrics.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.min_depth is not None:
            warnings.warn(
                "MuPCConfig.min_depth is deprecated and ignored. "
                "muPC scaling now uses a=1/sqrt(fan_in*K) based on graph "
                "topology (in-degree K) instead of depth metrics.",
                DeprecationWarning,
                stacklevel=2,
            )


def compute_mupc_scalings(
    nodes: Dict[str, Any],
    edges: Dict[str, Any],
    config: MuPCConfig,
    node_order: Optional[List[str]] = None,
) -> Dict[str, MuPCScalingFactors]:
    """
    Compute per-node MuPCScalingFactors from graph topology.

    Uses a unified forward scaling formula for all non-source nodes:

        a_k = gain / sqrt(fan_in_k * K)

    where fan_in_k comes from each node class's get_weight_fan_in() method
    (which returns 1 for weightless nodes like IdentityNode), K is the
    node's in-degree, and gain = activation.variance_gain() compensates
    for activation-induced variance contraction (Kaiming convention).

    For output nodes (include_output=True):

        a_k = gain / (fan_in_k * sqrt(K))

    Args:
        nodes: Dictionary mapping node names to finalized NodeBase instances
               (each has .node_info with shape, in_edges, out_edges, etc.)
        edges: Dictionary mapping edge keys to EdgeInfo objects
        config: MuPCConfig with scaling parameters.
        node_order: Optional topological ordering of node names. If provided,
                    nodes are processed in this order for determinism.

    Returns:
        Dictionary mapping node names to MuPCScalingFactors instances.
        Terminal input nodes (in_degree=0) get None.
        Terminal output nodes (out_degree=0) get None unless include_output=True.
    """
    # Detect output nodes (out_degree=0).
    output_nodes = {
        name for name, node in nodes.items() if node.node_info.out_degree == 0
    }

    # Use topological order if provided, otherwise iterate dict order
    iteration_order = node_order if node_order is not None else nodes.keys()

    scalings = {}

    for node_name in iteration_order:
        node = nodes[node_name]
        node_info = node.node_info
        node_class = node_info.node_class

        # Terminal input nodes (in_degree=0): no scaling needed.
        if node_info.in_degree == 0:
            scalings[node_name] = None
            continue

        # Output nodes: skip unless include_output is set.
        # With softmax+CE, excluding output avoids over-compressing logits.
        # With MSE/Gaussian energy, including output matches jpc reference.
        is_output = node_name in output_nodes
        if is_output and not config.include_output:
            scalings[node_name] = None
            continue

        K = node_info.in_degree
        node_config = node_info.node_config

        # Activation-aware gain for variance preservation (Kaiming convention).
        # Compensates for activation-induced variance contraction so that
        # Var(act(z)) ≈ 1 when pre-activations have Var(z) = gain^2.
        activation = node_info.activation
        if activation is not None:
            gain = type(activation).variance_gain(activation.config)
            jac_gain = type(activation).jacobian_gain(activation.config)
        else:
            gain = 1.0
            jac_gain = 1.0

        forward_scale = {}
        topdown_grad_scale = {}
        weight_grad_scale = {}

        for edge_key in node_info.in_edges:
            edge_info = edges[edge_key]
            source_node = nodes[edge_info.source]
            source_shape = source_node.node_info.shape

            # Unified fan_in from node class. Weighted nodes return their
            # weight-matrix fan_in (Kaiming convention); weightless nodes
            # (e.g. IdentityNode) return 1.
            fan_in = node_class.get_weight_fan_in(source_shape, node_config)

            # Forward scaling formula with activation gain:
            #   Hidden: a = gain/sqrt(fan_in * K) — maintains O(1) post-activation variance
            #   Output: a = gain/(fan_in * sqrt(K)) — matches muPC O(1/N) convention
            if is_output:
                a = gain / (fan_in * math.sqrt(K))
            else:
                a = gain / math.sqrt(fan_in * K)

            forward_scale[edge_key] = a

            # Top-down gradient scaling: c_td = a * jacobian_gain.
            #
            # Two components:
            # 1. Chain rule correction (a): _apply_forward_scaling pre-scales
            #    inputs (x → a*x) outside the value_and_grad closure. Autodiff
            #    yields dE/d(a*x); multiplying by a restores dE/dx.
            # 2. Jacobian compensation (jacobian_gain): the per-hop Jacobian
            #    diag(act'(z)) @ (a*W) has RMS singular value ≈ gain*rms(act'(z)),
            #    which is < 1 for saturating activations (e.g. tanh). Multiplying
            #    by jacobian_gain = 1/(gain*rms(act'(z))) normalizes the expected
            #    per-hop gradient propagation factor to ~1.0, preventing
            #    exponential gradient vanishing in deep networks.
            #    For identity/ReLU/LeakyReLU, jacobian_gain = 1.0 (no correction).
            topdown_grad_scale[edge_key] = a * jac_gain

            # Weight gradient scaling: 1.0 (optimizer handles magnitude).
            weight_grad_scale[edge_key] = 1.0

        # Self-gradient scaling: 1.0
        # The self-gradient (dE/dz from energy_functional) is already O(1)
        # when the forward scaling maintains O(1) activations.
        self_grad_scale = 1.0

        scalings[node_name] = MuPCScalingFactors(
            forward_scale=forward_scale,
            self_grad_scale=self_grad_scale,
            topdown_grad_scale=topdown_grad_scale,
            weight_grad_scale=weight_grad_scale,
        )

    return scalings
