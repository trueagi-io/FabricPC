"""
muPC (Maximal Update Parameterization for Predictive Coding) scaling computation.

This module computes per-node scaling factors that maintain O(1) variance of
activations, prediction errors, and gradients across networks of arbitrary
width and topology. Based on:

  - Depth-muP (Yang et al.) — width+depth scaling for deep networks
  - muPC (Innocenti et al., arXiv:2505.13124) — adaptation to predictive coding

Forward scaling for variance-scaled nodes uses a depth+fan_in formula:

    a_k = gain / sqrt(fan_in_k * K * L)

where fan_in_k is the weight-matrix fan_in (from get_weight_fan_in()),
K is the node's in-degree, gain = activation.variance_gain() is the
Kaiming-style gain, and L is the residual depth (number of SkipConnection
nodes along the longest path in the graph, minimum 1).

For pure sequential chains (no SkipConnection nodes), L=1 and the
formula reduces to a = gain/sqrt(fan_in * K). For residual networks
with D skip connections, L=D and the depth factor bounds total variance
growth to (1 + gain²/L)^L ≈ e^{gain²} — finite regardless of depth.

Skip connections (nodes with apply_variance_scaling=False) pass through
at scale 1.0. This preserves the identity mapping that carries signal
through deep residual networks. Without this, in-degree scaling attenuates
skip paths by 1/sqrt(K), causing exponential signal decay (0.707^L).

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

from fabricpc.nodes.skip_connection import SkipConnection


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
                "muPC scaling now uses a=gain/sqrt(fan_in*K*L) based on graph "
                "topology (in-degree K, depth L) instead of depth metrics.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.min_depth is not None:
            warnings.warn(
                "MuPCConfig.min_depth is deprecated and ignored. "
                "muPC scaling now uses a=gain/sqrt(fan_in*K*L) based on graph "
                "topology (in-degree K, depth L) instead of depth metrics.",
                DeprecationWarning,
                stacklevel=2,
            )


def _count_skip_depth(
    nodes: Dict[str, Any],
    edges: Dict[str, Any],
    node_order: List[str],
) -> int:
    """
    Count the number of unscaled (SkipConnection) nodes along the longest
    path in the graph. This is the "residual depth" — the number of
    variance-accumulating merge points where skip and compute paths sum.

    In a pure sequential chain (no SkipConnection nodes), returns 0.
    In a ResNet with D residual blocks, returns D.

    The caller uses max(skip_depth, 1) as L in the scaling formula so that
    pure chains degenerate to a = gain/sqrt(fan_in * K) (no depth factor).
    """
    # Count skip nodes along the longest path (BFS on topological order)
    skip_counts: Dict[str, int] = {}

    for node_name in node_order:
        node = nodes[node_name]
        node_info = node.node_info
        node_class = node_info.node_class

        if node_info.in_degree == 0:
            skip_counts[node_name] = 0
            continue

        # Find max skip count among predecessors
        max_pred_skips = 0
        for edge_key in node_info.in_edges:
            edge_info = edges[edge_key]
            pred_skips = skip_counts.get(edge_info.source, 0)
            max_pred_skips = max(max_pred_skips, pred_skips)

        # SkipConnection nodes increment the skip depth count
        if issubclass(node_class, SkipConnection):
            skip_counts[node_name] = max_pred_skips + 1
        else:
            skip_counts[node_name] = max_pred_skips

    return max(skip_counts.values()) if skip_counts else 0


def compute_mupc_scalings(
    nodes: Dict[str, Any],
    edges: Dict[str, Any],
    config: MuPCConfig,
    node_order: Optional[List[str]] = None,
) -> Dict[str, MuPCScalingFactors]:
    """
    Compute per-node MuPCScalingFactors from graph topology.

    Uses a depth+fan_in scaling formula for variance-scaled nodes:

        a_k = gain / sqrt(fan_in_k * K * L)

    where K is the number of variance-scaled incoming edges, L is the
    effective depth, and gain = activation.variance_gain().

    Nodes with apply_variance_scaling=False (e.g., SkipConnection) get
    scale 1.0 on all edges — the identity mapping is preserved.

    For output nodes (include_output=True):

        a_k = gain / (fan_in_k * sqrt(K * L))

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
    iteration_order = node_order if node_order is not None else list(nodes.keys())

    # Compute residual depth: number of SkipConnection (unscaled) nodes
    # along the longest path. For pure chains this is 0, for ResNets it's
    # the number of residual blocks. Using max(skip_depth, 1) ensures
    # pure chains degenerate to the original a = gain/sqrt(fan_in * K).
    skip_depth = _count_skip_depth(nodes, edges, iteration_order)
    L = max(skip_depth, 1)

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

        # Check if this node applies variance scaling
        applies_scaling = getattr(node_class, "apply_variance_scaling", True)

        if not applies_scaling:
            # Skip connection node: all edges pass through at scale 1.0.
            # No variance scaling, no gradient scaling adjustments.
            forward_scale = {ek: 1.0 for ek in node_info.in_edges}
            topdown_grad_scale = {ek: 1.0 for ek in node_info.in_edges}
            weight_grad_scale = {ek: 1.0 for ek in node_info.in_edges}

            scalings[node_name] = MuPCScalingFactors(
                forward_scale=forward_scale,
                self_grad_scale=1.0,
                topdown_grad_scale=topdown_grad_scale,
                weight_grad_scale=weight_grad_scale,
            )
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

            # Forward scaling formula with activation gain and depth:
            #   Hidden: a = gain/sqrt(fan_in * K * L)
            #       — K handles multi-input variance amplification
            #       — L bounds total variance growth to (1+1/L)^L ≈ e
            #   Output: a = gain/(fan_in * sqrt(K * L))
            #       — matches muPC O(1/N) convention
            if is_output:
                a = gain / (fan_in * math.sqrt(K * L))
            else:
                a = gain / math.sqrt(fan_in * K * L)

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
