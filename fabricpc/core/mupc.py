"""
muPC (Maximal Update Parameterization for Predictive Coding) scaling computation.

This module computes per-node scaling factors that maintain O(1) variance of
activations, prediction errors, and gradients across networks of arbitrary
width and topology. Based on:

  - Depth-muP (Yang et al.) — width+depth scaling for deep networks
  - muPC (Innocenti et al., arXiv:2505.13124) — adaptation to predictive coding

Scaling is computed per in-edge, based on the target slot's properties:

  - Edges arriving at a slot with is_variance_scalable=True get the full
    muPC scaling formula:

        a = gain / sqrt(fan_in * K_slot * L)

    where fan_in is the weight-matrix fan_in (from get_weight_fan_in()),
    K_slot is the in-degree of the *specific slot* (not the whole node),
    gain = activation.variance_gain() is the Kaiming-style gain, and
    L is the residual depth.

  - Edges arriving at a slot with is_variance_scalable=False pass through
    at scale 1.0. This preserves the identity mapping that carries signal
    through deep residual networks.

Residual depth L is the number of nodes along the longest path that have
at least one slot with is_skip_connection=True. These are the
variance-accumulating merge points where skip and compute paths sum.
For pure sequential chains (no skip connections), L=1. For residual
networks with D blocks, L=D. Slots with is_variance_scalable=False but
is_skip_connection=False (e.g., metadata like attention masks) do not
contribute to L.

  input → h1 → skip1(+) → h2 → skip2(+) → h3 → skip3(+) → output
           │      ↑         │      ↑         │      ↑
           └──────┘         └──────┘         └──────┘
              L=1              L=2              L=3

Top-down gradient scaling combines chain rule correction with Jacobian
compensation for deep gradient propagation:

    c_td = a * jacobian_gain

Chain rule correction (a): Because _apply_forward_scaling pre-scales
inputs (x -> a*x) before the value_and_grad closure, autodiff yields
dE/d(a*x). Multiplying by a restores dE/dx.

Jacobian compensation (jacobian_gain): The per-hop Jacobian
diag(act'(z)) @ (a*W) has RMS singular value ~ gain * rms(act'(z)),
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


def _count_skip_connections_depth(
    nodes: Dict[str, Any],
    edges: Dict[str, Any],
    node_order: List[str],
) -> int:
    """
    Count the number of nodes with skip connection slots along the longest
    path in the graph. This is the "residual depth" — the number of
    variance-accumulating merge points where skip and compute paths sum.

    A node counts as a merge point if it has at least one slot with
    is_skip_connection=True. Slots that are merely non-scalable (e.g.,
    metadata like attention masks) do not count.

    In a pure sequential chain (no skip connections), returns 0.
    In a ResNet with D residual blocks, returns D.

    The caller uses max(skip_depth, 1) as L in the scaling formula so that
    pure chains degenerate to a = gain/sqrt(fan_in * K) (no depth factor).
    """
    skip_counts: Dict[str, int] = {}

    for node_name in node_order:
        node = nodes[node_name]
        node_info = node.node_info

        if node_info.in_degree == 0:
            skip_counts[node_name] = 0
            continue

        # Find max skip count among predecessors
        max_pred_skips = 0
        for edge_key in node_info.in_edges:
            edge_info = edges[edge_key]
            pred_skips = skip_counts.get(edge_info.source, 0)
            max_pred_skips = max(max_pred_skips, pred_skips)

        # Check if this node has any skip connection slot
        has_skip = any(s.is_skip_connection for s in node_info.slots.values())

        if has_skip:
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

    Uses edge-based scaling: each in-edge is scaled according to the
    target slot's is_variance_scalable flag. Non-scalable slots (e.g.,
    skip/residual connections) pass through at scale 1.0. Scalable slots
    get the full muPC formula:

        a = gain / sqrt(fan_in * K_slot * L)

    where K_slot is the in-degree of the *specific slot* (not the whole
    node), L is the residual depth, and gain = activation.variance_gain().

    For output nodes (include_output=True):

        a = gain / (fan_in * sqrt(K_slot * L))

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

    # Compute residual depth: number of nodes with non-scalable slots
    # along the longest path. For pure chains this is 0, for ResNets it's
    # the number of residual blocks. Using max(depth, 1) ensures
    # pure chains degenerate to the original a = gain/sqrt(fan_in * K).
    skip_depth = _count_skip_connections_depth(nodes, edges, iteration_order)
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

        node_config = node_info.node_config

        # Activation-aware gain for variance preservation (Kaiming convention).
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
            slot_name = edge_info.slot
            slot_info = node_info.slots[slot_name]

            if not slot_info.is_variance_scalable:
                # Non-scalable slot: identity pass-through at scale 1.0
                forward_scale[edge_key] = 1.0
                topdown_grad_scale[edge_key] = 1.0
                weight_grad_scale[edge_key] = 1.0
            else:
                # Scalable slot: full muPC formula
                # K_slot = in-degree of this specific slot
                K_slot = len(slot_info.in_neighbors)

                source_node = nodes[edge_info.source]
                source_shape = source_node.node_info.shape

                # Unified fan_in from node class. Weighted nodes return their
                # weight-matrix fan_in (Kaiming convention); weightless nodes
                # (e.g. IdentityNode) return 1.
                fan_in = node_class.get_weight_fan_in(source_shape, node_config)

                # Forward scaling formula with activation gain and depth:
                #   Hidden: a = gain/sqrt(fan_in * K_slot * L)
                #       — K_slot handles multi-input variance amplification per slot
                #       — L bounds total variance growth to (1+1/L)^L ~ e
                #   Output: a = gain/(fan_in * sqrt(K_slot * L))
                #       — matches muPC O(1/N) convention
                if is_output:
                    a = gain / (fan_in * math.sqrt(K_slot * L))
                else:
                    a = gain / math.sqrt(fan_in * K_slot * L)

                forward_scale[edge_key] = a

                # Top-down gradient scaling: c_td = a * jacobian_gain.
                #
                # Two components:
                # 1. Chain rule correction (a): _apply_forward_scaling pre-scales
                #    inputs (x -> a*x) outside the value_and_grad closure. Autodiff
                #    yields dE/d(a*x); multiplying by a restores dE/dx.
                # 2. Jacobian compensation (jacobian_gain): the per-hop Jacobian
                #    diag(act'(z)) @ (a*W) has RMS singular value ~ gain*rms(act'(z)),
                #    which is < 1 for saturating activations. Multiplying by
                #    jacobian_gain = 1/(gain*rms(act'(z))) normalizes the expected
                #    per-hop gradient propagation factor to ~1.0.
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
