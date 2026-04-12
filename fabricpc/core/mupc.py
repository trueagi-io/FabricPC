"""
muPC (Maximal Update Parameterization for Predictive Coding) scaling computation.

This module computes per-node scaling factors that maintain O(1) variance of
activations, prediction errors, and gradients across networks of arbitrary
width and topology. Based on:

  - Depth-muP (Yang et al.) — width+depth scaling for deep networks
  - muPC (Innocenti et al., arXiv:2505.13124) — adaptation to predictive coding

Forward scaling uses a unified per-edge formula:

    a_k = 1 / sqrt(fan_in_k * K)

where fan_in_k is the weight-matrix fan_in (from get_weight_fan_in()) and
K is the node's in-degree. Weightless nodes (e.g. IdentityNode) return
fan_in=1, so the formula reduces to a=1/sqrt(K) — compensating only for
multi-edge summation variance amplification.

Top-down gradient scaling restores the chain rule factor lost when
forward scaling is applied outside the differentiation:

    c_td = a_k  (same as the forward scaling factor)

Because _apply_forward_scaling pre-scales inputs (x → a*x) before the
value_and_grad closure, autodiff yields dE/d(a*x). The gradient w.r.t.
the original input x is dE/dx = a * dE/d(a*x) by chain rule. Setting
c_td = a restores this factor, matching what jpc computes by
differentiating through the scaling.

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
            presynaptic nodes. Equals the forward_scale (a) to restore the
            chain rule factor lost when scaling is applied outside autodiff.
            Applied after autodiff in forward_inference().
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

        a_k = 1 / sqrt(fan_in_k * K)

    where fan_in_k comes from each node class's get_weight_fan_in() method
    (which returns 1 for weightless nodes like IdentityNode) and K is the
    node's in-degree (number of input edges).

    For output nodes (include_output=True):

        a_k = 1 / (fan_in_k * sqrt(K))

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

            # Forward scaling formula:
            #   Hidden: a = 1/sqrt(fan_in * K)  — maintains O(1) pre-activation variance
            #   Output: a = 1/(fan_in * sqrt(K)) — matches muPC O(1/N) convention
            if is_output:
                a = 1.0 / (fan_in * math.sqrt(K))
            else:
                a = 1.0 / math.sqrt(fan_in * K)

            forward_scale[edge_key] = a

            # Top-down gradient scaling: c_td = a (chain rule correction).
            #
            # _apply_forward_scaling pre-scales inputs (x → a*x) outside the
            # value_and_grad closure. Autodiff then yields dE/d(a*x), but the
            # gradient w.r.t. the original presynaptic latent x is:
            #
            #   dE/dx = a * dE/d(a*x)    (by chain rule)
            #
            # Setting c_td = a restores this factor. Without it, each hop
            # through the network drops the top-down gradient by a factor of a,
            # causing exponential gradient vanishing in deep networks:
            # gradient ∝ a^L → 0 for large L.
            topdown_grad_scale[edge_key] = a

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
