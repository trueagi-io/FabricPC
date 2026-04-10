"""
muPC (Maximal Update Parameterization for Predictive Coding) scaling computation.

This module computes per-node scaling factors that maintain O(1) variance of
activations, prediction errors, and gradients across networks of arbitrary
width and depth. Based on:

  - Depth-muP (Yang et al.) — width+depth scaling for deep networks
  - muPC (Innocenti et al., arXiv:2505.13124) — adaptation to predictive coding

Extended with a three-way gradient decomposition for arbitrary graph topologies:

  1. Forward scaling (a_l per edge): maintains O(1) activation variance
  2. Self-gradient scaling (c_self per node): controls dE/dz_self magnitude
  3. Top-down gradient scaling (c_td per edge): normalizes W^T * epsilon
  4. Weight gradient scaling (c_w per edge): controls dE/dW magnitude

Usage:
    from fabricpc.core.mupc import MuPCConfig
    from fabricpc.core.depth_metric import ShortestPathDepth

    structure = graph(
        nodes=[...], edges=[...], task_map=...,
        inference=InferenceSGD(...),
        scaling=MuPCConfig(depth_metric=ShortestPathDepth()),
    )
"""

import math
from typing import Dict, Set, Any, Optional
from dataclasses import dataclass, field

from fabricpc.core.depth_metric import DepthMetricBase, ShortestPathDepth


@dataclass(frozen=True)
class MuPCScaling:
    """
    Per-node scaling factors for muPC parameterization.

    All dictionaries are keyed by edge_key (e.g., "h1->h2:in").

    Attributes:
        forward_scale: Per-edge forward scaling factor (a_l).
            Applied to inputs before the node's forward() call.
        self_grad_scale: Scalar scaling for the node's self-gradient (dE/dz_self).
            Applied in energy_functional().
        topdown_grad_scale: Per-edge scaling for the top-down gradient to
            presynaptic nodes. Applied after autodiff in forward_inference().
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
        depth_metric: DepthMetricBase instance for computing effective depth.
            Default: ShortestPathDepth().
        min_depth: Minimum effective depth used in scaling formulas.
            Prevents division by zero and excessive scaling for shallow paths.
            Default: 1.
    """

    depth_metric: DepthMetricBase = field(default_factory=ShortestPathDepth)
    min_depth: int = 1


def compute_mupc_scalings(
    nodes: Dict[str, Any],
    edges: Dict[str, Any],
    output_nodes: Set[str],
    config: MuPCConfig,
) -> Dict[str, MuPCScaling]:
    """
    Compute per-node MuPCScaling from graph topology.

    Args:
        nodes: Dictionary mapping node names to finalized NodeBase instances
               (each has .node_info with shape, in_edges, out_edges, etc.)
        edges: Dictionary mapping edge keys to EdgeInfo objects
        output_nodes: Set of node names that are clamped to targets during
                      training (typically from task_map["y"]).
        config: MuPCConfig with depth metric and parameters.

    Returns:
        Dictionary mapping node names to MuPCScaling instances.
        Source nodes (in_degree=0) get None (no scaling needed).
    """
    import numpy as np

    # Compute effective depth for each node
    depths = config.depth_metric.compute(nodes, edges)

    scalings = {}

    for node_name, node in nodes.items():
        node_info = node.node_info

        # Source nodes: no incoming edges, no scaling needed
        if node_info.in_degree == 0:
            scalings[node_name] = None
            continue

        is_output = node_name in output_nodes
        fan_out = int(np.prod(node_info.shape))
        effective_depth = max(depths.get(node_name, 1), config.min_depth)

        forward_scale = {}
        topdown_grad_scale = {}
        weight_grad_scale = {}

        for edge_key in node_info.in_edges:
            edge_info = edges[edge_key]
            source_node = nodes[edge_info.source]
            source_shape = source_node.node_info.shape
            fan_in = int(np.prod(source_shape))

            # Forward scaling: a_l
            if is_output:
                # Output node: stronger 1/fan_in scaling (muPC-specific)
                a = 1.0 / fan_in
            else:
                # Hidden node: 1/sqrt(fan_in * L)
                a = 1.0 / math.sqrt(fan_in * effective_depth)

            forward_scale[edge_key] = a

            # Top-down gradient scaling: c_td
            # Goal: make W^T * epsilon have O(1) variance per component
            # Autodiff already includes a_l from input pre-scaling.
            # The autodiff result has variance ~ a^2 * fan_out.
            # We want total = c_td * a * sqrt(fan_out) = O(1)
            # So c_td = 1 / (a * sqrt(fan_out))
            #         = sqrt(fan_in * L) / sqrt(fan_out)  for hidden
            #         = fan_in / sqrt(fan_out)             for output
            if fan_out > 0:
                c_td = 1.0 / (a * math.sqrt(fan_out))
            else:
                c_td = 1.0
            topdown_grad_scale[edge_key] = c_td

            # Weight gradient scaling: 1.0 (let optimizer handle magnitude)
            weight_grad_scale[edge_key] = 1.0

        # Self-gradient scaling: 1.0
        # The self-gradient (dE/dz from energy_functional) is already O(1)
        # when the forward scaling maintains O(1) activations.
        self_grad_scale = 1.0

        scalings[node_name] = MuPCScaling(
            forward_scale=forward_scale,
            self_grad_scale=self_grad_scale,
            topdown_grad_scale=topdown_grad_scale,
            weight_grad_scale=weight_grad_scale,
        )

    return scalings
