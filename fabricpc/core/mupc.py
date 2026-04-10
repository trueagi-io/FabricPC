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
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from fabricpc.core.depth_metric import DepthMetricBase, ShortestPathDepth


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
    config: MuPCConfig,
) -> Dict[str, MuPCScalingFactors]:
    """
    Compute per-node MuPCScalingFactors from graph topology.

    Output nodes are detected automatically as terminal nodes (out_degree=0),
    symmetric to depth metrics using in_degree=0 for source nodes. Output nodes
    receive the stronger 1/fan_in forward scaling instead of 1/sqrt(fan_in*L).

    Args:
        nodes: Dictionary mapping node names to finalized NodeBase instances
               (each has .node_info with shape, in_edges, out_edges, etc.)
        edges: Dictionary mapping edge keys to EdgeInfo objects
        config: MuPCConfig with depth metric and parameters.

    Returns:
        Dictionary mapping node names to MuPCScalingFactors instances.
        Source nodes (in_degree=0) get None (no scaling needed).
    """
    import numpy as np

    # Compute effective depth for each node
    depths = config.depth_metric.compute(nodes, edges)

    # Detect output nodes as terminal nodes (out_degree=0), symmetric to
    # depth metrics using in_degree=0 for source nodes.
    output_nodes = {
        name for name, node in nodes.items() if node.node_info.out_degree == 0
    }

    scalings = {}

    for node_name, node in nodes.items():
        node_info = node.node_info

        # Source nodes: no incoming edges, no scaling needed
        if node_info.in_degree == 0:
            scalings[node_name] = None
            continue

        is_output = node_name in output_nodes
        # fan_out in the Xavier/Kaiming sense: output dimensionality of the
        # weight matrix (number of node features), not graph out-degree.
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
            # Goal: normalize the gradient sent to presynaptic nodes to O(1).
            #
            # Chain rule mechanics:
            #   1. _apply_forward_scaling multiplies inputs by a before forward().
            #   2. jax.value_and_grad(forward, argnums=1) differentiates w.r.t.
            #      the *scaled* inputs, yielding dE/d(a*x).
            #   3. dE/d(a*x) ~ W^T @ epsilon, which has variance ~fan_out
            #      (fan_out rows of unit-variance W entries).
            #   4. The gradient sent to the presynaptic node is c_td * dE/d(a*x).
            #      The presynaptic latent update accumulates this directly.
            #   5. The factor a from input pre-scaling is absorbed into the
            #      differentiated function, so the effective variance contribution
            #      to the presynaptic update is c_td^2 * a^2 * fan_out.
            #   6. For O(1): c_td^2 * a^2 * fan_out = 1
            #      => c_td = 1 / (a * sqrt(fan_out))
            if fan_out > 0:
                c_td = 1.0 / (a * math.sqrt(fan_out))
            else:
                c_td = 1.0

            # Scaling in the backward pass to presynaptic nodes is less crucial because of diminishing energy in deeper layers, but we include it for completeness and to maintain O(1) gradients at the presynaptic nodes.
            topdown_grad_scale[edge_key] = c_td

            # Weight gradient scaling: 1.0 (let optimizer handle magnitude for now. keep as placeholder for future exploration)
            # For non-square weight matrices, the variance of dE/dW can depend on both fan_in and fan_out. A more careful scaling could be derived, but for now we keep it at 1.0 and rely on the optimizer's learning rate to manage it.
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
