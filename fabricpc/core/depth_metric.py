"""
Depth metric classes for computing effective depth of nodes in arbitrary graphs.

Effective depth is used by muPC scaling to determine per-node scaling factors.
The depth of a node affects the forward scaling factor: deeper nodes get
stronger damping to prevent signal explosion across residual connections.

Users can create custom depth metrics by extending DepthMetricBase:

    class MyDepthMetric(DepthMetricBase):
        def compute(self, nodes, edges):
            # Return {node_name: effective_depth} for each node
            ...
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from collections import deque


class DepthMetricBase(ABC):
    """
    Abstract base class for computing effective depth of each node in a graph.

    Effective depth is defined relative to source nodes (nodes with in_degree=0).
    Subclasses implement different strategies for measuring depth in arbitrary
    graph topologies.
    """

    @abstractmethod
    def compute(
        self,
        nodes: Dict[str, Any],
        edges: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Compute effective depth for each node.

        Args:
            nodes: Dictionary mapping node names to finalized NodeBase instances
                   (each has .node_info with in_degree, out_edges, etc.)
            edges: Dictionary mapping edge keys to EdgeInfo objects

        Returns:
            Dictionary mapping node names to integer effective depth.
            Source nodes (in_degree=0) have depth 0.
        """
        pass


class ShortestPathDepth(DepthMetricBase):
    """
    Effective depth = shortest path from any source node.

    Uses BFS from all source nodes simultaneously. This is the conservative
    choice: nodes reachable by a short path get a smaller depth, resulting
    in weaker damping. Suitable when short-circuit paths (skip connections)
    are the dominant signal pathway.
    """

    def compute(self, nodes, edges):
        # Find source nodes (in_degree=0)
        sources = [
            name for name, node in nodes.items() if node.node_info.in_degree == 0
        ]

        # BFS from all sources simultaneously
        depth = {}
        queue = deque()
        for s in sources:
            depth[s] = 0
            queue.append(s)

        while queue:
            node_name = queue.popleft()
            current_depth = depth[node_name]
            node = nodes[node_name]

            for out_edge_key in node.node_info.out_edges:
                edge_info = edges[out_edge_key]
                target = edge_info.target
                new_depth = current_depth + 1

                if target not in depth or new_depth < depth[target]:
                    depth[target] = new_depth
                    queue.append(target)

        return depth


class LongestPathDepth(DepthMetricBase):
    """
    Effective depth = longest path from any source node.

    Uses dynamic programming on the topological order. This is the aggressive
    choice: nodes get depth equal to their longest dependency chain, resulting
    in stronger damping. Suitable for deep networks where the longest path
    determines the signal accumulation.
    """

    def compute(self, nodes, edges):
        # Topological sort via Kahn's algorithm
        in_degree = {name: node.node_info.in_degree for name, node in nodes.items()}
        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        topo_order = []

        while queue:
            node_name = queue.popleft()
            topo_order.append(node_name)
            for out_edge_key in nodes[node_name].node_info.out_edges:
                target = edges[out_edge_key].target
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        # DP: longest path from any source
        depth = {name: 0 for name in nodes}
        for node_name in topo_order:
            current_depth = depth[node_name]
            for out_edge_key in nodes[node_name].node_info.out_edges:
                target = edges[out_edge_key].target
                depth[target] = max(depth[target], current_depth + 1)

        return depth


class FixedDepth(DepthMetricBase):
    """
    User-specified fixed depth for all nodes.

    Useful for testing or when the user wants to manually control the
    depth factor in the scaling computation.

    Args:
        depth: Fixed depth value applied to all non-source nodes.
               Source nodes (in_degree=0) always get depth 0.
    """

    def __init__(self, depth: int):
        if depth < 1:
            raise ValueError(f"Fixed depth must be >= 1, got {depth}")
        self._depth = depth

    def compute(self, nodes, edges):
        return {
            name: 0 if node.node_info.in_degree == 0 else self._depth
            for name, node in nodes.items()
        }
