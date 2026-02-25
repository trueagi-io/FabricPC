"""Core JAX types for predictive coding networks."""

from __future__ import annotations

from typing import Dict, Any, Tuple, NamedTuple, Sequence
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotInfo:
    """Metadata for an input slot to a node."""

    name: str
    parent_node: str
    is_multi_input: bool
    in_neighbors: Tuple[str, ...]


@dataclass(frozen=True)
class NodeInfo:
    """Static metadata and runtime object refs for a node in the graph."""

    name: str
    shape: Tuple[int, ...]
    node_type: str
    node: Any
    activation: Any
    energy: Any
    latent_init: Any
    node_config: Dict[str, Any]
    slots: Dict[str, SlotInfo]
    in_degree: int
    out_degree: int
    in_edges: Tuple[str, ...]
    out_edges: Tuple[str, ...]


@dataclass(frozen=True)
class EdgeInfo:
    """Metadata for a single directed graph edge."""

    key: str
    source: str
    target: str
    slot: str

    @classmethod
    def from_refs(
        cls, source_node: Any, target_node: Any, slot: str = "in"
    ) -> "EdgeInfo":
        """Construct EdgeInfo from concrete node object references."""
        source_name = getattr(source_node, "name", None)
        target_name = getattr(target_node, "name", None)

        if not source_name or not target_name:
            raise ValueError(
                "source_node and target_node must be node objects with a 'name' attribute"
            )
        if source_name == target_name:
            raise ValueError(f"self-edge at: {source_name} is not allowed")

        edge_key = f"{source_name}->{target_name}:{slot}"
        return cls(key=edge_key, source=source_name, target=target_name, slot=slot)


class NodeParams(NamedTuple):
    """Parameters for a single node (weights, biases, etc.)."""

    weights: Dict[str, jnp.ndarray]
    biases: Dict[str, jnp.ndarray]


class GraphParams(NamedTuple):
    """Learnable parameters of the predictive coding network."""

    nodes: Dict[str, NodeParams]

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        total_params = 0
        for node_params in self.nodes.values():
            if "weights" in node_params._fields:
                total_params += sum(w.size for w in node_params.weights.values())
            if "biases" in node_params._fields:
                total_params += sum(b.size for b in node_params.biases.values())
        return f"GraphParams(nodes={n_nodes}, total_params={total_params})"


class NodeState(NamedTuple):
    """Dynamic state of a node during inference."""

    z_latent: jnp.ndarray
    z_mu: jnp.ndarray
    error: jnp.ndarray
    energy: jnp.ndarray
    pre_activation: jnp.ndarray
    latent_grad: jnp.ndarray
    substructure: Dict[str, jnp.ndarray]


class GraphState(NamedTuple):
    """Dynamic state of the network during inference."""

    nodes: Dict[str, NodeState]
    batch_size: int

    def __repr__(self) -> str:
        return f"GraphState(nodes={len(self.nodes)}, batch_size={self.batch_size})"


class GraphStructure(NamedTuple):
    """Static graph topology (compile-time constant)."""

    nodes: Dict[str, NodeInfo]
    edges: Dict[str, EdgeInfo]
    task_map: Dict[str, str]
    node_order: Tuple[str, ...]
    config: Dict[str, Any]

    def __repr__(self) -> str:
        return f"GraphStructure(nodes={len(self.nodes)}, edges={len(self.edges)})"

    @classmethod
    def from_objects(
        cls,
        nodes: Sequence[Any],
        edges: Sequence[EdgeInfo],
        task_map: Dict[str, str] | None = None,
        graph_state_initializer: Any = None,
    ) -> "GraphStructure":
        """Construct a GraphStructure from node instances and EdgeInfo objects."""
        if nodes is None or edges is None:
            raise ValueError("nodes and edges are required")

        node_by_name: Dict[str, Any] = {}
        for node in nodes:
            node_name = getattr(node, "name", None)
            if not node_name:
                raise ValueError("all nodes must provide a 'name' attribute")
            if node_name in node_by_name:
                raise ValueError(f"duplicate node '{node_name}', names must be unique")
            node_by_name[node_name] = node

        edges_by_key: Dict[str, EdgeInfo] = {}
        for edge in edges:
            if not isinstance(edge, EdgeInfo):
                raise TypeError("edges must be EdgeInfo instances")
            if edge.key in edges_by_key:
                raise ValueError(f"duplicate edge: {edge.key}")
            edges_by_key[edge.key] = edge

        for edge in edges_by_key.values():
            if edge.source not in node_by_name:
                raise ValueError(f"edge source node '{edge.source}' does not exist")
            if edge.target not in node_by_name:
                raise ValueError(f"edge target node '{edge.target}' does not exist")

        node_infos: Dict[str, NodeInfo] = {}
        for node_name, node in node_by_name.items():
            in_edges = {k: e for k, e in edges_by_key.items() if e.target == node_name}
            out_edges = {k: e for k, e in edges_by_key.items() if e.source == node_name}
            node_info = node.build_info(in_edges, out_edges)
            node_infos[node_name] = node_info

        node_order = cls._topological_sort(node_infos, edges_by_key)
        cfg = {"graph_state_initializer": graph_state_initializer}

        return cls(
            nodes=node_infos,
            edges=edges_by_key,
            task_map=dict(task_map or {}),
            node_order=node_order,
            config=cfg,
        )

    @staticmethod
    def _topological_sort(
        nodes: Dict[str, "NodeInfo"], edges: Dict[str, "EdgeInfo"]
    ) -> Tuple[str, ...]:
        in_degree = {name: info.in_degree for name, info in nodes.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            node_name = queue.pop(0)
            result.append(node_name)

            node_info = nodes[node_name]
            for out_edge_key in node_info.out_edges:
                edge_info = edges[out_edge_key]
                target_name = edge_info.target
                in_degree[target_name] -= 1
                if in_degree[target_name] == 0:
                    queue.append(target_name)

        if len(result) != len(nodes):
            print("Warning: Graph contains cycles, using partial topological order")

        return tuple(result)


# Register as pytrees for JAX transformations

tree_util.register_pytree_node(
    GraphParams,
    lambda gp: ((gp.nodes,), None),
    lambda aux, children: GraphParams(*children),
)

tree_util.register_pytree_node(
    NodeParams,
    lambda np: ((np.weights, np.biases), None),
    lambda aux, children: NodeParams(*children),
)

tree_util.register_pytree_node(
    NodeState,
    lambda ns: (
        (
            ns.z_latent,
            ns.z_mu,
            ns.error,
            ns.energy,
            ns.pre_activation,
            ns.latent_grad,
            ns.substructure,
        ),
        None,
    ),
    lambda aux, children: NodeState(*children),
)

tree_util.register_pytree_node(
    GraphState,
    lambda gs: ((gs.nodes,), (gs.batch_size,)),
    lambda aux, children: GraphState(children[0], aux[0]),
)

tree_util.register_pytree_node(
    GraphStructure,
    lambda gs: ((), (gs.nodes, gs.edges, gs.task_map, gs.node_order, gs.config)),
    lambda aux, _: GraphStructure(*aux),
)
