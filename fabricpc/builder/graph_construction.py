"""Graph builder that assembles nodes and edges into a GraphStructure."""

import types
from dataclasses import replace
from typing import List, Dict, Any, Optional, Tuple, Union
from fabricpc.core.types import GraphStructure, NodeInfo, EdgeInfo, SlotInfo
from fabricpc.core.inference import InferenceBase
from fabricpc.core.mupc import MuPCConfig, compute_mupc_scalings
from fabricpc.builder.edge import Edge, SlotRef
from fabricpc.nodes.base import NodeBase
from fabricpc.graph_initialization.state_initializer import (
    StateInitBase,
    FeedforwardStateInit,
)


class TaskMap:
    """
    Maps task names (x, y, etc.) to nodes.
    Accepts node objects or node name strings.
    """

    def __init__(self, **kwargs):
        mapping: Dict[str, str] = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                mapping[key] = value
            else:
                # NodeBase instance
                mapping[key] = value.name
        self._map = types.MappingProxyType(mapping)  # Immutable dictionary

    def to_dict(self) -> Dict[str, str]:
        return dict(self._map)


def _build_slots(node: NodeBase, in_edges: Dict[str, EdgeInfo]) -> Dict[str, SlotInfo]:
    """Build SlotInfo objects from node's slot specs and incoming edges."""
    slot_specs = type(node).get_slots()
    slots = {}

    for slot_name, slot_spec in slot_specs.items():
        # Find source nodes connecting to this slot
        in_neighbors = [e.source for e in in_edges.values() if e.slot == slot_name]

        # Validate single-input constraint
        if not slot_spec.is_multi_input and len(in_neighbors) > 1:
            raise ValueError(
                f"Slot '{slot_name}' in node '{node.name}' is single-input "
                f"but has {len(in_neighbors)} connections"
            )

        slots[slot_name] = SlotInfo(
            name=slot_name,
            parent_node=node.name,
            is_multi_input=slot_spec.is_multi_input,
            is_variance_scalable=slot_spec.is_variance_scalable,
            is_skip_connection=slot_spec.is_skip_connection,
            in_neighbors=tuple(in_neighbors),
        )

    return slots


def _topological_sort(
    nodes: Dict[str, NodeBase], edge_infos: Dict[str, EdgeInfo]
) -> Tuple[str, ...]:
    """BFS-based topological sort."""
    in_degree = {name: node.node_info.in_degree for name, node in nodes.items()}
    queue = [name for name, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        node_name = queue.pop(0)
        result.append(node_name)

        for out_edge_key in nodes[node_name].node_info.out_edges:
            edge_info = edge_infos[out_edge_key]
            target_name = edge_info.target
            in_degree[target_name] -= 1
            if in_degree[target_name] == 0:
                queue.append(target_name)

    if len(result) != len(nodes):
        print("Warning: Graph contains cycles, using partial topological order")

    return tuple(result)


def graph(
    nodes: List[NodeBase],
    edges: List[Edge],
    task_map: TaskMap,
    inference: InferenceBase,
    graph_state_initializer: Optional[StateInitBase] = None,
    scaling=None,
) -> GraphStructure:
    """
    Build a GraphStructure from node objects, edge objects, and a task map.

    This is the primary entry point for constructing predictive coding graphs.
    Uses copy-on-finalize: original node objects are not modified.

    Args:
        nodes: List of NodeBase instances
        edges: List of Edge instances
        task_map: TaskMap instance or dict mapping task names to node names
        inference: InferenceBase instance for inference algorithm
        graph_state_initializer: Optional StateInitBase instance
            (default: FeedforwardStateInit())
        scaling: Optional MuPCConfig instance for muPC parameterization.
            When provided, per-node scaling factors are computed from graph
            topology and attached to each NodeInfo.scaling_config.

    Returns:
        GraphStructure with finalized nodes, edges, and topology
    """
    # 1. Build EdgeInfo objects from Edge objects
    edge_infos = {}
    for edge in edges:
        source_name = edge.source.name
        target_name = edge.target_node.name
        target_slot = edge.target_slot
        key = f"{source_name}->{target_name}:{target_slot}"
        edge_infos[key] = EdgeInfo(
            key=key, source=source_name, target=target_name, slot=target_slot
        )

    # 2. Build node names set for validation
    node_names = {node.name for node in nodes}

    # 3. Validate edge endpoints
    for edge_key, edge_info in edge_infos.items():
        if edge_info.source == edge_info.target:
            raise ValueError(f"Self-edge not allowed: '{edge_key}'")
        if edge_info.source not in node_names:
            raise ValueError(f"Edge source node '{edge_info.source}' does not exist")
        if edge_info.target not in node_names:
            raise ValueError(f"Edge target node '{edge_info.target}' does not exist")

    # 4. For each node: resolve defaults, build slots, build NodeInfo, copy-on-finalize
    finalized_nodes = {}
    for node in nodes:
        name = node.name

        # Validate unique names
        if name in finalized_nodes:
            raise ValueError(f"Duplicate node name '{name}'")

        # Find edges for this node
        in_edges = {k: e for k, e in edge_infos.items() if e.target == name}
        out_edges = {k: e for k, e in edge_infos.items() if e.source == name}

        # Build slots
        slots = _build_slots(node, in_edges)

        # Validate incoming edges connect to valid slots
        for edge_key, edge in in_edges.items():
            if edge.slot not in slots:
                raise ValueError(
                    f"Edge '{edge_key}' connects to non-existent slot '{edge.slot}' "
                    f"in node '{name}'. Available slots: {list(slots.keys())}"
                )

        node_info = NodeInfo(
            name=name,
            shape=node.shape,
            node_type=type(node).__name__,
            node_class=type(node),
            node_config=node._extra_config,
            activation=node._activation,
            energy=node._energy,
            latent_init=node._latent_init,
            weight_init=node._weight_init,
            slots=slots,
            in_degree=len(in_edges),
            out_degree=len(out_edges),
            in_edges=tuple(in_edges.keys()),
            out_edges=tuple(out_edges.keys()),
        )
        finalized_nodes[name] = node._with_graph_info(node_info)

    # 5. Topological sort
    node_order = _topological_sort(finalized_nodes, edge_infos)

    # 5b. Compute and attach muPC scalings if requested
    if scaling is not None:
        if not isinstance(scaling, MuPCConfig):
            raise TypeError(
                f"scaling must be a MuPCConfig instance, got {type(scaling)}"
            )

        mupc_scalings = compute_mupc_scalings(
            finalized_nodes, edge_infos, scaling, node_order
        )

        # Attach scaling_config to each NodeInfo via copy-on-finalize
        updated_nodes = {}
        for name, node in finalized_nodes.items():
            node_scaling = mupc_scalings.get(name)
            if node_scaling is not None:
                new_info = replace(node.node_info, scaling_config=node_scaling)
                updated_nodes[name] = node._with_graph_info(new_info)
            else:
                updated_nodes[name] = node
        finalized_nodes = updated_nodes

    # 6. Resolve task map
    if isinstance(task_map, TaskMap):
        task_map_dict = task_map.to_dict()
    elif isinstance(task_map, dict):
        task_map_dict = task_map
    else:
        raise TypeError(f"task_map must be TaskMap or dict, got {type(task_map)}")

    # 7. Build GraphStructure
    gs_config = {
        "graph_state_initializer": graph_state_initializer or FeedforwardStateInit(),
        "inference": inference,
    }

    return GraphStructure(
        nodes=finalized_nodes,
        edges=edge_infos,
        task_map=task_map_dict,
        node_order=node_order,
        config=gs_config,
    )
