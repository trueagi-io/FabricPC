"""Graph builder that assembles nodes and edges into a GraphStructure."""

from typing import List, Dict, Any, Optional
from fabricpc.core.types import GraphStructure, NodeInfo, EdgeInfo, SlotInfo
from fabricpc.builder.edge import Edge, SlotRef


class TaskMap:
    """
    Maps task names (x, y, etc.) to nodes.
    Accepts node objects or node name strings.
    """

    def __init__(self, **kwargs):
        self._map = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                self._map[key] = value
            else:
                # NodeBase instance
                self._map[key] = value.name

    def to_dict(self):
        return dict(self._map)


def _build_slots(node, in_edges):
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
            in_neighbors=tuple(in_neighbors),
        )

    return slots


def _topological_sort(nodes, edge_infos):
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


def graph(nodes, edges, task_map, graph_state_initializer=None):
    """
    Build a GraphStructure from node objects, edge objects, and a task map.

    This is the primary entry point for constructing predictive coding graphs.
    Uses copy-on-finalize: original node objects are not modified.

    Args:
        nodes: List of NodeBase instances
        edges: List of Edge instances
        task_map: TaskMap instance or dict mapping task names to node names
        graph_state_initializer: Optional StateInitBase instance
            (default: FeedforwardStateInit())

    Returns:
        GraphStructure with finalized nodes, edges, and topology
    """
    from fabricpc.core.activations import IdentityActivation
    from fabricpc.core.energy import GaussianEnergy
    from fabricpc.core.initializers import NormalInitializer

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

        # Resolve activation/energy/latent_init: use node's instance, or class default
        activation = node._activation
        if activation is None:
            default_act = getattr(type(node), "DEFAULT_ACTIVATION", None)
            activation = default_act() if default_act else IdentityActivation()

        energy = node._energy
        if energy is None:
            default_energy = getattr(type(node), "DEFAULT_ENERGY", None)
            energy = default_energy() if default_energy else GaussianEnergy()

        latent_init = node._latent_init
        if latent_init is None:
            default_init = getattr(type(node), "DEFAULT_LATENT_INIT", None)
            if default_init is not None:
                latent_init = default_init()
            else:
                latent_init = NormalInitializer()

        node_info = NodeInfo(
            name=name,
            shape=node.shape,
            node_type=type(node).__name__,
            node_config=node._extra_config,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            slots=slots,
            in_degree=len(in_edges),
            out_degree=len(out_edges),
            in_edges=tuple(in_edges.keys()),
            out_edges=tuple(out_edges.keys()),
        )
        finalized_nodes[name] = node._with_graph_info(node_info)

    # 5. Topological sort
    node_order = _topological_sort(finalized_nodes, edge_infos)

    # 6. Resolve task map
    if isinstance(task_map, TaskMap):
        task_map_dict = task_map.to_dict()
    elif isinstance(task_map, dict):
        task_map_dict = task_map
    else:
        raise TypeError(f"task_map must be TaskMap or dict, got {type(task_map)}")

    # 7. Build GraphStructure
    from fabricpc.graph.state_initializer import FeedforwardStateInit

    gs_config = {
        "graph_state_initializer": graph_state_initializer or FeedforwardStateInit(),
    }

    return GraphStructure(
        nodes=finalized_nodes,
        edges=edge_infos,
        task_map=task_map_dict,
        node_order=node_order,
        config=gs_config,
    )
