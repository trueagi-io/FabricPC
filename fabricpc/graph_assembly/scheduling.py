"""Deterministic node-visit schedules for assembled graphs."""

from abc import ABC, abstractmethod
from collections import deque
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Set, Tuple


class GraphCycleError(ValueError):
    """Raised when a DAG-only operation encounters a directed cycle."""


class TopologySchedulerBase(ABC):
    """Base class for static graph topology schedulers."""

    def __init__(self, **config):
        self.config = MappingProxyType(config)

    @staticmethod
    @abstractmethod
    def compute_schedule(nodes, edges, config) -> Tuple[str, ...]:
        """Return the complete node visit sequence for a graph."""
        raise NotImplementedError


def first_occurrence_order(schedule: Iterable[str]) -> Tuple[str, ...]:
    """Deduplicate a schedule while preserving first occurrence order."""
    seen: Set[str] = set()
    order = []
    for name in schedule:
        if name not in seen:
            seen.add(name)
            order.append(name)
    return tuple(order)


class DAGScheduler(TopologySchedulerBase):
    """Legacy-compatible Kahn/BFS order with explicit cycle rejection."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_schedule(nodes, edges, config) -> Tuple[str, ...]:
        del config
        in_degree = {name: node.node_info.in_degree for name, node in nodes.items()}
        queue = deque(name for name, degree in in_degree.items() if degree == 0)
        result = []

        while queue:
            node_name = queue.popleft()
            result.append(node_name)
            for edge_key in nodes[node_name].node_info.out_edges:
                target = edges[edge_key].target
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        if len(result) != len(nodes):
            visited = set(result)
            unordered = tuple(name for name in nodes if name not in visited)
            raise GraphCycleError(
                "Graph contains a directed cycle; a DAG schedule cannot be "
                f"computed. Unordered nodes: {unordered}. Pass "
                "topology_scheduler=UnrolledCycleScheduler(...) explicitly "
                "for a cyclic sPC graph. EPCInference supports DAGs only."
            )

        return tuple(result)


def _successors(nodes, edges) -> Dict[str, Tuple[str, ...]]:
    return {
        name: tuple(edges[key].target for key in node.node_info.out_edges)
        for name, node in nodes.items()
    }


def _strongly_connected_components(nodes, edges) -> Tuple[Tuple[str, ...], ...]:
    """Iterative Tarjan SCC decomposition in deterministic graph order."""
    successors = _successors(nodes, edges)
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    tarjan_stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[Tuple[str, ...]] = []
    next_index = 0

    for root in nodes:
        if root in indices:
            continue

        indices[root] = next_index
        lowlink[root] = next_index
        next_index += 1
        tarjan_stack.append(root)
        on_stack.add(root)
        # Mutable frames: [node, next-successor-index, parent].
        dfs_stack: List[List[Any]] = [[root, 0, None]]

        while dfs_stack:
            node_name, successor_index, parent = dfs_stack[-1]
            node_successors = successors[node_name]

            if successor_index < len(node_successors):
                successor = node_successors[successor_index]
                dfs_stack[-1][1] += 1
                if successor not in indices:
                    indices[successor] = next_index
                    lowlink[successor] = next_index
                    next_index += 1
                    tarjan_stack.append(successor)
                    on_stack.add(successor)
                    dfs_stack.append([successor, 0, node_name])
                elif successor in on_stack:
                    lowlink[node_name] = min(lowlink[node_name], indices[successor])
                continue

            dfs_stack.pop()
            if parent is not None:
                lowlink[parent] = min(lowlink[parent], lowlink[node_name])

            if lowlink[node_name] == indices[node_name]:
                component = []
                while True:
                    member = tarjan_stack.pop()
                    on_stack.remove(member)
                    component.append(member)
                    if member == node_name:
                        break
                components.append(tuple(component))

    return tuple(components)


def _component_member_order(component: Set[str], nodes, edges) -> Tuple[str, ...]:
    """Entry-first BFS order for the members of one SCC."""
    node_names = tuple(nodes)
    entries = []
    for name in node_names:
        if name not in component:
            continue
        has_external_input = any(
            edges[key].source not in component for key in nodes[name].node_info.in_edges
        )
        if has_external_input:
            entries.append(name)

    if not entries:
        entries = [next(name for name in node_names if name in component)]

    queue = deque(entries)
    seen = set(entries)
    order = []
    while queue:
        name = queue.popleft()
        order.append(name)
        for edge_key in nodes[name].node_info.out_edges:
            target = edges[edge_key].target
            if target in component and target not in seen:
                seen.add(target)
                queue.append(target)

    # An SCC is reachable from any seed, but retain a deterministic safeguard
    # for malformed custom graph metadata.
    order.extend(name for name in node_names if name in component and name not in seen)
    return tuple(order)


class UnrolledCycleScheduler(TopologySchedulerBase):
    """Visit cyclic SCCs repeatedly while visiting acyclic nodes once.

    This scheduler supplies deterministic propagation for cyclic sPC graphs.
    It is not an ePC time-unrolling scheme; ``EPCInference`` rejects cyclic
    topologies independently of this schedule.
    """

    def __init__(self, num_unrolls: int = 3):
        if isinstance(num_unrolls, bool) or not isinstance(num_unrolls, int):
            raise TypeError("num_unrolls must be an integer")
        if num_unrolls < 1:
            raise ValueError("num_unrolls must be >= 1")
        super().__init__(num_unrolls=num_unrolls)

    @staticmethod
    def compute_schedule(nodes, edges, config) -> Tuple[str, ...]:
        num_unrolls = config["num_unrolls"]
        components = _strongly_connected_components(nodes, edges)
        node_position = {name: index for index, name in enumerate(nodes)}
        normalized_components = [
            tuple(sorted(component, key=node_position.__getitem__))
            for component in components
        ]
        component_of = {
            name: component_index
            for component_index, component in enumerate(normalized_components)
            for name in component
        }

        component_indegree = {index: 0 for index in range(len(components))}
        component_successors = {index: [] for index in range(len(components))}
        seen_component_edges = set()
        for source_name, node in nodes.items():
            source_component = component_of[source_name]
            for edge_key in node.node_info.out_edges:
                target_name = edges[edge_key].target
                target_component = component_of[target_name]
                component_edge = (source_component, target_component)
                if (
                    source_component != target_component
                    and component_edge not in seen_component_edges
                ):
                    seen_component_edges.add(component_edge)
                    component_successors[source_component].append(target_component)
                    component_indegree[target_component] += 1

        queue = deque()
        queued = set()
        for name in nodes:
            component_index = component_of[name]
            if (
                component_indegree[component_index] == 0
                and component_index not in queued
            ):
                queued.add(component_index)
                queue.append(component_index)

        schedule = []
        while queue:
            component_index = queue.popleft()
            component = set(normalized_components[component_index])
            member_order = _component_member_order(component, nodes, edges)
            has_self_edge = any(
                edges[key].target == name
                for name in component
                for key in nodes[name].node_info.out_edges
            )
            repeats = num_unrolls if len(component) > 1 or has_self_edge else 1
            for _ in range(repeats):
                schedule.extend(member_order)

            for target_component in component_successors[component_index]:
                component_indegree[target_component] -= 1
                if component_indegree[target_component] == 0:
                    queue.append(target_component)

        return tuple(schedule)
