"""Builder primitives for constructing predictive coding graphs."""

from fabricpc.builder.edge import Edge, SlotRef
from fabricpc.builder.namespace import GraphNamespace
from fabricpc.builder.graph_builder import graph, TaskMap

__all__ = [
    "Edge",
    "SlotRef",
    "GraphNamespace",
    "graph",
    "TaskMap",
]
