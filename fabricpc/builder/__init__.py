"""Builder primitives for constructing predictive coding graphs."""

from fabricpc.builder.edge import Edge, SlotRef
from fabricpc.builder.namespace import GraphNamespace
from fabricpc.builder.graph_builder import graph, TaskMap

# TODO rename this module to graph_builder and rename graph_builder.py to graph_construction.py or something like that. The current name is too generic and doesn't reflect the fact that this module contains the main graph construction function (graph) and related primitives (Edge, SlotRef, TaskMap).
__all__ = [
    "Edge",
    "SlotRef",
    "GraphNamespace",
    "graph",
    "TaskMap",
]
