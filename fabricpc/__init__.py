"""FabricPC-JAX: Predictive Coding Networks in JAX."""

from importlib.metadata import version

__version__ = version("fabricpc")

from fabricpc import core, graph, nodes, training, utils

from fabricpc.graph import create_pc_graph
from fabricpc.training import train_pcn, evaluate_pcn

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.types import EdgeInfo
from fabricpc.nodes import LinearNode, TransformerBlockNode, LinearExplicitGrad

__all__ = [
    "create_pc_graph",
    "train_pcn",
    "evaluate_pcn",
    "GraphParams",
    "GraphState",
    "GraphStructure",
    "EdgeInfo",
    "LinearNode",
    "TransformerBlockNode",
    "LinearExplicitGrad",
    "core",
    "graph",
    "nodes",
    "training",
    "utils",
]
