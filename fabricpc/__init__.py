"""
FabricPC-JAX: Predictive Coding Networks in JAX
================================================

A functional, high-performance implementation of predictive coding networks
using JAX for automatic differentiation, JIT compilation, and multi-device parallelism.

Key Features:
- Functional programming paradigm (immutable data structures)
- JIT-compiled inference and training loops
- Multi-GPU/TPU support with pmap
- XLA optimization for maximum performance

Example:
    >>> from fabricpc.graph import create_pc_graph
    >>> from fabricpc.training import train_pcn
    >>>
    >>> params, structure = create_pc_graph(config)
    >>> params = train_pcn(params, structure, train_loader, config)
"""

__version__ = "0.2.0"

from fabricpc import core, graph, nodes, training
from fabricpc.core import types, activations, inference, initialization
from fabricpc.graph import graph_net
from fabricpc.nodes import base, linear
from fabricpc.training import train, optimizers, multi_gpu

__all__ = [
    "core",
    "models",
    "nodes",
    "training",
    "types",
    "activations",
    "inference",
    "initialization",
    "graph_net",
    "base",
    "linear",
    "train",
    "optimizers",
    "multi_gpu",
]
