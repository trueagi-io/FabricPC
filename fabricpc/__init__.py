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
    >>> from fabricpc import create_pc_graph, train_pcn, evaluate_pcn
    >>>
    >>> params, structure = create_pc_graph(config, rng_key)
    >>> trained_params, history, _ = train_pcn(params, structure, train_loader, config)
    >>> metrics = evaluate_pcn(trained_params, structure, test_loader, config)
"""

from importlib.metadata import version

__version__ = version("fabricpc")

# Submodules (for advanced use)
from fabricpc import core, graph, nodes, training, utils

# Core API - what most users need
from fabricpc.graph import create_pc_graph
from fabricpc.training import train_pcn, evaluate_pcn

# Types - for type hints
from fabricpc.core.types import GraphParams, GraphState, GraphStructure

__all__ = [
    # Core API (common use)
    "create_pc_graph",
    "train_pcn",
    "evaluate_pcn",
    # Types (for type hints)
    "GraphParams",
    "GraphState",
    "GraphStructure",
    # Submodules (advanced use)
    "core",
    "graph",
    "nodes",
    "training",
    "utils",
]
