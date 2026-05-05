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
    >>> from fabricpc.nodes import Linear
    >>> from fabricpc.core.topology import Edge
    >>> from fabricpc.graph_assembly import TaskMap, graph
    >>> from fabricpc.graph_initialization import initialize_params
    >>> from fabricpc.training import train_pcn, evaluate_pcn
    >>>
    >>> # Define nodes
    >>> input_node = Linear(shape=(784,), name="input")
    >>> hidden = Linear(shape=(128,), name="hidden")
    >>> output = Linear(shape=(10,), name="output")
    >>>
    >>> # Build graph
    >>> structure = graph(
    ...     nodes=[input_node, hidden, output],
    ...     edges=[
    ...         Edge(source=input_node, target=hidden.slot("in")),
    ...         Edge(source=hidden, target=output.slot("in")),
    ...     ],
    ...     task_map=TaskMap(x=input_node, y=output),
    ...     inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    ... )
    >>> params = initialize_params(structure, rng_key)
    >>> trained_params, history, _ = train_pcn(params, structure, train_loader, config)
    >>> metrics = evaluate_pcn(trained_params, structure, test_loader, config)
"""

from importlib.metadata import version

__version__ = version("fabricpc")

# Submodules (for advanced use)
# nodes must precede graph_assembly: graph_assembly imports nodes.base,
# and nodes.transformer_v2 imports back from graph_assembly. Loading nodes
# first ensures nodes.base is fully initialized before graph_assembly runs.
from fabricpc import (
    core,
    graph_initialization,
    nodes,
    training,
    utils,
    graph_assembly,
    experiments,
)

# Core API - what most users need
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import train_pcn, evaluate_pcn

# Types - for type hints
from fabricpc.core.types import GraphParams, GraphState, GraphStructure

__all__ = [
    # Core API (common use)
    "initialize_params",
    "train_pcn",
    "evaluate_pcn",
    # Types (for type hints)
    "GraphParams",
    "GraphState",
    "GraphStructure",
    # Submodules (advanced use)
    "core",
    "graph_assembly",
    "graph_initialization",
    "nodes",
    "training",
    "utils",
    "experiments",
]
