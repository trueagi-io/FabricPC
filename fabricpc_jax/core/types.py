"""
Core JAX types for predictive coding networks.

All types are immutable and registered as JAX pytrees for automatic differentiation.
"""

from typing import Dict, Any, Tuple, NamedTuple
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeInfo:
    """Metadata for a single node in the graph."""

    name: str
    dim: int
    activation_config: Dict[str, Any]  # {"type": "sigmoid", ...}
    node_config: Dict[str, Any]
    slot: Dict[str, Any]  # {"in": "multi", ...}
    in_degree: int  # Number of incoming edges
    out_degree: int  # Number of outgoing edges
    in_edges: Tuple[str, ...]  # Tuple of edge keys
    out_edges: Tuple[str, ...]  # Tuple of edge keys

@dataclass(frozen=True)
class SlotInfo:
    """Metadata for an input slot to a node."""
    name: str
    parent_node: str
    is_multi_input: bool
    in_neighbors: Tuple[str, ...]  # Tuple of source node names

@dataclass(frozen=True)
class EdgeInfo:
    """Metadata for a single edge in the graph."""

    key: str  # "source->target:slot"
    source: str
    target: str
    slot: str


class GraphParams(NamedTuple):
    """
    Learnable parameters of the predictive coding network.

    Attributes:
        weights: Dictionary mapping edge keys to weight matrices
        biases: Dictionary mapping node names to bias vectors
    """

    weights: Dict[str, jnp.ndarray]  # {edge_key: weight_matrix}
    biases: Dict[str, jnp.ndarray]  # {node_name: bias_vector}

    def __repr__(self) -> str:
        n_weights = len(self.weights)
        n_biases = len(self.biases)
        total_params = sum(w.size for w in self.weights.values()) + sum(
            b.size for b in self.biases.values()
        )
        return f"GraphParams(edges={n_weights}, nodes={n_biases}, total_params={total_params})"


class GraphState(NamedTuple):
    """
    Dynamic state of the network during inference.

    All states are dictionaries mapping node names to arrays.

    Attributes:
        z_latent: Latent states (what the network infers)
        z_mu: Predicted expectations (what the network predicts)
        error: Prediction errors (z_latent - z_mu)
        pre_activation: Pre-activation values (before activation function)
        gain_mod_error: Gain-modulated errors (error * activation_derivative)
    """

    z_latent: Dict[str, jnp.ndarray]
    z_mu: Dict[str, jnp.ndarray]
    error: Dict[str, jnp.ndarray]
    pre_activation: Dict[str, jnp.ndarray]
    gain_mod_error: Dict[str, jnp.ndarray]

    def __repr__(self) -> str:
        n_nodes = len(self.z_latent)
        batch_size = next(iter(self.z_latent.values())).shape[0] if self.z_latent else 0
        return f"GraphState(nodes={n_nodes}, batch_size={batch_size})"


class GraphStructure(NamedTuple):
    """
    Static graph topology (compile-time constant).

    This structure is immutable and marked as static during JIT compilation.

    Attributes:
        nodes: Dictionary mapping node names to NodeInfo
        edges: Dictionary mapping edge keys to EdgeInfo
        task_map: Dictionary mapping task names to node names
        node_order: Topological order for forward pass
    """

    nodes: Dict[str, NodeInfo]
    edges: Dict[str, EdgeInfo]
    task_map: Dict[str, str]
    node_order: Tuple[str, ...]  # Topological sort for inference

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        return f"GraphStructure(nodes={n_nodes}, edges={n_edges})"


# Register as pytrees for JAX transformations
tree_util.register_pytree_node(
    GraphParams,
    lambda gp: ((gp.weights, gp.biases), None),
    lambda aux, children: GraphParams(*children),
)

tree_util.register_pytree_node(
    GraphState,
    lambda gs: (
        (gs.z_latent, gs.z_mu, gs.error, gs.pre_activation, gs.gain_mod_error),
        None,
    ),
    lambda aux, children: GraphState(*children),
)

# GraphStructure is static, so we register it as having no dynamic components
tree_util.register_pytree_node(
    GraphStructure,
    lambda gs: ((), (gs.nodes, gs.edges, gs.task_map, gs.node_order)),
    lambda aux, _: GraphStructure(*aux),  # Reconstruct from aux data
)
