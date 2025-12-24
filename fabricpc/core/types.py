"""
Core JAX types for predictive coding networks.

All types are immutable and registered as JAX pytrees for automatic differentiation.
"""

from typing import Dict, Any, Tuple, NamedTuple
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotInfo:
    """Metadata for an input slot to a node."""
    name: str  # Slot name (e.g., "in")
    parent_node: str  # Name of the parent node
    is_multi_input: bool  # True if slot accepts multiple edges, False for single edge
    in_neighbors: Tuple[str, ...]  # Tuple of node names connecting to this slot

@dataclass(frozen=True)
class NodeInfo:
    """Metadata for a single node in the graph.

    The shape field represents the output dimensions of the node, excluding
    the batch dimension. Batch is always the first dimension in arrays
    and is stored in GraphState.batch_size at runtime.

    Examples:
        - Linear layer: shape=(128,) for 128-dimensional output
        - Image: shape=(28, 28, 1) for 28x28 grayscale image (NHWC format)
        - Sequence: shape=(100, 64) for 100 timesteps, 64 features
    """

    name: str
    shape: Tuple[int, ...]  # Output shape excluding batch dimension
    node_type: str  # "linear", "transformer", etc.
    node_config: Dict[str, Any]  # Includes "activation" and "energy" sub-configs
    slots: Dict[str, SlotInfo]  # {"in": SlotInfo, ...}
    in_degree: int  # Number of incoming edges
    out_degree: int  # Number of outgoing edges
    in_edges: Tuple[str, ...]  # Tuple of edge keys
    out_edges: Tuple[str, ...]  # Tuple of edge keys

@dataclass(frozen=True)
class EdgeInfo:
    """Metadata for a single edge in the graph."""

    key: str  # "source->target:slot"
    source: str
    target: str
    slot: str

    # Config schema for edge configuration (used during graph construction)
    CONFIG_SCHEMA: Dict[str, Any] = None  # Class attribute, set below

    @classmethod
    def from_config(cls, edge_config: dict) -> "EdgeInfo":
        """
        Construct an EdgeInfo from config.

        Args:
            edge_config: Edge configuration dictionary

        Returns:
            Validated EdgeInfo instance

        Raises:
            ConfigValidationError: If validation fails
        """
        from fabricpc.core.config import validate_config, ConfigValidationError

        validated = validate_config(cls.CONFIG_SCHEMA, edge_config, context="edge")

        source = validated["source_name"]
        target = validated["target_name"]
        slot = validated["slot"]
        edge_key = f"{source}->{target}:{slot}"

        if source == target:
            raise ValueError(f"self-edge at: {source} is not allowed")

        return cls(key=edge_key, source=source, target=target, slot=slot)


# Set CONFIG_SCHEMA as a class attribute (can't be done inside frozen dataclass)
EdgeInfo.CONFIG_SCHEMA = {
    "source_name": {
        "type": str,
        "required": True,
        "description": "Name of the source node"
    },
    "target_name": {
        "type": str,
        "required": True,
        "description": "Name of the target node"
    },
    "slot": {
        "type": str,
        "default": "in",
        "description": "Target slot name"
    }
}

class NodeParams(NamedTuple):
    """Parameters for a single node (weights, biases, etc.)."""
    weights: Dict[str, jnp.ndarray]  # Named weight matrices, where name identifies the substructure of the node for the parameters
    biases: Dict[str, jnp.ndarray]   # Named bias vectors

class GraphParams(NamedTuple):
    """
    Learnable parameters of the predictive coding network.

    Now organized by node rather than edge, supporting complex nodes with
    multiple internal parameters (e.g., transformer blocks).

    Attributes:
        nodes: Dictionary mapping node names to their parameters
            Each node has a dict of weights and a dict of biases
            Complex nodes may have multiple named weight/bias matrices
    """

    nodes: Dict[str, NodeParams]  # {node_name: NodeParams}

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        total_params = 0
        for node_params in self.nodes.values():
            if "weights" in node_params._fields:
                total_params += sum(w.size for w in node_params.weights.values())
            if "biases" in node_params._fields:
                total_params += sum(b.size for b in node_params.biases.values())
        return f"GraphParams(nodes={n_nodes}, total_params={total_params})"

class NodeState(NamedTuple):
    """
    Dynamic state of the Node during inference.

    Attributes:
        z_latent: Latent states (what the network infers)
        z_mu: Predicted expectations (what the network predicts)
        error: Prediction errors (z_latent - z_mu)
        energy: Energy
        pre_activation: Pre-activation values (before activation function)
        latent_grad: Gradients w.r.t. latent states for inference updates
        substructure: Dictionary of node internal states for complex nodes
    """

    z_latent: jnp.ndarray
    z_mu: jnp.ndarray
    error: jnp.ndarray
    energy: jnp.ndarray  # per-sample energy, shape (batch_size,)
    pre_activation: jnp.ndarray
    latent_grad: jnp.ndarray  # For local gradient accumulation
    substructure: Dict[str, jnp.ndarray]  # substructure of node internal states

class GraphState(NamedTuple):
    """
    Dynamic state of the network during inference.

    All states are dictionaries mapping node names to arrays.

    Attributes:
        nodes: Dictionary mapping node names to NodeState
        batch_size: Current batch size
    """
    nodes: Dict[str, NodeState]  # {node_name: NodeState}
    batch_size: int

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        return f"GraphState(nodes={n_nodes}, batch_size={self.batch_size})"

class GraphStructure(NamedTuple):
    """
    Static graph topology (compile-time constant).

    This structure is immutable and marked as static during JIT compilation.

    Attributes:
        nodes: Dictionary mapping node names to NodeInfo
        edges: Dictionary mapping edge keys to EdgeInfo
        task_map: Dictionary mapping task names to node names
        node_order: Topological order for forward pass
        config: Graph configuration
    """

    nodes: Dict[str, NodeInfo]
    edges: Dict[str, EdgeInfo]
    task_map: Dict[str, str]
    node_order: Tuple[str, ...]  # Topological sort for inference
    config: Dict[str, Any]  # Graph configuration

    # CONFIG_SCHEMA defined below class

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        return f"GraphStructure(nodes={n_nodes}, edges={n_edges})"

    @classmethod
    def from_config(cls, config: dict) -> "GraphStructure":
        """
        Construct a GraphStructure from configuration dictionary.

        Args:
            config: Configuration dictionary with node_list, edge_list, task_map

        Returns:
            Immutable GraphStructure with validated components

        Raises:
            ValueError: If graph is misspecified
            ConfigValidationError: If config validation fails
        """
        # Lazy imports to avoid circular dependencies
        from fabricpc.core.config import validate_config, ConfigValidationError
        from fabricpc.nodes import get_node_class, validate_node_config, NodeBase

        # 1. Validate graph-level config
        validated_graph = validate_config(
            cls.CONFIG_SCHEMA, config, context="graph"
        )

        node_list = validated_graph["node_list"]
        edge_list = validated_graph["edge_list"]
        task_map = validated_graph["task_map"]

        # 2. Build edges - delegate to EdgeInfo.from_config()
        edges: Dict[str, EdgeInfo] = {}
        for edge_config in edge_list:
            edge = EdgeInfo.from_config(edge_config)
            if edge.key in edges:
                raise ConfigValidationError(f"duplicate edge: {edge.key}")
            edges[edge.key] = edge

        # 3. Build nodes - delegate to node class from_config()
        nodes: Dict[str, NodeInfo] = {}
        for node_config in node_list:
            # Validate base requirements (name, shape, type)
            validated_base = validate_node_config(NodeBase, node_config)
            name = validated_base["name"]
            node_type = validated_base["type"].lower()

            # Validate unique node names
            if name in nodes:
                raise ValueError(f"duplicate node '{name}', names must be unique")

            # Find edges for this node
            in_edges_dict = {k: e for k, e in edges.items() if e.target == name}
            out_edges_dict = {k: e for k, e in edges.items() if e.source == name}

            # Delegate node construction to node class
            node_class = get_node_class(node_type)
            nodes[name] = node_class.from_config(node_config, in_edges_dict, out_edges_dict)

        # 4. Validate edge endpoints exist
        for edge_key, edge_info in edges.items():
            if edge_info.source not in nodes:
                raise ValueError(f"edge source node '{edge_info.source}' does not exist")
            if edge_info.target not in nodes:
                raise ValueError(f"edge target node '{edge_info.target}' does not exist")

        # 5. Compute topological order
        node_order = cls._topological_sort(nodes, edges)

        return cls(nodes=nodes, edges=edges, task_map=task_map, node_order=node_order, config=validated_graph)

    @staticmethod
    def _topological_sort(
        nodes: Dict[str, "NodeInfo"],
        edges: Dict[str, "EdgeInfo"]
    ) -> Tuple[str, ...]:
        """
        Compute topological ordering of nodes for feedforward traversal.

        Args:
            nodes: Dictionary of node information
            edges: Dictionary of edge information

        Returns:
            Tuple of node names in topological order

        Note:
            If the graph contains cycles, some nodes may be omitted from the order.
        """
        # Count in-degrees
        in_degree = {name: info.in_degree for name, info in nodes.items()}

        # Queue of nodes, begin with nodes having no incoming edges
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            node_name = queue.pop(0)
            result.append(node_name)

            # Reduce in-degree of neighbors
            node_info = nodes[node_name]
            for out_edge_key in node_info.out_edges:
                edge_info = edges[out_edge_key]
                target_name = edge_info.target
                in_degree[target_name] -= 1

                if in_degree[target_name] == 0:
                    # Dependencies have been processed, now add next node to the queue
                    queue.append(target_name)

        if len(result) != len(nodes):
            print("Warning: Graph contains cycles, using partial topological order")

        return tuple(result)

# Config schema for graph configuration (validated during graph construction)
# Node and edge configs are validated by their respective class CONFIG_SCHEMAs
GraphStructure.CONFIG_SCHEMA = {
    "node_list": {
        "type": list,
        "required": True,
        "description": "List of node configurations"
    },
    "edge_list": {
        "type": list,
        "required": True,
        "description": "List of edge configurations"
    },
    "task_map": {
        "type": dict,
        "required": True,
        "description": "Mapping of task names to node names"
    },
    "graph_state_initializer": {
        "type": dict,
        "default": {"type": "feedforward"},
        "description": "Graph-level latent state initialization configuration"
    }
}


# Register as pytrees for JAX transformations
tree_util.register_pytree_node(
    GraphParams,
    lambda gp: ((gp.nodes,), None),
    lambda aux, children: GraphParams(*children),
)

tree_util.register_pytree_node(
    NodeParams,
    lambda np: ((np.weights, np.biases), None),
    lambda aux, children: NodeParams(*children),
)

tree_util.register_pytree_node(
    NodeState,
    lambda ns: (
        (ns.z_latent, ns.z_mu, ns.error, ns.energy, ns.pre_activation, ns.latent_grad, ns.substructure),
        None,
    ),
    lambda aux, children: NodeState(*children),
)

tree_util.register_pytree_node(
    GraphState,
    lambda gs: (
        (gs.nodes,),        # Dynamic children (differentiable)
        (gs.batch_size,),   # Static auxiliary data (metadata)
    ),
    lambda aux, children: GraphState(children[0], aux[0]),
)

# GraphStructure is static, so we register it as having no dynamic components
tree_util.register_pytree_node(
    GraphStructure,
    lambda gs: ((), (gs.nodes, gs.edges, gs.task_map, gs.node_order, gs.config)),
    lambda aux, _: GraphStructure(*aux),  # Reconstruct from aux data
)
