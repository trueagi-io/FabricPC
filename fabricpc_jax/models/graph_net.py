"""
Graph-based predictive coding network construction for JAX.

This module provides functions to build graph structures, initialize parameters,
and create complete PC models from configuration dictionaries.
"""

from typing import Dict, Tuple, List, Any
import jax
import jax.numpy as jnp

from fabricpc_jax.core.types import (
    NodeInfo,
    EdgeInfo,
    GraphParams,
    GraphState,
    GraphStructure,
)
from fabricpc_jax.core.initialization import (
    initialize_weights,
    initialize_state_values,
    parse_state_init_config,
    get_default_weight_init,
    get_default_state_init,
)


def build_graph_structure(config: dict) -> GraphStructure:
    """
    Convert configuration dictionary to static GraphStructure.

    Expected config format (matches PyTorch API):
    {
        "node_list": [
            {"name": "x", "dim": 784, "type": "linear", "activation": {"type": "identity"}},
            {"name": "h", "dim": 256, "type": "linear", "activation": {"type": "relu"}},
            ...
        ],
        "edge_list": [
            {"source_name": "x", "target_name": "h", "slot": "in"},
            ...
        ],
        "task_map": {
            "x": "x",  # task_name -> node_name
            "y": "y"
        }
    }

    Args:
        config: Configuration dictionary

    Returns:
        Immutable GraphStructure

    Notes:
        - Node types supported: "linear" (currently the only node type)
        - Slot names: "in" for standard input (empty string "" also supported for backward compatibility)
        - Activation types: "identity", "sigmoid", "tanh", "relu", "leaky_relu", "hard_tanh"
    """
    if "node_list" not in config:
        raise ValueError("config['node_list'] is required")
    if "edge_list" not in config:
        raise ValueError("config['edge_list'] is required")
    if "task_map" not in config:
        raise ValueError("config['task_map'] is required")

    node_list = config["node_list"]
    edge_list = config["edge_list"]
    task_map = config["task_map"]

    # Validate unique node names
    node_names = [node["name"] for node in node_list]
    if len(node_names) != len(set(node_names)):
        raise ValueError("Node names must be unique")

    # Build edge dictionary first
    edges: Dict[str, EdgeInfo] = {}
    for edge_config in edge_list:
        source = edge_config["source_name"]
        target = edge_config["target_name"]
        # Default slot is "in" to match PyTorch API (empty string for backward compatibility)
        slot = edge_config.get("slot", "in")

        # Create edge key: source->target:slot
        edge_key = f"{source}->{target}:{slot}"

        if edge_key in edges:
            raise ValueError(f"Duplicate edge: {edge_key}")
        if source == target:
            raise ValueError(f"Self-loops not allowed: {source}")

        edges[edge_key] = EdgeInfo(key=edge_key, source=source, target=target, slot=slot)

    # Build nodes with neighbor information
    nodes: Dict[str, NodeInfo] = {}

    for node_config in node_list:
        name = node_config["name"]
        dim = node_config["dim"]
        activation_config = node_config["activation"]

        # Validate node type (optional but recommended)
        node_type = node_config.get("type", "")
        if node_type.lower() not in ["linear"]:
            raise ValueError(
                f"Node '{name}' has unsupported type '{node_type}'. "
                f"Currently only 'linear' node type is supported."
            )

        # Find incoming and outgoing edges
        in_edges: List[str] = []
        out_edges: List[str] = []

        for edge_key, edge_info in edges.items():
            if edge_info.target == name:
                in_edges.append(edge_key)
            if edge_info.source == name:
                out_edges.append(edge_key)

        nodes[name] = NodeInfo(
            name=name,
            dim=dim,
            node_config=node_config,
            activation=activation_config,
            in_degree=len(in_edges),
            out_degree=len(out_edges),
            in_edges=tuple(in_edges),
            out_edges=tuple(out_edges),
        )

    # Compute topological order for efficient traversal
    node_order = topological_sort(nodes, edges)

    return GraphStructure(
        nodes=nodes, edges=edges, task_map=task_map, node_order=node_order
    )

def change_node_order(structure: GraphStructure, node_order: Tuple[str, ...]):
    """
    Override the topological order of nodes for feedforward initialization. Can be useful for cyclic graphs.
    Fallback initialization will be applied first to any nodes absent from the node_order tuple.
    Must recompile the model after calling this function.
    """
    # Structure is immutable, create a new one
    return GraphStructure(
        nodes=structure.nodes,
        edges=structure.edges,
        task_map=structure.task_map,
        node_order=node_order
    )

def topological_sort(
    nodes: Dict[str, NodeInfo], edges: Dict[str, EdgeInfo]
) -> Tuple[str, ...]:
    """
    Compute topological ordering of nodes for feedforward initialization
        Args:
        nodes: Dictionary of node information
        edges: Dictionary of edge information

    Returns:
        Tuple of node names in topological order
    """
    # Count in-degrees
    in_degree = {name: info.in_degree for name, info in nodes.items()}

    # Queue of nodes with no incoming edges
    queue = [name for name, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        # Pop a node with no incoming edges
        node_name = queue.pop(0)
        result.append(node_name)

        # Reduce in-degree of neighbors
        node_info = nodes[node_name]
        for out_edge_key in node_info.out_edges:
            edge_info = edges[out_edge_key]
            target_name = edge_info.target
            in_degree[target_name] -= 1

            if in_degree[target_name] == 0:
                queue.append(target_name)

    if len(result) != len(nodes):
        raise Warning("Graph contains cycles, some nodes will be randomly initialized")
    else:
        print("Graph is a DAG, all nodes will be initialized in topological order")

    return tuple(result)


def initialize_params(
    structure: GraphStructure,
    key: jax.random.PRNGKey,
    weight_init_config: Dict[str, Any] = None,
) -> GraphParams:
    """
    Initialize model parameters (weights and biases).

    Args:
        structure: Graph structure
        key: JAX random key
        weight_init_config: Weight initialization configuration dict
            Examples:
            - {"type": "normal", "mean": 0, "std": 0.05}
            - {"type": "uniform", "min": -0.1, "max": 0.1}
            - {"type": "zeros"}

    Returns:
        Initialized GraphParams
    """
    weights = {}
    biases = {}

    # Use default if not provided
    if weight_init_config is None:
        weight_init_config = get_default_weight_init()

    # Split key for each parameter
    num_params = len(structure.edges) + len(structure.nodes)
    keys = jax.random.split(key, num_params)
    key_idx = 0

    # Initialize weights for each edge
    for edge_key, edge_info in structure.edges.items():
        source_node = structure.nodes[edge_info.source]
        target_node = structure.nodes[edge_info.target]

        # Compute input dimension for target node (sum of all incoming source dims)
        in_dims = []
        for in_edge_key in target_node.in_edges:
            in_edge_info = structure.edges[in_edge_key]
            in_dims.append(structure.nodes[in_edge_info.source].dim)
        total_in_dim = sum(in_dims)

        # Weight matrix: (total_in_dim, target_dim)
        # All incoming edges to a node share the same weight matrix
        if edge_key == target_node.in_edges[0]:
            # Only initialize once for the first incoming edge
            W = initialize_weights(
                weight_init_config, keys[key_idx], (total_in_dim, target_node.dim)
            )
            key_idx += 1
        else:
            # Reuse weight matrix from first edge
            first_edge_key = target_node.in_edges[0]
            W = weights[first_edge_key]

        weights[edge_key] = W

    # Initialize biases for each node
    for node_name, node_info in structure.nodes.items():
        # Bias vector: (1, dim) for broadcasting
        b = jnp.zeros((1, node_info.dim))
        biases[node_name] = b

    return GraphParams(weights=weights, biases=biases)


def initialize_state(
    structure: GraphStructure,
    batch_size: int,
    clamps: Dict[str, jnp.ndarray] = None,
    state_init_config: Dict[str, Any] = None,
    params: GraphParams = None,
) -> GraphState:
    """
    Initialize graph state for inference.

    Args:
        structure: Graph structure
        batch_size: Batch size
        clamps: Optional dictionary of clamped values
        state_init_config: State initialization configuration dict
            Examples:
            - {"type": "zeros"}
            - {"type": "normal", "mean": 0, "std": 0.01}
            - {"type": "feedforward", "fallback": {"type": "normal", "std": 0.01}}
        params: Parameters (required for feedforward init)

    Returns:
        Initial GraphState
    """
    z_latent = {}
    z_mu = {}
    error = {}
    pre_activation = {}
    gain_mod_error = {}

    clamps = clamps or {}

    # Use default if not provided
    if state_init_config is None:
        state_init_config = get_default_state_init()

    # Parse initialization config
    init_method, fallback_config = parse_state_init_config(state_init_config)

    # Initialize all nodes
    for node_name, node_info in structure.nodes.items():
        shape = (batch_size, node_info.dim)

        # Initialize z_latent
        if node_name in clamps:
            # Use clamped value
            z_latent[node_name] = clamps[node_name]
        elif init_method == "zeros":
            z_latent[node_name] = jnp.zeros(shape)
        elif init_method == "uniform":
            # Direct uniform initialization
            key = jax.random.PRNGKey(hash(node_name) % (2**32))
            z_latent[node_name] = initialize_state_values(fallback_config, key, shape)
        elif init_method == "normal":
            # Direct normal initialization
            key = jax.random.PRNGKey(hash(node_name) % (2**32))
            z_latent[node_name] = initialize_state_values(fallback_config, key, shape)
        elif init_method == "feedforward":
            # Initialize with fallback first, will be overwritten in feedforward pass
            key = jax.random.PRNGKey(hash(node_name) % (2**32))
            z_latent[node_name] = initialize_state_values(fallback_config, key, shape)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # Initialize other state components to zeros
        z_mu[node_name] = jnp.zeros(shape)
        error[node_name] = jnp.zeros(shape)
        pre_activation[node_name] = jnp.zeros(shape)
        gain_mod_error[node_name] = jnp.zeros(shape)

    # Feedforward initialization (like PyTorch version)
    if init_method == "feedforward" and params is not None:
        from fabricpc_jax.core.inference import compute_projection

        # Process nodes in topological order
        for node_name in structure.node_order:
            if node_name not in clamps:
                # Compute forward projection
                z_mu_init, pre_act_init = compute_projection(
                    params, z_latent, node_name, structure
                )
                z_latent[node_name] = z_mu_init
                z_mu[node_name] = z_mu_init
                pre_activation[node_name] = pre_act_init

    return GraphState(
        z_latent=z_latent,
        z_mu=z_mu,
        error=error,
        pre_activation=pre_activation,
        gain_mod_error=gain_mod_error,
    )


def create_pc_graph(
    config: dict,
    key: jax.random.PRNGKey,
    weight_init_config: Dict[str, Any] = None
) -> Tuple[GraphParams, GraphStructure]:
    """
    Create a complete PC graph from configuration.

    This is the main entry point for creating a JAX PC model.

    Args:
        config: Configuration dictionary with node_list, edge_list, task_map
        key: JAX random key for initialization
        weight_init_config: Weight initialization configuration dict
            Examples:
            - {"type": "normal", "mean": 0, "std": 0.05}
            - {"type": "uniform", "min": -0.1, "max": 0.1}
            - None (uses default: normal with std=0.05)

    Returns:
        Tuple of (params, structure)

    Example:
        >>> config = {
        ...     "node_list": [
        ...         {"name": "pixels", "dim": 784, "type": "linear", "activation": {"type": "identity"}},
        ...         {"name": "hidden", "dim": 256, "type": "linear", "activation": {"type": "sigmoid"}},
        ...         {"name": "class", "dim": 10, "type": "linear", "activation": {"type": "sigmoid"}},
        ...     ],
        ...     "edge_list": [
        ...         {"source_name": "pixels", "target_name": "hidden", "slot": "in"},
        ...         {"source_name": "hidden", "target_name": "class", "slot": "in"},
        ...     ],
        ...     "task_map": {"x": "pixels", "y": "class"}
        ... }
        >>> params, structure = create_pc_graph(config, jax.random.PRNGKey(0))
    """
    structure = build_graph_structure(config)
    params = initialize_params(structure, key, weight_init_config)
    return params, structure
