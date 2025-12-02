"""
Graph-based predictive coding network construction for JAX with local Hebbian learning.

This module provides functions to build graph structures with node classes,
validate slot connections, and initialize parameters at the node level.

Edges direct the flow of information through the graph in inference and training.
- Node output is passed to input slots of post-synaptic nodes
- Node gets gradient contributions from post-synaptic nodes by querying its out_neighbors on each outgoing edge for the gradient contributions of that particular edge.
"""

from typing import Dict, Tuple, List, Any
import jax
import jax.numpy as jnp
from fabricpc.core.types import (
    NodeInfo,
    EdgeInfo,
    NodeParams,
    GraphParams,
    NodeState,
    GraphState,
    GraphStructure,
    SlotInfo,
)
from fabricpc.core.initialization import (
    initialize_state_values,
    parse_state_init_config,
    get_default_state_init,
)
from fabricpc.nodes import get_node_class_from_type
from fabricpc.core.inference import gather_inputs
from fabricpc.utils.helpers import update_node_in_state


def validate_node_and_build_slots(
    node_config: dict,
    node_name: str,
    edges: Dict[str, EdgeInfo],
) -> Dict[str, SlotInfo]:
    """
    Build and validate slots for a node based on its type and incoming edges.

    Args:
        node_config: Node configuration dictionary
        node_name: Name of the node
        edges: Dictionary of all edges in the graph

    Returns:
        Dictionary mapping slot names to SlotInfo objects

    Raises:
        ValueError: If edges connect to non-existent slots or violate slot constraints
    """
    default_node = "Linear"  # TODO change default to empty or raise error when missing from config
    node_type = node_config.get("type", default_node).lower()
    node_class = get_node_class_from_type(node_type)

    # Get slot specifications from node class
    slot_specs = node_class.get_slots()

    # Build SlotInfo objects
    slots = {}
    for slot_name, slot_spec in slot_specs.items():
        # Find incoming edges for this slot
        in_neighbors = []
        for edge_key, edge_info in edges.items():
            if edge_info.target == node_name and edge_info.slot == slot_name:
                in_neighbors.append(edge_info.source)

        # Validate slot constraints
        if not slot_spec.is_multi_input and len(in_neighbors) > 1:
            raise ValueError(
                f"Slot '{slot_name}' in node '{node_name}' is single-input "
                f"but has {len(in_neighbors)} connections"
            )

        slots[slot_name] = SlotInfo(
            name=slot_name,
            parent_node=node_name,
            is_multi_input=slot_spec.is_multi_input,
            in_neighbors=tuple(in_neighbors),
        )

    # Validate that all incoming edges connect to valid slots
    for edge_key, edge_info in edges.items():
        if edge_info.target == node_name:
            if edge_info.slot not in slots:
                raise ValueError(
                    f"Edge '{edge_key}' connects to non-existent slot '{edge_info.slot}' "
                    f"in node '{node_name}'. Available slots: {list(slots.keys())}"
                )

    return slots


def build_graph_structure(config: dict) -> GraphStructure:
    """
    Convert configuration dictionary to static GraphStructure with slot validation.

    Expected config format:
    {
        "node_list": [
            {
                "name": "x",
                "dim": 784,
                "type": "linear",
                "activation": {"type": "identity"},
                "weight_init": {"type": "normal", "std": 0.05}  # Optional
            },
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
        Immutable GraphStructure with validated slots

    Raises:
        ValueError: If configuration is invalid or slots don't match
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
        # find the duplicate nodes
        duplicates = set(
            name for name in node_names if node_names.count(name) > 1
        )
        raise ValueError(f"duplicated nodes {duplicates}, names must be unique")

    # Build edge dictionary first
    edges: Dict[str, EdgeInfo] = {}
    for edge_config in edge_list:
        source = edge_config["source_name"]
        target = edge_config["target_name"]
        # Default slot is "in" for backward compatibility
        slot = edge_config.get("slot", "in")

        # Create edge key: source->target:slot
        edge_key = f"{source}->{target}:{slot}"

        if edge_key in edges:
            raise ValueError(f"duplicate edge: {edge_key}")
        if source == target:
            raise ValueError(f"self-edge at: {source} is not allowed")
        if source not in node_names:
            raise ValueError(f"edge source node '{source}' does not exist")
        if target not in node_names:
            raise ValueError(f"edge target node '{target}' does not exist")

        edges[edge_key] = EdgeInfo(key=edge_key, source=source, target=target, slot=slot)

    # Build nodes with validated slots
    nodes: Dict[str, NodeInfo] = {}

    for node_config in node_list:
        name = node_config["name"]
        dim = node_config["dim"]
        node_type = node_config.get("type", "linear").lower()
        activation_config = node_config.get("activation", {"type": "identity"})

        # Validate and build slots
        slots = validate_node_and_build_slots(node_config, name, edges)

        # Find incoming and outgoing edges
        in_edges: List[str] = []
        out_edges: List[str] = []

        for edge_key, edge_info in edges.items():
            if edge_info.target == name:
                in_edges.append(edge_key)
            if edge_info.source == name:
                out_edges.append(edge_key)

        # Construct the node object
        nodes[name] = NodeInfo(
            name=name,
            dim=dim,
            node_type=node_type,
            node_config=node_config,
            activation_config=activation_config,
            slots=slots,
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


def topological_sort(
    nodes: Dict[str, NodeInfo], edges: Dict[str, EdgeInfo]
) -> Tuple[str, ...]:
    """
    Compute topological ordering of nodes for feedforward initialization.

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
        # Graph contains cycles
        print("Warning: Graph contains cycles, using partial topological order")

    return tuple(result)


def initialize_params(
    structure: GraphStructure,
    rng_key: jax.Array,  # from jax.random.PRNGKey
) -> GraphParams:
    """
    Initialize model parameters at the node level.

    Each node class handles its own parameter initialization,
    supporting complex nodes with multiple internal parameters.

    Args:
        structure: Graph structure
        rng_key: JAX random key

    Returns:
        GraphParams with node-based parameter organization
    """
    node_params = {}  # type: Dict[str, NodeParams]

    # Split key for each node
    num_nodes = len([n for n in structure.nodes.values() if n.in_degree > 0])
    if num_nodes > 0:
        keys = jax.random.split(rng_key, num_nodes)
    else:
        keys = []
    key_idx = 0

    for node_name, node_info in structure.nodes.items():
        # Skip source nodes (no parameters)
        if node_info.in_degree == 0:
            node_params[node_name] = NodeParams(weights={}, biases={})
            continue

        # Get node class
        node_class = get_node_class_from_type(node_info.node_type)

        # Get the input dimensions for each edge
        input_dims = {}
        for edge_key in node_info.in_edges:
            edge_info = structure.edges[edge_key]
            source_node = structure.nodes[edge_info.source]
            total_dim = source_node.dim
            input_dims[edge_key] = total_dim

        # Initialize parameters of the node
        params_obj = node_class.initialize_params(
            keys[key_idx],
            node_info.dim,
            input_dims,
            node_info.node_config
        )
        key_idx += 1
        node_params[node_name] = params_obj

    return GraphParams(nodes=node_params)


def initialize_state(
    structure: GraphStructure,
    batch_size: int,
    rng_key: jax.Array,
    clamps: Dict[str, jnp.ndarray] = None,
    state_init_config: Dict[str, Any] = None,
    params: GraphParams = None,
) -> GraphState:
    """
    Initialize graph state for inference.

    Args:
        structure: Graph structure
        batch_size: Batch size
        rng_key: JAX random key for state initialization
        clamps: Optional dictionary of clamped values, keyed on node names
        state_init_config: State initialization configuration dict
        params: GraphParams (required for feedforward init)

    Returns:
        Initial GraphState with latent gradients
    """
    clamps = clamps or {}

    # Use default if not provided
    if state_init_config is None:
        state_init_config = get_default_state_init()

    # Parse initialization config
    init_method, fallback_config = parse_state_init_config(state_init_config)

    # Split rng_key for each node
    node_names = list(structure.nodes.keys())
    node_keys = jax.random.split(rng_key, len(node_names))
    node_key_map = dict(zip(node_names, node_keys))
    node_state_dict = {}

    # Initialize all nodes, respecting clamps
    for node_name, node_info in structure.nodes.items():
        shape = (batch_size, node_info.dim)

        # Initialize z_latent
        if node_name in clamps:
            # Use clamped value
            z_latent = clamps[node_name]
        elif init_method == "zeros":
            z_latent = jnp.zeros(shape)
        elif init_method in ["uniform", "normal"]:
            # Direct initialization with split key
            z_latent = initialize_state_values(
                fallback_config, node_key_map[node_name], shape
            )
        elif init_method == "feedforward":
            # Initialize with fallback first using split key
            z_latent = initialize_state_values(
                fallback_config, node_key_map[node_name], shape
            )
        else:
            raise ValueError(f"unknown init_method: {init_method}")

        # Initialize latents, set other state components to zeros
        node_state_dict[node_name] = NodeState(
            z_latent=z_latent,  # init latent state
            z_mu=jnp.zeros(shape),
            error=jnp.zeros(shape),
            energy=jnp.zeros((batch_size,)),  # Per-sample energy
            pre_activation=jnp.zeros(shape),
            gain_mod_error=jnp.zeros(shape),
            latent_grad=jnp.zeros(shape),
            substructure={},
        )

    # Create a draft graph state
    state = GraphState(
        nodes=node_state_dict,
        batch_size=batch_size,
    )
    # Feedforward initialization if requested
    if init_method == "feedforward" and params is not None:

        # Process nodes in topological order
        for node_name in structure.node_order:
            node_info = structure.nodes[node_name]
            if node_info.in_degree > 0:

                # Collect edge inputs
                node_state = state.nodes[node_name]
                node_params = params.nodes[node_name]
                node_class = get_node_class_from_type(node_info.node_type)
                edge_inputs = gather_inputs(node_info, structure, state)

                # Compute forward projection
                _, node_state = node_class.forward(node_params, edge_inputs, node_state, node_info)

                # Update the state with feedforward values
                if node_name not in clamps:
                    # z_latent <- z_mu_init
                    node_state = node_state._replace(z_latent=node_state.z_mu)
                else:
                    # Respect clamped values
                    node_state = node_state._replace(z_latent=clamps[node_name])
                # Update state
                state = state._replace(nodes={**state.nodes, node_name: node_state})

    return state

def set_latents_to_clamps(
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
) -> GraphState:
    """
    Set the latent states of specified nodes to their clamped values.

    Args:
        state: Current graph state
        clamps: Dictionary of clamped values, keyed on node names

    Returns:
        Updated GraphState with latents set to clamped values
    """
    for node_name, clamp_value in clamps.items():
        if node_name in state.nodes:
            state = update_node_in_state(state, node_name, z_latent=clamp_value)
    return state

def create_pc_graph(
    config: dict,
    rng_key: jax.Array,  # from jax.random.PRNGKey
) -> Tuple[GraphParams, GraphStructure]:
    """
    Create a complete PC graph with local Hebbian learning.

    This is the main entry point for creating a JAX PC model with the new architecture.

    Args:
        config: Configuration dictionary with node_list, edge_list, task_map
        rng_key: JAX random key for initialization

    Returns:
        Tuple of (params, structure)

    Example:
        >>> config = {
        ...     "node_list": [
        ...         {
        ...             "name": "pixels",
        ...             "dim": 784,
        ...             "type": "linear",
        ...             "activation": {"type": "identity"},
        ...             "weight_init": {"type": "xavier"}
        ...         },
        ...         {
        ...             "name": "hidden",
        ...             "dim": 256,
        ...             "type": "linear",
        ...             "activation": {"type": "relu"}
        ...         },
        ...         {
        ...             "name": "class",
        ...             "dim": 10,
        ...             "type": "linear",
        ...             "activation": {"type": "softmax"}
        ...         },
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
    params = initialize_params(structure, rng_key)
    return params, structure