"""Graph construction and parameter utilities for object-based FabricPC graphs."""

from typing import Dict, Tuple, Any, Optional, Sequence
import jax
import jax.numpy as jnp
from fabricpc.core.types import (
    EdgeInfo,
    NodeParams,
    GraphParams,
    NodeState,
    GraphState,
    GraphStructure,
)
from fabricpc.utils.helpers import update_node_in_state
from fabricpc.core.inference import gather_inputs
from fabricpc.graph.state_initializer import FeedforwardStateInit


def build_graph_structure(
    config: Optional[dict] = None,
    *,
    nodes: Optional[Sequence[Any]] = None,
    edges: Optional[Sequence[EdgeInfo]] = None,
    task_map: Optional[Dict[str, str]] = None,
    graph_state_initializer: Optional[Any] = None,
) -> GraphStructure:
    """Build GraphStructure from concrete nodes and edges."""
    if config is not None:
        raise ValueError(
            "Config-dict graph construction has been removed. "
            "Construct node objects directly and pass nodes/edges."
        )

    if nodes is None or edges is None:
        raise ValueError("object mode requires `nodes` and `edges`")

    if graph_state_initializer is None:
        graph_state_initializer = FeedforwardStateInit()

    return GraphStructure.from_objects(
        nodes=nodes,
        edges=edges,
        task_map=task_map,
        graph_state_initializer=graph_state_initializer,
    )


def compute_local_weight_gradients(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """Compute local weight gradients for each node using its own error signal."""
    gradients = {}

    for node_name, node_info in structure.nodes.items():
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        in_edges_data = gather_inputs(node_info, structure, final_state)
        node_cls = node_info.node.__class__
        _, grad_params = node_cls.forward_learning(
            params.nodes[node_name],
            in_edges_data,
            final_state.nodes[node_name],
            node_info,
        )

        gradients[node_name] = grad_params

    return GraphParams(nodes=gradients)


def initialize_params(
    structure: GraphStructure,
    rng_key: jax.Array,
) -> GraphParams:
    """Initialize model parameters at the node level."""
    node_params = {}  # type: Dict[str, NodeParams]

    num_nodes = len([n for n in structure.nodes.values() if n.in_degree > 0])
    keys = jax.random.split(rng_key, num_nodes) if num_nodes > 0 else []
    key_idx = 0

    for node_name, node_info in structure.nodes.items():
        if node_info.in_degree == 0:
            node_params[node_name] = NodeParams(weights={}, biases={})
            continue

        input_shapes = {}
        for edge_key in node_info.in_edges:
            edge_info = structure.edges[edge_key]
            source_node = structure.nodes[edge_info.source]
            input_shapes[edge_key] = source_node.shape

        node_cls = node_info.node.__class__
        params_obj = node_cls.initialize_params(
            keys[key_idx], node_info.shape, input_shapes, node_info.node_config
        )
        key_idx += 1
        node_params[node_name] = params_obj

    return GraphParams(nodes=node_params)


def set_latents_to_clamps(
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
) -> GraphState:
    """Set latent states of specified nodes to clamped values."""
    for node_name, clamp_value in clamps.items():
        if node_name in state.nodes:
            state = update_node_in_state(state, node_name, z_latent=clamp_value)
    return state


def create_pc_graph(
    config: Optional[dict] = None,
    rng_key: Optional[jax.Array] = None,
    *,
    nodes: Optional[Sequence[Any]] = None,
    edges: Optional[Sequence[EdgeInfo]] = None,
    task_map: Optional[Dict[str, str]] = None,
    graph_state_initializer: Optional[Any] = None,
) -> Tuple[GraphParams, GraphStructure]:
    """Create a complete PC graph from node objects and edge references."""
    if rng_key is None:
        raise ValueError("`rng_key` is required")

    structure = build_graph_structure(
        config=config,
        nodes=nodes,
        edges=edges,
        task_map=task_map,
        graph_state_initializer=graph_state_initializer,
    )
    params = initialize_params(structure, rng_key)
    return params, structure
