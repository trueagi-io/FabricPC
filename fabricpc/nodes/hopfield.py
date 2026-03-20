"""
Hopfield node for associative memory in predictive coding networks.

Implements a Hopfield memory layer that stores bipolar patterns via the
Hebbian learning rule and recalls them through PC energy minimization.

Architecture (self-loops not allowed, so use a feedback node):

    input_node ──[W_in]──> HopfieldNode ──────> feedback_node
                                ^                      |
                                └──────[W_hop]---------┘

At PC equilibrium: z* = tanh(alpha * x_query + W_hop @ z*)
This IS the continuous Hopfield network with external input drive.

Usage:
    from fabricpc.nodes.hopfield import HopfieldNode, store_patterns, recall

    node = HopfieldNode(shape=(784,), name="memory")
    # ... build graph, initialize params ...
    params = store_patterns(params, patterns, structure)
    recalled = recall(params, structure, noisy_query, rng_key)
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.nodes.identity import IdentityNode
from fabricpc.core.types import NodeParams, NodeState, NodeInfo, GraphParams
from fabricpc.core.activations import TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, ZerosInitializer

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase
    from fabricpc.core.types import GraphStructure


# =============================================================================
# Hopfield Node
# =============================================================================


class HopfieldNode(FlattenInputMixin, NodeBase):
    """
    Hopfield memory node for associative pattern recall.

    A Linear-like node with tanh activation designed for Hopfield networks.
    Receives two inputs:
    - Direct input (query/probe pattern)
    - Recurrent feedback (via a feedback IdentityNode for iterative recall)

    The forward pass computes:
        z_mu = tanh(W_input @ x_query + W_feedback @ z_feedback)

    With Hebbian weights in W_feedback and scaled identity in W_input,
    PC inference converges to the nearest stored attractor pattern.

    Args:
        shape: Pattern dimensionality, e.g. (784,) for MNIST
        name: Node name
        activation: Activation function (default: TanhActivation for continuous Hopfield)
        energy: Energy functional (default: GaussianEnergy)
        use_bias: Whether to include bias (default: False for Hopfield)
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = TanhActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = False,
        latent_init: Optional[InitializerBase] = NormalInitializer(mean=0.0, std=0.01),
        weight_init: Optional[InitializerBase] = NormalInitializer(mean=0.0, std=0.01),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single multi-input slot accepting query + feedback edges."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        """
        Initialize weight matrices for each incoming edge.

        Each edge gets a (in_features, out_features) weight matrix.
        For Hopfield, these will be overwritten by store_patterns().
        """
        from fabricpc.core.initializers import initialize

        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.01)

        key_w, key_b = jax.random.split(key)

        weights_dict = {}
        edge_keys = list(input_shapes.keys())
        w_keys = jax.random.split(key_w, len(edge_keys))

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_features = int(np.prod(in_shape))
            out_features = int(np.prod(node_shape))
            weight_shape = (in_features, out_features)
            weights_dict[edge_key] = initialize(w_keys[i], weight_shape, weight_init)

        use_bias = config.get("use_bias", False)
        biases = {}
        if use_bias:
            out_features = int(np.prod(node_shape))
            biases["b"] = jnp.zeros((out_features,))

        return NodeParams(weights=weights_dict, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Hopfield forward pass: z_mu = activation(sum_edges(W_e @ x_e) + bias).

        Flattens inputs for matrix multiplication, then applies activation.
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape

        # Linear combination of all inputs
        pre_activation = FlattenInputMixin.compute_linear(
            inputs, params.weights, batch_size, out_shape
        )

        # Add bias if present
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation (tanh for continuous Hopfield)
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        # Prediction error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Compute energy and accumulate self-latent gradient
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


# =============================================================================
# Hebbian Storage
# =============================================================================


def hebbian_weight_matrix(patterns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Hebbian weight matrix: W = (1/P) * X^T @ X, zero diagonal.

    Classic Hopfield storage rule. Capacity ~0.14 * N patterns.
    Works best with random/uncorrelated patterns.

    Args:
        patterns: (P, N) bipolar {-1, +1} patterns

    Returns:
        (N, N) symmetric weight matrix with zero diagonal
    """
    P, N = patterns.shape
    W = jnp.dot(patterns.T, patterns) / P
    W = W * (1.0 - jnp.eye(N))
    return W


def pseudoinverse_weight_matrix(patterns: jnp.ndarray) -> jnp.ndarray:
    """
    Compute pseudo-inverse (projection) weight matrix for correlated patterns.

    W = X^T @ (X @ X^T)^{-1} @ X, zero diagonal.

    Unlike Hebbian, this guarantees W @ p_i ≈ p_i for all stored patterns
    even when patterns are highly correlated (e.g., MNIST digit prototypes).

    Args:
        patterns: (P, N) bipolar {-1, +1} patterns

    Returns:
        (N, N) weight matrix with zero diagonal
    """
    P, N = patterns.shape
    # (P, P) Gram matrix + small regularization for numerical stability
    gram = patterns @ patterns.T + 1e-6 * jnp.eye(P)
    gram_inv = jnp.linalg.inv(gram)
    W = patterns.T @ gram_inv @ patterns
    W = W * (1.0 - jnp.eye(N))
    return W


def store_patterns(
    params: GraphParams,
    patterns: jnp.ndarray,
    structure: "GraphStructure",
    memory_node_name: str = "memory",
    input_node_name: str = "input",
    feedback_node_name: str = "feedback",
    input_strength: float = 0.5,
    method: str = "pseudoinverse",
) -> GraphParams:
    """
    Store patterns in the Hopfield network.

    Sets:
    - W_input = input_strength * I  (query injection)
    - W_feedback = storage matrix   (pattern storage)

    Args:
        params: Current graph parameters
        patterns: (P, N) bipolar patterns to store
        structure: Graph structure
        memory_node_name: Name of the HopfieldNode
        input_node_name: Name of the input IdentityNode
        feedback_node_name: Name of the feedback IdentityNode
        input_strength: Scaling for the input drive (default: 0.5)
        method: Storage method — "hebbian" or "pseudoinverse" (default).
                Use "pseudoinverse" for correlated patterns like MNIST.

    Returns:
        Updated GraphParams with stored weights
    """
    P, N = patterns.shape
    if method == "pseudoinverse":
        W_heb = pseudoinverse_weight_matrix(patterns)
    else:
        W_heb = hebbian_weight_matrix(patterns)

    old_memory = params.nodes[memory_node_name]
    new_weights = {}

    for edge_key in old_memory.weights:
        if f"{input_node_name}->{memory_node_name}" in edge_key:
            new_weights[edge_key] = input_strength * jnp.eye(N)
        elif f"{feedback_node_name}->{memory_node_name}" in edge_key:
            new_weights[edge_key] = W_heb
        else:
            # Fallback: keep existing
            new_weights[edge_key] = old_memory.weights[edge_key]

    new_memory = NodeParams(weights=new_weights, biases=old_memory.biases)
    return GraphParams(nodes={**params.nodes, memory_node_name: new_memory})


# =============================================================================
# Graph Construction Helper
# =============================================================================


def build_hopfield_graph(
    pattern_size: int,
    infer_steps: int = 100,
    eta_infer: float = 0.1,
    input_name: str = "input",
    memory_name: str = "memory",
    feedback_name: str = "feedback",
):
    """
    Build a complete Hopfield network graph with recurrent feedback.

    Architecture:
        input (N) --[W_in]--> memory (N) ----> feedback (N)
                                  ^                 |
                                  +----[W_hop]------+

    Args:
        pattern_size: Dimensionality of patterns (e.g. 784 for MNIST)
        infer_steps: Number of PC inference iterations for recall
        eta_infer: Inference learning rate
        input_name: Name for input node
        memory_name: Name for memory node
        feedback_name: Name for feedback node

    Returns:
        GraphStructure ready for initialize_params()
    """
    from fabricpc.builder import Edge, TaskMap, graph
    from fabricpc.core.inference import InferenceSGD
    from fabricpc.graph.state_initializer import GlobalStateInit

    input_node = IdentityNode(shape=(pattern_size,), name=input_name)

    memory_node = HopfieldNode(
        shape=(pattern_size,),
        name=memory_name,
    )

    feedback_node = IdentityNode(shape=(pattern_size,), name=feedback_name)

    structure = graph(
        nodes=[input_node, memory_node, feedback_node],
        edges=[
            Edge(source=input_node, target=memory_node.slot("in")),
            Edge(source=memory_node, target=feedback_node.slot("in")),
            Edge(source=feedback_node, target=memory_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
        graph_state_initializer=GlobalStateInit(),
    )
    return structure


# =============================================================================
# Recall
# =============================================================================


def _recall_core(
    params: GraphParams,
    structure: "GraphStructure",
    query: jnp.ndarray,
    rng_key: jax.Array,
    memory_node_name: str = "memory",
    feedback_node_name: str = "feedback",
):
    """
    Core recall logic: initialize state, run inference, return results.

    Initializes memory and feedback nodes to the query pattern for
    faster convergence to the nearest stored attractor.

    Returns:
        (recalled, final_state, squeeze)
    """
    from fabricpc.core.inference import run_inference
    from fabricpc.graph.state_initializer import initialize_graph_state
    from fabricpc.utils.helpers import update_node_in_state

    squeeze = False
    if query.ndim == 1:
        query = query[None, :]
        squeeze = True

    clamps = {"input": query}

    state = initialize_graph_state(
        structure,
        batch_size=query.shape[0],
        rng_key=rng_key,
        clamps=clamps,
        params=params,
    )

    # Initialize memory and feedback to query for better convergence
    # (starts near the correct basin of attraction)
    state = update_node_in_state(state, memory_node_name, z_latent=query)
    state = update_node_in_state(state, feedback_node_name, z_latent=query)

    final_state = run_inference(params, state, clamps, structure)
    recalled = final_state.nodes[memory_node_name].z_latent

    return recalled, final_state, squeeze


def recall(
    params: GraphParams,
    structure: "GraphStructure",
    query: jnp.ndarray,
    rng_key: jax.Array,
    memory_node_name: str = "memory",
) -> jnp.ndarray:
    """
    Recall stored patterns given a (possibly noisy) query.

    Args:
        params: Graph parameters with stored Hebbian weights
        structure: Graph structure
        query: Query pattern(s), shape (N,) or (batch, N)
        rng_key: Random key for state initialization
        memory_node_name: Name of the memory node

    Returns:
        Recalled pattern(s), same shape as query (continuous values in ~[-1, +1])
    """
    recalled, _, squeeze = _recall_core(
        params, structure, query, rng_key, memory_node_name
    )
    if squeeze:
        recalled = recalled[0]
    return recalled


def recall_with_energy(
    params: GraphParams,
    structure: "GraphStructure",
    query: jnp.ndarray,
    rng_key: jax.Array,
    memory_node_name: str = "memory",
    feedback_node_name: str = "feedback",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Recall stored patterns and return both recalled patterns and energy.

    Args:
        params: Graph parameters with stored Hebbian weights
        structure: Graph structure
        query: Query pattern(s), shape (N,) or (batch, N)
        rng_key: Random key for state initialization
        memory_node_name: Name of the memory node
        feedback_node_name: Name of the feedback node

    Returns:
        recalled: Recalled pattern(s), same shape as query
        energy: Per-sample total energy, shape (batch,) or scalar
    """
    recalled, final_state, squeeze = _recall_core(
        params, structure, query, rng_key, memory_node_name, feedback_node_name
    )

    # Sum energy from all non-source nodes
    batch_size = recalled.shape[0] if recalled.ndim > 1 else 1
    batch_energy = jnp.zeros(batch_size)
    for name, node in structure.nodes.items():
        if node.node_info.in_degree > 0:
            batch_energy = batch_energy + final_state.nodes[name].energy

    if squeeze:
        recalled = recalled[0]
        batch_energy = batch_energy[0]

    return recalled, batch_energy


def pattern_overlap(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Normalized overlap between bipolar patterns: sign(a) . sign(b) / N."""
    return jnp.dot(jnp.sign(a), jnp.sign(b)) / a.shape[-1]


def add_noise(pattern: jnp.ndarray, noise_level: float, rng_key: jax.Array) -> jnp.ndarray:
    """Flip bits with given probability."""
    flip = jax.random.bernoulli(rng_key, p=noise_level, shape=pattern.shape)
    return pattern * (1.0 - 2.0 * flip.astype(jnp.float32))
