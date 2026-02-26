"""
Base node classes for JAX predictive coding networks.

This module provides the abstract base class for all node types, defining the
interface for custom transfer functions, multiple input slots, and local gradient computation.
All node methods are pure functions (no side effects) for JAX compatibility.

User Extensibility
------------------
Users can create custom nodes by extending NodeBase:

    class MyNode(NodeBase):
        DEFAULT_ACTIVATION = IdentityActivation
        DEFAULT_ENERGY = GaussianEnergy
        DEFAULT_LATENT_INIT = NormalInitializer

        def __init__(self, shape, name, activation=None, energy=None, **kwargs):
            super().__init__(shape=shape, name=name, activation=activation,
                             energy=energy, **kwargs)

        @staticmethod
        def get_slots():
            return {"in": SlotSpec(name="in", is_multi_input=True)}

        @staticmethod
        def initialize_params(key, node_shape, input_shapes, config):
            ...

        @staticmethod
        def forward(params, inputs, state, node_info):
            ...
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import copy
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from fabricpc.core.types import NodeParams, NodeState, NodeInfo, SlotInfo, EdgeInfo


@dataclass(frozen=True)
class SlotSpec:
    """Specification for an input slot to a node."""

    name: str
    is_multi_input: bool  # True = multiple inputs allowed, False = single input only


@dataclass(frozen=True)
class Slot:
    """Runtime slot information with connected edges."""

    spec: SlotSpec
    in_neighbors: Dict[str, str]  # edge_key -> source_node_name mapping


class FlattenInputMixin:
    """
    Mixin providing flatten/reshape utilities for dense (fully-connected) nodes.

    Use this mixin when your node needs to:
    - Flatten arbitrary-shaped inputs to 2D for matrix multiplication
    - Reshape flat outputs back to a target shape
    """

    @staticmethod
    def flatten_input(x: jnp.ndarray) -> jnp.ndarray:
        """
        Flatten input tensor to 2D: (batch, *shape) -> (batch, numel).

        Args:
            x: Input tensor with batch dimension first

        Returns:
            Flattened tensor of shape (batch, numel)
        """
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    @staticmethod
    def reshape_output(x_flat: jnp.ndarray, out_shape: Tuple[int, ...]) -> jnp.ndarray:
        """
        Reshape flat tensor to target shape: (batch, numel) -> (batch, *out_shape).

        Args:
            x_flat: Flat tensor of shape (batch, numel)
            out_shape: Target shape (excluding batch dimension)

        Returns:
            Reshaped tensor of shape (batch, *out_shape)
        """
        batch_size = x_flat.shape[0]
        return x_flat.reshape(batch_size, *out_shape)

    @staticmethod
    def compute_linear(
        inputs: Dict[str, jnp.ndarray],
        weights: Dict[str, jnp.ndarray],
        batch_size: int,
        out_shape: Tuple[int, ...],
    ) -> jnp.ndarray:
        """
        Compute linear transformation: sum of (flattened_input @ weight) for each edge.

        Args:
            inputs: Dictionary mapping edge keys to input tensors
            weights: Dictionary mapping edge keys to weight matrices (in_numel, out_numel)
            batch_size: Batch size for output initialization
            out_shape: Target output shape (excluding batch)

        Returns:
            Pre-activation tensor of shape (batch, *out_shape)
        """
        import numpy as np

        out_numel = int(np.prod(out_shape))
        pre_activation_flat = jnp.zeros((batch_size, out_numel))

        for edge_key, x in inputs.items():
            x_flat = FlattenInputMixin.flatten_input(x)
            pre_activation_flat = pre_activation_flat + jnp.matmul(
                x_flat, weights[edge_key]
            )

        return FlattenInputMixin.reshape_output(pre_activation_flat, out_shape)


class NodeBase(ABC):
    """
    Abstract base class for all predictive coding nodes.

    All computation methods are pure functions (static, no side effects) for JAX
    compatibility. Nodes can have multiple input slots and custom transfer functions.

    Nodes are instantiated with their configuration, then finalized by the
    graph() builder which attaches topology info via copy-on-finalize.

    Class-level defaults (override in subclasses):
        DEFAULT_ACTIVATION: Activation class (e.g., IdentityActivation)
        DEFAULT_ENERGY: Energy class (e.g., GaussianEnergy)
        DEFAULT_LATENT_INIT: Initializer class (e.g., NormalInitializer)
    """

    # Subclasses should override these with activation/energy/initializer CLASSES (not instances)
    # The graph builder will call these with () to create instances if the user didn't provide one
    DEFAULT_ACTIVATION = None  # Set in subclass, e.g., IdentityActivation
    DEFAULT_ENERGY = None  # Set in subclass, e.g., GaussianEnergy
    DEFAULT_LATENT_INIT = None  # Set in subclass, e.g., NormalInitializer

    def __init__(
        self,
        shape,
        name,
        activation=None,
        energy=None,
        latent_init=None,
        **extra_config,
    ):
        """
        Initialize a node descriptor.

        Args:
            shape: Output shape tuple (excluding batch dimension)
            name: Node name. Automatically prefixed with current GraphNamespace.
            activation: ActivationBase instance, or None (uses class DEFAULT_ACTIVATION)
            energy: EnergyFunctional instance, or None (uses class DEFAULT_ENERGY)
            latent_init: InitializerBase instance, or None (uses class DEFAULT_LATENT_INIT)
            **extra_config: Node-specific config (use_bias, flatten_input, etc.)
        """
        from fabricpc.builder.namespace import _get_current_namespace

        ns = _get_current_namespace()
        self._name = f"{ns}/{name}" if ns else name
        self._shape = tuple(shape)
        self._activation = activation
        self._energy = energy
        self._latent_init = latent_init
        self._extra_config = extra_config
        self._node_info = None  # Set by graph builder (copy-on-finalize)

    @property
    def name(self) -> str:
        """Node name, including namespace prefix if any."""
        return self._name

    @property
    def shape(self) -> Tuple[int, ...]:
        """Output shape excluding batch dimension."""
        return self._shape

    @property
    def node_info(self) -> NodeInfo:
        """NodeInfo with topology info. None until graph() is called."""
        return self._node_info

    def slot(self, slot_name: str):
        """
        Create a SlotRef for connecting edges to a specific slot.

        Args:
            slot_name: Name of the slot (e.g., "in", "mask")

        Returns:
            SlotRef pointing to this node's slot

        Raises:
            KeyError: If slot_name is not defined for this node type
        """
        from fabricpc.builder.edge import SlotRef

        slot_specs = type(self).get_slots()
        if slot_name not in slot_specs:
            raise KeyError(
                f"Node '{self._name}' has no slot '{slot_name}'. "
                f"Available: {list(slot_specs.keys())}"
            )
        return SlotRef(node=self, slot=slot_name)

    def _with_graph_info(self, node_info: NodeInfo) -> "NodeBase":
        """
        Copy-on-finalize: return a shallow copy with graph topology info attached.

        The original node object is not modified.
        """
        new = copy.copy(self)
        new._node_info = node_info
        return new

    # =========================================================================
    # Abstract methods - subclasses must implement
    # =========================================================================

    @staticmethod
    @abstractmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Define the input slots for this node type.
        Create as many named input slots as needed, and specify whether each slot allows multiple inputs.
        Don't set is_multi_input=True unless you intend to aggregate an arbitrary number of inputs to a single named slot and create appropriate parameters and forward logic to handle that.

        Returns:
            Dictionary mapping slot names to SlotSpec objects

        Example:
            return {
                "in": SlotSpec(name="in", is_multi_input=True),
                "gate": SlotSpec(name="gate", is_multi_input=False)
            }
        """
        pass

    @staticmethod
    @abstractmethod
    def initialize_params(
        key: jax.Array,  # from jax.random.PRNGKey
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],  # edge_key -> source shape
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Define and initialize the parameters required for the node.

        Args:
            key: JAX random key
            node_shape: Output shape of this node (excluding batch dimension)
            input_shapes: Dictionary mapping edge keys to source node shapes
            config: Node configuration (may contain initialization settings)

        Returns:
            NodeParams with initialized weights and biases
        """
        pass

    @staticmethod
    @abstractmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # keyed on EdgeInfo.key -> inputs data
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        """
        Forward pass through the node, returning energy scalar and updated state.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: NodeState for this node
            node_info: NodeInfo object (contains activation, energy, etc.)

        Returns:
            Tuple of (total_energy, NodeState)
                - total_energy: scalar energy value for this node
                - NodeState: updated node state (z_mu, pre_activation, etc.)
        """
        pass

    # =========================================================================
    # Default implementations - can be overridden for explicit gradients
    # =========================================================================

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray]]:
        """
        Forward pass: updates node state and computes gradients w.r.t. inputs.
        Don't override this method. Instead, implement forward() and JAX will handle the gradients.

        PC has two contributions to latent gradients during inference:
        1. Gradient w.r.t. inputs (delE/delX): used to update the latent states of in-neighbor nodes during inference.
        2. Gradient w.r.t. node's self latent state (delE/delZ): computed in the energy functional and accumulated to the node gradient (latent_grad).

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: NodeState for this node
            node_info: NodeInfo object
            is_clamped: Whether this node is clamped to data

        Returns:
            Tuple of (NodeState, gradient_wrt_inputs):
                - NodeState: updated node state (z_mu, pre_activation, etc.)
                - gradient_wrt_inputs: dictionary of gradients w.r.t. each input edge
        """
        node_class = node_info.node_class

        # Handle terminal nodes
        if node_info.in_degree == 0:
            # No inputs!
            # This is a terminal input node of the graph. It might be clamped to data, or it might be a source of top-down predictions. Either way, the gradients are zero.

            # Update z_mu <-- z_latent, so error is zero.
            new_state = state._replace(
                z_mu=state.z_latent,
                error=jnp.zeros_like(state.error),
                pre_activation=jnp.zeros_like(state.pre_activation),
            )
            # Update the state's energy and self-latent gradient; will be zero since z_mu = z_latent
            new_state = node_class.energy_functional(new_state, node_info)
            # Gradient to inputs is zero since there are no inputs
            input_grads = {
                edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs
            }

        elif node_info.out_degree == 0 and not is_clamped:
            # No post-synaptic targets and no clamped data!
            # This happens for output nodes when the model is run in inference/evaluation mode (not training)
            # Compute its projection (z_mu) but no gradient since it doesn't contribute to any error.
            total_energy, new_state = node_class.forward(
                params, inputs, state, node_info
            )
            # Update keeping the projection, but zero error.
            new_state = new_state._replace(
                z_latent=new_state.z_mu,
                error=jnp.zeros_like(new_state.error),
                energy=jnp.zeros_like(new_state.energy),
                latent_grad=jnp.zeros_like(new_state.latent_grad),
            )
            input_grads = {
                edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs
            }

        else:
            # Internal node or a clamped output node. Compute the energy and gradients.
            # Use JAX's value_and_grad to compute gradients w.r.t. inputs
            (total_energy, new_state), input_grads = jax.value_and_grad(
                node_class.forward, argnums=1, has_aux=True
            )(params, inputs, state, node_info)
            # TODO if using preactivation latents, need to wrap the node_class.forward() with method to apply pre-synaptic activation function to the inputs first.
            # TODO Refactor node_class.forward()
            #   - node_class.forward only computes the projection z_mu
            #   - Remove pre-activation from NodeState; it's unnecessary to store!
            #   - Wrapper method here computes:
            #       - error = state.z_latent - z_mu
            #       - state = node_class.energy_functional(state, node_info)
            #       - total_energy = jnp.sum(state.energy)

        return new_state, input_grads

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass: update state and compute gradients of weights for local learning.
        # Don't override this method. Instead, implement forward() and JAX will handle the gradients.

        The local gradient for weights is: delE/delW

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
            state: NodeState for this node
            node_info: NodeInfo object

        Returns:
            Tuple of (NodeState, params_grad):
                - NodeState: updated node state (z_mu, pre_activation, etc.)
                - params_grad: NodeParams containing weight and bias gradients
        """
        node_class = node_info.node_class

        # Use JAX's value_and_grad to compute gradients w.r.t. params
        (total_energy, new_state), params_grad = jax.value_and_grad(
            node_class.forward, argnums=0, has_aux=True
        )(params, inputs, state, node_info)

        return new_state, params_grad

    @staticmethod
    def energy_functional(state: NodeState, node_info: NodeInfo) -> NodeState:
        """
        Compute energy and its derivative w.r.t. the node's latent state.

        Uses the energy instance stored in node_info.energy.

        Args:
            state: NodeState object (contains z_latent, z_mu, etc.)
            node_info: NodeInfo object (contains energy instance)

        Returns:
            Updated NodeState with energy and latent_grad
        """
        energy_obj = node_info.energy
        if energy_obj is None:
            raise ValueError(
                f"Node '{node_info.name}' has no energy functional configured."
            )

        energy_cls = type(energy_obj)
        config = energy_obj.config

        energy = energy_cls.energy(state.z_latent, state.z_mu, config)
        grad = energy_cls.grad_latent(state.z_latent, state.z_mu, config)

        latent_grad = state.latent_grad + grad
        state = state._replace(energy=energy, latent_grad=latent_grad)

        return state
