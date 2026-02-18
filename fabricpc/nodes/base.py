"""
Base node classes for JAX predictive coding networks.

This module provides the abstract base class for all node types, defining the
interface for custom transfer functions, multiple input slots, and local gradient computation.
All node methods are pure functions (no side effects) for JAX compatibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
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

    Example usage:
        @register_node("my_dense")
        class MyDenseNode(FlattenInputMixin, NodeBase):
            @staticmethod
            def forward(params, inputs, state, node_info):
                batch_size = state.z_latent.shape[0]
                out_shape = node_info.shape

                # Flatten inputs and compute linear transformation
                pre_activation = FlattenInputMixin.compute_linear(
                    inputs, params.weights, batch_size, out_shape
                )
                ...
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

        Flattens each input, applies matmul with corresponding weight matrix,
        accumulates results, and reshapes to output shape.

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

    All methods are pure functions (no side effects) for JAX compatibility.
    Nodes can have multiple input slots and custom transfer functions.

    Required class attributes (validated during registration):
        - CONFIG_SCHEMA: dict specifying node configuration validation
        - DEFAULT_ENERGY_CONFIG: dict specifying default energy functional

    Example:
        @register_node("my_node")
        class MyNode(NodeBase):
            CONFIG_SCHEMA = {
                "my_param": {"type": int, "default": 10}
            }
            ...
    """

    # Base config schema for all nodes - defines required fields
    BASE_CONFIG_SCHEMA: Dict[str, Any] = {
        "name": {"type": str, "required": True, "description": "Node name"},
        "shape": {"type": tuple, "required": True, "description": "Output shape"},
        "type": {"type": str, "required": True, "description": "Node type"},
    }

    # CONFIG_SCHEMA is required - subclasses must define it
    # Use empty dict {} if no additional config parameters are needed
    CONFIG_SCHEMA: Dict[str, Any]

    # DEFAULT_ENERGY_CONFIG - Can be overridden by default in subclass and per-node via node_config["energy"]
    DEFAULT_ENERGY_CONFIG: Dict[str, Any] = {"type": "gaussian"}

    # Default activation config - can be overridden by subclass or per-node via node_config["activation"]
    DEFAULT_ACTIVATION_CONFIG: Dict[str, Any] = {"type": "identity"}

    # Node-level state initialization config
    DEFAULT_LATENT_INIT: Dict[str, Any] = {"type": "normal"}

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
        Describe the weights and biases structure in your docstring.

        Args:
            key: JAX random key
            node_shape: Output shape of this node (excluding batch dimension)
            input_shapes: Dictionary mapping edge keys to source node shapes
            config: Node configuration (may contain initialization settings)

        Returns:
            NodeParams with initialized weights and biases

        Example:
            from fabricpc.core.initializers import initialize
            in_features = next(iter(input_shapes.values()))[-1]  # Assuming single input edge and last dimension is feature size
            out_features = node_shape[-1]

            weights = {"a->b:in": initialize(config, key, (in_features, out_features))}
            biases = {"bias": jnp.zeros((1, out_features))}
            return NodeParams(weights=weights, biases=biases)
        """
        pass

    @staticmethod
    def forward_inference(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        state: NodeState,  # state object for the present node
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
            state: state object for the present node
            node_info: NodeInfo object (contains activation function, etc.)
            is_clamped: Whether this node is clamped to data

        Returns:
            Tuple of (NodeState, gradient_wrt_inputs):
                - NodeState: updated node state (z_mu, pre_activation, etc.)
                - gradient_wrt_inputs: dictionary of gradients w.r.t. each input edge
        """
        from fabricpc.nodes import get_node_class

        node_class = get_node_class(node_info.node_type)

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
                node_class.forward, argnums=1, has_aux=True  # inputs
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
        state: NodeState,  # state object for the present node
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass: update state and compute gradients of weights for local learning.
        # Don't override this method. Instead, implement forward() and JAX will handle the gradients.

        The local gradient for weights is: delE/delW

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
            state: state object for the present node
            node_info: NodeInfo object

        Returns:
            Tuple of (NodeState, params_grad):
                - NodeState: updated node state (z_mu, pre_activation, etc.)
                - params_grad: NodeParams containing weight and bias gradients
        """
        from fabricpc.nodes import get_node_class

        node_class = get_node_class(node_info.node_type)

        # Use JAX's value_and_grad to compute gradients w.r.t. params
        (total_energy, new_state), params_grad = jax.value_and_grad(
            node_class.forward, argnums=0, has_aux=True  # params
        )(params, inputs, state, node_info)

        return new_state, params_grad

    @staticmethod
    @abstractmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # EdgeInfo.key -> inputs data
        state: NodeState,  # state object for the present node
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        """
        Forward pass through the node, returning energy scalar and updated state.
        Computes:
            forward projection -> compute error & update state -> compute energy & update state -> total energy

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: state object for the present node
            node_info: NodeInfo object (contains activation function, etc.)

        Returns:
            Tuple of (total_energy, NodeState):
                - total_energy: scalar energy value for this node
                - NodeState: updated node state (z_mu, pre_activation, etc.)
        """
        pass
        """ 
        # example
        
        # handle source nodes
        if node_info.in_degree == 0:
            # Source nodes: no inputs
            z_mu = state.z_latent  # prediction is the latent state itself
            pre_activation = jnp.zeros_like(state.z_latent)
            error = jnp.zeros_like(state.z_latent)
        else:
            pre_activation = zeros_like(state.z_latent)
            for edge_key, x in inputs.items():
                pre_activation += jnp.matmul(x, params.weights[edge_key])
            if "bias" in params:
                pre_activation += params.biases["bias"]
            
            # Apply activation function
            activation_fn, _ = get_activation(node_info.node_config["activation"])
            z_mu = activation_fn(pre_activation)
    
            # Error
            error = state.z_latent - z_mu

        # Update node state before computing energy
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error
        )

        # Compute energy, accumulate the self-latent gradient
        state = node_class.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state
        """

    @staticmethod
    def energy_functional(state: NodeState, node_info: NodeInfo) -> NodeState:
        """
        Compute the energy and the derivative with respect to the node's latent state.
        # Don't override this method. Instead, extend base class EnergyFunctional and specify your class in the node config. This method will look up the energy functional from the registry and apply it.

        The energy functional to use is determined by node_info.node_config["energy"].
        If not specified, the node class's DEFAULT_ENERGY_CONFIG is used during
        graph construction.

        Energy config format:
            {
                "energy": {
                    "type": "gaussian",  # or "binaryce", "crossentropy", etc.
                    "precision": 1.0     # energy-specific parameters
                }
            }

        Args:
            state: NodeState object (contains z_latent, z_mu, etc.)
            node_info: NodeInfo object (contains energy config in node_config)

        Returns:
            Updated NodeState with:
                energy: energy value per batch element, shape (batch_size,)
                latent_grad: derivative of energy w.r.t. z_latent (accumulated)
        """
        from fabricpc.core.energy import get_energy_and_gradient

        # Get energy config from node_config (should be set during graph construction)
        energy_config = node_info.node_config.get("energy", None)
        if energy_config is None:
            raise ValueError(
                f"graph was improperly constructed. Node {node_info.name} is missing default energy functional."
            )

        # Compute energy and gradient using the energy registry
        energy, grad = get_energy_and_gradient(
            state.z_latent, state.z_mu, energy_config
        )

        # Accumulate gradient with existing latent_grad
        latent_grad = state.latent_grad + grad

        # Update node state
        state = state._replace(energy=energy, latent_grad=latent_grad)

        return state

    @staticmethod
    def get_energy_functional(energy_name: str):
        """
        Retrieve an energy functional class by name.

        Args:
            energy_name: Name of the energy functional (e.g., "gaussian", "bernoulli")

        Returns:
            The EnergyFunctional class

        Example:
            energy_class = NodeBase.get_energy_functional("bernoulli")
            energy = energy_class.energy(z_latent, z_mu, config)
        """
        from fabricpc.core.energy import get_energy_class

        return get_energy_class(energy_name)

    @classmethod
    def from_config(
        cls,
        node_config: Dict[str, Any],
        in_edges: Dict[str, EdgeInfo],
        out_edges: Dict[str, EdgeInfo],
    ) -> NodeInfo:
        """
        Validate config and construct node components.
        # Don't override this method. Instead, implement get_slots() and set DEFAULT_ENERGY_CONFIG, DEFAULT_ACTIVATION_CONFIG, and DEFAULT_LATENT_INIT as needed.

        This method centralizes node construction logic, delegating subnode
        construction (slots, energy, activation) to the object's class.

        Args:
            node_config: Raw node configuration dictionary
            in_edges: Dictionary of incoming edges (edge_key -> EdgeInfo)
            out_edges: Dictionary of outgoing edges (edge_key -> EdgeInfo)

        Returns:
            NodeInfo object with validated config and constructed components

        Raises:
            ValueError: If slot constraints are violated
            ConfigValidationError: If config validation fails
        """
        from fabricpc.nodes.registry import validate_node_config

        # 1. Validate node-specific config against CONFIG_SCHEMA
        validated_config = validate_node_config(cls, node_config)

        node_name = validated_config["name"]

        # 2. Build and validate slots
        slots = cls._build_slots(node_name, in_edges)

        # 3. Resolve energy config (user override or class default)
        validated_config["energy"] = cls._resolve_energy_config(node_config)

        # 4. Resolve activation config (user override or class default)
        validated_config["activation"] = cls._resolve_activation_config(node_config)

        # 5. Resolve state initialization config (user override or class default)
        validated_config["latent_init"] = cls._resolve_state_init_config(node_config)

        # Construct the node object with validated config (includes defaults)
        return NodeInfo(
            name=node_name,
            shape=tuple(validated_config["shape"]),
            node_type=validated_config["type"],
            node_config=validated_config,
            slots=slots,
            in_degree=len(in_edges),
            out_degree=len(out_edges),
            in_edges=tuple(in_edges.keys()),
            out_edges=tuple(out_edges.keys()),
        )

    @classmethod
    def _build_slots(
        cls, node_name: str, in_edges: Dict[str, EdgeInfo]
    ) -> Dict[str, SlotInfo]:
        """
        Build SlotInfo objects from slot specs and incoming edges.

        Args:
            node_name: Name of this node
            in_edges: Dictionary of incoming edges (edge_key -> EdgeInfo)

        Returns:
            Dictionary mapping slot names to SlotInfo objects

        Raises:
            ValueError: If slot constraints are violated (e.g., multiple inputs to single-input slot)
        """
        slot_specs = cls.get_slots()
        slots = {}

        for slot_name, slot_spec in slot_specs.items():
            # Find node names that are in-neighbors to this slot
            in_neighbors = [
                edge.source for edge in in_edges.values() if edge.slot == slot_name
            ]

            # Validate constraints
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
        for edge_key, edge in in_edges.items():
            if edge.slot not in slots:
                raise ValueError(
                    f"Edge '{edge_key}' connects to non-existent slot '{edge.slot}' "
                    f"in node '{node_name}'. Available slots: {list(slots.keys())}"
                )

        return slots

    @classmethod
    def _resolve_energy_config(cls, node_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve energy config with validation.

        Uses node_config["energy"] if specified, otherwise falls back to
        cls.DEFAULT_ENERGY_CONFIG. Validates against energy class CONFIG_SCHEMA.

        Args:
            node_config: Node configuration dictionary

        Returns:
            Validated energy config dict
        """
        from fabricpc.core.config import transform_shorthand
        from fabricpc.core.energy import get_energy_class, validate_energy_config

        energy_config = node_config.get("energy")

        if energy_config is None:
            energy_config = cls.DEFAULT_ENERGY_CONFIG.copy()
        elif isinstance(energy_config, str):
            energy_config = {"type": energy_config}

        energy_class = get_energy_class(energy_config["type"])
        return validate_energy_config(energy_class, energy_config)

    @classmethod
    def _resolve_activation_config(cls, node_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve activation config with validation.

        Uses node_config["activation"] if specified, otherwise falls back to
        cls.DEFAULT_ACTIVATION_CONFIG. Validates against activation class CONFIG_SCHEMA.

        Args:
            node_config: Node configuration dictionary

        Returns:
            Validated activation config dict
        """
        from fabricpc.core.activations import (
            get_activation_class,
            validate_activation_config,
        )

        act_config = node_config.get("activation")

        if act_config is None:
            act_config = cls.DEFAULT_ACTIVATION_CONFIG.copy()
        elif isinstance(act_config, str):
            act_config = {"type": act_config}

        act_class = get_activation_class(act_config["type"])
        return validate_activation_config(act_class, act_config)

    @classmethod
    def _resolve_state_init_config(cls, node_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve state initialization config with validation.

        Uses node_config["latent_init"] if specified, otherwise base node default. Validates against InitializerBase class CONFIG_SCHEMA.

        Args:
            node_config: Node configuration dictionary

        Returns:
            Validated state initialization config dict, or None if not specified
        """
        from fabricpc.core.initializers import (
            get_initializer_class,
            validate_initializer_config,
        )

        state_init_config = node_config.get("latent_init")

        if state_init_config is None:
            state_init_config = cls.DEFAULT_LATENT_INIT.copy()
        elif isinstance(state_init_config, str):
            state_init_config = {"type": state_init_config}

        init_class = get_initializer_class(state_init_config["type"])
        return validate_initializer_config(init_class, state_init_config)
