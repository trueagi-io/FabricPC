"""
Base node classes for JAX predictive coding networks.

This module provides the abstract base class for all node types, defining the
interface for custom transfer functions, multiple input slots, and local gradient computation.
All node methods are pure functions (no side effects) for JAX compatibility.

User Extensibility
------------------
Users can create custom nodes by extending NodeBase:

    class MyNode(NodeBase):
        def __init__(self, shape, name,
                     activation=IdentityActivation(),
                     energy=GaussianEnergy(),
                     latent_init=NormalInitializer(),
                     **kwargs):
            super().__init__(shape=shape, name=name, activation=activation,
                             energy=energy, latent_init=latent_init, **kwargs)

        @staticmethod
        def get_slots():
            return {"in": SlotSpec(name="in", is_multi_input=True)}

        @staticmethod
        def initialize_params(key, node_shape, input_shapes, weight_init, config):
            ...

        @staticmethod
        def forward(params, inputs, state, node_info):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import copy
import types
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from fabricpc.core.types import NodeParams, NodeState, NodeInfo, SlotInfo, EdgeInfo

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


@dataclass(frozen=True)
class SlotSpec:
    """Specification for an input slot to a node."""

    name: str
    is_multi_input: bool  # True = multiple inputs allowed, False = single input only
    is_variance_scalable: bool = (
        True  # False = muPC leaves edges to this slot unscaled (scale 1.0)
    )
    is_skip_connection: bool = (
        False  # True = identity bypass path that counts toward muPC depth L
    )

    def __post_init__(self):
        if self.is_skip_connection and self.is_variance_scalable:
            raise ValueError(
                "is_skip_connection and is_variance_scalable were both set to True, but skip connection slots should NOT be subject to muPC variance scaling"
            )


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

    Subclasses set concrete default instances for activation, energy, latent_init,
    and weight_init in their ``__init__`` parameter defaults.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = None,
        energy: Optional[EnergyFunctional] = None,
        latent_init: Optional[InitializerBase] = None,
        weight_init: Optional[InitializerBase] = None,
        **extra_config,
    ):
        """
        Initialize a node descriptor.

        Args:
            shape: Output shape tuple (excluding batch dimension)
            name: Node name. Automatically prefixed with current GraphNamespace.
            activation: ActivationBase instance, or None
            energy: EnergyFunctional instance, or None
            latent_init: InitializerBase instance, or None
            weight_init: InitializerBase instance, or None
            **extra_config: Node-specific config (use_bias, flatten_input, etc.)
        """
        from fabricpc.builder.namespace import _get_current_namespace

        ns = _get_current_namespace()
        self._name = f"{ns}/{name}" if ns else name
        self._shape = tuple(shape)
        self._activation = activation
        self._energy = energy
        self._latent_init = latent_init
        self._weight_init = weight_init
        self._extra_config = types.MappingProxyType(
            extra_config
        )  # Immutable dictionary
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
        weight_init: Optional[InitializerBase],
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Define and initialize the parameters required for the node.

        Args:
            key: JAX random key
            node_shape: Output shape of this node (excluding batch dimension)
            input_shapes: Dictionary mapping edge keys to source node shapes
            weight_init: InitializerBase instance for weight initialization, or None
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
    # muPC fan_in for scaling — override per node type
    # =========================================================================

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """
        Return weight-matrix fan_in for muPC scaling (Kaiming convention).

        This is the number of input units that contribute to each output unit
        of the weight matrix. Override in subclasses for node-specific logic
        (e.g., Conv2D uses C_in * kH * kW instead of H * W * C).

        - flatten_input=True (dense): all dims flattened → prod(source_shape)
        - flatten_input=False (per-position): last-axis features only

        Args:
            source_shape: Shape of the source (presynaptic) node, excluding batch.
            config: Node configuration dictionary (e.g., kernel_size, flatten_input).

        Returns:
            Integer fan_in for the weight matrix connecting source to this node.
        """
        if config.get("flatten_input", False):
            return int(np.prod(source_shape))
        # Typically nodes operate on the last (feature dimension)
        return source_shape[-1]

    # =========================================================================
    # Default implementations - can be overridden for explicit gradients
    # =========================================================================

    @staticmethod
    def forward_and_latent_grads(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Forward pass with autodiff: computes updated state, gradients w.r.t.
        inputs (for updating upstream latents), and the self-latent gradient
        (dE/dz_latent).
        Called in the inference phase of predictive coding.

        Override this method to implement explicit (non-autodiff) gradient
        computation. When overriding, use ``energy.grad_latent()`` and
        ``activation.derivative()`` (or ``activation.jacobian()``) for
        analytical gradients.

        muPC scaling is NOT applied here — it is handled by the callsite
        (inference loop). Node methods are pure autodiff. The returned
        ``self_grad`` is the contribution from this node only, so the
        callsite scales it independently and adds it to ``state.latent_grad``
        without re-scaling pre-existing accumulated contributions from
        downstream successors.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
                (already muPC-scaled by the callsite when scaling is active)
            state: NodeState for this node
            node_info: NodeInfo object
            is_clamped: Whether this node is clamped to data

        Returns:
            Tuple of (NodeState, input_grads, self_grad):
                - NodeState: updated state (z_mu, pre_activation, error,
                  energy). ``latent_grad`` is *not* modified here.
                - input_grads: dict of gradients w.r.t. each input edge
                  (dE/d_input per edge), unscaled.
                - self_grad: dE/dz_latent contribution from this node,
                  unscaled, same shape as ``state.z_latent``.
        """
        node_class = node_info.node_class

        # Handle terminal nodes
        if node_info.in_degree == 0:
            # No inputs!
            # This is a terminal input node of the graph. It might be clamped to data, or it might be a source of top-down predictions. Either way, the gradients are zero.

            # Update z_mu <-- z_latent, so error is zero. Cast to z_mu's dtype:
            # source-node z_latent may be an integer clamp (e.g. token indices),
            # but the rest of the NodeState stays float for the inference carry.
            new_state = state._replace(
                z_mu=state.z_latent.astype(state.z_mu.dtype),
                error=jnp.zeros_like(state.error),
                pre_activation=jnp.zeros_like(state.pre_activation),
            )
            # Update the state's energy; will be zero since z_mu = z_latent
            new_state = node_class.energy_functional(new_state, node_info)
            # No inputs, no contribution to latent_grad of self or upstream
            input_grads = {
                edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs
            }
            self_grad = jnp.zeros_like(state.latent_grad)

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
            self_grad = jnp.zeros_like(state.z_latent)

        else:
            # Internal or clamped output node: autodiff for input AND self-latent gradients.
            # Extract z_latent as a separate differentiable argument via closure.
            def energy_fn(input_args, z_latent):
                s = state._replace(z_latent=z_latent)
                total_energy, new_s = node_class.forward(
                    params, input_args, s, node_info
                )
                return total_energy, new_s

            (total_energy, new_state), (input_grads, self_grad) = jax.value_and_grad(
                energy_fn, argnums=(0, 1), has_aux=True
            )(inputs, state.z_latent)

        return new_state, input_grads, self_grad

    @staticmethod
    def forward_and_weight_grads(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass with autodiff: computes the node's local energy gradient w.r.t. weights.
        Called in the learning phase of predictive coding.

        Override this method to implement explicit weight gradient computation
        or apply node-specific post-processing (e.g., LayerNorm compensation).

        muPC scaling is NOT applied here — it is handled by the callsite
        (learning loop). Node methods are pure autodiff.

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
                (already muPC-scaled by the callsite when scaling is active)
            state: NodeState for this node
            node_info: NodeInfo object

        Returns:
            Tuple of (NodeState, params_grad):
                - NodeState: updated node state
                - params_grad: NodeParams containing weight and bias gradients
        """
        node_class = node_info.node_class

        (total_energy, new_state), params_grad = jax.value_and_grad(
            node_class.forward, argnums=0, has_aux=True
        )(params, inputs, state, node_info)

        return new_state, params_grad

    @staticmethod
    def energy_functional(state: NodeState, node_info: NodeInfo) -> NodeState:
        """
        Compute energy E(z_latent, z_mu) and update state.

        The self-latent gradient (dE/dz_latent) is NOT computed here — it
        is obtained via autodiff in forward_and_latent_grads(). For explicit
        gradient overrides, use ``energy.grad_latent()`` directly.

        Args:
            state: NodeState object (contains z_latent, z_mu, etc.)
            node_info: NodeInfo object (contains energy instance)

        Returns:
            Updated NodeState with energy field set
        """
        energy_obj = node_info.energy
        if energy_obj is None:
            raise ValueError(
                f"Node '{node_info.name}' has no energy functional configured."
            )

        energy_cls = type(energy_obj)
        config = energy_obj.config

        energy = energy_cls.energy(state.z_latent, state.z_mu, config)
        return state._replace(energy=energy)
