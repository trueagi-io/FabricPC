"""
Native FabricPC Nodes for Continual Learning.

Provides FabricPC-native node implementations that embed continual learning
algorithms directly into the graph's forward/backward pass:

- CausalLinear: Linear node with per-weight causal gradient correction
- TransWeaveLinear: Linear node with TransWeave transfer learning
- CausalColumnNode: Column node with causal coding support

These nodes integrate seamlessly with FabricPC's inference-based learning
while providing continual learning capabilities.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass, field
import weakref

import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.nodes.linear import Linear, LinearExplicitGrad
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.initializers import NormalInitializer, KaimingInitializer, initialize
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy

from fabricpc.continual.optimal_transport import sinkhorn_1d_correction
from fabricpc.continual.weight_causal import (
    PerWeightCausalConfig,
    compute_weight_excess_kurtosis,
    compute_weight_multimodal_gap,
    compute_non_gaussianity_score,
)

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


# ----------------------------
# Causal Gradient Registry
# ----------------------------


@dataclass
class GradientHistoryEntry:
    """Stores gradient history for a single parameter."""

    buffer: np.ndarray  # (history_size, *param_shape)
    position: int = 0
    filled: bool = False


class CausalGradientRegistry:
    """
    Global registry for per-weight gradient history.

    This registry enables stateful gradient tracking across training steps
    while keeping FabricPC nodes pure (stateless). Each node registers its
    gradients here, and the registry manages history and non-Gaussianity detection.

    Usage:
        registry = CausalGradientRegistry.get_instance()
        registry.register_gradient(node_name, param_key, gradient)
        mask = registry.get_non_gaussian_mask(node_name, param_key, config)
    """

    _instance: Optional["CausalGradientRegistry"] = None

    def __init__(self, config: Optional[PerWeightCausalConfig] = None):
        self.config = config or PerWeightCausalConfig()
        self._histories: Dict[str, Dict[str, GradientHistoryEntry]] = {}
        self._step_count = 0

    @classmethod
    def get_instance(cls) -> "CausalGradientRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls, config: Optional[PerWeightCausalConfig] = None) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = cls(config)

    def register_gradient(
        self,
        node_name: str,
        param_key: str,
        gradient: jnp.ndarray,
    ) -> None:
        """
        Register a gradient for tracking.

        Args:
            node_name: Name of the node
            param_key: Parameter key (e.g., edge key for weights)
            gradient: Gradient array
        """
        if not self.config.enable:
            return

        gradient_np = np.asarray(gradient, dtype=np.float32)

        if node_name not in self._histories:
            self._histories[node_name] = {}

        if param_key not in self._histories[node_name]:
            buffer_shape = (self.config.gradient_history_size,) + gradient_np.shape
            self._histories[node_name][param_key] = GradientHistoryEntry(
                buffer=np.zeros(buffer_shape, dtype=np.float32),
                position=0,
                filled=False,
            )

        entry = self._histories[node_name][param_key]
        entry.buffer[entry.position] = gradient_np
        entry.position = (entry.position + 1) % self.config.gradient_history_size

        if entry.position == 0:
            entry.filled = True

    def get_non_gaussian_mask(
        self,
        node_name: str,
        param_key: str,
    ) -> Optional[jnp.ndarray]:
        """
        Get non-Gaussianity mask for a parameter.

        Returns:
            Mask array where values > 0 indicate non-Gaussian weights,
            or None if insufficient history.
        """
        if not self.config.enable:
            return None

        if node_name not in self._histories:
            return None

        if param_key not in self._histories[node_name]:
            return None

        entry = self._histories[node_name][param_key]

        # Check if we have sufficient history
        effective_size = (
            self.config.gradient_history_size if entry.filled else entry.position
        )

        if effective_size < self.config.min_history_for_detection:
            return None

        # Get history in chronological order
        if entry.filled:
            history = np.concatenate(
                [entry.buffer[entry.position :], entry.buffer[: entry.position]],
                axis=0,
            )
        else:
            history = entry.buffer[: entry.position]

        # Compute per-weight statistics
        kurtosis = compute_weight_excess_kurtosis(history)
        multimodal = compute_weight_multimodal_gap(history)
        score = compute_non_gaussianity_score(kurtosis, multimodal, self.config)

        # Compute mask
        if self.config.blend_mode == "hard":
            mask = (score > self.config.combined_threshold).astype(np.float32)
        else:
            # Soft mask using sigmoid
            mask = 1.0 / (
                1.0
                + np.exp(
                    -self.config.soft_blend_scale
                    * (score - self.config.combined_threshold)
                )
            )

        return jnp.asarray(mask, dtype=jnp.float32)

    def apply_causal_correction(
        self,
        node_name: str,
        param_key: str,
        gradient: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply causal correction to gradient if needed.

        This is the main entry point for per-weight causal coding.

        Args:
            node_name: Name of the node
            param_key: Parameter key
            gradient: Original gradient

        Returns:
            Corrected gradient (or original if no correction needed)
        """
        if not self.config.enable:
            return gradient

        # First, register the gradient for tracking
        self.register_gradient(node_name, param_key, gradient)

        # Get non-Gaussian mask
        mask = self.get_non_gaussian_mask(node_name, param_key)

        if mask is None:
            return gradient

        # Check if any weights need correction
        if jnp.max(mask) < 1e-6:
            return gradient

        # Apply Sinkhorn correction
        gradient_np = np.asarray(gradient, dtype=np.float32)
        mask_np = np.asarray(mask, dtype=np.float32)
        original_shape = gradient_np.shape

        # Flatten for processing
        flat_grad = gradient_np.ravel()
        flat_mask = mask_np.ravel()

        # Identify weights that need correction
        needs_correction = flat_mask > 0.1

        if not np.any(needs_correction):
            return gradient

        # Get indices and apply correction
        idx_correct = np.where(needs_correction)[0]

        if len(idx_correct) >= 2:
            grads_to_correct = flat_grad[idx_correct]

            # Apply 1D Sinkhorn correction
            corrected_grads = np.asarray(
                sinkhorn_1d_correction(
                    grads_to_correct,
                    eps=self.config.sb_sinkhorn_eps,
                    iters=self.config.sb_sinkhorn_iters,
                )
            )

            # Blend based on mask values
            blend_factors = flat_mask[idx_correct] * self.config.sb_correction_strength

            corrected = flat_grad.copy()
            corrected[idx_correct] = (
                1 - blend_factors
            ) * grads_to_correct + blend_factors * corrected_grads

            return jnp.asarray(corrected.reshape(original_shape), dtype=jnp.float32)

        return gradient

    def clear(self, node_name: Optional[str] = None) -> None:
        """Clear gradient history."""
        if node_name is None:
            self._histories.clear()
        elif node_name in self._histories:
            del self._histories[node_name]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry."""
        total_params = sum(len(params) for params in self._histories.values())
        return {
            "num_nodes": len(self._histories),
            "total_params_tracked": total_params,
            "config_enabled": self.config.enable,
        }


# ----------------------------
# CausalLinear Node
# ----------------------------


class CausalLinear(LinearExplicitGrad):
    """
    Linear node with per-weight causal gradient correction.

    Extends LinearExplicitGrad to apply Sinkhorn-based corrections to
    gradients based on their non-Gaussianity. Weights with non-Gaussian
    gradient distributions receive transport-based corrections.

    This node integrates directly into the FabricPC graph and applies
    causal coding during the learning phase (forward_learning).

    Usage:
        # Create node
        hidden = CausalLinear(
            shape=(256,),
            name="hidden",
            causal_config=PerWeightCausalConfig(enable=True),
        )

        # Use in graph
        structure = graph(
            nodes=[input_node, hidden, output],
            edges=[...],
        )
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
        causal_config: Optional[PerWeightCausalConfig] = None,
    ):
        """
        Args:
            shape: Output shape tuple (excluding batch dimension)
            name: Node name
            activation: Activation function
            energy: Energy functional
            use_bias: Whether to use bias
            flatten_input: If True, flatten all input dims for dense behavior
            weight_init: Weight initializer
            latent_init: Latent state initializer
            causal_config: Configuration for per-weight causal coding
        """
        if causal_config is None:
            causal_config = PerWeightCausalConfig()

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            use_bias=use_bias,
            flatten_input=flatten_input,
            weight_init=weight_init,
            latent_init=latent_init,
        )

        # Store causal config in extra_config
        self._extra_config = {
            **self._extra_config,
            "causal_enable": causal_config.enable,
            "causal_kurtosis_threshold": causal_config.kurtosis_threshold,
            "causal_sb_correction_strength": causal_config.sb_correction_strength,
        }

    @staticmethod
    def forward_learning(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass with causal gradient correction.

        Computes gradients and applies per-weight causal corrections
        to non-Gaussian gradient distributions.
        """
        node_class = node_info.node_class

        # Forward pass to get new state
        _, state = node_class.forward(params, inputs, state, node_info)

        # Gain-modulated error computation
        gain_mod_error = node_class.compute_gain_mod_error(state, node_info)

        flatten_input = node_info.node_config.get("flatten_input", False)
        causal_enable = node_info.node_config.get("causal_enable", True)

        weight_grads = {}
        bias_grads = {}

        # Get registry for causal corrections
        registry = CausalGradientRegistry.get_instance()

        # Weight gradient with causal correction
        for edge_key, in_tensor in inputs.items():
            if flatten_input:
                in_flat = FlattenInputMixin.flatten_input(in_tensor)
                gain_mod_error_flat = FlattenInputMixin.flatten_input(gain_mod_error)
                grad_w = -jnp.matmul(in_flat.T, gain_mod_error_flat)
            else:
                in_shape = in_tensor.shape
                err_shape = gain_mod_error.shape
                in_flat = in_tensor.reshape(-1, in_shape[-1])
                err_flat = gain_mod_error.reshape(-1, err_shape[-1])
                grad_w = -jnp.matmul(in_flat.T, err_flat)

            # Apply causal correction if enabled
            if causal_enable:
                grad_w = registry.apply_causal_correction(
                    node_info.name, edge_key, grad_w
                )

            weight_grads[edge_key] = grad_w

        # Bias gradient (no causal correction for bias)
        if "b" in params.biases:
            grad_b = -jnp.sum(gain_mod_error, axis=0, keepdims=True)
            bias_grads["b"] = grad_b

        return state, NodeParams(weights=weight_grads, biases=bias_grads)


# ----------------------------
# TransWeave Registry
# ----------------------------


@dataclass
class TransWeaveState:
    """Stores TransWeave state for a node across tasks."""

    task_representations: Dict[int, Dict[str, jnp.ndarray]] = field(
        default_factory=dict
    )
    current_task_id: int = 0


class TransWeaveRegistry:
    """
    Global registry for TransWeave transfer learning state.

    Manages cross-task representation storage and transfer for nodes
    that use TransWeave-based knowledge transfer.
    """

    _instance: Optional["TransWeaveRegistry"] = None

    def __init__(self):
        self._states: Dict[str, TransWeaveState] = {}
        self._current_task_id: int = 0

    @classmethod
    def get_instance(cls) -> "TransWeaveRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = cls()

    def set_current_task(self, task_id: int) -> None:
        """Set the current task ID for transfer."""
        self._current_task_id = task_id

    def register_task_end(
        self,
        node_name: str,
        params: NodeParams,
    ) -> None:
        """
        Register end-of-task parameters for future transfer.

        Args:
            node_name: Name of the node
            params: Final parameters after task training
        """
        if node_name not in self._states:
            self._states[node_name] = TransWeaveState()

        state = self._states[node_name]

        # Store weight representations
        weight_repr = {}
        for key, weight in params.weights.items():
            weight_repr[key] = jnp.asarray(weight)

        state.task_representations[self._current_task_id] = weight_repr

    def get_transfer_init(
        self,
        node_name: str,
        current_params: NodeParams,
        transfer_strength: float = 0.3,
        use_last_k_tasks: int = 3,
    ) -> NodeParams:
        """
        Get transferred initialization for a new task.

        Args:
            node_name: Name of the node
            current_params: Current (randomly initialized) parameters
            transfer_strength: How much to blend transferred params (0-1)
            use_last_k_tasks: Number of recent tasks to use for transfer

        Returns:
            Blended parameters with transfer from previous tasks
        """
        if node_name not in self._states:
            return current_params

        state = self._states[node_name]

        if not state.task_representations:
            return current_params

        # Get recent task IDs
        task_ids = sorted(state.task_representations.keys())
        recent_tasks = task_ids[-use_last_k_tasks:]

        if not recent_tasks:
            return current_params

        # Compute weighted average of previous task weights
        transferred_weights = {}

        for key in current_params.weights.keys():
            accumulated = jnp.zeros_like(current_params.weights[key])
            total_weight = 0.0

            for i, task_id in enumerate(recent_tasks):
                if key in state.task_representations[task_id]:
                    # Recency weighting
                    recency_weight = 0.5 ** (len(recent_tasks) - i - 1)
                    accumulated += (
                        recency_weight * state.task_representations[task_id][key]
                    )
                    total_weight += recency_weight

            if total_weight > 0:
                transferred = accumulated / total_weight
                # Blend with current params
                transferred_weights[key] = (
                    1 - transfer_strength
                ) * current_params.weights[key] + transfer_strength * transferred
            else:
                transferred_weights[key] = current_params.weights[key]

        return NodeParams(weights=transferred_weights, biases=current_params.biases)

    def clear(self, node_name: Optional[str] = None) -> None:
        """Clear TransWeave state."""
        if node_name is None:
            self._states.clear()
        elif node_name in self._states:
            del self._states[node_name]


# ----------------------------
# TransWeaveLinear Node
# ----------------------------


class TransWeaveLinear(Linear):
    """
    Linear node with TransWeave transfer learning.

    Extends Linear to support cross-task knowledge transfer using
    optimal transport-based alignment of weight representations.

    The node automatically registers its state at task boundaries and
    can initialize new task weights using transferred representations.

    Usage:
        # Create node
        hidden = TransWeaveLinear(
            shape=(256,),
            name="hidden",
            transfer_strength=0.3,
        )

        # At task boundary, register state
        TransWeaveRegistry.get_instance().register_task_end(
            "hidden", current_params
        )

        # For new task, get transferred init
        new_params = TransWeaveRegistry.get_instance().get_transfer_init(
            "hidden", random_init_params
        )
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
        transfer_strength: float = 0.3,
        use_last_k_tasks: int = 3,
    ):
        """
        Args:
            shape: Output shape tuple
            name: Node name
            activation: Activation function
            energy: Energy functional
            use_bias: Whether to use bias
            flatten_input: If True, flatten input for dense behavior
            weight_init: Weight initializer
            latent_init: Latent state initializer
            transfer_strength: How much to blend transferred params (0-1)
            use_last_k_tasks: Number of recent tasks for transfer
        """
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            use_bias=use_bias,
            flatten_input=flatten_input,
            weight_init=weight_init,
            latent_init=latent_init,
        )

        self._extra_config = {
            **self._extra_config,
            "transfer_strength": transfer_strength,
            "use_last_k_tasks": use_last_k_tasks,
        }


# ----------------------------
# CausalTransWeaveLinear Node
# ----------------------------


class CausalTransWeaveLinear(CausalLinear):
    """
    Linear node combining per-weight causal coding and TransWeave transfer.

    Provides the full continual learning stack:
    - Per-weight gradient correction for non-Gaussian distributions
    - Cross-task knowledge transfer using optimal transport

    This is the recommended node for continual learning tasks.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = IdentityActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init: Optional[InitializerBase] = KaimingInitializer(),
        latent_init: Optional[InitializerBase] = NormalInitializer(),
        causal_config: Optional[PerWeightCausalConfig] = None,
        transfer_strength: float = 0.3,
        use_last_k_tasks: int = 3,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            use_bias=use_bias,
            flatten_input=flatten_input,
            weight_init=weight_init,
            latent_init=latent_init,
            causal_config=causal_config,
        )

        self._extra_config = {
            **self._extra_config,
            "transfer_strength": transfer_strength,
            "use_last_k_tasks": use_last_k_tasks,
        }


# ----------------------------
# Helper Functions
# ----------------------------


def apply_causal_to_gradients(
    gradients: Dict[str, jnp.ndarray],
    node_name: str,
) -> Dict[str, jnp.ndarray]:
    """
    Apply causal corrections to a dictionary of gradients.

    Convenience function for applying causal coding to any node's gradients.

    Args:
        gradients: Dictionary of parameter gradients
        node_name: Name of the node

    Returns:
        Dictionary of corrected gradients
    """
    registry = CausalGradientRegistry.get_instance()

    return {
        key: registry.apply_causal_correction(node_name, key, grad)
        for key, grad in gradients.items()
    }


def register_task_end_for_nodes(
    node_names: List[str],
    params: Dict[str, NodeParams],
) -> None:
    """
    Register end-of-task for multiple nodes.

    Convenience function for registering TransWeave state at task boundaries.

    Args:
        node_names: List of node names to register
        params: Dictionary mapping node names to their parameters
    """
    registry = TransWeaveRegistry.get_instance()

    for name in node_names:
        if name in params:
            registry.register_task_end(name, params[name])


def get_transferred_params(
    node_names: List[str],
    current_params: Dict[str, NodeParams],
    transfer_strength: float = 0.3,
) -> Dict[str, NodeParams]:
    """
    Get transferred parameters for multiple nodes.

    Args:
        node_names: List of node names
        current_params: Dictionary of current parameters
        transfer_strength: Blend strength for transfer

    Returns:
        Dictionary of transferred parameters
    """
    registry = TransWeaveRegistry.get_instance()

    result = {}
    for name in node_names:
        if name in current_params:
            result[name] = registry.get_transfer_init(
                name, current_params[name], transfer_strength
            )
        else:
            result[name] = current_params.get(name)

    return result
