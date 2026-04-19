"""
Gradient Protection for Continual Learning.

Implements shell-based gradient masking to prevent catastrophic forgetting:
- Protected center (shell 0): Near-zero gradient updates
- Stable inner (shell 1): Reduced gradient updates
- Disposable outer (shell 2): Full gradient updates

Also supports column-based masking where inactive columns receive zero gradients.

Reference: Two-Level HiBaCaML Decomposition (among-columns + within-column shells)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import tree_util

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # type: ignore


@dataclass
class GradientProtectionConfig:
    """Configuration for gradient protection."""

    # Enable gradient protection
    enable: bool = True

    # Shell-based protection scales (lower = more protection)
    # Shell 0: protected center, Shell 1: stable inner, Shell 2: disposable outer
    shell_scales: Tuple[float, ...] = (0.0, 0.1, 1.0)

    # Column-based protection
    protect_inactive_columns: bool = True
    inactive_column_scale: float = 0.0  # 0.0 = completely frozen

    # Shared columns (always active)
    shared_column_scale: float = 0.5  # Partial protection for shared columns

    # Gradient clipping (applied after masking)
    max_gradient_norm: Optional[float] = 1.0


@dataclass
class GradientProtectionState:
    """State tracking for gradient protection."""

    # Current task ID
    task_id: int = 0

    # Active column indices
    active_columns: Tuple[int, ...] = ()

    # Shared column indices
    shared_columns: Tuple[int, ...] = ()

    # Shell assignments per column: Dict[column_id, np.ndarray of shell indices per neuron]
    column_shell_assignments: Dict[int, np.ndarray] = field(default_factory=dict)

    # Number of columns
    num_columns: int = 32

    # Shell sizes
    shell_sizes: Tuple[int, ...] = (8, 16, 8)

    # Memory dimension per column (for computing input masks on downstream layers)
    memory_dim: int = 64

    # Active output classes for this task (for output layer gradient masking)
    active_classes: Tuple[int, ...] = ()

    # Total number of output classes
    num_output_classes: int = 10

    # Aggregator partitioning for task-specific pathways
    aggregator_dim: int = 128  # Total aggregator neurons
    num_tasks: int = 5  # Number of tasks (for partitioning)


class GradientProtector:
    """
    Applies gradient protection based on shell assignments and column activity.

    This is the core mechanism for preventing catastrophic forgetting without
    experience replay. Gradients are scaled based on:

    1. Column activity: Inactive columns receive zero/minimal gradients
    2. Shell membership: Inner shell neurons receive reduced gradients

    Usage:
        protector = GradientProtector(config)
        protector.set_active_columns(active_cols, shared_cols)
        protector.set_shell_assignments(column_shell_assignments)

        # During training:
        masked_grads = protector.apply_protection(grads, params)
    """

    def __init__(self, config: GradientProtectionConfig):
        self.config = config
        self.state = GradientProtectionState()

    def set_task(self, task_id: int) -> None:
        """Set current task ID."""
        self.state.task_id = task_id

    def set_active_columns(
        self,
        active_columns: Sequence[int],
        shared_columns: Sequence[int],
        num_columns: int,
    ) -> None:
        """Set which columns are active for current task."""
        self.state.active_columns = tuple(active_columns)
        self.state.shared_columns = tuple(shared_columns)
        self.state.num_columns = num_columns

    def set_shell_assignments(
        self,
        column_shell_assignments: Dict[int, np.ndarray],
        shell_sizes: Tuple[int, ...],
    ) -> None:
        """Set shell assignments for each column."""
        self.state.column_shell_assignments = column_shell_assignments
        self.state.shell_sizes = shell_sizes

    def set_active_classes(
        self,
        active_classes: Sequence[int],
        num_output_classes: int = 10,
    ) -> None:
        """Set which output classes are active for current task."""
        self.state.active_classes = tuple(active_classes)
        self.state.num_output_classes = num_output_classes

    def set_aggregator_config(
        self,
        aggregator_dim: int,
        num_tasks: int,
    ) -> None:
        """Set aggregator partitioning configuration."""
        self.state.aggregator_dim = aggregator_dim
        self.state.num_tasks = num_tasks

    def compute_column_mask(self) -> np.ndarray:
        """
        Compute protection mask for each column.

        Returns:
            Array of shape (num_columns,) with scale factors per column
        """
        mask = np.ones(self.state.num_columns, dtype=np.float32)

        if not self.config.enable:
            return mask

        active_set = set(self.state.active_columns)
        shared_set = set(self.state.shared_columns)

        for col_id in range(self.state.num_columns):
            if col_id in shared_set:
                # Shared columns get partial protection
                mask[col_id] = self.config.shared_column_scale
            elif col_id in active_set:
                # Active non-shared columns get full updates
                mask[col_id] = 1.0
            else:
                # Inactive columns get minimal updates
                mask[col_id] = self.config.inactive_column_scale

        return mask

    def compute_shell_mask_for_column(self, column_id: int) -> np.ndarray:
        """
        Compute protection mask for neurons within a column based on shell.

        Args:
            column_id: Column index

        Returns:
            Array of shape (num_neurons,) with scale factors per neuron
        """
        num_neurons = sum(self.state.shell_sizes)
        mask = np.ones(num_neurons, dtype=np.float32)

        if not self.config.enable:
            return mask

        shell_assignments = self.state.column_shell_assignments.get(column_id)
        if shell_assignments is None:
            # No shell info, default to outer shell (full updates)
            return mask

        # Apply shell-based scaling
        for neuron_idx, shell_id in enumerate(shell_assignments):
            shell_id = int(shell_id)
            if shell_id < len(self.config.shell_scales):
                mask[neuron_idx] = self.config.shell_scales[shell_id]
            else:
                mask[neuron_idx] = 1.0  # Unknown shell, allow updates

        return mask

    def compute_input_column_mask(self, input_dim: int) -> Optional[np.ndarray]:
        """
        Compute INPUT dimension mask for layers receiving flattened column outputs.

        For a layer receiving input from ColumnNode (flattened from num_columns x memory_dim),
        this returns a mask that protects input connections from frozen columns.

        Args:
            input_dim: Input dimension of the layer

        Returns:
            Array of shape (input_dim,) with scale factors, or None if not applicable
        """
        num_columns = self.state.num_columns
        memory_dim = self.state.memory_dim

        expected_dim = num_columns * memory_dim

        if input_dim != expected_dim or num_columns == 0:
            # Not a downstream layer from ColumnNode
            return None

        column_mask = self.compute_column_mask()

        # Expand column mask to per-neuron: each column controls memory_dim neurons
        input_mask = np.zeros(input_dim, dtype=np.float32)
        for col_idx in range(num_columns):
            start_idx = col_idx * memory_dim
            end_idx = start_idx + memory_dim
            input_mask[start_idx:end_idx] = column_mask[col_idx]

        return input_mask

    def compute_aggregator_task_mask(self, aggregator_dim: int) -> Optional[np.ndarray]:
        """
        Compute mask for aggregator OUTPUT neurons based on task partitioning.

        Partitions aggregator neurons among tasks so each task has its own
        dedicated subset. This creates task-specific pathways through the network.

        Args:
            aggregator_dim: Number of aggregator output neurons

        Returns:
            Array of shape (aggregator_dim,) with 1.0 for current task's neurons,
            0.0 for other tasks' neurons. Returns None if not applicable.
        """
        if aggregator_dim != self.state.aggregator_dim:
            # Dimension mismatch, not the aggregator
            return None

        num_tasks = self.state.num_tasks
        if num_tasks <= 0:
            return None

        task_id = self.state.task_id

        # Partition neurons: each task gets aggregator_dim // num_tasks neurons
        neurons_per_task = aggregator_dim // num_tasks
        remainder = aggregator_dim % num_tasks

        # Task i gets neurons from start_i to end_i
        # Extra neurons go to earlier tasks
        start_idx = task_id * neurons_per_task + min(task_id, remainder)
        extra = 1 if task_id < remainder else 0
        end_idx = start_idx + neurons_per_task + extra

        mask = np.zeros(aggregator_dim, dtype=np.float32)
        mask[start_idx:end_idx] = 1.0

        return mask

    def compute_output_class_mask(self, output_dim: int) -> Optional[np.ndarray]:
        """
        Compute OUTPUT dimension mask for output layer based on active classes.

        Only active classes receive gradients; inactive class outputs are frozen
        to prevent catastrophic forgetting of previously learned classes.

        Args:
            output_dim: Output dimension (number of classes)

        Returns:
            Array of shape (output_dim,) with scale factors, or None if not applicable
        """
        if output_dim != self.state.num_output_classes:
            # Not the output layer
            return None

        if not self.state.active_classes:
            # No active classes specified, allow all
            return None

        # Mask: 1.0 for active classes, 0.0 for inactive
        mask = np.zeros(output_dim, dtype=np.float32)
        for class_idx in self.state.active_classes:
            if 0 <= class_idx < output_dim:
                mask[class_idx] = 1.0

        return mask

    def apply_protection(
        self,
        grads: Any,
        params: Any,
    ) -> Any:
        """
        Apply gradient protection based on column activity and shell assignments.

        For ColumnNode parameters (with col_X_proj naming):
        - Mask entire column gradients based on column activity
        - Within active columns, mask based on shell assignments

        For other parameters:
        - Apply global gradient clipping only

        Args:
            grads: PyTree of gradients (same structure as params)
            params: PyTree of parameters

        Returns:
            Protected gradients with same structure
        """
        if not self.config.enable:
            return grads

        column_mask = self.compute_column_mask()

        def apply_mask_to_node(node_name: str, node_grads, node_params):
            """Apply protection to gradients for a single node."""
            if node_grads is None:
                return node_grads

            weights_grads = node_grads.weights
            biases_grads = node_grads.biases
            num_columns = self.state.num_columns

            # Apply column-wise masking for ColumnNode weights
            masked_weights = {}
            for weight_name, weight_grad in weights_grads.items():
                if weight_name.startswith("col_") and "_proj" in weight_name:
                    # ColumnNode: Extract column index "col_X_proj" -> X
                    try:
                        parts = weight_name.split("_")
                        col_idx = int(parts[1])
                        col_scale = column_mask[col_idx]

                        # Also apply shell masking to output dimension
                        shell_mask = self.compute_shell_mask_for_column(col_idx)

                        # For weight shape (input_dim, output_dim), mask output dim
                        if weight_grad.ndim == 2 and weight_grad.shape[1] == len(
                            shell_mask
                        ):
                            # Scale by column activity and shell membership
                            combined_mask = col_scale * shell_mask
                            masked_grad = weight_grad * combined_mask[None, :]
                        else:
                            # Shape mismatch, just apply column scale
                            masked_grad = weight_grad * col_scale

                        masked_weights[weight_name] = masked_grad
                    except (IndexError, ValueError):
                        # Can't parse column index, keep original
                        masked_weights[weight_name] = weight_grad

                elif weight_name.endswith(":in") and weight_grad.ndim == 2:
                    # Standard Linear layer
                    # Weight shape is (input_dim, output_dim)
                    input_dim = weight_grad.shape[0]
                    output_dim = weight_grad.shape[1]

                    masked_grad = weight_grad

                    # Check if this is the aggregator (receives column outputs)
                    input_mask = self.compute_input_column_mask(input_dim)
                    agg_output_mask = self.compute_aggregator_task_mask(output_dim)

                    if input_mask is not None and agg_output_mask is not None:
                        # This is the aggregator - allow learning task pathways
                        # Gradients flow where BOTH:
                        # 1. Input is from active OR shared columns (not frozen)
                        # 2. Output is current task's aggregator neurons
                        # Shared columns allow the task to use learned features.

                        # Build a 2D mask: (input_dim, output_dim)
                        # Current task can learn from shared + own columns -> own neurons
                        input_mask_jax = jnp.array(input_mask)
                        agg_output_mask_jax = jnp.array(agg_output_mask)

                        # Outer product gives us the combined mask
                        # Only positions where BOTH input AND output are 1.0 get gradients
                        combined_mask = (
                            input_mask_jax[:, None] * agg_output_mask_jax[None, :]
                        )
                        masked_grad = masked_grad * combined_mask

                    elif input_mask is not None:
                        # Just column input masking (fallback)
                        input_mask_jax = jnp.array(input_mask)
                        masked_grad = masked_grad * input_mask_jax[:, None]

                    # Check if this is the output layer
                    output_class_mask = self.compute_output_class_mask(output_dim)
                    if output_class_mask is not None:
                        # Apply output class mask: only current task's classes get gradients
                        output_class_mask_jax = jnp.array(output_class_mask)
                        masked_grad = masked_grad * output_class_mask_jax[None, :]

                        # Also apply aggregator task partition to INPUT dimension
                        # This ensures only current task's aggregator neurons contribute!
                        agg_input_mask = self.compute_aggregator_task_mask(input_dim)
                        if agg_input_mask is not None:
                            agg_input_mask_jax = jnp.array(agg_input_mask)
                            masked_grad = masked_grad * agg_input_mask_jax[:, None]

                    masked_weights[weight_name] = masked_grad

                else:
                    # Other weights (e.g., small layers, output), keep original
                    masked_weights[weight_name] = weight_grad

            # Biases: apply column mask for ColumnNode, output class mask for output layer
            masked_biases = {}
            for bias_name, bias_grad in biases_grads.items():
                if bias_name == "column_bias" and bias_grad.ndim >= 1:
                    # ColumnNode bias: shape (num_columns, memory_dim) or (num_columns,)
                    if bias_grad.shape[0] == len(column_mask):
                        if bias_grad.ndim == 2:
                            masked_biases[bias_name] = bias_grad * column_mask[:, None]
                        else:
                            masked_biases[bias_name] = bias_grad * column_mask
                    else:
                        masked_biases[bias_name] = bias_grad

                elif bias_name == "b" and bias_grad.ndim == 1:
                    # Standard Linear bias (named "b" in FabricPC)
                    output_dim = bias_grad.shape[0]
                    masked_bias = bias_grad

                    # Check if this is output layer bias - mask inactive classes
                    output_class_mask = self.compute_output_class_mask(output_dim)
                    if output_class_mask is not None:
                        output_class_mask_jax = jnp.array(output_class_mask)
                        masked_bias = masked_bias * output_class_mask_jax

                    # Check if this is aggregator bias - apply task partitioning
                    agg_mask = self.compute_aggregator_task_mask(output_dim)
                    if agg_mask is not None:
                        agg_mask_jax = jnp.array(agg_mask)
                        masked_bias = masked_bias * agg_mask_jax

                    masked_biases[bias_name] = masked_bias

                else:
                    masked_biases[bias_name] = bias_grad

            # Reconstruct NodeParams-like structure
            from fabricpc.core.types import NodeParams

            return NodeParams(weights=masked_weights, biases=masked_biases)

        # Apply to all nodes in the gradient tree
        if hasattr(grads, "nodes"):
            # GraphParams structure
            masked_nodes = {}
            params_nodes = params.nodes if hasattr(params, "nodes") else {}
            for node_name, node_grads in grads.nodes.items():
                node_params = params_nodes.get(node_name)
                masked_nodes[node_name] = apply_mask_to_node(
                    node_name, node_grads, node_params
                )

            from fabricpc.core.types import GraphParams

            return GraphParams(nodes=masked_nodes)
        else:
            # Unknown structure, return original
            return grads

    def apply_gradient_clipping(self, grads: Any) -> Any:
        """Apply gradient norm clipping."""
        if self.config.max_gradient_norm is None:
            return grads

        max_norm = self.config.max_gradient_norm

        # Compute global gradient norm
        def compute_norm(tree):
            leaves = tree_util.tree_leaves(tree)
            return jnp.sqrt(
                sum(jnp.sum(jnp.square(x)) for x in leaves if x is not None)
            )

        grad_norm = compute_norm(grads)
        scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))

        def scale_leaf(x):
            if x is None:
                return x
            return x * scale

        return tree_util.tree_map(scale_leaf, grads)

    def get_protection_stats(self) -> Dict[str, Any]:
        """Get statistics about current protection state."""
        column_mask = self.compute_column_mask()

        num_fully_protected = np.sum(column_mask == 0.0)
        num_partially_protected = np.sum((column_mask > 0.0) & (column_mask < 1.0))
        num_unprotected = np.sum(column_mask == 1.0)

        stats = {
            "task_id": self.state.task_id,
            "num_columns": self.state.num_columns,
            "num_active": len(self.state.active_columns),
            "num_shared": len(self.state.shared_columns),
            "columns_fully_protected": int(num_fully_protected),
            "columns_partially_protected": int(num_partially_protected),
            "columns_unprotected": int(num_unprotected),
            "column_mask_mean": float(np.mean(column_mask)),
        }

        # Shell statistics
        if self.state.column_shell_assignments:
            shell_counts = {s: 0 for s in range(len(self.config.shell_scales))}
            for assignments in self.state.column_shell_assignments.values():
                for shell_id in assignments:
                    shell_id = int(shell_id)
                    if shell_id in shell_counts:
                        shell_counts[shell_id] += 1
            stats["shell_distribution"] = shell_counts

        return stats


# ------------------------------------
# Optax-compatible Gradient Transform
# ------------------------------------

try:
    import optax
    from optax._src.base import GradientTransformation, EmptyState

    class GradientProtectionState(optax.EmptyState):
        """Empty state for gradient protection transform (stateless)."""

        pass

    def gradient_protection_transform(
        protector: GradientProtector,
    ) -> optax.GradientTransformation:
        """
        Create an optax gradient transformation that applies gradient protection.

        This transformation can be chained with other optimizers:
            optimizer = optax.chain(
                gradient_protection_transform(protector),
                optax.adamw(learning_rate=1e-3)
            )

        Args:
            protector: Configured GradientProtector instance

        Returns:
            Optax GradientTransformation
        """

        def init_fn(params):
            return EmptyState()

        def update_fn(updates, state, params=None):
            # Apply gradient protection mask
            if protector.config.enable:
                protected_updates = protector.apply_protection(updates, params)
                # Apply gradient clipping
                protected_updates = protector.apply_gradient_clipping(protected_updates)
            else:
                protected_updates = updates
            return protected_updates, state

        return optax.GradientTransformation(init_fn, update_fn)

except ImportError:
    # optax not available, skip transformation
    def gradient_protection_transform(protector):
        raise ImportError("optax is required for gradient_protection_transform")
