"""
Custom FabricPC Nodes for Continual Learning.

Implements specialized nodes for the hierarchical column-based architecture
used in Split-MNIST continual learning experiments.
"""

from typing import Dict, Any, Tuple, Optional, Sequence
import math
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.initializers import NormalInitializer, initialize
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import GaussianEnergy

# Module-level variables for current task context.
# These are set by SequentialTrainer before training/evaluation so continual
# routing remains active even when the graph has no explicit task/mask edges.
_CURRENT_TASK_ID = 0
_CURRENT_SUPPORT_COLS: Optional[Tuple[int, ...]] = None


def set_current_task_id(task_id: int) -> None:
    """Set the current task ID for ComposerNode attention routing."""
    global _CURRENT_TASK_ID
    _CURRENT_TASK_ID = task_id


def get_current_task_id() -> int:
    """Get the current task ID for ComposerNode attention routing."""
    return _CURRENT_TASK_ID


def set_current_support_cols(active_columns: Sequence[int] | None) -> None:
    """Set the active support columns for continual routing."""
    global _CURRENT_SUPPORT_COLS
    if active_columns is None:
        _CURRENT_SUPPORT_COLS = None
    else:
        _CURRENT_SUPPORT_COLS = tuple(int(col) for col in active_columns)


def get_current_support_mask(batch_size: int, num_columns: int) -> jnp.ndarray:
    """Build a batch mask from the current support selection."""
    if _CURRENT_SUPPORT_COLS is None:
        return jnp.ones((batch_size, num_columns), dtype=jnp.float32)

    mask = np.zeros((batch_size, num_columns), dtype=np.float32)
    for col_idx in _CURRENT_SUPPORT_COLS:
        if 0 <= col_idx < num_columns:
            mask[:, col_idx] = 1.0
    return jnp.array(mask)


class PatchEmbedNode(NodeBase, FlattenInputMixin):
    """
    Patch embedding node for image tokenization.

    Splits an image into non-overlapping patches and projects each patch
    to an embedding dimension. Optionally adds positional encodings.

    Input: (batch, H, W, C) or (batch, H*W*C) if flatten_input=True
    Output: (batch, num_patches, embed_dim)

    Args:
        shape: Output shape (num_patches, embed_dim)
        patch_size: Size of each square patch
        image_size: Input image size (assumes square images)
        add_positional: Whether to add learnable positional embeddings
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        name: str,
        patch_size: int = 4,
        image_size: int = 28,
        add_positional: bool = True,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
        **kwargs,
    ):
        num_patches = (image_size // patch_size) ** 2
        embed_dim = shape[1] if len(shape) == 2 else shape[0]
        computed_shape = (num_patches, embed_dim)

        super().__init__(
            shape=computed_shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            patch_size=patch_size,
            image_size=image_size,
            add_positional=add_positional,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.02)

        patch_size = config.get("patch_size", 4)
        add_positional = config.get("add_positional", True)
        num_patches, embed_dim = node_shape

        # Calculate input patch dimension
        patch_dim = patch_size * patch_size  # Grayscale

        keys = jax.random.split(key, 3)

        # Patch projection: (patch_dim) -> (embed_dim)
        weights = {
            "patch_proj": initialize(keys[0], (patch_dim, embed_dim), weight_init)
        }

        biases = {"patch_bias": jnp.zeros((embed_dim,))}

        # Positional embeddings
        if add_positional:
            weights["pos_embed"] = initialize(
                keys[1], (num_patches, embed_dim), weight_init
            )

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        patch_size = config.get("patch_size", 4)
        image_size = config.get("image_size", 28)
        add_positional = config.get("add_positional", True)

        # Get input image
        x = list(inputs.values())[0]  # (batch, 784) or (batch, 28, 28, 1)
        batch_size = x.shape[0]

        # Reshape to image format if flat
        if x.ndim == 2:
            x = x.reshape(batch_size, image_size, image_size, 1)

        # Extract patches: (batch, H, W, 1) -> (batch, num_patches, patch_dim)
        num_patches_per_side = image_size // patch_size
        num_patches = num_patches_per_side**2
        patch_dim = patch_size * patch_size

        # Use reshape and transpose to extract patches
        x = x.reshape(
            batch_size,
            num_patches_per_side,
            patch_size,
            num_patches_per_side,
            patch_size,
            1,
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, H', W', p, p, C)
        patches = x.reshape(batch_size, num_patches, patch_dim)

        # Project patches
        patch_embed = jnp.matmul(patches, params.weights["patch_proj"])
        patch_embed = patch_embed + params.biases["patch_bias"]

        # Add positional embeddings
        if add_positional and "pos_embed" in params.weights:
            patch_embed = patch_embed + params.weights["pos_embed"]

        # Apply activation
        activation = node_info.activation
        z_mu = type(activation).forward(patch_embed, activation.config)

        # Compute error and energy
        error = state.z_latent - z_mu
        state = state._replace(
            pre_activation=patch_embed,
            z_mu=z_mu,
            error=error,
        )

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


class ColumnNode(NodeBase):
    """
    Memory column node for continual learning.

    Each column maintains a memory representation that can be selectively
    activated based on task requirements. Supports shell-based hierarchical
    organization.

    Input: (batch, num_patches, embed_dim) from PatchEmbed
    Output: (batch, num_columns, memory_dim)

    Args:
        shape: Output shape (num_columns, memory_dim)
        num_shells: Number of hierarchical shell levels
        shell_sizes: Tuple of columns per shell
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        name: str,
        num_shells: int = 3,
        shell_sizes: Tuple[int, ...] = (8, 16, 8),
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_shells=num_shells,
            shell_sizes=shell_sizes,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "in": SlotSpec(name="in", is_multi_input=False),
            "mask": SlotSpec(name="mask", is_multi_input=False),  # Optional active mask
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.02)

        num_columns, memory_dim = node_shape

        # Get input shape from the edge connected to "in" slot
        # input_shapes keys are edge keys like "source->target:slot"
        in_shape = None
        for edge_key, shape in input_shapes.items():
            if ":in" in edge_key:
                in_shape = shape
                break

        if in_shape is None:
            # Fallback for backwards compatibility
            in_shape = (784,)  # Default to flattened MNIST

        # Compute input dimension (flatten if needed)
        input_dim = int(np.prod(in_shape))

        keys = jax.random.split(key, num_columns + 2)

        # Per-column projections - each column has separate weights
        weights = {}
        for col_idx in range(num_columns):
            weights[f"col_{col_idx}_proj"] = initialize(
                keys[col_idx], (input_dim, memory_dim), weight_init
            )

        biases = {"column_bias": jnp.zeros((num_columns, memory_dim))}

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        num_columns, memory_dim = node_info.shape

        # Get patch embeddings
        x = inputs.get("in", inputs.get(list(inputs.keys())[0]))
        batch_size = x.shape[0]

        # Flatten patch embeddings
        x_flat = x.reshape(batch_size, -1)

        # Get optional mask
        mask = inputs.get("mask", None)
        if mask is None:
            mask = get_current_support_mask(batch_size, num_columns)

        # Compute column outputs
        col_outputs = []
        for col_idx in range(num_columns):
            col_proj = params.weights[f"col_{col_idx}_proj"]
            col_out = jnp.matmul(x_flat, col_proj)
            col_outputs.append(col_out)

        # Stack: (batch, num_columns, memory_dim)
        column_out = jnp.stack(col_outputs, axis=1)
        column_out = column_out + params.biases["column_bias"]

        # Apply mask
        column_out = column_out * mask[:, :, None]

        # Apply activation
        activation = node_info.activation
        z_mu = type(activation).forward(column_out, activation.config)

        # Compute error
        error = state.z_latent - z_mu
        state = state._replace(
            pre_activation=column_out,
            z_mu=z_mu,
            error=error,
        )

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


class ComposerNode(NodeBase):
    """
    Attention-based composer node for combining column outputs.

    Uses multi-head self-attention to aggregate information across columns,
    with task-specific query vectors and gating mechanisms.

    Input: (batch, num_columns, memory_dim) from ColumnNode
    Output: (batch, hidden_dim)

    Args:
        shape: Output shape (hidden_dim,)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate (only used in training)
    """

    def __init__(
        self,
        shape: Tuple[int],
        name: str,
        num_heads: int = 2,
        num_layers: int = 1,
        num_tasks: int = 5,
        gate_temp: float = 0.5,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_heads=num_heads,
            num_layers=num_layers,
            num_tasks=num_tasks,
            gate_temp=gate_temp,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "in": SlotSpec(name="in", is_multi_input=False),
            "task_id": SlotSpec(name="task_id", is_multi_input=False),
            "mask": SlotSpec(name="mask", is_multi_input=False),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.02)

        hidden_dim = node_shape[0]
        num_heads = config.get("num_heads", 2)
        num_layers = config.get("num_layers", 1)
        num_tasks = config.get("num_tasks", 5)

        # Get input dimensions from edge connected to "in" slot
        # input_shapes keys are edge keys like "source->target:in"
        in_shape = None
        for edge_key, shape in input_shapes.items():
            if ":in" in edge_key:
                in_shape = shape
                break

        if in_shape is None:
            # Fallback for backwards compatibility
            in_shape = (32, 64)

        if len(in_shape) == 2:
            num_columns, input_dim = in_shape
        else:
            input_dim = in_shape[0]
            num_columns = 32

        head_dim = hidden_dim // num_heads
        keys = jax.random.split(key, 10 + num_layers * 4)
        key_idx = 0

        weights = {}
        biases = {}

        # Input projection
        weights["input_proj"] = initialize(
            keys[key_idx], (input_dim, hidden_dim), weight_init
        )
        key_idx += 1
        biases["input_proj_bias"] = jnp.zeros((hidden_dim,))

        # Attention layers
        for layer in range(num_layers):
            # QKV projections
            weights[f"layer_{layer}_q"] = initialize(
                keys[key_idx], (hidden_dim, hidden_dim), weight_init
            )
            key_idx += 1
            weights[f"layer_{layer}_k"] = initialize(
                keys[key_idx], (hidden_dim, hidden_dim), weight_init
            )
            key_idx += 1
            weights[f"layer_{layer}_v"] = initialize(
                keys[key_idx], (hidden_dim, hidden_dim), weight_init
            )
            key_idx += 1
            weights[f"layer_{layer}_out"] = initialize(
                keys[key_idx], (hidden_dim, hidden_dim), weight_init
            )
            key_idx += 1

            biases[f"layer_{layer}_out_bias"] = jnp.zeros((hidden_dim,))

        # Task queries
        weights["task_queries"] = initialize(
            keys[key_idx], (num_tasks, hidden_dim), weight_init
        )
        key_idx += 1

        # Gate projection
        weights["gate_proj"] = initialize(keys[key_idx], (hidden_dim, 1), weight_init)
        key_idx += 1
        biases["gate_bias"] = jnp.zeros((1,))

        # Output projection
        weights["output_proj"] = initialize(
            keys[key_idx], (hidden_dim, hidden_dim), weight_init
        )
        biases["output_bias"] = jnp.zeros((hidden_dim,))

        # Layer norms (as scale + shift)
        for layer in range(num_layers + 1):
            weights[f"ln_{layer}_scale"] = jnp.ones((hidden_dim,))
            biases[f"ln_{layer}_shift"] = jnp.zeros((hidden_dim,))

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        num_heads = config.get("num_heads", 2)
        num_layers = config.get("num_layers", 1)
        gate_temp = config.get("gate_temp", 0.5)
        hidden_dim = node_info.shape[0]
        head_dim = hidden_dim // num_heads

        # Get column outputs
        x = inputs.get("in", inputs.get(list(inputs.keys())[0]))
        batch_size = x.shape[0]
        num_columns = x.shape[1]

        # Get task_id (as one-hot or scalar from input, or from module-level variable)
        task_id_input = inputs.get("task_id", None)
        if task_id_input is not None:
            if task_id_input.ndim == 2:  # One-hot
                task_id = jnp.argmax(task_id_input[0])
            else:
                task_id = task_id_input[0].astype(jnp.int32)
        else:
            # Fallback to module-level task_id (set by trainer via set_current_task_id)
            task_id = get_current_task_id()

        # Get mask
        mask = inputs.get("mask", None)
        if mask is None:
            mask = get_current_support_mask(batch_size, num_columns)

        # Input projection
        x = (
            jnp.matmul(x, params.weights["input_proj"])
            + params.biases["input_proj_bias"]
        )

        # Layer norm
        x = ComposerNode._layer_norm(
            x, params.weights["ln_0_scale"], params.biases["ln_0_shift"]
        )

        # Attention layers
        for layer in range(num_layers):
            # Self-attention
            Q = jnp.matmul(x, params.weights[f"layer_{layer}_q"])
            K = jnp.matmul(x, params.weights[f"layer_{layer}_k"])
            V = jnp.matmul(x, params.weights[f"layer_{layer}_v"])

            # Reshape for multi-head attention
            Q = Q.reshape(batch_size, num_columns, num_heads, head_dim).transpose(
                0, 2, 1, 3
            )
            K = K.reshape(batch_size, num_columns, num_heads, head_dim).transpose(
                0, 2, 1, 3
            )
            V = V.reshape(batch_size, num_columns, num_heads, head_dim).transpose(
                0, 2, 1, 3
            )

            # Attention scores
            scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

            # Apply mask
            mask_expanded = mask[:, None, None, :]  # (B, 1, 1, C)
            scores = jnp.where(mask_expanded > 0, scores, -1e9)

            # Softmax
            attn = jax.nn.softmax(scores, axis=-1)

            # Apply attention
            out = jnp.matmul(attn, V)
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, num_columns, hidden_dim)

            # Output projection
            out = jnp.matmul(out, params.weights[f"layer_{layer}_out"])
            out = out + params.biases[f"layer_{layer}_out_bias"]

            # Residual + layer norm
            x = ComposerNode._layer_norm(
                x + out,
                params.weights[f"ln_{layer + 1}_scale"],
                params.biases[f"ln_{layer + 1}_shift"],
            )

        # Compute gate probabilities using mask-based gating (no task-specific queries)
        # This avoids JAX JIT issues with task_id being captured as a constant
        # The mask already encodes which columns are active for the current task
        gate_logits = jnp.matmul(x, params.weights["gate_proj"]).squeeze(-1)
        gate_logits = jnp.where(mask > 0, gate_logits, -1e9)
        gate_probs = jax.nn.softmax(gate_logits / gate_temp, axis=-1)

        # Weighted aggregation
        pooled = jnp.sum(gate_probs[:, :, None] * x, axis=1)

        # Output projection
        output = (
            jnp.matmul(pooled, params.weights["output_proj"])
            + params.biases["output_bias"]
        )

        # Apply activation
        activation = node_info.activation
        z_mu = type(activation).forward(output, activation.config)

        # Compute error
        error = state.z_latent - z_mu
        state = state._replace(
            pre_activation=output,
            z_mu=z_mu,
            error=error,
        )

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state

    @staticmethod
    def _layer_norm(x, scale, shift, eps=1e-5):
        """Apply layer normalization."""
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / jnp.sqrt(var + eps)
        return normalized * scale + shift


class ClassifierNode(NodeBase, FlattenInputMixin):
    """
    Multi-task classifier head node.

    Supports multiple task-specific heads for continual learning scenarios.
    Can operate in shared-head mode (single head for all tasks) or
    task-specific mode (separate head per task).

    Input: (batch, hidden_dim) from ComposerNode or ColumnNode
    Output: (batch, num_classes)
    """

    def __init__(
        self,
        shape: Tuple[int],
        name: str,
        num_tasks: int = 5,
        task_specific_heads: bool = False,
        activation=SoftmaxActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_tasks=num_tasks,
            task_specific_heads=task_specific_heads,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "in": SlotSpec(name="in", is_multi_input=False),
            "task_id": SlotSpec(name="task_id", is_multi_input=False),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.02)

        num_classes = node_shape[0]
        num_tasks = config.get("num_tasks", 5)
        task_specific = config.get("task_specific_heads", False)

        # Get input dimension
        in_shape = input_shapes.get("in", (64,))
        input_dim = in_shape[0] if len(in_shape) == 1 else int(np.prod(in_shape))

        weights = {}
        biases = {}

        if task_specific:
            # Separate head per task
            keys = jax.random.split(key, num_tasks)
            for task_id in range(num_tasks):
                weights[f"head_{task_id}"] = initialize(
                    keys[task_id], (input_dim, num_classes), weight_init
                )
                biases[f"head_{task_id}_bias"] = jnp.zeros((num_classes,))
        else:
            # Shared head
            weights["head"] = initialize(key, (input_dim, num_classes), weight_init)
            biases["head_bias"] = jnp.zeros((num_classes,))

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        task_specific = config.get("task_specific_heads", False)

        # Get input
        x = inputs.get("in", inputs.get(list(inputs.keys())[0]))
        batch_size = x.shape[0]

        # Flatten if needed
        if x.ndim > 2:
            x = x.reshape(batch_size, -1)

        # Get task_id
        task_id_input = inputs.get("task_id", None)
        if task_id_input is not None:
            if task_id_input.ndim == 2:
                task_id = int(jnp.argmax(task_id_input[0]))
            else:
                task_id = int(task_id_input[0])
        else:
            task_id = 0

        # Apply classifier head
        if task_specific:
            logits = jnp.matmul(x, params.weights[f"head_{task_id}"])
            logits = logits + params.biases[f"head_{task_id}_bias"]
        else:
            logits = jnp.matmul(x, params.weights["head"])
            logits = logits + params.biases["head_bias"]

        # Apply activation (softmax for classification)
        activation = node_info.activation
        z_mu = type(activation).forward(logits, activation.config)

        # Compute error
        error = state.z_latent - z_mu
        state = state._replace(
            pre_activation=logits,
            z_mu=z_mu,
            error=error,
        )

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


class GatedColumnNode(NodeBase):
    """
    Column node with learnable gating for support selection.

    Combines the column processing with a gating mechanism that learns
    which columns should be active for each task.

    Input: (batch, input_dim)
    Output: (batch, output_dim)

    The gate is controlled by the support selection mechanism and
    can be dynamically adjusted during continual learning.
    """

    def __init__(
        self,
        shape: Tuple[int],
        name: str,
        num_columns: int = 32,
        shared_columns: int = 8,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_columns=num_columns,
            shared_columns=shared_columns,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "in": SlotSpec(name="in", is_multi_input=False),
            "gate": SlotSpec(name="gate", is_multi_input=False),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.02)

        output_dim = node_shape[0]
        num_columns = config.get("num_columns", 32)

        # Get input dimension
        in_shape = input_shapes.get("in", (784,))
        input_dim = in_shape[0] if len(in_shape) == 1 else int(np.prod(in_shape))

        column_dim = output_dim // num_columns

        keys = jax.random.split(key, num_columns + 1)

        weights = {}
        for col_idx in range(num_columns):
            weights[f"col_{col_idx}"] = initialize(
                keys[col_idx], (input_dim, column_dim), weight_init
            )

        biases = {"bias": jnp.zeros((output_dim,))}

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        num_columns = config.get("num_columns", 32)
        output_dim = node_info.shape[0]
        column_dim = output_dim // num_columns

        # Get input
        x = inputs.get("in", inputs.get(list(inputs.keys())[0]))
        batch_size = x.shape[0]

        # Flatten if needed
        if x.ndim > 2:
            x = x.reshape(batch_size, -1)

        # Get gate (default: all ones = all columns active)
        gate = inputs.get("gate", jnp.ones((batch_size, num_columns)))

        # Compute column outputs
        col_outputs = []
        for col_idx in range(num_columns):
            col_out = jnp.matmul(x, params.weights[f"col_{col_idx}"])
            col_out = col_out * gate[:, col_idx : col_idx + 1]
            col_outputs.append(col_out)

        # Concatenate columns
        output = jnp.concatenate(col_outputs, axis=-1)
        output = output + params.biases["bias"]

        # Apply activation
        activation = node_info.activation
        z_mu = type(activation).forward(output, activation.config)

        # Compute error
        error = state.z_latent - z_mu
        state = state._replace(
            pre_activation=output,
            z_mu=z_mu,
            error=error,
        )

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


class PartitionedAggregator(NodeBase):
    """
    Aggregator with task-specific partitioned pathways for continual learning.

    This node creates TRUE architectural isolation between tasks by having
    separate weight matrices for each task's columns. Unlike gradient masking,
    there are no connections between task partitions - the weights simply don't exist.

    Architecture:
    - Input: (batch, num_columns, memory_dim) from ColumnNode
    - Shared columns (always first shared_columns) → shared partition
    - Task-specific columns → task-specific partition (separate weights per task)
    - Output: (batch, shared_dim + task_dim)

    Connection Pattern:
    - Shared cols [0:shared_columns] → weights_shared: (shared_cols * mem_dim, shared_dim)
    - Task t cols [start_t:end_t] → weights_task_t: (topk_nonshared * mem_dim, task_dim)

    Why this works:
    - No gradient flow between task partitions (architectural isolation)
    - Shared partition enables knowledge transfer
    - Each task's columns only connect to that task's partition

    Args:
        shape: Output shape (shared_dim + task_dim,)
        num_tasks: Number of tasks (default: 5 for Split-MNIST)
        shared_columns: Number of shared columns (default: 2)
        topk_nonshared: Columns per task (default: 4)
        shared_dim: Output neurons for shared pathway (default: 32)
        task_dim: Output neurons per task pathway (default: 64)
        memory_dim: Input memory dimension per column (default: 64)
    """

    def __init__(
        self,
        shape: Tuple[int],
        name: str,
        num_tasks: int = 5,
        shared_columns: int = 2,
        topk_nonshared: int = 4,
        shared_dim: int = 32,
        task_dim: int = 64,
        memory_dim: int = 64,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
        **kwargs,
    ):
        # Output shape should match shared_dim + task_dim
        actual_shape = (shared_dim + task_dim,)

        super().__init__(
            shape=actual_shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_tasks=num_tasks,
            shared_columns=shared_columns,
            topk_nonshared=topk_nonshared,
            shared_dim=shared_dim,
            task_dim=task_dim,
            memory_dim=memory_dim,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "in": SlotSpec(name="in", is_multi_input=False),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Dict[str, Any] = {},
    ) -> NodeParams:
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.02)

        from fabricpc.core.initializers import initialize

        num_tasks = config.get("num_tasks", 5)
        shared_columns = config.get("shared_columns", 2)
        topk_nonshared = config.get("topk_nonshared", 4)
        shared_dim = config.get("shared_dim", 32)
        task_dim = config.get("task_dim", 64)
        memory_dim = config.get("memory_dim", 64)

        # Split keys for all weight matrices
        keys = jax.random.split(key, num_tasks + 2)

        weights = {}
        biases = {}

        # Shared pathway weights: (shared_columns * memory_dim, shared_dim)
        shared_input_dim = shared_columns * memory_dim
        weights["shared"] = initialize(
            keys[0], (shared_input_dim, shared_dim), weight_init
        )
        biases["shared_bias"] = jnp.zeros((shared_dim,))

        # Task-specific pathway weights: one per task
        # Each task's columns: (topk_nonshared * memory_dim, task_dim)
        task_input_dim = topk_nonshared * memory_dim
        for task_id in range(num_tasks):
            weights[f"task_{task_id}"] = initialize(
                keys[task_id + 1], (task_input_dim, task_dim), weight_init
            )
            biases[f"task_{task_id}_bias"] = jnp.zeros((task_dim,))

        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        config = node_info.node_config
        num_tasks = config.get("num_tasks", 5)
        shared_columns = config.get("shared_columns", 2)
        topk_nonshared = config.get("topk_nonshared", 4)
        shared_dim = config.get("shared_dim", 32)
        task_dim = config.get("task_dim", 64)
        memory_dim = config.get("memory_dim", 64)

        # Get input: (batch, num_columns, memory_dim)
        x = inputs.get("in", inputs.get(list(inputs.keys())[0]))
        batch_size = x.shape[0]

        # Get current task_id from module-level variable
        task_id = get_current_task_id()

        # Clamp task_id to valid range
        task_id = max(0, min(task_id, num_tasks - 1))

        # === Shared pathway ===
        # Extract shared columns: (batch, shared_columns, memory_dim)
        x_shared = x[:, :shared_columns, :]
        # Flatten: (batch, shared_columns * memory_dim)
        x_shared_flat = x_shared.reshape(batch_size, -1)
        # Project: (batch, shared_dim)
        shared_out = jnp.matmul(x_shared_flat, params.weights["shared"])
        shared_out = shared_out + params.biases["shared_bias"]

        # === Task-specific pathway ===
        # Calculate which columns belong to this task
        # Non-shared columns start after shared_columns
        # Task t's columns: [shared_columns + t*topk_nonshared : shared_columns + (t+1)*topk_nonshared]
        task_col_start = shared_columns + task_id * topk_nonshared
        task_col_end = task_col_start + topk_nonshared

        # Extract task columns: (batch, topk_nonshared, memory_dim)
        x_task = jax.lax.dynamic_slice(
            x,
            (0, task_col_start, 0),
            (batch_size, topk_nonshared, memory_dim),
        )
        # Flatten: (batch, topk_nonshared * memory_dim)
        x_task_flat = x_task.reshape(batch_size, -1)

        # Project using task-specific weights: (batch, task_dim)
        # Use lax.switch for JIT-compatible task selection
        def get_task_output(t):
            return (
                jnp.matmul(x_task_flat, params.weights[f"task_{t}"])
                + params.biases[f"task_{t}_bias"]
            )

        # Create branches for each task
        branches = [lambda t=t: get_task_output(t) for t in range(num_tasks)]
        task_out = jax.lax.switch(task_id, branches)

        # === Combine pathways ===
        # Output: (batch, shared_dim + task_dim)
        output = jnp.concatenate([shared_out, task_out], axis=-1)

        # Apply activation
        activation = node_info.activation
        z_mu = type(activation).forward(output, activation.config)

        # Compute error
        error = state.z_latent - z_mu
        state = state._replace(
            pre_activation=output,
            z_mu=z_mu,
            error=error,
        )

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state
