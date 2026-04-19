"""
Augmentation mechanisms for Continual Learning.

Implements Cloud augmentation and TransWeave transport for
improving support selection and knowledge transfer.
"""

from typing import Dict, List, Tuple, Optional, Sequence, Any
import numpy as np
import jax.numpy as jnp


class CloudAugmenter:
    """
    Cloud augmentation for support and composer evaluation.

    Generates variants of inputs using transformations like:
    - Shifts (left, right, up, down)
    - Noise injection
    - Masking
    - Contrast adjustment

    Used to evaluate support columns under different conditions
    and improve robustness of selection.
    """

    def __init__(
        self,
        shift_pixels: int = 1,
        noise_std: float = 0.06,
        mask_patch: int = 7,
        contrast_scale: float = 0.90,
        image_size: int = 28,
    ):
        """
        Initialize cloud augmenter.

        Args:
            shift_pixels: Number of pixels to shift
            noise_std: Standard deviation for noise injection
            mask_patch: Size of masking patch
            contrast_scale: Contrast adjustment factor
            image_size: Size of input images
        """
        self.shift_pixels = shift_pixels
        self.noise_std = noise_std
        self.mask_patch = mask_patch
        self.contrast_scale = contrast_scale
        self.image_size = image_size

    def generate_variants(
        self,
        images: jnp.ndarray,
        variant_types: Sequence[str] = ("identity", "shift_left", "noise"),
        rng_key: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Generate augmented variants of input images.

        Args:
            images: Input images (batch, H, W, C) or (batch, H*W)
            variant_types: Types of variants to generate
            rng_key: JAX random key for stochastic augmentations

        Returns:
            Dictionary mapping variant name to augmented images
        """
        # Reshape to image format if flat
        original_shape = images.shape
        if images.ndim == 2:
            batch_size = images.shape[0]
            images = images.reshape(batch_size, self.image_size, self.image_size, 1)
        else:
            batch_size = images.shape[0]

        variants = {}

        for vtype in variant_types:
            if vtype == "identity":
                variants[vtype] = images

            elif vtype == "shift_left":
                shifted = jnp.roll(images, -self.shift_pixels, axis=2)
                # Zero out wrapped pixels
                shifted = shifted.at[:, :, -self.shift_pixels :, :].set(0)
                variants[vtype] = shifted

            elif vtype == "shift_right":
                shifted = jnp.roll(images, self.shift_pixels, axis=2)
                shifted = shifted.at[:, :, : self.shift_pixels, :].set(0)
                variants[vtype] = shifted

            elif vtype == "shift_up":
                shifted = jnp.roll(images, -self.shift_pixels, axis=1)
                shifted = shifted.at[:, -self.shift_pixels :, :, :].set(0)
                variants[vtype] = shifted

            elif vtype == "shift_down":
                shifted = jnp.roll(images, self.shift_pixels, axis=1)
                shifted = shifted.at[:, : self.shift_pixels, :, :].set(0)
                variants[vtype] = shifted

            elif vtype == "noise":
                if rng_key is not None:
                    noise = jax.random.normal(rng_key, images.shape) * self.noise_std
                else:
                    noise = (
                        np.random.randn(*images.shape).astype(np.float32)
                        * self.noise_std
                    )
                    noise = jnp.array(noise)
                variants[vtype] = images + noise

            elif vtype == "mask":
                # Random rectangular mask
                masked = images.copy()
                h_start = np.random.randint(0, self.image_size - self.mask_patch)
                w_start = np.random.randint(0, self.image_size - self.mask_patch)
                masked = masked.at[
                    :,
                    h_start : h_start + self.mask_patch,
                    w_start : w_start + self.mask_patch,
                    :,
                ].set(0)
                variants[vtype] = masked

            elif vtype == "contrast":
                mean = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
                contrasted = mean + self.contrast_scale * (images - mean)
                variants[vtype] = contrasted

            else:
                # Unknown variant type, use identity
                variants[vtype] = images

        # Reshape back to original format if needed
        if len(original_shape) == 2:
            variants = {k: v.reshape(batch_size, -1) for k, v in variants.items()}

        return variants


class TransWeaveTransport:
    """
    TransWeave optimal transport for knowledge transfer.

    Uses Sinkhorn algorithm to compute soft transport maps between
    column representations across tasks, enabling knowledge transfer
    without catastrophic forgetting.
    """

    def __init__(
        self,
        sinkhorn_eps: float = 0.30,
        sinkhorn_iters: int = 20,
        identity_bonus: float = 0.12,
        transport_diag_mix: float = 0.35,
    ):
        """
        Initialize TransWeave transport.

        Args:
            sinkhorn_eps: Regularization parameter for Sinkhorn
            sinkhorn_iters: Number of Sinkhorn iterations
            identity_bonus: Bonus for identity (diagonal) transport
            transport_diag_mix: Mixing weight for diagonal transport
        """
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.identity_bonus = identity_bonus
        self.transport_diag_mix = transport_diag_mix

    def compute_cost_matrix(
        self,
        source_repr: jnp.ndarray,
        target_repr: jnp.ndarray,
        metric: str = "cosine",
    ) -> jnp.ndarray:
        """
        Compute cost matrix between source and target representations.

        Args:
            source_repr: Source representations (num_source, dim)
            target_repr: Target representations (num_target, dim)
            metric: Distance metric ("cosine" or "euclidean")

        Returns:
            Cost matrix (num_source, num_target)
        """
        if metric == "cosine":
            # Normalize
            source_norm = source_repr / (
                jnp.linalg.norm(source_repr, axis=1, keepdims=True) + 1e-8
            )
            target_norm = target_repr / (
                jnp.linalg.norm(target_repr, axis=1, keepdims=True) + 1e-8
            )
            # Cosine similarity -> cost (1 - sim)
            similarity = jnp.matmul(source_norm, target_norm.T)
            cost = 1.0 - similarity
        else:
            # Euclidean distance
            diff = source_repr[:, None, :] - target_repr[None, :, :]
            cost = jnp.sqrt(jnp.sum(diff**2, axis=-1))

        return cost

    def sinkhorn_transport(
        self,
        cost: jnp.ndarray,
        source_weights: Optional[jnp.ndarray] = None,
        target_weights: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute optimal transport plan using Sinkhorn algorithm.

        Args:
            cost: Cost matrix (n, m)
            source_weights: Source marginal weights (n,)
            target_weights: Target marginal weights (m,)

        Returns:
            Transport plan (n, m)
        """
        n, m = cost.shape

        # Default to uniform weights
        if source_weights is None:
            source_weights = jnp.ones(n) / n
        if target_weights is None:
            target_weights = jnp.ones(m) / m

        # Add identity bonus to diagonal (if square)
        if n == m and self.identity_bonus > 0:
            identity_cost = jnp.eye(n) * (-self.identity_bonus)
            cost = cost + identity_cost

        # Initialize
        K = jnp.exp(-cost / self.sinkhorn_eps)
        u = jnp.ones(n)
        v = jnp.ones(m)

        # Sinkhorn iterations
        for _ in range(self.sinkhorn_iters):
            u = source_weights / (jnp.matmul(K, v) + 1e-10)
            v = target_weights / (jnp.matmul(K.T, u) + 1e-10)

        # Transport plan
        transport = u[:, None] * K * v[None, :]

        # Mix with diagonal transport
        if n == m and self.transport_diag_mix > 0:
            diag_transport = jnp.eye(n) / n
            transport = (
                1 - self.transport_diag_mix
            ) * transport + self.transport_diag_mix * diag_transport

        return transport

    def apply_transport(
        self,
        source_features: jnp.ndarray,
        transport_plan: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply transport plan to transfer features.

        Args:
            source_features: Source feature matrix (num_source, dim)
            transport_plan: Transport plan (num_source, num_target)

        Returns:
            Transported features (num_target, dim)
        """
        # Normalize transport plan rows
        plan_normalized = transport_plan / (
            jnp.sum(transport_plan, axis=0, keepdims=True) + 1e-10
        )
        # Transport: weighted combination of source features
        transported = jnp.matmul(plan_normalized.T, source_features)
        return transported


class TransWeaveManager:
    """
    Manager for TransWeave transport across tasks.

    Maintains task-specific representations and computes transport
    maps for knowledge transfer during continual learning.
    """

    def __init__(
        self,
        num_columns: int,
        feature_dim: int,
        use_last_k_tasks: int = 3,
        source_local_weight: float = 0.50,
        source_block_weight: float = 0.30,
        source_global_weight: float = 0.20,
        **transport_kwargs,
    ):
        """
        Initialize TransWeave manager.

        Args:
            num_columns: Number of columns
            feature_dim: Feature dimension per column
            use_last_k_tasks: Number of recent tasks to use as sources
            source_local_weight: Weight for local (task-specific) features
            source_block_weight: Weight for block-level features
            source_global_weight: Weight for global features
            **transport_kwargs: Arguments for TransWeaveTransport
        """
        self.num_columns = num_columns
        self.feature_dim = feature_dim
        self.use_last_k_tasks = use_last_k_tasks
        self.source_local_weight = source_local_weight
        self.source_block_weight = source_block_weight
        self.source_global_weight = source_global_weight

        self.transport = TransWeaveTransport(**transport_kwargs)

        # Task representations: task_id -> column features
        self.task_representations: Dict[int, jnp.ndarray] = {}

        # Transport history
        self.transport_history: List[Dict[str, Any]] = []

    def register_task_representation(
        self,
        task_id: int,
        column_features: jnp.ndarray,
    ):
        """
        Register column representations for a task.

        Args:
            task_id: Task ID
            column_features: Column feature matrix (num_columns, feature_dim)
        """
        self.task_representations[task_id] = column_features

    def compute_transfer(
        self,
        target_task_id: int,
        target_features: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute transferred features from previous tasks.

        Args:
            target_task_id: Target task ID
            target_features: Target column features (num_columns, feature_dim)

        Returns:
            Tuple of (transferred_features, metadata)
        """
        # Get source tasks
        available_tasks = sorted(
            [t for t in self.task_representations.keys() if t < target_task_id]
        )

        if not available_tasks:
            return target_features, {"source_tasks": [], "transport_applied": False}

        # Use last k tasks
        source_tasks = available_tasks[-self.use_last_k_tasks :]

        # Compute weighted transfer from each source
        transferred = jnp.zeros_like(target_features)
        total_weight = 0.0

        transport_details = []

        for source_task in source_tasks:
            source_features = self.task_representations[source_task]

            # Compute cost and transport
            cost = self.transport.compute_cost_matrix(source_features, target_features)
            plan = self.transport.sinkhorn_transport(cost)

            # Apply transport
            source_transferred = self.transport.apply_transport(source_features, plan)

            # Weight based on recency
            recency_weight = 1.0 / (target_task_id - source_task)
            transferred = transferred + recency_weight * source_transferred
            total_weight += recency_weight

            transport_details.append(
                {
                    "source_task": source_task,
                    "recency_weight": float(recency_weight),
                    "transport_cost": float(jnp.sum(cost * plan)),
                }
            )

        # Normalize
        if total_weight > 0:
            transferred = transferred / total_weight

        # Mix with target features
        final_features = (
            self.source_local_weight * target_features
            + (1 - self.source_local_weight) * transferred
        )

        metadata = {
            "source_tasks": source_tasks,
            "transport_applied": True,
            "transport_details": transport_details,
        }

        self.transport_history.append(
            {
                "target_task": target_task_id,
                **metadata,
            }
        )

        return final_features, metadata

    def get_weakness_scores(
        self,
        task_id: int,
        column_features: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute per-column weakness scores based on transport analysis.

        Columns that transport poorly may be "weak" and candidates
        for retraining or demotion.

        Args:
            task_id: Current task ID
            column_features: Column features (num_columns, feature_dim)

        Returns:
            Weakness scores per column (num_columns,)
        """
        if len(self.task_representations) == 0:
            return jnp.zeros(self.num_columns)

        weakness_scores = jnp.zeros(self.num_columns)

        for source_task, source_features in self.task_representations.items():
            if source_task >= task_id:
                continue

            # Compute cost matrix
            cost = self.transport.compute_cost_matrix(source_features, column_features)

            # Per-column minimum cost (how well each target column matches sources)
            min_costs = jnp.min(cost, axis=0)
            weakness_scores = weakness_scores + min_costs

        # Normalize
        num_sources = sum(1 for t in self.task_representations if t < task_id)
        if num_sources > 0:
            weakness_scores = weakness_scores / num_sources

        return weakness_scores


# Import JAX here to avoid circular imports
try:
    import jax
except ImportError:
    jax = None
