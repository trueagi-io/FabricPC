"""
TransWeave Transfer Learning for Continual Learning.

Implements TransWeave optimal transport at two levels as described in V20.2b:

1. **Composer-level TransWeave**: Transfers learned composition patterns across tasks.
   The composer network that combines column outputs benefits from previous tasks'
   learned attention/composition strategies via Sinkhorn transport.

2. **Within-column Shell Demotion TransWeave**: Transfers shell membership patterns
   to guide neuron demotion between rings (protected center -> stable inner tiers ->
   disposable outer tiers). Uses transport to identify which neurons should move
   between shells based on cross-task activation patterns.

Reference: v_20_b_vs_v_18.pdf Sections 4.6, 5.2, 7.7
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # type: ignore

# Import configs from central config module
from fabricpc.continual.config import (
    ComposerTransWeaveConfig,
    ShellDemotionTransWeaveConfig,
)

# Import optimal transport utilities from shared module
from fabricpc.continual.optimal_transport import (
    sinkhorn_transport,
    cosine_cost_matrix,
    euclidean_cost_matrix,
)

# ----------------------------
# Composer-level TransWeave
# ----------------------------


@dataclass
class ComposerRepresentation:
    """Representation of composer state for a task."""

    task_id: int
    attention_weights: np.ndarray  # (num_heads, num_columns, num_columns)
    query_projections: np.ndarray  # (num_heads, hidden_dim, key_dim)
    key_projections: np.ndarray  # (num_heads, hidden_dim, key_dim)
    value_projections: np.ndarray  # (num_heads, hidden_dim, value_dim)
    output_projection: np.ndarray  # (num_heads * value_dim, output_dim)
    gate_logits: Optional[np.ndarray] = None  # (num_columns,)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComposerTransferResult:
    """Result of composer-level transfer."""

    transferred_attention: np.ndarray
    transferred_queries: np.ndarray
    transferred_keys: np.ndarray
    transferred_values: np.ndarray
    transport_plans: List[np.ndarray]
    source_tasks: List[int]
    transfer_strength: float
    diagnostics: Dict[str, float]


class ComposerTransWeave:
    """
    TransWeave transport for composer-level transfer learning.

    The composer network combines outputs from multiple columns. This class
    enables transfer of learned composition patterns (attention weights,
    projections) from previous tasks to new tasks via optimal transport.

    Key insight from V20.2b: The composer should benefit from prior learned
    attention patterns, but in a way that preserves task-specific adaptations.
    Transport provides a principled way to align and transfer these patterns.
    """

    def __init__(self, config: ComposerTransWeaveConfig):
        self.config = config

        # Task representations storage
        self.task_representations: Dict[int, ComposerRepresentation] = {}

        # Transfer history for diagnostics
        self.transfer_history: List[Dict[str, Any]] = []

    def register_task(
        self,
        task_id: int,
        attention_weights: np.ndarray,
        query_projections: np.ndarray,
        key_projections: np.ndarray,
        value_projections: np.ndarray,
        output_projection: np.ndarray,
        gate_logits: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register composer state for a completed task.

        Args:
            task_id: Task identifier
            attention_weights: Learned attention patterns (heads, cols, cols)
            query_projections: Query projection matrices
            key_projections: Key projection matrices
            value_projections: Value projection matrices
            output_projection: Final output projection
            gate_logits: Optional gating logits for columns
            metadata: Optional additional metadata
        """
        self.task_representations[task_id] = ComposerRepresentation(
            task_id=task_id,
            attention_weights=np.asarray(attention_weights),
            query_projections=np.asarray(query_projections),
            key_projections=np.asarray(key_projections),
            value_projections=np.asarray(value_projections),
            output_projection=np.asarray(output_projection),
            gate_logits=np.asarray(gate_logits) if gate_logits is not None else None,
            metadata=metadata or {},
        )

    def compute_transfer(
        self,
        target_task_id: int,
        current_attention: np.ndarray,
        current_queries: np.ndarray,
        current_keys: np.ndarray,
        current_values: np.ndarray,
        column_features: Optional[np.ndarray] = None,
    ) -> ComposerTransferResult:
        """
        Compute transferred composer representations from previous tasks.

        Uses Sinkhorn transport to align attention patterns and projections
        from source tasks to the target task's column structure.

        Args:
            target_task_id: Target task ID
            current_attention: Current attention weights (heads, cols, cols)
            current_queries: Current query projections
            current_keys: Current key projections
            current_values: Current value projections
            column_features: Optional column features for cost computation

        Returns:
            ComposerTransferResult with transferred representations
        """
        config = self.config

        # Get available source tasks
        source_task_ids = sorted(
            [t for t in self.task_representations.keys() if t < target_task_id]
        )

        # Check warmup
        if len(source_task_ids) < config.warmup_tasks:
            return ComposerTransferResult(
                transferred_attention=current_attention,
                transferred_queries=current_queries,
                transferred_keys=current_keys,
                transferred_values=current_values,
                transport_plans=[],
                source_tasks=[],
                transfer_strength=0.0,
                diagnostics={"warmup": True, "num_sources": len(source_task_ids)},
            )

        # Use last k tasks
        source_task_ids = source_task_ids[-config.use_last_k_tasks :]

        # Accumulate transfers with recency weighting
        transferred_attention = np.zeros_like(current_attention)
        transferred_queries = np.zeros_like(current_queries)
        transferred_keys = np.zeros_like(current_keys)
        transferred_values = np.zeros_like(current_values)

        transport_plans = []
        total_weight = 0.0
        transport_costs = []

        for i, source_task_id in enumerate(source_task_ids):
            source_repr = self.task_representations[source_task_id]

            # Recency weight (more recent = higher weight)
            recency = len(source_task_ids) - i
            weight = config.recency_decay ** (len(source_task_ids) - recency)

            # Compute transport plan based on attention similarity
            # Cost: how different are the attention patterns?
            source_attn_flat = source_repr.attention_weights.reshape(
                source_repr.attention_weights.shape[0], -1
            ).T  # (cols*cols, heads)
            target_attn_flat = current_attention.reshape(
                current_attention.shape[0], -1
            ).T  # (cols*cols, heads)

            cost = cosine_cost_matrix(source_attn_flat, target_attn_flat)

            # Sinkhorn transport
            plan = sinkhorn_transport(
                cost,
                eps=config.sinkhorn_eps,
                iters=config.sinkhorn_iters,
                identity_bonus=config.identity_bonus,
            )
            transport_plans.append(plan)
            transport_costs.append(float(np.sum(cost * plan)))

            # Apply transport to attention weights
            num_heads = current_attention.shape[0]
            num_cols = current_attention.shape[1]

            # Transport attention: for each head, transport the attention matrix
            for h in range(num_heads):
                source_attn_h = source_repr.attention_weights[h]  # (cols, cols)
                # Use diagonal of plan as column-to-column mapping
                col_plan = (
                    plan[:num_cols, :num_cols]
                    if plan.shape[0] >= num_cols
                    else np.eye(num_cols)
                )
                col_plan = col_plan / (col_plan.sum(axis=0, keepdims=True) + 1e-10)
                transported_attn = col_plan.T @ source_attn_h @ col_plan
                transferred_attention[h] += weight * transported_attn

            # Transport projections (simpler: weighted average)
            if config.transfer_query_keys:
                transferred_queries += weight * source_repr.query_projections
                transferred_keys += weight * source_repr.key_projections

            if config.transfer_value_projections:
                transferred_values += weight * source_repr.value_projections

            total_weight += weight

        # Normalize
        if total_weight > 0:
            transferred_attention /= total_weight
            transferred_queries /= total_weight
            transferred_keys /= total_weight
            transferred_values /= total_weight

        # Blend with current (transfer_strength controls mixing)
        strength = config.transfer_strength
        final_attention = (
            1 - strength
        ) * current_attention + strength * transferred_attention
        final_queries = (
            1 - strength
        ) * current_queries + strength * transferred_queries
        final_keys = (1 - strength) * current_keys + strength * transferred_keys
        final_values = (1 - strength) * current_values + strength * transferred_values

        # Diagnostics
        diagnostics = {
            "num_sources": len(source_task_ids),
            "total_weight": total_weight,
            "mean_transport_cost": (
                float(np.mean(transport_costs)) if transport_costs else 0.0
            ),
            "transfer_strength": strength,
        }

        # Record history
        self.transfer_history.append(
            {
                "target_task": target_task_id,
                "source_tasks": source_task_ids,
                **diagnostics,
            }
        )

        return ComposerTransferResult(
            transferred_attention=final_attention,
            transferred_queries=final_queries,
            transferred_keys=final_keys,
            transferred_values=final_values,
            transport_plans=transport_plans,
            source_tasks=source_task_ids,
            transfer_strength=strength,
            diagnostics=diagnostics,
        )

    def compute_attention_regularization(
        self,
        attention_weights: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute regularization terms for attention patterns.

        Args:
            attention_weights: Attention weights (heads, cols, cols)

        Returns:
            Dict with orthogonality_loss and sparsity_loss
        """
        config = self.config
        num_heads = attention_weights.shape[0]

        # Orthogonality: encourage diverse attention patterns across heads
        ortho_loss = 0.0
        if num_heads > 1:
            for h1 in range(num_heads):
                for h2 in range(h1 + 1, num_heads):
                    a1 = attention_weights[h1].ravel()
                    a2 = attention_weights[h2].ravel()
                    # Cosine similarity
                    sim = np.dot(a1, a2) / (
                        np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-8
                    )
                    ortho_loss += sim**2

            ortho_loss /= num_heads * (num_heads - 1) / 2

        # Sparsity: encourage focused attention
        sparsity_loss = 0.0
        for h in range(num_heads):
            attn = attention_weights[h]
            # Entropy-based sparsity (lower = more sparse)
            attn_flat = attn.ravel()
            attn_flat = attn_flat / (attn_flat.sum() + 1e-10)
            entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
            max_entropy = np.log(attn_flat.size)
            sparsity_loss += entropy / max_entropy

        sparsity_loss /= num_heads

        return {
            "orthogonality_loss": config.orthogonality_weight * ortho_loss,
            "sparsity_loss": config.sparsity_weight * sparsity_loss,
            "total_reg_loss": (
                config.orthogonality_weight * ortho_loss
                + config.sparsity_weight * sparsity_loss
            ),
        }

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "task_representations": {
                k: {
                    "task_id": v.task_id,
                    "attention_weights": v.attention_weights.tolist(),
                    "query_projections": v.query_projections.tolist(),
                    "key_projections": v.key_projections.tolist(),
                    "value_projections": v.value_projections.tolist(),
                    "output_projection": v.output_projection.tolist(),
                    "gate_logits": (
                        v.gate_logits.tolist() if v.gate_logits is not None else None
                    ),
                    "metadata": v.metadata,
                }
                for k, v in self.task_representations.items()
            },
            "transfer_history": self.transfer_history,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self.task_representations = {}
        for k, v in state.get("task_representations", {}).items():
            self.task_representations[int(k)] = ComposerRepresentation(
                task_id=v["task_id"],
                attention_weights=np.array(v["attention_weights"]),
                query_projections=np.array(v["query_projections"]),
                key_projections=np.array(v["key_projections"]),
                value_projections=np.array(v["value_projections"]),
                output_projection=np.array(v["output_projection"]),
                gate_logits=(
                    np.array(v["gate_logits"]) if v["gate_logits"] is not None else None
                ),
                metadata=v.get("metadata", {}),
            )
        self.transfer_history = state.get("transfer_history", [])


# ----------------------------
# Within-Column Shell Demotion TransWeave
# ----------------------------


@dataclass
class ShellState:
    """State of shell structure within a column."""

    column_id: int
    task_id: int
    shell_assignments: np.ndarray  # (num_neurons,) - shell index per neuron
    neuron_activities: np.ndarray  # (num_neurons,) - activity levels
    shell_occupancies: np.ndarray  # (num_shells,) - neurons per shell
    demotion_scores: np.ndarray  # (num_neurons,) - demotion pressure per neuron
    promotion_scores: np.ndarray  # (num_neurons,) - promotion potential per neuron


@dataclass
class ShellDemotionResult:
    """Result of shell demotion transport analysis."""

    demotion_candidates: List[Tuple[int, int, int]]  # (neuron_id, from_shell, to_shell)
    promotion_candidates: List[
        Tuple[int, int, int]
    ]  # (neuron_id, from_shell, to_shell)
    transport_plan: np.ndarray  # Transport between shells
    shell_transition_matrix: np.ndarray  # Probability of transitions
    diagnostics: Dict[str, float]


class ShellDemotionTransWeave:
    """
    TransWeave transport for within-column shell demotion.

    Implements the V18/V20 radial shell semantics:
    - Protected center (shell 0): Most stable, rarely demoted
    - Stable inner tiers (shell 1): Moderately stable
    - Disposable outer tiers (shell 2): Task-local, frequently recycled

    Uses Sinkhorn transport to:
    1. Identify neurons whose cross-task activation patterns suggest demotion
    2. Find neurons that should be promoted to more stable shells
    3. Balance shell occupancy across tasks

    Key insight: Transport mass between shell representations across tasks
    indicates how neuron roles should shift.
    """

    def __init__(
        self,
        config: ShellDemotionTransWeaveConfig,
        num_columns: int,
    ):
        self.config = config
        self.num_columns = num_columns

        # Per-column shell state history
        self.column_histories: Dict[int, List[ShellState]] = {
            i: [] for i in range(num_columns)
        }

        # Running activity estimates per column
        self.activity_emas: Dict[int, np.ndarray] = {}

        # Demotion/promotion history
        self.transition_history: List[Dict[str, Any]] = []

    def _get_num_neurons(self) -> int:
        """Get total neurons per column from shell sizes."""
        return sum(self.config.shell_sizes)

    def _shell_boundaries(self) -> List[Tuple[int, int]]:
        """Get (start, end) indices for each shell."""
        boundaries = []
        start = 0
        for size in self.config.shell_sizes:
            boundaries.append((start, start + size))
            start += size
        return boundaries

    def register_shell_state(
        self,
        column_id: int,
        task_id: int,
        shell_assignments: np.ndarray,
        neuron_activities: np.ndarray,
    ) -> None:
        """
        Register shell state for a column after a task.

        Args:
            column_id: Column identifier
            task_id: Task identifier
            shell_assignments: Shell index per neuron
            neuron_activities: Activity level per neuron (e.g., mean activation)
        """
        shell_assignments = np.asarray(shell_assignments, dtype=np.int32)
        neuron_activities = np.asarray(neuron_activities, dtype=np.float64)

        num_shells = self.config.num_shells
        num_neurons = len(shell_assignments)

        # Compute shell occupancies
        shell_occupancies = np.array(
            [np.sum(shell_assignments == s) for s in range(num_shells)]
        )

        # Update activity EMA
        if column_id not in self.activity_emas:
            self.activity_emas[column_id] = neuron_activities.copy()
        else:
            decay = self.config.activity_ema_decay
            self.activity_emas[column_id] = (
                decay * self.activity_emas[column_id] + (1 - decay) * neuron_activities
            )

        # Compute demotion scores (low activity = candidate for demotion)
        activity_ema = self.activity_emas[column_id]
        mean_activity = np.mean(activity_ema) + 1e-8
        demotion_scores = 1.0 - (activity_ema / (2 * mean_activity)).clip(0, 1)

        # Compute promotion scores (high activity = candidate for promotion)
        promotion_scores = (activity_ema / (2 * mean_activity)).clip(0, 1)

        # Apply shell-based modifiers
        for s, (start, end) in enumerate(self._shell_boundaries()):
            shell_mask = shell_assignments == s
            if s == 0:
                # Protected center: reduce demotion, increase promotion threshold
                demotion_scores[shell_mask] *= 0.3
            elif s == num_shells - 1:
                # Outer shell: increase demotion pressure
                demotion_scores[shell_mask] *= 1.5

        state = ShellState(
            column_id=column_id,
            task_id=task_id,
            shell_assignments=shell_assignments,
            neuron_activities=neuron_activities,
            shell_occupancies=shell_occupancies,
            demotion_scores=demotion_scores,
            promotion_scores=promotion_scores,
        )

        self.column_histories[column_id].append(state)

        # Keep limited history
        max_history = self.config.use_last_k_tasks + 2
        if len(self.column_histories[column_id]) > max_history:
            self.column_histories[column_id] = self.column_histories[column_id][
                -max_history:
            ]

    def compute_demotion_transport(
        self,
        column_id: int,
        current_activities: np.ndarray,
        current_assignments: np.ndarray,
    ) -> ShellDemotionResult:
        """
        Compute shell demotion/promotion recommendations using transport.

        Uses Sinkhorn transport between shell activity distributions across
        tasks to identify neurons that should transition between shells.

        Args:
            column_id: Column identifier
            current_activities: Current neuron activities
            current_assignments: Current shell assignments

        Returns:
            ShellDemotionResult with demotion/promotion candidates
        """
        config = self.config
        num_shells = config.num_shells
        num_neurons = self._get_num_neurons()

        current_activities = np.asarray(current_activities, dtype=np.float64)
        current_assignments = np.asarray(current_assignments, dtype=np.int32)

        # Get historical states
        history = self.column_histories.get(column_id, [])

        if len(history) < 1:
            # No history - return empty result
            return ShellDemotionResult(
                demotion_candidates=[],
                promotion_candidates=[],
                transport_plan=np.eye(num_shells),
                shell_transition_matrix=np.eye(num_shells),
                diagnostics={"no_history": True},
            )

        # Build shell activity distributions
        # Current: activity distribution per shell
        current_shell_activities = []
        for s in range(num_shells):
            shell_mask = current_assignments == s
            if np.any(shell_mask):
                shell_act = np.mean(current_activities[shell_mask])
            else:
                shell_act = 0.0
            current_shell_activities.append(shell_act)
        current_shell_activities = np.array(current_shell_activities)

        # Historical: average shell activities
        historical_shell_activities = np.zeros(num_shells)
        for state in history[-config.use_last_k_tasks :]:
            for s in range(num_shells):
                shell_mask = state.shell_assignments == s
                if np.any(shell_mask):
                    historical_shell_activities[s] += np.mean(
                        state.neuron_activities[shell_mask]
                    )
        historical_shell_activities /= len(history[-config.use_last_k_tasks :])

        # Compute transport between shell distributions
        # Cost: difference in activity patterns, scaled appropriately for Sinkhorn
        cost = np.zeros((num_shells, num_shells))
        for i in range(num_shells):
            for j in range(num_shells):
                # Shell distance cost - scaled down to work with eps
                # Adjacent shells should have moderate cost, not prohibitive
                shell_distance = abs(i - j) * 0.15  # Scale factor
                activity_diff = abs(
                    historical_shell_activities[i] - current_shell_activities[j]
                )
                cost[i, j] = shell_distance + activity_diff * 0.5

        # Stability bonus: small preference for staying in place
        # Should be small relative to shell_distance cost
        cost -= np.eye(num_shells) * config.stability_bonus * 0.5

        # Normalize activities as weights
        hist_weights = historical_shell_activities / (
            historical_shell_activities.sum() + 1e-10
        )
        curr_weights = current_shell_activities / (
            current_shell_activities.sum() + 1e-10
        )

        # Sinkhorn transport
        transport_plan = sinkhorn_transport(
            cost,
            source_weights=hist_weights,
            target_weights=curr_weights,
            eps=config.sinkhorn_eps,
            iters=config.sinkhorn_iters,
            identity_bonus=config.identity_bonus,
        )

        # Compute transition matrix (normalized per row)
        shell_transition = transport_plan / (
            transport_plan.sum(axis=1, keepdims=True) + 1e-10
        )

        # Identify demotion candidates
        # A neuron is a demotion candidate if:
        # 1. It's not in the outer shell already
        # 2. Transport mass suggests moving to a higher shell index
        # 3. Its activity is below threshold

        demotion_candidates = []
        promotion_candidates = []

        # Get activity EMA
        if column_id in self.activity_emas:
            activity_ema = self.activity_emas[column_id]
        else:
            activity_ema = current_activities

        for s in range(num_shells - 1):  # Can't demote from outer shell
            shell_mask = current_assignments == s
            neuron_indices = np.where(shell_mask)[0]

            # Transport mass to next shell
            demotion_mass = shell_transition[s, s + 1] if s + 1 < num_shells else 0

            if demotion_mass > config.demotion_threshold:
                # Find lowest-activity neurons in this shell
                shell_activities = activity_ema[neuron_indices]
                sorted_indices = np.argsort(shell_activities)

                # Respect protected center
                if s == 0:
                    protected_count = int(
                        len(neuron_indices) * config.protected_center_fraction
                    )
                    sorted_indices = sorted_indices[protected_count:]

                # Add candidates up to max
                for idx in sorted_indices[: config.max_demotions_per_step]:
                    neuron_id = neuron_indices[idx]
                    demotion_candidates.append((neuron_id, s, s + 1))

        # Identify promotion candidates
        for s in range(1, num_shells):  # Can't promote from center
            shell_mask = current_assignments == s
            neuron_indices = np.where(shell_mask)[0]

            # Transport mass to previous shell
            promotion_mass = shell_transition[s, s - 1] if s > 0 else 0

            if promotion_mass > config.promotion_threshold:
                # Find highest-activity neurons in this shell
                shell_activities = activity_ema[neuron_indices]
                sorted_indices = np.argsort(shell_activities)[::-1]  # Descending

                # Check min occupancy constraint for target shell
                target_occupancy = np.sum(current_assignments == (s - 1))
                target_size = config.shell_sizes[s - 1]

                if target_occupancy < target_size:
                    # Room for promotion
                    for idx in sorted_indices[: config.max_demotions_per_step]:
                        neuron_id = neuron_indices[idx]
                        promotion_candidates.append((neuron_id, s, s - 1))

        # Diagnostics
        diagnostics = {
            "num_demotion_candidates": len(demotion_candidates),
            "num_promotion_candidates": len(promotion_candidates),
            "transport_cost": float(np.sum(cost * transport_plan)),
            "shell_transition_entropy": float(
                -np.sum(shell_transition * np.log(shell_transition + 1e-10))
            ),
        }

        # Record history
        self.transition_history.append(
            {
                "column_id": column_id,
                "demotion_candidates": demotion_candidates,
                "promotion_candidates": promotion_candidates,
                **diagnostics,
            }
        )

        return ShellDemotionResult(
            demotion_candidates=demotion_candidates,
            promotion_candidates=promotion_candidates,
            transport_plan=transport_plan,
            shell_transition_matrix=shell_transition,
            diagnostics=diagnostics,
        )

    def apply_transitions(
        self,
        current_assignments: np.ndarray,
        demotion_result: ShellDemotionResult,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Apply shell transitions to assignments.

        Args:
            current_assignments: Current shell assignments
            demotion_result: Result from compute_demotion_transport

        Returns:
            Tuple of (new_assignments, transition_counts)
        """
        new_assignments = current_assignments.copy()

        demotions_applied = 0
        promotions_applied = 0

        # Apply demotions
        for neuron_id, from_shell, to_shell in demotion_result.demotion_candidates:
            if new_assignments[neuron_id] == from_shell:
                new_assignments[neuron_id] = to_shell
                demotions_applied += 1

        # Apply promotions
        for neuron_id, from_shell, to_shell in demotion_result.promotion_candidates:
            if new_assignments[neuron_id] == from_shell:
                new_assignments[neuron_id] = to_shell
                promotions_applied += 1

        return new_assignments, {
            "demotions_applied": demotions_applied,
            "promotions_applied": promotions_applied,
        }

    def get_shell_statistics(self, column_id: int) -> Dict[str, Any]:
        """Get statistics for a column's shell structure."""
        history = self.column_histories.get(column_id, [])

        if not history:
            return {"no_history": True}

        latest = history[-1]
        num_shells = self.config.num_shells

        stats = {
            "task_id": latest.task_id,
            "shell_occupancies": latest.shell_occupancies.tolist(),
            "mean_demotion_score": float(np.mean(latest.demotion_scores)),
            "mean_promotion_score": float(np.mean(latest.promotion_scores)),
        }

        # Per-shell statistics
        for s in range(num_shells):
            shell_mask = latest.shell_assignments == s
            if np.any(shell_mask):
                stats[f"shell_{s}_mean_activity"] = float(
                    np.mean(latest.neuron_activities[shell_mask])
                )
                stats[f"shell_{s}_occupancy"] = int(np.sum(shell_mask))

        return stats

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "column_histories": {
                col_id: [
                    {
                        "column_id": s.column_id,
                        "task_id": s.task_id,
                        "shell_assignments": s.shell_assignments.tolist(),
                        "neuron_activities": s.neuron_activities.tolist(),
                        "shell_occupancies": s.shell_occupancies.tolist(),
                        "demotion_scores": s.demotion_scores.tolist(),
                        "promotion_scores": s.promotion_scores.tolist(),
                    }
                    for s in states
                ]
                for col_id, states in self.column_histories.items()
            },
            "activity_emas": {k: v.tolist() for k, v in self.activity_emas.items()},
            "transition_history": self.transition_history,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self.column_histories = {}
        for col_id, states in state.get("column_histories", {}).items():
            self.column_histories[int(col_id)] = [
                ShellState(
                    column_id=s["column_id"],
                    task_id=s["task_id"],
                    shell_assignments=np.array(s["shell_assignments"]),
                    neuron_activities=np.array(s["neuron_activities"]),
                    shell_occupancies=np.array(s["shell_occupancies"]),
                    demotion_scores=np.array(s["demotion_scores"]),
                    promotion_scores=np.array(s["promotion_scores"]),
                )
                for s in states
            ]

        self.activity_emas = {
            int(k): np.array(v) for k, v in state.get("activity_emas", {}).items()
        }
        self.transition_history = state.get("transition_history", [])


# ----------------------------
# Unified TransWeave Manager
# ----------------------------


class TransWeaveManager:
    """
    Unified manager for multi-level TransWeave transfer learning.

    Coordinates:
    1. Composer-level transfer (attention patterns, projections)
    2. Within-column shell demotion transfer (neuron shell assignments)

    Provides a single interface for the trainer to interact with.
    """

    def __init__(
        self,
        num_columns: int,
        composer_config: Optional[ComposerTransWeaveConfig] = None,
        shell_config: Optional[ShellDemotionTransWeaveConfig] = None,
    ):
        self.num_columns = num_columns

        # Initialize components
        if composer_config is None:
            composer_config = ComposerTransWeaveConfig()
        self.composer_config = composer_config

        if shell_config is None:
            shell_config = ShellDemotionTransWeaveConfig()
        self.shell_config = shell_config

        self.composer_transweave = ComposerTransWeave(composer_config)
        self.shell_transweave = ShellDemotionTransWeave(shell_config, num_columns)

    def register_task_end(
        self,
        task_id: int,
        # Composer state
        attention_weights: Optional[np.ndarray] = None,
        query_projections: Optional[np.ndarray] = None,
        key_projections: Optional[np.ndarray] = None,
        value_projections: Optional[np.ndarray] = None,
        output_projection: Optional[np.ndarray] = None,
        # Shell state per column
        column_shell_states: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        """
        Register end-of-task state for transfer learning.

        Args:
            task_id: Completed task ID
            attention_weights: Composer attention weights
            query_projections: Composer query projections
            key_projections: Composer key projections
            value_projections: Composer value projections
            output_projection: Composer output projection
            column_shell_states: Dict mapping column_id -> (shell_assignments, neuron_activities)
        """
        # Register composer state
        if (
            self.composer_config.enable
            and attention_weights is not None
            and query_projections is not None
        ):
            self.composer_transweave.register_task(
                task_id=task_id,
                attention_weights=attention_weights,
                query_projections=query_projections,
                key_projections=key_projections,
                value_projections=value_projections,
                output_projection=output_projection,
            )

        # Register shell states per column
        if self.shell_config.enable and column_shell_states is not None:
            for col_id, (assignments, activities) in column_shell_states.items():
                self.shell_transweave.register_shell_state(
                    column_id=col_id,
                    task_id=task_id,
                    shell_assignments=assignments,
                    neuron_activities=activities,
                )

    def compute_transfers(
        self,
        target_task_id: int,
        # Current composer state
        current_attention: Optional[np.ndarray] = None,
        current_queries: Optional[np.ndarray] = None,
        current_keys: Optional[np.ndarray] = None,
        current_values: Optional[np.ndarray] = None,
        # Current shell state per column
        column_current_states: Optional[
            Dict[int, Tuple[np.ndarray, np.ndarray]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Compute all transfers for a new task.

        Args:
            target_task_id: New task ID
            current_attention: Current composer attention
            current_queries: Current query projections
            current_keys: Current key projections
            current_values: Current value projections
            column_current_states: Dict mapping column_id -> (assignments, activities)

        Returns:
            Dict with composer_transfer and shell_demotions
        """
        results = {}

        # Composer transfer
        if self.composer_config.enable and current_attention is not None:
            composer_result = self.composer_transweave.compute_transfer(
                target_task_id=target_task_id,
                current_attention=current_attention,
                current_queries=current_queries,
                current_keys=current_keys,
                current_values=current_values,
            )
            results["composer_transfer"] = composer_result

        # Shell demotion per column
        if self.shell_config.enable and column_current_states is not None:
            shell_results = {}
            for col_id, (assignments, activities) in column_current_states.items():
                shell_result = self.shell_transweave.compute_demotion_transport(
                    column_id=col_id,
                    current_activities=activities,
                    current_assignments=assignments,
                )
                shell_results[col_id] = shell_result
            results["shell_demotions"] = shell_results

        return results

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all TransWeave components."""
        stats = {}

        # Composer stats
        if self.composer_config.enable:
            stats["composer_tasks_registered"] = len(
                self.composer_transweave.task_representations
            )
            if self.composer_transweave.transfer_history:
                latest = self.composer_transweave.transfer_history[-1]
                stats["composer_latest_transfer_cost"] = latest.get(
                    "mean_transport_cost", 0
                )

        # Shell stats (aggregate across columns)
        if self.shell_config.enable:
            total_demotions = 0
            total_promotions = 0
            for entry in self.shell_transweave.transition_history:
                total_demotions += entry.get("num_demotion_candidates", 0)
                total_promotions += entry.get("num_promotion_candidates", 0)

            stats["total_demotion_candidates"] = total_demotions
            stats["total_promotion_candidates"] = total_promotions

        return stats

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "composer": self.composer_transweave.save_state(),
            "shell": self.shell_transweave.save_state(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        if "composer" in state:
            self.composer_transweave.load_state(state["composer"])
        if "shell" in state:
            self.shell_transweave.load_state(state["shell"])
