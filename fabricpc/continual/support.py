"""
Support Selection System for Continual Learning.

Manages which memory columns are active for each task, using:
- Support banks for storing successful patterns
- Demotion banks for tracking demoted columns
- Hybrid selector policy combining multiple strategies
- Trust controller for adjusting policy influence
- Experience replay buffer for preventing catastrophic forgetting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence, Any, Iterator, TYPE_CHECKING
import numpy as np
import jax.numpy as jnp

if TYPE_CHECKING:
    from .config import SupportConfig


@dataclass
class SupportState:
    """
    State of the support selection system.

    Tracks which columns are active, their history, and selection scores.
    """

    # Current active column indices (non-shared)
    active_nonshared: Tuple[int, ...]

    # All active indices (shared + non-shared)
    active_all: Tuple[int, ...]

    # Number of columns
    num_columns: int
    num_shared: int
    topk_nonshared: int

    # Per-column statistics
    task_usage_count: Dict[int, int] = field(default_factory=dict)
    column_scores: Optional[np.ndarray] = None
    last_support_score_table: List[Dict[str, Any]] = field(default_factory=list)

    # Causal guidance state
    causal_fingerprint_sim: Optional[np.ndarray] = None
    causal_confidence: Optional[np.ndarray] = None
    causal_fingerprint_mean: Optional[np.ndarray] = None
    causal_effective_scale: float = 0.0
    causal_mix_gate: float = 0.0

    def active_indices(self) -> Tuple[int, ...]:
        """Return all active column indices."""
        return self.active_all

    def active_mask(self, batch_size: int = 1) -> jnp.ndarray:
        """
        Return a binary mask for active columns.

        Args:
            batch_size: Batch size for the mask

        Returns:
            Array of shape (batch_size, num_columns) with 1s for active columns
        """
        mask = np.zeros((batch_size, self.num_columns), dtype=np.float32)
        for idx in self.active_all:
            mask[:, idx] = 1.0
        return jnp.array(mask)

    def shared_indices(self) -> Tuple[int, ...]:
        """Return shared column indices (always active)."""
        return tuple(range(self.num_shared))


def create_initial_support_state(
    num_columns: int,
    num_shared: int,
    topk_nonshared: int,
    initial_nonshared: Optional[Sequence[int]] = None,
) -> SupportState:
    """
    Create initial support state.

    Args:
        num_columns: Total number of columns
        num_shared: Number of always-active shared columns
        topk_nonshared: Number of non-shared columns to select
        initial_nonshared: Initial non-shared column indices (optional)

    Returns:
        Initial SupportState
    """
    shared = tuple(range(num_shared))

    if initial_nonshared is None:
        # Select first topk non-shared columns
        nonshared_pool = list(range(num_shared, num_columns))
        initial_nonshared = tuple(nonshared_pool[:topk_nonshared])
    else:
        initial_nonshared = tuple(initial_nonshared)

    all_active = shared + initial_nonshared

    return SupportState(
        active_nonshared=initial_nonshared,
        active_all=all_active,
        num_columns=num_columns,
        num_shared=num_shared,
        topk_nonshared=topk_nonshared,
        column_scores=np.zeros(num_columns),
    )


@dataclass
class SupportBankRow:
    """A single row in the support bank."""

    task_id: int
    support_cols: Tuple[int, ...]
    accuracy: float
    loss: float
    timestamp: int
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SupportBank:
    """
    Bank of support patterns for replay-based continual learning.

    Stores successful support column configurations and their outcomes
    for use in selecting supports for new tasks.
    """

    def __init__(
        self,
        max_rows: int = 1000,
        feature_dim: int = 64,
    ):
        self.max_rows = max_rows
        self.feature_dim = feature_dim
        self.rows: List[SupportBankRow] = []
        self._timestamp = 0

    def add(
        self,
        task_id: int,
        support_cols: Sequence[int],
        accuracy: float,
        loss: float,
        features: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a support pattern to the bank."""
        row = SupportBankRow(
            task_id=task_id,
            support_cols=tuple(support_cols),
            accuracy=accuracy,
            loss=loss,
            timestamp=self._timestamp,
            features=features,
            metadata=metadata or {},
        )
        self.rows.append(row)
        self._timestamp += 1

        # Trim if over capacity
        if len(self.rows) > self.max_rows:
            # Remove oldest rows
            self.rows = self.rows[-self.max_rows :]

    def query(
        self,
        task_id: Optional[int] = None,
        min_accuracy: float = 0.0,
        topk: int = 10,
    ) -> List[SupportBankRow]:
        """
        Query the bank for relevant support patterns.

        Args:
            task_id: Filter by task ID (optional)
            min_accuracy: Minimum accuracy threshold
            topk: Maximum number of results

        Returns:
            List of matching SupportBankRow objects
        """
        results = []
        for row in self.rows:
            if task_id is not None and row.task_id != task_id:
                continue
            if row.accuracy < min_accuracy:
                continue
            results.append(row)

        # Sort by accuracy descending
        results.sort(key=lambda r: r.accuracy, reverse=True)
        return results[:topk]

    def get_column_statistics(self) -> Dict[int, Dict[str, float]]:
        """
        Compute per-column statistics from bank history.

        Returns:
            Dict mapping column index to statistics dict
        """
        stats: Dict[int, Dict[str, List[float]]] = {}

        for row in self.rows:
            for col in row.support_cols:
                if col not in stats:
                    stats[col] = {"accuracies": [], "losses": []}
                stats[col]["accuracies"].append(row.accuracy)
                stats[col]["losses"].append(row.loss)

        result = {}
        for col, data in stats.items():
            result[col] = {
                "mean_accuracy": np.mean(data["accuracies"]),
                "mean_loss": np.mean(data["losses"]),
                "count": len(data["accuracies"]),
            }
        return result

    def get_mean_accuracy_by_column(self, num_columns: int) -> np.ndarray:
        """
        Compute mean accuracy per column as a dense vector.

        This provides a batched form for support ranking logic.
        Columns with no history receive 0.
        """
        if num_columns <= 0:
            return np.zeros((0,), dtype=np.float64)
        if not self.rows:
            return np.zeros((num_columns,), dtype=np.float64)

        flat_cols = []
        flat_accs = []
        for row in self.rows:
            if not row.support_cols:
                continue
            cols = np.asarray(row.support_cols, dtype=np.int32)
            valid = (cols >= 0) & (cols < num_columns)
            cols = cols[valid]
            if cols.size == 0:
                continue
            flat_cols.append(cols)
            flat_accs.append(np.full(cols.shape, row.accuracy, dtype=np.float64))

        if not flat_cols:
            return np.zeros((num_columns,), dtype=np.float64)

        all_cols = np.concatenate(flat_cols)
        all_accs = np.concatenate(flat_accs)
        acc_sum = np.bincount(all_cols, weights=all_accs, minlength=num_columns)
        count = np.bincount(all_cols, minlength=num_columns)
        return acc_sum / np.clip(count, 1, None)

    def __len__(self) -> int:
        return len(self.rows)


class DemotionBank:
    """
    Bank tracking demoted columns that performed poorly.

    Used to avoid re-selecting columns that have been found to
    interfere with learning.
    """

    def __init__(self, max_rows: int = 500):
        self.max_rows = max_rows
        self.demotions: List[Dict[str, Any]] = []

    def add_demotion(
        self,
        task_id: int,
        column_idx: int,
        reason: str,
        score_before: float,
        score_after: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a column demotion."""
        self.demotions.append(
            {
                "task_id": task_id,
                "column_idx": column_idx,
                "reason": reason,
                "score_before": score_before,
                "score_after": score_after,
                "metadata": metadata or {},
            }
        )

        if len(self.demotions) > self.max_rows:
            self.demotions = self.demotions[-self.max_rows :]

    def is_demoted(self, column_idx: int, task_id: Optional[int] = None) -> bool:
        """Check if a column has been demoted."""
        for d in self.demotions:
            if d["column_idx"] == column_idx:
                if task_id is None or d["task_id"] == task_id:
                    return True
        return False

    def get_demoted_columns(self, task_id: Optional[int] = None) -> List[int]:
        """Get list of demoted column indices."""
        demoted = set()
        for d in self.demotions:
            if task_id is None or d["task_id"] == task_id:
                demoted.add(d["column_idx"])
        return list(demoted)


class HybridSelectorPolicy:
    """
    Hybrid policy for support column selection.

    Combines multiple strategies:
    1. Prior knowledge from support bank
    2. Learned selector model
    3. KNN-based similarity matching
    4. Random exploration
    """

    def __init__(
        self,
        num_columns: int,
        num_shared: int,
        topk_nonshared: int,
        config: Optional["SupportConfig"] = None,
    ):
        self.num_columns = num_columns
        self.num_shared = num_shared
        self.topk_nonshared = topk_nonshared
        self.config = config

        # Column scores (learned from history)
        self.column_scores = np.zeros(num_columns)

        # Task-specific preferences
        self.task_preferences: Dict[int, np.ndarray] = {}

        # History for learning
        self.history: List[Dict[str, Any]] = []

    def select_support(
        self,
        task_id: int,
        support_bank: Optional[SupportBank] = None,
        demotion_bank: Optional[DemotionBank] = None,
        current_features: Optional[np.ndarray] = None,
        exploration_rate: float = 0.1,
        diversify_columns: bool = True,
        diversification_penalty: float = 2.0,
    ) -> Tuple[int, ...]:
        """
        Select non-shared support columns for a task.

        Args:
            task_id: Current task ID
            support_bank: Bank of previous support patterns
            demotion_bank: Bank of demoted columns
            current_features: Feature vector for similarity matching
            exploration_rate: Probability of random exploration
            diversify_columns: If True, penalize columns used by other tasks
            diversification_penalty: Penalty scale for used columns

        Returns:
            Tuple of selected non-shared column indices
        """
        nonshared_pool = np.arange(self.num_shared, self.num_columns, dtype=np.int32)

        # Remove demoted columns
        if demotion_bank is not None:
            demoted = np.asarray(
                demotion_bank.get_demoted_columns(task_id), dtype=np.int32
            )
            if demoted.size > 0:
                nonshared_pool = nonshared_pool[~np.isin(nonshared_pool, demoted)]

        if nonshared_pool.size < self.topk_nonshared:
            # Not enough columns, use all available
            return tuple(int(c) for c in nonshared_pool.tolist())

        # Compute selection scores
        scores = self.column_scores[nonshared_pool].astype(np.float64, copy=True)

        # 2. Task-specific preferences
        if task_id in self.task_preferences:
            scores += self.task_preferences[task_id][nonshared_pool]

        # 3. Support bank similarity (if available)
        if support_bank is not None and len(support_bank) > 0:
            scores += (
                0.5
                * support_bank.get_mean_accuracy_by_column(self.num_columns)[
                    nonshared_pool
                ]
            )

        # 4. Column diversification penalty for continual learning
        # Penalize columns that were used by OTHER tasks to encourage fresh columns
        if diversify_columns and support_bank is not None:
            for row in support_bank.rows:
                if row.task_id != task_id:
                    # Penalize columns used by other tasks
                    for col in row.support_cols:
                        if col in nonshared_pool:
                            pool_idx = np.where(nonshared_pool == col)[0]
                            if pool_idx.size > 0:
                                scores[pool_idx[0]] -= diversification_penalty

        # 5. Random exploration
        if np.random.random() < exploration_rate:
            # Select randomly
            selected_indices = np.random.choice(
                nonshared_pool.size, size=self.topk_nonshared, replace=False
            )
        else:
            # Select top-k by score
            selected_indices = np.argpartition(scores, -self.topk_nonshared)[
                -self.topk_nonshared :
            ]
            selected_indices = selected_indices[
                np.argsort(scores[selected_indices])[::-1]
            ]

        selected_cols = tuple(
            int(col) for col in nonshared_pool[selected_indices].tolist()
        )
        return selected_cols

    def update_from_outcome(
        self,
        task_id: int,
        support_cols: Sequence[int],
        accuracy: float,
        loss: float,
        learning_rate: float = 0.1,
    ):
        """
        Update policy based on observed outcome.

        Args:
            task_id: Task ID
            support_cols: Selected support columns
            accuracy: Achieved accuracy
            loss: Achieved loss
            learning_rate: Learning rate for score updates
        """
        # Update column scores
        reward = accuracy - 0.5  # Center around 0.5

        for col in support_cols:
            self.column_scores[col] += learning_rate * reward

        # Update task preferences
        if task_id not in self.task_preferences:
            self.task_preferences[task_id] = np.zeros(self.num_columns)

        for col in support_cols:
            self.task_preferences[task_id][col] += learning_rate * reward

        # Record history
        self.history.append(
            {
                "task_id": task_id,
                "support_cols": tuple(support_cols),
                "accuracy": accuracy,
                "loss": loss,
            }
        )

    def get_column_rankings(self) -> List[Tuple[int, float]]:
        """Get columns ranked by score."""
        rankings = [(i, self.column_scores[i]) for i in range(self.num_columns)]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class TrustController:
    """
    Controller for adjusting trust in the selector policy.

    Monitors policy performance and adjusts its influence on
    support selection decisions.
    """

    def __init__(
        self,
        initial_trust: float = 0.5,
        min_trust: float = 0.1,
        max_trust: float = 0.9,
        learning_rate: float = 0.05,
    ):
        self.trust = initial_trust
        self.min_trust = min_trust
        self.max_trust = max_trust
        self.learning_rate = learning_rate

        self.history: List[Dict[str, float]] = []

    def update(self, policy_accuracy: float, baseline_accuracy: float):
        """
        Update trust based on policy vs baseline performance.

        Args:
            policy_accuracy: Accuracy achieved with policy selection
            baseline_accuracy: Accuracy achieved with baseline selection
        """
        improvement = policy_accuracy - baseline_accuracy

        # Increase trust if policy outperforms baseline
        if improvement > 0:
            self.trust = min(
                self.max_trust, self.trust + self.learning_rate * improvement
            )
        else:
            self.trust = max(
                self.min_trust, self.trust + self.learning_rate * improvement
            )

        self.history.append(
            {
                "policy_accuracy": policy_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "improvement": improvement,
                "trust": self.trust,
            }
        )

    def get_trust(self) -> float:
        """Get current trust level."""
        return self.trust

    def should_use_policy(self) -> bool:
        """Determine whether to use policy selection."""
        return np.random.random() < self.trust


class SupportManager:
    """
    High-level manager for the support selection system.

    Coordinates the support bank, demotion bank, selector policy,
    trust controller, and causal components.
    """

    def __init__(
        self,
        num_columns: int,
        num_shared: int,
        topk_nonshared: int,
        config: Optional["SupportConfig"] = None,
        num_tasks: int = 5,
    ):
        self.num_columns = num_columns
        self.num_shared = num_shared
        self.topk_nonshared = topk_nonshared
        self.config = config
        self.num_tasks = num_tasks

        # Initialize components
        self.support_bank = SupportBank()
        self.demotion_bank = DemotionBank()
        self.selector_policy = HybridSelectorPolicy(
            num_columns, num_shared, topk_nonshared, config
        )
        self.trust_controller = TrustController()

        # Current state
        self.current_state = create_initial_support_state(
            num_columns, num_shared, topk_nonshared
        )

        # Causal components (lazy import to avoid circular dependencies)
        from .causal import (
            CausalFingerprintBank,
            CausalContributionPredictor,
            CausalSelectorTrustController,
            CausalSupportFeatureBuilder,
        )

        self.causal_bank = CausalFingerprintBank.create(num_columns, num_tasks)
        self.causal_predictor = (
            CausalContributionPredictor(config=config) if config else None
        )
        self.causal_trust = (
            CausalSelectorTrustController(config=config) if config else None
        )
        self.causal_feature_builder = CausalSupportFeatureBuilder.from_config(
            num_columns=num_columns,
            num_tasks=num_tasks,
            topk_nonshared=topk_nonshared,
        )

    def select_support_for_task(
        self,
        task_id: int,
        features: Optional[np.ndarray] = None,
    ) -> SupportState:
        """
        Select support columns for a task.

        Args:
            task_id: Task ID
            features: Optional feature vector for similarity matching

        Returns:
            Updated SupportState with selected columns
        """
        # For continual learning, use sequential column assignment first
        # to maximize column isolation between tasks
        num_nonshared = self.num_columns - self.num_shared
        columns_per_task = self.topk_nonshared

        # Determine if we can assign fresh columns to this task
        # Tasks 0, 1, 2, ... get columns [2,3], [4,5], [6,7], etc.
        # until we run out of fresh columns
        max_isolated_tasks = num_nonshared // columns_per_task

        if task_id < max_isolated_tasks:
            # Assign fresh non-shared columns sequentially
            start_col = self.num_shared + task_id * columns_per_task
            selected = tuple(range(start_col, start_col + columns_per_task))
        else:
            # Out of fresh columns, use policy or baseline with recycling
            use_policy = self.trust_controller.should_use_policy()

            if use_policy:
                selected = self.selector_policy.select_support(
                    task_id,
                    support_bank=self.support_bank,
                    demotion_bank=self.demotion_bank,
                    current_features=features,
                )
            else:
                # Baseline: cycle through available columns
                cycle_idx = task_id % max_isolated_tasks
                start_col = self.num_shared + cycle_idx * columns_per_task
                selected = tuple(range(start_col, start_col + columns_per_task))

        # Apply causal guidance if available and configured
        if self.config is not None and self.config.causal_max_effective_scale > 0:
            selected = self._apply_causal_guidance(task_id, list(selected))

        # Update state
        shared = tuple(range(self.num_shared))
        all_active = shared + selected

        self.current_state = SupportState(
            active_nonshared=selected,
            active_all=all_active,
            num_columns=self.num_columns,
            num_shared=self.num_shared,
            topk_nonshared=self.topk_nonshared,
            task_usage_count=self.current_state.task_usage_count,
            column_scores=self.current_state.column_scores,
        )

        # Update usage count
        if task_id not in self.current_state.task_usage_count:
            self.current_state.task_usage_count[task_id] = 0
        self.current_state.task_usage_count[task_id] += 1

        # Set causal guidance on state for tracking
        self.set_causal_guidance_on_state(task_id)

        return self.current_state

    def _apply_causal_guidance(
        self,
        task_id: int,
        initial_selected: List[int],
    ) -> Tuple[int, ...]:
        """
        Apply causal guidance to refine column selection.

        Uses the CausalContributionPredictor to score candidate columns
        and potentially swap in better ones.

        Args:
            task_id: Current task ID
            initial_selected: Initially selected non-shared columns

        Returns:
            Refined selection as tuple
        """
        if self.causal_predictor is None or not self.causal_predictor.trained:
            return tuple(initial_selected)

        if self.causal_trust is None:
            return tuple(initial_selected)

        # Get current trust state
        diag = self.causal_trust.last_diag
        mix_gate = diag.get("mix_gate", 0.0)

        # If mix_gate is too low, skip causal guidance
        if mix_gate < 0.05:
            return tuple(initial_selected)

        # Get fingerprint data for feature building
        fp_mean = None
        fp_conf = None
        if self.causal_bank is not None:
            fp_mean = self.causal_bank.mean_gain()
            fp_conf = self.causal_bank.column_confidence(
                self.config.causal_similarity_conf_target if self.config else 8.0
            )

        # Build features for all candidate columns
        all_nonshared = list(range(self.num_shared, self.num_columns))
        unchosen = [c for c in all_nonshared if c not in initial_selected]

        if not unchosen:
            return tuple(initial_selected)

        # Simple similarity functions
        def struct_sim(i: int, j: int) -> float:
            return 1.0 if i == j else 0.0

        def causal_sim(i: int, j: int) -> float:
            if fp_mean is None:
                return 0.0
            if i >= fp_mean.shape[0] or j >= fp_mean.shape[0]:
                return 0.0
            vi = fp_mean[i]
            vj = fp_mean[j]
            ni = np.linalg.norm(vi)
            nj = np.linalg.norm(vj)
            if ni < 1e-6 or nj < 1e-6:
                return 0.0
            return float(np.dot(vi, vj) / (ni * nj))

        # Build placeholder certificate arrays
        cert_general = np.zeros(self.num_columns)
        cert_specific = np.zeros(self.num_columns)
        cert_demotion = np.zeros(self.num_columns)
        cert_saturation = np.zeros(self.num_columns)
        novelty = np.zeros(self.num_columns)
        saturation = np.zeros(self.num_columns)
        recent_penalty = np.zeros(self.num_columns)
        reserve_bonus = np.zeros(self.num_columns)
        base_z = np.zeros(self.num_columns)

        # Predict scores for unchosen columns
        candidate_features = self.causal_feature_builder.build_features_batch(
            indices=unchosen,
            roles="challenger",
            chosen_sets=[tuple(initial_selected)] * len(unchosen),
            base_z=base_z,
            cert_general=cert_general,
            cert_specific=cert_specific,
            cert_demotion=cert_demotion,
            cert_saturation=cert_saturation,
            novelty=novelty,
            saturation=saturation,
            recent_penalty=recent_penalty,
            reserve_bonus=reserve_bonus,
            fingerprint_mean=fp_mean,
            fingerprint_confidence=fp_conf,
            current_task_id=task_id,
        )
        candidate_pred = self.causal_predictor.predict(candidate_features)
        candidate_order = np.argsort(candidate_pred)[::-1]

        # Consider swapping in top candidates if they score well
        # Use mix_gate to control how aggressive we are
        selected = list(initial_selected)
        swap_threshold = 0.01 * mix_gate  # Higher mix_gate = lower threshold

        chosen_contexts = [tuple(c for c in selected if c != col) for col in selected]
        chosen_features = self.causal_feature_builder.build_features_batch(
            indices=selected,
            roles="reuse",
            chosen_sets=chosen_contexts,
            base_z=base_z,
            cert_general=cert_general,
            cert_specific=cert_specific,
            cert_demotion=cert_demotion,
            cert_saturation=cert_saturation,
            novelty=novelty,
            saturation=saturation,
            recent_penalty=recent_penalty,
            reserve_bonus=reserve_bonus,
            fingerprint_mean=fp_mean,
            fingerprint_confidence=fp_conf,
            current_task_id=task_id,
        )
        chosen_pred = self.causal_predictor.predict(chosen_features)
        worst_idx = int(np.argmin(chosen_pred))
        worst_score = float(chosen_pred[worst_idx])

        for order_idx in candidate_order[:2]:  # Consider top 2 candidates
            cand_col = unchosen[int(order_idx)]
            cand_score = float(candidate_pred[int(order_idx)])
            if cand_score > swap_threshold:
                # Swap if candidate is significantly better
                if cand_score - worst_score > swap_threshold:
                    selected[worst_idx] = cand_col
                    break  # Only one swap per selection

        return tuple(selected)

    def record_outcome(
        self,
        task_id: int,
        accuracy: float,
        loss: float,
        features: Optional[np.ndarray] = None,
    ):
        """
        Record the outcome of training with current support selection.

        Args:
            task_id: Task ID
            accuracy: Achieved accuracy
            loss: Achieved loss
            features: Optional feature vector
        """
        # Add to support bank
        self.support_bank.add(
            task_id=task_id,
            support_cols=self.current_state.active_nonshared,
            accuracy=accuracy,
            loss=loss,
            features=features,
        )

        # Update selector policy
        self.selector_policy.update_from_outcome(
            task_id=task_id,
            support_cols=self.current_state.active_nonshared,
            accuracy=accuracy,
            loss=loss,
        )

    def demote_column(
        self,
        task_id: int,
        column_idx: int,
        reason: str,
        score_before: float,
        score_after: float,
    ):
        """Record a column demotion."""
        self.demotion_bank.add_demotion(
            task_id=task_id,
            column_idx=column_idx,
            reason=reason,
            score_before=score_before,
            score_after=score_after,
        )

    def get_active_mask(self, batch_size: int = 1) -> jnp.ndarray:
        """Get binary mask for active columns."""
        return self.current_state.active_mask(batch_size)

    def save_state(self) -> Dict[str, Any]:
        """Serialize manager state for checkpointing."""
        return {
            "current_state": {
                "active_nonshared": self.current_state.active_nonshared,
                "active_all": self.current_state.active_all,
                "num_columns": self.current_state.num_columns,
                "num_shared": self.current_state.num_shared,
                "topk_nonshared": self.current_state.topk_nonshared,
                "task_usage_count": dict(self.current_state.task_usage_count),
                "column_scores": (
                    self.current_state.column_scores.tolist()
                    if self.current_state.column_scores is not None
                    else None
                ),
            },
            "support_bank_rows": [
                {
                    "task_id": r.task_id,
                    "support_cols": r.support_cols,
                    "accuracy": r.accuracy,
                    "loss": r.loss,
                    "timestamp": r.timestamp,
                }
                for r in self.support_bank.rows
            ],
            "demotion_bank": self.demotion_bank.demotions,
            "selector_policy": {
                "column_scores": self.selector_policy.column_scores.tolist(),
                "task_preferences": {
                    str(k): v.tolist()
                    for k, v in self.selector_policy.task_preferences.items()
                },
            },
            "trust_controller": {
                "trust": self.trust_controller.trust,
                "history": self.trust_controller.history,
            },
            # Causal components
            "causal_bank": (
                self.causal_bank.save_state() if self.causal_bank else None
            ),
            "causal_predictor": (
                self.causal_predictor.save_state() if self.causal_predictor else None
            ),
            "causal_trust": (
                self.causal_trust.save_state() if self.causal_trust else None
            ),
        }

    def load_state(self, state: Dict[str, Any]):
        """Restore manager state from checkpoint."""
        cs = state["current_state"]
        self.current_state = SupportState(
            active_nonshared=tuple(cs["active_nonshared"]),
            active_all=tuple(cs["active_all"]),
            num_columns=cs["num_columns"],
            num_shared=cs["num_shared"],
            topk_nonshared=cs["topk_nonshared"],
            task_usage_count=cs["task_usage_count"],
            column_scores=(
                np.array(cs["column_scores"]) if cs["column_scores"] else None
            ),
        )

        # Restore support bank
        self.support_bank.rows = []
        for r in state["support_bank_rows"]:
            self.support_bank.rows.append(
                SupportBankRow(
                    task_id=r["task_id"],
                    support_cols=tuple(r["support_cols"]),
                    accuracy=r["accuracy"],
                    loss=r["loss"],
                    timestamp=r["timestamp"],
                )
            )

        # Restore demotion bank
        self.demotion_bank.demotions = state["demotion_bank"]

        # Restore selector policy
        sp = state["selector_policy"]
        self.selector_policy.column_scores = np.array(sp["column_scores"])
        self.selector_policy.task_preferences = {
            int(k): np.array(v) for k, v in sp["task_preferences"].items()
        }

        # Restore trust controller
        tc = state["trust_controller"]
        self.trust_controller.trust = tc["trust"]
        self.trust_controller.history = tc["history"]

        # Restore causal components
        if "causal_bank" in state and self.causal_bank is not None:
            self.causal_bank.load_state(state["causal_bank"])
        if "causal_predictor" in state and self.causal_predictor is not None:
            self.causal_predictor.load_state(state["causal_predictor"])
        if "causal_trust" in state and self.causal_trust is not None:
            self.causal_trust.load_state(state["causal_trust"])

    def update_causal_from_audit(
        self,
        audit_rows: List[Dict[str, Any]],
        current_task_id: int,
        old_task: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Update causal bank and predictor from audit results.

        Args:
            audit_rows: List of audit/swap result dictionaries
            current_task_id: Current task ID
            old_task: Previous task ID (if any)

        Returns:
            Training metrics dictionary
        """
        if self.causal_bank is None:
            return {"causal_selector_corr": 0.0, "causal_selector_mae": 0.0}

        # Update fingerprint bank
        self.causal_bank.update_from_support_rows(audit_rows, current_task_id, old_task)

        # Train predictor if we have enough examples
        metrics = {"causal_selector_corr": 0.0, "causal_selector_mae": 0.0}
        if self.causal_predictor is not None:
            metrics = self.causal_predictor.train_if_ready()

        return metrics

    def get_causal_guidance(
        self,
        task_id: int,
        effective_internal_trust: float = 0.5,
        recent_agreement: float = 0.0,
        recent_rows: int = 0,
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float, float
    ]:
        """
        Get causal guidance for support selection.

        Args:
            task_id: Current task ID
            effective_internal_trust: Trust score from support state
            recent_agreement: Recent prediction agreement rate
            recent_rows: Number of recent audit rows

        Returns:
            Tuple of (fingerprint_sim, confidence, mean_gain, effective_scale, mix_gate)
        """
        if self.causal_bank is None:
            return None, None, None, 0.0, 0.0

        # Get fingerprint similarity and confidence
        config = self.config
        target_count = config.causal_similarity_conf_target if config else 8.0
        fp_sim, fp_conf = self.causal_bank.similarity_matrix(target_count)
        fp_mean = self.causal_bank.mean_gain()

        # Compute trust gates
        effective_scale = 0.0
        mix_gate = 0.0
        if self.causal_trust is not None and self.causal_predictor is not None:
            diag = self.causal_trust.compute(
                self.causal_predictor,
                effective_internal_trust,
                recent_agreement,
                recent_rows,
            )
            effective_scale = diag.get("effective_scale", 0.0)
            mix_gate = diag.get("mix_gate", 0.0)

        return fp_sim, fp_conf, fp_mean, effective_scale, mix_gate

    def set_causal_guidance_on_state(
        self,
        task_id: int,
        effective_internal_trust: float = 0.5,
        recent_agreement: float = 0.0,
        recent_rows: int = 0,
    ) -> None:
        """
        Set causal guidance on current support state.

        Args:
            task_id: Current task ID
            effective_internal_trust: Trust score from support state
            recent_agreement: Recent prediction agreement rate
            recent_rows: Number of recent audit rows
        """
        fp_sim, fp_conf, fp_mean, eff_scale, mix_gate = self.get_causal_guidance(
            task_id, effective_internal_trust, recent_agreement, recent_rows
        )

        self.current_state.causal_fingerprint_sim = fp_sim
        self.current_state.causal_confidence = fp_conf
        self.current_state.causal_fingerprint_mean = fp_mean
        self.current_state.causal_effective_scale = eff_scale
        self.current_state.causal_mix_gate = mix_gate

    def add_causal_examples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add examples to causal predictor.

        Args:
            X: Feature array (N, feature_dim)
            y: Target array (N,)
            w: Weight array (N,)
            meta: Optional metadata list
        """
        if self.causal_predictor is not None:
            self.causal_predictor.add_examples(X, y, w, meta)


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Stores samples from previous tasks to be replayed during training
    on new tasks, preventing catastrophic forgetting.

    Uses reservoir sampling to maintain a fixed-size buffer with
    representative samples from all seen tasks.
    """

    def __init__(
        self,
        max_samples_per_task: int = 500,
        max_total_samples: int = 5000,
    ):
        """
        Initialize replay buffer.

        Args:
            max_samples_per_task: Maximum samples to store per task
            max_total_samples: Maximum total samples in buffer
        """
        self.max_samples_per_task = max_samples_per_task
        self.max_total_samples = max_total_samples

        # Storage: task_id -> (images, labels)
        self._buffers: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._counts: Dict[int, int] = {}

    def add_task_samples(
        self,
        task_id: int,
        images: np.ndarray,
        labels: np.ndarray,
        replace: bool = False,
    ):
        """
        Add samples from a task to the buffer.

        Uses reservoir sampling if more samples than max_samples_per_task.

        Args:
            task_id: Task identifier
            images: Array of images (N, ...)
            labels: Array of labels (N, ...) - can be one-hot or integer
            replace: Whether to replace existing samples for this task
        """
        n_samples = len(images)

        if replace or task_id not in self._buffers:
            # Initialize or replace buffer for this task
            if n_samples <= self.max_samples_per_task:
                # Store all samples
                self._buffers[task_id] = (images.copy(), labels.copy())
                self._counts[task_id] = n_samples
            else:
                # Reservoir sampling
                indices = np.random.choice(
                    n_samples, size=self.max_samples_per_task, replace=False
                )
                self._buffers[task_id] = (
                    images[indices].copy(),
                    labels[indices].copy(),
                )
                self._counts[task_id] = self.max_samples_per_task
        else:
            # Append to existing (with reservoir sampling if needed)
            existing_imgs, existing_labels = self._buffers[task_id]
            combined_imgs = np.concatenate([existing_imgs, images], axis=0)
            combined_labels = np.concatenate([existing_labels, labels], axis=0)

            if len(combined_imgs) <= self.max_samples_per_task:
                self._buffers[task_id] = (combined_imgs, combined_labels)
                self._counts[task_id] = len(combined_imgs)
            else:
                # Reservoir sampling
                indices = np.random.choice(
                    len(combined_imgs), size=self.max_samples_per_task, replace=False
                )
                self._buffers[task_id] = (
                    combined_imgs[indices].copy(),
                    combined_labels[indices].copy(),
                )
                self._counts[task_id] = self.max_samples_per_task

        # Enforce total limit by reducing oldest tasks
        self._enforce_total_limit()

    def _enforce_total_limit(self):
        """Reduce buffer sizes to stay within total limit."""
        total = sum(self._counts.values())
        if total <= self.max_total_samples:
            return

        # Reduce proportionally from all tasks
        reduction_factor = self.max_total_samples / total

        for task_id in list(self._buffers.keys()):
            imgs, labels = self._buffers[task_id]
            new_size = max(1, int(len(imgs) * reduction_factor))
            if new_size < len(imgs):
                indices = np.random.choice(len(imgs), size=new_size, replace=False)
                self._buffers[task_id] = (imgs[indices], labels[indices])
                self._counts[task_id] = new_size

    def sample(
        self,
        batch_size: int,
        exclude_task: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample a batch from the replay buffer.

        Args:
            batch_size: Number of samples to return
            exclude_task: Task ID to exclude (e.g., current task)

        Returns:
            Tuple of (images, labels) or None if buffer is empty
        """
        task_ids = np.array(
            [task_id for task_id in self._buffers.keys() if task_id != exclude_task],
            dtype=np.int32,
        )
        if task_ids.size == 0:
            return None

        counts = np.array(
            [self._counts[task_id] for task_id in task_ids], dtype=np.int32
        )
        total_count = int(np.sum(counts))
        if total_count == 0:
            return None

        actual_batch_size = min(batch_size, total_count)
        global_indices = np.random.choice(
            total_count, size=actual_batch_size, replace=False
        )
        cumulative = np.cumsum(counts)
        task_positions = np.searchsorted(cumulative, global_indices, side="right")
        starts = cumulative - counts
        local_indices = global_indices - starts[task_positions]

        sampled_imgs = []
        sampled_labels = []
        for task_pos in np.unique(task_positions):
            mask = task_positions == task_pos
            task_id = int(task_ids[task_pos])
            imgs, labels = self._buffers[task_id]
            task_local_indices = local_indices[mask]
            sampled_imgs.append(imgs[task_local_indices])
            sampled_labels.append(labels[task_local_indices])

        return np.concatenate(sampled_imgs, axis=0), np.concatenate(
            sampled_labels, axis=0
        )

    def sample_batches(
        self,
        num_batches: int,
        batch_size: int,
        exclude_task: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample many replay batches at once.

        Produces a dense batch matrix so interleaved training can prepare an
        epoch's replay samples in one vectorized step instead of sampling once
        per batch from Python.

        Args:
            num_batches: Number of replay batches to sample
            batch_size: Samples per replay batch
            exclude_task: Task ID to exclude

        Returns:
            Tuple of (images, labels) with leading shape
            (num_batches, actual_batch_size, ...) or None if buffer is empty
        """
        if num_batches <= 0 or batch_size <= 0:
            return None

        task_ids = np.array(
            [task_id for task_id in self._buffers.keys() if task_id != exclude_task],
            dtype=np.int32,
        )
        if task_ids.size == 0:
            return None

        counts = np.array(
            [self._counts[task_id] for task_id in task_ids], dtype=np.int32
        )
        total_count = int(np.sum(counts))
        if total_count == 0:
            return None

        actual_batch_size = min(batch_size, total_count)
        random_scores = np.random.random((num_batches, total_count))
        global_indices = np.argpartition(
            random_scores, kth=actual_batch_size - 1, axis=1
        )[:, :actual_batch_size]
        chosen_scores = np.take_along_axis(random_scores, global_indices, axis=1)
        row_order = np.argsort(chosen_scores, axis=1)
        global_indices = np.take_along_axis(global_indices, row_order, axis=1)

        cumulative = np.cumsum(counts)
        flat_indices = global_indices.reshape(-1)
        task_positions = np.searchsorted(cumulative, flat_indices, side="right")
        starts = cumulative - counts
        local_indices = flat_indices - starts[task_positions]

        first_task_id = int(task_ids[0])
        example_images, example_labels = self._buffers[first_task_id]
        images_shape = (num_batches, actual_batch_size) + example_images.shape[1:]
        labels_shape = (num_batches, actual_batch_size) + example_labels.shape[1:]
        sampled_images = np.empty(images_shape, dtype=example_images.dtype)
        sampled_labels = np.empty(labels_shape, dtype=example_labels.dtype)

        batch_rows = np.repeat(np.arange(num_batches), actual_batch_size)
        batch_cols = np.tile(np.arange(actual_batch_size), num_batches)
        for task_pos in np.unique(task_positions):
            mask = task_positions == task_pos
            task_id = int(task_ids[int(task_pos)])
            task_images, task_labels = self._buffers[task_id]
            sampled_images[batch_rows[mask], batch_cols[mask]] = task_images[
                local_indices[mask]
            ]
            sampled_labels[batch_rows[mask], batch_cols[mask]] = task_labels[
                local_indices[mask]
            ]

        return sampled_images, sampled_labels

    def sample_by_task(
        self,
        samples_per_task: int,
        exclude_task: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample equal numbers from each task.

        Args:
            samples_per_task: Samples to take from each task
            exclude_task: Task ID to exclude

        Returns:
            Tuple of (images, labels) or None if buffer is empty
        """
        available_tasks = [t for t in self._buffers.keys() if t != exclude_task]

        if not available_tasks:
            return None

        sampled_imgs = []
        sampled_labels = []

        for task_id in available_tasks:
            imgs, labels = self._buffers[task_id]
            n = min(samples_per_task, len(imgs))
            if n > 0:
                indices = np.random.choice(len(imgs), size=n, replace=False)
                sampled_imgs.append(imgs[indices])
                sampled_labels.append(labels[indices])

        if not sampled_imgs:
            return None

        return np.concatenate(sampled_imgs, axis=0), np.concatenate(
            sampled_labels, axis=0
        )

    def get_task_ids(self) -> List[int]:
        """Return list of task IDs in buffer."""
        return list(self._buffers.keys())

    def get_task_count(self, task_id: int) -> int:
        """Return sample count for a task."""
        return self._counts.get(task_id, 0)

    def total_samples(self) -> int:
        """Return total samples in buffer."""
        return sum(self._counts.values())

    def __len__(self) -> int:
        """Return number of tasks in buffer."""
        return len(self._buffers)

    def clear(self):
        """Clear all samples from buffer."""
        self._buffers.clear()
        self._counts.clear()

    def save_state(self) -> Dict[str, Any]:
        """Save buffer state for checkpointing."""
        return {
            "buffers": {
                str(k): {
                    "images": v[0].tolist(),
                    "labels": v[1].tolist(),
                }
                for k, v in self._buffers.items()
            },
            "counts": {str(k): v for k, v in self._counts.items()},
            "max_samples_per_task": self.max_samples_per_task,
            "max_total_samples": self.max_total_samples,
        }

    def load_state(self, state: Dict[str, Any]):
        """Load buffer state from checkpoint."""
        self.max_samples_per_task = state.get(
            "max_samples_per_task", self.max_samples_per_task
        )
        self.max_total_samples = state.get("max_total_samples", self.max_total_samples)

        self._buffers = {}
        for k, v in state.get("buffers", {}).items():
            task_id = int(k)
            self._buffers[task_id] = (
                np.array(v["images"]),
                np.array(v["labels"]),
            )

        self._counts = {int(k): v for k, v in state.get("counts", {}).items()}
