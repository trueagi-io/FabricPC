"""
Support Selection System for Continual Learning.

Manages which memory columns are active for each task, using:
- Support banks for storing successful patterns
- Demotion banks for tracking demoted columns
- Hybrid selector policy combining multiple strategies
- Trust controller for adjusting policy influence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence, Any
import numpy as np
import jax.numpy as jnp


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
    ) -> Tuple[int, ...]:
        """
        Select non-shared support columns for a task.

        Args:
            task_id: Current task ID
            support_bank: Bank of previous support patterns
            demotion_bank: Bank of demoted columns
            current_features: Feature vector for similarity matching
            exploration_rate: Probability of random exploration

        Returns:
            Tuple of selected non-shared column indices
        """
        nonshared_pool = list(range(self.num_shared, self.num_columns))

        # Remove demoted columns
        if demotion_bank is not None:
            demoted = set(demotion_bank.get_demoted_columns(task_id))
            nonshared_pool = [c for c in nonshared_pool if c not in demoted]

        if len(nonshared_pool) < self.topk_nonshared:
            # Not enough columns, use all available
            return tuple(nonshared_pool)

        # Compute selection scores
        scores = np.zeros(len(nonshared_pool))

        # 1. Prior from column history
        for i, col in enumerate(nonshared_pool):
            scores[i] += self.column_scores[col]

        # 2. Task-specific preferences
        if task_id in self.task_preferences:
            for i, col in enumerate(nonshared_pool):
                scores[i] += self.task_preferences[task_id][col]

        # 3. Support bank similarity (if available)
        if support_bank is not None and len(support_bank) > 0:
            bank_stats = support_bank.get_column_statistics()
            for i, col in enumerate(nonshared_pool):
                if col in bank_stats:
                    scores[i] += 0.5 * bank_stats[col]["mean_accuracy"]

        # 4. Random exploration
        if np.random.random() < exploration_rate:
            # Select randomly
            selected_indices = np.random.choice(
                len(nonshared_pool), size=self.topk_nonshared, replace=False
            )
        else:
            # Select top-k by score
            selected_indices = np.argsort(scores)[-self.topk_nonshared :]

        selected_cols = tuple(nonshared_pool[i] for i in selected_indices)
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
    and trust controller.
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
        # Decide whether to use policy or baseline
        use_policy = self.trust_controller.should_use_policy()

        if use_policy:
            selected = self.selector_policy.select_support(
                task_id,
                support_bank=self.support_bank,
                demotion_bank=self.demotion_bank,
                current_features=features,
            )
        else:
            # Baseline: use current selection or random
            if len(self.current_state.active_nonshared) == self.topk_nonshared:
                selected = self.current_state.active_nonshared
            else:
                nonshared_pool = list(range(self.num_shared, self.num_columns))
                indices = np.random.choice(
                    len(nonshared_pool), size=self.topk_nonshared, replace=False
                )
                selected = tuple(nonshared_pool[i] for i in indices)

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

        return self.current_state

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
