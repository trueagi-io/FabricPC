"""
Causal coding components for FabricPC continual learning.

Ported from mnist_audit_guided_generality_v20_2b notebook (PyTorch)
to JAX/NumPy-based implementation.

Components:
- CausalFingerprintBank: Tracks per-column, per-task gain statistics
- CausalContributionPredictor: Ridge regression for column value prediction
- CausalSelectorTrustController: Multi-gate confidence blending
- SB Distribution Clarity Penalty: Kurtosis + Sinkhorn transport penalties
- Routing Bonus System: Multi-factor column scoring
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from .config import SupportConfig

# ----------------------------
# Utility functions
# ----------------------------


def weighted_corr(
    x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
) -> float:
    """Compute weighted correlation coefficient."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return 0.0
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, dtype=np.float64).ravel()
        w = np.clip(w, 1e-6, None)
    w = w / np.sum(w).clip(1e-6)
    mx = np.sum(w * x)
    my = np.sum(w * y)
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx < 1e-8 or vy < 1e-8:
        return 0.0
    cov = np.sum(w * (x - mx) * (y - my))
    corr = cov / np.sqrt(vx * vy + 1e-8)
    return float(corr)


def weighted_mae(
    pred: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
) -> float:
    """Compute weighted mean absolute error."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if pred.size == 0:
        return 0.0
    if w is None:
        return float(np.mean(np.abs(pred - y)))
    w = np.asarray(w, dtype=np.float64).ravel()
    w = np.clip(w, 1e-6, None)
    w = w / np.sum(w).clip(1e-6)
    return float(np.sum(w * np.abs(pred - y)))


def zscore_nonshared(
    values: np.ndarray, num_shared: int, eps: float = 1e-6
) -> np.ndarray:
    """Compute z-scores for non-shared columns only."""
    values = np.asarray(values, dtype=np.float64)
    nonshared = values[num_shared:]
    if nonshared.size == 0:
        return values.copy()
    mean = np.mean(nonshared)
    std = np.std(nonshared).clip(eps)
    result = values.copy()
    result[num_shared:] = (nonshared - mean) / std
    return result


def normalize_vectors(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """L2-normalize vectors along specified axis."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


# ----------------------------
# SB Distribution Clarity
# ----------------------------


def compute_excess_kurtosis(x: np.ndarray) -> float:
    """
    Compute excess kurtosis of array.

    Excess kurtosis = kurtosis - 3 (so Gaussian has excess kurtosis of 0).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-8:
        return 0.0
    z = (x - mean) / std
    return float(np.mean(z**4) - 3.0)


def compute_multimodal_gap(x: np.ndarray) -> float:
    """
    Compute multimodal gap metric.

    Measures separation between lower and upper halves of distribution.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 4:
        return 0.0
    median = np.median(x)
    lower = x[x <= median]
    upper = x[x > median]
    if lower.size == 0 or upper.size == 0:
        return 0.0
    lower_mean = np.mean(lower)
    upper_mean = np.mean(upper)
    std = np.std(x)
    if std < 1e-8:
        return 0.0
    return float(abs(upper_mean - lower_mean) / std)


def sinkhorn_transport_cost(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.35,
    iters: int = 8,
    mode: str = "huber",
) -> float:
    """
    Compute Sinkhorn optimal transport cost between two distributions.

    Args:
        x: Source distribution samples (n,) or (n, d)
        y: Target distribution samples (m,) or (m, d)
        eps: Regularization parameter
        iters: Number of Sinkhorn iterations
        mode: Cost function ("huber", "l1", "l2")

    Returns:
        Transport cost
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n, m = x.shape[0], y.shape[0]
    if n == 0 or m == 0:
        return 0.0

    # Compute cost matrix
    diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # (n, m, d)
    if mode == "l2":
        C = np.sum(diff**2, axis=-1)
    elif mode == "l1":
        C = np.sum(np.abs(diff), axis=-1)
    else:  # huber
        delta = 1.0
        abs_diff = np.abs(diff)
        huber = np.where(
            abs_diff <= delta,
            0.5 * diff**2,
            delta * (abs_diff - 0.5 * delta),
        )
        C = np.sum(huber, axis=-1)

    # Sinkhorn algorithm
    K = np.exp(-C / eps)
    a = np.ones(n) / n
    b = np.ones(m) / m

    u = np.ones(n)
    for _ in range(iters):
        v = b / (K.T @ u + 1e-10)
        u = a / (K @ v + 1e-10)

    # Compute transport cost
    P = np.outer(u, v) * K
    cost = np.sum(P * C)
    return float(cost)


@dataclass
class SBClarityPenaltyResult:
    """Result of SB clarity penalty computation."""

    penalty: float
    mean_kurtosis: float
    mean_transport: float
    active_cols: int
    mean_alpha: float
    column_details: List[Dict[str, float]]


def compute_sb_clarity_penalty(
    column_outputs: Dict[int, np.ndarray],
    config: SupportConfig,
    gaussian_reference: Optional[np.ndarray] = None,
) -> SBClarityPenaltyResult:
    """
    Compute SB distribution clarity penalty.

    For each active column, penalizes:
    1. Excess kurtosis beyond threshold
    2. Multimodal gap beyond threshold
    3. Sinkhorn transport cost from Gaussian reference

    Args:
        column_outputs: Dict mapping column_idx -> output array
        config: SupportConfig with SB parameters
        gaussian_reference: Optional pre-generated Gaussian samples

    Returns:
        SBClarityPenaltyResult with penalty and diagnostics
    """
    if not config.sb_enable or not column_outputs:
        return SBClarityPenaltyResult(
            penalty=0.0,
            mean_kurtosis=0.0,
            mean_transport=0.0,
            active_cols=0,
            mean_alpha=0.0,
            column_details=[],
        )

    details = []
    total_penalty = 0.0
    total_kurtosis = 0.0
    total_transport = 0.0
    total_alpha = 0.0

    for col_idx, outputs in column_outputs.items():
        outputs = np.asarray(outputs, dtype=np.float64).ravel()
        if outputs.size < 4:
            continue

        # Compute kurtosis and multimodal gap
        kurtosis = compute_excess_kurtosis(outputs)
        multimodal = compute_multimodal_gap(outputs)

        # Non-Gaussianity score
        kurt_excess = max(0.0, abs(kurtosis) - config.sb_kurtosis_threshold)
        multi_excess = max(0.0, multimodal - config.sb_multimodal_threshold)
        non_gauss = kurt_excess + multi_excess

        # Generate Gaussian reference if not provided
        if gaussian_reference is None:
            ref = np.random.randn(outputs.size) * config.sb_gaussian_ref_scale
        else:
            ref = gaussian_reference

        # Compute transport cost
        transport = sinkhorn_transport_cost(
            outputs,
            ref,
            eps=config.sb_sinkhorn_eps,
            iters=config.sb_sinkhorn_iters,
            mode=config.sb_cost_mode,
        )

        # Compute alpha scaling
        alpha = config.sb_alpha_scale * (non_gauss + transport)
        alpha = min(alpha, config.sb_alpha_max)

        # Column penalty
        col_penalty = alpha * transport

        details.append(
            {
                "column_idx": col_idx,
                "kurtosis": kurtosis,
                "multimodal": multimodal,
                "transport": transport,
                "alpha": alpha,
                "penalty": col_penalty,
            }
        )

        total_penalty += col_penalty
        total_kurtosis += abs(kurtosis)
        total_transport += transport
        total_alpha += alpha

    n_cols = len(details)
    if n_cols == 0:
        return SBClarityPenaltyResult(
            penalty=0.0,
            mean_kurtosis=0.0,
            mean_transport=0.0,
            active_cols=0,
            mean_alpha=0.0,
            column_details=[],
        )

    return SBClarityPenaltyResult(
        penalty=total_penalty * config.sb_clarity_weight,
        mean_kurtosis=total_kurtosis / n_cols,
        mean_transport=total_transport / n_cols,
        active_cols=n_cols,
        mean_alpha=total_alpha / n_cols,
        column_details=details,
    )


# ----------------------------
# Routing Bonus System
# ----------------------------


@dataclass
class RoutingCertificates:
    """Certificate arrays for routing bonus computation."""

    shared_mass: np.ndarray  # (num_columns,)
    specific_load: np.ndarray  # (num_columns,)
    demotion_pressure: np.ndarray  # (num_columns,)
    novelty_bonus: np.ndarray  # (num_columns,)
    redundancy_penalty: np.ndarray  # (num_columns,)
    stability_score: np.ndarray  # (num_columns,)


@dataclass
class RoutingBonusResult:
    """Result of routing bonus computation."""

    scores: np.ndarray  # (num_columns,)
    certificates: RoutingCertificates
    z_scores: Dict[str, np.ndarray]


def compute_routing_bonus(
    certificates: RoutingCertificates,
    config: SupportConfig,
    nonshared_indices: List[int],
    num_shared: int,
) -> RoutingBonusResult:
    """
    Compute routing bonus scores for column selection.

    Combines z-scored certificate terms:
    - shared_mass: General usefulness (positive)
    - specific_load: Task-specific contribution (negative)
    - demotion_pressure: Negative signal from demotions (negative)
    - novelty_bonus: Diversity encouragement (positive)
    - redundancy_penalty: Overlap penalty (negative)
    - stability_score: Consistency bonus (positive)

    Returns combined scores for ranking columns.
    """
    if not config.route_enable or len(nonshared_indices) == 0:
        n = certificates.shared_mass.shape[0]
        return RoutingBonusResult(
            scores=np.zeros(n),
            certificates=certificates,
            z_scores={},
        )

    # Z-score each certificate for non-shared columns
    z_shared = zscore_nonshared(certificates.shared_mass, num_shared)
    z_specific = zscore_nonshared(certificates.specific_load, num_shared)
    z_demotion = zscore_nonshared(certificates.demotion_pressure, num_shared)
    z_novelty = zscore_nonshared(certificates.novelty_bonus, num_shared)
    z_redundancy = zscore_nonshared(certificates.redundancy_penalty, num_shared)
    z_stability = zscore_nonshared(certificates.stability_score, num_shared)

    # Combine with configured scales
    scores = (
        config.route_cert_mix_scale * z_shared
        - config.route_cert_mix_scale * z_specific
        - config.route_demotion_scale * z_demotion
        + config.route_novelty_scale * z_novelty
        - config.route_diversity_scale * z_redundancy
        + config.route_stability_scale * z_stability
    )

    z_scores = {
        "z_shared": z_shared,
        "z_specific": z_specific,
        "z_demotion": z_demotion,
        "z_novelty": z_novelty,
        "z_redundancy": z_redundancy,
        "z_stability": z_stability,
    }

    return RoutingBonusResult(
        scores=scores,
        certificates=certificates,
        z_scores=z_scores,
    )


# ----------------------------
# Causal Fingerprint Bank
# ----------------------------


@dataclass
class CausalFingerprintBank:
    """
    Bank of per-column, per-task gain statistics for causal similarity.

    Tracks loss improvements when columns are swapped in/out during audits.
    """

    num_columns: int
    num_tasks: int
    gain_sum: np.ndarray  # shape: (num_columns, num_tasks)
    gain_count: np.ndarray  # shape: (num_columns, num_tasks)

    @classmethod
    def create(cls, num_columns: int, num_tasks: int) -> "CausalFingerprintBank":
        """Factory method to create a new bank."""
        return cls(
            num_columns=num_columns,
            num_tasks=num_tasks,
            gain_sum=np.zeros((num_columns, num_tasks), dtype=np.float64),
            gain_count=np.zeros((num_columns, num_tasks), dtype=np.float64),
        )

    def update_from_support_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        current_task_id: int,
        old_task: Optional[int] = None,
    ) -> None:
        """Update gain statistics from audit/swap rows."""
        for row in rows:
            c = int(row.get("swap_in", -1))
            if c < 0 or c >= self.num_columns:
                continue

            # Current task gain
            cur_gain = float(
                row.get("chosen_current_loss", 0.0) - row.get("alt_current_loss", 0.0)
            )
            if current_task_id < self.num_tasks:
                self.gain_sum[c, current_task_id] += cur_gain
                self.gain_count[c, current_task_id] += 1.0

            # Old task gain (if applicable)
            if old_task is not None and old_task < self.num_tasks:
                old_gain = float(
                    row.get("chosen_old_loss", 0.0) - row.get("alt_old_loss", 0.0)
                )
                self.gain_sum[c, old_task] += old_gain
                self.gain_count[c, old_task] += 1.0

    def mean_gain(self) -> np.ndarray:
        """Returns (num_columns, num_tasks) mean gain matrix."""
        return self.gain_sum / np.clip(self.gain_count, 1.0, None)

    def column_confidence(self, target_count: float) -> np.ndarray:
        """Returns (num_columns,) confidence scores."""
        total = np.sum(self.gain_count, axis=1)
        return np.sqrt(total / max(1e-6, float(target_count))).clip(0.0, 1.0)

    def similarity_matrix(self, target_count: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (similarity_matrix, confidence_vector).

        Similarity is cosine similarity of mean gain vectors.
        """
        conf = self.column_confidence(target_count)
        vec = self.mean_gain()
        vec = normalize_vectors(vec, axis=-1)
        sim = vec @ vec.T
        np.fill_diagonal(sim, 1.0)
        return sim, conf

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "num_columns": self.num_columns,
            "num_tasks": self.num_tasks,
            "gain_sum": self.gain_sum.copy(),
            "gain_count": self.gain_count.copy(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self.num_columns = state["num_columns"]
        self.num_tasks = state["num_tasks"]
        self.gain_sum = np.asarray(state["gain_sum"], dtype=np.float64)
        self.gain_count = np.asarray(state["gain_count"], dtype=np.float64)


# ----------------------------
# Causal Contribution Predictor
# ----------------------------


@dataclass
class CausalContributionPredictor:
    """
    Ridge regression predictor for column contribution scores.

    Input: Feature vector per column (default 21-dim)
    Output: Predicted contribution score
    """

    config: SupportConfig
    X_buffer: List[np.ndarray] = field(default_factory=list)
    y_buffer: List[np.ndarray] = field(default_factory=list)
    w_buffer: List[np.ndarray] = field(default_factory=list)
    meta_buffer: List[Dict[str, Any]] = field(default_factory=list)

    # Learned parameters
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    beta_: Optional[np.ndarray] = None
    trained: bool = False
    input_dim: Optional[int] = None

    # Diagnostics
    last_corr: float = 0.0
    last_mae: float = 0.0

    def add_examples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        meta: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add training examples."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)

        if X.size == 0:
            return

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)
        if w.ndim == 0:
            w = w.reshape(1)

        self.X_buffer.append(X)
        self.y_buffer.append(y.ravel())
        self.w_buffer.append(w.ravel())

        if meta is not None:
            self.meta_buffer.extend(meta)

        if self.input_dim is None:
            self.input_dim = X.shape[-1]

    def num_examples(self) -> int:
        """Return total example count."""
        return sum(x.shape[0] for x in self.X_buffer)

    def _stack(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack all buffers into single arrays."""
        X = np.vstack(self.X_buffer).astype(np.float64)
        y = np.concatenate(self.y_buffer).astype(np.float64)
        w = np.concatenate(self.w_buffer).astype(np.float64)
        w = np.clip(w, 1e-4, None)
        return X, y, w

    def train_if_ready(self) -> Dict[str, float]:
        """
        Train predictor if sufficient examples.
        Returns diagnostic metrics.
        """
        n = self.num_examples()
        if self.input_dim is None or n < self.config.causal_min_examples:
            self.last_corr = 0.0
            self.last_mae = 0.0
            return {"causal_selector_corr": 0.0, "causal_selector_mae": 0.0}

        X, y, w = self._stack()

        # Standardize features
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0).clip(1e-5)
        Xn = (X - self.mean_) / self.std_

        # Add intercept term
        ones = np.ones((Xn.shape[0], 1))
        Xaug = np.hstack([ones, Xn])

        # Weighted least squares with ridge regularization
        sw = np.sqrt(w).reshape(-1, 1)
        Xw = Xaug * sw
        yw = y.reshape(-1, 1) * sw

        d = Xaug.shape[1]
        reg = np.eye(d) * self.config.causal_ridge_lambda
        reg[0, 0] = 0.0  # Don't regularize intercept

        # Solve normal equations
        A = Xw.T @ Xw + reg
        b = Xw.T @ yw
        self.beta_ = np.linalg.solve(A, b).ravel()
        self.trained = True

        # Compute diagnostics
        pred = self.predict(X)
        self.last_corr = weighted_corr(pred, y, w)
        self.last_mae = weighted_mae(pred, y, w)

        return {
            "causal_selector_corr": float(self.last_corr),
            "causal_selector_mae": float(self.last_mae),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict contribution scores."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if (
            not self.trained
            or self.beta_ is None
            or self.mean_ is None
            or self.std_ is None
        ):
            return np.zeros(X.shape[0])

        Xn = (X - self.mean_) / self.std_
        ones = np.ones((Xn.shape[0], 1))
        Xaug = np.hstack([ones, Xn])
        return Xaug @ self.beta_

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "X_buffer": [x.copy() for x in self.X_buffer],
            "y_buffer": [y.copy() for y in self.y_buffer],
            "w_buffer": [w.copy() for w in self.w_buffer],
            "meta_buffer": copy.deepcopy(self.meta_buffer),
            "mean_": None if self.mean_ is None else self.mean_.copy(),
            "std_": None if self.std_ is None else self.std_.copy(),
            "beta_": None if self.beta_ is None else self.beta_.copy(),
            "trained": self.trained,
            "input_dim": self.input_dim,
            "last_corr": self.last_corr,
            "last_mae": self.last_mae,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self.X_buffer = [np.asarray(x) for x in state.get("X_buffer", [])]
        self.y_buffer = [np.asarray(y) for y in state.get("y_buffer", [])]
        self.w_buffer = [np.asarray(w) for w in state.get("w_buffer", [])]
        self.meta_buffer = copy.deepcopy(state.get("meta_buffer", []))
        self.mean_ = None if state.get("mean_") is None else np.asarray(state["mean_"])
        self.std_ = None if state.get("std_") is None else np.asarray(state["std_"])
        self.beta_ = None if state.get("beta_") is None else np.asarray(state["beta_"])
        self.trained = state.get("trained", False)
        self.input_dim = state.get("input_dim")
        self.last_corr = state.get("last_corr", 0.0)
        self.last_mae = state.get("last_mae", 0.0)


# ----------------------------
# Causal Selector Trust Controller
# ----------------------------


@dataclass
class AgreementTracker:
    """
    Tracks agreement between predictor predictions and actual outcomes.

    Stores recent predictions and computes agreement rate when outcomes are known.
    """

    max_history: int = 100
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    recent_agreements: List[float] = field(default_factory=list)
    _pred_task_ids: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _pred_column_ids: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _pred_scores: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _pred_role_codes: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _pred_size: int = field(default=0, init=False, repr=False)
    _pred_pos: int = field(default=0, init=False, repr=False)
    _out_task_ids: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _out_column_ids: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _out_gains: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _out_size: int = field(default=0, init=False, repr=False)
    _out_pos: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._allocate_buffers()

    def _capacity(self) -> int:
        return max(1, self.max_history * 2)

    def _allocate_buffers(self) -> None:
        cap = self._capacity()
        self._pred_task_ids = np.zeros(cap, dtype=np.int32)
        self._pred_column_ids = np.zeros(cap, dtype=np.int32)
        self._pred_scores = np.zeros(cap, dtype=np.float64)
        self._pred_role_codes = np.zeros(cap, dtype=np.int8)
        self._out_task_ids = np.zeros(cap, dtype=np.int32)
        self._out_column_ids = np.zeros(cap, dtype=np.int32)
        self._out_gains = np.zeros(cap, dtype=np.float64)
        self._pred_size = 0
        self._pred_pos = 0
        self._out_size = 0
        self._out_pos = 0

    @staticmethod
    def _role_to_code(role: str) -> int:
        return {"reuse": 0, "diverse": 1, "challenger": 2}.get(role, -1)

    @staticmethod
    def _code_to_role(code: int) -> str:
        return {0: "reuse", 1: "diverse", 2: "challenger"}.get(code, "unknown")

    @staticmethod
    def _ordered_ring(arr: np.ndarray, size: int, pos: int) -> np.ndarray:
        if size == 0:
            return arr[:0]
        if size < arr.shape[0]:
            return arr[:size].copy()
        return np.concatenate([arr[pos:], arr[:pos]], axis=0)

    def _prediction_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._pred_task_ids is not None
        assert self._pred_column_ids is not None
        assert self._pred_scores is not None
        assert self._pred_role_codes is not None
        return (
            self._ordered_ring(self._pred_task_ids, self._pred_size, self._pred_pos),
            self._ordered_ring(self._pred_column_ids, self._pred_size, self._pred_pos),
            self._ordered_ring(self._pred_scores, self._pred_size, self._pred_pos),
            self._ordered_ring(self._pred_role_codes, self._pred_size, self._pred_pos),
        )

    def _outcome_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._out_task_ids is not None
        assert self._out_column_ids is not None
        assert self._out_gains is not None
        return (
            self._ordered_ring(self._out_task_ids, self._out_size, self._out_pos),
            self._ordered_ring(self._out_column_ids, self._out_size, self._out_pos),
            self._ordered_ring(self._out_gains, self._out_size, self._out_pos),
        )

    def record_prediction(
        self,
        task_id: int,
        column_idx: int,
        predicted_score: float,
        role: str = "challenger",
    ) -> None:
        """Record a predictor's score for a column."""
        assert self._pred_task_ids is not None
        assert self._pred_column_ids is not None
        assert self._pred_scores is not None
        assert self._pred_role_codes is not None
        self.predictions.append(
            {
                "task_id": task_id,
                "column_idx": column_idx,
                "predicted_score": predicted_score,
                "role": role,
            }
        )
        if len(self.predictions) > self.max_history:
            self.predictions = self.predictions[-self.max_history :]
        self._pred_task_ids[self._pred_pos] = task_id
        self._pred_column_ids[self._pred_pos] = column_idx
        self._pred_scores[self._pred_pos] = predicted_score
        self._pred_role_codes[self._pred_pos] = self._role_to_code(role)
        self._pred_pos = (self._pred_pos + 1) % self._capacity()
        self._pred_size = min(self._pred_size + 1, self._capacity())
        if self._pred_size > self.max_history:
            task_ids, col_ids, scores, role_codes = self._prediction_arrays()
            keep = self.max_history
            self._pred_task_ids[:keep] = task_ids[-keep:]
            self._pred_column_ids[:keep] = col_ids[-keep:]
            self._pred_scores[:keep] = scores[-keep:]
            self._pred_role_codes[:keep] = role_codes[-keep:]
            self._pred_size = keep
            self._pred_pos = keep % self._capacity()

    def record_outcome(
        self,
        task_id: int,
        column_idx: int,
        actual_gain: float,
    ) -> None:
        """Record actual outcome for a column from audit."""
        assert self._out_task_ids is not None
        assert self._out_column_ids is not None
        assert self._out_gains is not None
        self.outcomes.append(
            {
                "task_id": task_id,
                "column_idx": column_idx,
                "actual_gain": actual_gain,
            }
        )
        if len(self.outcomes) > self.max_history:
            self.outcomes = self.outcomes[-self.max_history :]
        self._out_task_ids[self._out_pos] = task_id
        self._out_column_ids[self._out_pos] = column_idx
        self._out_gains[self._out_pos] = actual_gain
        self._out_pos = (self._out_pos + 1) % self._capacity()
        self._out_size = min(self._out_size + 1, self._capacity())
        if self._out_size > self.max_history:
            task_ids, col_ids, gains = self._outcome_arrays()
            keep = self.max_history
            self._out_task_ids[:keep] = task_ids[-keep:]
            self._out_column_ids[:keep] = col_ids[-keep:]
            self._out_gains[:keep] = gains[-keep:]
            self._out_size = keep
            self._out_pos = keep % self._capacity()

    def compute_recent_agreement(self, window: int = 20) -> Tuple[float, int]:
        """
        Compute agreement rate over recent predictions/outcomes.

        Agreement is measured as correlation between predicted scores
        and actual gains for matched (task_id, column_idx) pairs.

        Returns:
            Tuple of (agreement_rate, num_matched_pairs)
        """
        pred_task_ids, pred_col_ids, pred_scores, _ = self._prediction_arrays()
        out_task_ids, out_col_ids, out_gains = self._outcome_arrays()

        if pred_scores.size == 0 or out_gains.size == 0:
            return 0.0, 0

        pred_task_ids = pred_task_ids[-self.max_history :]
        pred_col_ids = pred_col_ids[-self.max_history :]
        pred_scores = pred_scores[-self.max_history :]
        out_task_ids = out_task_ids[-self.max_history :]
        out_col_ids = out_col_ids[-self.max_history :]
        out_gains = out_gains[-self.max_history :]

        pred_keys = np.rec.fromarrays([pred_task_ids, pred_col_ids], names="task,col")
        out_keys = np.rec.fromarrays([out_task_ids, out_col_ids], names="task,col")
        order = np.argsort(out_keys, kind="mergesort")
        sorted_out_keys = out_keys[order]
        positions = np.searchsorted(sorted_out_keys, pred_keys)
        valid = positions < sorted_out_keys.shape[0]
        valid &= (
            sorted_out_keys[np.clip(positions, 0, sorted_out_keys.shape[0] - 1)]
            == pred_keys
        )

        if not np.any(valid):
            return 0.0, 0

        matched_pred = pred_scores[valid][-window:]
        matched_actual = out_gains[order][positions[valid]][-window:]

        if len(matched_pred) < 4:
            return 0.0, len(matched_pred)

        # Compute correlation as agreement measure
        pred_arr = np.asarray(matched_pred, dtype=np.float64)
        actual_arr = np.asarray(matched_actual, dtype=np.float64)

        # Normalize to avoid scale issues
        pred_std = np.std(pred_arr)
        actual_std = np.std(actual_arr)

        if pred_std < 1e-8 or actual_std < 1e-8:
            # No variance - check if signs match
            pred_signs = np.sign(pred_arr)
            actual_signs = np.sign(actual_arr)
            agreement = np.mean(pred_signs == actual_signs)
            return float(agreement), len(pred_arr)

        # Correlation coefficient
        corr = np.corrcoef(pred_arr, actual_arr)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # Convert correlation to agreement (0 to 1 scale)
        # corr of 1 = perfect agreement, corr of -1 = perfect disagreement
        agreement = (corr + 1.0) / 2.0

        self.recent_agreements.append(agreement)
        if len(self.recent_agreements) > self.max_history:
            self.recent_agreements = self.recent_agreements[-self.max_history :]

        return float(agreement), len(pred_arr)

    def get_smoothed_agreement(self, window: int = 10) -> float:
        """Get smoothed agreement over recent computations."""
        if not self.recent_agreements:
            return 0.0
        recent = self.recent_agreements[-window:]
        return float(np.mean(recent))

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        pred_task_ids, pred_col_ids, pred_scores, pred_role_codes = (
            self._prediction_arrays()
        )
        out_task_ids, out_col_ids, out_gains = self._outcome_arrays()
        return {
            "max_history": self.max_history,
            "predictions": [
                {
                    "task_id": int(task_id),
                    "column_idx": int(column_idx),
                    "predicted_score": float(score),
                    "role": self._code_to_role(int(role_code)),
                }
                for task_id, column_idx, score, role_code in zip(
                    pred_task_ids, pred_col_ids, pred_scores, pred_role_codes
                )
            ],
            "outcomes": [
                {
                    "task_id": int(task_id),
                    "column_idx": int(column_idx),
                    "actual_gain": float(gain),
                }
                for task_id, column_idx, gain in zip(
                    out_task_ids, out_col_ids, out_gains
                )
            ],
            "recent_agreements": list(self.recent_agreements),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self.max_history = state.get("max_history", 100)
        self._allocate_buffers()
        self.predictions = []
        self.outcomes = []
        self.recent_agreements = list(state.get("recent_agreements", []))
        for pred in state.get("predictions", []):
            self.record_prediction(
                pred["task_id"],
                pred["column_idx"],
                pred["predicted_score"],
                pred.get("role", "challenger"),
            )
        for outcome in state.get("outcomes", []):
            self.record_outcome(
                outcome["task_id"],
                outcome["column_idx"],
                outcome["actual_gain"],
            )


@dataclass
class CausalSelectorTrustController:
    """
    Multi-gate trust controller for blending causal predictions with fallback.

    Gates:
    - coverage_gate: Based on number of training examples
    - agreement_gate: Based on recent prediction agreement
    - trend_gate: EMA-based trend detection
    - structural_gate: Based on internal trust metrics

    Output: effective_scale for blending causal vs certificate scores
    """

    config: SupportConfig
    agreement_ema: Optional[float] = None
    agreement_tracker: AgreementTracker = field(default_factory=AgreementTracker)
    last_diag: Dict[str, float] = field(
        default_factory=lambda: {
            "coverage_gate": 0.0,
            "agreement_gate": 0.0,
            "trend_gate": 0.0,
            "structural_gate": 0.0,
            "noise_floor": 0.0,
            "effective_scale": 0.0,
            "mix_gate": 0.0,
        }
    )
    last_effective_scale: float = 0.0

    def compute(
        self,
        predictor: CausalContributionPredictor,
        effective_internal_trust: float,
        recent_agreement: float,
        recent_rows: int,
    ) -> Dict[str, float]:
        """
        Compute trust gates and effective scale.

        Args:
            predictor: The causal predictor to check
            effective_internal_trust: Trust score from support state
            recent_agreement: Recent prediction agreement rate
            recent_rows: Number of recent audit rows

        Returns:
            Dict with gate values and effective_scale
        """
        n = predictor.num_examples()
        min_ex = self.config.causal_min_examples
        target_ex = max(self.config.causal_target_examples, min_ex)

        if n < min_ex:
            self.last_diag = {
                "coverage_gate": 0.0,
                "agreement_gate": 0.0,
                "trend_gate": 0.0,
                "structural_gate": float(effective_internal_trust),
                "noise_floor": 0.0,
                "effective_scale": 0.0,
                "mix_gate": 0.0,
            }
            self.last_effective_scale = 0.0
            return dict(self.last_diag)

        # Coverage gate: ramps up from min_examples to target_examples
        coverage_gate = max(0.0, min(1.0, (n - min_ex) / max(1.0, target_ex - min_ex)))

        # Noise floor based on number of recent rows
        noise_floor = min(
            0.25, max(0.05, 1.0 / math.sqrt(max(4.0, float(recent_rows))))
        )

        # Agreement gate
        if recent_agreement <= noise_floor:
            agreement_gate = 0.0
        else:
            raw = (recent_agreement - noise_floor) / max(
                1e-6, self.config.causal_agreement_target - noise_floor
            )
            agreement_gate = math.sqrt(max(0.0, min(1.0, raw)))

        # Trend gate: penalizes drops in agreement
        if self.agreement_ema is None:
            self.agreement_ema = recent_agreement
        trend_drop = max(0.0, self.agreement_ema - recent_agreement)
        trend_gate = math.exp(-trend_drop / max(1e-6, self.config.causal_trend_tau))
        self.agreement_ema = 0.75 * self.agreement_ema + 0.25 * recent_agreement

        # Structural gate: bounded internal trust
        structural_gate = max(
            self.config.structural_trust_floor,
            min(self.config.structural_trust_target, effective_internal_trust),
        )

        # Combine all gates
        effective_scale = (
            self.config.causal_max_effective_scale
            * math.sqrt(max(0.0, coverage_gate))
            * agreement_gate
            * trend_gate
            * structural_gate
        )
        effective_scale = max(
            0.0, min(self.config.causal_max_effective_scale, effective_scale)
        )

        # Mix gate is normalized effective scale
        if self.config.causal_max_effective_scale <= 1e-8:
            mix_gate = 0.0
        else:
            mix_gate = effective_scale / self.config.causal_max_effective_scale

        self.last_effective_scale = effective_scale
        self.last_diag = {
            "coverage_gate": float(coverage_gate),
            "agreement_gate": float(agreement_gate),
            "trend_gate": float(trend_gate),
            "structural_gate": float(structural_gate),
            "noise_floor": float(noise_floor),
            "effective_scale": float(effective_scale),
            "mix_gate": float(mix_gate),
        }
        return dict(self.last_diag)

    def record_prediction(
        self,
        task_id: int,
        column_idx: int,
        predicted_score: float,
        role: str = "challenger",
    ) -> None:
        """Record a predictor's score for tracking agreement."""
        self.agreement_tracker.record_prediction(
            task_id, column_idx, predicted_score, role
        )

    def record_outcome(
        self,
        task_id: int,
        column_idx: int,
        actual_gain: float,
    ) -> None:
        """Record actual outcome from audit for agreement tracking."""
        self.agreement_tracker.record_outcome(task_id, column_idx, actual_gain)

    def get_recent_agreement(self) -> Tuple[float, int]:
        """Get recent agreement rate and sample count."""
        return self.agreement_tracker.compute_recent_agreement()

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "agreement_ema": self.agreement_ema,
            "last_diag": dict(self.last_diag),
            "last_effective_scale": self.last_effective_scale,
            "agreement_tracker": self.agreement_tracker.save_state(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self.agreement_ema = state.get("agreement_ema")
        self.last_diag = state.get("last_diag", dict(self.last_diag))
        self.last_effective_scale = state.get("last_effective_scale", 0.0)
        if "agreement_tracker" in state:
            self.agreement_tracker.load_state(state["agreement_tracker"])


# ----------------------------
# Causal Support Feature Builder
# ----------------------------


@dataclass
class CausalSupportFeatureBuilder:
    """
    Builds 21-dimensional feature vectors for causal contribution prediction.

    Features:
    0. base_z: Base z-score
    1-4. cert_general, cert_specific, cert_demotion, cert_saturation
    5-8. novelty, saturation, recent_penalty, reserve_bonus
    9. reserve_flag (0 or 1)
    10-13. fingerprint features (cur, old, abs, conf)
    14-15. struct_max, causal_max similarity
    16-18. role one-hot (reuse, diverse, challenger)
    19. context_size (normalized)
    20. task_pos (normalized)
    """

    num_columns: int
    num_tasks: int
    topk_nonshared: int
    reserve_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        num_columns: int,
        num_tasks: int,
        topk_nonshared: int,
        reserve_indices: Optional[List[int]] = None,
    ) -> "CausalSupportFeatureBuilder":
        """Factory from config values."""
        return cls(
            num_columns=num_columns,
            num_tasks=num_tasks,
            topk_nonshared=topk_nonshared,
            reserve_indices=reserve_indices or [],
        )

    def build_feature(
        self,
        idx: int,
        role: str,
        chosen: Sequence[int],
        base_z: np.ndarray,
        cert_general: np.ndarray,
        cert_specific: np.ndarray,
        cert_demotion: np.ndarray,
        cert_saturation: np.ndarray,
        novelty: np.ndarray,
        saturation: np.ndarray,
        recent_penalty: np.ndarray,
        reserve_bonus: np.ndarray,
        fingerprint_mean: Optional[np.ndarray],
        fingerprint_confidence: Optional[np.ndarray],
        struct_similarity_fn: Callable[[int, int], float],
        causal_similarity_fn: Callable[[int, int], float],
        current_task_id: int,
    ) -> np.ndarray:
        """Build 21-dim feature vector for column idx."""
        # Fingerprint features
        if fingerprint_mean is not None and idx < fingerprint_mean.shape[0]:
            fp = fingerprint_mean[idx]
            fp_conf = (
                float(fingerprint_confidence[idx])
                if fingerprint_confidence is not None
                else 0.0
            )
            fp_cur = float(fp[current_task_id]) if current_task_id < fp.size else 0.0
            fp_old = (
                float(np.mean(fp[:current_task_id])) if current_task_id > 0 else 0.0
            )
            fp_abs = float(np.mean(np.abs(fp))) if fp.size > 0 else 0.0
        else:
            fp_conf = 0.0
            fp_cur = 0.0
            fp_old = 0.0
            fp_abs = 0.0

        # Similarity to chosen columns
        if chosen:
            struct_sims = [struct_similarity_fn(idx, j) for j in chosen]
            causal_sims = [causal_similarity_fn(idx, j) for j in chosen]
            struct_max = max(struct_sims)
            causal_max = max(causal_sims)
        else:
            struct_max = 0.0
            causal_max = 0.0

        # Role one-hot encoding
        role_map = {
            "reuse": [1.0, 0.0, 0.0],
            "diverse": [0.0, 1.0, 0.0],
            "challenger": [0.0, 0.0, 1.0],
        }
        role_one_hot = role_map.get(role, [0.0, 0.0, 0.0])

        # Reserve flag
        reserve_flag = 1.0 if idx in self.reserve_indices else 0.0

        # Context size (normalized)
        context_size = float(len(chosen)) / max(1.0, float(self.topk_nonshared - 1))

        # Task position (normalized)
        task_pos = float(current_task_id) / max(1.0, float(self.num_tasks - 1))

        # Build feature vector
        vals = [
            float(base_z[idx]),
            float(cert_general[idx]),
            float(cert_specific[idx]),
            float(cert_demotion[idx]),
            float(cert_saturation[idx]),
            float(novelty[idx]),
            float(saturation[idx]),
            float(recent_penalty[idx]),
            float(reserve_bonus[idx]),
            reserve_flag,
            fp_cur,
            fp_old,
            fp_abs,
            fp_conf,
            struct_max,
            causal_max,
            role_one_hot[0],
            role_one_hot[1],
            role_one_hot[2],
            context_size,
            task_pos,
        ]
        return np.array(vals, dtype=np.float64)

    def build_features_batch(
        self,
        indices: Sequence[int],
        roles: Sequence[str] | str,
        chosen_sets: Sequence[Sequence[int]],
        base_z: np.ndarray,
        cert_general: np.ndarray,
        cert_specific: np.ndarray,
        cert_demotion: np.ndarray,
        cert_saturation: np.ndarray,
        novelty: np.ndarray,
        saturation: np.ndarray,
        recent_penalty: np.ndarray,
        reserve_bonus: np.ndarray,
        fingerprint_mean: Optional[np.ndarray],
        fingerprint_confidence: Optional[np.ndarray],
        current_task_id: int,
    ) -> np.ndarray:
        """
        Build a batch of 21-dimensional feature vectors.

        This is a vectorized replacement for repeated `build_feature()` calls in
        support selection and audit processing. It uses batched JAX array ops for
        the feature construction hot path and returns a NumPy array for the
        downstream predictor.
        """
        jax_dtype = jnp.float32
        idx_arr = np.asarray(indices, dtype=np.int32).ravel()
        n = idx_arr.shape[0]
        if n == 0:
            return np.zeros((0, 21), dtype=np.float64)

        if isinstance(roles, str):
            role_list = [roles] * n
        else:
            role_list = list(roles)
        if len(role_list) != n:
            raise ValueError(
                f"roles length {len(role_list)} does not match indices length {n}"
            )

        chosen_list = [tuple(chosen) for chosen in chosen_sets]
        if len(chosen_list) != n:
            raise ValueError(
                f"chosen_sets length {len(chosen_list)} does not match indices length {n}"
            )

        max_chosen = max((len(chosen) for chosen in chosen_list), default=0)
        chosen_padded = np.zeros((n, max_chosen), dtype=np.int32)
        chosen_mask = np.zeros((n, max_chosen), dtype=bool)
        chosen_lengths = np.zeros((n,), dtype=np.int32)
        for i, chosen in enumerate(chosen_list):
            chosen_lengths[i] = len(chosen)
            if chosen:
                chosen_padded[i, : len(chosen)] = chosen
                chosen_mask[i, : len(chosen)] = True

        num_columns = len(base_z)
        valid_idx = (idx_arr >= 0) & (idx_arr < num_columns)
        idx_clipped = np.clip(idx_arr, 0, max(0, num_columns - 1))

        idx_j = jnp.asarray(idx_clipped, dtype=jnp.int32)
        valid_idx_j = jnp.asarray(valid_idx)
        chosen_padded_j = jnp.asarray(chosen_padded, dtype=jnp.int32)
        chosen_mask_j = jnp.asarray(chosen_mask)
        chosen_lengths_j = jnp.asarray(chosen_lengths, dtype=jax_dtype)

        def gather_feature(values: np.ndarray) -> jnp.ndarray:
            arr = jnp.asarray(values, dtype=jax_dtype)
            gathered = arr[idx_j]
            return jnp.where(valid_idx_j, gathered, 0.0)

        reserve_flag = np.isin(
            idx_arr, np.asarray(self.reserve_indices, dtype=np.int32)
        )
        reserve_flag_j = jnp.asarray(reserve_flag, dtype=jax_dtype)

        fp_cur = jnp.zeros((n,), dtype=jax_dtype)
        fp_old = jnp.zeros((n,), dtype=jax_dtype)
        fp_abs = jnp.zeros((n,), dtype=jax_dtype)
        fp_conf = jnp.zeros((n,), dtype=jax_dtype)
        causal_max = jnp.zeros((n,), dtype=jax_dtype)

        if fingerprint_mean is not None and np.size(fingerprint_mean) > 0:
            fp_mean = jnp.asarray(fingerprint_mean, dtype=jax_dtype)
            fp_rows = fp_mean.shape[0]
            fp_valid_idx = valid_idx_j & (idx_j < fp_rows)
            fp_idx = jnp.clip(idx_j, 0, max(0, fp_rows - 1))
            gathered_fp = fp_mean[fp_idx]
            if current_task_id < fp_mean.shape[1]:
                fp_cur = jnp.where(fp_valid_idx, gathered_fp[:, current_task_id], 0.0)
            if current_task_id > 0:
                fp_old_means = jnp.mean(fp_mean[:, :current_task_id], axis=1)
                fp_old = jnp.where(fp_valid_idx, fp_old_means[fp_idx], 0.0)
            fp_abs_means = jnp.mean(jnp.abs(fp_mean), axis=1)
            fp_abs = jnp.where(fp_valid_idx, fp_abs_means[fp_idx], 0.0)
            if fingerprint_confidence is not None:
                fp_conf_arr = jnp.asarray(fingerprint_confidence, dtype=jax_dtype)
                fp_conf = jnp.where(fp_valid_idx, fp_conf_arr[fp_idx], 0.0)

            if max_chosen > 0:
                chosen_valid = (
                    chosen_mask_j & (chosen_padded_j >= 0) & (chosen_padded_j < fp_rows)
                )
                chosen_idx = jnp.clip(chosen_padded_j, 0, max(0, fp_rows - 1))
                normalized = fp_mean / jnp.clip(
                    jnp.linalg.norm(fp_mean, axis=1, keepdims=True), 1e-6, None
                )
                current_vectors = normalized[fp_idx]
                chosen_vectors = normalized[chosen_idx]
                sims = jnp.sum(current_vectors[:, None, :] * chosen_vectors, axis=-1)
                sims = jnp.where(chosen_valid, sims, -jnp.inf)
                causal_max = jnp.where(
                    chosen_lengths_j > 0,
                    jnp.max(sims, axis=1),
                    0.0,
                )
                causal_max = jnp.where(jnp.isfinite(causal_max), causal_max, 0.0)

        struct_max = jnp.zeros((n,), dtype=jax_dtype)
        if max_chosen > 0:
            struct_max = jnp.any(
                chosen_mask_j & (chosen_padded_j == idx_j[:, None]), axis=1
            ).astype(jax_dtype)

        role_map = {
            "reuse": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "diverse": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "challenger": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        role_one_hot = np.stack(
            [role_map.get(role, np.zeros(3, dtype=np.float64)) for role in role_list],
            axis=0,
        )
        role_one_hot_j = jnp.asarray(role_one_hot, dtype=jax_dtype)

        context_denom = max(1.0, float(self.topk_nonshared - 1))
        context_size = chosen_lengths_j / context_denom
        task_pos = jnp.full(
            (n,),
            float(current_task_id) / max(1.0, float(self.num_tasks - 1)),
            dtype=jax_dtype,
        )

        features = jnp.column_stack(
            [
                gather_feature(base_z),
                gather_feature(cert_general),
                gather_feature(cert_specific),
                gather_feature(cert_demotion),
                gather_feature(cert_saturation),
                gather_feature(novelty),
                gather_feature(saturation),
                gather_feature(recent_penalty),
                gather_feature(reserve_bonus),
                reserve_flag_j,
                fp_cur,
                fp_old,
                fp_abs,
                fp_conf,
                struct_max,
                causal_max,
                role_one_hot_j[:, 0],
                role_one_hot_j[:, 1],
                role_one_hot_j[:, 2],
                context_size,
                task_pos,
            ]
        )
        return np.asarray(features, dtype=np.float64)
