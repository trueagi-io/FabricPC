"""
Optimal Transport Utilities for Continual Learning.

Provides Sinkhorn-based optimal transport algorithms used by:
- TransWeave transfer learning (transweave.py)
- Per-weight causal coding (weight_causal.py)

This module centralizes optimal transport functionality to avoid code duplication.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ----------------------------
# Core Sinkhorn Transport
# ----------------------------


def sinkhorn_transport(
    cost: np.ndarray,
    source_weights: Optional[np.ndarray] = None,
    target_weights: Optional[np.ndarray] = None,
    eps: float = 0.25,
    iters: int = 15,
    identity_bonus: float = 0.0,
) -> np.ndarray:
    """
    Compute optimal transport plan using Sinkhorn algorithm.

    Solves the entropy-regularized optimal transport problem:
        min_P <C, P> - eps * H(P)
        s.t. P @ 1 = source_weights, P.T @ 1 = target_weights

    Args:
        cost: Cost matrix (n, m)
        source_weights: Source marginal weights (n,). Defaults to uniform.
        target_weights: Target marginal weights (m,). Defaults to uniform.
        eps: Entropy regularization parameter. Higher = more diffuse transport.
        iters: Number of Sinkhorn iterations.
        identity_bonus: Bonus for diagonal (identity) transport. Only for square matrices.

    Returns:
        Transport plan (n, m) - doubly stochastic matrix satisfying marginal constraints.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape

    # Default to uniform weights
    if source_weights is None:
        source_weights = np.ones(n) / n
    if target_weights is None:
        target_weights = np.ones(m) / m

    source_weights = np.asarray(source_weights, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)

    # Normalize weights
    source_weights = source_weights / (np.sum(source_weights) + 1e-10)
    target_weights = target_weights / (np.sum(target_weights) + 1e-10)

    # Add identity bonus to diagonal (if square)
    if n == m and identity_bonus > 0:
        identity_cost = np.eye(n) * (-identity_bonus)
        cost = cost + identity_cost

    # Initialize Gibbs kernel
    K = np.exp(-cost / (eps + 1e-10))

    # Sinkhorn iterations
    u = np.ones(n)
    v = np.ones(m)

    for _ in range(iters):
        u = source_weights / (K @ v + 1e-10)
        v = target_weights / (K.T @ u + 1e-10)

    # Transport plan
    transport = np.outer(u, v) * K

    return transport


def sinkhorn_1d_correction(
    values: np.ndarray,
    eps: float = 0.1,
    iters: int = 5,
) -> np.ndarray:
    """
    Compute Sinkhorn-based correction for 1D distribution.

    Maps empirical distribution towards Gaussian via optimal transport.
    Used for gradient Gaussianization in per-weight causal coding.

    Args:
        values: 1D array of values to correct
        eps: Regularization parameter (relative to variance)
        iters: Number of Sinkhorn iterations

    Returns:
        Corrected values with more Gaussian-like distribution
    """
    n = len(values)
    if n < 2:
        return values

    # Sort values to get empirical CDF
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]

    # Generate target Gaussian quantiles (same mean/std as input)
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-8:
        return values

    # Gaussian quantiles
    probs = (np.arange(n) + 0.5) / n
    target = mean + std * np.sqrt(2) * erfinv_approx(2 * probs - 1)

    # Cost matrix (squared distance)
    C = (sorted_vals.reshape(-1, 1) - target.reshape(1, -1)) ** 2

    # Sinkhorn algorithm
    K = np.exp(-C / (eps * std**2 + 1e-8))
    u = np.ones(n)
    v = np.ones(n)

    for _ in range(iters):
        u = 1.0 / (K @ v + 1e-10)
        v = 1.0 / (K.T @ u + 1e-10)

    # Transport plan
    P = np.outer(u, v) * K
    P = P / (P.sum(axis=1, keepdims=True) + 1e-10)

    # Barycentric projection: corrected values
    corrected_sorted = P @ target

    # Unsort to original order
    corrected = np.empty_like(values)
    corrected[sorted_idx] = corrected_sorted

    return corrected


# ----------------------------
# Mathematical Utilities
# ----------------------------


def erfinv_approx(x: np.ndarray) -> np.ndarray:
    """
    Approximate inverse error function.

    Uses rational approximation for |x| < 1.
    Required for computing Gaussian quantiles in sinkhorn_1d_correction.

    Args:
        x: Input array with values in (-1, 1)

    Returns:
        Approximate erfinv(x)
    """
    x = np.clip(x, -0.99999, 0.99999)

    # Approximation coefficients (Winitzki, 2008)
    a = 0.147
    ln_term = np.log(1 - x**2)
    term1 = 2 / (np.pi * a) + ln_term / 2
    term2 = ln_term / a

    sign = np.sign(x)
    result = sign * np.sqrt(np.sqrt(term1**2 - term2) - term1)

    return result


# ----------------------------
# Cost Matrix Helpers
# ----------------------------


def cosine_cost_matrix(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine distance cost matrix.

    Args:
        source: Source representations (n, d)
        target: Target representations (m, d)

    Returns:
        Cost matrix (n, m) where cost[i,j] = 1 - cosine_sim(source[i], target[j])
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # Normalize
    source_norm = source / (np.linalg.norm(source, axis=1, keepdims=True) + 1e-8)
    target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity
    similarity = source_norm @ target_norm.T

    # Convert to cost (1 - similarity)
    cost = 1.0 - similarity

    return cost


def euclidean_cost_matrix(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Compute squared Euclidean distance cost matrix.

    Args:
        source: Source representations (n, d)
        target: Target representations (m, d)

    Returns:
        Cost matrix (n, m) where cost[i,j] = ||source[i] - target[j]||^2
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    # ||s - t||^2 = ||s||^2 + ||t||^2 - 2*s.t
    source_sq = np.sum(source**2, axis=1, keepdims=True)
    target_sq = np.sum(target**2, axis=1, keepdims=True)

    cost = source_sq + target_sq.T - 2 * (source @ target.T)
    cost = np.maximum(cost, 0.0)  # Numerical stability

    return cost
