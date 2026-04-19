"""
Optimal Transport Utilities for Continual Learning.

Provides Sinkhorn-based optimal transport algorithms used by:
- TransWeave transfer learning (transweave.py)
- Per-weight causal coding (weight_causal.py)

This module centralizes optimal transport functionality to avoid code duplication.
Uses JAX for efficient computation with NumPy fallback.
"""

from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING
import math

# Try to import JAX, fall back to NumPy
try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    import numpy as np

    HAS_JAX = True
    ArrayType = Union[np.ndarray, jnp.ndarray]
except ImportError:
    import numpy as np
    import numpy as jnp  # type: ignore

    HAS_JAX = False
    ArrayType = np.ndarray


# ----------------------------
# Core Sinkhorn Transport
# ----------------------------


def sinkhorn_transport(
    cost: ArrayType,
    source_weights: Optional[ArrayType] = None,
    target_weights: Optional[ArrayType] = None,
    eps: float = 0.25,
    iters: int = 15,
    identity_bonus: float = 0.0,
) -> ArrayType:
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
    cost = jnp.asarray(cost, dtype=jnp.float32)
    n, m = cost.shape

    # Default to uniform weights
    if source_weights is None:
        source_weights = jnp.ones(n) / n
    else:
        source_weights = jnp.asarray(source_weights, dtype=jnp.float32)

    if target_weights is None:
        target_weights = jnp.ones(m) / m
    else:
        target_weights = jnp.asarray(target_weights, dtype=jnp.float32)

    # Normalize weights
    source_weights = source_weights / (jnp.sum(source_weights) + 1e-10)
    target_weights = target_weights / (jnp.sum(target_weights) + 1e-10)

    # Add identity bonus to diagonal (if square)
    if n == m and identity_bonus > 0:
        identity_cost = jnp.eye(n) * (-identity_bonus)
        cost = cost + identity_cost

    # Initialize Gibbs kernel
    K = jnp.exp(-cost / (eps + 1e-10))

    # Sinkhorn iterations
    u = jnp.ones(n)
    v = jnp.ones(m)

    if HAS_JAX:
        # Use lax.fori_loop for efficient JAX compilation
        def sinkhorn_step(_, uv):
            u, v = uv
            u = source_weights / (K @ v + 1e-10)
            v = target_weights / (K.T @ u + 1e-10)
            return (u, v)

        u, v = lax.fori_loop(0, iters, sinkhorn_step, (u, v))
    else:
        for _ in range(iters):
            u = source_weights / (K @ v + 1e-10)
            v = target_weights / (K.T @ u + 1e-10)

    # Transport plan
    transport = jnp.outer(u, v) * K

    return transport


def sinkhorn_1d_correction(
    values: ArrayType,
    eps: float = 0.1,
    iters: int = 5,
) -> ArrayType:
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
    values = jnp.asarray(values, dtype=jnp.float32)
    n = len(values)

    if n < 2:
        return values

    # Sort values to get empirical CDF
    sorted_idx = jnp.argsort(values)
    sorted_vals = values[sorted_idx]

    # Generate target Gaussian quantiles (same mean/std as input)
    mean = jnp.mean(values)
    std = jnp.std(values)

    # Early return if std is too small
    if HAS_JAX:
        # For JAX, we need to handle this without Python conditionals on traced values
        # We'll compute anyway and blend with original based on std
        std_safe = jnp.maximum(std, 1e-8)
    else:
        if std < 1e-8:
            return values
        std_safe = std

    # Gaussian quantiles
    probs = (jnp.arange(n) + 0.5) / n
    target = mean + std_safe * jnp.sqrt(2.0) * erfinv_approx(2 * probs - 1)

    # Cost matrix (squared distance)
    C = (sorted_vals.reshape(-1, 1) - target.reshape(1, -1)) ** 2

    # Sinkhorn algorithm
    K = jnp.exp(-C / (eps * std_safe**2 + 1e-8))
    u = jnp.ones(n)
    v = jnp.ones(n)

    if HAS_JAX:

        def sinkhorn_step(_, uv):
            u, v = uv
            u = 1.0 / (K @ v + 1e-10)
            v = 1.0 / (K.T @ u + 1e-10)
            return (u, v)

        u, v = lax.fori_loop(0, iters, sinkhorn_step, (u, v))
    else:
        for _ in range(iters):
            u = 1.0 / (K @ v + 1e-10)
            v = 1.0 / (K.T @ u + 1e-10)

    # Transport plan
    P = jnp.outer(u, v) * K
    P = P / (P.sum(axis=1, keepdims=True) + 1e-10)

    # Barycentric projection: corrected values
    corrected_sorted = P @ target

    # Unsort to original order using inverse permutation
    inverse_idx = jnp.argsort(sorted_idx)
    corrected = corrected_sorted[inverse_idx]

    # If std was too small, return original values
    if HAS_JAX:
        # Blend based on whether std was valid
        blend = jnp.where(std < 1e-8, 0.0, 1.0)
        corrected = blend * corrected + (1 - blend) * values
    # For NumPy path, we already returned early

    return corrected


# ----------------------------
# Mathematical Utilities
# ----------------------------


def erfinv_approx(x: ArrayType) -> ArrayType:
    """
    Approximate inverse error function.

    Uses rational approximation for |x| < 1.
    Required for computing Gaussian quantiles in sinkhorn_1d_correction.

    Args:
        x: Input array with values in (-1, 1)

    Returns:
        Approximate erfinv(x)
    """
    x = jnp.clip(x, -0.99999, 0.99999)

    # Approximation coefficients (Winitzki, 2008)
    a = 0.147
    ln_term = jnp.log(1 - x**2)
    term1 = 2 / (math.pi * a) + ln_term / 2
    term2 = ln_term / a

    sign = jnp.sign(x)
    result = sign * jnp.sqrt(jnp.sqrt(term1**2 - term2) - term1)

    return result


# ----------------------------
# Cost Matrix Helpers
# ----------------------------


def cosine_cost_matrix(
    source: ArrayType,
    target: ArrayType,
) -> ArrayType:
    """
    Compute cosine distance cost matrix.

    Args:
        source: Source representations (n, d)
        target: Target representations (m, d)

    Returns:
        Cost matrix (n, m) where cost[i,j] = 1 - cosine_sim(source[i], target[j])
    """
    source = jnp.asarray(source, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)

    # Normalize
    source_norm = source / (jnp.linalg.norm(source, axis=1, keepdims=True) + 1e-8)
    target_norm = target / (jnp.linalg.norm(target, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity
    similarity = source_norm @ target_norm.T

    # Convert to cost (1 - similarity)
    cost = 1.0 - similarity

    return cost


def euclidean_cost_matrix(
    source: ArrayType,
    target: ArrayType,
) -> ArrayType:
    """
    Compute squared Euclidean distance cost matrix.

    Args:
        source: Source representations (n, d)
        target: Target representations (m, d)

    Returns:
        Cost matrix (n, m) where cost[i,j] = ||source[i] - target[j]||^2
    """
    source = jnp.asarray(source, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)

    # ||s - t||^2 = ||s||^2 + ||t||^2 - 2*s.t
    source_sq = jnp.sum(source**2, axis=1, keepdims=True)
    target_sq = jnp.sum(target**2, axis=1, keepdims=True)

    cost = source_sq + target_sq.T - 2 * (source @ target.T)
    cost = jnp.maximum(cost, 0.0)  # Numerical stability

    return cost
