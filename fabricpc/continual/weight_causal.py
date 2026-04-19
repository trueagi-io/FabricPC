"""
Per-Weight Causal Coding for Continual Learning.

Implements causal coding at the per-weight level as described in V20.2b+:
- Standard learning (Adam/SGD) for weights with Gaussian-like gradient distributions
- Sinkhorn-based (SB) learning for weights with extreme non-Gaussianity

This enables fine-grained causal control where individual weight updates
are modulated based on their gradient distribution characteristics.

Key components:
- WeightGradientTracker: Tracks per-weight gradient history
- PerWeightKurtosisComputer: Computes kurtosis for weight gradients
- AdaptiveWeightUpdater: Switches between standard and SB updates
- SinkhornWeightCorrection: Applies transport-based correction

Reference: v_20_b_vs_v_18.pdf Section 10, 13.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # type: ignore

# Import optimal transport utilities from shared module
from fabricpc.continual.optimal_transport import (
    sinkhorn_1d_correction,
    erfinv_approx,
)

# Type alias for array types
ArrayType = np.ndarray  # For type hints, actual implementation uses jnp when available

# ----------------------------
# Configuration
# ----------------------------


@dataclass
class PerWeightCausalConfig:
    """Configuration for per-weight causal coding."""

    # Enable per-weight causal coding
    enable: bool = True

    # Gradient tracking
    gradient_history_size: int = 32  # Number of recent gradients to track per weight
    min_history_for_detection: int = 8  # Minimum history before computing kurtosis

    # Non-Gaussianity thresholds
    kurtosis_threshold: float = 2.0  # Excess kurtosis threshold for non-Gaussianity
    multimodal_threshold: float = 0.5  # Multimodal gap threshold
    combined_threshold: float = 1.5  # Combined non-Gaussianity score threshold

    # Sinkhorn parameters for SB update
    sb_sinkhorn_eps: float = 0.1  # Regularization for Sinkhorn
    sb_sinkhorn_iters: int = 5  # Sinkhorn iterations
    sb_correction_strength: float = 0.3  # How much to blend SB correction

    # Adaptive blending
    blend_mode: str = "soft"  # "hard" (binary) or "soft" (smooth transition)
    soft_blend_scale: float = 1.0  # Scale for soft sigmoid transition

    # Per-layer control
    skip_bias_weights: bool = True  # Skip bias terms (typically more Gaussian)
    skip_small_weights: bool = True  # Skip weights below threshold
    small_weight_threshold: float = 1e-6  # Threshold for small weights

    # Statistics tracking
    track_statistics: bool = True  # Track per-weight statistics for debugging
    stats_update_every: int = 10  # Update summary stats every N steps


# ----------------------------
# Utility Functions (JAX-accelerated)
# ----------------------------


def compute_weight_excess_kurtosis(gradients: ArrayType) -> ArrayType:
    """
    Compute excess kurtosis for each weight from gradient history.

    Uses JAX for efficient computation when available.

    Args:
        gradients: Array of shape (history_size, *weight_shape)

    Returns:
        Array of shape (*weight_shape) with per-weight excess kurtosis
    """
    gradients = jnp.asarray(gradients, dtype=jnp.float32)

    if gradients.shape[0] < 4:
        return jnp.zeros(gradients.shape[1:], dtype=jnp.float32)

    # Compute along history axis (axis=0)
    mean = jnp.mean(gradients, axis=0)
    std = jnp.std(gradients, axis=0)

    # Avoid division by zero
    std = jnp.clip(std, 1e-8, None)

    # Standardize
    z = (gradients - mean) / std

    # Fourth moment minus 3 (excess kurtosis)
    kurtosis = jnp.mean(z**4, axis=0) - 3.0

    return kurtosis


def compute_weight_multimodal_gap(gradients: ArrayType) -> ArrayType:
    """
    Compute multimodal gap metric for each weight.

    Measures separation between lower and upper halves of gradient distribution.
    Uses JAX for efficient computation when available.

    Args:
        gradients: Array of shape (history_size, *weight_shape)

    Returns:
        Array of shape (*weight_shape) with per-weight multimodal gap
    """
    gradients = jnp.asarray(gradients, dtype=jnp.float32)

    if gradients.shape[0] < 4:
        return jnp.zeros(gradients.shape[1:], dtype=jnp.float32)

    # Compute median and std
    median = jnp.median(gradients, axis=0, keepdims=True)
    std = jnp.std(gradients, axis=0)
    std = jnp.clip(std, 1e-8, None)

    # Split into lower and upper halves
    lower_mask = gradients <= median
    upper_mask = gradients > median

    # Compute means for each half using masked operations
    # For JAX compatibility, use where with sum/count instead of nanmean
    lower_sum = jnp.sum(jnp.where(lower_mask, gradients, 0.0), axis=0)
    lower_count = jnp.sum(lower_mask.astype(jnp.float32), axis=0)
    lower_mean = lower_sum / jnp.maximum(lower_count, 1.0)

    upper_sum = jnp.sum(jnp.where(upper_mask, gradients, 0.0), axis=0)
    upper_count = jnp.sum(upper_mask.astype(jnp.float32), axis=0)
    upper_mean = upper_sum / jnp.maximum(upper_count, 1.0)

    gap = jnp.abs(upper_mean - lower_mean) / std

    return gap


def compute_non_gaussianity_score(
    kurtosis: ArrayType,
    multimodal: ArrayType,
    config: PerWeightCausalConfig,
) -> ArrayType:
    """
    Compute combined non-Gaussianity score.

    Uses JAX for efficient computation when available.

    Args:
        kurtosis: Per-weight excess kurtosis
        multimodal: Per-weight multimodal gap
        config: Configuration

    Returns:
        Combined non-Gaussianity score per weight
    """
    kurtosis = jnp.asarray(kurtosis, dtype=jnp.float32)
    multimodal = jnp.asarray(multimodal, dtype=jnp.float32)

    # Excess beyond thresholds
    kurt_excess = jnp.maximum(0.0, jnp.abs(kurtosis) - config.kurtosis_threshold)
    multi_excess = jnp.maximum(0.0, multimodal - config.multimodal_threshold)

    # Combined score
    score = kurt_excess + multi_excess

    return score


# ----------------------------
# Weight Gradient Tracker
# ----------------------------


@dataclass
class WeightGradientTracker:
    """
    Tracks per-weight gradient history for non-Gaussianity detection.

    Maintains a rolling buffer of recent gradients for each parameter.
    """

    config: PerWeightCausalConfig
    _buffers: Dict[str, np.ndarray] = field(default_factory=dict)
    _positions: Dict[str, int] = field(default_factory=dict)
    _filled: Dict[str, bool] = field(default_factory=dict)
    _step_count: int = 0

    def update(self, param_name: str, gradient: np.ndarray) -> None:
        """
        Add gradient to tracking buffer.

        Args:
            param_name: Name of the parameter
            gradient: Gradient array for this parameter
        """
        if not self.config.enable:
            return

        gradient = np.asarray(gradient)

        # Initialize buffer if needed
        if param_name not in self._buffers:
            buffer_shape = (self.config.gradient_history_size,) + gradient.shape
            self._buffers[param_name] = np.zeros(buffer_shape, dtype=np.float64)
            self._positions[param_name] = 0
            self._filled[param_name] = False

        # Add to circular buffer
        pos = self._positions[param_name]
        self._buffers[param_name][pos] = gradient
        self._positions[param_name] = (pos + 1) % self.config.gradient_history_size

        if pos + 1 >= self.config.gradient_history_size:
            self._filled[param_name] = True

        self._step_count += 1

    def get_history(self, param_name: str) -> Optional[np.ndarray]:
        """
        Get gradient history for a parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            Array of shape (history_size, *weight_shape) or None if not tracked
        """
        if param_name not in self._buffers:
            return None

        buffer = self._buffers[param_name]

        if self._filled[param_name]:
            # Full buffer - reorder to chronological order
            pos = self._positions[param_name]
            return np.concatenate([buffer[pos:], buffer[:pos]], axis=0)
        else:
            # Partial buffer - return filled portion
            pos = self._positions[param_name]
            if pos == 0:
                return None
            return buffer[:pos]

    def has_sufficient_history(self, param_name: str) -> bool:
        """Check if parameter has enough history for kurtosis computation."""
        if param_name not in self._buffers:
            return False

        if self._filled[param_name]:
            return True

        return self._positions[param_name] >= self.config.min_history_for_detection

    def get_effective_size(self, param_name: str) -> int:
        """Get the effective history size for a parameter."""
        if param_name not in self._buffers:
            return 0

        if self._filled[param_name]:
            return self.config.gradient_history_size

        return self._positions[param_name]

    def clear(self, param_name: Optional[str] = None) -> None:
        """Clear tracking buffers."""
        if param_name is None:
            self._buffers.clear()
            self._positions.clear()
            self._filled.clear()
        elif param_name in self._buffers:
            del self._buffers[param_name]
            del self._positions[param_name]
            del self._filled[param_name]

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "buffers": {k: v.tolist() for k, v in self._buffers.items()},
            "positions": dict(self._positions),
            "filled": dict(self._filled),
            "step_count": self._step_count,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self._buffers = {k: np.array(v) for k, v in state.get("buffers", {}).items()}
        self._positions = dict(state.get("positions", {}))
        self._filled = dict(state.get("filled", {}))
        self._step_count = state.get("step_count", 0)


# ----------------------------
# Per-Weight Non-Gaussianity Detector
# ----------------------------


@dataclass
class PerWeightNonGaussianityResult:
    """Result of per-weight non-Gaussianity detection."""

    param_name: str
    kurtosis: np.ndarray  # Per-weight excess kurtosis
    multimodal: np.ndarray  # Per-weight multimodal gap
    non_gaussian_score: np.ndarray  # Combined score
    non_gaussian_mask: np.ndarray  # Boolean mask of non-Gaussian weights
    fraction_non_gaussian: float  # Fraction of weights flagged
    mean_kurtosis: float
    mean_multimodal: float
    max_kurtosis: float


@dataclass
class PerWeightNonGaussianityDetector:
    """
    Detects which weights have extreme non-Gaussianity in their gradient distributions.

    Uses kurtosis and multimodal gap metrics to identify weights that would
    benefit from Sinkhorn-based updates rather than standard Adam/SGD.
    """

    config: PerWeightCausalConfig
    tracker: WeightGradientTracker = field(default_factory=lambda: None)  # type: ignore

    def __post_init__(self):
        if self.tracker is None:
            self.tracker = WeightGradientTracker(config=self.config)

    def update_gradients(self, param_name: str, gradient: np.ndarray) -> None:
        """Update gradient history for a parameter."""
        self.tracker.update(param_name, gradient)

    def detect(self, param_name: str) -> Optional[PerWeightNonGaussianityResult]:
        """
        Detect non-Gaussianity for a parameter's weights.

        Uses JAX for efficient computation when available.

        Args:
            param_name: Name of the parameter

        Returns:
            PerWeightNonGaussianityResult or None if insufficient history
        """
        if not self.tracker.has_sufficient_history(param_name):
            return None

        history = self.tracker.get_history(param_name)
        if history is None:
            return None

        # Compute per-weight statistics (JAX-accelerated)
        kurtosis = compute_weight_excess_kurtosis(history)
        multimodal = compute_weight_multimodal_gap(history)
        score = compute_non_gaussianity_score(kurtosis, multimodal, self.config)

        # Determine non-Gaussian mask
        if self.config.blend_mode == "hard":
            mask = (score > self.config.combined_threshold).astype(jnp.float32)
        else:
            # Soft mask using sigmoid (JAX-compatible)
            mask = 1.0 / (
                1.0
                + jnp.exp(
                    -self.config.soft_blend_scale
                    * (score - self.config.combined_threshold)
                )
            )

        # Compute summary statistics
        n_total = kurtosis.size
        if self.config.blend_mode == "hard":
            n_non_gaussian = float(jnp.sum(mask))
        else:
            n_non_gaussian = float(jnp.sum(mask > 0.5))

        frac_non_gaussian = float(n_non_gaussian) / max(1, n_total)

        # Convert to numpy arrays for storage in result (keeps compatibility)
        return PerWeightNonGaussianityResult(
            param_name=param_name,
            kurtosis=np.asarray(kurtosis),
            multimodal=np.asarray(multimodal),
            non_gaussian_score=np.asarray(score),
            non_gaussian_mask=np.asarray(mask),
            fraction_non_gaussian=frac_non_gaussian,
            mean_kurtosis=float(jnp.mean(jnp.abs(kurtosis))),
            mean_multimodal=float(jnp.mean(multimodal)),
            max_kurtosis=float(jnp.max(jnp.abs(kurtosis))),
        )

    def detect_all(
        self, param_names: List[str]
    ) -> Dict[str, PerWeightNonGaussianityResult]:
        """Detect non-Gaussianity for multiple parameters."""
        results = {}
        for name in param_names:
            result = self.detect(name)
            if result is not None:
                results[name] = result
        return results

    def get_summary_stats(self, param_names: List[str]) -> Dict[str, float]:
        """Get summary statistics across all parameters."""
        results = self.detect_all(param_names)

        if not results:
            return {
                "mean_fraction_non_gaussian": 0.0,
                "mean_kurtosis": 0.0,
                "mean_multimodal": 0.0,
                "max_kurtosis": 0.0,
                "num_params_tracked": 0,
            }

        fracs = [r.fraction_non_gaussian for r in results.values()]
        kurts = [r.mean_kurtosis for r in results.values()]
        mults = [r.mean_multimodal for r in results.values()]
        max_kurts = [r.max_kurtosis for r in results.values()]

        return {
            "mean_fraction_non_gaussian": float(np.mean(fracs)),
            "mean_kurtosis": float(np.mean(kurts)),
            "mean_multimodal": float(np.mean(mults)),
            "max_kurtosis": float(np.max(max_kurts)),
            "num_params_tracked": len(results),
        }

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {"tracker": self.tracker.save_state()}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        if "tracker" in state:
            self.tracker.load_state(state["tracker"])


# ----------------------------
# Sinkhorn Weight Correction (JAX-accelerated)
# ----------------------------


def compute_sinkhorn_weight_correction(
    gradient: ArrayType,
    non_gaussian_mask: ArrayType,
    config: PerWeightCausalConfig,
) -> ArrayType:
    """
    Compute Sinkhorn-based correction for non-Gaussian weights.

    For weights flagged as non-Gaussian, applies transport-based correction
    to move gradient distribution towards Gaussian. Uses JAX for efficient
    computation when available.

    Args:
        gradient: Current gradient array
        non_gaussian_mask: Mask indicating non-Gaussian weights (0-1)
        config: Configuration

    Returns:
        Corrected gradient array
    """
    gradient = jnp.asarray(gradient, dtype=jnp.float32)
    mask = jnp.asarray(non_gaussian_mask, dtype=jnp.float32)
    original_shape = gradient.shape

    # If no weights need correction, return original
    if jnp.max(mask) < 1e-6:
        return gradient

    # Flatten for processing
    flat_grad = gradient.ravel()
    flat_mask = mask.ravel()

    # Identify weights that need correction
    needs_correction = flat_mask > 0.1

    if not jnp.any(needs_correction):
        return gradient

    # Get indices of weights needing correction
    idx_correct = jnp.where(needs_correction)[0]

    if len(idx_correct) >= 2:
        # Get gradients for these weights
        grads_to_correct = flat_grad[idx_correct]

        # Apply 1D Sinkhorn correction (already JAX-accelerated)
        corrected_grads = sinkhorn_1d_correction(
            grads_to_correct,
            eps=config.sb_sinkhorn_eps,
            iters=config.sb_sinkhorn_iters,
        )

        # Blend based on mask values
        blend_factors = flat_mask[idx_correct] * config.sb_correction_strength

        # Compute blended correction
        blended = (
            1 - blend_factors
        ) * grads_to_correct + blend_factors * corrected_grads

        # Update the corrected values at the indices
        # Use JAX's functional update pattern
        corrected = flat_grad.at[idx_correct].set(blended)
    else:
        corrected = flat_grad

    return corrected.reshape(original_shape)


# ----------------------------
# Adaptive Weight Updater
# ----------------------------


@dataclass
class AdaptiveWeightUpdateResult:
    """Result of adaptive weight update."""

    corrected_gradients: Dict[str, np.ndarray]
    standard_update_fraction: float  # Fraction using standard update
    sb_update_fraction: float  # Fraction using SB update
    diagnostics: Dict[str, Any]


class AdaptiveWeightUpdater:
    """
    Adaptive weight updater that switches between standard and SB updates.

    For each parameter:
    1. Tracks gradient history
    2. Detects non-Gaussianity per weight
    3. Applies standard update for Gaussian weights
    4. Applies SB-corrected update for non-Gaussian weights

    This implements the per-weight do-influence gating described in V20.2b+.
    """

    def __init__(self, config: PerWeightCausalConfig):
        self.config = config
        self.detector = PerWeightNonGaussianityDetector(config=config)
        self._step_count = 0
        self._stats_history: List[Dict[str, float]] = []

    def update_and_correct(
        self,
        gradients: Dict[str, np.ndarray],
        is_bias: Optional[Dict[str, bool]] = None,
    ) -> AdaptiveWeightUpdateResult:
        """
        Update gradient tracking and compute corrected gradients.

        Args:
            gradients: Dict mapping param_name -> gradient array
            is_bias: Optional dict indicating which params are biases

        Returns:
            AdaptiveWeightUpdateResult with corrected gradients
        """
        if is_bias is None:
            is_bias = {}

        corrected_gradients = {}
        total_weights = 0
        total_standard = 0
        total_sb = 0
        diagnostics_per_param = {}

        for param_name, grad in gradients.items():
            grad = np.asarray(grad)

            # Skip bias weights if configured
            if self.config.skip_bias_weights and is_bias.get(param_name, False):
                corrected_gradients[param_name] = grad
                total_weights += grad.size
                total_standard += grad.size
                continue

            # Skip small weights if configured
            if self.config.skip_small_weights:
                if np.max(np.abs(grad)) < self.config.small_weight_threshold:
                    corrected_gradients[param_name] = grad
                    total_weights += grad.size
                    total_standard += grad.size
                    continue

            # Update gradient tracking
            self.detector.update_gradients(param_name, grad)

            # Detect non-Gaussianity
            detection = self.detector.detect(param_name)

            if detection is None:
                # Not enough history yet - use standard update
                corrected_gradients[param_name] = grad
                total_weights += grad.size
                total_standard += grad.size
            else:
                # Apply correction based on detection
                corrected = compute_sinkhorn_weight_correction(
                    grad,
                    detection.non_gaussian_mask,
                    self.config,
                )
                corrected_gradients[param_name] = corrected

                # Track statistics
                total_weights += grad.size
                if self.config.blend_mode == "hard":
                    n_sb = np.sum(detection.non_gaussian_mask)
                else:
                    n_sb = np.sum(detection.non_gaussian_mask > 0.5)
                total_sb += n_sb
                total_standard += grad.size - n_sb

                diagnostics_per_param[param_name] = {
                    "fraction_non_gaussian": detection.fraction_non_gaussian,
                    "mean_kurtosis": detection.mean_kurtosis,
                    "max_kurtosis": detection.max_kurtosis,
                }

        self._step_count += 1

        # Compute overall fractions
        if total_weights > 0:
            standard_frac = total_standard / total_weights
            sb_frac = total_sb / total_weights
        else:
            standard_frac = 1.0
            sb_frac = 0.0

        # Store stats periodically
        if self.config.track_statistics:
            if self._step_count % self.config.stats_update_every == 0:
                self._stats_history.append(
                    {
                        "step": self._step_count,
                        "standard_fraction": standard_frac,
                        "sb_fraction": sb_frac,
                        "num_params": len(gradients),
                    }
                )
                # Keep last 100 entries
                if len(self._stats_history) > 100:
                    self._stats_history = self._stats_history[-100:]

        return AdaptiveWeightUpdateResult(
            corrected_gradients=corrected_gradients,
            standard_update_fraction=standard_frac,
            sb_update_fraction=sb_frac,
            diagnostics={
                "per_param": diagnostics_per_param,
                "step": self._step_count,
            },
        )

    def get_summary_stats(self, param_names: List[str]) -> Dict[str, float]:
        """Get summary statistics for tracked parameters."""
        stats = self.detector.get_summary_stats(param_names)
        stats["step_count"] = self._step_count
        return stats

    def get_stats_history(self) -> List[Dict[str, float]]:
        """Get history of statistics over time."""
        return list(self._stats_history)

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "detector": self.detector.save_state(),
            "step_count": self._step_count,
            "stats_history": list(self._stats_history),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        if "detector" in state:
            self.detector.load_state(state["detector"])
        self._step_count = state.get("step_count", 0)
        self._stats_history = list(state.get("stats_history", []))


# ----------------------------
# JAX Integration Helpers
# ----------------------------


if HAS_JAX:

    def pytree_to_dict(params: Any, prefix: str = "") -> Dict[str, np.ndarray]:
        """
        Convert JAX pytree to flat dictionary of arrays.

        Args:
            params: JAX pytree (nested dict/tuple of arrays)
            prefix: Prefix for parameter names

        Returns:
            Flat dict mapping names to numpy arrays
        """
        result = {}

        if isinstance(params, dict):
            for key, value in params.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                result.update(pytree_to_dict(value, new_prefix))
        elif isinstance(params, (list, tuple)):
            for i, value in enumerate(params):
                new_prefix = f"{prefix}[{i}]"
                result.update(pytree_to_dict(value, new_prefix))
        elif hasattr(params, "__jax_array__") or isinstance(
            params, (np.ndarray, jnp.ndarray)
        ):
            result[prefix] = np.asarray(params)

        return result

    def dict_to_pytree(flat_dict: Dict[str, np.ndarray], template: Any) -> Any:
        """
        Convert flat dictionary back to JAX pytree structure.

        Args:
            flat_dict: Flat dict mapping names to arrays
            template: Original pytree structure for shape reference

        Returns:
            Reconstructed pytree
        """

        def _reconstruct(obj, prefix=""):
            if isinstance(obj, dict):
                return {
                    key: _reconstruct(value, f"{prefix}.{key}" if prefix else key)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [
                    _reconstruct(value, f"{prefix}[{i}]") for i, value in enumerate(obj)
                ]
            elif isinstance(obj, tuple):
                return tuple(
                    _reconstruct(value, f"{prefix}[{i}]") for i, value in enumerate(obj)
                )
            else:
                # Leaf node - get from flat dict
                if prefix in flat_dict:
                    return jnp.array(flat_dict[prefix])
                return obj

        return _reconstruct(template)


# ----------------------------
# Main Integration Class
# ----------------------------


class PerWeightCausalLearner:
    """
    Main integration class for per-weight causal learning.

    Wraps the adaptive weight updater and provides easy integration
    with FabricPC training loops.

    Usage:
        learner = PerWeightCausalLearner(config)

        # In training loop:
        gradients = compute_gradients(params, batch)
        result = learner.process_gradients(gradients)
        params = optimizer.apply(result.corrected_gradients, params)
    """

    def __init__(self, config: Optional[PerWeightCausalConfig] = None):
        if config is None:
            config = PerWeightCausalConfig()
        self.config = config
        self.updater = AdaptiveWeightUpdater(config)

    def process_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        is_bias: Optional[Dict[str, bool]] = None,
    ) -> AdaptiveWeightUpdateResult:
        """
        Process gradients through per-weight causal system.

        Args:
            gradients: Dict mapping param_name -> gradient array
            is_bias: Optional dict indicating which params are biases

        Returns:
            AdaptiveWeightUpdateResult with corrected gradients
        """
        if not self.config.enable:
            # Bypass - return original gradients
            return AdaptiveWeightUpdateResult(
                corrected_gradients=gradients,
                standard_update_fraction=1.0,
                sb_update_fraction=0.0,
                diagnostics={"bypassed": True},
            )

        return self.updater.update_and_correct(gradients, is_bias)

    def process_jax_gradients(
        self,
        gradients: Any,
        is_bias_fn: Optional[Callable[[str], bool]] = None,
    ) -> Tuple[Any, AdaptiveWeightUpdateResult]:
        """
        Process JAX pytree gradients through per-weight causal system.

        Args:
            gradients: JAX pytree of gradients
            is_bias_fn: Optional function to determine if param is bias

        Returns:
            Tuple of (corrected pytree, result with diagnostics)
        """
        if not HAS_JAX:
            raise ImportError("JAX is required for process_jax_gradients")

        if not self.config.enable:
            return gradients, AdaptiveWeightUpdateResult(
                corrected_gradients={},
                standard_update_fraction=1.0,
                sb_update_fraction=0.0,
                diagnostics={"bypassed": True},
            )

        # Convert to flat dict
        flat_grads = pytree_to_dict(gradients)

        # Determine bias flags
        is_bias = {}
        if is_bias_fn is not None:
            for name in flat_grads:
                is_bias[name] = is_bias_fn(name)

        # Process
        result = self.updater.update_and_correct(flat_grads, is_bias)

        # Convert back to pytree
        corrected_pytree = dict_to_pytree(result.corrected_gradients, gradients)

        return corrected_pytree, result

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        return self.updater.get_summary_stats(
            list(self.updater.detector.tracker._buffers.keys())
        )

    def save_state(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        return {
            "updater": self.updater.save_state(),
            "config": {
                "enable": self.config.enable,
                "gradient_history_size": self.config.gradient_history_size,
                "kurtosis_threshold": self.config.kurtosis_threshold,
                "sb_correction_strength": self.config.sb_correction_strength,
            },
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        if "updater" in state:
            self.updater.load_state(state["updater"])
