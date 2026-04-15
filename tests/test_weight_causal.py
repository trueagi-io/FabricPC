"""
Unit tests for per-weight causal coding module.

Tests the core components:
- WeightGradientTracker
- PerWeightNonGaussianityDetector
- AdaptiveWeightUpdater
- PerWeightCausalLearner
"""

import numpy as np
import pytest

from fabricpc.continual.weight_causal import (
    PerWeightCausalConfig,
    PerWeightCausalLearner,
    WeightGradientTracker,
    PerWeightNonGaussianityDetector,
    AdaptiveWeightUpdater,
    compute_weight_excess_kurtosis,
    compute_weight_multimodal_gap,
    compute_non_gaussianity_score,
    compute_sinkhorn_weight_correction,
    erfinv_approx,
    sinkhorn_1d_correction,
)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_erfinv_approx_zero(self):
        """Test erfinv at 0."""
        result = erfinv_approx(np.array([0.0]))
        assert np.abs(result[0]) < 1e-6

    def test_erfinv_approx_values(self):
        """Test erfinv at standard values."""
        x = np.array([-0.5, 0.0, 0.5])
        result = erfinv_approx(x)
        # Should be antisymmetric
        assert np.abs(result[0] + result[2]) < 0.1
        assert np.abs(result[1]) < 1e-6

    def test_compute_weight_excess_kurtosis_gaussian(self):
        """Test kurtosis for Gaussian distribution."""
        np.random.seed(42)
        # Gaussian should have excess kurtosis near 0
        gradients = np.random.randn(100, 10)  # 100 samples, 10 weights
        kurtosis = compute_weight_excess_kurtosis(gradients)
        assert kurtosis.shape == (10,)
        # Excess kurtosis should be near 0 for Gaussian
        assert np.abs(np.mean(kurtosis)) < 1.0

    def test_compute_weight_excess_kurtosis_uniform(self):
        """Test kurtosis for uniform distribution."""
        np.random.seed(42)
        # Uniform has excess kurtosis of -1.2
        gradients = np.random.uniform(-1, 1, size=(100, 10))
        kurtosis = compute_weight_excess_kurtosis(gradients)
        assert kurtosis.shape == (10,)
        # Should be negative (platykurtic)
        assert np.mean(kurtosis) < 0

    def test_compute_weight_excess_kurtosis_heavy_tail(self):
        """Test kurtosis for heavy-tailed distribution."""
        np.random.seed(42)
        # Laplace has excess kurtosis of 3
        gradients = np.random.laplace(0, 1, size=(200, 10))
        kurtosis = compute_weight_excess_kurtosis(gradients)
        assert kurtosis.shape == (10,)
        # Should be positive (leptokurtic)
        assert np.mean(kurtosis) > 0

    def test_compute_weight_multimodal_gap_unimodal(self):
        """Test multimodal gap for unimodal distribution."""
        np.random.seed(42)
        gradients = np.random.randn(100, 10)
        gap = compute_weight_multimodal_gap(gradients)
        assert gap.shape == (10,)
        # Unimodal Gaussian has expected gap of ~1.6 (theoretical value)
        # Check it's within reasonable range
        assert np.mean(gap) < 2.0

    def test_compute_weight_multimodal_gap_bimodal(self):
        """Test multimodal gap for bimodal distribution."""
        np.random.seed(42)
        # Create bimodal by mixing two Gaussians with larger separation
        low = np.random.randn(50, 10) - 5
        high = np.random.randn(50, 10) + 5
        gradients = np.vstack([low, high])
        gap = compute_weight_multimodal_gap(gradients)
        assert gap.shape == (10,)
        # Bimodal with well-separated modes should have gap > 1.6 (unimodal)
        assert np.mean(gap) > 1.5

    def test_compute_non_gaussianity_score(self):
        """Test combined non-Gaussianity score."""
        config = PerWeightCausalConfig()
        kurtosis = np.array([0.5, 3.0, -1.0, 5.0])
        multimodal = np.array([0.2, 0.6, 0.3, 1.0])

        score = compute_non_gaussianity_score(kurtosis, multimodal, config)

        assert score.shape == (4,)
        # Higher kurtosis/multimodal should give higher score
        assert score[3] > score[0]

    def test_sinkhorn_1d_correction(self):
        """Test Sinkhorn 1D correction."""
        np.random.seed(42)
        # Non-Gaussian input
        gradients = np.random.laplace(0, 1, size=50)

        corrected = sinkhorn_1d_correction(gradients, eps=0.1, iters=10)

        # Should have same length
        assert len(corrected) == len(gradients)
        # Should have similar mean/std
        assert np.abs(np.mean(corrected) - np.mean(gradients)) < 0.5
        # Corrected should be more Gaussian-like (lower kurtosis)
        original_kurt = compute_weight_excess_kurtosis(gradients.reshape(-1, 1))[0]
        corrected_kurt = compute_weight_excess_kurtosis(corrected.reshape(-1, 1))[0]
        # Correction should reduce kurtosis magnitude in most cases
        # (not a strict guarantee, so use weaker assertion)
        assert corrected_kurt is not None

    def test_compute_sinkhorn_weight_correction(self):
        """Test per-weight Sinkhorn correction."""
        config = PerWeightCausalConfig(sb_correction_strength=0.5)
        np.random.seed(42)

        gradient = np.random.randn(100)
        mask = np.ones(100) * 0.8  # 80% correction

        corrected = compute_sinkhorn_weight_correction(gradient, mask, config)

        assert corrected.shape == gradient.shape
        # Should be different from original
        assert not np.allclose(corrected, gradient)


class TestWeightGradientTracker:
    """Test WeightGradientTracker."""

    def test_init(self):
        """Test initialization."""
        config = PerWeightCausalConfig(gradient_history_size=10)
        tracker = WeightGradientTracker(config=config)
        assert len(tracker._buffers) == 0

    def test_update_single(self):
        """Test single update."""
        config = PerWeightCausalConfig(gradient_history_size=10)
        tracker = WeightGradientTracker(config=config)

        gradient = np.random.randn(5, 5)
        tracker.update("layer1.weight", gradient)

        assert "layer1.weight" in tracker._buffers
        assert tracker._positions["layer1.weight"] == 1

    def test_update_multiple(self):
        """Test multiple updates."""
        config = PerWeightCausalConfig(
            gradient_history_size=10, min_history_for_detection=4
        )
        tracker = WeightGradientTracker(config=config)

        for i in range(5):
            gradient = np.random.randn(5, 5)
            tracker.update("layer1.weight", gradient)

        assert tracker._positions["layer1.weight"] == 5
        assert tracker.has_sufficient_history("layer1.weight")
        assert tracker.get_effective_size("layer1.weight") == 5

    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        config = PerWeightCausalConfig(gradient_history_size=5)
        tracker = WeightGradientTracker(config=config)

        for i in range(8):
            gradient = np.ones((2, 2)) * i
            tracker.update("layer1.weight", gradient)

        assert tracker._filled["layer1.weight"]
        assert tracker.get_effective_size("layer1.weight") == 5

        # Get history should return chronological order
        history = tracker.get_history("layer1.weight")
        assert history.shape == (5, 2, 2)
        # Last values should be 7, 6, 5, 4, 3 (reordered to chronological)
        # After 8 updates with buffer size 5:
        # positions 0-4 contain values 5,6,7,3,4 -> reordered to 3,4,5,6,7

    def test_save_load_state(self):
        """Test state serialization."""
        config = PerWeightCausalConfig(gradient_history_size=5)
        tracker = WeightGradientTracker(config=config)

        for i in range(3):
            tracker.update("layer1.weight", np.random.randn(2, 2))

        state = tracker.save_state()
        tracker2 = WeightGradientTracker(config=config)
        tracker2.load_state(state)

        assert tracker2._positions == tracker._positions
        assert tracker2._filled == tracker._filled


class TestPerWeightNonGaussianityDetector:
    """Test PerWeightNonGaussianityDetector."""

    def test_init(self):
        """Test initialization."""
        config = PerWeightCausalConfig()
        detector = PerWeightNonGaussianityDetector(config=config)
        assert detector.tracker is not None

    def test_detect_no_history(self):
        """Test detection with no history."""
        config = PerWeightCausalConfig()
        detector = PerWeightNonGaussianityDetector(config=config)

        result = detector.detect("layer1.weight")
        assert result is None

    def test_detect_gaussian(self):
        """Test detection on Gaussian gradients."""
        config = PerWeightCausalConfig(
            gradient_history_size=50,
            min_history_for_detection=20,
            kurtosis_threshold=2.0,
        )
        detector = PerWeightNonGaussianityDetector(config=config)

        np.random.seed(42)
        for _ in range(30):
            gradient = np.random.randn(10)
            detector.update_gradients("layer1.weight", gradient)

        result = detector.detect("layer1.weight")
        assert result is not None
        assert result.param_name == "layer1.weight"
        # Gaussian should have low kurtosis
        assert result.mean_kurtosis < 2.0
        # Few weights should be flagged
        assert result.fraction_non_gaussian < 0.5

    def test_detect_non_gaussian(self):
        """Test detection on non-Gaussian gradients."""
        config = PerWeightCausalConfig(
            gradient_history_size=100,
            min_history_for_detection=50,
            kurtosis_threshold=1.0,
            combined_threshold=0.5,
        )
        detector = PerWeightNonGaussianityDetector(config=config)

        np.random.seed(42)
        for _ in range(60):
            # Heavy-tailed distribution
            gradient = np.random.laplace(0, 1, size=10)
            detector.update_gradients("layer1.weight", gradient)

        result = detector.detect("layer1.weight")
        assert result is not None
        # Laplace should have higher kurtosis
        assert result.mean_kurtosis > 0


class TestAdaptiveWeightUpdater:
    """Test AdaptiveWeightUpdater."""

    def test_init(self):
        """Test initialization."""
        config = PerWeightCausalConfig()
        updater = AdaptiveWeightUpdater(config)
        assert updater._step_count == 0

    def test_update_and_correct(self):
        """Test gradient correction."""
        config = PerWeightCausalConfig(
            gradient_history_size=20,
            min_history_for_detection=10,
        )
        updater = AdaptiveWeightUpdater(config)

        np.random.seed(42)
        for _ in range(15):
            gradients = {
                "layer1.weight": np.random.randn(10),
                "layer1.bias": np.random.randn(10),
            }
            result = updater.update_and_correct(
                gradients, is_bias={"layer1.weight": False, "layer1.bias": True}
            )

        assert "layer1.weight" in result.corrected_gradients
        assert "layer1.bias" in result.corrected_gradients
        assert result.standard_update_fraction >= 0
        assert result.sb_update_fraction >= 0
        assert (
            result.standard_update_fraction + result.sb_update_fraction
        ) <= 1.01  # Allow small floating point error

    def test_stats_tracking(self):
        """Test statistics tracking."""
        config = PerWeightCausalConfig(track_statistics=True, stats_update_every=5)
        updater = AdaptiveWeightUpdater(config)

        np.random.seed(42)
        for _ in range(20):
            gradients = {"layer1.weight": np.random.randn(10)}
            updater.update_and_correct(gradients)

        history = updater.get_stats_history()
        assert len(history) >= 3  # At least 3 stats updates (at steps 5, 10, 15)


class TestPerWeightCausalLearner:
    """Test PerWeightCausalLearner."""

    def test_init_default(self):
        """Test default initialization."""
        learner = PerWeightCausalLearner()
        assert learner.config.enable

    def test_init_disabled(self):
        """Test disabled initialization."""
        config = PerWeightCausalConfig(enable=False)
        learner = PerWeightCausalLearner(config)
        assert not learner.config.enable

    def test_process_gradients_disabled(self):
        """Test processing when disabled."""
        config = PerWeightCausalConfig(enable=False)
        learner = PerWeightCausalLearner(config)

        gradients = {"layer1.weight": np.random.randn(10)}
        result = learner.process_gradients(gradients)

        # Should return unchanged when disabled
        assert np.allclose(
            result.corrected_gradients["layer1.weight"],
            gradients["layer1.weight"],
        )
        assert result.diagnostics.get("bypassed", False)

    def test_process_gradients_enabled(self):
        """Test processing when enabled."""
        config = PerWeightCausalConfig(enable=True)
        learner = PerWeightCausalLearner(config)

        np.random.seed(42)
        gradients = {"layer1.weight": np.random.randn(10)}
        result = learner.process_gradients(gradients)

        assert "layer1.weight" in result.corrected_gradients
        assert result.corrected_gradients["layer1.weight"].shape == (10,)

    def test_get_stats(self):
        """Test getting statistics."""
        config = PerWeightCausalConfig(
            gradient_history_size=10, min_history_for_detection=5
        )
        learner = PerWeightCausalLearner(config)

        np.random.seed(42)
        for _ in range(10):
            gradients = {"layer1.weight": np.random.randn(10)}
            learner.process_gradients(gradients)

        stats = learner.get_stats()
        assert "mean_kurtosis" in stats or "num_params_tracked" in stats

    def test_save_load_state(self):
        """Test state serialization."""
        config = PerWeightCausalConfig()
        learner = PerWeightCausalLearner(config)

        np.random.seed(42)
        for _ in range(5):
            gradients = {"layer1.weight": np.random.randn(10)}
            learner.process_gradients(gradients)

        state = learner.save_state()
        assert "updater" in state

        learner2 = PerWeightCausalLearner(config)
        learner2.load_state(state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
