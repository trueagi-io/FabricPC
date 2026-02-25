#!/usr/bin/env python3
"""
Tests for dashboarding tracker utility methods that do not require Aim installed.
"""

from fabricpc.utils.dashboarding.trackers import AimExperimentTracker, TrackingConfig


class DummyRun:
    """Minimal Aim Run stub for unit testing."""

    def __init__(self):
        self.calls = []

    def track(self, value, name, step=None, epoch=None, context=None):
        self.calls.append(
            {
                "value": value,
                "name": name,
                "step": step,
                "epoch": epoch,
                "context": context,
            }
        )

    def close(self):
        return None


def test_track_inference_dynamics_from_history_logs_expected_metrics():
    """Tracker should emit energy/grad/error metrics with expected step scaling."""
    config = TrackingConfig(
        track_inference_dynamics=True,
        inference_nodes_to_track=["h1"],
    )
    tracker = AimExperimentTracker(config=config)
    tracker._run = DummyRun()
    tracker._initialized = True

    inference_history = [
        {"h1": {"energy": 1.0, "latent_grad_norm": 0.5, "error_norm": 0.25}},
        {"h1": {"energy": 0.7, "latent_grad_norm": 0.2, "error_norm": 0.1}},
    ]

    tracker.track_inference_dynamics_from_history(
        inference_history,
        epoch=2,
        batch=4,
        collect_every=5,
    )

    calls = tracker._run.calls
    assert len(calls) == 6  # 2 steps * 3 metrics

    names = [c["name"] for c in calls]
    assert names == [
        "inference_energy",
        "inference_grad_norm",
        "inference_error_norm",
        "inference_energy",
        "inference_grad_norm",
        "inference_error_norm",
    ]

    assert calls[0]["step"] == 0
    assert calls[3]["step"] == 5
    assert calls[0]["context"]["node"] == "h1"
    assert calls[0]["context"]["epoch"] == 2
    assert calls[0]["context"]["batch"] == 4
