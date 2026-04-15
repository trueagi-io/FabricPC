"""
Results container for continual learning benchmarks.

Stores all outputs from a benchmark evaluation including
accuracy matrices, metrics, and configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cl_benchmark.evaluation.protocol import BenchmarkConfig


@dataclass
class BenchmarkResults:
    """
    Container for benchmark evaluation results.

    Stores raw data (accuracy matrices) and computed metrics
    for a single model evaluated on a benchmark.

    Attributes:
        config: Configuration used for the benchmark
        accuracy_matrices: List of accuracy matrices (one per run)
        training_times: Training time per task per run
        forgetting: Per-task forgetting (averaged across runs)
        backward_transfer: BWT value (averaged)
        forward_transfer: FWT value (averaged)
        average_accuracy: Final average accuracy (averaged)
        accuracy_mean: Mean of average accuracy across runs
        accuracy_std: Std of average accuracy across runs
        forgetting_mean: Mean forgetting across runs
        forgetting_std: Std forgetting across runs
        model_name: Optional name of the model
        timestamp: When the benchmark was run
    """

    config: BenchmarkConfig
    accuracy_matrices: List[np.ndarray] = field(default_factory=list)
    training_times: List[List[float]] = field(default_factory=list)

    # Computed metrics (populated after evaluation)
    forgetting: np.ndarray = field(default_factory=lambda: np.array([]))
    backward_transfer: float = 0.0
    forward_transfer: float = 0.0
    average_accuracy: float = 0.0

    # Aggregated statistics
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    forgetting_mean: float = 0.0
    forgetting_std: float = 0.0

    # Metadata
    model_name: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_summary_metrics(self):
        """Compute summary metrics from raw accuracy matrices."""
        if not self.accuracy_matrices:
            return

        from cl_benchmark.metrics import (
            compute_forgetting,
            compute_backward_transfer,
            compute_average_accuracy,
        )

        # Compute per-run metrics
        avg_accs = []
        forgettings = []
        bwts = []

        for matrix in self.accuracy_matrices:
            avg_accs.append(compute_average_accuracy(matrix))
            forgettings.append(compute_forgetting(matrix))
            bwts.append(compute_backward_transfer(matrix))

        # Aggregate
        self.average_accuracy = float(np.mean(avg_accs))
        self.accuracy_mean = float(np.mean(avg_accs))
        self.accuracy_std = float(np.std(avg_accs))

        # Stack forgetting arrays and compute mean
        if forgettings:
            stacked_forgetting = np.stack(forgettings, axis=0)
            self.forgetting = np.mean(stacked_forgetting, axis=0)
            # Average forgetting (excluding last task)
            avg_fgt = [np.mean(f[:-1]) if len(f) > 1 else 0.0 for f in forgettings]
            self.forgetting_mean = float(np.mean(avg_fgt))
            self.forgetting_std = float(np.std(avg_fgt))

        self.backward_transfer = float(np.mean(bwts))

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "accuracy_matrices": [m.tolist() for m in self.accuracy_matrices],
            "training_times": self.training_times,
            "forgetting": self.forgetting.tolist() if len(self.forgetting) > 0 else [],
            "backward_transfer": self.backward_transfer,
            "forward_transfer": self.forward_transfer,
            "average_accuracy": self.average_accuracy,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "forgetting_mean": self.forgetting_mean,
            "forgetting_std": self.forgetting_std,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResults":
        """Create results from dictionary."""
        config = BenchmarkConfig.from_dict(d["config"])
        results = cls(config=config)

        results.accuracy_matrices = [np.array(m) for m in d["accuracy_matrices"]]
        results.training_times = d.get("training_times", [])
        results.forgetting = np.array(d.get("forgetting", []))
        results.backward_transfer = d.get("backward_transfer", 0.0)
        results.forward_transfer = d.get("forward_transfer", 0.0)
        results.average_accuracy = d.get("average_accuracy", 0.0)
        results.accuracy_mean = d.get("accuracy_mean", 0.0)
        results.accuracy_std = d.get("accuracy_std", 0.0)
        results.forgetting_mean = d.get("forgetting_mean", 0.0)
        results.forgetting_std = d.get("forgetting_std", 0.0)
        results.model_name = d.get("model_name", "unknown")
        results.timestamp = d.get("timestamp", "")

        return results

    def to_json(self, path: str) -> None:
        """Save results to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkResults":
        """Load results from JSON file."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def print_summary(self) -> None:
        """Print a summary of the benchmark results."""
        print("=" * 60)
        print(f"Benchmark Results: {self.model_name}")
        print("=" * 60)
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Epochs per task: {self.config.epochs_per_task}")
        print(f"Number of runs: {len(self.accuracy_matrices)}")
        print("-" * 60)
        print(f"Average Accuracy: {self.accuracy_mean:.4f} +/- {self.accuracy_std:.4f}")
        print(
            f"Average Forgetting: {self.forgetting_mean:.4f} +/- {self.forgetting_std:.4f}"
        )
        print(f"Backward Transfer: {self.backward_transfer:.4f}")
        print("=" * 60)

        # Per-task final accuracy
        if self.accuracy_matrices:
            print("\nPer-task Final Accuracy:")
            mean_matrix = np.mean(self.accuracy_matrices, axis=0)
            final_accs = mean_matrix[-1, :]
            for i, acc in enumerate(final_accs):
                print(f"  Task {i}: {acc:.4f}")

    def get_mean_accuracy_matrix(self) -> np.ndarray:
        """Get mean accuracy matrix across all runs."""
        if not self.accuracy_matrices:
            return np.array([])
        return np.mean(self.accuracy_matrices, axis=0)

    def get_std_accuracy_matrix(self) -> np.ndarray:
        """Get std accuracy matrix across all runs."""
        if not self.accuracy_matrices:
            return np.array([])
        return np.std(self.accuracy_matrices, axis=0)
