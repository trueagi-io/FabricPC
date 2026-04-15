"""
Benchmark configuration for continual learning evaluation.

Defines the evaluation protocol including dataset, training parameters,
and output settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkConfig:
    """
    Configuration for continual learning benchmark evaluation.

    This defines the evaluation protocol that will be applied
    to all models being benchmarked.

    Attributes:
        dataset_name: Name of registered dataset (e.g., "split-mnist")
        dataset_kwargs: Additional arguments for dataset initialization
        epochs_per_task: Number of training epochs per task
        batch_size: Batch size for training and evaluation
        num_runs: Number of independent runs with different seeds
        seeds: Explicit seeds for each run (auto-generated if None)
        save_results: Whether to save results to disk
        output_dir: Directory for saving results
        verbose: Whether to print progress during evaluation

    Example:
        >>> config = BenchmarkConfig(
        ...     dataset_name="split-mnist",
        ...     epochs_per_task=5,
        ...     num_runs=5,
        ... )
    """

    # Dataset configuration
    dataset_name: str = "split-mnist"
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    epochs_per_task: int = 5
    batch_size: int = 256

    # Evaluation configuration
    num_runs: int = 5
    seeds: Optional[List[int]] = None

    # Output configuration
    save_results: bool = True
    output_dir: str = "./benchmark_results"
    verbose: bool = True

    def __post_init__(self):
        """Generate seeds if not provided."""
        if self.seeds is None:
            self.seeds = list(range(self.num_runs))
        elif len(self.seeds) != self.num_runs:
            raise ValueError(
                f"Number of seeds ({len(self.seeds)}) must match "
                f"num_runs ({self.num_runs})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_kwargs": self.dataset_kwargs,
            "epochs_per_task": self.epochs_per_task,
            "batch_size": self.batch_size,
            "num_runs": self.num_runs,
            "seeds": self.seeds,
            "save_results": self.save_results,
            "output_dir": self.output_dir,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary."""
        return cls(**d)


# Preset configurations for common benchmarks
PRESET_CONFIGS = {
    "split-mnist-quick": BenchmarkConfig(
        dataset_name="split-mnist",
        epochs_per_task=1,
        num_runs=1,
        verbose=True,
    ),
    "split-mnist-standard": BenchmarkConfig(
        dataset_name="split-mnist",
        epochs_per_task=5,
        num_runs=5,
    ),
    "split-mnist-full": BenchmarkConfig(
        dataset_name="split-mnist",
        epochs_per_task=10,
        num_runs=10,
    ),
    "permuted-mnist-standard": BenchmarkConfig(
        dataset_name="permuted-mnist",
        dataset_kwargs={"num_tasks": 10},
        epochs_per_task=5,
        num_runs=5,
    ),
    "split-cifar10-standard": BenchmarkConfig(
        dataset_name="split-cifar10",
        epochs_per_task=10,
        num_runs=5,
    ),
}


def get_preset_config(name: str) -> BenchmarkConfig:
    """
    Get a preset benchmark configuration.

    Args:
        name: Name of preset (e.g., "split-mnist-standard")

    Returns:
        BenchmarkConfig for the preset

    Raises:
        ValueError: If preset name is not recognized
    """
    if name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: '{name}'. Available presets: {available}")

    return PRESET_CONFIGS[name]
