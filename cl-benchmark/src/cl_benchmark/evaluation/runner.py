"""
Benchmark runner for continual learning evaluation.

Orchestrates the training and evaluation loop according
to the evaluation protocol.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import numpy as np

from cl_benchmark.datasets import get_dataset
from cl_benchmark.evaluation.protocol import BenchmarkConfig
from cl_benchmark.evaluation.results import BenchmarkResults
from cl_benchmark.metrics import (
    compute_accuracy_matrix,
    compute_forgetting,
    compute_backward_transfer,
    compute_average_accuracy,
)

if TYPE_CHECKING:
    from cl_benchmark.protocols import ContinualModel, ContinualDataset


class BenchmarkRunner:
    """
    Runs continual learning benchmark evaluation.

    Executes the evaluation protocol:
    1. For each run (with different random seed):
       a. Initialize/reset model
       b. For each task in sequence:
          - Train on current task
          - Evaluate on all tasks
          - Record accuracy matrix row
    2. Compute metrics across runs
    3. Return BenchmarkResults

    Example:
        >>> from cl_benchmark import BenchmarkRunner, BenchmarkConfig
        >>>
        >>> config = BenchmarkConfig(
        ...     dataset_name="split-mnist",
        ...     epochs_per_task=5,
        ...     num_runs=5,
        ... )
        >>> runner = BenchmarkRunner(config)
        >>>
        >>> # Your model must implement ContinualModel protocol
        >>> results = runner.evaluate(my_model)
        >>> results.print_summary()
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._dataset = None

    @property
    def dataset(self) -> "ContinualDataset":
        """Lazily load dataset."""
        if self._dataset is None:
            self._dataset = get_dataset(
                self.config.dataset_name,
                **self.config.dataset_kwargs,
            )
        return self._dataset

    def evaluate(
        self,
        model: "ContinualModel",
        model_factory: Optional[Callable[[], "ContinualModel"]] = None,
        model_name: str = "model",
    ) -> BenchmarkResults:
        """
        Run full benchmark evaluation.

        For multiple runs, you should provide model_factory to create
        fresh model instances for each run. If only model is provided,
        it will be used for a single run.

        Args:
            model: Model to evaluate (used if model_factory is None)
            model_factory: Optional factory function to create fresh models
            model_name: Name for the model in results

        Returns:
            BenchmarkResults containing all metrics and raw data
        """
        results = BenchmarkResults(
            config=self.config,
            model_name=model_name,
        )

        for run_idx, seed in enumerate(self.config.seeds):
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"Run {run_idx + 1}/{self.config.num_runs} (seed={seed})")
                print(f"{'='*60}")

            # Get model for this run
            if model_factory is not None:
                run_model = model_factory()
            else:
                run_model = model
                if run_idx > 0:
                    print(
                        "Warning: Using same model for multiple runs. "
                        "Consider providing model_factory for independent runs."
                    )

            # Run single evaluation
            matrix, times = self._evaluate_single_run(run_model, seed)

            results.accuracy_matrices.append(matrix)
            results.training_times.append(times)

        # Compute summary metrics
        results.compute_summary_metrics()

        # Save if configured
        if self.config.save_results:
            self._save_results(results)

        return results

    def _evaluate_single_run(
        self,
        model: "ContinualModel",
        seed: int,
    ) -> tuple[np.ndarray, list[float]]:
        """
        Run evaluation for a single seed.

        Args:
            model: Model to evaluate
            seed: Random seed for this run

        Returns:
            Tuple of (accuracy_matrix, training_times)
        """
        num_tasks = self.dataset.num_tasks
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        training_times = []

        # Set numpy random seed
        np.random.seed(seed)

        for task_idx, task in enumerate(self.dataset):
            if self.config.verbose:
                print(f"\nTask {task_idx}: classes {task.classes}")

            # Train on current task
            start_time = time.time()
            model.train_on_task(
                task_id=task_idx,
                train_data=task,
                epochs=self.config.epochs_per_task,
            )
            train_time = time.time() - start_time
            training_times.append(train_time)

            if self.config.verbose:
                print(f"  Training time: {train_time:.2f}s")

            # Evaluate on all tasks
            row = compute_accuracy_matrix(
                model,
                self.dataset,
                task_idx + 1,
                self.config.batch_size,
            )
            accuracy_matrix[task_idx] = row

            if self.config.verbose:
                print(f"  Accuracies: {row[:task_idx+1]}")

        return accuracy_matrix, training_times

    def _save_results(self, results: BenchmarkResults) -> None:
        """Save results to output directory."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with model name and timestamp
        filename = f"{results.model_name}_{results.timestamp.replace(':', '-')}.json"
        filepath = output_dir / filename

        results.to_json(str(filepath))

        if self.config.verbose:
            print(f"\nResults saved to: {filepath}")

    def compare(
        self,
        models: Dict[str, "ContinualModel"],
        model_factories: Optional[Dict[str, Callable[[], "ContinualModel"]]] = None,
    ) -> Dict[str, BenchmarkResults]:
        """
        Compare multiple models on the benchmark.

        Args:
            models: Dictionary mapping model names to model instances
            model_factories: Optional dictionary mapping names to factory functions

        Returns:
            Dictionary mapping model names to their BenchmarkResults
        """
        all_results = {}

        for name, model in models.items():
            if self.config.verbose:
                print(f"\n{'#'*60}")
                print(f"Evaluating: {name}")
                print(f"{'#'*60}")

            factory = model_factories.get(name) if model_factories else None

            results = self.evaluate(
                model=model,
                model_factory=factory,
                model_name=name,
            )
            all_results[name] = results

        # Print comparison
        if self.config.verbose:
            self._print_comparison(all_results)

        return all_results

    def _print_comparison(self, results: Dict[str, BenchmarkResults]) -> None:
        """Print comparison table."""
        print("\n" + "=" * 80)
        print("Comparison Summary")
        print("=" * 80)
        print(f"{'Model':<20} {'Avg Acc':<15} {'Forgetting':<15} {'BWT':<15}")
        print("-" * 80)

        # Sort by accuracy
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].accuracy_mean,
            reverse=True,
        )

        for name, res in sorted_results:
            acc = f"{res.accuracy_mean:.4f}±{res.accuracy_std:.4f}"
            fgt = f"{res.forgetting_mean:.4f}±{res.forgetting_std:.4f}"
            bwt = f"{res.backward_transfer:.4f}"
            print(f"{name:<20} {acc:<15} {fgt:<15} {bwt:<15}")

        print("=" * 80)
