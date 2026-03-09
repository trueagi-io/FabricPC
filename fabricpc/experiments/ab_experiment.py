"""A/B experiment harness for comparing training methods and architectures.

Provides a reusable framework for running paired statistical comparisons
between two training configurations (different architectures, different
training algorithms, or both).

Example::

    from fabricpc.experiments import ExperimentArm, ABExperiment
    import optax

    optimizer = optax.adam(1e-3)

    arm_a = ExperimentArm(
        name="Lateral",
        model_factory=create_lateral_model,
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=pc_config,
    )
    arm_b = ExperimentArm(
        name="MLP",
        model_factory=create_mlp_model,
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=pc_config,
    )

    experiment = ABExperiment(
        arm_a=arm_a,
        arm_b=arm_b,
        metric="accuracy",
        data_loader_factory=make_loaders,
        n_trials=10,
    )
    results = experiment.run()
    results.print_summary()
"""

from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict, Any, List
import time

import numpy as np
import jax
import optax

from fabricpc.experiments.statistics import (
    descriptive_stats,
    paired_ttest,
    cohens_d,
    estimate_required_n,
)

# Type aliases
ModelFactory = Callable[[jax.Array], Tuple[Any, Any]]  # rng_key -> (params, structure)
TrainFn = Callable  # (params, structure, loader, config, rng_key, verbose=...) -> (params, history, epoch_results)
EvalFn = Callable  # (params, structure, loader, config, rng_key) -> dict
DataLoaderFactory = Callable[
    [int], Tuple[Any, Any]
]  # seed -> (train_loader, test_loader)


@dataclass(frozen=True)
class ExperimentArm:
    """One arm (condition) of an A/B experiment.

    Args:
        name: Human-readable name for this arm (e.g., "PC-sigmoid").
        model_factory: Callable taking a JAX rng_key and returning
            (GraphParams, GraphStructure). Called fresh each trial.
        train_fn: Training function with signature matching train_pcn.
        eval_fn: Evaluation function with signature matching evaluate_pcn.
        optimizer: Optax optimizer (e.g., optax.adam(1e-3)).
        train_config: Training configuration dict (scalar hyperparams only).
    """

    name: str
    model_factory: ModelFactory
    train_fn: TrainFn
    eval_fn: EvalFn
    optimizer: optax.GradientTransformation
    train_config: dict


@dataclass
class TrialResult:
    """Result of a single trial for one arm."""

    metric_value: float
    train_time: float
    all_metrics: Dict[str, float]


@dataclass
class ABResults:
    """Results from a completed A/B experiment.

    Holds per-trial data for both arms and provides analysis and reporting
    via :meth:`print_summary`.
    """

    arm_a_name: str
    arm_b_name: str
    metric: str
    n_trials: int
    arm_a_trials: List[TrialResult]
    arm_b_trials: List[TrialResult]
    seeds: List[int]
    total_time: float
    num_epochs: int

    @property
    def arm_a_metrics(self) -> np.ndarray:
        return np.array([t.metric_value for t in self.arm_a_trials])

    @property
    def arm_b_metrics(self) -> np.ndarray:
        return np.array([t.metric_value for t in self.arm_b_trials])

    @property
    def arm_a_times(self) -> np.ndarray:
        return np.array([t.train_time for t in self.arm_a_trials])

    @property
    def arm_b_times(self) -> np.ndarray:
        return np.array([t.train_time for t in self.arm_b_trials])

    def print_summary(self) -> None:
        """Print a complete ASCII summary of the experiment results."""
        a_vals = self.arm_a_metrics
        b_vals = self.arm_b_metrics

        # Detect rate metrics (0-1 range) for percentage display
        is_rate = (
            np.all(a_vals >= 0)
            and np.all(a_vals <= 1)
            and np.all(b_vals >= 0)
            and np.all(b_vals <= 1)
        )
        scale = 100.0 if is_rate else 1.0
        pct = "%" if is_rate else ""

        print("=" * 70)
        print(f"A/B Experiment: {self.arm_a_name} vs {self.arm_b_name}")
        print("=" * 70)
        # Training time comparison
        a_epoch_times = self.arm_a_times / self.num_epochs
        b_epoch_times = self.arm_b_times / self.num_epochs
        a_t = descriptive_stats(a_epoch_times)
        b_t = descriptive_stats(b_epoch_times)

        print()
        print("--- Training Time per Epoch ---")
        print(f"{self.arm_a_name}: {a_t.mean:.3f} +/- {a_t.se:.3f}s")
        print(f"{self.arm_b_name}: {b_t.mean:.3f} +/- {b_t.se:.3f}s")
        if b_t.mean > 0:
            print(
                f"Ratio: {self.arm_a_name} is "
                f"{a_t.mean / b_t.mean:.2f}x {self.arm_b_name} time"
            )

        print()
        print(f"Total wall time: {self.total_time:.1f}s")
        print("=" * 70)

        print(f"Metric: {self.metric}")
        print(f"Trials: {self.n_trials}")
        print(f"Epochs per trial: {self.num_epochs}")
        print(f"Design: Paired (same seed per trial)")
        print()

        # Per-trial table
        col_a = f"{self.arm_a_name}{pct}"
        col_b = f"{self.arm_b_name}{pct}"
        col_d = f"Diff{pct}"
        header = f"{'Trial':<8} {'Seed':<8} {col_a:<20} {col_b:<20} {col_d:<12}"
        print(header)
        print("-" * len(header))
        for i in range(self.n_trials):
            diff = (a_vals[i] - b_vals[i]) * scale
            print(
                f"{i+1:<8} {self.seeds[i]:<8} "
                f"{a_vals[i]*scale:<20.2f} "
                f"{b_vals[i]*scale:<20.2f} "
                f"{diff:<+12.2f}"
            )
        print("-" * len(header))

        # Descriptive stats
        a_stats = descriptive_stats(a_vals * scale)
        b_stats = descriptive_stats(b_vals * scale)
        print()
        print(
            f"{self.arm_a_name}: {a_stats.mean:.2f} +/- {a_stats.se:.2f}{pct}"
            f"  (mean +/- SE, SD={a_stats.std:.2f}{pct})"
        )
        print(
            f"{self.arm_b_name}: {b_stats.mean:.2f} +/- {b_stats.se:.2f}{pct}"
            f"  (mean +/- SE, SD={b_stats.std:.2f}{pct})"
        )

        # Statistical tests (require n >= 2)
        if self.n_trials >= 2:
            ttest = paired_ttest(a_vals, b_vals)
            print()
            print("--- Paired t-test ---")
            print(
                f"Mean difference ({self.arm_a_name} - {self.arm_b_name}): "
                f"{ttest.mean_difference * scale:+.2f}{pct}"
            )
            print(f"t-statistic: {ttest.t_statistic:.4f}")
            print(f"p-value: {ttest.p_value:.4f}, N = {ttest.n}")
            print(
                f"Significant at p<0.05: "
                f"{'YES' if ttest.significant_at_05 else 'NO'}"
            )

            effect = cohens_d(a_vals, b_vals)
            print()
            print("--- Effect Size ---")
            print(f"Cohen's d (paired): {effect.d:.4f}")
            print(f"Interpretation: {effect.magnitude}")

            req_n = estimate_required_n(effect.d)
            print()
            print("--- Power Analysis ---")
            print(f"Estimated trials needed for p<0.05 with 80% power: {req_n}")
            if req_n >= 999999:
                print("  -> Effect size is ~zero; no finite sample can detect it.")
            elif req_n <= self.n_trials:
                print(f"  -> Current n_trials ({self.n_trials}) is sufficient.")
            else:
                print(
                    f"  -> Current n_trials ({self.n_trials}) may be underpowered. "
                    f"Consider increasing to {req_n}."
                )
        else:
            print()
            print("--- Statistical tests require n_trials >= 2 ---")
            diff = float(np.mean(a_vals - b_vals))
            print(
                f"Mean difference ({self.arm_a_name} - {self.arm_b_name}): "
                f"{diff * scale:+.2f}{pct}"
            )


class ABExperiment:
    """Runner for paired A/B experiments.

    Runs N independent trials where each trial initializes fresh models
    for both arms using the same RNG seed (paired design), trains both,
    evaluates both, and records the specified metric.

    Args:
        arm_a: First experimental condition.
        arm_b: Second experimental condition.
        metric: Key in eval_fn's return dict to compare
            (e.g., "accuracy", "perplexity").
        data_loader_factory: Callable taking an int seed and returning
            (train_loader, test_loader).
        n_trials: Number of independent paired trials.
        seed_offset: Base seed offset. Trial i uses
            seed = seed_offset + i * 1000.
        verbose: If True, print per-epoch training output.
    """

    def __init__(
        self,
        arm_a: ExperimentArm,
        arm_b: ExperimentArm,
        metric: str,
        data_loader_factory: DataLoaderFactory,
        n_trials: int = 10,
        seed_offset: int = 0,
        verbose: bool = False,
    ):
        self.arm_a = arm_a
        self.arm_b = arm_b
        self.metric = metric
        self.data_loader_factory = data_loader_factory
        self.n_trials = n_trials
        self.seed_offset = seed_offset
        self.verbose = verbose

    def _run_arm_trial(
        self,
        arm: ExperimentArm,
        trial_seed: int,
        train_loader: Any,
        test_loader: Any,
    ) -> TrialResult:
        """Run a single trial for one arm."""
        master_key = jax.random.PRNGKey(trial_seed)
        graph_key, train_key, eval_key = jax.random.split(master_key, 3)

        params, structure = arm.model_factory(graph_key)

        t0 = time.time()
        trained_params, _, _ = arm.train_fn(
            params,
            structure,
            train_loader,
            arm.optimizer,
            arm.train_config,
            train_key,
            verbose=self.verbose,
        )
        train_time = time.time() - t0

        metrics = arm.eval_fn(
            trained_params,
            structure,
            test_loader,
            arm.train_config,
            eval_key,
        )

        if self.metric not in metrics:
            available = ", ".join(sorted(metrics.keys()))
            raise KeyError(
                f"Metric '{self.metric}' not found in eval results for arm "
                f"'{arm.name}'. Available: {available}"
            )

        return TrialResult(
            metric_value=float(metrics[self.metric]),
            train_time=train_time,
            all_metrics={k: float(v) for k, v in metrics.items()},
        )

    def run(self) -> ABResults:
        """Execute the full experiment: all trials for both arms.

        Returns:
            ABResults containing per-trial data and analysis methods.
        """
        arm_a_trials: List[TrialResult] = []
        arm_b_trials: List[TrialResult] = []
        seeds: List[int] = []

        num_epochs = self.arm_a.train_config.get(
            "num_epochs",
            self.arm_b.train_config.get("num_epochs", 1),
        )

        total_start = time.time()

        for trial_idx in range(self.n_trials):
            trial_seed = self.seed_offset + trial_idx * 1000
            seeds.append(trial_seed)

            print(
                f"--- Trial {trial_idx + 1}/{self.n_trials} " f"(seed={trial_seed}) ---"
            )

            # Arm A
            train_loader, test_loader = self.data_loader_factory(trial_seed)
            result_a = self._run_arm_trial(
                self.arm_a, trial_seed, train_loader, test_loader
            )
            arm_a_trials.append(result_a)
            print(
                f"  {self.arm_a.name}: {self.metric}="
                f"{result_a.metric_value:.4f}  "
                f"(train: {result_a.train_time:.1f}s)"
            )

            # Arm B (recreate loaders for isolation)
            train_loader, test_loader = self.data_loader_factory(trial_seed)
            result_b = self._run_arm_trial(
                self.arm_b, trial_seed, train_loader, test_loader
            )
            arm_b_trials.append(result_b)
            print(
                f"  {self.arm_b.name}: {self.metric}="
                f"{result_b.metric_value:.4f}  "
                f"(train: {result_b.train_time:.1f}s)"
            )

        total_time = time.time() - total_start

        return ABResults(
            arm_a_name=self.arm_a.name,
            arm_b_name=self.arm_b.name,
            metric=self.metric,
            n_trials=self.n_trials,
            arm_a_trials=arm_a_trials,
            arm_b_trials=arm_b_trials,
            seeds=seeds,
            total_time=total_time,
            num_epochs=num_epochs,
        )
