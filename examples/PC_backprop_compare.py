"""
Statistical Comparison: Predictive Coding vs Backpropagation on MNIST
=====================================================================

Runs multiple independent training trials for both PC and backprop,
then performs statistical analysis to compare test accuracies.

Reports:
- Per-trial accuracy results table
- Mean +/- standard error for each method
- Paired t-test with p-value
- Cohen's d effect size
- Power analysis: estimated n_trials needed for significance

Usage:
    python examples/PC_backprop_compare.py                # 10 trials (default)
    python examples/PC_backprop_compare.py --n_trials 20  # 20 trials
    python examples/PC_backprop_compare.py --verbose       # show per-epoch output
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import importlib.util
import sys
import time

import numpy as np
import jax
from scipy import stats

from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    SigmoidActivation,
    SoftmaxActivation,
    ReLUActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.training.train_backprop import train_backprop, evaluate_backprop
from fabricpc.utils.data.dataloader import MnistLoader

# Import train_config and batch_size from mnist_demo without triggering examples/__init__.py
_demo_path = os.path.join(os.path.dirname(__file__), "mnist_demo.py")
_spec = importlib.util.spec_from_file_location("mnist_demo", _demo_path)
_mnist_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mnist_demo)
train_config = _mnist_demo.train_config
batch_size = _mnist_demo.batch_size

jax.config.update("jax_default_prng_impl", "threefry2x32")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of PC vs Backprop on MNIST"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of independent training trials per method (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output for each trial",
    )
    return parser.parse_args()


def create_pc_model(rng_key):
    """Create PC model with sigmoid activations."""
    pixels = Linear(shape=(784,), name="pixels")
    hidden1 = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1")
    hidden2 = Linear(shape=(64,), activation=SigmoidActivation(), name="hidden2")
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
    )
    structure = graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def create_backprop_model(rng_key):
    """Create backprop model with ReLU activations (to avoid vanishing gradients)."""
    pixels = Linear(shape=(784,), name="pixels")
    hidden1 = Linear(shape=(256,), activation=ReLUActivation(), name="hidden1")
    hidden2 = Linear(shape=(64,), activation=ReLUActivation(), name="hidden2")
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
    )
    structure = graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def run_single_trial(method, trial_seed, verbose=False):
    """Run a single training trial and return test accuracy and training time.

    Args:
        method: "pc" or "backprop"
        trial_seed: Random seed for this trial
        verbose: Whether to print per-epoch progress

    Returns:
        (accuracy, train_time) where accuracy is in [0, 1] and
        train_time is wall-clock seconds for training.
    """
    master_key = jax.random.PRNGKey(trial_seed)
    graph_key, train_key, eval_key = jax.random.split(master_key, 3)

    train_loader = MnistLoader(
        "train",
        batch_size=batch_size,
        tensor_format="flat",
        shuffle=True,
        seed=trial_seed,
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, tensor_format="flat", shuffle=False
    )

    if method == "pc":
        params, structure = create_pc_model(graph_key)
        t0 = time.time()
        trained_params, _, _ = train_pcn(
            params, structure, train_loader, train_config, train_key, verbose=verbose
        )
        train_time = time.time() - t0
        metrics = evaluate_pcn(
            trained_params, structure, test_loader, train_config, eval_key
        )
    elif method == "backprop":
        params, structure = create_backprop_model(graph_key)
        t0 = time.time()
        trained_params, _, _ = train_backprop(
            params, structure, train_loader, train_config, train_key, verbose=verbose
        )
        train_time = time.time() - t0
        metrics = evaluate_backprop(
            trained_params, structure, test_loader, train_config, eval_key
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(metrics["accuracy"]), train_time


def compute_cohens_d(group1, group2):
    """Compute Cohen's d for paired samples.

    Uses the standard deviation of the differences as the denominator,
    which is appropriate for paired designs.
    """
    diff = group1 - group2
    return np.mean(diff) / np.std(diff, ddof=1)


def estimate_required_n(observed_d, alpha=0.05, power=0.80):
    """Estimate required sample size for a paired t-test.

    Uses the approximation: n = ((z_alpha + z_beta) / d)^2

    Args:
        observed_d: Observed Cohen's d (effect size)
        alpha: Significance level (two-sided)
        power: Desired statistical power

    Returns:
        Estimated number of paired trials needed, or inf if effect is ~zero
    """
    if abs(observed_d) < 1e-10:
        return float("inf")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / observed_d) ** 2
    return int(np.ceil(n))


def main():
    args = parse_args()
    n_trials = args.n_trials

    print("=" * 70)
    print("Statistical Comparison: Predictive Coding vs Backpropagation")
    print("=" * 70)
    print(f"Dataset: MNIST")
    print(f"Architecture: 784 -> 256 -> 64 -> 10")
    print(f"PC activations: sigmoid | Backprop activations: relu")
    print(f"Epochs per trial: {train_config['num_epochs']}")
    print(f"Number of trials: {n_trials}")
    print(f"Paired design: Yes (same seed for PC and backprop per trial)")
    print()

    pc_accuracies = np.zeros(n_trials)
    bp_accuracies = np.zeros(n_trials)
    pc_train_times = np.zeros(n_trials)
    bp_train_times = np.zeros(n_trials)
    num_epochs = train_config["num_epochs"]

    total_start = time.time()

    for trial in range(n_trials):
        trial_seed = trial * 1000
        print(f"--- Trial {trial + 1}/{n_trials} (seed={trial_seed}) ---")

        pc_acc, pc_tt = run_single_trial("pc", trial_seed, verbose=args.verbose)
        pc_accuracies[trial] = pc_acc
        pc_train_times[trial] = pc_tt
        print(
            f"  PC:       {pc_acc * 100:.2f}%  (train: {pc_tt:.1f}s, {pc_tt/num_epochs:.2f}s/epoch)"
        )

        bp_acc, bp_tt = run_single_trial("backprop", trial_seed, verbose=args.verbose)
        bp_accuracies[trial] = bp_acc
        bp_train_times[trial] = bp_tt
        print(
            f"  Backprop: {bp_acc * 100:.2f}%  (train: {bp_tt:.1f}s, {bp_tt/num_epochs:.2f}s/epoch)"
        )

    total_time = time.time() - total_start

    # ================================================================
    # Results Table
    # ================================================================
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Trial':<8} {'PC Acc (%)':<14} {'BP Acc (%)':<14} {'Diff (%)':<12}")
    print("-" * 48)
    for i in range(n_trials):
        diff = (pc_accuracies[i] - bp_accuracies[i]) * 100
        print(
            f"{i+1:<8} {pc_accuracies[i]*100:<14.2f} "
            f"{bp_accuracies[i]*100:<14.2f} {diff:<+12.2f}"
        )
    print("-" * 48)

    # ================================================================
    # Summary Statistics
    # ================================================================
    pc_mean = np.mean(pc_accuracies) * 100
    bp_mean = np.mean(bp_accuracies) * 100
    pc_se = np.std(pc_accuracies, ddof=1) / np.sqrt(n_trials) * 100
    bp_se = np.std(bp_accuracies, ddof=1) / np.sqrt(n_trials) * 100
    pc_sd = np.std(pc_accuracies, ddof=1) * 100
    bp_sd = np.std(bp_accuracies, ddof=1) * 100

    print()
    print(f"PC:       {pc_mean:.2f} +/- {pc_se:.2f}%  (mean +/- SE, SD={pc_sd:.2f}%)")
    print(f"Backprop: {bp_mean:.2f} +/- {bp_se:.2f}%  (mean +/- SE, SD={bp_sd:.2f}%)")

    # ================================================================
    # Statistical Test: Paired t-test
    # ================================================================
    differences = pc_accuracies - bp_accuracies

    if n_trials < 2:
        print()
        print("--- Statistical tests require n_trials >= 2 ---")
        print(f"Mean difference (PC - BP): {np.mean(differences)*100:+.2f}%")
    else:
        t_stat, p_value = stats.ttest_rel(pc_accuracies, bp_accuracies)

        print()
        print("--- Paired t-test ---")
        print(f"Mean difference (PC - BP): {np.mean(differences)*100:+.2f}%")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}, N = {n_trials}")
        print(f"Significant at p<0.05: {'YES' if p_value < 0.05 else 'NO'}")

        # ================================================================
        # Effect Size
        # ================================================================
        d = compute_cohens_d(pc_accuracies, bp_accuracies)

        magnitude = "negligible"
        if abs(d) >= 0.8:
            magnitude = "large"
        elif abs(d) >= 0.5:
            magnitude = "medium"
        elif abs(d) >= 0.2:
            magnitude = "small"

        print()
        print("--- Effect Size ---")
        print(f"Cohen's d (paired): {d:.4f}")
        print(f"Interpretation: {magnitude}")

        # ================================================================
        # Power Analysis
        # ================================================================
        required_n = estimate_required_n(d, alpha=0.05, power=0.80)

        print()
        print("--- Power Analysis ---")
        print(f"Estimated trials needed for p<0.05 with 80% power: {required_n}")
        if isinstance(required_n, float) and required_n == float("inf"):
            print("  -> Effect size is ~zero; no finite sample can detect it.")
        elif required_n <= n_trials:
            print(f"  -> Current n_trials ({n_trials}) is sufficient.")
        else:
            print(
                f"  -> Current n_trials ({n_trials}) may be underpowered. "
                f"Consider increasing to {required_n}."
            )

    # ================================================================
    # Training Time Comparison (per epoch)
    # ================================================================
    pc_epoch_times = pc_train_times / num_epochs
    bp_epoch_times = bp_train_times / num_epochs

    pc_t_mean = np.mean(pc_epoch_times)
    bp_t_mean = np.mean(bp_epoch_times)
    pc_t_se = (
        np.std(pc_epoch_times, ddof=1) / np.sqrt(len(pc_epoch_times))
        if len(pc_epoch_times) > 1
        else 0.0
    )
    bp_t_se = (
        np.std(bp_epoch_times, ddof=1) / np.sqrt(len(bp_epoch_times))
        if len(bp_epoch_times) > 1
        else 0.0
    )

    print()
    print(f"--- Training Time per Epoch ---")
    print(f"PC:       {pc_t_mean:.3f} +/- {pc_t_se:.3f}s")
    print(f"Backprop: {bp_t_mean:.3f} +/- {bp_t_se:.3f}s")
    if bp_t_mean > 0:
        print(f"Ratio:    PC is {pc_t_mean / bp_t_mean:.2f}x backprop time")

    # ================================================================
    # Timing
    # ================================================================
    print()
    print(f"Total wall time: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
