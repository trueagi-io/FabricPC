"""
Split-MNIST Continual Learning with Causal Guidance

Demonstrates the causal coding features for continual learning:
- Support swap audit system for generating causal training data
- CausalContributionPredictor for learning column value
- CausalSelectorTrustController with agreement tracking
- Causal-guided column selection

The causal system learns from audit results to predict which columns
will be beneficial for new tasks, improving support selection over time.

Usage:
    python examples/split_mnist_causal.py
    python examples/split_mnist_causal.py --quick-smoke
    python examples/split_mnist_causal.py --num-tasks 3
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import argparse
import time

import jax
import numpy as np

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SoftmaxActivation,
    ReLUActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer

from fabricpc.continual.config import make_config, ExperimentConfig
from fabricpc.continual.data import SplitMnistLoader
from fabricpc.continual.trainer import SequentialTrainer


def create_network_structure(config: ExperimentConfig):
    """
    Create the FabricPC graph structure for continual learning.

    Architecture:
        input (784) -> hidden1 (256) -> hidden2 (128) -> output (10)
    """
    # Input node
    pixels = IdentityNode(shape=(784,), name="pixels")

    # Hidden layers
    hidden1 = Linear(
        shape=(256,),
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        name="hidden1",
        weight_init=XavierInitializer(),
    )

    hidden2 = Linear(
        shape=(128,),
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        name="hidden2",
        weight_init=XavierInitializer(),
    )

    # Output layer
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="output",
        weight_init=XavierInitializer(),
    )

    # Build graph
    structure = graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )

    return structure


def print_causal_summary(trainer: SequentialTrainer):
    """Print detailed causal system summary."""
    print("\n" + "=" * 60)
    print("CAUSAL SYSTEM SUMMARY")
    print("=" * 60)

    # Predictor state
    predictor = trainer.support_manager.causal_predictor
    if predictor:
        print("\nCausal Contribution Predictor:")
        print(f"  Total training examples: {predictor.num_examples()}")
        print(f"  Model trained: {predictor.trained}")
        if predictor.trained:
            print(f"  Last correlation: {predictor.last_corr:.4f}")
            print(f"  Last MAE: {predictor.last_mae:.4f}")
            if predictor.beta_ is not None:
                print(f"  Feature weights (top 5 by magnitude):")
                feature_names = [
                    "base_z",
                    "cert_general",
                    "cert_specific",
                    "cert_demotion",
                    "cert_saturation",
                    "novelty",
                    "saturation",
                    "recent_penalty",
                    "reserve_bonus",
                    "reserve_flag",
                    "fp_cur",
                    "fp_old",
                    "fp_abs",
                    "fp_conf",
                    "struct_max",
                    "causal_max",
                    "role_reuse",
                    "role_diverse",
                    "role_challenger",
                    "context_size",
                    "task_pos",
                ]
                # Skip intercept (index 0)
                weights = predictor.beta_[1:] if len(predictor.beta_) > 1 else []
                if len(weights) > 0:
                    indexed = [
                        (
                            i,
                            w,
                            feature_names[i] if i < len(feature_names) else f"feat_{i}",
                        )
                        for i, w in enumerate(weights)
                    ]
                    indexed.sort(key=lambda x: abs(x[1]), reverse=True)
                    for i, w, name in indexed[:5]:
                        print(f"    {name}: {w:.4f}")

    # Trust controller state
    trust = trainer.support_manager.causal_trust
    if trust:
        print("\nCausal Trust Controller:")
        diag = trust.last_diag
        print(f"  Coverage gate:   {diag.get('coverage_gate', 0):.4f}")
        print(f"  Agreement gate:  {diag.get('agreement_gate', 0):.4f}")
        print(f"  Trend gate:      {diag.get('trend_gate', 0):.4f}")
        print(f"  Structural gate: {diag.get('structural_gate', 0):.4f}")
        print(f"  Effective scale: {diag.get('effective_scale', 0):.4f}")
        print(f"  Mix gate:        {diag.get('mix_gate', 0):.4f}")

        # Agreement tracker
        tracker = trust.agreement_tracker
        print("\n  Agreement Tracker:")
        print(f"    Predictions recorded: {len(tracker.predictions)}")
        print(f"    Outcomes recorded: {len(tracker.outcomes)}")
        agreement, n_matched = tracker.compute_recent_agreement()
        print(f"    Recent agreement: {agreement:.4f} (n={n_matched})")

    # Fingerprint bank
    bank = trainer.support_manager.causal_bank
    if bank:
        print("\nCausal Fingerprint Bank:")
        print(f"  Shape: {bank.gain_sum.shape} (columns x tasks)")
        total_obs = bank.gain_count.sum()
        print(f"  Total observations: {total_obs:.0f}")
        conf = bank.column_confidence(
            trainer.config.support.causal_similarity_conf_target
        )
        print(f"  Mean column confidence: {conf.mean():.4f}")
        print(f"  Max column confidence: {conf.max():.4f}")

        # Show top columns by mean gain
        mean_gain = bank.mean_gain()
        col_scores = mean_gain.mean(axis=1)
        top_cols = np.argsort(col_scores)[-5:][::-1]
        print(f"  Top 5 columns by mean gain:")
        for col in top_cols:
            print(f"    Column {col}: {col_scores[col]:.4f}")


def print_accuracy_matrix(trainer: SequentialTrainer):
    """Print the accuracy matrix."""
    matrix = trainer.accuracy_matrix()
    n_tasks = matrix.shape[1] if matrix.size > 0 else 0

    print("\nAccuracy Matrix (rows=trained up to, cols=evaluated on):")
    header = "        " + "  ".join(f"Task{i}" for i in range(n_tasks))
    print(header)

    for i, row in enumerate(matrix):
        row_str = f"After{i}: "
        row_str += "  ".join(f"{v:.4f}" if v > 0 else "  -   " for v in row)
        print(row_str)


def main():
    parser = argparse.ArgumentParser(
        description="Split-MNIST Continual Learning with Causal Guidance"
    )
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Run quick smoke test with minimal training",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=5,
        help="Number of tasks to train (1-5)",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="pc",
        choices=["pc", "backprop"],
        help="Training mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--causal-scale",
        type=float,
        default=0.5,
        help="Max effective scale for causal guidance (0 to disable)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Split-MNIST Continual Learning with Causal Guidance")
    print("=" * 60)

    # Configuration
    config = make_config(quick_smoke=args.quick_smoke)
    config.seed = args.seed
    config.training.training_mode = args.training_mode

    # Set number of tasks
    num_tasks = min(5, max(1, args.num_tasks))
    config.num_tasks = num_tasks
    all_pairs = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
    config.task_pairs = all_pairs[:num_tasks]

    # Enable causal guidance
    config.support.causal_max_effective_scale = args.causal_scale
    config.support.causal_min_examples = 4 if args.quick_smoke else 36

    print(f"\nConfiguration:")
    print(f"  Training mode: {config.training.training_mode}")
    print(f"  Epochs per task: {config.training.epochs_per_task}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Number of tasks: {num_tasks}")
    print(f"  Task pairs: {config.task_pairs}")
    print(f"  Causal max scale: {config.support.causal_max_effective_scale}")
    print(f"  Causal min examples: {config.support.causal_min_examples}")

    # Create network structure
    print("\nBuilding network structure...")
    structure = create_network_structure(config)

    # Create trainer
    print("Initializing trainer...")
    trainer = SequentialTrainer(structure, config)

    # Load data
    print("Loading Split-MNIST data...")
    loader = SplitMnistLoader(
        task_pairs=config.task_pairs,
        batch_size=config.training.batch_size,
        seed=config.seed,
    )

    # Train on each task
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    start_time = time.time()
    for task_data in loader.tasks:
        summary = trainer.train_task(task_data, verbose=True)

        # Print causal metrics summary
        print(
            f"\n  Causal Status: examples={summary.causal_selector_examples:.0f}, "
            f"corr={summary.causal_selector_corr:.4f}, "
            f"agree={summary.causal_selector_agreement_gate:.3f}, "
            f"mix={summary.causal_selector_mix_gate:.3f}"
        )

    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print_accuracy_matrix(trainer)

    print(f"\nPerformance Metrics:")
    print(f"  Average forgetting: {trainer.get_forgetting_metric():.4f}")
    print(f"  Total training time: {total_time:.1f}s")

    # Final task accuracies
    final_accs = (
        trainer.accuracy_matrix()[-1] if trainer.accuracy_matrix().size > 0 else []
    )
    if len(final_accs) > 0:
        print(f"  Final mean accuracy: {np.mean(final_accs[final_accs > 0]):.4f}")

    # Print causal system summary
    print_causal_summary(trainer)

    # Summary table
    print("\n" + "=" * 60)
    print("TASK SUMMARY")
    print("=" * 60)
    print(
        f"{'Task':<6} {'Classes':<10} {'Accuracy':<10} {'Examples':<10} "
        f"{'Corr':<8} {'Mix Gate':<10}"
    )
    print("-" * 60)
    for summary in trainer.summaries:
        print(
            f"{summary.task_id:<6} {str(summary.classes):<10} "
            f"{summary.test_accuracy:<10.4f} "
            f"{summary.causal_selector_examples:<10.0f} "
            f"{summary.causal_selector_corr:<8.4f} "
            f"{summary.causal_selector_mix_gate:<10.4f}"
        )

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
