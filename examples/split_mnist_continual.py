"""
Split-MNIST Continual Learning Example

Demonstrates the full continual learning pipeline using FabricPC:
- Sequential task training on digit pairs
- Support column selection
- Accuracy matrix tracking
- Forgetting analysis

Usage:
    python examples/split_mnist_continual.py
    python examples/split_mnist_continual.py --quick-smoke
    python examples/split_mnist_continual.py --training-mode backprop
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SigmoidActivation,
    SoftmaxActivation,
    ReLUActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer

from fabricpc.continual.config import make_config, ExperimentConfig
from fabricpc.continual.data import SplitMnistLoader, build_split_mnist_loaders
from fabricpc.continual.trainer import SequentialTrainer
from fabricpc.continual.utils import (
    plot_accuracy_curves,
    plot_accuracy_matrix,
    plot_forgetting_analysis,
    print_summary_table,
    save_summaries_json,
    save_accuracy_matrix,
    create_run_directory,
    save_experiment_config,
)


def create_network_structure(config: ExperimentConfig):
    """
    Create the FabricPC graph structure for continual learning.

    Architecture:
        input (784) -> hidden1 (256) -> hidden2 (128) -> output (10)

    For a more sophisticated architecture with columns and composers,
    use the custom nodes from fabricpc.continual.nodes.
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


def main():
    parser = argparse.ArgumentParser(description="Split-MNIST Continual Learning")
    parser.add_argument(
        "--quick-smoke",
        action="store_true",
        help="Run quick smoke test with minimal training",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="pc",
        choices=["pc", "backprop", "hybrid"],
        help="Training mode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/split_mnist",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs per task"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Split-MNIST Continual Learning with FabricPC")
    print("=" * 60)

    # Configuration
    config = make_config(quick_smoke=args.quick_smoke)
    config.seed = args.seed
    config.training.training_mode = args.training_mode

    if args.epochs is not None:
        config.training.epochs_per_task = args.epochs

    print(f"\nConfiguration:")
    print(f"  Training mode: {config.training.training_mode}")
    print(f"  Epochs per task: {config.training.epochs_per_task}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Task pairs: {config.task_pairs}")
    print(f"  Quick smoke: {args.quick_smoke}")

    # Initialize JAX
    jax.config.update("jax_default_prng_impl", "threefry2x32")
    master_key = jax.random.PRNGKey(config.seed)
    init_key, train_key = jax.random.split(master_key)

    # Create network
    print("\nCreating network...")
    structure = create_network_structure(config)
    params = initialize_params(structure, init_key)

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Nodes: {len(structure.nodes)}")
    print(f"  Edges: {len(structure.edges)}")
    print(f"  Parameters: {total_params:,}")

    # Create optimizer
    optimizer = optax.adamw(
        config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create trainer
    trainer = SequentialTrainer(
        structure=structure,
        config=config,
        params=params,
        optimizer=optimizer,
        rng_key=train_key,
    )

    # Load data
    print("\nLoading Split-MNIST data...")
    tasks = build_split_mnist_loaders(config, data_root="./data")
    print(f"  Tasks: {len(tasks)}")
    for task in tasks:
        print(
            f"    Task {task.task_id}: classes {task.classes}, "
            f"train batches: {len(task.train_loader)}, "
            f"test batches: {len(task.test_loader)}"
        )

    # Create output directory
    run_dir = create_run_directory(args.output_dir, "split_mnist", config.seed)
    print(f"\nOutput directory: {run_dir}")

    # Save config
    save_experiment_config(config, run_dir / "config.json")

    # Train sequentially
    print("\n" + "=" * 60)
    print("Starting Sequential Training")
    print("=" * 60)

    start_time = time.time()

    for task_data in tasks:
        summary = trainer.train_task(task_data, verbose=True)

        # Save checkpoint after each task
        checkpoint_path = run_dir / f"checkpoint_task_{task_data.task_id}.npz"
        trainer.save_checkpoint(str(checkpoint_path))

    total_time = time.time() - start_time

    # Results
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    print_summary_table(trainer.summaries)

    print(f"\nTotal training time: {total_time:.1f}s")
    print(f"Average time per task: {total_time / len(tasks):.1f}s")

    # Accuracy matrix
    acc_matrix = trainer.accuracy_matrix()
    print("\nAccuracy Matrix:")
    print(acc_matrix)

    # Forgetting metric
    forgetting = trainer.get_forgetting_metric()
    print(f"\nAverage forgetting: {forgetting:.4f}")

    # Save results
    save_summaries_json(trainer.summaries, run_dir / "summaries.json")
    save_accuracy_matrix(acc_matrix, run_dir / "accuracy_matrix.csv")

    # Plotting (requires plotly; kaleido for PNG export)
    try:
        print("\nGenerating plots...")
        plot_accuracy_curves(
            trainer.summaries, save_path=run_dir / "accuracy_curves.png", show=False
        )
        plot_accuracy_matrix(
            acc_matrix, save_path=run_dir / "accuracy_matrix.png", show=False
        )
        plot_forgetting_analysis(
            acc_matrix, save_path=run_dir / "forgetting_analysis.png", show=False
        )
        print(f"Plots saved to {run_dir}")
    except Exception as e:
        print(f"Could not generate plots: {e}")

    print(f"\nResults saved to: {run_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
