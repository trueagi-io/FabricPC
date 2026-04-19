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
    IdentityActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import (
    XavierInitializer,
    NormalInitializer,
    ZerosInitializer,
)

from fabricpc.continual.config import make_config, ExperimentConfig
from fabricpc.continual.nodes import ColumnNode, ComposerNode, PartitionedAggregator
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


def create_network_structure(config: ExperimentConfig, use_columns: bool = True):
    """
    Create the FabricPC graph structure for continual learning.

    If use_columns=True (default):
        Uses ColumnNode architecture with true column isolation for CL:
        input (784) -> columns (num_columns, memory_dim) -> aggregator -> output (10)

        Each column has separate weights (col_X_proj), enabling gradient protection
        to freeze specific columns without affecting others.

        If use_attention_aggregator=True, uses ComposerNode with attention-based
        task routing instead of simple Linear aggregator.

    If use_columns=False:
        Uses standard Linear architecture (no column isolation):
        input (784) -> hidden1 (256) -> hidden2 (128) -> output (10)
    """
    # Input node
    pixels = IdentityNode(shape=(784,), name="pixels")

    if use_columns:
        # Modular ColumnNode architecture for true column isolation
        num_columns = config.columns.num_columns
        memory_dim = config.columns.memory_dim
        use_attention = config.columns.use_attention_aggregator

        # ColumnNode: separate weights per column (col_X_proj)
        # Each column processes the input independently
        columns = ColumnNode(
            shape=(num_columns, memory_dim),
            name="columns",
            num_shells=3,
            shell_sizes=config.shell_demotion_transweave.shell_sizes,
            activation=ReLUActivation(),
            energy=GaussianEnergy(),
            weight_init=NormalInitializer(mean=0.0, std=0.02),
        )

        aggregator_dim = config.columns.aggregator_dim
        use_partitioned = config.columns.use_partitioned_aggregator

        if use_partitioned:
            # Partitioned aggregator with TRUE architectural isolation
            # Each task has dedicated weight matrices - no gradient flow between tasks
            shared_dim = config.columns.partitioned_shared_dim
            task_dim = config.columns.partitioned_task_dim
            aggregator = PartitionedAggregator(
                shape=(shared_dim + task_dim,),
                name="aggregator",
                num_tasks=config.num_tasks,
                shared_columns=config.columns.shared_columns,
                topk_nonshared=config.columns.topk_nonshared,
                shared_dim=shared_dim,
                task_dim=task_dim,
                memory_dim=memory_dim,
                activation=ReLUActivation(),
                energy=GaussianEnergy(),
                weight_init=NormalInitializer(mean=0.0, std=0.02),
            )
        elif use_attention:
            # Attention-based aggregator with task-specific routing
            # Uses self-attention over columns + task query for weighted aggregation
            aggregator = ComposerNode(
                shape=(aggregator_dim,),
                name="aggregator",
                num_heads=config.columns.attention_num_heads,
                num_layers=config.columns.attention_num_layers,
                num_tasks=config.num_tasks,
                gate_temp=0.5,
                activation=ReLUActivation(),
                energy=GaussianEnergy(),
                weight_init=NormalInitializer(mean=0.0, std=0.02),
            )
        else:
            # Simple linear aggregator: flattens and projects columns
            aggregator = Linear(
                shape=(aggregator_dim,),
                activation=ReLUActivation(),
                energy=GaussianEnergy(),
                name="aggregator",
                weight_init=XavierInitializer(),
                flatten_input=True,  # Flatten (num_columns, memory_dim) to (num_columns * memory_dim)
            )

        # Output layer
        output = Linear(
            shape=(10,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="output",
            weight_init=XavierInitializer(),
        )

        # Build graph with column architecture
        if use_attention:
            # ComposerNode receives columns directly (no flatten needed)
            structure = graph(
                nodes=[pixels, columns, aggregator, output],
                edges=[
                    Edge(source=pixels, target=columns.slot("in")),
                    Edge(source=columns, target=aggregator.slot("in")),
                    Edge(source=aggregator, target=output.slot("in")),
                ],
                task_map=TaskMap(x=pixels, y=output),
                inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
            )
        else:
            structure = graph(
                nodes=[pixels, columns, aggregator, output],
                edges=[
                    Edge(source=pixels, target=columns.slot("in")),
                    Edge(source=columns, target=aggregator.slot("in")),
                    Edge(source=aggregator, target=output.slot("in")),
                ],
                task_map=TaskMap(x=pixels, y=output),
                inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
            )
    else:
        # Standard Linear architecture (no column isolation)
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

        output = Linear(
            shape=(10,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="output",
            weight_init=XavierInitializer(),
        )

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
        default="../results/split_mnist",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs per task"
    )
    parser.add_argument(
        "--no-columns",
        action="store_true",
        help="Use standard Linear architecture instead of ColumnNode (disables column isolation)",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Use attention-based aggregation (ComposerNode) instead of Linear",
    )
    parser.add_argument(
        "--ewc",
        action="store_true",
        help="Enable EWC (Elastic Weight Consolidation) for reduced forgetting",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=5000.0,
        help="EWC regularization strength (default: 5000.0)",
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="Use PartitionedAggregator with true architectural isolation between tasks",
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

    # Enable attention-based aggregation if requested
    if args.attention:
        config.columns.use_attention_aggregator = True
        # Use lower learning rate for attention stability
        config.training.learning_rate = 0.0003

    # Enable EWC (Elastic Weight Consolidation) if requested
    if args.ewc:
        config.ewc.enable = True
        config.ewc.lambda_ewc = args.ewc_lambda

    # Enable partitioned aggregator for true architectural isolation
    if args.partitioned:
        config.columns.use_partitioned_aggregator = True
        # Disable attention if partitioned is selected
        config.columns.use_attention_aggregator = False

    print(f"\nConfiguration:")
    print(f"  Training mode: {config.training.training_mode}")
    print(f"  Epochs per task: {config.training.epochs_per_task}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Grad clip norm: {config.training.grad_clip_norm}")
    print(f"  Task pairs: {config.task_pairs}")
    print(f"  Quick smoke: {args.quick_smoke}")
    print(f"  Column architecture: {not args.no_columns}")
    print(f"  Attention aggregator: {config.columns.use_attention_aggregator}")
    print(f"  Partitioned aggregator: {config.columns.use_partitioned_aggregator}")
    print(
        f"  Columns: {config.columns.num_columns} (shared: {config.columns.shared_columns})"
    )
    if config.columns.use_partitioned_aggregator:
        print(
            f"  Partitioned dims: shared={config.columns.partitioned_shared_dim}, "
            f"task={config.columns.partitioned_task_dim}"
        )
    print(f"  EWC enabled: {config.ewc.enable}")
    if config.ewc.enable:
        print(f"  EWC lambda: {config.ewc.lambda_ewc}")

    # Initialize JAX
    jax.config.update("jax_default_prng_impl", "threefry2x32")
    master_key = jax.random.PRNGKey(config.seed)
    init_key, train_key = jax.random.split(master_key)

    # Create network
    use_columns = not args.no_columns
    print(f"\nCreating network (columns={use_columns})...")
    structure = create_network_structure(config, use_columns=use_columns)
    params = initialize_params(structure, init_key)

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Nodes: {len(structure.nodes)}")
    print(f"  Edges: {len(structure.edges)}")
    print(f"  Parameters: {total_params:,}")

    # Create optimizer (with gradient clipping for attention mode)
    if config.columns.use_attention_aggregator:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.training.grad_clip_norm),
            optax.adamw(
                config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            ),
        )
    else:
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
