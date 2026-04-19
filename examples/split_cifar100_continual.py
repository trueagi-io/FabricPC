"""
Split-CIFAR-100 Continual Learning Example

Extends the Split-MNIST continual learning pipeline to CIFAR-100:
- 20 tasks of 5 classes each (class-incremental)
- 3072-dim flattened RGB input, 100-way softmax output
- Same ColumnNode / PartitionedAggregator / attention architectures

Usage:
    python examples/split_cifar100_continual.py
    python examples/split_cifar100_continual.py --quick-smoke
    python examples/split_cifar100_continual.py --training-mode backprop
    python examples/split_cifar100_continual.py --partitioned
    python examples/split_cifar100_continual.py --num-tasks 10 --classes-per-task 10
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import argparse
import time
from pathlib import Path

import jax
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SoftmaxActivation,
    ReLUActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import (
    XavierInitializer,
    NormalInitializer,
)

from fabricpc.continual.config import make_config, ExperimentConfig
from fabricpc.continual.nodes import ColumnNode, ComposerNode, PartitionedAggregator
from fabricpc.continual.data_cifar import build_split_cifar100_loaders
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

INPUT_DIM = 32 * 32 * 3  # 3072


def _build_class_groups(num_tasks: int, classes_per_task: int):
    return tuple(
        tuple(range(i * classes_per_task, (i + 1) * classes_per_task))
        for i in range(num_tasks)
    )


def create_network_structure(config: ExperimentConfig, use_columns: bool = True):
    """
    FabricPC graph structure for Split-CIFAR-100.

    Same topology as the MNIST example; only the input/output dims and the
    MLP hidden widths are scaled up for RGB images and 100-way output.
    """
    pixels = IdentityNode(shape=(INPUT_DIM,), name="pixels")

    if use_columns:
        num_columns = config.columns.num_columns
        memory_dim = config.columns.memory_dim
        use_attention = config.columns.use_attention_aggregator

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
            aggregator = Linear(
                shape=(aggregator_dim,),
                activation=ReLUActivation(),
                energy=GaussianEnergy(),
                name="aggregator",
                weight_init=XavierInitializer(),
                flatten_input=True,
            )

        output = Linear(
            shape=(config.num_output_classes,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="output",
            weight_init=XavierInitializer(),
        )

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
        # Plain MLP baseline — scaled up for RGB inputs.
        hidden1 = Linear(
            shape=(1024,),
            activation=ReLUActivation(),
            energy=GaussianEnergy(),
            name="hidden1",
            weight_init=XavierInitializer(),
        )
        hidden2 = Linear(
            shape=(512,),
            activation=ReLUActivation(),
            energy=GaussianEnergy(),
            name="hidden2",
            weight_init=XavierInitializer(),
        )
        output = Linear(
            shape=(config.num_output_classes,),
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


def _configure_for_cifar100(
    config: ExperimentConfig,
    num_tasks: int,
    classes_per_task: int,
    quick_smoke: bool,
) -> ExperimentConfig:
    """Override Split-MNIST defaults with CIFAR-100-sized task/column settings."""
    config.num_tasks = num_tasks
    config.num_output_classes = 100
    config.task_pairs = _build_class_groups(num_tasks, classes_per_task)

    # Column capacity: shared + num_tasks * topk_nonshared.
    # CIFAR-100 needs more capacity than MNIST; keep modest for compute.
    if quick_smoke:
        # Minimal but valid for the requested num_tasks
        config.columns.topk_nonshared = 2
        config.columns.shared_columns = 2
        config.columns.num_columns = (
            config.columns.shared_columns + num_tasks * config.columns.topk_nonshared
        )
        config.columns.memory_dim = 8
        config.columns.aggregator_dim = 64
    else:
        config.columns.topk_nonshared = 3
        config.columns.shared_columns = 4
        config.columns.num_columns = (
            config.columns.shared_columns + num_tasks * config.columns.topk_nonshared
        )
        config.columns.memory_dim = 128
        config.columns.aggregator_dim = 256
        config.columns.partitioned_shared_dim = 48
        config.columns.partitioned_task_dim = 48

    # Support manager topk_nonshared must match columns.topk_nonshared
    config.support.topk_nonshared = config.columns.topk_nonshared
    # The CIFAR continual setup benefits from actually enabling the learned
    # causal selector; the baseline config leaves it disabled at 0.0.
    config.support.causal_max_effective_scale = 0.5

    return config


def main():
    parser = argparse.ArgumentParser(description="Split-CIFAR-100 Continual Learning")
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
        default="../results/split_cifar100",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs per task"
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=20,
        help="Number of sequential tasks (default: 20)",
    )
    parser.add_argument(
        "--classes-per-task",
        type=int,
        default=5,
        help="Classes grouped into each task (default: 5)",
    )
    parser.add_argument(
        "--no-columns",
        action="store_true",
        help="Use standard Linear architecture instead of ColumnNode",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Use attention-based aggregation (ComposerNode)",
    )
    parser.add_argument(
        "--ewc",
        action="store_true",
        help="Enable EWC (Elastic Weight Consolidation)",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=5000.0,
        help="EWC regularization strength (default: 5000.0)",
    )
    parser.add_argument(
        "--causal-scale",
        type=float,
        default=None,
        help="Override support.causal_max_effective_scale (default: config value)",
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="Use PartitionedAggregator (true architectural isolation)",
    )
    args = parser.parse_args()

    if args.num_tasks * args.classes_per_task > 100:
        parser.error("num_tasks * classes_per_task must be <= 100 for CIFAR-100")

    print("=" * 60)
    print("Split-CIFAR-100 Continual Learning with FabricPC")
    print("=" * 60)

    config = make_config(quick_smoke=args.quick_smoke)
    config.seed = args.seed
    config.training.training_mode = args.training_mode

    if args.epochs is not None:
        config.training.epochs_per_task = args.epochs

    if args.quick_smoke and args.epochs is None:
        # Keep smoke fast — 2 train batches, 1 epoch, 5 tasks by default
        config.training.fast_dev_max_train_batches = 2
        config.training.fast_dev_max_test_batches = 2

    config = _configure_for_cifar100(
        config,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        quick_smoke=args.quick_smoke,
    )

    if args.attention:
        config.columns.use_attention_aggregator = True
        config.training.learning_rate = 0.0003

    if args.ewc:
        config.ewc.enable = True
        config.ewc.lambda_ewc = args.ewc_lambda

    if args.causal_scale is not None:
        config.support.causal_max_effective_scale = args.causal_scale

    if args.partitioned:
        config.columns.use_partitioned_aggregator = True
        config.columns.use_attention_aggregator = False

    print("\nConfiguration:")
    print(f"  Training mode: {config.training.training_mode}")
    print(f"  Epochs per task: {config.training.epochs_per_task}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Grad clip norm: {config.training.grad_clip_norm}")
    print(f"  Num tasks: {config.num_tasks}")
    print(f"  Classes per task: {args.classes_per_task}")
    print(f"  Output classes: {config.num_output_classes}")
    print(f"  Quick smoke: {args.quick_smoke}")
    print(f"  Column architecture: {not args.no_columns}")
    print(f"  Attention aggregator: {config.columns.use_attention_aggregator}")
    print(f"  Partitioned aggregator: {config.columns.use_partitioned_aggregator}")
    print(
        f"  Columns: {config.columns.num_columns} "
        f"(shared: {config.columns.shared_columns}, "
        f"topk_nonshared: {config.columns.topk_nonshared}, "
        f"memory_dim: {config.columns.memory_dim})"
    )
    print(f"  EWC enabled: {config.ewc.enable}")
    if config.ewc.enable:
        print(f"  EWC lambda: {config.ewc.lambda_ewc}")
    print(f"  Causal scale: {config.support.causal_max_effective_scale}")

    jax.config.update("jax_default_prng_impl", "threefry2x32")
    master_key = jax.random.PRNGKey(config.seed)
    init_key, train_key = jax.random.split(master_key)

    use_columns = not args.no_columns
    print(f"\nCreating network (columns={use_columns})...")
    structure = create_network_structure(config, use_columns=use_columns)
    params = initialize_params(structure, init_key)

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Nodes: {len(structure.nodes)}")
    print(f"  Edges: {len(structure.edges)}")
    print(f"  Parameters: {total_params:,}")

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

    trainer = SequentialTrainer(
        structure=structure,
        config=config,
        params=params,
        optimizer=optimizer,
        rng_key=train_key,
    )

    print("\nLoading Split-CIFAR-100 data...")
    tasks = build_split_cifar100_loaders(config, data_root="./data")
    print(f"  Tasks: {len(tasks)}")
    for task in tasks[:5]:
        print(
            f"    Task {task.task_id}: classes {task.classes}, "
            f"train batches: {len(task.train_loader)}, "
            f"test batches: {len(task.test_loader)}"
        )
    if len(tasks) > 5:
        print(f"    ... and {len(tasks) - 5} more tasks")

    run_dir = create_run_directory(args.output_dir, "split_cifar100", config.seed)
    print(f"\nOutput directory: {run_dir}")

    save_experiment_config(config, run_dir / "config.json")

    print("\n" + "=" * 60)
    print("Starting Sequential Training")
    print("=" * 60)

    start_time = time.time()
    for task_data in tasks:
        trainer.train_task(task_data, verbose=True)
        checkpoint_path = run_dir / f"checkpoint_task_{task_data.task_id}.npz"
        trainer.save_checkpoint(str(checkpoint_path))
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)

    print_summary_table(trainer.summaries)

    print(f"\nWall-clock run time: {total_time:.1f}s")
    print(f"Average wall time per task: {total_time / len(tasks):.1f}s")

    acc_matrix = trainer.accuracy_matrix()
    print("\nAccuracy Matrix:")
    print(acc_matrix)

    forgetting = trainer.get_forgetting_metric()
    print(f"\nAverage forgetting: {forgetting:.4f}")

    save_summaries_json(trainer.summaries, run_dir / "summaries.json")
    save_accuracy_matrix(acc_matrix, run_dir / "accuracy_matrix.csv")

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
