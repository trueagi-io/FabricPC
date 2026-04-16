"""
muPC Scaling — FC-ResNet on MNIST
==================================

Demonstrates muPC (Maximal Update Parameterization for Predictive Coding)
on a deep fully-connected residual network trained on MNIST, using only
FabricPC's native graph components. muPC scaling is computed automatically
from the graph topology — no manual scaling needed.

Architecture:
    input(784) -> stem(W, W=64) -> [N residual blocks] -> output(10)

Each residual block:
    prev -> Linear(W, Tanh) -> SkipConnection(sum)
      |                              ^
      +------------------------------+  (identity skip path)

SkipConnection nodes have apply_variance_scaling=False, so muPC preserves
the identity mapping through deep networks without signal decay.

Usage:
    python examples/mupc_demo.py
    python examples/mupc_demo.py --num_blocks 4 --verbose
    python examples/mupc_demo.py --num_blocks 32 --num_epochs 6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import time
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.nodes.skip_connection import SkipConnection
from fabricpc.builder import Edge, TaskMap, graph, GraphNamespace
from fabricpc.graph import initialize_params
from fabricpc.core.activations import TanhActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader

jax.config.update("jax_default_prng_impl", "threefry2x32")


def parse_args():
    parser = argparse.ArgumentParser(description="muPC demo: FC-ResNet on MNIST")
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=16,
        help="Number of residual blocks (default: 16)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden layer width (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Training epochs (default: 4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=0.003,
        help="Inference rate (default: 0.003)",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="Inference steps per sample (default: 4*(num_blocks+2))",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="Learning rate (default: 0.002)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output",
    )
    return parser.parse_args()


def build_fc_resnet(num_blocks, hidden_dim, infer_steps=None, *, eta_infer):
    """
    Build an FC-ResNet for MNIST with muPC scaling.

    Architecture:
        input(784) -> stem(hidden_dim) -> [N residual blocks] -> output(10)

    Each residual block has a Linear transform path and a SkipConnection
    that sums the transform output with the identity skip. SkipConnection
    nodes disable muPC variance scaling to preserve the identity mapping.

    Args:
        num_blocks: Number of residual blocks.
        hidden_dim: Width of hidden layers.
        infer_steps: Inference steps per sample. Default: max(20, 4*(num_blocks+2)).
        eta_infer: Inference rate.

    Returns:
        GraphStructure with muPC scaling.
    """
    if infer_steps is None:
        infer_steps = max(20, 4 * (num_blocks + 2))
    mupc_init = MuPCInitializer()

    # Input node
    input_node = IdentityNode(shape=(784,), name="input")

    # Stem: projects 784 -> hidden_dim (no skip connection)
    stem = Linear(
        shape=(hidden_dim,),
        weight_init=mupc_init,
        flatten_input=True,
        name="stem",
    )

    all_nodes = [input_node, stem]
    all_edges = [Edge(source=input_node, target=stem.slot("in"))]

    # Residual blocks
    prev = stem
    for i in range(num_blocks):
        with GraphNamespace(f"block{i}"):
            linear = Linear(
                shape=(hidden_dim,),
                activation=TanhActivation(),
                weight_init=mupc_init,
                name="linear",
            )
            skip = SkipConnection(
                shape=(hidden_dim,),
                name="sum",
            )

        all_nodes.extend([linear, skip])
        all_edges.extend(
            [
                Edge(source=prev, target=linear.slot("in")),  # transform path
                Edge(source=prev, target=skip.slot("in")),  # skip/identity path
                Edge(source=linear, target=skip.slot("in")),  # merge into sum
            ]
        )
        prev = skip

    # Output classifier
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=XavierInitializer(),
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=prev, target=output.slot("in")))

    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
        scaling=MuPCConfig(include_output=False),
    )

    return structure


def main():
    args = parse_args()

    print("=" * 60)
    print("muPC Demo: FC-ResNet on MNIST")
    print("=" * 60)

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    structure = build_fc_resnet(
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )
    params = initialize_params(structure, graph_key)

    print(
        f"\nArchitecture: input(784) -> stem({args.hidden_dim})"
        f" -> {args.num_blocks} residual blocks -> output(10)"
    )
    print(f"Model: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    for name, node in structure.nodes.items():
        ni = node.node_info
        if ni.scaling_config is not None:
            fwd_scales = list(ni.scaling_config.forward_scale.values())
            tag = f"fwd_scale={fwd_scales[0]:.6f}" if fwd_scales else ""
        else:
            tag = "no scaling"
        print(f"  {name}: shape={ni.shape}, {tag}")

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data
    train_loader = MnistLoader(
        "train",
        batch_size=args.batch_size,
        tensor_format="flat",
        shuffle=True,
        seed=42,
    )
    test_loader = MnistLoader(
        "test",
        batch_size=args.batch_size,
        tensor_format="flat",
        shuffle=False,
    )

    # Train
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    train_config = {"num_epochs": args.num_epochs}

    print(
        f"\nTraining for {args.num_epochs} epochs "
        f"(JIT compilation on first batch)..."
    )
    start_time = time.time()

    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=args.verbose,
    )

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.1f}s ({elapsed / args.num_epochs:.1f}s per epoch)")

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

    if metrics["accuracy"] >= 0.90:
        print("PASS: accuracy >= 90%")
    else:
        print(f"BELOW TARGET: {metrics['accuracy']*100:.1f}% < 90%")


if __name__ == "__main__":
    main()
