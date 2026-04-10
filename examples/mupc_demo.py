"""
muPC Scaling — CIFAR-100 Conv Demo
===================================

Demonstrates muPC (Maximal Update Parameterization for Predictive Coding)
on a small convolutional network trained on CIFAR-100.

Architecture:
    input(32,32,3) -> Conv3x3(32,32,32) -> Conv3x3(16,16,64, stride=2)
    -> Conv3x3(8,8,128, stride=2) -> Linear(100, softmax, CE)

Key patterns:
    - MuPCInitializer() on all parameterized nodes (weights ~ N(0, gain^2))
    - MuPCConfig(depth_metric=ShortestPathDepth()) in graph() builder
    - Per-edge forward/gradient scaling computed automatically from topology

Usage:
    python examples/mupc_demo.py
    python examples/mupc_demo.py --num_epochs 5 --verbose
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import argparse
import time
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode
from custom_node import Conv2DNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.depth_metric import ShortestPathDepth
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import Cifar100Loader

jax.config.update("jax_default_prng_impl", "threefry2x32")


def parse_args():
    parser = argparse.ArgumentParser(
        description="muPC scaling demo: small ConvNet on CIFAR-100"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output",
    )
    return parser.parse_args()


def create_mupc_convnet():
    """
    Build a small convolutional network with muPC parameterization.

    Architecture:
        input(32,32,3) -> conv1(32,32,32) 3x3 ReLU
        -> conv2(16,16,64) 3x3 stride=2 ReLU
        -> conv3(8,8,128) 3x3 stride=2 ReLU
        -> output(100) flatten -> softmax + CE

    Returns:
        GraphStructure with muPC scaling attached to each node.
    """
    mupc_init = MuPCInitializer()

    input_node = IdentityNode(
        shape=(32, 32, 3),
        name="input",
    )

    conv1 = Conv2DNode(
        shape=(32, 32, 32),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=mupc_init,
        name="conv1",
    )

    conv2 = Conv2DNode(
        shape=(16, 16, 64),
        kernel_size=(3, 3),
        stride=(2, 2),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=mupc_init,
        name="conv2",
    )

    conv3 = Conv2DNode(
        shape=(8, 8, 128),
        kernel_size=(3, 3),
        stride=(2, 2),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=mupc_init,
        name="conv3",
    )

    output = Linear(
        shape=(100,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=mupc_init,
        name="output",
    )

    structure = graph(
        nodes=[input_node, conv1, conv2, conv3, output],
        edges=[
            Edge(source=input_node, target=conv1.slot("in")),
            Edge(source=conv1, target=conv2.slot("in")),
            Edge(source=conv2, target=conv3.slot("in")),
            Edge(source=conv3, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
        scaling=MuPCConfig(depth_metric=ShortestPathDepth()),
    )

    return structure


def main():
    args = parse_args()

    print("=" * 60)
    print("muPC Demo: ConvNet on CIFAR-100")
    print("=" * 60)

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model with muPC scaling
    structure = create_mupc_convnet()
    params = initialize_params(structure, graph_key)

    print(f"\nModel: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    for name, node in structure.nodes.items():
        ni = node.node_info
        scaling_tag = ""
        if ni.scaling_config is not None:
            fwd_scales = list(ni.scaling_config.forward_scale.values())
            scaling_tag = f"  fwd_scale={fwd_scales[0]:.4f}" if fwd_scales else ""
        print(f"  {name}: shape={ni.shape}, type={ni.node_type}{scaling_tag}")

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data
    train_loader = Cifar100Loader(
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
    )
    test_loader = Cifar100Loader(
        "test",
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Train
    optimizer = optax.adamw(0.001, weight_decay=0.01)
    train_config = {"num_epochs": args.num_epochs}

    print(
        f"\nTraining for {args.num_epochs} epochs (JIT compilation on first batch)..."
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
    print(f"Test Energy:   {metrics['energy']:.4f}")


if __name__ == "__main__":
    main()
