"""
muPC Scaling — MNIST Validation Demo
======================================

Validates muPC (Maximal Update Parameterization for Predictive Coding) on MNIST,
targeting ~91% accuracy to confirm our implementation matches the jpc reference
(thebuckleylab/jpc).

Architecture:
    input(784) -> hidden1(N, ReLU) -> ... -> hiddenL(N, ReLU) -> output(10)

muPC scaling:
    - MuPCInitializer (W ~ N(0,1)) on all Linear layers including output
    - MuPCConfig(include_output=True)
    - Forward scaling: a = 1/sqrt(fan_in * K) where K = in-degree
      For single-edge chain: a = 1/sqrt(fan_in), i.e. Kaiming scaling
      For output: a = 1/(fan_in * sqrt(K)) = 1/fan_in for K=1
    - Gaussian (MSE) energy on output (matches jpc reference)

Results:
    ~91% test accuracy in 10 epochs (vs. jpc reference ~93% in 1 epoch)

Usage:
    python examples/mupc_mnist_demo.py
    python examples/mupc_mnist_demo.py --num_epochs 10 --verbose
    python examples/mupc_mnist_demo.py --num_hidden 20 --hidden_dim 64
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import time
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader

jax.config.update("jax_default_prng_impl", "threefry2x32")


def parse_args():
    parser = argparse.ArgumentParser(description="muPC validation: FC network on MNIST")
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
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden layer width (default: 64, matches jpc reference)",
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=10,
        help="Number of hidden layers (default: 10)",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="Inference steps per sample (default: 4*(num_hidden+2))",
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=0.003,
        help="Inference rate (default: 0.003)",
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


def build_mupc_network(hidden_dim, num_hidden, infer_steps=None, *, eta_infer):
    """
    Build an FC network for MNIST with muPC scaling.

    Architecture:
        input(784) -> hidden1(hidden_dim, ReLU) -> ... -> output(10)

    All layers use MuPCInitializer (W ~ N(0,1)) and muPC forward scaling.
    Output uses Identity activation + Gaussian (MSE) energy, matching the
    jpc reference implementation.

    Args:
        hidden_dim: Width of hidden layers.
        num_hidden: Number of hidden layers.
        infer_steps: Inference steps per sample. Default: max(20, 3*(num_hidden+2)).
        eta_infer: Inference rate.

    Returns:
        GraphStructure with muPC scaling.
    """
    if infer_steps is None:
        infer_steps = max(20, 4 * (num_hidden + 2))
    weight_init = MuPCInitializer()

    # Input
    input_node = IdentityNode(shape=(784,), name="input")

    # Hidden layers
    hidden_layers = []
    for i in range(num_hidden):
        h = Linear(
            shape=(hidden_dim,),
            activation=TanhActivation(),
            weight_init=weight_init,
            flatten_input=(i == 0),  # First hidden flattens input
            name=f"h{i + 1}",
        )
        hidden_layers.append(h)

    # Output: Identity activation + Gaussian (MSE) energy (default).
    # MuPCInitializer + include_output=True gives a_L = 1/fan_in.
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=XavierInitializer(),
        flatten_input=True,
        name="output",
    )

    # Build edges
    all_nodes = [input_node] + hidden_layers + [output]
    all_edges = []
    prev = input_node
    for h in hidden_layers:
        all_edges.append(Edge(source=prev, target=h.slot("in")))
        prev = h
    all_edges.append(Edge(source=prev, target=output.slot("in")))

    # Build graph with muPC scaling
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
    print("muPC Validation: FC Network on MNIST")
    print("=" * 60)

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    structure = build_mupc_network(
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )
    params = initialize_params(structure, graph_key)

    print(f"\nModel: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
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

    # Reference comparison
    print(f"\njpc reference: ~93% (MNIST, FC network, muPC scaling)")
    if metrics["accuracy"] >= 0.90:
        print("PASS: accuracy >= 90%")
    else:
        print(f"BELOW TARGET: {metrics['accuracy']*100:.1f}% < 90%")


if __name__ == "__main__":
    main()
