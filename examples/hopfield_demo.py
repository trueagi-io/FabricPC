"""
Hopfield Network on Binarized MNIST — FabricPC Demo
====================================================

Trains a predictive coding network using HopfieldNode (tanh activation)
as the hidden layer on binarized MNIST. Achieves high classification
accuracy with supervised cross-entropy training.

Architecture:
    input(784) ──> HopfieldNode(256, tanh) ──> Linear(10, softmax, CE)

The HopfieldNode provides the same tanh-based dynamics used in
continuous Hopfield networks, applied here as a learned hidden
representation for classification.

Usage:
    python examples/hopfield_demo.py
    python examples/hopfield_demo.py --num_epochs 30
    python examples/hopfield_demo.py --hidden_size 512
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.graph import initialize_params
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.nodes.hopfield import HopfieldNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.binarized_mnist import load_binarized_mnist


# =========================================================================
# Data Loader
# =========================================================================


class HopfieldMnistLoader:
    """Yields (image, one_hot_label) batches for supervised training."""

    def __init__(self, images, labels, num_classes, batch_size,
                 shuffle=True, seed=42):
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.images)
        idx = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for s in range(0, n, self.batch_size):
            e = min(s + self.batch_size, n)
            batch_imgs = jnp.array(self.images[idx[s:e]])
            batch_lbls = jax.nn.one_hot(
                jnp.array(self.labels[idx[s:e]]), self.num_classes
            )
            yield batch_imgs, batch_lbls

    def __len__(self):
        return (len(self.images) + self.batch_size - 1) // self.batch_size


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Hopfield Network on Binarized MNIST"
    )
    parser.add_argument(
        "--digits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Digit classes to store (default: all 10)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--infer_steps", type=int, default=20,
        help="PC inference steps per training batch (default: 20)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512,
        help="HopfieldNode hidden layer size (default: 512)",
    )
    parser.add_argument(
        "--n_train", type=int, default=5000,
        help="Training images per digit (default: 5000)",
    )
    parser.add_argument(
        "--n_test", type=int, default=50,
        help="Test images per digit (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=200,
        help="Batch size (default: 200)",
    )
    args = parser.parse_args()

    jax.config.update("jax_default_prng_impl", "threefry2x32")
    rng_key = jax.random.PRNGKey(42)
    num_classes = len(args.digits)

    print("=" * 70)
    print("Hopfield Network on Binarized MNIST — FabricPC")
    print("=" * 70)

    # --- Load dataset ---
    print("\nLoading binarized MNIST...")
    train_images, train_labels, test_images, test_labels = load_binarized_mnist()
    print(f"  Train: {train_images.shape}  Test: {test_images.shape}")

    digits = args.digits
    N = train_images.shape[1]  # 784
    H = args.hidden_size
    print(f"  Digits: {digits}, input: {N}, hidden: {H}, classes: {num_classes}")

    # --- Sample train and test subsets ---
    train_subset_imgs = []
    train_subset_lbls = []
    for d in digits:
        mask = train_labels == d
        idx = np.where(mask)[0][: args.n_train]
        train_subset_imgs.append(train_images[idx])
        train_subset_lbls.append(train_labels[idx])
    train_subset_imgs = np.concatenate(train_subset_imgs)
    train_subset_lbls = np.concatenate(train_subset_lbls)

    test_subset_imgs = []
    test_subset_lbls = []
    for d in digits:
        mask = test_labels == d
        idx = np.where(mask)[0][: args.n_test]
        test_subset_imgs.append(test_images[idx])
        test_subset_lbls.append(test_labels[idx])
    test_subset_imgs = np.concatenate(test_subset_imgs)
    test_subset_lbls = np.concatenate(test_subset_lbls)

    print(f"  Train subset: {len(train_subset_imgs)} images ({args.n_train} per digit)")
    print(f"  Test subset:  {len(test_subset_imgs)} images ({args.n_test} per digit)")

    # --- Build network: input -> HopfieldNode -> classification output ---
    print("\nBuilding network...")
    input_node = IdentityNode(shape=(N,), name="pixels")
    hidden_node = HopfieldNode(
        shape=(H,),
        name="hopfield",
        weight_init=XavierInitializer(),
    )
    output_node = Linear(
        shape=(num_classes,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )

    structure = graph(
        nodes=[input_node, hidden_node, output_node],
        edges=[
            Edge(source=input_node, target=hidden_node.slot("in")),
            Edge(source=hidden_node, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=args.infer_steps),
    )

    rng_key, graph_key, train_key = jax.random.split(rng_key, 3)
    params = initialize_params(structure, graph_key)

    print(f"  Nodes: {len(structure.nodes)}, Edges: {len(structure.edges)}")
    for name, node in structure.nodes.items():
        info = node.node_info
        print(
            f"    {name}: shape={info.shape}, type={info.node_type}, "
            f"in={info.in_degree}, out={info.out_degree}"
        )
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"  Total parameters: {total_params:,}")

    # --- Training setup ---
    train_loader = HopfieldMnistLoader(
        train_subset_imgs, train_subset_lbls, num_classes,
        batch_size=args.batch_size, shuffle=True,
    )
    train_eval_loader = HopfieldMnistLoader(
        train_subset_imgs, train_subset_lbls, num_classes,
        batch_size=args.batch_size, shuffle=False,
    )
    optimizer = optax.adamw(args.lr, weight_decay=0.1)

    # Track per-epoch training energy
    batch_energy_tracker = []

    def iter_cb(_epoch_idx, _batch_idx, energy):
        normalized = float(energy) / args.batch_size
        batch_energy_tracker.append(normalized)
        return normalized

    def epoch_cb(epoch_idx, params, structure, _config, rng_key):
        train_energy = (
            sum(batch_energy_tracker) / len(batch_energy_tracker)
            if batch_energy_tracker
            else 0.0
        )
        batch_energy_tracker.clear()

        rng_key, eval_rk = jax.random.split(rng_key)
        metrics = evaluate_pcn(
            params, structure, train_eval_loader, {}, eval_rk
        )
        train_acc = metrics["accuracy"]

        print(
            f"Epoch {epoch_idx + 1:3d}/{args.num_epochs}, "
            f"energy: {train_energy:10.4f}, "
            f"accuracy: {train_acc * 100:6.2f}%"
        )

    # --- Train ---
    print("\n" + "=" * 70)
    print(f"Training — {args.num_epochs} epochs, {args.infer_steps} inference steps")
    print("=" * 70 + "\n")

    start_time = time.time()
    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config={"num_epochs": args.num_epochs},
        rng_key=train_key,
        verbose=False,
        iter_callback=iter_cb,
        epoch_callback=epoch_cb,
    )
    total_time = time.time() - start_time
    print(f"Training time: {total_time:.1f}s")

    # --- Evaluate on test set ---
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)

    test_loader = HopfieldMnistLoader(
        test_subset_imgs, test_subset_lbls, num_classes,
        batch_size=args.batch_size, shuffle=False,
    )
    eval_key = jax.random.PRNGKey(99)
    test_metrics = evaluate_pcn(
        trained_params, structure, test_loader, {}, eval_key
    )

    print(f"  Test Energy:   {test_metrics['energy']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"\n{len(structure.nodes)} nodes, {len(structure.edges)} edges, "
          f"{total_params:,} parameters")
    print("\nDone.")


if __name__ == "__main__":
    main()
