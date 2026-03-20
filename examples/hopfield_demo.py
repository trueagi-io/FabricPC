"""
Hopfield Network on Binarized MNIST — FabricPC Demo
====================================================

Trains a Hopfield memory network on binarized MNIST using predictive
coding weight learning (not one-shot Hebbian). Each epoch is a full pass
over the training data, updating weights via PC local gradients.

Architecture:
    input_node ──[W_in]──> HopfieldNode ──────> feedback_node
                                ^                      |
                                └────[W_hop]───────────┘

During training, both input and feedback (target) are clamped to the
same pattern. The network learns weights that minimize PC energy.
During recall, only input is clamped; the recurrent loop reconstructs
the nearest stored pattern.

Usage:
    python examples/hopfield_demo.py
    python examples/hopfield_demo.py --num_epochs 30
    python examples/hopfield_demo.py --digits 0 1 2 3 4
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
from fabricpc.nodes import IdentityNode
from fabricpc.nodes.hopfield import HopfieldNode, recall_with_energy
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph.state_initializer import GlobalStateInit
from fabricpc.training import train_pcn
from fabricpc.utils.data.binarized_mnist import (
    load_binarized_mnist,
    get_unique_digit_prototypes,
)


# =========================================================================
# Data Loader
# =========================================================================


class BipolarMnistLoader:
    """Yields (image, image) batches for Hopfield training (x = y = pattern)."""

    def __init__(self, images, batch_size, shuffle=True, seed=42):
        self.images = images
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
            batch = jnp.array(self.images[idx[s:e]])
            yield batch, batch  # input = target

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
        "--num_epochs", type=int, default=50,
        help="Number of training epochs (default: 50)",
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
        "--n_train", type=int, default=100,
        help="Training images per digit (default: 100)",
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

    print("=" * 70)
    print("Hopfield Network on Binarized MNIST — FabricPC")
    print("=" * 70)

    # --- Load dataset ---
    print("\nLoading binarized MNIST...")
    train_images, train_labels, test_images, test_labels = load_binarized_mnist()
    print(f"  Train: {train_images.shape}  Test: {test_images.shape}")

    # --- Create digit prototypes (for accuracy measurement) ---
    digits = args.digits
    prototypes, digit_ids = get_unique_digit_prototypes(
        train_images, train_labels, digits=digits
    )
    prototypes = jnp.array(prototypes)
    P, N = prototypes.shape
    digit_ids_arr = jnp.array(digit_ids)

    print(f"  Digits: {digit_ids}, pattern size: {N}")

    # --- Sample train and test subsets ---
    train_subset = []
    for d in digit_ids:
        mask = train_labels == d
        idx = np.where(mask)[0][: args.n_train]
        train_subset.append(train_images[idx])
    train_subset = np.concatenate(train_subset)

    test_subset_imgs = []
    test_subset_lbls = []
    for d in digit_ids:
        mask = test_labels == d
        idx = np.where(mask)[0][: args.n_test]
        test_subset_imgs.append(test_images[idx])
        test_subset_lbls.append(test_labels[idx])
    test_subset_imgs = jnp.array(np.concatenate(test_subset_imgs))
    test_subset_lbls = np.concatenate(test_subset_lbls)
    n_test_total = len(test_subset_imgs)

    print(f"  Train subset: {len(train_subset)} images ({args.n_train} per digit)")
    print(f"  Test subset:  {n_test_total} images ({args.n_test} per digit)")

    # --- Build Hopfield graph (with y target for training) ---
    print("\nBuilding Hopfield network...")
    input_node = IdentityNode(shape=(N,), name="input")
    memory_node = HopfieldNode(shape=(N,), name="memory")
    feedback_node = IdentityNode(shape=(N,), name="feedback")

    structure = graph(
        nodes=[input_node, memory_node, feedback_node],
        edges=[
            Edge(source=input_node, target=memory_node.slot("in")),
            Edge(source=memory_node, target=feedback_node.slot("in")),
            Edge(source=feedback_node, target=memory_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=feedback_node),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=args.infer_steps),
        graph_state_initializer=GlobalStateInit(),
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
    train_loader = BipolarMnistLoader(
        train_subset, batch_size=args.batch_size, shuffle=True
    )
    optimizer = optax.adam(args.lr)

    # Track per-epoch training energy
    batch_energy_tracker = []

    def iter_cb(_epoch_idx, _batch_idx, energy):
        normalized = float(energy) / args.batch_size
        batch_energy_tracker.append(normalized)
        return normalized

    def epoch_cb(epoch_idx, params, structure, _config, rng_key):
        # Training energy from tracked batches
        train_energy = (
            sum(batch_energy_tracker) / len(batch_energy_tracker)
            if batch_energy_tracker
            else 0.0
        )
        batch_energy_tracker.clear()

        # Training accuracy via recall on train set
        correct = 0
        total = 0
        for start in range(0, len(train_subset), args.batch_size):
            end = min(start + args.batch_size, len(train_subset))
            batch_imgs = jnp.array(train_subset[start:end])
            batch_lbls = train_labels_subset[start:end]

            rng_key, rk = jax.random.split(rng_key)
            recalled, _ = recall_with_energy(params, structure, batch_imgs, rk)

            sign_recalled = jnp.sign(recalled)
            overlaps = sign_recalled @ prototypes.T / N
            predicted_idx = jnp.argmax(overlaps, axis=1)
            predicted_digits = digit_ids_arr[predicted_idx]

            correct += int(jnp.sum(predicted_digits == batch_lbls))
            total += end - start

        train_acc = correct / total
        print(
            f"Epoch {epoch_idx + 1:3d}/{args.num_epochs}, "
            f"energy: {train_energy:10.4f}, "
            f"accuracy: {train_acc * 100:6.2f}%"
        )

    # Build train labels for accuracy eval
    train_labels_subset = []
    for d in digit_ids:
        mask = train_labels == d
        idx = np.where(mask)[0][: args.n_train]
        train_labels_subset.append(train_labels[idx])
    train_labels_subset = np.concatenate(train_labels_subset)

    # --- Train ---
    print("\n" + "=" * 70)
    print(f"Training — {args.num_epochs} epochs, {args.infer_steps} inference steps, "
          f"noise=0.00")
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

    correct = 0
    total = 0
    total_energy = 0.0
    eval_key = jax.random.PRNGKey(99)

    for start in range(0, n_test_total, args.batch_size):
        end = min(start + args.batch_size, n_test_total)
        batch_imgs = test_subset_imgs[start:end]
        batch_lbls = test_subset_lbls[start:end]

        eval_key, rk = jax.random.split(eval_key)
        recalled, energy = recall_with_energy(
            trained_params, structure, batch_imgs, rk
        )

        sign_recalled = jnp.sign(recalled)
        overlaps = sign_recalled @ prototypes.T / N
        predicted_idx = jnp.argmax(overlaps, axis=1)
        predicted_digits = digit_ids_arr[predicted_idx]

        correct += int(jnp.sum(predicted_digits == batch_lbls))
        total_energy += float(jnp.sum(energy))
        total += end - start

    test_energy = total_energy / total
    accuracy = correct / total

    print(f"  Test Energy:   {test_energy:.4f}")
    print(f"  Test Accuracy: {accuracy * 100:.2f}%")
    print(f"\n{len(structure.nodes)} nodes, {len(structure.edges)} edges, "
          f"{total_params:,} parameters")
    print("\nDone.")


if __name__ == "__main__":
    main()
