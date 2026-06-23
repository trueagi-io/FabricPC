"""
Predictive Coding Network — MNIST Demo
=======================================

Train a small predictive coding network on MNIST using the object API.

Architecture::

    pixels(784) ──→ hidden1(256) ──→ hidden2(64) ──→ class(10)
     Identity        Sigmoid          Sigmoid        Softmax+CE

For optimizer selection and advanced controls, see mnist_advanced.py.

Checkpointing
-------------
Run with ``--checkpoint <dir>`` to demonstrate model checkpointing: the demo
trains half the epochs, saves params + optimizer state + structure to <dir>,
reloads them with ``load_checkpoint`` (verifying the params round-trip exactly),
then continues training the remaining epochs from the restored checkpoint::

    python examples/mnist_demo.py --checkpoint /tmp/mnist_ckpt --epochs 6

Results:
Avg training time: 1.37s per epoch
20 epochs
Test Accuracy: 98.14%
4 nodes, 3 edges, 218,058 parameters
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()  # options: "cpu", "cuda", "tpu"

import jax
from fabricpc.nodes import Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
import optax
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc import save_checkpoint, load_checkpoint
from fabricpc.utils.data.dataloader import MnistLoader
import argparse
import time

jax.config.update("jax_default_prng_impl", "threefry2x32")

# --- Network ---

pixels = IdentityNode(shape=(784,), name="pixels")
hidden1 = Linear(
    shape=(256,),
    activation=SigmoidActivation(),
    name="hidden1",
    weight_init=XavierInitializer(),
)
hidden2 = Linear(
    shape=(64,),
    activation=SigmoidActivation(),
    name="hidden2",
    weight_init=XavierInitializer(),
)
output = Linear(
    shape=(10,),
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),
    name="class",
    weight_init=XavierInitializer(),
)

# x= and y= tell the trainer which nodes are inputs and targets
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

# --- Hyperparameters ---

batch_size = 200
optimizer = optax.adamw(0.001, weight_decay=0.1)

# --- Train & Evaluate ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Directory to demonstrate save/load: train half the epochs, save, "
        "reload, and continue training the rest from the checkpoint.",
    )
    args = parser.parse_args()

    master_rng_key = jax.random.PRNGKey(0)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)

    train_loader = MnistLoader(
        "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, tensor_format="flat", shuffle=False
    )

    def evaluate(p, label):
        metrics = evaluate_pcn(p, structure, test_loader, {}, eval_key)
        print(f"{label} Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
        return metrics["accuracy"]

    if args.checkpoint is None:
        # --- Standard run: train all epochs straight through. ---
        print("\nTraining (JIT compilation on first batch)...")
        start_time = time.time()
        trained_params, energy_history, _ = train_pcn(
            params=params,
            structure=structure,
            train_loader=train_loader,
            optimizer=optimizer,
            config={"num_epochs": args.epochs},
            rng_key=train_key,
            verbose=True,
        )
        elapsed = time.time() - start_time
        print(f"Avg training time: {elapsed / args.epochs:.2f}s per epoch")
        print("\nEvaluating...")
        evaluate(trained_params, "")
    else:
        # --- Checkpoint demo: train half -> save -> load -> train the rest. ---
        first = args.epochs // 2
        rest = args.epochs - first

        print(f"\nTraining first {first} epochs...")
        trained_params, _, _ = train_pcn(
            params=params,
            structure=structure,
            train_loader=train_loader,
            optimizer=optimizer,
            config={"num_epochs": first},
            rng_key=train_key,
            verbose=True,
        )
        acc_mid = evaluate(trained_params, "[checkpoint] mid-training")

        # Persist params + optimizer state + structure. NOTE: train_pcn manages
        # its own optimizer state internally and neither returns nor accepts one,
        # so this demo cannot thread real optimizer momentum across the save/load
        # boundary — we capture a fresh opt_state purely to exercise the full
        # save/load API. Bitwise optimizer-state resume parity (continuing with
        # the *restored* opt_state) is proven in tests/test_serialization.py
        # (test_resume_training_parity) via the lower-level train_step.
        opt_state = optimizer.init(trained_params)
        save_checkpoint(
            args.checkpoint,
            trained_params,
            opt_state=opt_state,
            structure=structure,
            metadata={"epochs_so_far": first},
            overwrite=True,
        )
        print(f"[checkpoint] saved to {args.checkpoint}")

        # Reload — structure comes back from disk, so we don't need the code that
        # built it. Pass the optimizer so its state pytree can be reconstructed.
        loaded = load_checkpoint(args.checkpoint, optimizer=optimizer)
        matches = all(
            bool((a == b).all())
            for a, b in zip(
                jax.tree_util.tree_leaves(trained_params),
                jax.tree_util.tree_leaves(loaded.params),
            )
        )
        print(f"[checkpoint] reloaded; params identical to pre-save: {matches}")

        print(f"\nContinuing {rest} more epochs from the checkpoint...")
        trained_params, _, _ = train_pcn(
            params=loaded.params,
            structure=loaded.structure,
            train_loader=train_loader,
            optimizer=optimizer,
            config={"num_epochs": rest},
            rng_key=train_key,
            verbose=True,
        )
        acc_final = evaluate(trained_params, "[checkpoint] final")
        print(
            f"[checkpoint] accuracy {acc_mid * 100:.2f}% (mid) -> "
            f"{acc_final * 100:.2f}% (after resuming)"
        )

    print(
        f"\n{len(structure.nodes)} nodes, {len(structure.edges)} edges, "
        f"{sum(p.size for p in jax.tree_util.tree_leaves(params)):,} parameters"
    )
