"""
MNIST Demo with IdentityNode
================================================

This example demonstrates the IdentityNode, which passes input through
unchanged with no learnable parameters. Useful for input nodes or
passthrough connections.

Based on the minimal MNIST demo, with an IdentityNode added between
hidden layers to demonstrate its usage.
"""

import os  # Set environment variables before importing JAX

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")  # "cpu", "cuda" or "tpu"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress XLA warnings
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
import time

# jax.config.update("jax_traceback_filtering", "off")

# Set random seed for reproducibility
jax.config.update(
    "jax_default_prng_impl", "threefry2x32"
)  # 'rbg' is faster than 'threefry2x32', but less reproducible across vmap

# ==============================================================================
# NETWORK DEFINITION: Object API with IdentityNode
# ==============================================================================
# fmt: off

# Create nodes - including an IdentityNode as a passthrough layer
pixels = Linear(shape=(784,), name="pixels")
hidden1 = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1")
passthrough = IdentityNode(shape=(256,), name="passthrough")  # Identity passthrough
hidden2 = Linear(shape=(64,), activation=SigmoidActivation(), name="hidden2")
output = Linear(shape=(10,), activation=SoftmaxActivation(), energy=CrossEntropyEnergy(), name="class")

# Build graph structure
structure = graph(
    nodes=[pixels, hidden1, passthrough, hidden2, output],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=passthrough.slot("in")),
        Edge(source=passthrough, target=hidden2.slot("in")),
        Edge(source=hidden2, target=output.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=output),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# Training hyperparameters
optimizer = optax.adamw(0.001, weight_decay=0.001)
train_config = {
    "num_epochs": 27,       # Number of training epochs
}
batch_size = 200

# fmt: on
if __name__ == "__main__":
    master_rng_key = jax.random.PRNGKey(0)

    # Split keys for different stages
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # ==============================================================================
    # CREATE MODEL: Initialize parameters from structure
    # ==============================================================================
    params = initialize_params(structure, graph_key)

    # ==============================================================================
    # LOAD DATA
    # ==============================================================================

    train_loader = MnistLoader(
        "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, tensor_format="flat", shuffle=False
    )

    # ==============================================================================
    # TRAIN (with automatic JIT compilation!)
    # ==============================================================================

    # A model consists of two parts: the parameters (weights) and the structure (graph architecture). The training loop uses both to perform inference and learning.

    print("\nTraining (JIT compilation on first batch)...")
    start_time = time.time()
    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
    )
    delta_t = time.time() - start_time
    print(
        f"Avg Training time: {delta_t / train_config['num_epochs']:.2f} seconds per epoch"
    )

    # ==============================================================================
    # EVALUATE
    # ==============================================================================

    print("\nEvaluating...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(
        f"Test energy: {metrics['energy']:.4f} Note: Energy will be zero in evaluation mode for graphs that are feed-forward in topology (no cycles) and use feed-forward initialization."
    )

    print(f"Model created: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    print(f"Total parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

    print("\n" + "=" * 70)
    print("This demo shows the IdentityNode in action!")
    print("The 'passthrough' node has no learnable parameters -")
    print("it simply passes data through unchanged.")
    print("\nUseful for:")
    print("  - Input nodes that don't need transformation")
    print("  - Routing data through complex graph topologies")
    print("  - Creating auxiliary connections without adding parameters")
    print("=" * 70)
