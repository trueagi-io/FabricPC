"""
MINIMAL Predictive Coding Network Example
================================================

This is the absolute SIMPLEST example showing how to:
1. Define a network with the new object API
2. Train it on MNIST
3. Get results

Total code: ~60 lines. That's it!
"""

import os  # Set environment variables before importing JAX

from fabricpc.core import InferenceSGDNormClip

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")  # "cpu", "cuda" or "tpu"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress XLA warnings

# Keep deterministic kernels and default to disabling Triton GEMM, which can
# trigger CUDA runtime errors on some GPUs for small/irregular matmuls.
# Triton tiling logic fails when it encounters certain fused operations where dimension bounds are not divisible by the tile size.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_deterministic_ops=true" not in _xla_flags:
    _xla_flags = (_xla_flags + " --xla_gpu_deterministic_ops=true").strip()
if os.environ.get("FABRICPC_DISABLE_TRITON_GEMM", "1") == "1":
    if "--xla_gpu_enable_triton_gemm=false" not in _xla_flags:
        _xla_flags = (_xla_flags + " --xla_gpu_enable_triton_gemm=false").strip()
os.environ["XLA_FLAGS"] = _xla_flags

import argparse
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
from fabricpc.training.natural_gradients import (
    scale_by_natural_gradient_diag,
    scale_by_natural_gradient_layerwise,
)
from fabricpc.utils.data.dataloader import MnistLoader
import time

# jax.config.update("jax_traceback_filtering", "off")

# Set random seed for reproducibility
jax.config.update(
    "jax_default_prng_impl", "threefry2x32"
)  # 'rbg' is faster than 'threefry2x32', but less reproducible across vmap

# ==============================================================================
# NETWORK DEFINITION: Object API
# ==============================================================================
# fmt: off

# Create nodes
pixels = IdentityNode(shape=(784,), name="pixels")
hidden1 = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1")
hidden2 = Linear(shape=(64,), activation=SigmoidActivation(), name="hidden2")
output = Linear(shape=(10,), activation=SoftmaxActivation(), energy=CrossEntropyEnergy(), name="class")

# Build graph structure
structure = graph(
    nodes=[pixels, hidden1, hidden2, output],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=hidden2.slot("in")),
        Edge(source=hidden2, target=output.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=output),  # Tell the trainer which nodes are inputs and targets for supervised learning
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# Training hyperparameters
train_config = {
    "num_epochs": 20,       # Number of training epochs
}
batch_size = 200

# fmt: on

OPTIMIZER_PRESETS = {
    "adam": lambda: optax.chain(optax.add_decayed_weights(0.001), optax.adam(0.001)),
    "adamw": lambda: optax.adamw(0.001, weight_decay=0.001),
    "sgd": lambda: optax.chain(
        optax.add_decayed_weights(0.001), optax.sgd(0.01, momentum=0.9)
    ),
    "ngd_diag": lambda: optax.chain(
        optax.add_decayed_weights(0.001),
        scale_by_natural_gradient_diag(fisher_decay=0.95, damping=1e-3),
        optax.scale(-0.0003),
    ),
    "ngd_layerwise": lambda: optax.chain(
        optax.add_decayed_weights(0.001),
        scale_by_natural_gradient_layerwise(fisher_decay=0.95, damping=1e-3),
        optax.scale(-0.001),
    ),
}


def parse_args() -> argparse.Namespace:
    """
    Parse CLI args.

    Env fallback:
        FABRICPC_OPTIMIZER={adam|adamw|sgd|ngd_diag|ngd_layerwise}
    """
    default_optimizer = os.environ.get("FABRICPC_OPTIMIZER", "adam")
    parser = argparse.ArgumentParser(description="FabricPC MNIST demo")
    parser.add_argument(
        "--optimizer",
        default=default_optimizer,
        help=(
            "Optimizer preset to use. "
            f"Choices: {', '.join(OPTIMIZER_PRESETS.keys())}. "
            "CLI flag overrides FABRICPC_OPTIMIZER."
        ),
    )
    args = parser.parse_args()
    if args.optimizer.lower() not in OPTIMIZER_PRESETS:
        valid = ", ".join(OPTIMIZER_PRESETS.keys())
        parser.error(f"unknown optimizer '{args.optimizer}'. valid choices: {valid}")
    return args


def get_optimizer(name: str) -> optax.GradientTransformation:
    """Return optimizer preset by name."""
    key = name.lower()
    if key not in OPTIMIZER_PRESETS:
        valid = ", ".join(OPTIMIZER_PRESETS.keys())
        raise ValueError(f"unknown optimizer '{name}'. valid choices: {valid}")
    return OPTIMIZER_PRESETS[key]()


if __name__ == "__main__":
    args = parse_args()

    optimizer = get_optimizer(args.optimizer)

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
    print(f"Using optimizer preset: {args.optimizer}")
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
    print("That's it! Want to change the architecture?")
    print("Just modify the node and edge definitions above:")
    print("  - Add more Linear nodes for deeper networks")
    print("  - Change 'shape' values to make layers wider/narrower")
    print("  - Modify edges to create different connection patterns")
    print("  - No need to change any other code!")
    print("\nJAX Benefits:")
    print("  ✓ Automatic JIT compilation (10-20x speedup)")
    print("  ✓ Functional programming (easier to debug)")
    print("  ✓ Multi-GPU ready (just add pmap!)")
    print("  ✓ TPU support out of the box")
    print("=" * 70)
