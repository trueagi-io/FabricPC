"""
Multi-GPU MNIST Example
==============================

This example demonstrates data-parallel training across multiple GPUs using pmap.

Key features:
- Automatic device detection
- Batch sharding across GPUs
- Gradient averaging with pmean
- Linear scaling with number of devices

Note: This will work with 1 GPU (falls back to single-GPU) but the real benefits
come with 2+ GPUs.
"""

import os  # set environment variables before importing JAX

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_PLATFORMS", "cuda"
)  # options: "cpu", "cuda" or "tpu" if available
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress XLA warnings
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import time

from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
import optax
from fabricpc.training import train_pcn_multi_gpu, evaluate_pcn_multi_gpu
from fabricpc.utils.data.dataloader import MnistLoader

# Set random seed
master_rng_key = jax.random.PRNGKey(0)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# fmt: off

pixels = Linear(shape=(784,), activation=IdentityActivation(), name="pixels")
hidden1 = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1")
hidden2 = Linear(shape=(64,), activation=SigmoidActivation(), name="hidden2")
class_node = Linear(shape=(10,), activation=SoftmaxActivation(), energy=CrossEntropyEnergy(), name="class")


structure = graph(
    nodes=[pixels, hidden1, hidden2, class_node],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=hidden2.slot("in")),
        Edge(source=hidden2, target=class_node.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=class_node),
)

optimizer = optax.adamw(0.001, weight_decay=0.001)
train_config = {
    "num_epochs": 20,
    "infer_steps": 20,
    "eta_infer": 0.05,
}

# fmt: on
# ==============================================================================
# DEVICE INFORMATION
# ==============================================================================

print("=" * 70)
print("Multi-GPU Predictive Coding - MNIST")
print("=" * 70)

n_devices = jax.device_count()
devices = jax.devices()

print(f"\n[Device Information]")
print(f"  Total devices: {n_devices}")
print(f"  Device types: {[d.device_kind for d in devices]}")
print(f"  Device IDs: {[d.id for d in devices]}")

if n_devices == 1:
    print(f"\n  ⚠ Only 1 device available.")
    print(f"     To see multi-GPU benefits, run on a machine with multiple GPUs")
else:
    print(f"\n  ✓ Multi-GPU training enabled!")
    print(f"     Expected speedup: ~{n_devices}x (linear scaling)")

# ==============================================================================
# CREATE MODEL
# ==============================================================================

print(f"\n[Model Architecture]")
params = initialize_params(structure, graph_key)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

print(f"  Nodes: {len(structure.nodes)}")
print(f"  Edges: {len(structure.edges)}")
print(f"  Parameters: {num_params:,}")

# ==============================================================================
# LOAD DATA
# ==============================================================================

print(f"\n[Data Loading]")

# Important: Batch size should be divisible by number of devices!
batch_size = 200 * n_devices  # Scale batch size with number of devices
print(f"  Batch size: {batch_size} ({batch_size // n_devices} per device)")

train_loader = MnistLoader(
    "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
)
test_loader = MnistLoader(
    "test", batch_size=batch_size, tensor_format="flat", shuffle=False
)

print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ==============================================================================
# TRAIN (Multi-GPU)
# ==============================================================================

print(f"\n[Training Configuration]")
print(f"  Epochs: {train_config['num_epochs']}")
print(f"  Optimizer: adamw")
print(f"  Learning rate: 0.001")
print(f"  Inference steps: {train_config['infer_steps']}")

print(f"\n[Training on {n_devices} device(s)]")
print("  (First batch will be slow due to pmap compilation)\n")

start_time = time.time()
trained_params = train_pcn_multi_gpu(
    params=params,
    structure=structure,
    train_loader=train_loader,
    optimizer=optimizer,
    config=train_config,
    rng_key=train_key,
    verbose=True,
)
training_time = time.time() - start_time

print(f"\n  Total training time: {training_time:.1f}s")
print(f"  Average time per epoch: {training_time / train_config['num_epochs']:.1f}s")
print(
    f"  Throughput: {train_loader.num_examples * train_config['num_epochs'] / training_time:.0f} samples/sec"
)

# ==============================================================================
# EVALUATE
# ==============================================================================

print(f"\n[Evaluation]")
metrics = evaluate_pcn_multi_gpu(
    trained_params, structure, test_loader, train_config, eval_key
)
print(f"  Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("Multi-GPU Training Complete!")
print("=" * 70)

if n_devices > 1:
    print(f"\n✓ Successfully trained on {n_devices} GPUs using pmap")
    print(f"✓ Automatic batch sharding across devices")
    print(f"✓ Gradient averaging with pmean")
    print(
        f"✓ Throughput: {train_loader.num_examples * train_config['num_epochs'] / training_time:.0f} samples/sec"
    )

    print(f"\nScaling Efficiency:")
    print(f"  - Linear scaling expected: {n_devices}x speedup")
    print(f"  - Actual throughput scaled by batch size increase")
else:
    print(f"\n✓ Trained on single GPU (multi-GPU code is ready)")
    print(f"  Run on multi-GPU machine to see parallelization benefits")

print("\nKey Takeaways:")
print("  • pmap provides data parallelism with minimal code changes")
print("  • Batch size scales with number of devices")
print("  • Gradients are automatically averaged across devices")
print("  • Nearly linear speedup with multiple GPUs")
print("=" * 70)
