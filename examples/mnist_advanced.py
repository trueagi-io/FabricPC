"""
ADVANCED Predictive Coding Network Example
=================================================

This example demonstrates:
1. Object-oriented network construction with direct node instantiation
2. Custom training configurations
3. Progress monitoring and checkpointing
4. Hyperparameter exploration

Compared to mnist_demo.py, this shows a customizable training loop
using the new object-oriented API with node objects.
"""

import os  # set environment variables before importing JAX

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_PLATFORMS", "cuda"
)  # options: "cpu", "cuda" or "tpu" if available
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress XLA warnings
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import time

from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params, FeedforwardStateInit
from fabricpc.core.activations import IdentityActivation, SigmoidActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import train_step, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader

# Set random seed and split for different stages
jax.config.update(
    "jax_default_prng_impl", "threefry2x32"
)  # 'rbg' is faster than 'threefry2x32', but less reproducible across vmaps
master_rng_key = jax.random.PRNGKey(42)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

# ==============================================================================
# ADVANCED NETWORK CONFIGURATION - OBJECT-ORIENTED API
# ==============================================================================

# Create nodes with explicit configuration
pixels = Linear(shape=(784,), activation=IdentityActivation(), name="pixels")
h1 = Linear(
    shape=(256,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=NormalInitializer(mean=0.0, std=0.05),
    name="h1",
)
h2 = Linear(
    shape=(128,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=NormalInitializer(mean=0.0, std=0.05),
    name="h2",
)
h3 = Linear(
    shape=(64,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=NormalInitializer(mean=0.0, std=0.05),
    name="h3",
)
class_node = Linear(
    shape=(10,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=NormalInitializer(mean=0.0, std=0.05),
    name="class",
)

structure = graph(
    nodes=[pixels, h1, h2, h3, class_node],
    edges=[
        Edge(source=pixels, target=h1.slot("in")),
        Edge(source=h1, target=h2.slot("in")),
        Edge(source=h2, target=h3.slot("in")),
        Edge(source=h3, target=class_node.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=class_node),
    graph_state_initializer=FeedforwardStateInit(),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# More sophisticated training configuration
optimizer = optax.adamw(0.001, weight_decay=0.001)
train_config = {}
batch_size = 200
num_epochs = 10

# ==============================================================================
# CREATE MODEL
# ==============================================================================

print("=" * 70)
print("Predictive Coding - Advanced MNIST Example")
print("=" * 70)

params = initialize_params(structure, graph_key)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

print(f"\n[Model Architecture]")
print(f"  Nodes: {len(structure.nodes)}")
print(f"  Edges: {len(structure.edges)}")
print(f"  Total parameters: {num_params:,}")
print(f"\n  Layer sizes: ", end="")
for node in structure.nodes.values():
    print(f"{node.node_info.shape} → ", end="")
print("(output)")

# ==============================================================================
# LOAD DATA
# ==============================================================================

print(f"\n[Data Loading]")

train_loader = MnistLoader(
    "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
)
test_loader = MnistLoader(
    "test", batch_size=batch_size, tensor_format="flat", shuffle=False
)

print(f"  Train samples: {train_loader.num_examples:,}")
print(f"  Test samples: {test_loader.num_examples:,}")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")

# ==============================================================================
# CUSTOM TRAINING LOOP WITH MONITORING
# ==============================================================================

print(f"\n[Training Configuration]")
print(f"  Optimizer: adamw")
print(f"  Learning rate: 0.001")
print(f"  Weight decay: 0.001")
print(f"  Inference steps: {structure.config['inference'].config['infer_steps']}")
print(f"  Inference eta: {structure.config['inference'].config['eta_infer']}")

opt_state = optimizer.init(params)

# JIT compile training step
print(f"\n[Compiling JIT functions...]")
jit_train_step = jax.jit(
    lambda p, o, b, k: train_step(p, o, b, structure, optimizer, k)
)

# Training loop with detailed monitoring
print(f"\n[Training for {num_epochs} epochs]")
print("  (First batch will be slow due to JIT compilation)\n")

best_accuracy = 0.0
training_history = []

# Prepare keys for all epochs and batches
num_batches = len(train_loader)
# split keys into (num_epochs x num_batches)
all_rng_keys = jax.random.split(train_key, num_epochs * num_batches)
all_rng_keys = all_rng_keys.reshape((num_epochs, num_batches, 2))

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_energies = []

    for batch_idx, (x, y) in enumerate(train_loader):
        batch = {"x": jnp.array(x), "y": y}

        # Training step with unique rng_key
        params, opt_state, energy, _ = jit_train_step(
            params, opt_state, batch, all_rng_keys[epoch, batch_idx]
        )
        epoch_energies.append(float(energy))

        # Progress indicator every n_batch_update batches
        n_batch_update = 100
        if (batch_idx + 1) % n_batch_update == 0:
            avg_energy = sum(epoch_energies[-n_batch_update:]) / len(
                epoch_energies[-n_batch_update:]
            )
            print(
                f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, energy: {avg_energy:.4f}"
            )

    epoch_time = time.time() - epoch_start
    avg_energy = sum(epoch_energies) / len(epoch_energies)

    # Evaluate on test set with unique eval key for this epoch
    epoch_eval_key, eval_key = jax.random.split(eval_key)
    metrics = evaluate_pcn(params, structure, test_loader, train_config, epoch_eval_key)
    accuracy = metrics["accuracy"] * 100  # convert to percentage

    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        print(f"  ★ New best accuracy: {accuracy:.2f}%")

    print(
        f"  Epoch {epoch+1}/{num_epochs} - energy: {avg_energy:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.1f}s"
    )

    training_history.append(
        {
            "epoch": epoch + 1,
            "energy": avg_energy,
            "accuracy": accuracy,
            "time": epoch_time,
        }
    )

# ==============================================================================
# FINAL EVALUATION
# ==============================================================================

print(f"\n[Final Results]")
print(f"  Best accuracy: {best_accuracy:.2f}%")
print(f"  Final accuracy: {training_history[-1]['accuracy']:.2f}%")
print(f"  Total training time: {sum(h['time'] for h in training_history):.1f}s")

# Print training history
print(f"\n[Training History]")
print("  Epoch | Energy    | Accuracy | Time")
print("  ------|---------|----------|------")
for h in training_history:
    print(
        f"  {h['epoch']:5d} | {h['energy']:7.4f} | {h['accuracy']:7.2f}% | {h['time']:4.1f}s"
    )

print("\n" + "=" * 70)
print("Advanced Training Complete!")
print("\nKey takeaways:")
print("  ✓ JIT compilation makes training fast")
print("  ✓ Monitoring and checkpointing are easy")
print("\nNext steps:")
print("  - Try even deeper architectures")
print("  - Experiment with different optimizers")
print("  - Add learning rate scheduling")
print("=" * 70)
