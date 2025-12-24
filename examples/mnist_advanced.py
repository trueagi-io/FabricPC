"""
ADVANCED Predictive Coding Network Example
=================================================

This example demonstrates:
1. Deeper network architectures
2. Custom training configurations
3. Progress monitoring and checkpointing
4. Hyperparameter exploration

Compared to mnist_demo.py, this shows a customizable training loop
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from fabricpc.graph import create_pc_graph
from fabricpc.training import train_step, create_optimizer, evaluate_pcn
from fabricpc.training.data_utils import OneHotWrapper

# Set random seed and split for different stages
jax.config.update('jax_default_prng_impl', 'threefry2x32')  # 'rbg' is faster than 'threefry2x32', but less reproducible across vmaps
master_rng_key = jax.random.PRNGKey(42)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)
import torch
import numpy as np
# Set seeds for torch data loaders
torch.manual_seed(0)
np.random.seed(0)

# ==============================================================================
# ADVANCED NETWORK CONFIGURATION - FULLY EXPLICIT (NO DEFAULTS)
# ==============================================================================
# fmt: off

# Detailed node configuration template
template_node = {
    "name": None,   # To be filled -> str
    "shape": None,  # To be filled -> tuple of ints
    "type": "linear",
    "activation": {"type": "sigmoid"},
    "energy": {"type": "gaussian", "precision": 1.0},
    "weight_init": {"type": "normal", "mean": 0.0, "std": 0.05},
    "use_bias": True,
    "flatten_input": False,
    "latent_init": None,  # Use graph-level default}
}

config = {
    # Deeper 3-hidden-layer network with fully explicit configs
    "node_list": [
        {   **template_node,
            "name": "pixels",  # Override template fields
            "shape": (784,),
            "activation": {"type": "identity"},
        },
        {   **template_node,
            "name": "h1",
            "shape": (256,),
        },
        {   **template_node,
            "name": "h2",
            "shape": (128,),
        },
        {   **template_node,
            "name": "h3",
            "shape": (64,),
        },
        {   **template_node,
            "name": "class",
            "shape": (10,),
        },
    ],

    "edge_list": [
        {"source_name": "pixels", "target_name": "h1",    "slot": "in"},
        {"source_name": "h1",     "target_name": "h2",    "slot": "in"},
        {"source_name": "h2",     "target_name": "h3",    "slot": "in"},
        {"source_name": "h3",     "target_name": "class", "slot": "in"},
    ],

    "task_map": {"x": "pixels", "y": "class"},

    # Graph-level state initialization (feedforward with normal fallback)
    "graph_state_initializer": {"type": "feedforward"},
}

# More sophisticated training configuration
train_config = {
    "infer_steps": 20,     # More inference steps for deeper network
    "eta_infer": 0.05,     # Inference learning rate
    "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.001},
}
batch_size = 200
num_epochs = 10

# fmt: on
# ==============================================================================
# CREATE MODEL
# ==============================================================================

print("="*70)
print("Predictive Coding - Advanced MNIST Example")
print("="*70)

params, structure = create_pc_graph(config, graph_key)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

print(f"\n[Model Architecture]")
print(f"  Nodes: {len(config['node_list'])}")
print(f"  Edges: {len(config['edge_list'])}")
print(f"  Total parameters: {num_params:,}")
print(f"\n  Layer sizes: ", end="")
for node in config['node_list']:
    print(f"{node['shape']} → ", end="")
print("(output)")

# ==============================================================================
# LOAD DATA
# ==============================================================================

print(f"\n[Data Loading]")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x_dat: x_dat.view(-1)),
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = OneHotWrapper(DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2
))
test_loader = OneHotWrapper(DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=2
))

print(f"  Train samples: {len(train_data):,}")
print(f"  Test samples: {len(test_data):,}")
print(f"  Batch size: {batch_size}")
print(f"  Train batches: {len(train_loader)}")

# ==============================================================================
# CUSTOM TRAINING LOOP WITH MONITORING
# ==============================================================================

print(f"\n[Training Configuration]")
print(f"  Optimizer: {train_config['optimizer']['type']}")
print(f"  Learning rate: {train_config['optimizer']['lr']}")
print(f"  Weight decay: {train_config['optimizer']['weight_decay']}")
print(f"  Inference steps: {train_config['infer_steps']}")
print(f"  Inference eta: {train_config['eta_infer']}")

infer_steps = train_config['infer_steps']
eta_infer = train_config['eta_infer']

# Create optimizer
optimizer = create_optimizer(train_config['optimizer'])
opt_state = optimizer.init(params)

# JIT compile training step
print(f"\n[Compiling JIT functions...]")
jit_train_step = jax.jit(
    lambda p, o, b, k: train_step(p, o, b, structure, optimizer, k, infer_steps, eta_infer)
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
        params, opt_state, energy, _ = jit_train_step(params, opt_state, batch, all_rng_keys[epoch, batch_idx])
        epoch_energies.append(float(energy))

        # Progress indicator every n_batch_update batches
        n_batch_update = 100
        if (batch_idx + 1) % n_batch_update == 0:
            avg_energy = sum(epoch_energies[-n_batch_update:]) / len(epoch_energies[-n_batch_update:])
            print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, energy: {avg_energy:.4f}")

    epoch_time = time.time() - epoch_start
    avg_energy = sum(epoch_energies) / len(epoch_energies)

    # Evaluate on test set with unique eval key for this epoch
    epoch_eval_key, eval_key = jax.random.split(eval_key)
    metrics = evaluate_pcn(params, structure, test_loader, train_config, epoch_eval_key)
    accuracy = metrics['accuracy'] * 100  # convert to percentage

    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        print(f"  ★ New best accuracy: {accuracy:.2f}%")

    print(f"  Epoch {epoch+1}/{num_epochs} - energy: {avg_energy:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.1f}s")

    training_history.append({
        'epoch': epoch + 1,
        'energy': avg_energy,
        'accuracy': accuracy,
        'time': epoch_time
    })

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
    print(f"  {h['epoch']:5d} | {h['energy']:7.4f} | {h['accuracy']:7.2f}% | {h['time']:4.1f}s")

print("\n" + "="*70)
print("Advanced Training Complete!")
print("\nKey takeaways:")
print("  ✓ JIT compilation makes training fast")
print("  ✓ Monitoring and checkpointing are easy")
print("\nNext steps:")
print("  - Try even deeper architectures")
print("  - Experiment with different optimizers")
print("  - Add learning rate scheduling")
print("="*70)
