"""
MINIMAL Predictive Coding Network Example
================================================

This is the absolute SIMPLEST example showing how to:
1. Define a network with a dictionary
2. Train it on MNIST
3. Get results

Total code: ~60 lines. That's it!
"""

import jax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os

from fabricpc.graph import create_pc_graph
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.training.data_utils import OneHotWrapper

os.environ["JAX_PLATFORMS"] = "cuda"  # change to "cpu", "cuda" or "tpu" if available

# jax.config.update("jax_traceback_filtering", "off")

# Set random seed for reproducibility
jax.config.update('jax_default_prng_impl', 'threefry2x32')  # 'rbg' is faster than 'threefry2x32', but less reproducible across vmap
master_rng_key = jax.random.PRNGKey(0)
import torch
import numpy as np
# Set seeds for torch data loaders
torch.manual_seed(0)
np.random.seed(0)

# Split keys for different stages
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

# ==============================================================================
# NETWORK DEFINITION: A dictionary!
# ==============================================================================
# fmt: off

config = {
    # Define nodes (layers)
    "node_list": [
        {"name": "pixels",  "dim": 784, "type": "linear", "activation": {"type": "identity"}},
        {"name": "hidden1", "dim": 256, "type": "linear", "activation": {"type": "sigmoid"}},
        {"name": "hidden2", "dim": 64,  "type": "linear", "activation": {"type": "sigmoid"}},
        {"name": "class",   "dim": 10,  "type": "linear", "activation": {"type": "sigmoid"}},
    ],

    # Connect nodes with edges
    "edge_list": [
        {"source_name": "pixels",  "target_name": "hidden1", "slot": "in"},
        {"source_name": "hidden1", "target_name": "hidden2", "slot": "in"},
        {"source_name": "hidden2", "target_name": "class",   "slot": "in"},
    ],

    # Map data loader tasks to nodes
    "task_map": {"x": "pixels", "y": "class"},
}

# Training hyperparameters
train_config = {
    "num_epochs": 20,       # Number of training epochs
    "infer_steps": 20,      # Inference steps
    "eta_infer": 0.05,      # Inference learning rate
    "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.001},
}
batch_size=200

# fmt: on
# ==============================================================================
# CREATE MODEL: One line!
# ==============================================================================
params, structure = create_pc_graph(config, graph_key)

print(f"Model created: {len(config['node_list'])} nodes, {len(config['edge_list'])} edges")
print(f"Total parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")
print("os.fork warning is harmless - due to PyTorch data loaders.")

# ==============================================================================
# LOAD DATA
# ==============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16)

train_loader = OneHotWrapper(train_loader)
test_loader = OneHotWrapper(test_loader)

# ==============================================================================
# TRAIN (with automatic JIT compilation!)
# ==============================================================================

print("\nTraining (JIT compilation on first batch)...")
start_time = time.time()
trained_params, energy_history, _ = train_pcn(
    params=params,
    structure=structure,
    train_loader=train_loader,
    config=train_config,
    rng_key=train_key,
    verbose=True,
)
delta_t = time.time() - start_time
print(f"Avg Training time: {delta_t / train_config["num_epochs"]:.2f} seconds per epoch")

# ==============================================================================
# EVALUATE
# ==============================================================================

print("\nEvaluating...")
metrics = evaluate_pcn(trained_params, structure, test_loader, train_config, eval_key)
print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Test Loss: {metrics['loss']:.4f}")

print("\n" + "="*70)
print("That's it! Want to change the architecture?")
print("Just modify the config dictionary above:")
print("  - Add more nodes to node_list for deeper networks")
print("  - Change 'dim' values to make layers wider/narrower")
print("  - Modify edge_list to create different connection patterns")
print("  - No need to change any other code!")
print("\nJAX Benefits:")
print("  ✓ Automatic JIT compilation (10-20x speedup)")
print("  ✓ Functional programming (easier to debug)")
print("  ✓ Multi-GPU ready (just add pmap!)")
print("  ✓ TPU support out of the box")
print("="*70)
