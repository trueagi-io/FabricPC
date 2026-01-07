"""
MNIST with Aim Experiment Tracking
==================================

This example demonstrates comprehensive experiment tracking for predictive
coding networks using Aim. It tracks:

1. Batch and epoch-level loss/accuracy metrics
2. Weight distributions per layer per epoch
3. Latent state distributions (z_latent, z_mu, pre_activation)
4. Per-node energy values
5. Inference dynamics: how energy and gradients evolve over inference steps

This provides deep insights for debugging and tuning predictive coding models.

After running, launch the Aim UI with:
    aim up

Requirements:
    pip install fabricpc[viz]  # Includes aim
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fabricpc.graph import create_pc_graph
from fabricpc.training import create_optimizer, evaluate_pcn
from fabricpc.training.data_utils import OneHotWrapper

# Import dashboarding utilities
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    is_aim_available,
    train_step_with_history,
    summarize_inference_convergence,
)

# Check if Aim is available
if not is_aim_available():
    print("WARNING: Aim is not installed. Install with: pip install aim")
    print("Tracking will be disabled. Continuing with training only...")
    TRACKING_ENABLED = False
else:
    TRACKING_ENABLED = True
    print("Aim is available. Experiment tracking enabled.")

# Set random seeds
jax.config.update("jax_default_prng_impl", "threefry2x32")
master_rng_key = jax.random.PRNGKey(42)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)
torch.manual_seed(0)
np.random.seed(0)

# ==============================================================================
# NETWORK CONFIGURATION
# ==============================================================================

template_node = {
    "name": None,
    "shape": None,
    "type": "linear",
    "activation": {"type": "sigmoid"},
    "energy": {"type": "gaussian", "precision": 1.0},
    "weight_init": {"type": "normal", "mean": 0.0, "std": 0.05},
    "use_bias": True,
    "flatten_input": False,
    "latent_init": None,
}

config = {
    "node_list": [
        {
            **template_node,
            "name": "pixels",
            "shape": (784,),
            "activation": {"type": "identity"},
        },
        {**template_node, "name": "h1", "shape": (256,)},
        {**template_node, "name": "h2", "shape": (128,)},
        {**template_node, "name": "h3", "shape": (64,)},
        {**template_node, "name": "class", "shape": (10,)},
    ],
    "edge_list": [
        {"source_name": "pixels", "target_name": "h1", "slot": "in"},
        {"source_name": "h1", "target_name": "h2", "slot": "in"},
        {"source_name": "h2", "target_name": "h3", "slot": "in"},
        {"source_name": "h3", "target_name": "class", "slot": "in"},
    ],
    "task_map": {"x": "pixels", "y": "class"},
    "graph_state_initializer": {
        "type": "feedforward",
        "fallback": {"type": "normal", "mean": 0.0, "std": 0.05},
    },
}

train_config = {
    "infer_steps": 20,
    "eta_infer": 0.05,
    "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.001},
}
batch_size = 200
num_epochs = 10

# ==============================================================================
# CREATE MODEL
# ==============================================================================

print("=" * 70)
print("MNIST with Aim Experiment Tracking")
print("=" * 70)

params, structure = create_pc_graph(config, graph_key)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

print(f"\n[Model Architecture]")
print(f"  Nodes: {len(config['node_list'])}")
print(f"  Total parameters: {num_params:,}")

# ==============================================================================
# LOAD DATA
# ==============================================================================

print(f"\n[Data Loading]")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x_dat: x_dat.view(-1)),
    ]
)

train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = OneHotWrapper(
    DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
)
test_loader = OneHotWrapper(
    DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
)

print(f"  Train samples: {len(train_data):,}")
print(f"  Test samples: {len(test_data):,}")
print(f"  Batch size: {batch_size}")

# ==============================================================================
# SETUP AIM TRACKING
# ==============================================================================

if TRACKING_ENABLED:
    print(f"\n[Setting up Aim Tracking]")

    tracking_config = TrackingConfig(
        experiment_name="mnist_pcn_tracking",
        run_name=f"5layer_lr{train_config['optimizer']['lr']}_infer{train_config['infer_steps']}",
        # Batch-level tracking
        track_batch_loss=True,
        track_batch_energy_per_node=True,
        # Epoch-level tracking
        track_epoch_loss=True,
        track_epoch_accuracy=True,
        track_weight_distributions=True,
        track_latent_distributions=True,
        track_preactivation_distributions=True,
        track_activation_distributions=True,
        # Inference dynamics
        track_inference_dynamics=True,
        inference_nodes_to_track=["h1", "h2", "h3", "class"],
        # Frequency controls
        weight_distribution_every_n_epochs=1,
        latent_distribution_every_n_batches=50,
    )

    tracker = AimExperimentTracker(config=tracking_config)

    # Log hyperparameters
    tracker.log_hyperparams(
        {
            "model_config": {
                "num_layers": len(config["node_list"]),
                "layer_sizes": [n["shape"][0] for n in config["node_list"]],
                "activation": "sigmoid",
                "energy_type": "gaussian",
            },
            "train_config": train_config,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
    )

    # Log graph structure
    tracker.log_graph_structure(structure)
    print("  Tracker initialized successfully")
else:
    tracker = None

# ==============================================================================
# TRAINING WITH DETAILED TRACKING
# ==============================================================================

print(f"\n[Training Configuration]")
print(f"  Optimizer: {train_config['optimizer']['type']}")
print(f"  Learning rate: {train_config['optimizer']['lr']}")
print(f"  Inference steps: {train_config['infer_steps']}")

infer_steps = train_config["infer_steps"]
eta_infer = train_config["eta_infer"]

optimizer = create_optimizer(train_config["optimizer"])
opt_state = optimizer.init(params)

# JIT compile training step with history
print(f"\n[Compiling JIT functions...]")
jit_train_step = jax.jit(
    lambda p, o, b, k: train_step_with_history(
        p,
        o,
        b,
        structure,
        optimizer,
        k,
        infer_steps,
        eta_infer,
        collect_every=5,  # Collect every 5th inference step
    )
)

print(f"\n[Training for {num_epochs} epochs with full tracking]")
print("  (First batch will be slow due to JIT compilation)\n")

best_accuracy = 0.0
training_history = []
global_step = 0

# Prepare RNG keys
num_batches = len(train_loader)
all_rng_keys = jax.random.split(train_key, num_epochs * num_batches)
all_rng_keys = all_rng_keys.reshape((num_epochs, num_batches, 2))

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_losses = []

    for batch_idx, (x, y) in enumerate(train_loader):
        batch = {"x": jnp.array(x), "y": y}

        # Training step with inference history
        params, opt_state, loss, final_state, inference_history = jit_train_step(
            params, opt_state, batch, all_rng_keys[epoch, batch_idx]
        )

        normalized_loss = float(loss) / batch_size
        epoch_losses.append(normalized_loss)

        # Track with Aim
        if tracker is not None:
            # Batch loss
            tracker.track_batch_loss(normalized_loss, epoch=epoch, batch=batch_idx)

            # Per-node energy
            tracker.track_batch_energy_per_node(
                final_state, structure, epoch=epoch, batch=batch_idx
            )

            # Latent distributions (at configured frequency)
            tracker.track_latent_distributions(
                final_state, epoch=epoch, batch=batch_idx
            )

            # Inference dynamics (track convergence every 100 batches)
            if batch_idx % 100 == 0:
                # Convert lightweight history to tracking format
                for step_idx, step_metrics in enumerate(inference_history):
                    for node_name, metrics in step_metrics.items():
                        if node_name in tracking_config.inference_nodes_to_track:
                            tracker._run.track(
                                metrics["energy"],
                                name="inference_energy",
                                step=step_idx * 5,  # multiply by collect_every
                                context={
                                    "node": node_name,
                                    "epoch": epoch,
                                    "batch": batch_idx,
                                },
                            )
                            tracker._run.track(
                                metrics["latent_grad_norm"],
                                name="inference_grad_norm",
                                step=step_idx * 5,
                                context={
                                    "node": node_name,
                                    "epoch": epoch,
                                    "batch": batch_idx,
                                },
                            )

        global_step += 1

        # Progress update
        n_batch_update = 100
        if (batch_idx + 1) % n_batch_update == 0:
            avg_loss = sum(epoch_losses[-n_batch_update:]) / len(
                epoch_losses[-n_batch_update:]
            )
            # Summarize inference convergence
            convergence = summarize_inference_convergence(inference_history)
            h1_final = convergence.get("h1", {}).get("final_energy", 0)
            print(
                f"  Epoch {epoch+1}/{num_epochs}, "
                f"Batch {batch_idx+1}/{len(train_loader)}, "
                f"Loss: {avg_loss:.4f}, "
                f"h1 Energy: {h1_final:.4f}"
            )

    epoch_time = time.time() - epoch_start
    avg_loss = sum(epoch_losses) / len(epoch_losses)

    # Track weight distributions at end of epoch
    if tracker is not None:
        tracker.track_weight_distributions(params, structure, epoch=epoch)

    # Evaluate on test set
    epoch_eval_key, eval_key = jax.random.split(eval_key)
    metrics = evaluate_pcn(params, structure, test_loader, train_config, epoch_eval_key)
    accuracy = metrics["accuracy"] * 100

    # Track epoch metrics
    if tracker is not None:
        tracker.track_epoch_metrics(
            {"loss": avg_loss, "accuracy": accuracy / 100},
            epoch=epoch,
            subset="val",
        )

    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        print(f"  * New best accuracy: {accuracy:.2f}%")

    print(
        f"  Epoch {epoch+1}/{num_epochs} - "
        f"Loss: {avg_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%, "
        f"Time: {epoch_time:.1f}s"
    )

    training_history.append(
        {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": accuracy,
            "time": epoch_time,
        }
    )

# ==============================================================================
# FINAL RESULTS
# ==============================================================================

print(f"\n[Final Results]")
print(f"  Best accuracy: {best_accuracy:.2f}%")
print(f"  Final accuracy: {training_history[-1]['accuracy']:.2f}%")
print(f"  Total training time: {sum(h['time'] for h in training_history):.1f}s")

# Track final metrics and close
if tracker is not None:
    tracker.track_epoch_metrics(
        {"final_best_accuracy": best_accuracy / 100},
        epoch=num_epochs - 1,
        subset="final",
    )
    tracker.close()
    print("\n[Experiment Tracking Complete]")
    print("  Run 'aim up' to view the dashboard")
    print("  Tracked metrics:")
    print("    - Batch-level: loss, per-node energy")
    print("    - Epoch-level: loss, accuracy, weight distributions")
    print("    - Distributions: z_latent, z_mu, pre_activation per node")
    print("    - Inference dynamics: energy convergence per step")

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
