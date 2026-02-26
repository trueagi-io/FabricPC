"""
MNIST with Aim Experiment Tracking
==================================

This example demonstrates comprehensive experiment tracking for predictive
coding networks using Aim. It tracks:

1. Batch and epoch-level energy/accuracy metrics
2. Weight distributions per layer per epoch
3. Latent state distributions (z_latent, z_mu, pre_activation)
4. Per-node energy values
5. Inference dynamics: how energy and gradients evolve over inference steps

This provides deep insights for debugging and tuning predictive coding models.

After running, launch the Aim UI with:
    aim up

Click on the link returned in the console to explore the dashboard.
Be sure to run the python script and start aim from the same working directory to ensure the tracking data is correctly linked.

Requirements:
    pip install fabricpc[viz]  # Includes aim
"""

import os  # set environment variables before importing JAX

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_PLATFORMS", "cuda"
)  # options: "cpu", "cuda" or "tpu" if available
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress XLA warnings

import time

import jax
import jax.numpy as jnp
from fabricpc.utils.data.dataloader import MnistLoader

from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params, FeedforwardStateInit
from fabricpc.core.activations import (
    IdentityActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import GaussianEnergy, CrossEntropyEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.training import create_optimizer, evaluate_pcn

# Import dashboarding utilities
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    is_aim_available,
    train_step_with_history,
    unstack_inference_history,
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

# ==============================================================================
# NETWORK CONFIGURATION
# ==============================================================================

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
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),
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
)

train_config = {
    "infer_steps": 20,
    "eta_infer": 0.05,
    "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.001},
}
batch_size = 200
num_epochs = 1

# ==============================================================================
# CREATE MODEL
# ==============================================================================

print("=" * 70)
print("MNIST with Aim Experiment Tracking")
print("=" * 70)

params = initialize_params(structure, graph_key)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

print(f"\n[Model Architecture]")
print(f"  Nodes: {len(structure.nodes)}")
print(f"  Total parameters: {num_params:,}")

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

# ==============================================================================
# SETUP AIM TRACKING
# ==============================================================================

if TRACKING_ENABLED:
    print(f"\n[Setting up Aim Tracking]")

    tracking_config = TrackingConfig(
        experiment_name="mnist_pcn_tracking",
        run_name=f"5layer_lr{train_config['optimizer']['lr']}_infer{train_config['infer_steps']}",
        # Batch-level tracking
        track_batch_energy=True,
        track_batch_energy_per_node=True,
        # Epoch-level tracking
        track_epoch_energy=True,
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
                "num_layers": len(structure.nodes),
                "layer_sizes": [
                    structure.nodes[n].node_info.shape for n in structure.node_order
                ],
                "activation": "sigmoid/softmax",
                "energy_type": "gaussian/cross_entropy",
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
    epoch_energies = []

    for batch_idx, (x, y) in enumerate(train_loader):
        batch = {"x": jnp.array(x), "y": y}

        energy = 0
        # Training step with inference history (returns stacked metrics)
        params, opt_state, energy, final_state, stacked_history = jit_train_step(
            params, opt_state, batch, all_rng_keys[epoch, batch_idx]
        )

        # Unstack inference history outside of JIT (converts JAX arrays to Python floats)
        inference_history = unstack_inference_history(stacked_history, collect_every=5)

        normalized_energy = float(energy) / batch_size
        epoch_energies.append(normalized_energy)

        # Track with Aim
        if tracker is not None:
            # Batch energy
            tracker.track_batch_energy(normalized_energy, epoch=epoch, batch=batch_idx)

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
            avg_energy = sum(epoch_energies[-n_batch_update:]) / len(
                epoch_energies[-n_batch_update:]
            )
            # Summarize inference convergence
            convergence = summarize_inference_convergence(inference_history)
            h1_final = convergence.get("h1", {}).get("final_energy", 0)
            print(
                f"  Epoch {epoch+1}/{num_epochs}, "
                f"Batch {batch_idx+1}/{len(train_loader)}, "
                f"energy: {avg_energy:.4f}, "
                f"h1 Energy: {h1_final:.4f}"
            )

    epoch_time = time.time() - epoch_start
    avg_energy = sum(epoch_energies) / len(epoch_energies)

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
            {"energy": avg_energy, "accuracy": accuracy / 100},
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
        f"energy: {avg_energy:.4f}, "
        f"Accuracy: {accuracy:.2f}%, "
        f"Time: {epoch_time:.1f}s"
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
    print("    - Batch-level: energy, per-node energy")
    print("    - Epoch-level: energy, accuracy, weight distributions")
    print("    - Distributions: z_latent, z_mu, pre_activation per node")
    print("    - Inference dynamics: energy convergence per step")

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
