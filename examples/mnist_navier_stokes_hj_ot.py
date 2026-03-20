"""
Predictive Coding Network — MNIST Navier-Stokes HW-OT Classification
=======================================

Train a predictive coding network on MNIST using the Navier-Stokes energy 
on an intermediate latent field and compare HJ-OT optimizer versus Adam.
"""

import os
import tempfile
import time
from typing import Iterator

import numpy as np
import matplotlib.pyplot as plt

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

# We use CPU to align with the smoke test unless GPU is readily available without OOM.
set_jax_flags_before_importing_jax(jax_platforms="cpu")
os.environ.setdefault("TFDS_DATA_DIR", os.path.join(tempfile.gettempdir(), "fabricpc_tfds"))

import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SoftmaxActivation, IdentityActivation
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.training.hj_ot import hj_ot_optimizer

jax.config.update("jax_default_prng_impl", "threefry2x32")


def mnist_to_uvp(images: np.ndarray) -> np.ndarray:
    """Map MNIST grayscale images to a simple `(u, v, p)` field."""
    zeros = np.zeros_like(images)
    return np.concatenate([images, images, zeros], axis=-1).astype(np.float32)


class MnistNavierStokesLoader:
    def __init__(self, split: str, batch_size: int, **loader_kwargs):
        self.loader = MnistLoader(
            split=split,
            batch_size=batch_size,
            tensor_format="NHWC",
            **loader_kwargs,
        )

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        for images, labels in self.loader:
            field = mnist_to_uvp(np.asarray(images))
            yield {"x": field, "y": np.asarray(labels)}

    def __len__(self) -> int:
        return len(self.loader)


def create_structure(use_navier_stokes: bool = True):
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")
    
    fluid_energy = NavierStokesEnergy(
        viscosity=0.1,
        data_weight=1.0,
        latent_ns_weight=0.1,
        prediction_ns_weight=0.1,
        momentum_weight=1.0,
        divergence_weight=1.0,
    ) if use_navier_stokes else GaussianEnergy()
    
    # Intermediate fluid representation layer
    fluid_layer = Linear(
        shape=(28, 28, 3),
        activation=IdentityActivation(),
        energy=fluid_energy,
        name="fluid",
    )
    # Output class probabilities
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        flatten_input=True,
    )

    return graph(
        nodes=[pixels, fluid_layer, output],
        edges=[
            Edge(source=pixels, target=fluid_layer.slot("in")),
            Edge(source=fluid_layer, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )

def train_and_eval(optimizer, name, use_navier_stokes=True, max_epochs=2):
    train_config = {"num_epochs": max_epochs}
    batch_size = 200

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    structure = create_structure(use_navier_stokes)
    params = initialize_params(structure, graph_key)

    train_loader = MnistNavierStokesLoader("train", batch_size=batch_size, shuffle=True, seed=42)
    test_loader = MnistNavierStokesLoader("test", batch_size=batch_size, shuffle=False)

    print(f"\n--- Training with {name} ---")
    
    accuracies = []
    
    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        metrics = evaluate_pcn(params, structure, test_loader, config, rng_key)
        acc = metrics['accuracy'] * 100
        print(f"[{name}] Epoch {epoch_idx + 1} Test Accuracy: {acc:.2f}%")
        accuracies.append(acc)
        return acc

    start_time = time.time()
    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
        epoch_callback=epoch_callback
    )
    elapsed = time.time() - start_time
    print(f"[{name}] Total training time: {elapsed:.2f}s")
    
    return accuracies

def main():
    max_epochs = 20
    
    # 1. HJ-OT with Low Viscosity (0.2)
    hj_ot_low_visc = hj_ot_optimizer(
        learning_rate=1e-3,
        viscosity=0.2,
        transport_cost=1e-4,
        dt=1.0
    )
    low_visc_accs = train_and_eval(hj_ot_low_visc, "HJ-OT (Visc 0.2)", use_navier_stokes=True, max_epochs=max_epochs)
    
    # 2. HJ-OT with High Viscosity (0.5)
    hj_ot_high_visc = hj_ot_optimizer(
        learning_rate=1e-3,
        viscosity=0.5,
        transport_cost=1e-4,
        dt=1.0
    )
    high_visc_accs = train_and_eval(hj_ot_high_visc, "HJ-OT (Visc 2.0)", use_navier_stokes=True, max_epochs=max_epochs)
    
    # Plotting
    epochs = range(1, max_epochs + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, low_visc_accs, 'o-', label='HJ-OT (Visc 0.5)', color='teal')
    plt.plot(epochs, high_visc_accs, 's-', label='HJ-OT (Visc 2.0)', color='crimson')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.title('HJ-OT Parameter Comparison (Viscosity 0.5 vs 2.0)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(os.getcwd(), 'examples', 'navier_stokes_accuracy_comparison.png')
    plt.savefig(plot_path)
    print(f"\nSaved visualization to {plot_path}")

if __name__ == "__main__":
    main()
