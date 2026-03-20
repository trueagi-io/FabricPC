"""
MNIST with Navier-Stokes Energy using standard Adam Optimizer
=======================================

This script tests the behavior of a standard Adam optimizer when faced 
with the Navier-Stokes physics constraints in a Predictive Coding Network.
"""

import os
import tempfile
import time
from typing import Iterator

import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SoftmaxActivation, IdentityActivation
from fabricpc.core.energy import CrossEntropyEnergy, NavierStokesEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")
os.environ.setdefault("TFDS_DATA_DIR", os.path.join(tempfile.gettempdir(), "fabricpc_tfds"))

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

def create_structure():
    pixels = IdentityNode(shape=(28, 28, 3), name="pixels")
    
    # Layer 1: Fluid Physics representation (H, W, C)
    fluid_layer = Linear(
        shape=(28, 28, 3),
        activation=IdentityActivation(),
        energy=NavierStokesEnergy(
            viscosity=0.1,
            data_weight=1.0,
            latent_ns_weight=0.1,
            prediction_ns_weight=0.1,
            momentum_weight=1.0,
            divergence_weight=1.0,
        ),
        name="fluid",
    )
    
    # Output class probabilities (flatten_input=True to bridge from 2D grid to 1D class probabilities)
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

def main():
    max_epochs = 20
    batch_size = 200
    train_config = {"num_epochs": max_epochs}
    
    # Standard Adam Optimizer (No HJ-OT)
    optimizer = optax.adam(1e-3)
    
    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    structure = create_structure()
    params = initialize_params(structure, graph_key)

    train_loader = MnistNavierStokesLoader("train", batch_size=batch_size, shuffle=True, seed=42)
    test_loader = MnistNavierStokesLoader("test", batch_size=batch_size, shuffle=False)

    print("\n--- Training Adam with Navier-Stokes Energy ---")
    
    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
        epoch_callback=lambda epoch_idx, params, structure, config, rng_key: 
            print(f"Epoch {epoch_idx+1} complete. Final Test Accuracy: {(evaluate_pcn(params, structure, test_loader, config, rng_key)['accuracy']*100):.2f}%")
    )

if __name__ == "__main__":
    main()
