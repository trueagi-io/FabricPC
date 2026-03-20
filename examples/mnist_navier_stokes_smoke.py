"""
MNIST Navier-Stokes Energy Smoke Test
=====================================

Load real MNIST images, adapt them into pseudo-fluid fields with channels
`(u, v, p)`, and run a short predictive-coding train/inference pass using
`NavierStokesEnergy`.

This is a runtime validation example, not a classification benchmark.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Iterator

import numpy as np

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")
os.environ.setdefault(
    "TFDS_DATA_DIR", os.path.join(tempfile.gettempdir(), "fabricpc_tfds")
)

import jax
import jax.numpy as jnp
import optax

from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import NavierStokesEnergy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.graph import initialize_graph_state, initialize_params
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.training import train_pcn
from fabricpc.training.hj_ot import hj_ot_optimizer  # Import the new Dual-Fluid optimizer
from fabricpc.utils.data.dataloader import MnistLoader

jax.config.update("jax_default_prng_impl", "threefry2x32")


def mnist_to_uvp(images: np.ndarray) -> np.ndarray:
    """Map MNIST grayscale images to a simple `(u, v, p)` field."""
    zeros = np.zeros_like(images)
    return np.concatenate([images, images, zeros], axis=-1).astype(np.float32)


class MnistFieldLoader:
    """Adapt MNIST batches to dict batches compatible with the training loop."""

    def __init__(self, split: str, batch_size: int, max_batches: int, **loader_kwargs):
        self.loader = MnistLoader(
            split=split,
            batch_size=batch_size,
            tensor_format="NHWC",
            **loader_kwargs,
        )
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        for batch_idx, (images, _) in enumerate(self.loader):
            if batch_idx >= self.max_batches:
                break
            field = mnist_to_uvp(np.asarray(images))
            yield {"x": field, "y": field}

    def __len__(self) -> int:
        return min(len(self.loader), self.max_batches)


def create_structure():
    input_node = IdentityNode(shape=(28, 28, 3), name="input")
    field_node = Linear(
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
        name="field",
    )

    return graph(
        nodes=[input_node, field_node],
        edges=[Edge(source=input_node, target=field_node.slot("in"))],
        task_map=TaskMap(x=input_node, y=field_node),
        inference=InferenceSGD(eta_infer=0.01, infer_steps=3),
    )


def mean_energy(history) -> float:
    values = [float(value) for epoch in history for value in epoch]
    return sum(values) / len(values) if values else math.nan


def held_out_batch_energy(params, structure, batch, rng_key) -> float:
    batch_size = batch["x"].shape[0]
    clamps = {
        structure.task_map["x"]: jnp.array(batch["x"]),
        structure.task_map["y"]: jnp.array(batch["y"]),
    }
    state = initialize_graph_state(
        structure,
        batch_size=batch_size,
        rng_key=rng_key,
        clamps=clamps,
        params=params,
    )
    final_state = run_inference(params, state, clamps, structure)
    return float(jnp.mean(final_state.nodes["field"].energy))


def main():
    batch_size = 16
    train_batches = 4
    test_batches = 1

    train_loader = MnistFieldLoader(
        "train",
        batch_size=batch_size,
        max_batches=train_batches,
        shuffle=True,
        seed=42,
    )
    test_loader = MnistFieldLoader(
        "test",
        batch_size=batch_size,
        max_batches=test_batches,
        shuffle=False,
    )

    raw_preview_loader = MnistLoader(
        "train",
        batch_size=batch_size,
        tensor_format="NHWC",
        shuffle=False,
    )
    preview_images, _ = next(iter(raw_preview_loader))
    preview_field = mnist_to_uvp(np.asarray(preview_images))

    print("MNIST Navier-Stokes smoke test")
    print(f"TFDS data dir: {os.environ['TFDS_DATA_DIR']}")
    print(f"Loaded MNIST batch shape: {tuple(preview_images.shape)}")
    print(f"Adapted field batch shape: {tuple(preview_field.shape)}")

    structure = create_structure()
    
    # Dual-Fluid: Use Hamilton-Jacobi Optimal Transport for weight propagation.
    # The network predicts Navier-Stokes fluids, while weights travel through
    # parameter space via HJ-OT transport viscous dynamics!
    optimizer = hj_ot_optimizer(
        learning_rate=1e-3,
        viscosity=0.9,
        transport_cost=1e-4,
        dt=1.0
    )
    
    train_config = {"num_epochs": 1}

    master_key = jax.random.PRNGKey(0)
    graph_key, train_key, eval_key = jax.random.split(master_key, 3)
    params = initialize_params(structure, graph_key)

    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
    )
    avg_training_energy = mean_energy(energy_history)

    held_out_batch = next(iter(test_loader))
    held_out_energy = held_out_batch_energy(
        trained_params, structure, held_out_batch, eval_key
    )

    finite_values = jnp.array([avg_training_energy, held_out_energy])
    print(f"Average training energy: {avg_training_energy:.6f}")
    print(f"Held-out batch energy: {held_out_energy:.6f}")
    print(f"All reported energies finite: {bool(jnp.all(jnp.isfinite(finite_values)))}")


if __name__ == "__main__":
    main()
