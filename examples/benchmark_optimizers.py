"""
Benchmark script comparing HJ-OT optimizer against Adam and SGD.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Iterator

import numpy as np

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax
set_jax_flags_before_importing_jax(jax_platforms="cpu")
os.environ.setdefault("TFDS_DATA_DIR", os.path.join(tempfile.gettempdir(), "fabricpc_tfds"))

import jax
import jax.numpy as jnp
import optax
import time
import matplotlib.pyplot as plt

from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import NavierStokesEnergy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.graph import initialize_graph_state, initialize_params
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.training import train_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc.training.hj_ot import hj_ot_optimizer

def mnist_to_uvp(images: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(images)
    return np.concatenate([images, images, zeros], axis=-1).astype(np.float32)

class MnistFieldLoader:
    def __init__(self, split: str, batch_size: int, max_batches: int, **loader_kwargs):
        self.loader = MnistLoader(
            split=split, batch_size=batch_size, tensor_format="NHWC", **loader_kwargs
        )
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        for batch_idx, (images, _) in enumerate(self.loader):
            if batch_idx >= self.max_batches:
                break
            field = mnist_to_uvp(np.asarray(images))
            yield {"x": field, "y": field}

    def __len__(self) -> int:
        return self.max_batches

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

def held_out_batch_energy(params, structure, batch, rng_key) -> float:
    batch_size = batch["x"].shape[0]
    clamps = {
        structure.task_map["x"]: jnp.array(batch["x"]),
        structure.task_map["y"]: jnp.array(batch["y"]),
    }
    state = initialize_graph_state(
        structure, batch_size=batch_size, rng_key=rng_key, clamps=clamps, params=params
    )
    final_state = run_inference(params, state, clamps, structure)
    return float(jnp.mean(final_state.nodes["field"].energy))

def mean_energy(history) -> float:
    values = [float(value) for epoch in history for value in epoch]
    return sum(values) / len(values) if values else math.nan

def benchmark_optimizer(name, optimizer, train_loader, test_loader, master_key, config):
    print(f"\\n--- Running benchmark for: {name} ---")
    structure = create_structure()
    graph_key, train_key, eval_key = jax.random.split(master_key, 3)
    params = initialize_params(structure, graph_key)
    
    start_time = time.time()
    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=config,
        rng_key=train_key,
        verbose=False,
    )
    elapsed = time.time() - start_time
    
    avg_training_energy = mean_energy(energy_history)
    
    held_out_batch = next(iter(test_loader))
    held_out_energy = held_out_batch_energy(
        trained_params, structure, held_out_batch, eval_key
    )
    
    # Calculate stability (variance of energy in the final epoch)
    final_epoch_energies = [float(v) for v in energy_history[-1]]
    energy_variance = np.var(final_epoch_energies) if len(final_epoch_energies) > 1 else 0.0
    
    print(f"[{name}] Completed in {elapsed:.2f} seconds")
    print(f"[{name}] Final Avg Training Energy: {avg_training_energy:.4f}")
    print(f"[{name}] Held-out Validation Energy: {held_out_energy:.4f}")
    print(f"[{name}] Stability (Final Epoch Energy Variance): {energy_variance:.4f}")
    return {
        "time": elapsed,
        "train_energy": avg_training_energy,
        "test_energy": held_out_energy,
        "variance": energy_variance
    }, energy_history

def main():
    batch_size = 32
    train_batches = 10
    test_batches = 1

    train_loader = MnistFieldLoader("train", batch_size, train_batches, shuffle=True, seed=42)
    test_loader = MnistFieldLoader("test", batch_size, test_batches, shuffle=False)

    config = {"num_epochs": 3}
    master_key = jax.random.PRNGKey(42)

    optimizers = {
        "Optax Adam": optax.adam(1e-3),
        "Optax SGD+Momentum": optax.sgd(1e-3, momentum=0.9),
        "Dual-Fluid HJ-OT": hj_ot_optimizer(learning_rate=1e-3, viscosity=0.9, transport_cost=1e-4, dt=1.0)
    }

    results = {}
    history_data = {}
    
    for name, opt in optimizers.items():
        results[name], history_data[name] = benchmark_optimizer(name, opt, train_loader, test_loader, master_key, config)

    # Filter out NaNs for the plot
    plt.figure(figsize=(10, 6))
    
    for name, history in history_data.items():
        # history is a list of epochs, each containing a list of batch energies
        flat_history = [float(val) for epoch in history for val in epoch]
        if np.any(np.isnan(flat_history)) or np.any(np.isinf(flat_history)):
            print(f"Skipping {name} from plot due to NaNs/Infs.")
            continue
            
        plt.plot(flat_history, label=name, marker='o', markersize=4, linestyle='-', linewidth=2)
        
    plt.title("Optimization Energy Landscape (Navier-Stokes MNIST)")
    plt.xlabel("Training Step (Batch)")
    plt.ylabel("Predictive Coding Free Energy")
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    
    save_path = '/Users/bitseat/.gemini/antigravity/brain/7ede49b9-a69b-48a6-871c-ed6a9149d7c5/optimization_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\\nSaved optimization plot to {save_path}")

    print("\\n\\n================= SUMMARY =================")
    for name, res in results.items():
        print(f"{name:25s} | Train Energy: {res['train_energy']:8.4f} | Test Energy: {res['test_energy']:8.4f} | Variance: {res['variance']:10.2f}")

if __name__ == "__main__":
    main()
