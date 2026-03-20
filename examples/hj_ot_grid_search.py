"""
HJ-OT Grid Search Tuning Script
===============================

Performs a grid search over viscosity and transport_cost for the 
HJ-OT optimizer in a Navier-Stokes constrained PCN.
"""

import os
import tempfile
import time
from typing import Iterator
import itertools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
from fabricpc.training.hj_ot import hj_ot_optimizer
from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")
os.environ.setdefault("TFDS_DATA_DIR", os.path.join(tempfile.gettempdir(), "fabricpc_tfds"))

jax.config.update("jax_default_prng_impl", "threefry2x32")

def mnist_to_uvp(images: np.ndarray) -> np.ndarray:
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
    
    # Grid search parameters
    viscosities = [0.9, 1.0]
    transport_costs = [1e-5]
    learning_rates = [3e-4, 1e-3, 3e-5]
    
    results = []
    
    # Pre-load data
    train_loader = MnistNavierStokesLoader("train", batch_size=batch_size, shuffle=True, seed=42)
    test_loader = MnistNavierStokesLoader("test", batch_size=batch_size, shuffle=False)
    
    master_rng_key = jax.random.PRNGKey(42)
    
    print(f"Starting Grid Search: {len(viscosities) * len(transport_costs) * len(learning_rates)} combinations")
    
    for lr, visc, cost in itertools.product(learning_rates, viscosities, transport_costs):
        print(f"\n--- Testing LR: {lr}, Viscosity: {visc}, Transport Cost: {cost} ---")
        
        optimizer = hj_ot_optimizer(
            learning_rate=lr,
            viscosity=visc,
            transport_cost=cost,
            dt=1.0
        )
        
        graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)
        structure = create_structure()
        params = initialize_params(structure, graph_key)
        
        start_time = time.time()
        try:
            trained_params, _, _ = train_pcn(
                params=params,
                structure=structure,
                train_loader=train_loader,
                optimizer=optimizer,
                config={"num_epochs": max_epochs},
                rng_key=train_key,
                verbose=False
            )
            
            metrics = evaluate_pcn(trained_params, structure, test_loader, {"num_epochs": max_epochs}, eval_key)
            accuracy = metrics['accuracy'] * 100
            elapsed = time.time() - start_time
            
            print(f"Result -> Accuracy: {accuracy:.2f}%, Time: {elapsed:.2f}s")
            results.append({
                "learning_rate": lr,
                "viscosity": visc,
                "transport_cost": cost,
                "accuracy": accuracy,
                "time": elapsed,
                "status": "success"
            })
        except Exception as e:
            print(f"Failed -> Error: {str(e)}")
            results.append({
                "learning_rate": lr,
                "viscosity": visc,
                "transport_cost": cost,
                "accuracy": 0.0,
                "time": 0.0,
                "status": f"failed: {str(e)}"
            })
            
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("hj_ot_grid_search_results.csv", index=False)
    print("\nResults saved to hj_ot_grid_search_results.csv")
    
    # Visualization: Heatmaps per Learning Rate
    for lr in learning_rates:
        lr_df = df[df["learning_rate"] == lr]
        if lr_df.empty:
            continue
            
        pivot_df = lr_df.pivot(index="viscosity", columns="transport_cost", values="accuracy")
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(pivot_df.values, cmap="viridis")
        plt.colorbar(im, label="Accuracy (%)")
        
        plt.xticks(range(len(transport_costs)), transport_costs)
        plt.yticks(range(len(viscosities)), viscosities)
        
        plt.xlabel("Transport Cost")
        plt.ylabel("Viscosity")
        plt.title(f"HJ-OT Heatmap (LR={lr}, {max_epochs} Epochs)")
        
        for i in range(len(viscosities)):
            for j in range(len(transport_costs)):
                val = pivot_df.values[i, j]
                plt.text(j, i, f"{val:.1f}", ha="center", va="center", color="white" if val < 50 else "black")
                
        plot_path = f"hj_ot_grid_search_heatmap_lr_{lr}.png"
        plt.savefig(plot_path)
        print(f"Heatmap saved to {plot_path}")

if __name__ == "__main__":
    main()
