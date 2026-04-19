"""
Split-MNIST Quick Smoke Test

Minimal example for testing the continual learning pipeline.
Runs very quickly with reduced epochs and batch sizes.

Usage:
    python examples/split_mnist_quick.py
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import jax
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD

from fabricpc.continual.config import make_config
from fabricpc.continual.data import build_split_mnist_loaders
from fabricpc.continual.trainer import SequentialTrainer
from fabricpc.continual.utils import print_summary_table


def main():
    print("Split-MNIST Quick Smoke Test")
    print("=" * 40)

    # Smoke test configuration
    config = make_config(quick_smoke=True)
    config.training.training_mode = "backprop"  # Faster for smoke test

    print(f"Epochs per task: {config.training.epochs_per_task}")
    print(f"Batch size: {config.training.batch_size}")

    # Initialize
    jax.config.update("jax_default_prng_impl", "threefry2x32")
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    # Simple network
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden = Linear(shape=(64,), activation=SigmoidActivation(), name="hidden")
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="output",
    )

    structure = graph(
        nodes=[pixels, hidden, output],
        edges=[
            Edge(source=pixels, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )

    params = initialize_params(structure, init_key)
    optimizer = optax.adam(0.01)

    # Create trainer
    trainer = SequentialTrainer(
        structure=structure,
        config=config,
        params=params,
        optimizer=optimizer,
        rng_key=train_key,
    )

    # Load data
    print("\nLoading data...")
    tasks = build_split_mnist_loaders(config, data_root="./data")

    # Train on first 2 tasks only for smoke test
    print("\nTraining...")
    for task_data in tasks[:2]:
        summary = trainer.train_task(task_data, verbose=True)

    # Results
    print_summary_table(trainer.summaries)

    # Accuracy matrix
    acc_matrix = trainer.accuracy_matrix()
    print("\nAccuracy Matrix:")
    print(acc_matrix)

    print("\nSmoke test passed!")


if __name__ == "__main__":
    main()
