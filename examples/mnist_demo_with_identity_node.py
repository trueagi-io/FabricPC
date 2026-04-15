"""
Predictive Coding Network — MNIST with IdentityNode
====================================================

Demonstrates the IdentityNode, which passes input through unchanged with
no learnable parameters. An IdentityNode is inserted between hidden layers
as a passthrough.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import jax
from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
import time

jax.config.update("jax_default_prng_impl", "threefry2x32")

# --- Network ---

pixels = Linear(shape=(784,), name="pixels")
hidden1 = Linear(
    shape=(256,),
    activation=SigmoidActivation(),
    name="hidden1",
)
passthrough = IdentityNode(shape=(256,), name="passthrough")
hidden2 = Linear(
    shape=(64,),
    activation=SigmoidActivation(),
    name="hidden2",
)
output = Linear(
    shape=(10,),
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),
    name="class",
)

structure = graph(
    nodes=[pixels, hidden1, passthrough, hidden2, output],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=passthrough.slot("in")),
        Edge(source=passthrough, target=hidden2.slot("in")),
        Edge(source=hidden2, target=output.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=output),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# --- Hyperparameters ---

optimizer = optax.adamw(0.001, weight_decay=0.001)
train_config = {"num_epochs": 27}
batch_size = 200

# --- Train & Evaluate ---

if __name__ == "__main__":
    master_rng_key = jax.random.PRNGKey(0)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)

    train_loader = MnistLoader(
        "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, tensor_format="flat", shuffle=False
    )

    print("\nTraining (JIT compilation on first batch)...")
    start_time = time.time()
    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
    )
    elapsed = time.time() - start_time
    print(f"Avg training time: {elapsed / train_config['num_epochs']:.2f}s per epoch")

    print("\nEvaluating...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Energy:   {metrics['energy']:.4f}")

    print(
        f"\n{len(structure.nodes)} nodes, {len(structure.edges)} edges, "
        f"{sum(p.size for p in jax.tree_util.tree_leaves(params)):,} parameters"
    )
