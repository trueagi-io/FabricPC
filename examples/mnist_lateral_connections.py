"""
Statistical Comparison: Lateral Connections vs Standard MLP on MNIST
=====================================================================

Compares two predictive coding architectures:
- Lateral: 6-node graph with lateral connections between hidden layers
- MLP: 4-node standard feedforward network (baseline)

Both are trained with identical PC hyperparameters to isolate the effect
of lateral connectivity on classification accuracy.

Usage:
    python examples/mnist_lateral_connections.py                # 10 trials (default)
    python examples/mnist_lateral_connections.py --n_trials 20  # 20 trials
    python examples/mnist_lateral_connections.py --verbose       # show per-epoch output
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import argparse

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.utils.data.dataloader import MnistLoader

jax.config.update("jax_default_prng_impl", "threefry2x32")

# Training hyperparameters (shared by both arms)
optimizer = optax.adamw(0.001, weight_decay=0.001)
train_config = {
    "num_epochs": 1,
}
batch_size = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of Lateral vs MLP on MNIST"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=3,
        help="Number of independent training trials per architecture (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output for each trial",
    )
    return parser.parse_args()


# fmt: off
def create_lateral_model(rng_key):
    """Create PC model with lateral connections between hidden layers."""
    pixels           = IdentityNode(shape=(784,), name="pixels")
    hidden1          = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1")
    hidden1_lateral  = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1_lateral")
    hidden2_lateral  = Linear(shape=(64,),  activation=SigmoidActivation(), name="hidden2_lateral")
    hidden2          = Linear(shape=(64,),  activation=SigmoidActivation(), name="hidden2")
    output           = Linear(shape=(10,),  activation=SoftmaxActivation(), energy=CrossEntropyEnergy(), name="class")

    structure = graph(
        nodes=[pixels, hidden1, hidden1_lateral, hidden2_lateral, hidden2, output],
        edges=[
            Edge(source=pixels,          target=hidden1.slot("in")),
            Edge(source=hidden1,         target=hidden2.slot("in")),
            Edge(source=hidden2,         target=output.slot("in")),
            Edge(source=pixels,          target=hidden1_lateral.slot("in")),
            Edge(source=hidden1_lateral, target=hidden1.slot("in")),
            Edge(source=hidden1_lateral, target=hidden2_lateral.slot("in")),
            Edge(source=hidden2_lateral, target=hidden2.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def create_mlp_model(rng_key):
    """Create standard MLP (no lateral connections) as baseline."""
    pixels   = IdentityNode(shape=(784,), name="pixels")
    hidden1  = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1")
    hidden2  = Linear(shape=(64,),  activation=SigmoidActivation(), name="hidden2")
    output   = Linear(shape=(10,),  activation=SoftmaxActivation(), energy=CrossEntropyEnergy(), name="class")

    structure = graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels,  target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
    params = initialize_params(structure, rng_key)
    return params, structure
# fmt: on


def main():
    args = parse_args()

    print("=" * 70)
    print("Statistical Comparison: Lateral Connections vs Standard MLP")
    print("=" * 70)
    print(f"Dataset: MNIST")
    print(f"Lateral: 784 -> [256 + 256_lat] -> [64 + 64_lat] -> 10  (6 nodes, 7 edges)")
    print(f"MLP:     784 -> 256 -> 64 -> 10                         (4 nodes, 3 edges)")
    print(f"Training: Predictive Coding (both arms)")
    print(f"Epochs per trial: {train_config['num_epochs']}")
    print(f"Number of trials: {args.n_trials}")
    print()

    arm_lateral = ExperimentArm(
        name="Lateral",
        model_factory=create_lateral_model,
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=train_config,
    )

    arm_mlp = ExperimentArm(
        name="MLP",
        model_factory=create_mlp_model,
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=train_config,
    )

    experiment = ABExperiment(
        arm_a=arm_lateral,
        arm_b=arm_mlp,
        metric="accuracy",
        data_loader_factory=lambda seed: (
            MnistLoader(
                "train",
                batch_size=batch_size,
                tensor_format="flat",
                shuffle=True,
                seed=seed,
            ),
            MnistLoader(
                "test",
                batch_size=batch_size,
                tensor_format="flat",
                shuffle=False,
            ),
        ),
        n_trials=args.n_trials,
        verbose=args.verbose,
    )

    results = experiment.run()
    results.print_summary()


if __name__ == "__main__":
    main()
