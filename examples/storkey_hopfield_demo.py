"""
Storkey Hopfield vs MLP — MNIST A/B Experiment
===============================================

Compares two predictive coding architectures:
- Hopfield: 4-node graph with StorkeyHopfield associative memory layer
- MLP: 4-node standard feedforward network (baseline from mnist_demo.py)

Both are trained with identical PC hyperparameters to isolate the effect
of the Hopfield attractor energy on classification accuracy.

Architecture:
    Hopfield: input(784) -> Linear(128, sigmoid) -> StorkeyHopfield(128, tanh) -> Linear(10, softmax, CE)
    MLP:      input(784) -> Linear(128, sigmoid) -> Linear(128, sigmoid) -> Linear(10, softmax, CE)

Usage:
    python examples/storkey_hopfield_demo.py
    python examples/storkey_hopfield_demo.py --n_trials 10
    python examples/storkey_hopfield_demo.py --verbose
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")

import jax
import argparse

from fabricpc.nodes import Linear, IdentityNode, StorkeyHopfield
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer, NormalInitializer
import optax
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.utils.data.dataloader import MnistLoader

jax.config.update("jax_default_prng_impl", "threefry2x32")

# Shared hyperparameters
optimizer = optax.adamw(0.001, weight_decay=0.1)
train_config = {"num_epochs": 1}
batch_size = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="A/B comparison: Storkey Hopfield vs MLP on MNIST"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of independent paired trials (default: 5)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Training epochs per trial (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output",
    )
    return parser.parse_args()


def create_hopfield_model(rng_key):
    """Create PC model with StorkeyHopfield hidden layer.

    Architecture: input(784) -> Linear(128) -> StorkeyHopfield(128) -> Linear(10)
    """
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden = Linear(
        shape=(128,),
        activation=SigmoidActivation(),
        name="hidden",
        weight_init=XavierInitializer(),
    )
    hopfield = StorkeyHopfield(
        shape=(128,),
        name="hopfield",
        hopfield_strength=1.0,
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )

    structure = graph(
        nodes=[pixels, hidden, hopfield, output],
        edges=[
            Edge(source=pixels, target=hidden.slot("in")),
            Edge(source=hidden, target=hopfield.slot("in")),
            Edge(source=hopfield, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def create_mlp_model(rng_key):
    """Create standard MLP baseline.

    Architecture: input(784) -> Linear(128) -> Linear(128) -> Linear(10)
    """
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden1 = Linear(
        shape=(128,),
        activation=SigmoidActivation(),
        name="hidden1",
        weight_init=XavierInitializer(),
    )
    hidden2 = Linear(
        shape=(128,),
        activation=SigmoidActivation(),
        name="hidden2",
        weight_init=XavierInitializer(),
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )

    structure = graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def main():
    args = parse_args()
    train_config["num_epochs"] = args.num_epochs

    print("=" * 70)
    print("A/B Experiment: Storkey Hopfield vs Standard MLP")
    print("=" * 70)
    print("Dataset: MNIST (real-valued)")
    print(
        "Hopfield: 784 -> 128(sigmoid) -> 128(StorkeyHopfield, tanh) -> 10(softmax, CE)"
    )
    print("MLP:      784 -> 128(sigmoid) -> 128(sigmoid) -> 10(softmax, CE)")
    print(f"Training: Predictive Coding (both arms)")
    print(f"Epochs per trial: {args.num_epochs}")
    print(f"Trials: {args.n_trials}")
    print()

    arm_hopfield = ExperimentArm(
        name="Hopfield",
        model_factory=create_hopfield_model,
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
        arm_a=arm_hopfield,
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
