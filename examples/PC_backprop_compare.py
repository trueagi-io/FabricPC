"""
Statistical Comparison: Predictive Coding vs Backpropagation on MNIST
=====================================================================

Runs multiple independent training trials for both PC and backprop,
then performs statistical analysis to compare test accuracies.

Architecture (identical topology, different activations)::

    PC:      pixels(784) ──→ hidden1(256,sigmoid) ──→ hidden2(64,sigmoid) ──→ class(10,softmax)
    Backprop: pixels(784) ──→ hidden1(256,relu) ──→ hidden2(64,relu) ──→ class(10,softmax)

Reports:
- Per-trial accuracy results table
- Mean +/- standard error for each method
- Paired t-test with p-value
- Cohen's d effect size
- Power analysis: estimated n_trials needed for significance

Usage:
    python examples/PC_backprop_compare.py                # 10 trials (default)
    python examples/PC_backprop_compare.py --n_trials 20  # 20 trials
    python examples/PC_backprop_compare.py --verbose       # show per-epoch output
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()  # "cpu", "cuda" or "tpu"

import jax
import argparse
import importlib.util

from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    SigmoidActivation,
    SoftmaxActivation,
    ReLUActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.training.train_backprop import train_backprop, evaluate_backprop
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.utils.data.dataloader import MnistLoader

# Import train_config and batch_size from mnist_demo without triggering examples/__init__.py
_demo_path = os.path.join(os.path.dirname(__file__), "mnist_demo.py")
_spec = importlib.util.spec_from_file_location("mnist_demo", _demo_path)
_mnist_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mnist_demo)
train_config = _mnist_demo.train_config
batch_size = _mnist_demo.batch_size
optimizer = optax.adamw(0.001, weight_decay=0.001)

jax.config.update("jax_default_prng_impl", "threefry2x32")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of PC vs Backprop on MNIST"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of independent training trials per method (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output for each trial",
    )
    return parser.parse_args()


def create_pc_model(rng_key):
    """Create PC model with sigmoid activations."""
    pixels = Linear(shape=(784,), name="pixels")
    hidden1 = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1")
    hidden2 = Linear(shape=(64,), activation=SigmoidActivation(), name="hidden2")
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
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


def create_backprop_model(rng_key):
    """Create backprop model with ReLU activations (to avoid vanishing gradients)."""
    pixels = Linear(shape=(784,), name="pixels")
    hidden1 = Linear(shape=(256,), activation=ReLUActivation(), name="hidden1")
    hidden2 = Linear(shape=(64,), activation=ReLUActivation(), name="hidden2")
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
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

    print("=" * 70)
    print("Statistical Comparison: Predictive Coding vs Backpropagation")
    print("=" * 70)
    print(f"Dataset: MNIST")
    print(f"Architecture: 784 -> 256 -> 64 -> 10")
    print(f"PC activations: sigmoid | Backprop activations: relu")
    print(f"Epochs per trial: {train_config['num_epochs']}")
    print(f"Number of trials: {args.n_trials}")
    print()

    arm_pc = ExperimentArm(
        name="PC",
        model_factory=create_pc_model,
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=train_config,
    )

    arm_bp = ExperimentArm(
        name="Backprop",
        model_factory=create_backprop_model,
        train_fn=train_backprop,
        eval_fn=evaluate_backprop,
        optimizer=optimizer,
        train_config=train_config,
    )

    experiment = ABExperiment(
        arm_a=arm_pc,
        arm_b=arm_bp,
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
