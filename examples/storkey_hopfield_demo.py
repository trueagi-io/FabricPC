"""
Storkey Hopfield vs MLP — MNIST A/B Experiment
===============================================

Compares two predictive coding architectures:
- Hopfield: 4-node graph with StorkeyHopfield associative memory layer
- MLP: 4-node standard feedforward network (baseline from mnist_demo.py)

Both are trained with identical PC hyperparameters to isolate the effect
of the Hopfield attractor energy on classification accuracy.

Architecture:
    Hopfield: input(784) -> Linear(128, tanh) -> StorkeyHopfield(128, tanh) -> Linear(10, softmax, CE)
    MLP:      input(784) -> Linear(128, tanh) -> Linear(128, tanh) -> Linear(10, softmax, CE)

Usage:
    python examples/storkey_hopfield_demo.py
    python examples/storkey_hopfield_demo.py --n_trials 15
    python examples/storkey_hopfield_demo.py --verbose
    python examples/storkey_hopfield_demo.py --sweep   # strength sweep mode
"""

from fabricpc.core import TanhActivation
from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")

import jax
import jax.numpy as jnp
import argparse
import numpy as np

from fabricpc.nodes import Linear, IdentityNode, StorkeyHopfield
from fabricpc.nodes.storkey_hopfield import inverse_softplus
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import SoftmaxActivation
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
        default=15,
        help="Number of independent paired trials (default: 15)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Training epochs per trial (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run hopfield_strength sweep over [0, 0.1, 0.5, 1, 2, 4, 8, 32, None(learnable)]",
    )
    return parser.parse_args()


def make_hopfield_factory(hopfield_strength=1.0):
    """Return a model factory closure with the given hopfield_strength."""

    def create_hopfield_model(rng_key):
        """Create PC model with StorkeyHopfield hidden layer.

        Architecture: input(784) -> Linear(128) -> StorkeyHopfield(128) -> Linear(10)
        """
        pixels = IdentityNode(shape=(784,), name="pixels")
        hidden = Linear(
            shape=(128,),
            activation=TanhActivation(),
            name="hidden",
            weight_init=XavierInitializer(),
        )
        hopfield = StorkeyHopfield(
            shape=(128,),
            name="hopfield",
            hopfield_strength=hopfield_strength,
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

    return create_hopfield_model


# Default factory for backward compatibility
create_hopfield_model = make_hopfield_factory()


def create_mlp_model(rng_key):
    """Create standard MLP baseline.

    Architecture: input(784) -> Linear(128) -> Linear(128) -> Linear(10)
    """
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden1 = Linear(
        shape=(128,),
        activation=TanhActivation(),
        name="hidden1",
        weight_init=XavierInitializer(),
    )
    hidden2 = Linear(
        shape=(128,),
        activation=TanhActivation(),
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


def _make_data_loader_factory(batch_size):
    """Return a data_loader_factory callable for ABExperiment."""
    return lambda seed: (
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
    )


def _get_learned_strength(params, structure):
    """Extract the effective hopfield_strength from trained params.

    Returns the softplus-transformed value if learnable, or the fixed
    config value otherwise.
    """
    for node_name, node_params in params.nodes.items():
        if "hopfield_strength" in node_params.biases:
            raw = node_params.biases["hopfield_strength"]
            return float(jax.nn.softplus(raw))
    return None


def run_sweep(args):
    """Run hopfield_strength sweep: each strength vs MLP baseline."""
    strengths = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0, None]

    print("=" * 70)
    print("Hopfield Strength Sweep vs MLP Baseline")
    print("=" * 70)
    print(
        "Architecture: 784 -> 128(tanh) -> 128(StorkeyHopfield, tanh) -> 10(softmax, CE)"
    )
    print("Baseline:     784 -> 128(tanh) -> 128(tanh) -> 10(softmax, CE)")
    print(f"Epochs per trial: {args.num_epochs}")
    print(f"Trials per strength: {args.n_trials}")
    print(f"Strengths: {[s if s is not None else 'learnable' for s in strengths]}")
    print()

    arm_mlp = ExperimentArm(
        name="MLP",
        model_factory=create_mlp_model,
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=train_config,
    )

    data_loader_factory = _make_data_loader_factory(batch_size)
    sweep_results = []

    for s in strengths:
        label = "learnable" if s is None else f"{s}"
        print(f"\n{'─'*70}")
        print(f"  hopfield_strength = {label}")
        print(f"{'─'*70}")

        arm_hop = ExperimentArm(
            name=f"Hop(s={label})",
            model_factory=make_hopfield_factory(s),
            train_fn=train_pcn,
            eval_fn=evaluate_pcn,
            optimizer=optimizer,
            train_config=train_config,
        )

        experiment = ABExperiment(
            arm_a=arm_hop,
            arm_b=arm_mlp,
            metric="accuracy",
            data_loader_factory=data_loader_factory,
            n_trials=args.n_trials,
            verbose=args.verbose,
        )
        result = experiment.run()

        hop_acc = result.arm_a_metrics
        mlp_acc = result.arm_b_metrics

        # For learnable strength, train one more model to extract final value
        learned_str = None
        if s is None:
            key = jax.random.PRNGKey(42)
            params, structure = make_hopfield_factory(None)(key)
            train_loader = MnistLoader(
                "train",
                batch_size=batch_size,
                tensor_format="flat",
                shuffle=True,
                seed=42,
            )
            params, _, _ = train_pcn(
                params,
                structure,
                train_loader,
                optimizer,
                train_config,
                key,
            )
            learned_str = _get_learned_strength(params, structure)

        sweep_results.append(
            {
                "strength": s,
                "label": label,
                "hop_mean": float(np.mean(hop_acc)),
                "hop_se": float(np.std(hop_acc, ddof=1) / np.sqrt(len(hop_acc))),
                "mlp_mean": float(np.mean(mlp_acc)),
                "mlp_se": float(np.std(mlp_acc, ddof=1) / np.sqrt(len(mlp_acc))),
                "diff_mean": float(np.mean(hop_acc - mlp_acc)),
                "learned_str": learned_str,
            }
        )

    # Summary table
    print("\n")
    print("=" * 70)
    print("STRENGTH SWEEP SUMMARY")
    print("=" * 70)
    print(
        f"{'Strength':<12} {'Hopfield%':>12} {'MLP%':>12} {'Diff%':>10} {'Learned':>10}"
    )
    print("─" * 70)
    for r in sweep_results:
        hop_str = f"{r['hop_mean']*100:.2f}±{r['hop_se']*100:.2f}"
        mlp_str = f"{r['mlp_mean']*100:.2f}±{r['mlp_se']*100:.2f}"
        diff_str = f"{r['diff_mean']*100:+.2f}"
        learned = f"{r['learned_str']:.3f}" if r["learned_str"] is not None else ""
        print(
            f"{r['label']:<12} {hop_str:>12} {mlp_str:>12} {diff_str:>10} {learned:>10}"
        )
    print("─" * 70)
    print(f"  Trials: {args.n_trials}, Epochs: {args.num_epochs}")


def run_single(args):
    """Run a single A/B experiment (original behavior)."""
    print("=" * 70)
    print("A/B Experiment: Storkey Hopfield vs Standard MLP")
    print("=" * 70)
    print("Dataset: MNIST (real-valued)")
    print("Hopfield: 784 -> 128(tanh) -> 128(StorkeyHopfield, tanh) -> 10(softmax, CE)")
    print("MLP:      784 -> 128(tanh) -> 128(tanh) -> 10(softmax, CE)")
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
        data_loader_factory=_make_data_loader_factory(batch_size),
        n_trials=args.n_trials,
        verbose=args.verbose,
    )

    results = experiment.run()
    results.print_summary()


def main():
    args = parse_args()
    train_config["num_epochs"] = args.num_epochs

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
