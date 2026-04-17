"""
StorkeyHopfield vs MLP — Few-Shot Noise Robustness Demo
========================================================

Demonstrates that the StorkeyHopfield attractor dynamics provide a
statistically significant advantage over a vanilla MLP under data scarcity
and input noise — conditions where memorized class prototypes and attractor
denoising should help.

Architecture (same 4-node graph for both arms)::

    Hopfield: pixels(784) ──→ hidden(128,tanh) ──→ hopfield(128,tanh) ──→ output(10,softmax+CE)
    MLP:      pixels(784) ──→ hidden(128,tanh) ──→ linear(128,tanh)   ──→ output(10,softmax+CE)

Experiment grid: K (shots per class) x noise_std
    - K controls data scarcity: fewer examples -> more reliance on attractor memory
    - noise_std controls input corruption: more noise -> more reliance on denoising

Hypothesis: Hopfield advantage (Delta accuracy) increases as K decreases
and noise_std increases.

Usage:
    python examples/storkey_hopfield_demo.py
    python examples/storkey_hopfield_demo.py --k_values 10,50 --noise_levels 0.0,1.0 --n_trials 5
    python examples/storkey_hopfield_demo.py --k_values 50 --noise_levels 0.0 --n_trials 2 --num_epochs 1

Results:
python examples/storkey_hopfield_demo.py --k_values 500 --noise_levels 2.0 --strength 2.0 --n_trials 5 --num_epochs 5
     K    Noise    Hopfield Accuracy %         MLP Accuracy %     Delta%    p-value   Sig        d
---------------------------------------------------------------------------------------------------
   500      2.0          65.17+/-0.42            61.36+/-0.32      +3.80     0.0001     *    6.885
---------------------------------------------------------------------------------------------------

python examples/storkey_hopfield_demo.py --k_values 5,10,20,50,100,500 --noise_levels 0.0,0.5,1.0,1.5,2.0 --strength 1.0 --n_trials 10 --num_epochs 5
RESULTS SUMMARY (sweep K examples per class, Gaussian noise with std n):
  Hopfield strength: 1.0
  Trials: 10, Epochs: 5
Delta Accuracy Heatmap (Hopfield - MLP, percentage points):

     K  n=0.0  n=0.5  n=1.0  n=1.5  n=2.0
-----------------------------------------
     5  -1.2   -0.9   -0.7   -0.5   -0.5
    10  -1.5   -1.1   -0.8   -0.3   -0.1
    20  -0.1   -0.0   +0.7   +1.7*  +2.4*
    50  -0.8*  -0.4   +0.7*  +1.7*  +2.6*
   100  -0.4*  +0.0   +1.0*  +1.9*  +3.0*
   500  +0.2   +0.7*  +1.9*  +2.9*  +3.8*

  * = significant at p<0.05
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import numpy as np
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode, StorkeyHopfield
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core import TanhActivation
from fabricpc.core.activations import SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.experiments.statistics import paired_ttest, cohens_d
from fabricpc.utils.data.dataloader import (
    FashionMnistLoader,
    FewShotLoader,
    NoisyTestLoader,
)

jax.config.update("jax_default_prng_impl", "threefry2x32")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Few-Shot + Noise: Storkey Hopfield vs MLP on Fashion-MNIST"
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="500",
        help="Comma-separated K (shots per class) values (default: 500)",
    )
    parser.add_argument(
        "--noise_levels",
        type=str,
        default="2.0",
        help="Comma-separated noise std values (default: 2.0)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of paired trials per condition (default: 10)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Training epochs per trial (default: 1)",
    )
    parser.add_argument(
        "--strength",
        type=str,
        default="2.0",
        help="Hopfield strength: float >=0 or 'None' for learnable (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model Factories
# ---------------------------------------------------------------------------

BATCH_SIZE = 64


def make_hopfield_factory(hopfield_strength=None):
    """Return a model factory closure with the given hopfield_strength."""

    def create_hopfield_model(rng_key):
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


def create_mlp_model(rng_key):
    """Standard MLP baseline with identical capacity."""
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


# ---------------------------------------------------------------------------
# Data Loader Factory
# ---------------------------------------------------------------------------


def make_data_factory(k_per_class, noise_std, batch_size):
    """Return a data_loader_factory(seed) for ABExperiment.

    Both arms receive identical training data (same K-shot subsample)
    and identical test noise (same seed).
    """

    def factory(seed):
        train_loader = FewShotLoader(
            dataset_name="fashion_mnist",
            split="train",
            k_per_class=k_per_class,
            batch_size=batch_size,
            num_classes=10,
            shuffle=True,
            seed=seed,
            tensor_format="flat",
            normalize_mean=0.2860,
            normalize_std=0.3530,
        )

        base_test_loader = FashionMnistLoader(
            split="test",
            batch_size=batch_size,
            shuffle=False,
            tensor_format="flat",
        )

        test_loader = NoisyTestLoader(
            base_loader=base_test_loader,
            noise_std=noise_std,
            seed=seed,
        )

        return train_loader, test_loader

    return factory


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    noise_levels = [float(n) for n in args.noise_levels.split(",")]

    if args.strength.lower() == "none":
        hopfield_strength = None
    else:
        hopfield_strength = float(args.strength)

    optimizer = optax.adamw(0.001, weight_decay=0.1)
    train_config = {"num_epochs": args.num_epochs}

    strength_label = (
        "learnable" if hopfield_strength is None else f"{hopfield_strength}"
    )

    print("=" * 70)
    print("Few-Shot + Noise Robustness: StorkeyHopfield vs MLP")
    print("=" * 70)
    print("Dataset: Fashion-MNIST")
    print("Hopfield: 784 -> 128(tanh) -> 128(StorkeyHopfield, tanh) -> 10(softmax, CE)")
    print("MLP:      784 -> 128(tanh) -> 128(tanh) -> 10(softmax, CE)")
    print(f"Hopfield strength: {strength_label}")
    print(f"Epochs per trial: {args.num_epochs}")
    print(f"Trials per condition: {args.n_trials}")
    print(f"K values (numer of examples per class): {k_values}")
    print(f"Noise levels (std dev): {noise_levels}")
    print()

    arm_hop = ExperimentArm(
        name="Hopfield",
        model_factory=make_hopfield_factory(hopfield_strength),
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

    # Collect results across the K x noise grid
    grid_results = []

    for k in k_values:
        for noise_std in noise_levels:
            print(f"\n{'='*70}")
            print(f"  K={k} shots/class, noise_std={noise_std}")
            print(f"{'='*70}")

            data_factory = make_data_factory(k, noise_std, BATCH_SIZE)

            experiment = ABExperiment(
                arm_a=arm_hop,
                arm_b=arm_mlp,
                metric="accuracy",
                data_loader_factory=data_factory,
                n_trials=args.n_trials,
                verbose=args.verbose,
            )

            result = experiment.run()

            hop_acc = result.arm_a_metrics
            mlp_acc = result.arm_b_metrics
            delta = hop_acc - mlp_acc

            row = {
                "k": k,
                "noise": noise_std,
                "hop_mean": float(np.mean(hop_acc)),
                "hop_se": (
                    float(np.std(hop_acc, ddof=1) / np.sqrt(len(hop_acc)))
                    if len(hop_acc) > 1
                    else 0.0
                ),
                "mlp_mean": float(np.mean(mlp_acc)),
                "mlp_se": (
                    float(np.std(mlp_acc, ddof=1) / np.sqrt(len(mlp_acc)))
                    if len(mlp_acc) > 1
                    else 0.0
                ),
                "delta_mean": float(np.mean(delta)),
            }

            if args.n_trials >= 2:
                ttest = paired_ttest(hop_acc, mlp_acc)
                effect = cohens_d(hop_acc, mlp_acc)
                row["p_value"] = ttest.p_value
                row["significant"] = ttest.significant_at_05
                row["cohens_d"] = effect.d
            else:
                row["p_value"] = float("nan")
                row["significant"] = False
                row["cohens_d"] = float("nan")

            grid_results.append(row)

            print(
                f"  -> Hopfield: {row['hop_mean']*100:.2f}%  "
                f"MLP: {row['mlp_mean']*100:.2f}%  "
                f"Delta: {row['delta_mean']*100:+.2f}%  "
                f"p={row['p_value']:.4f}"
            )

    # ---------------------------------------------------------------------------
    # Summary Tables
    # ---------------------------------------------------------------------------

    print("\n\n")
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Hopfield strength: {strength_label}")
    print(f"  Trials: {args.n_trials}, Epochs: {args.num_epochs}")
    print()

    # Per-condition table
    header = f"{'K':>6} {'Noise':>8} {'Hopfield%':>12} {'MLP%':>12} {'Delta%':>10} {'p-value':>10} {'Sig':>5} {'d':>8}"
    print(header)
    print("-" * len(header))
    for r in grid_results:
        hop_str = f"{r['hop_mean']*100:.2f}+/-{r['hop_se']*100:.2f}"
        mlp_str = f"{r['mlp_mean']*100:.2f}+/-{r['mlp_se']*100:.2f}"
        delta_str = f"{r['delta_mean']*100:+.2f}"
        p_str = f"{r['p_value']:.4f}" if not np.isnan(r["p_value"]) else "n/a"
        sig_str = "*" if r["significant"] else ""
        d_str = f"{r['cohens_d']:.3f}" if not np.isnan(r["cohens_d"]) else "n/a"
        print(
            f"{r['k']:>6} {r['noise']:>8.1f} {hop_str:>12} {mlp_str:>12} "
            f"{delta_str:>10} {p_str:>10} {sig_str:>5} {d_str:>8}"
        )
    print("-" * len(header))

    # Delta heatmap (K x noise)
    if len(k_values) > 1 and len(noise_levels) > 1:
        print()
        print("Delta Accuracy Heatmap (Hopfield - MLP, percentage points):")
        print()

        # Header row: noise levels
        noise_header = f"{'K':>6}" + "".join(f"  n={n:.1f}" for n in noise_levels)
        print(noise_header)
        print("-" * len(noise_header))

        for k in k_values:
            row_str = f"{k:>6}"
            for noise_std in noise_levels:
                match = [
                    r for r in grid_results if r["k"] == k and r["noise"] == noise_std
                ]
                if match:
                    d = match[0]["delta_mean"] * 100
                    sig = "*" if match[0]["significant"] else " "
                    row_str += f" {d:+5.1f}{sig}"
                else:
                    row_str += "    n/a"
            print(row_str)
        print()
        print("  * = significant at p<0.05")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
