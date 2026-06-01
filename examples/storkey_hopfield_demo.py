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
   500      2.0          64.94+/-0.33            60.33+/-0.34      +4.61      0.0000     *    11.115
---------------------------------------------------------------------------------------------------

python examples/storkey_hopfield_demo.py --k_values 5,10,20,50,100,500 --noise_levels 0.0,0.5,1.0,1.5,2.0 --strength 1.0 --n_trials 10 --num_epochs 5
RESULTS SUMMARY (sweep K examples per class, Gaussian noise with std n):
  Hopfield strength: 1.0
  Trials: 10, Epochs: 5
Delta Accuracy Heatmap (Hopfield - MLP, percentage points):

     K  n=0.0  n=0.5  n=1.0  n=1.5  n=2.0
-----------------------------------------
     5  -1.2   -1.0   -0.7   -0.5   -0.4
    10  -2.3   -1.9   -0.8   +0.1   +0.2
    20  -0.5   -0.4   +0.5*  +1.3*  +2.1*
    50  -0.4   -0.2   +0.7*  +1.8*  +2.5*
   100  -0.4*  -0.2   +0.8*  +2.0*  +2.8*
   500  +0.3   +0.7*  +1.7*  +3.1*  +4.3*

  * = significant at p<0.05
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import numpy as np
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode, StorkeyHopfield
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
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
        "--all_hopfield",
        action="store_true",
        default=False,
        help=(
            "Fully-Hopfield MLP: run a Hopfield hidden layer at input width (784) "
            "with no entry projection (only the Linear softmax head remains), "
            "compared vs a vanilla MLP of the same shape. Default (off) reproduces "
            "the original single-Hopfield-node architecture (784->128(Linear)->"
            "128(StorkeyHopfield)->10)."
        ),
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


def make_model_factory(hidden_type, width, depth, hopfield_strength):
    """Return a model_factory(rng_key) for a classifier whose hidden stack is
    either all StorkeyHopfield ('hopfield') or all Linear ('linear') nodes.

    StorkeyHopfield is width-preserving (square W), so a Linear entry projection
    (784 -> width) is inserted only when width != 784. The output is always a
    Linear softmax + CrossEntropy head -- the classifier readout (an output node,
    not a hidden node), so "all hidden nodes Hopfield" still holds when
    hidden_type == 'hopfield'.

    Reproduces the original demo arms when (width=128, depth=1):
        'hopfield' -> pixels(784) -> entry(128,Linear) -> hidden0(128,SH) -> class(10)
        'linear'   -> pixels(784) -> entry(128,Linear) -> hidden0(128,Linear) -> class(10)
    Fully-Hopfield arm when (width=784, depth=1):
        'hopfield' -> pixels(784) -> hidden0(784,SH) -> class(10)   (no entry projection)
    """

    def create_model(rng_key):
        pixels = IdentityNode(shape=(784,), name="pixels")
        nodes = [pixels]
        edges = []
        prev = pixels

        # Entry projection only needed to change width into the hidden stack.
        if width != 784:
            entry = Linear(
                shape=(width,),
                activation=TanhActivation(),
                name="entry",
                weight_init=XavierInitializer(),
            )
            nodes.append(entry)
            edges.append(Edge(source=prev, target=entry.slot("in")))
            prev = entry

        # Hidden stack: all StorkeyHopfield or all Linear, at constant width.
        for i in range(depth):
            if hidden_type == "hopfield":
                hidden = StorkeyHopfield(
                    shape=(width,),
                    name=f"hidden{i}",
                    hopfield_strength=hopfield_strength,
                )
            else:
                hidden = Linear(
                    shape=(width,),
                    activation=TanhActivation(),
                    name=f"hidden{i}",
                    weight_init=XavierInitializer(),
                )
            nodes.append(hidden)
            edges.append(Edge(source=prev, target=hidden.slot("in")))
            prev = hidden

        # Linear softmax classifier head (output node, not a hidden node).
        output = Linear(
            shape=(10,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
            weight_init=XavierInitializer(),
        )
        nodes.append(output)
        edges.append(Edge(source=prev, target=output.slot("in")))

        structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=pixels, y=output),
            inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return create_model


def arch_str(hidden_type, width, depth):
    """Human-readable architecture string for the header print."""
    parts = ["784"]
    if width != 784:
        parts.append(f"{width}(Linear, tanh)")
    node = "StorkeyHopfield" if hidden_type == "hopfield" else "Linear"
    for _ in range(depth):
        parts.append(f"{width}({node}, tanh)")
    parts.append("10(softmax, CE)")
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Data Loader Factory
# ---------------------------------------------------------------------------


def make_data_factory(k_per_class, noise_std, batch_size):
    """Return a data_loader_factory(seed) for ABExperiment.

    Both arms receive identical training data (same K-shot subsample)
    and identical test noise within a trial.

    Seeds are domain-separated: the K-shot subsample is a function of
    the trial seed only (so the same training data is reused across
    noise_levels for a clean isolation of the noise effect), while the
    test-noise seed mixes (trial, noise_std) so different noise levels
    draw independent realizations rather than rescaled copies of one
    shared noise stream.
    """
    # Stable integer fingerprint of noise_std used as RNG entropy.
    noise_level_id = int(round(noise_std * 1_000_000))

    def factory(seed):
        train_seed = int(np.random.SeedSequence([seed, 0]).generate_state(1)[0])
        noise_seed = int(
            np.random.SeedSequence([seed, 1, noise_level_id]).generate_state(1)[0]
        )

        train_loader = FewShotLoader(
            dataset_name="fashion_mnist",
            split="train",
            k_per_class=k_per_class,
            batch_size=batch_size,
            num_classes=10,
            shuffle=True,
            seed=train_seed,
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
            seed=noise_seed,
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

    # Architecture: --all_hopfield runs the Hopfield layer at input width (784)
    # with no entry projection (fully-Hopfield MLP); default keeps the original
    # 128-width single-Hopfield-node architecture.
    hidden_width = 784 if args.all_hopfield else 128
    hidden_depth = 1

    title = (
        "Fully-Hopfield MLP vs vanilla MLP"
        if args.all_hopfield
        else "Few-Shot + Noise Robustness: StorkeyHopfield vs MLP"
    )

    print("=" * 70)
    print(title)
    print("=" * 70)
    print("Dataset: Fashion-MNIST")
    print(f"Hopfield: {arch_str('hopfield', hidden_width, hidden_depth)}")
    print(f"MLP:      {arch_str('linear', hidden_width, hidden_depth)}")
    print(f"Hopfield strength: {strength_label}")
    print(f"Epochs per trial: {args.num_epochs}")
    print(f"Trials per condition: {args.n_trials}")
    print(f"K values (numer of examples per class): {k_values}")
    print(f"Noise levels (std dev): {noise_levels}")
    print()

    arm_hop = ExperimentArm(
        name="Hopfield",
        model_factory=make_model_factory(
            "hopfield", hidden_width, hidden_depth, hopfield_strength
        ),
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=train_config,
    )

    arm_mlp = ExperimentArm(
        name="MLP",
        model_factory=make_model_factory(
            "linear", hidden_width, hidden_depth, hopfield_strength
        ),
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
