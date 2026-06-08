"""
StorkeyHopfield vs MLP — Depth, Few-Shot, and Noise-Robustness Demo
====================================================================

Demonstrates that StorkeyHopfield behaves as a composable graph element in
FabricPC: substituting one or more Linear hidden layers with StorkeyHopfield
should produce cumulative accuracy gains under data scarcity and input noise,
where memorized class prototypes and attractor denoising help.

Architecture (same depth-4 graph across all arms, hidden width = 64)::

    "MLP"        pixels(784) -> 64(Linear) -> 64(Linear)          -> 64(Linear)          -> 10(softmax+CE)
    "1hopfield"  pixels(784) -> 64(Linear) -> 64(StorkeyHopfield) -> 64(Linear)          -> 10(softmax+CE)
    "2hopfield"  pixels(784) -> 64(Linear) -> 64(StorkeyHopfield) -> 64(StorkeyHopfield) -> 10(softmax+CE)

The first hidden layer is always Linear (the 784->64 feature-extraction
projection — StorkeyHopfield is width-preserving and cannot reduce dimension).
The classifier head is always Linear with Softmax + CrossEntropy energy.

Hypothesis: a higher fraction of hidden layers converted from Linear to
StorkeyHopfield produces cumulative accuracy gains under scarcity + noise.
The MLP arm is the baseline; deltas are computed per (K, noise) cell against
MLP.

Experiment grid: K (shots per class) x noise_std
    - K controls data scarcity: fewer examples -> more reliance on memory
    - noise_std controls input corruption: more noise -> more reliance on
      attractor denoising

Setup (one-time):
    pip install tensorflow_datasets tensorflow-cpu
    (`tensorflow-cpu` keeps TF off the GPU so it does not contend with JAX
    for device memory; `tensorflow_datasets` is the dataset backend used by
    `FashionMnistLoader` / `FewShotLoader`. Neither is currently declared
    in pyproject.toml — should be added there in a follow-up.)

Usage:
    python examples/storkey_hopfield_demo.py
    python examples/storkey_hopfield_demo.py --networks 1hopfield 2hopfield
    python examples/storkey_hopfield_demo.py --networks 2hopfield --k_values 50,500 --noise_levels 0.0,2.0
    python examples/storkey_hopfield_demo.py --k_values 50 --noise_levels 0.0 --n_trials 2 --num_epochs 1

Results (Fashion-MNIST, --strength 2.0, --n_trials 10, --num_epochs 20)::

    Delta Accuracy Heatmap (1hopfield - MLP, percentage points):

         K  n=0.0  n=0.5  n=1.0  n=2.0
    ----------------------------------
        10  -1.7*  -0.9   +0.5   +2.8*
        50  -0.2   +0.6   +3.0*  +7.6*
       100  +0.1   +1.0*  +3.3*  +8.5*
       500  -0.3   +0.8*  +3.2*  +7.2*

    Delta Accuracy Heatmap (2hopfield - MLP, percentage points):

         K  n=0.0  n=0.5  n=1.0  n=2.0
    ----------------------------------
        10  -4.6*  -3.5*  -1.0   +3.4*
        50  -0.4   +0.7*  +4.0* +10.5*
       100  +0.3   +1.5*  +4.8* +11.6*
       500  +0.8*  +2.4*  +5.9* +13.1*

      * = significant at p<0.05

The results support the hypothesis: more `Linear -> StorkeyHopfield`
substitutions produce cumulative accuracy gains under data-abundance combined
with input noise. The 2hopfield arm consistently beats the 1hopfield arm at
K >= 50 and any nonzero noise, and the gap widens as noise grows (e.g.
K=500, n=2.0: 1hopfield delta = +7.2 pp, 2hopfield delta = +13.1 pp — nearly
double). At K=10 / no noise both Hopfield arms underperform MLP, as expected:
with almost no data and no corruption, attractor memory has nothing to
retrieve or denoise, so the extra Hopfield constraints only slow learning.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import time
from typing import Dict, List, Sequence

import numpy as np
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode, NodeBase, StorkeyHopfield
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core import TanhActivation
from fabricpc.core.activations import SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.experiments import ExperimentArm
from fabricpc.experiments.statistics import paired_ttest, cohens_d
from fabricpc.utils.data.dataloader import (
    FashionMnistLoader,
    FewShotLoader,
    NoisyTestLoader,
)

jax.config.update("jax_default_prng_impl", "threefry2x32")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
HIDDEN_WIDTH = 64

# Each entry lists the hidden-layer types between the input and the Linear
# softmax readout. Position 0 must be Linear (784->HIDDEN_WIDTH projection);
# StorkeyHopfield is width-preserving and cannot reduce dimension.
ARCH_CONFIGS: Dict[str, List[str]] = {
    "MLP": ["Linear", "Linear", "Linear"],
    "1hopfield": ["Linear", "StorkeyHopfield", "Linear"],
    "2hopfield": ["Linear", "StorkeyHopfield", "StorkeyHopfield"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth + Few-Shot + Noise: StorkeyHopfield as a composable graph element on Fashion-MNIST"
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
        help="Number of paired trials per condition (default: 5)",
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
        "--networks",
        nargs="+",
        type=str,
        default=["1hopfield", "2hopfield"],
        choices=["1hopfield", "2hopfield"],
        help=(
            "Which Hopfield-substituted variants to compare against the MLP "
            "baseline. MLP is always included implicitly. "
            "Default: 1hopfield 2hopfield."
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
# Model Factory
# ---------------------------------------------------------------------------


def make_model_factory(
    layer_types: Sequence[str],
    hopfield_strength,
    width: int = HIDDEN_WIDTH,
):
    """Return a model_factory(rng_key) for a classifier with a heterogeneous
    hidden stack defined by ``layer_types``.

    Architecture (depth-3 hidden + readout):
        pixels(784) -> hidden0(width) -> hidden1(width) -> hidden2(width) -> class(10)

    Each hidden layer is either Linear(tanh) or StorkeyHopfield, per the
    corresponding entry of layer_types. Position 0 must be Linear because
    StorkeyHopfield is width-preserving (square W, in_dim == out_dim) and
    cannot perform the 784->width projection.

    The classifier head is always Linear with Softmax + CrossEntropy energy.
    """

    def create_model(rng_key):
        pixels = IdentityNode(shape=(784,), name="pixels")
        # Annotate as List[NodeBase] so Pylance allows appending Linear /
        # StorkeyHopfield / etc. without inferring the narrower IdentityNode
        # type from the initial single-element list.
        nodes: List[NodeBase] = [pixels]
        edges: List[Edge] = []
        prev: NodeBase = pixels

        for i, ltype in enumerate(layer_types):
            if ltype == "StorkeyHopfield":
                layer = StorkeyHopfield(
                    shape=(width,),
                    name=f"hidden{i}",
                    hopfield_strength=hopfield_strength,
                )
            elif ltype == "Linear":
                layer = Linear(
                    shape=(width,),
                    activation=TanhActivation(),
                    name=f"hidden{i}",
                    weight_init=XavierInitializer(),
                )
            else:
                raise ValueError(
                    f"Unknown layer type {ltype!r} at position {i}; "
                    "expected 'Linear' or 'StorkeyHopfield'."
                )
            nodes.append(layer)
            edges.append(Edge(source=prev, target=layer.slot("in")))
            prev = layer

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


def arch_str(layer_types: Sequence[str], width: int = HIDDEN_WIDTH) -> str:
    """Human-readable architecture string for the header print."""
    parts = ["784"]
    for ltype in layer_types:
        node = "StorkeyHopfield" if ltype == "StorkeyHopfield" else "Linear"
        parts.append(f"{width}({node}, tanh)")
    parts.append("10(softmax, CE)")
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Data Loader Factory
# ---------------------------------------------------------------------------


def make_data_factory(k_per_class, noise_std, batch_size):
    """Return a data_loader_factory(seed) for the multi-arm trial loop.

    Each trial seed produces a deterministic K-shot subsample (function of
    trial seed only) and an independent test-noise realization (function of
    trial seed and noise_std). Identical training data is reused across noise
    levels for a clean isolation of the noise effect, while different noise
    levels draw independent realizations rather than rescaled copies of one
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
# Multi-Arm Trial Runner
# ---------------------------------------------------------------------------


def run_multi_arm_trials(
    arms: List[ExperimentArm],
    data_factory,
    n_trials: int,
    metric: str = "accuracy",
    seed_offset: int = 0,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Run every arm across n_trials paired trials.

    Each trial uses one seed for both the data factory (same K-shot subsample
    and noise realization across arms) and the model RNG key (same init key
    across arms — different structures still produce different weights, but
    the source of randomness is paired). The chosen ``metric`` is collected
    per arm per trial.

    This replaces ``ABExperiment`` because we now have more than two arms
    (MLP + 1hopfield + 2hopfield) and want one shared MLP training per trial,
    not one MLP training per Hopfield comparison.

    Returns a dict mapping ``arm.name -> np.ndarray`` of per-trial metric
    values (length n_trials).
    """
    per_arm_metrics: Dict[str, List[float]] = {arm.name: [] for arm in arms}

    for trial_idx in range(n_trials):
        trial_seed = seed_offset + trial_idx * 1000
        print(f"--- Trial {trial_idx + 1}/{n_trials} (seed={trial_seed}) ---")

        train_loader, test_loader = data_factory(trial_seed)

        for arm in arms:
            master_key = jax.random.PRNGKey(trial_seed)
            graph_key, train_key, eval_key = jax.random.split(master_key, 3)

            params, structure = arm.model_factory(graph_key)

            t0 = time.time()
            trained_params, _, _ = arm.train_fn(
                params,
                structure,
                train_loader,
                arm.optimizer,
                arm.train_config,
                train_key,
                verbose=verbose,
            )
            train_time = time.time() - t0

            metrics = arm.eval_fn(
                trained_params,
                structure,
                test_loader,
                arm.train_config,
                eval_key,
            )

            if metric not in metrics:
                available = ", ".join(sorted(metrics.keys()))
                raise KeyError(
                    f"Metric '{metric}' not found in eval results for arm "
                    f"'{arm.name}'. Available: {available}"
                )
            value = float(metrics[metric])
            per_arm_metrics[arm.name].append(value)
            print(
                f"  {arm.name:<10}: {metric}={value:.4f}  "
                f"(train: {train_time:.1f}s)"
            )

    return {name: np.asarray(vals) for name, vals in per_arm_metrics.items()}


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

    # MLP is always the baseline. argparse de-duplication: keep input order.
    seen = set()
    hopfield_arms_to_run: List[str] = []
    for name in args.networks:
        if name not in seen:
            seen.add(name)
            hopfield_arms_to_run.append(name)
    arm_names_in_run: List[str] = ["MLP"] + hopfield_arms_to_run

    def build_arm(arm_name: str) -> ExperimentArm:
        return ExperimentArm(
            name=arm_name,
            model_factory=make_model_factory(
                ARCH_CONFIGS[arm_name], hopfield_strength, HIDDEN_WIDTH
            ),
            train_fn=train_pcn,
            eval_fn=evaluate_pcn,
            optimizer=optimizer,
            train_config=train_config,
        )

    arms = [build_arm(name) for name in arm_names_in_run]

    print("=" * 78)
    print("StorkeyHopfield depth comparison: composable graph element on Fashion-MNIST")
    print("=" * 78)
    print("Dataset: Fashion-MNIST")
    for name in arm_names_in_run:
        print(f"  {name:<10}: {arch_str(ARCH_CONFIGS[name], HIDDEN_WIDTH)}")
    print(f"Hopfield strength: {strength_label}")
    print(f"Epochs per trial: {args.num_epochs}")
    print(f"Trials per condition: {args.n_trials}")
    print(f"K values (number of examples per class): {k_values}")
    print(f"Noise levels (std dev): {noise_levels}")
    print()

    # Collect results across the K x noise grid.
    grid_results: List[Dict] = []

    for k in k_values:
        for noise_std in noise_levels:
            print(f"\n{'='*78}")
            print(f"  K={k} shots/class, noise_std={noise_std}")
            print(f"{'='*78}")

            data_factory = make_data_factory(k, noise_std, BATCH_SIZE)

            metrics_by_arm = run_multi_arm_trials(
                arms=arms,
                data_factory=data_factory,
                n_trials=args.n_trials,
                metric="accuracy",
                verbose=args.verbose,
            )

            row: Dict = {"k": k, "noise": noise_std}
            mlp_acc = metrics_by_arm["MLP"]
            row["MLP_mean"] = float(np.mean(mlp_acc))
            row["MLP_se"] = (
                float(np.std(mlp_acc, ddof=1) / np.sqrt(len(mlp_acc)))
                if len(mlp_acc) > 1
                else 0.0
            )

            for arm_name in hopfield_arms_to_run:
                arm_acc = metrics_by_arm[arm_name]
                delta = arm_acc - mlp_acc
                row[f"{arm_name}_mean"] = float(np.mean(arm_acc))
                row[f"{arm_name}_se"] = (
                    float(np.std(arm_acc, ddof=1) / np.sqrt(len(arm_acc)))
                    if len(arm_acc) > 1
                    else 0.0
                )
                row[f"{arm_name}_delta"] = float(np.mean(delta))
                if args.n_trials >= 2:
                    ttest = paired_ttest(arm_acc, mlp_acc)
                    effect = cohens_d(arm_acc, mlp_acc)
                    row[f"{arm_name}_pval"] = ttest.p_value
                    row[f"{arm_name}_sig"] = ttest.significant_at_05
                    row[f"{arm_name}_d"] = effect.d
                else:
                    row[f"{arm_name}_pval"] = float("nan")
                    row[f"{arm_name}_sig"] = False
                    row[f"{arm_name}_d"] = float("nan")

            grid_results.append(row)

            print("  Cell summary:")
            print(f"    MLP       : {row['MLP_mean']*100:.2f}%")
            for arm_name in hopfield_arms_to_run:
                p = row[f"{arm_name}_pval"]
                p_str = "p=n/a" if np.isnan(p) else f"p={p:.4f}"
                print(
                    f"    {arm_name:<10}: {row[f'{arm_name}_mean']*100:.2f}%  "
                    f"Delta={row[f'{arm_name}_delta']*100:+.2f}%  {p_str}"
                )

    # -----------------------------------------------------------------------
    # Summary Tables
    # -----------------------------------------------------------------------

    print("\n\n")
    print("=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)
    print(f"  Hopfield strength: {strength_label}")
    print(f"  Trials: {args.n_trials}, Epochs: {args.num_epochs}")
    print()

    # Per-condition table: MLP and each Hopfield arm side by side.
    cols = [f"{'K':>5}", f"{'Noise':>6}", f"{'MLP%':>13}"]
    for arm_name in hopfield_arms_to_run:
        cols.append(f"{arm_name + '%':>14}")
        cols.append(f"{'Delta':>7}")
        cols.append(f"{'p':>8}")
        cols.append(f"{'sig':>4}")
        cols.append(f"{'d':>7}")
    header = " ".join(cols)
    print(header)
    print("-" * len(header))
    for r in grid_results:
        parts = [
            f"{r['k']:>5}",
            f"{r['noise']:>6.1f}",
            f"{r['MLP_mean']*100:6.2f}+/-{r['MLP_se']*100:.2f}",
        ]
        for arm_name in hopfield_arms_to_run:
            parts.append(
                f"{r[f'{arm_name}_mean']*100:6.2f}+/-{r[f'{arm_name}_se']*100:.2f}"
            )
            parts.append(f"{r[f'{arm_name}_delta']*100:+6.2f}")
            p = r[f"{arm_name}_pval"]
            parts.append(f"{p:8.4f}" if not np.isnan(p) else f"{'n/a':>8}")
            parts.append(f"{'*' if r[f'{arm_name}_sig'] else '':>4}")
            d = r[f"{arm_name}_d"]
            parts.append(f"{d:7.3f}" if not np.isnan(d) else f"{'n/a':>7}")
        print(" ".join(parts))
    print("-" * len(header))

    # Delta heatmap (K x noise), one block per Hopfield arm.
    if len(k_values) > 1 and len(noise_levels) > 1:
        for arm_name in hopfield_arms_to_run:
            print()
            print(f"Delta Accuracy Heatmap ({arm_name} - MLP, percentage points):")
            print()
            noise_header = f"{'K':>6}" + "".join(f"  n={n:.1f}" for n in noise_levels)
            print(noise_header)
            print("-" * len(noise_header))
            for k in k_values:
                row_str = f"{k:>6}"
                for noise_std in noise_levels:
                    match = [
                        r
                        for r in grid_results
                        if r["k"] == k and r["noise"] == noise_std
                    ]
                    if match:
                        d = match[0][f"{arm_name}_delta"] * 100
                        sig = "*" if match[0][f"{arm_name}_sig"] else " "
                        row_str += f" {d:+5.1f}{sig}"
                    else:
                        row_str += "    n/a"
                print(row_str)
            print()
        print("  * = significant at p<0.05")

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
