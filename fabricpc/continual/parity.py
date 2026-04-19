"""
Notebook-parity regression harness for Split-MNIST continual learning.

This module provides representative V18-like and V20.2b-like presets for the
current FabricPC port. The goal is not to claim exact notebook reproduction.
The goal is to provide stable regression benchmarks that future refactors can
run and compare against checked-in baselines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
from fabricpc.continual.config import ExperimentConfig, make_config
from fabricpc.continual.data import build_split_mnist_loaders
from fabricpc.continual.trainer import SequentialTrainer


@dataclass(frozen=True)
class ParityProfile:
    """Named notebook-parity profile."""

    name: str
    description: str


@dataclass(frozen=True)
class MetricTolerance:
    """Absolute tolerance used for a benchmark metric."""

    abs_tol: float


@dataclass(frozen=True)
class ParityMetrics:
    """Summary metrics for a parity benchmark run."""

    profile: str
    seed: int
    num_tasks: int
    final_mean_accuracy: float
    average_forgetting: float
    support_diversity: int
    mean_test_accuracy: float
    mean_causal_examples: float
    mean_causal_mix_gate: float
    mean_transweave_sources: float
    total_training_time_s: float


@dataclass(frozen=True)
class MetricCheck:
    metric: str
    observed: float
    expected: float
    abs_diff: float
    abs_tol: float
    passed: bool


@dataclass(frozen=True)
class ParityComparison:
    """Comparison result for one parity run against a baseline."""

    profile: str
    passed: bool
    checks: Tuple[MetricCheck, ...]


PROFILES: Dict[str, ParityProfile] = {
    "v18_like": ParityProfile(
        name="v18_like",
        description=(
            "Teacher-first / later mechanisms muted. Representative of a V18-style "
            "control regime in the current port, not an exact notebook reimplementation."
        ),
    ),
    "v20_2b_like": ParityProfile(
        name="v20_2b_like",
        description=(
            "Replay-assisted / conservative online guidance enabled. Representative of "
            "the V20.2b control regime described in the notebooks and PDF."
        ),
    ),
}


DEFAULT_TOLERANCES: Dict[str, MetricTolerance] = {
    "final_mean_accuracy": MetricTolerance(abs_tol=0.03),
    "average_forgetting": MetricTolerance(abs_tol=0.03),
    "support_diversity": MetricTolerance(abs_tol=1.0),
    "mean_test_accuracy": MetricTolerance(abs_tol=0.03),
    "mean_causal_examples": MetricTolerance(abs_tol=4.0),
    "mean_causal_mix_gate": MetricTolerance(abs_tol=0.15),
    "mean_transweave_sources": MetricTolerance(abs_tol=1.0),
}


def create_parity_network_structure(config: ExperimentConfig):
    """Create the shared benchmark network used by the parity harness."""
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden1 = Linear(
        shape=(256,),
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        name="hidden1",
        weight_init=XavierInitializer(),
    )
    hidden2 = Linear(
        shape=(128,),
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        name="hidden2",
        weight_init=XavierInitializer(),
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="output",
        weight_init=XavierInitializer(),
    )
    return graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )


def make_parity_config(profile_name: str, seed: int = 42) -> ExperimentConfig:
    """Create a representative config preset for notebook-parity benchmarking."""
    if profile_name not in PROFILES:
        raise KeyError(f"Unknown parity profile: {profile_name}")

    cfg = make_config(quick_smoke=True)
    cfg.seed = seed
    cfg.training.training_mode = "pc"
    cfg.num_tasks = 3
    cfg.task_pairs = ((0, 1), (2, 3), (4, 5))
    cfg.training.fast_dev_max_train_batches = 4
    cfg.training.fast_dev_max_test_batches = 4

    if profile_name == "v18_like":
        cfg.support.replay_bank_support_enable = False
        cfg.support.causal_max_effective_scale = 0.0
        cfg.support.route_enable = False
        cfg.support.sb_enable = False
        cfg.typing.enable_demotion = False
        cfg.typing.replay_bank_demotion_enable = False
        cfg.transweave.enable = False
        cfg.composer_transweave.enable = False
        cfg.shell_demotion_transweave.enable = False
        cfg.per_weight_causal.enable = False
        cfg.cloud.enable = False
        cfg.composer.enable = False
        cfg.hierarchy.enable = False
    elif profile_name == "v20_2b_like":
        cfg.support.replay_bank_support_enable = True
        cfg.support.causal_max_effective_scale = 0.5
        cfg.support.route_enable = True
        cfg.support.sb_enable = True
        cfg.typing.enable_demotion = True
        cfg.typing.replay_bank_demotion_enable = True
        cfg.transweave.enable = True
        cfg.composer_transweave.enable = True
        cfg.shell_demotion_transweave.enable = True
        cfg.per_weight_causal.enable = True
        cfg.cloud.enable = True
        cfg.composer.enable = True
        cfg.hierarchy.enable = True

    return cfg


def compute_parity_metrics(
    profile_name: str,
    seed: int,
    trainer: SequentialTrainer,
    total_training_time_s: float,
) -> ParityMetrics:
    """Summarize a completed parity run."""
    acc_matrix = trainer.accuracy_matrix()
    final_row = (
        acc_matrix[-1] if acc_matrix.size > 0 else np.array([], dtype=np.float64)
    )
    final_row = final_row[final_row > 0]
    support_diversity = len(
        {tuple(summary.support_cols) for summary in trainer.summaries}
    )

    return ParityMetrics(
        profile=profile_name,
        seed=seed,
        num_tasks=len(trainer.summaries),
        final_mean_accuracy=float(np.mean(final_row)) if final_row.size > 0 else 0.0,
        average_forgetting=float(trainer.get_forgetting_metric()),
        support_diversity=int(support_diversity),
        mean_test_accuracy=(
            float(np.mean([summary.test_accuracy for summary in trainer.summaries]))
            if trainer.summaries
            else 0.0
        ),
        mean_causal_examples=(
            float(
                np.mean(
                    [summary.causal_selector_examples for summary in trainer.summaries]
                )
            )
            if trainer.summaries
            else 0.0
        ),
        mean_causal_mix_gate=(
            float(
                np.mean(
                    [summary.causal_selector_mix_gate for summary in trainer.summaries]
                )
            )
            if trainer.summaries
            else 0.0
        ),
        mean_transweave_sources=(
            float(
                np.mean(
                    [
                        summary.transweave_composer_sources
                        for summary in trainer.summaries
                    ]
                )
            )
            if trainer.summaries
            else 0.0
        ),
        total_training_time_s=float(total_training_time_s),
    )


def run_parity_profile(
    profile_name: str,
    *,
    seed: int = 42,
    data_root: str = "./data",
) -> ParityMetrics:
    """Run a notebook-parity profile end to end."""
    cfg = make_parity_config(profile_name, seed=seed)

    jax.config.update("jax_default_prng_impl", "threefry2x32")
    master_key = jax.random.PRNGKey(seed)
    init_key, train_key = jax.random.split(master_key)

    structure = create_parity_network_structure(cfg)
    params = initialize_params(structure, init_key)
    optimizer = optax.adamw(
        cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    trainer = SequentialTrainer(
        structure=structure,
        config=cfg,
        params=params,
        optimizer=optimizer,
        rng_key=train_key,
    )

    tasks = build_split_mnist_loaders(cfg, data_root=data_root)
    start = time.time()
    for task_data in tasks:
        trainer.train_task(task_data, verbose=False)
    total_training_time_s = time.time() - start

    return compute_parity_metrics(profile_name, seed, trainer, total_training_time_s)


def load_parity_baselines(path: Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def save_parity_baselines(
    metrics_by_profile: Dict[str, ParityMetrics],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(metrics) for name, metrics in metrics_by_profile.items()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def compare_against_baseline(
    observed: ParityMetrics,
    baseline: Dict[str, Any],
    tolerances: Optional[Dict[str, MetricTolerance]] = None,
) -> ParityComparison:
    """Compare one observed parity run against a baseline payload."""
    tolerances = tolerances or DEFAULT_TOLERANCES
    observed_dict = asdict(observed)
    checks = []
    for metric, tolerance in tolerances.items():
        expected = float(baseline[metric])
        actual = float(observed_dict[metric])
        diff = abs(actual - expected)
        checks.append(
            MetricCheck(
                metric=metric,
                observed=actual,
                expected=expected,
                abs_diff=diff,
                abs_tol=tolerance.abs_tol,
                passed=diff <= tolerance.abs_tol,
            )
        )
    passed = all(check.passed for check in checks)
    return ParityComparison(
        profile=observed.profile,
        passed=passed,
        checks=tuple(checks),
    )
