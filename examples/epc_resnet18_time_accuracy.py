"""Validation-selected, time-budgeted ePC versus sPC ResNet-18 follow-up.

This experiment is deliberately split into two commands so the ePC
configuration is selected without constructing either holdout loader.

Tune on train[:80%], validate on train[80%:90%]:

    python examples/epc_resnet18_time_accuracy.py --mode tune

After copying the printed SELECTED values, lock them into the final command:

    python examples/epc_resnet18_time_accuracy.py --mode final \
        --epc_steps SELECTED_STEPS --epc_lr SELECTED_LR

The final phase trains on the same 40k examples, records validation curves,
and evaluates the fresh train[90%:] holdout plus the official CIFAR-10 test
split exactly once per trained model. The official test split is secondary:
it was already observed by the earlier two-epoch benchmark.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
from dataclasses import dataclass
import importlib.util
import os
import time
from typing import Iterable, Optional, Sequence, Tuple

import jax
import numpy as np
import optax

from fabricpc.core import EPCInference, InferenceBase, InferenceSGDNormClip
from fabricpc.experiments.statistics import cohens_d, paired_ttest
from fabricpc.training import evaluate_pcn, train_pcn
from fabricpc.utils.data.dataloader import Cifar10Loader

TRAIN_SPLIT = "train[:80%]"
VALIDATION_SPLIT = "train[80%:90%]"
HOLDOUT_SPLIT = "train[90%:]"
OFFICIAL_TEST_SPLIT = "test"

DEFAULT_TUNE_SEEDS = (271828, 314159)
DEFAULT_FINAL_SEEDS = (0, 1000, 2000)
DEFAULT_STEP_GRID = (2, 5, 10)
DEFAULT_LR_GRID = (3e-4, 1e-3, 3e-3)

_DEMO_PATH = os.path.join(os.path.dirname(__file__), "resnet18_cifar10_demo.py")
_SPEC = importlib.util.spec_from_file_location("resnet18_cifar10_demo", _DEMO_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load ResNet demo from {_DEMO_PATH}")
_RESNET_DEMO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RESNET_DEMO)


@dataclass(frozen=True)
class CurvePoint:
    epoch: int
    training_seconds: float
    validation_accuracy: float
    validation_energy: float


@dataclass(frozen=True)
class TrainingResult:
    seed: int
    name: str
    training_seconds: float
    curve: Tuple[CurvePoint, ...]
    holdout_accuracy: Optional[float] = None
    official_test_accuracy: Optional[float] = None


@dataclass(frozen=True)
class TuningObservation:
    steps: int
    learning_rate: float
    seed: int
    validation_accuracy: float
    training_seconds: float


@dataclass(frozen=True)
class CandidateSummary:
    steps: int
    learning_rate: float
    mean_validation_accuracy: float
    se_validation_accuracy: float
    mean_training_seconds: float


def _parse_csv_ints(value: str) -> Tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("grid values must be unique")
    return parsed


def _parse_csv_nonnegative_ints(value: str) -> Tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "expected comma-separated nonnegative integers"
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be unique")
    return parsed


def _parse_csv_floats(value: str) -> Tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(not np.isfinite(item) or item <= 0.0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive floats")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("grid values must be unique")
    return parsed


def _mean_se(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(np.mean(array)), float(np.std(array, ddof=1) / np.sqrt(array.size))


def summarize_candidates(
    observations: Iterable[TuningObservation],
) -> Tuple[CandidateSummary, ...]:
    grouped = {}
    for observation in observations:
        if observation.steps < 1:
            raise ValueError("tuning inference steps must be positive")
        if not np.isfinite(observation.learning_rate) or observation.learning_rate <= 0:
            raise ValueError("tuning learning rates must be finite and positive")
        if not np.isfinite(observation.validation_accuracy) or not (
            0.0 <= observation.validation_accuracy <= 1.0
        ):
            raise ValueError(
                "tuning validation accuracies must be finite probabilities"
            )
        if (
            not np.isfinite(observation.training_seconds)
            or observation.training_seconds <= 0
        ):
            raise ValueError("tuning times must be finite and positive")
        key = (observation.steps, observation.learning_rate)
        grouped.setdefault(key, []).append(observation)
    if not grouped:
        raise ValueError("at least one tuning observation is required")

    seed_sets = []
    for rows in grouped.values():
        seeds = [row.seed for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError("each tuning candidate must have unique seeds")
        seed_sets.append(set(seeds))
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("all tuning candidates must use the same seeds")

    summaries = []
    for (steps, learning_rate), rows in grouped.items():
        mean_accuracy, se_accuracy = _mean_se([row.validation_accuracy for row in rows])
        mean_seconds, _ = _mean_se([row.training_seconds for row in rows])
        summaries.append(
            CandidateSummary(
                steps=steps,
                learning_rate=learning_rate,
                mean_validation_accuracy=mean_accuracy,
                se_validation_accuracy=se_accuracy,
                mean_training_seconds=mean_seconds,
            )
        )
    return tuple(sorted(summaries, key=lambda row: (row.steps, row.learning_rate)))


def select_candidate(observations: Iterable[TuningObservation]) -> CandidateSummary:
    """Select mean validation accuracy; ties prefer fewer steps, then lower LR."""
    summaries = summarize_candidates(observations)
    return min(
        summaries,
        key=lambda row: (
            -row.mean_validation_accuracy,
            row.steps,
            row.learning_rate,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tune", "final"), required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--tune_epochs", type=int, default=3)
    parser.add_argument(
        "--tune_seeds",
        type=_parse_csv_nonnegative_ints,
        default=DEFAULT_TUNE_SEEDS,
    )
    parser.add_argument("--steps_grid", type=_parse_csv_ints, default=DEFAULT_STEP_GRID)
    parser.add_argument("--lr_grid", type=_parse_csv_floats, default=DEFAULT_LR_GRID)
    parser.add_argument("--epc_steps", type=int)
    parser.add_argument("--epc_lr", type=float)
    parser.add_argument("--epc_eta", type=float, default=0.1)
    parser.add_argument("--epc_epochs", type=int, default=13)
    parser.add_argument("--spc_steps", type=int, default=120)
    parser.add_argument("--spc_eta", type=float, default=0.1)
    parser.add_argument("--spc_lr", type=float, default=0.001)
    parser.add_argument("--spc_epochs", type=int, default=2)
    parser.add_argument(
        "--final_seeds",
        type=_parse_csv_nonnegative_ints,
        default=DEFAULT_FINAL_SEEDS,
    )
    args = parser.parse_args()

    positive_ints = {
        "batch_size": args.batch_size,
        "tune_epochs": args.tune_epochs,
        "epc_epochs": args.epc_epochs,
        "spc_steps": args.spc_steps,
        "spc_epochs": args.spc_epochs,
    }
    for name, value in positive_ints.items():
        if value < 1:
            parser.error(f"--{name} must be >= 1")
    positive_floats = {
        "epc_eta": args.epc_eta,
        "spc_eta": args.spc_eta,
        "spc_lr": args.spc_lr,
    }
    for name, value in positive_floats.items():
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and > 0")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight_decay must be finite and >= 0")
    if args.mode == "final":
        if args.epc_steps is None or args.epc_steps < 1:
            parser.error("--epc_steps must be supplied and >= 1 in final mode")
        if args.epc_lr is None or not np.isfinite(args.epc_lr) or args.epc_lr <= 0:
            parser.error("--epc_lr must be supplied and > 0 in final mode")
    return args


def _optimizer(learning_rate, weight_decay, epochs, steps_per_epoch):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )
    return optax.adamw(schedule, weight_decay=weight_decay)


def _create_model(rng_key, inference):
    return _RESNET_DEMO._create_mupc_model(
        rng_key,
        inference=inference,
        activation=_RESNET_DEMO.get_activation("relu"),
    )


def _run_training(
    *,
    name: str,
    seed: int,
    inference: InferenceBase,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    evaluate_holdout: bool,
) -> TrainingResult:
    master_key = jax.random.PRNGKey(seed)
    graph_key, train_key, validation_key, holdout_key, test_key = jax.random.split(
        master_key, 5
    )
    params, structure = _create_model(graph_key, inference)
    train_loader = Cifar10Loader(
        TRAIN_SPLIT, batch_size=batch_size, shuffle=True, seed=seed
    )
    validation_loader = Cifar10Loader(
        VALIDATION_SPLIT, batch_size=batch_size, shuffle=False, seed=seed
    )
    optimizer = _optimizer(learning_rate, weight_decay, epochs, len(train_loader))
    config = {"num_epochs": epochs}

    curve = []
    evaluation_seconds = 0.0
    start = time.perf_counter()

    def epoch_callback(epoch_idx, current_params, current_structure, _, __):
        nonlocal evaluation_seconds
        before_evaluation = time.perf_counter()
        training_seconds = before_evaluation - start - evaluation_seconds
        metrics = evaluate_pcn(
            current_params,
            current_structure,
            validation_loader,
            config,
            validation_key,
        )
        evaluation_seconds += time.perf_counter() - before_evaluation
        point = CurvePoint(
            epoch=epoch_idx + 1,
            training_seconds=training_seconds,
            validation_accuracy=float(metrics["accuracy"]),
            validation_energy=float(metrics["energy"]),
        )
        curve.append(point)
        print(
            "CURVE "
            f"name={name} seed={seed} epoch={point.epoch} "
            f"train_seconds={point.training_seconds:.6f} "
            f"validation_accuracy={point.validation_accuracy:.8f} "
            f"validation_energy={point.validation_energy:.8g}",
            flush=True,
        )
        return metrics

    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=config,
        rng_key=train_key,
        verbose=False,
        use_tqdm=False,
        epoch_callback=epoch_callback,
    )
    training_seconds = time.perf_counter() - start - evaluation_seconds

    holdout_accuracy = None
    official_test_accuracy = None
    if evaluate_holdout:
        holdout_loader = Cifar10Loader(
            HOLDOUT_SPLIT, batch_size=batch_size, shuffle=False, seed=seed
        )
        official_test_loader = Cifar10Loader(
            OFFICIAL_TEST_SPLIT,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        holdout_metrics = evaluate_pcn(
            trained_params, structure, holdout_loader, config, holdout_key
        )
        test_metrics = evaluate_pcn(
            trained_params, structure, official_test_loader, config, test_key
        )
        holdout_accuracy = float(holdout_metrics["accuracy"])
        official_test_accuracy = float(test_metrics["accuracy"])

    result = TrainingResult(
        seed=seed,
        name=name,
        training_seconds=training_seconds,
        curve=tuple(curve),
        holdout_accuracy=holdout_accuracy,
        official_test_accuracy=official_test_accuracy,
    )
    print(
        "RUN_RESULT "
        f"name={name} seed={seed} epochs={epochs} "
        f"train_seconds={training_seconds:.6f} "
        f"final_validation_accuracy={curve[-1].validation_accuracy:.8f} "
        f"holdout_accuracy={holdout_accuracy} "
        f"official_test_accuracy={official_test_accuracy}",
        flush=True,
    )
    return result


def run_tuning(args):
    observations = []
    print("PHASE tune", flush=True)
    print(
        f"SPLITS train={TRAIN_SPLIT} validation={VALIDATION_SPLIT} "
        f"holdout=UNTOUCHED official_test=UNTOUCHED",
        flush=True,
    )
    for steps in args.steps_grid:
        for learning_rate in args.lr_grid:
            for seed in args.tune_seeds:
                name = f"ePC-{steps}-lr-{learning_rate:g}"
                result = _run_training(
                    name=name,
                    seed=seed,
                    inference=EPCInference(eta_infer=args.epc_eta, infer_steps=steps),
                    epochs=args.tune_epochs,
                    learning_rate=learning_rate,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    evaluate_holdout=False,
                )
                observations.append(
                    TuningObservation(
                        steps=steps,
                        learning_rate=learning_rate,
                        seed=seed,
                        validation_accuracy=result.curve[-1].validation_accuracy,
                        training_seconds=result.training_seconds,
                    )
                )

    for summary in summarize_candidates(observations):
        print(
            "TUNE_SUMMARY "
            f"steps={summary.steps} lr={summary.learning_rate:.8g} "
            f"mean_validation_accuracy={summary.mean_validation_accuracy:.8f} "
            f"se_validation_accuracy={summary.se_validation_accuracy:.8f} "
            f"mean_train_seconds={summary.mean_training_seconds:.6f}",
            flush=True,
        )
    selected = select_candidate(observations)
    print(
        "SELECTED "
        f"steps={selected.steps} lr={selected.learning_rate:.8g} "
        f"mean_validation_accuracy={selected.mean_validation_accuracy:.8f} "
        f"se_validation_accuracy={selected.se_validation_accuracy:.8f}",
        flush=True,
    )


def _print_final_summary(results: Sequence[TrainingResult]):
    by_name = {}
    for result in results:
        by_name.setdefault(result.name, []).append(result)

    if len(by_name) != 2:
        raise ValueError("final comparison requires exactly one ePC and one sPC arm")
    epc_names = [name for name in by_name if name.startswith("ePC")]
    spc_names = [name for name in by_name if name.startswith("sPC")]
    if len(epc_names) != 1 or len(spc_names) != 1:
        raise ValueError("final comparison requires named ePC and sPC arms")
    arm_rows = list(by_name.values())
    seed_sets = [{row.seed for row in rows} for rows in arm_rows]
    if any(len(seeds) != len(rows) for seeds, rows in zip(seed_sets, arm_rows)):
        raise ValueError("each final arm must have unique seeds")
    if seed_sets[0] != seed_sets[1]:
        raise ValueError("final arms must use identical paired seeds")
    for result in results:
        if not result.curve:
            raise ValueError("each final result must contain a validation curve")
        if result.holdout_accuracy is None or result.official_test_accuracy is None:
            raise ValueError("each final result must contain both endpoint accuracies")
        endpoint_values = np.asarray(
            [
                result.training_seconds,
                result.curve[-1].validation_accuracy,
                result.holdout_accuracy,
                result.official_test_accuracy,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(endpoint_values)):
            raise ValueError("all final endpoints must be finite")
        if endpoint_values[0] <= 0 or np.any(
            (endpoint_values[1:] < 0) | (endpoint_values[1:] > 1)
        ):
            raise ValueError("final time must be positive and accuracies probabilities")

    for name, rows in by_name.items():
        train_mean, train_se = _mean_se([row.training_seconds for row in rows])
        validation_mean, validation_se = _mean_se(
            [row.curve[-1].validation_accuracy for row in rows]
        )
        holdout_mean, holdout_se = _mean_se(
            [float(row.holdout_accuracy) for row in rows]
        )
        test_mean, test_se = _mean_se(
            [float(row.official_test_accuracy) for row in rows]
        )
        print(
            "FINAL_SUMMARY "
            f"name={name} n={len(rows)} "
            f"mean_train_seconds={train_mean:.6f} se_train_seconds={train_se:.6f} "
            f"mean_validation_accuracy={validation_mean:.8f} "
            f"se_validation_accuracy={validation_se:.8f} "
            f"mean_holdout_accuracy={holdout_mean:.8f} "
            f"se_holdout_accuracy={holdout_se:.8f} "
            f"mean_official_test_accuracy={test_mean:.8f} "
            f"se_official_test_accuracy={test_se:.8f}",
            flush=True,
        )

    epc_name = epc_names[0]
    spc_name = spc_names[0]
    epc_by_seed = {row.seed: row for row in by_name[epc_name]}
    spc_by_seed = {row.seed: row for row in by_name[spc_name]}
    paired_seeds = sorted(seed_sets[0])
    epc_rows = [epc_by_seed[seed] for seed in paired_seeds]
    spc_rows = [spc_by_seed[seed] for seed in paired_seeds]
    epc_holdout = np.asarray([row.holdout_accuracy for row in epc_rows])
    spc_holdout = np.asarray([row.holdout_accuracy for row in spc_rows])
    epc_times = np.asarray([row.training_seconds for row in epc_rows])
    spc_times = np.asarray([row.training_seconds for row in spc_rows])
    holdout_test = paired_ttest(epc_holdout, spc_holdout)
    effect = cohens_d(epc_holdout, spc_holdout)
    time_ratio = float(np.mean(epc_times) / np.mean(spc_times))
    accuracy_pass = bool(np.mean(epc_holdout) >= np.mean(spc_holdout))
    time_pass = bool(np.mean(epc_times) <= np.mean(spc_times))
    print(
        "PAIRED_HOLDOUT "
        f"mean_difference={holdout_test.mean_difference:.8f} "
        f"t={holdout_test.t_statistic:.8f} p={holdout_test.p_value:.8f} "
        f"cohens_d={effect.d:.8f}",
        flush=True,
    )
    print(
        "PRACTICAL_GATE "
        f"accuracy_pass={accuracy_pass} time_pass={time_pass} "
        f"time_ratio={time_ratio:.8f} overall_pass={accuracy_pass and time_pass}",
        flush=True,
    )


def run_final(args):
    print("PHASE final", flush=True)
    print(
        f"LOCKED epc_steps={args.epc_steps} epc_lr={args.epc_lr:.8g} "
        f"epc_epochs={args.epc_epochs} spc_steps={args.spc_steps} "
        f"spc_lr={args.spc_lr:.8g} spc_epochs={args.spc_epochs}",
        flush=True,
    )
    print(
        f"SPLITS train={TRAIN_SPLIT} validation={VALIDATION_SPLIT} "
        f"holdout={HOLDOUT_SPLIT} official_test={OFFICIAL_TEST_SPLIT}",
        flush=True,
    )
    results = []
    for seed in args.final_seeds:
        results.append(
            _run_training(
                name=f"ePC-{args.epc_steps}-lr-{args.epc_lr:g}",
                seed=seed,
                inference=EPCInference(
                    eta_infer=args.epc_eta, infer_steps=args.epc_steps
                ),
                epochs=args.epc_epochs,
                learning_rate=args.epc_lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                evaluate_holdout=True,
            )
        )
        results.append(
            _run_training(
                name=f"sPC-{args.spc_steps}-lr-{args.spc_lr:g}",
                seed=seed,
                inference=InferenceSGDNormClip(
                    eta_infer=args.spc_eta,
                    infer_steps=args.spc_steps,
                    max_norm=1.0,
                ),
                epochs=args.spc_epochs,
                learning_rate=args.spc_lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                evaluate_holdout=True,
            )
        )
    _print_final_summary(results)


def main():
    args = parse_args()
    if args.mode == "tune":
        run_tuning(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
