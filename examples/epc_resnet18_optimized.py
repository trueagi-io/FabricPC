"""Independently tune ePC and sPC, then compare accuracy against GPU time.

Tuning constructs only the training and validation loaders:

    python examples/epc_resnet18_optimized.py --mode tune

Final mode requires the complete lock printed by tuning and evaluates the
held-out endpoint exactly once at each run's best-validation checkpoint.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
from dataclasses import dataclass
import importlib.util
import os
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

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

SCREEN_SEED = 271828
CONFIRM_SEED = 314159
FINAL_SEEDS = (0, 1000, 2000)

EPC_STEP_GRID = (2, 5, 10)
EPC_LR_GRID = (3e-5, 1e-4, 3e-4)
SPC_STEP_GRID = (120,)
SPC_LR_GRID = (3e-4, 1e-3, 3e-3)
DECAY_EPOCH_GRID = (3, 5)

OBSERVATION_EPOCHS = 8
WARMUP_EPOCHS = 0.25
FLOOR_RATIO = 0.01
STABILITY_TAIL_EPOCHS = 3
STABILITY_TOLERANCE = 0.01
NONINFERIORITY_MARGIN = 0.01
SHORTLIST_SIZE = 2

_DEMO_PATH = os.path.join(os.path.dirname(__file__), "resnet18_cifar10_demo.py")
_SPEC = importlib.util.spec_from_file_location("resnet18_cifar10_demo", _DEMO_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load ResNet demo from {_DEMO_PATH}")
_RESNET_DEMO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RESNET_DEMO)


@dataclass(frozen=True)
class Recipe:
    solver: str
    inference_steps: int
    inference_rate: float
    peak_learning_rate: float
    decay_epochs: int

    @property
    def name(self) -> str:
        return (
            f"{self.solver}-s{self.inference_steps}"
            f"-lr{self.peak_learning_rate:g}-d{self.decay_epochs}"
        )


@dataclass(frozen=True)
class CurvePoint:
    epoch: int
    training_seconds: float
    validation_accuracy: float
    validation_energy: float
    best_accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    recipe: Recipe
    seed: int
    total_training_seconds: float
    curve: Tuple[CurvePoint, ...]
    best_epoch: int
    best_training_seconds: float
    best_validation_accuracy: float
    final_validation_accuracy: float
    endpoint_accuracy: Optional[float] = None
    endpoint_seconds: Optional[float] = None


@dataclass(frozen=True)
class CandidateSummary:
    recipe: Recipe
    mean_best_validation_accuracy: float
    se_best_validation_accuracy: float
    mean_time_to_best: float
    stable: bool


def _parse_csv_positive_ints(value: str) -> Tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be unique")
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


def _parse_csv_positive_floats(value: str) -> Tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(not np.isfinite(item) or item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive floats")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be unique")
    return parsed


def _mean_se(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError("summary values must be finite")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(np.mean(array)), float(np.std(array, ddof=1) / np.sqrt(array.size))


def make_candidate_grid(
    solver: str,
    step_grid: Iterable[int],
    learning_rate_grid: Iterable[float],
    decay_epoch_grid: Iterable[int],
    inference_rate: float,
) -> Tuple[Recipe, ...]:
    if solver not in ("ePC", "sPC"):
        raise ValueError("solver must be ePC or sPC")
    return tuple(
        Recipe(solver, steps, inference_rate, learning_rate, decay_epochs)
        for steps in step_grid
        for learning_rate in learning_rate_grid
        for decay_epochs in decay_epoch_grid
    )


def _learning_rate_schedule(
    recipe: Recipe,
    steps_per_epoch: int,
    warmup_epochs: float,
    floor_ratio: float,
):
    warmup_steps = max(1, round(warmup_epochs * steps_per_epoch))
    decay_steps = max(warmup_steps + 1, round(recipe.decay_epochs * steps_per_epoch))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=recipe.peak_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=recipe.peak_learning_rate * floor_ratio,
    )


def _optimizer(
    recipe: Recipe,
    weight_decay: float,
    steps_per_epoch: int,
    warmup_epochs: float,
    floor_ratio: float,
):
    schedule = _learning_rate_schedule(
        recipe, steps_per_epoch, warmup_epochs, floor_ratio
    )
    return optax.adamw(schedule, weight_decay=weight_decay)


def _inference(recipe: Recipe) -> InferenceBase:
    if recipe.solver == "ePC":
        return EPCInference(
            eta_infer=recipe.inference_rate,
            infer_steps=recipe.inference_steps,
        )
    if recipe.solver == "sPC":
        return InferenceSGDNormClip(
            eta_infer=recipe.inference_rate,
            infer_steps=recipe.inference_steps,
            max_norm=1.0,
        )
    raise ValueError(f"unsupported solver: {recipe.solver}")


def _create_model(rng_key, inference):
    return _RESNET_DEMO._create_mupc_model(
        rng_key,
        inference=inference,
        activation=_RESNET_DEMO.get_activation("relu"),
    )


def is_stable(
    result: TrainingResult,
    tail_epochs: int = STABILITY_TAIL_EPOCHS,
    tolerance: float = STABILITY_TOLERANCE,
) -> bool:
    if tail_epochs < 1 or len(result.curve) < tail_epochs:
        return False
    accuracies = np.asarray(
        [point.validation_accuracy for point in result.curve], dtype=np.float64
    )
    if not np.all(np.isfinite(accuracies)):
        return False
    tail = accuracies[-tail_epochs:]
    return bool(
        np.max(tail) - np.min(tail) <= tolerance
        and np.max(accuracies) - np.max(tail) <= tolerance
    )


def time_to_accuracy(result: TrainingResult, target: float) -> Optional[float]:
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("target must be a finite probability")
    for point in result.curve:
        if point.validation_accuracy >= target:
            return point.training_seconds
    return None


def _run_training(
    *,
    recipe: Recipe,
    seed: int,
    epochs: int,
    weight_decay: float,
    batch_size: int,
    warmup_epochs: float,
    floor_ratio: float,
    stability_tail_epochs: int,
    stability_tolerance: float,
    evaluate_endpoint: bool,
) -> TrainingResult:
    master_key = jax.random.PRNGKey(seed)
    graph_key, train_key, validation_key, endpoint_key = jax.random.split(master_key, 4)
    params, structure = _create_model(graph_key, _inference(recipe))
    train_loader = Cifar10Loader(
        TRAIN_SPLIT, batch_size=batch_size, shuffle=True, seed=seed
    )
    validation_loader = Cifar10Loader(
        VALIDATION_SPLIT, batch_size=batch_size, shuffle=False, seed=seed
    )
    optimizer = _optimizer(
        recipe,
        weight_decay,
        len(train_loader),
        warmup_epochs,
        floor_ratio,
    )
    config = {"num_epochs": epochs}

    curve = []
    best_params = None
    best_epoch = 0
    best_training_seconds = 0.0
    best_validation_accuracy = -np.inf
    callback_seconds = 0.0
    start = time.perf_counter()

    def epoch_callback(epoch_idx, current_params, current_structure, _, __):
        nonlocal best_params
        nonlocal best_epoch
        nonlocal best_training_seconds
        nonlocal best_validation_accuracy
        nonlocal callback_seconds

        callback_start = time.perf_counter()
        training_seconds = callback_start - start - callback_seconds
        metrics = evaluate_pcn(
            current_params,
            current_structure,
            validation_loader,
            config,
            validation_key,
        )
        accuracy = float(metrics["accuracy"])
        energy = float(metrics["energy"])
        if not np.isfinite(accuracy) or not np.isfinite(energy):
            raise FloatingPointError("validation metrics must be finite")
        if not 0 <= accuracy <= 1:
            raise ValueError("validation accuracy must be a probability")
        if accuracy > best_validation_accuracy:
            best_params = current_params
            best_epoch = epoch_idx + 1
            best_training_seconds = training_seconds
            best_validation_accuracy = accuracy
        point = CurvePoint(
            epoch=epoch_idx + 1,
            training_seconds=training_seconds,
            validation_accuracy=accuracy,
            validation_energy=energy,
            best_accuracy=best_validation_accuracy,
        )
        curve.append(point)
        print(
            "CURVE "
            f"name={recipe.name} seed={seed} epoch={point.epoch} "
            f"train_seconds={point.training_seconds:.6f} "
            f"validation_accuracy={point.validation_accuracy:.8f} "
            f"best_accuracy={point.best_accuracy:.8f} "
            f"validation_energy={point.validation_energy:.8g}",
            flush=True,
        )
        callback_seconds += time.perf_counter() - callback_start
        return metrics

    train_pcn(
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
    total_training_seconds = time.perf_counter() - start - callback_seconds
    if best_params is None or not curve:
        raise RuntimeError("training produced no checkpoint")

    endpoint_accuracy = None
    endpoint_seconds = None
    if evaluate_endpoint:
        endpoint_loader = Cifar10Loader(
            HOLDOUT_SPLIT,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        endpoint_start = time.perf_counter()
        endpoint_metrics = evaluate_pcn(
            best_params,
            structure,
            endpoint_loader,
            config,
            endpoint_key,
        )
        endpoint_seconds = time.perf_counter() - endpoint_start
        endpoint_accuracy = float(endpoint_metrics["accuracy"])
        if not np.isfinite(endpoint_accuracy) or not 0 <= endpoint_accuracy <= 1:
            raise FloatingPointError("endpoint accuracy must be a finite probability")

    result = TrainingResult(
        recipe=recipe,
        seed=seed,
        total_training_seconds=total_training_seconds,
        curve=tuple(curve),
        best_epoch=best_epoch,
        best_training_seconds=best_training_seconds,
        best_validation_accuracy=best_validation_accuracy,
        final_validation_accuracy=curve[-1].validation_accuracy,
        endpoint_accuracy=endpoint_accuracy,
        endpoint_seconds=endpoint_seconds,
    )
    print(
        "RUN_RESULT "
        f"name={recipe.name} seed={seed} epochs={epochs} "
        f"total_train_seconds={result.total_training_seconds:.6f} "
        f"best_epoch={result.best_epoch} "
        f"time_to_best={result.best_training_seconds:.6f} "
        f"best_validation_accuracy={result.best_validation_accuracy:.8f} "
        f"final_validation_accuracy={result.final_validation_accuracy:.8f} "
        f"stable={is_stable(result, stability_tail_epochs, stability_tolerance)} "
        f"endpoint_accuracy={endpoint_accuracy} "
        f"endpoint_seconds={endpoint_seconds}",
        flush=True,
    )
    return result


def _recipe_rank(result: TrainingResult, tail_epochs: int, tolerance: float):
    return (
        not is_stable(result, tail_epochs, tolerance),
        -result.best_validation_accuracy,
        result.best_training_seconds,
        result.recipe.inference_steps,
        result.recipe.peak_learning_rate,
        result.recipe.decay_epochs,
    )


def shortlist_candidates(
    screen_results: Iterable[TrainingResult],
    shortlist_size: int,
    tail_epochs: int = STABILITY_TAIL_EPOCHS,
    tolerance: float = STABILITY_TOLERANCE,
) -> Tuple[Recipe, ...]:
    rows = tuple(screen_results)
    if shortlist_size < 1 or shortlist_size > len(rows):
        raise ValueError("shortlist size must be within the candidate count")
    if len({row.recipe for row in rows}) != len(rows):
        raise ValueError("screen results must contain unique recipes")
    return tuple(
        row.recipe
        for row in sorted(
            rows, key=lambda row: _recipe_rank(row, tail_epochs, tolerance)
        )[:shortlist_size]
    )


def summarize_candidate(
    recipe: Recipe,
    results: Sequence[TrainingResult],
    tail_epochs: int = STABILITY_TAIL_EPOCHS,
    tolerance: float = STABILITY_TOLERANCE,
) -> CandidateSummary:
    if not results or any(result.recipe != recipe for result in results):
        raise ValueError("candidate results must be nonempty and use one recipe")
    seeds = [result.seed for result in results]
    if len(seeds) != len(set(seeds)):
        raise ValueError("candidate results must use unique seeds")
    mean_accuracy, se_accuracy = _mean_se(
        [result.best_validation_accuracy for result in results]
    )
    mean_time, _ = _mean_se([result.best_training_seconds for result in results])
    return CandidateSummary(
        recipe=recipe,
        mean_best_validation_accuracy=mean_accuracy,
        se_best_validation_accuracy=se_accuracy,
        mean_time_to_best=mean_time,
        stable=all(is_stable(result, tail_epochs, tolerance) for result in results),
    )


def select_recipe(
    candidate_results: Dict[Recipe, Sequence[TrainingResult]],
    tail_epochs: int = STABILITY_TAIL_EPOCHS,
    tolerance: float = STABILITY_TOLERANCE,
) -> CandidateSummary:
    summaries = tuple(
        summarize_candidate(recipe, tuple(results), tail_epochs, tolerance)
        for recipe, results in candidate_results.items()
    )
    selectable = [summary for summary in summaries if summary.stable]
    if not selectable:
        raise RuntimeError("no candidate is stable on every tuning seed")
    return min(
        selectable,
        key=lambda summary: (
            -summary.mean_best_validation_accuracy,
            summary.mean_time_to_best,
            summary.recipe.inference_steps,
            summary.recipe.peak_learning_rate,
            summary.recipe.decay_epochs,
        ),
    )


def stable_plateau_floor(
    results: Sequence[TrainingResult],
    tail_epochs: int,
    tolerance: float = STABILITY_TOLERANCE,
) -> float:
    if not results or any(
        not is_stable(result, tail_epochs, tolerance) for result in results
    ):
        raise ValueError("target requires stable selected results")
    return float(
        min(
            point.validation_accuracy
            for result in results
            for point in result.curve[-tail_epochs:]
        )
    )


def _run_recipe(recipe: Recipe, seed: int, args, evaluate_endpoint: bool):
    return _run_training(
        recipe=recipe,
        seed=seed,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        floor_ratio=args.floor_ratio,
        stability_tail_epochs=args.stability_tail_epochs,
        stability_tolerance=args.stability_tolerance,
        evaluate_endpoint=evaluate_endpoint,
    )


def run_tuning(args):
    epc_candidates = make_candidate_grid(
        "ePC",
        args.epc_steps_grid,
        args.epc_lr_grid,
        args.decay_epochs_grid,
        args.epc_eta,
    )
    spc_candidates = make_candidate_grid(
        "sPC",
        args.spc_steps_grid,
        args.spc_lr_grid,
        args.decay_epochs_grid,
        args.spc_eta,
    )
    by_solver = {"ePC": epc_candidates, "sPC": spc_candidates}
    all_results: Dict[Recipe, list] = {}
    tuning_cost = {"ePC": 0.0, "sPC": 0.0}

    print("PHASE tune", flush=True)
    print(
        f"SPLITS train={TRAIN_SPLIT} validation={VALIDATION_SPLIT} " "holdout=SEALED",
        flush=True,
    )
    print(
        f"DESIGN epochs={args.epochs} screen_seed={args.screen_seed} "
        f"confirm_seed={args.confirm_seed} shortlist={args.shortlist_size}",
        flush=True,
    )

    shortlists = {}
    for solver, candidates in by_solver.items():
        screen_rows = []
        for recipe in candidates:
            result = _run_recipe(recipe, args.screen_seed, args, False)
            all_results.setdefault(recipe, []).append(result)
            tuning_cost[solver] += result.total_training_seconds
            screen_rows.append(result)
        shortlists[solver] = shortlist_candidates(
            screen_rows,
            args.shortlist_size,
            args.stability_tail_epochs,
            args.stability_tolerance,
        )
        print(
            f"SHORTLIST solver={solver} "
            f"recipes={','.join(recipe.name for recipe in shortlists[solver])}",
            flush=True,
        )

    for solver, recipes in shortlists.items():
        for recipe in recipes:
            result = _run_recipe(recipe, args.confirm_seed, args, False)
            all_results[recipe].append(result)
            tuning_cost[solver] += result.total_training_seconds

    selected = {}
    for solver, recipes in shortlists.items():
        candidate_rows = {recipe: tuple(all_results[recipe]) for recipe in recipes}
        for recipe, rows in candidate_rows.items():
            summary = summarize_candidate(
                recipe,
                rows,
                args.stability_tail_epochs,
                args.stability_tolerance,
            )
            print(
                "CANDIDATE_SUMMARY "
                f"solver={solver} name={recipe.name} stable={summary.stable} "
                f"mean_best_validation_accuracy="
                f"{summary.mean_best_validation_accuracy:.8f} "
                f"se_best_validation_accuracy="
                f"{summary.se_best_validation_accuracy:.8f} "
                f"mean_time_to_best={summary.mean_time_to_best:.6f}",
                flush=True,
            )
        selected[solver] = select_recipe(
            candidate_rows,
            args.stability_tail_epochs,
            args.stability_tolerance,
        )

    selected_epc = selected["ePC"]
    selected_spc = selected["sPC"]
    spc_results = tuple(all_results[selected_spc.recipe])
    target = stable_plateau_floor(
        spc_results,
        args.stability_tail_epochs,
        args.stability_tolerance,
    )

    for solver, selection in selected.items():
        for result in all_results[selection.recipe]:
            crossing = time_to_accuracy(result, target)
            print(
                "TUNE_TARGET "
                f"solver={solver} name={selection.recipe.name} seed={result.seed} "
                f"target={target:.8f} time_to_target={crossing}",
                flush=True,
            )
        print(
            f"TUNING_COST solver={solver} "
            f"training_seconds={tuning_cost[solver]:.6f}",
            flush=True,
        )

    epc = selected_epc.recipe
    spc = selected_spc.recipe
    print(
        "LOCKED "
        f"epc_steps={epc.inference_steps} epc_lr={epc.peak_learning_rate:.8g} "
        f"epc_decay_epochs={epc.decay_epochs} "
        f"spc_steps={spc.inference_steps} spc_lr={spc.peak_learning_rate:.8g} "
        f"spc_decay_epochs={spc.decay_epochs} "
        f"target_accuracy={target:.8f}",
        flush=True,
    )


def _final_summary(results: Sequence[TrainingResult], target: float, margin: float):
    by_solver = {
        solver: [result for result in results if result.recipe.solver == solver]
        for solver in ("ePC", "sPC")
    }
    if any(not rows for rows in by_solver.values()):
        raise ValueError("final results require both solvers")
    epc_by_seed = {result.seed: result for result in by_solver["ePC"]}
    spc_by_seed = {result.seed: result for result in by_solver["sPC"]}
    if len(epc_by_seed) != len(by_solver["ePC"]) or len(spc_by_seed) != len(
        by_solver["sPC"]
    ):
        raise ValueError("final solver seeds must be unique")
    if set(epc_by_seed) != set(spc_by_seed):
        raise ValueError("final solver seeds must be paired")

    paired_seeds = sorted(epc_by_seed)
    epc_rows = [epc_by_seed[seed] for seed in paired_seeds]
    spc_rows = [spc_by_seed[seed] for seed in paired_seeds]
    if any(result.endpoint_accuracy is None for result in results):
        raise ValueError("every final result requires an endpoint accuracy")

    for solver, rows in by_solver.items():
        endpoint_mean, endpoint_se = _mean_se(
            [float(row.endpoint_accuracy) for row in rows]
        )
        best_mean, best_se = _mean_se([row.best_validation_accuracy for row in rows])
        best_time_mean, best_time_se = _mean_se(
            [row.best_training_seconds for row in rows]
        )
        print(
            "FINAL_SUMMARY "
            f"solver={solver} n={len(rows)} "
            f"mean_best_validation_accuracy={best_mean:.8f} "
            f"se_best_validation_accuracy={best_se:.8f} "
            f"mean_endpoint_accuracy={endpoint_mean:.8f} "
            f"se_endpoint_accuracy={endpoint_se:.8f} "
            f"mean_time_to_best={best_time_mean:.6f} "
            f"se_time_to_best={best_time_se:.6f}",
            flush=True,
        )

    epc_endpoint = np.asarray(
        [float(result.endpoint_accuracy) for result in epc_rows], dtype=np.float64
    )
    spc_endpoint = np.asarray(
        [float(result.endpoint_accuracy) for result in spc_rows], dtype=np.float64
    )
    endpoint_test = paired_ttest(epc_endpoint, spc_endpoint)
    endpoint_effect = cohens_d(epc_endpoint, spc_endpoint)
    endpoint_difference = float(np.mean(epc_endpoint - spc_endpoint))
    noninferior = endpoint_difference + margin >= -1e-12

    epc_target_times = [time_to_accuracy(result, target) for result in epc_rows]
    spc_target_times = [time_to_accuracy(result, target) for result in spc_rows]
    all_reached = all(
        value is not None for value in epc_target_times + spc_target_times
    )
    faster = False
    time_ratio = None
    if all_reached:
        mean_epc_time = float(np.mean(epc_target_times))
        mean_spc_time = float(np.mean(spc_target_times))
        time_ratio = mean_epc_time / mean_spc_time
        faster = mean_epc_time < mean_spc_time

    print(
        "PAIRED_ENDPOINT "
        f"mean_difference={endpoint_difference:.8f} "
        f"t={endpoint_test.t_statistic:.8f} p={endpoint_test.p_value:.8f} "
        f"cohens_d={endpoint_effect.d:.8f} margin={margin:.8f} "
        f"noninferior={noninferior}",
        flush=True,
    )
    print(
        "PERFORMANCE_GATE "
        f"target={target:.8f} all_reached={all_reached} faster={faster} "
        f"time_ratio={time_ratio} noninferior={noninferior} "
        f"overall_pass={all_reached and faster and noninferior}",
        flush=True,
    )


def run_final(args):
    epc_recipe = Recipe(
        "ePC", args.epc_steps, args.epc_eta, args.epc_lr, args.epc_decay_epochs
    )
    spc_recipe = Recipe(
        "sPC", args.spc_steps, args.spc_eta, args.spc_lr, args.spc_decay_epochs
    )
    print("PHASE final", flush=True)
    print(
        "LOCKED "
        f"epc_steps={epc_recipe.inference_steps} "
        f"epc_lr={epc_recipe.peak_learning_rate:.8g} "
        f"epc_decay_epochs={epc_recipe.decay_epochs} "
        f"spc_steps={spc_recipe.inference_steps} "
        f"spc_lr={spc_recipe.peak_learning_rate:.8g} "
        f"spc_decay_epochs={spc_recipe.decay_epochs} "
        f"target_accuracy={args.target_accuracy:.8f}",
        flush=True,
    )
    print(
        f"SPLITS train={TRAIN_SPLIT} validation={VALIDATION_SPLIT} "
        f"holdout={HOLDOUT_SPLIT}",
        flush=True,
    )

    results = []
    for seed in args.final_seeds:
        for recipe in (epc_recipe, spc_recipe):
            result = _run_recipe(recipe, seed, args, True)
            results.append(result)
            print(
                "FINAL_RESULT "
                f"solver={recipe.solver} name={recipe.name} seed={seed} "
                f"best_epoch={result.best_epoch} "
                f"time_to_best={result.best_training_seconds:.6f} "
                f"best_validation_accuracy={result.best_validation_accuracy:.8f} "
                f"time_to_target={time_to_accuracy(result, args.target_accuracy)} "
                f"endpoint_accuracy={result.endpoint_accuracy}",
                flush=True,
            )
    _final_summary(results, args.target_accuracy, args.noninferiority_margin)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tune", "final"), required=True)
    parser.add_argument("--epochs", type=int, default=OBSERVATION_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=float, default=WARMUP_EPOCHS)
    parser.add_argument("--floor_ratio", type=float, default=FLOOR_RATIO)
    parser.add_argument("--epc_eta", type=float, default=0.1)
    parser.add_argument("--spc_eta", type=float, default=0.1)
    parser.add_argument(
        "--epc_steps_grid", type=_parse_csv_positive_ints, default=EPC_STEP_GRID
    )
    parser.add_argument(
        "--epc_lr_grid", type=_parse_csv_positive_floats, default=EPC_LR_GRID
    )
    parser.add_argument(
        "--spc_steps_grid", type=_parse_csv_positive_ints, default=SPC_STEP_GRID
    )
    parser.add_argument(
        "--spc_lr_grid", type=_parse_csv_positive_floats, default=SPC_LR_GRID
    )
    parser.add_argument(
        "--decay_epochs_grid",
        type=_parse_csv_positive_ints,
        default=DECAY_EPOCH_GRID,
    )
    parser.add_argument("--screen_seed", type=int, default=SCREEN_SEED)
    parser.add_argument("--confirm_seed", type=int, default=CONFIRM_SEED)
    parser.add_argument("--shortlist_size", type=int, default=SHORTLIST_SIZE)
    parser.add_argument(
        "--stability_tail_epochs", type=int, default=STABILITY_TAIL_EPOCHS
    )
    parser.add_argument(
        "--stability_tolerance", type=float, default=STABILITY_TOLERANCE
    )
    parser.add_argument("--epc_steps", type=int)
    parser.add_argument("--epc_lr", type=float)
    parser.add_argument("--epc_decay_epochs", type=int)
    parser.add_argument("--spc_steps", type=int)
    parser.add_argument("--spc_lr", type=float)
    parser.add_argument("--spc_decay_epochs", type=int)
    parser.add_argument("--target_accuracy", type=float)
    parser.add_argument(
        "--noninferiority_margin", type=float, default=NONINFERIORITY_MARGIN
    )
    parser.add_argument(
        "--final_seeds",
        type=_parse_csv_nonnegative_ints,
        default=FINAL_SEEDS,
    )
    args = parser.parse_args()

    positive_ints = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "shortlist_size": args.shortlist_size,
        "stability_tail_epochs": args.stability_tail_epochs,
    }
    for name, value in positive_ints.items():
        if value < 1:
            parser.error(f"--{name} must be >= 1")
    positive_floats = {
        "warmup_epochs": args.warmup_epochs,
        "floor_ratio": args.floor_ratio,
        "epc_eta": args.epc_eta,
        "spc_eta": args.spc_eta,
        "stability_tolerance": args.stability_tolerance,
    }
    for name, value in positive_floats.items():
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and > 0")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--weight_decay must be finite and >= 0")
    if args.screen_seed < 0 or args.confirm_seed < 0:
        parser.error("tuning seeds must be nonnegative")
    if args.screen_seed == args.confirm_seed:
        parser.error("screen and confirmation seeds must differ")
    if args.stability_tail_epochs > args.epochs:
        parser.error("stability tail cannot exceed observation epochs")
    if not 0 < args.floor_ratio <= 1:
        parser.error("--floor_ratio must be <= 1")
    if not 0 <= args.noninferiority_margin <= 1:
        parser.error("--noninferiority_margin must be between 0 and 1")
    if len(args.final_seeds) < 2:
        parser.error("at least two final seeds are required")
    if set(args.final_seeds) & {args.screen_seed, args.confirm_seed}:
        parser.error("final seeds must be disjoint from tuning seeds")

    if args.mode == "final":
        required_positive = {
            "epc_steps": args.epc_steps,
            "epc_lr": args.epc_lr,
            "epc_decay_epochs": args.epc_decay_epochs,
            "spc_steps": args.spc_steps,
            "spc_lr": args.spc_lr,
            "spc_decay_epochs": args.spc_decay_epochs,
            "target_accuracy": args.target_accuracy,
        }
        for name, value in required_positive.items():
            if value is None or not np.isfinite(value) or value <= 0:
                parser.error(f"--{name} must be supplied and > 0 in final mode")
        if args.target_accuracy > 1:
            parser.error("--target_accuracy must be <= 1")
    return args


def main():
    args = parse_args()
    if args.mode == "tune":
        run_tuning(args)
    else:
        run_final(args)


if __name__ == "__main__":
    main()
