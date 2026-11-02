"""Run one trial of one benchmark row and save the result as JSON."""

import itertools
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import jax

from fabricpc.bench.run.compute import count_compute
from fabricpc.bench.results.manifest import SCHEMA_VERSION
from fabricpc.bench.run.diagnostics import diagnose
from fabricpc.bench.run.measure import epc_regime, memory_snapshot, time_steps
from fabricpc.bench.definitions.registry import BenchmarkRow
from fabricpc.bench.run.zoo import save_params
from fabricpc.training import InferenceRateController, evaluate, train
from fabricpc.training.trainer import build_clamps, convert_batch

# The trainer knows two algorithms. Both PC solvers use "pc"; the solver
# itself lives inside the graph the row's model factory built.
_TRAINER_ALGORITHM = {"spc": "pc", "epc": "pc", "backprop": "backprop"}

# Any fixed number works; it just keeps the timing key apart from the three
# keys the experiment framework uses.
_TIMING_STREAM = 7


def total_train_steps(train_loader, num_epochs: float) -> int:
    """Optimizer updates in a run: batches per epoch times epochs, rounded up."""
    return math.ceil(len(train_loader) * float(num_epochs))


DEFAULT_CURVE_BATCHES = 10


class _BatchList:
    """A fixed list of batches that iterates the same way every time."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class _Curve:
    """Scores the model on a small fixed slice of the test set after every
    epoch, so a run that learns and then falls apart shows it as it happens
    (the final score still uses the whole test set)."""

    def __init__(
        self, structure, test_loader, config, key, algorithm, n_batches, metrics=None
    ):
        self.enabled = n_batches > 0
        self.points: List[Dict[str, float]] = []
        self.seconds = 0.0
        self._metrics = metrics
        if self.enabled:
            self._args = (structure, config, key, algorithm)
            self._batches = _BatchList(itertools.islice(test_loader, n_batches))

    def on_epoch(self, ctx):
        t0 = time.perf_counter()
        _, config, key, algorithm = self._args
        # ctx.structure, not the one training started with: a followed ePC
        # rate changes the structure during the run.
        raw = evaluate(
            ctx.params,
            ctx.structure,
            self._batches,
            config,
            key,
            algorithm=algorithm,
            metrics=self._metrics,
        )
        point = {"epoch": int(ctx.epoch_idx) + 1}
        point.update({k: float(v) for k, v in raw.items()})
        point["train_energy"] = float(ctx.metrics.get("energy", float("nan")))
        self.points.append(point)
        self.seconds += time.perf_counter() - t0
        return None


FORWARD_PREFIX = "forward_"


def _forward_metrics(
    params, structure, test_loader, config, key, algorithm, metrics, scores=None
):
    """The trained model scored with one plain forward pass, as forward_<name>.

    PC evaluation leaves the output free and lets it keep settling. With a
    cross-entropy output that pushes the outputs to be over-confident, so a
    PC row's loss and perplexity can look worse than the weights really are.
    The forward pass is how the model is used after training, and it scores
    every method the same way. For backprop the normal evaluation already is
    the forward pass, so its numbers are just copied.
    """
    if algorithm == "backprop":
        return {FORWARD_PREFIX + k: v for k, v in metrics.items()}
    raw = evaluate(
        params,
        structure,
        test_loader,
        config,
        key,
        algorithm="backprop",
        metrics=scores,
    )
    return {FORWARD_PREFIX + k: float(v) for k, v in raw.items()}


def seed_for_trial(trial: int, seed_offset: int = 0) -> int:
    """Same rule as PlannedMultiContrastExperiment, so arms line up."""
    return seed_offset + trial * 1000


@dataclass(frozen=True)
class TrialResult:
    row_id: str
    trial: int
    seed: int
    algorithm: str
    n_params: int
    metrics: Dict[str, float] = field(default_factory=dict)
    compile_time_s: float = 0.0
    step_time_ms: float = 0.0
    train_time_s: float = 0.0
    memory_bytes: Optional[int] = None  # held after warmup
    # Most held by the end of the trial, process-wide. On GPU this includes
    # compile scratch space and can be the same for every algorithm; see
    # step_memory for the compiled training step's own needs.
    peak_memory_bytes: Optional[int] = None
    step_memory: Optional[Dict[str, int]] = None
    # Test metrics and mean training energy after every epoch, on the first
    # ``curve_batches`` test batches (None when turned off).
    curve: Optional[List[Dict[str, float]]] = None
    # Backprop alignment and stiffness at init and after training
    # (fabricpc.bench.run.diagnostics); None when turned off.
    diagnostics: Optional[Dict[str, Dict[str, object]]] = None
    # ePC rows: EPCInference.regime at init and after training (else None)
    epc_regime: Optional[Dict[str, Dict[str, object]]] = None
    # Rows with rate_control: the controller's summary and every probe
    rate_control: Optional[Dict[str, object]] = None
    num_epochs: float = 0.0
    compute: Dict[str, float] = field(default_factory=dict)
    achieved_tflops: float = 0.0  # flops per update / measured step time
    checkpoint: Optional[str] = None  # zoo path of the trained params
    status: str = "ok"
    error: Optional[str] = None
    schema_version: int = SCHEMA_VERSION


def run_trial(
    row: BenchmarkRow,
    trial: int,
    out_dir,
    *,
    loaders=None,
    num_epochs: Optional[float] = None,
    warmup_steps: int = 5,
    timed_steps: int = 30,
    seed_offset: int = 0,
    zoo_dir=None,
    curve_batches: int = DEFAULT_CURVE_BATCHES,
    diagnostics: bool = True,
) -> TrialResult:
    """Train the row's model once, measure it, and write ``trial<i>.json``.

    ``loaders`` lets a test pass a small fake dataset. ``num_epochs`` lets a
    caller shorten a run. Both default to what the row says. ``zoo_dir``
    saves the trained params there; None skips saving.
    """
    seed = seed_for_trial(trial, seed_offset)
    out_path = Path(out_dir) / row.id / f"trial{trial}.json"
    algorithm = _TRAINER_ALGORITHM[row.algorithm]
    epochs = float(
        num_epochs if num_epochs is not None else row.train_config["num_epochs"]
    )

    try:
        # Split the seed exactly as PlannedMultiContrastExperiment does, so a
        # trial here trains the same model the experiment framework would.
        # The timing key is drawn off to the side and never touches training.
        master_key = jax.random.PRNGKey(seed)
        graph_key, train_key, eval_key = jax.random.split(master_key, 3)
        timing_key = jax.random.fold_in(master_key, _TIMING_STREAM)

        params, structure = row.model_factory(graph_key)
        n_params = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))
        # Timing reads a few batches, so training gets fresh loaders after it
        # (the framework also builds fresh loaders for every arm).
        timing_loader, _ = loaders or row.loader_factory(seed)
        optimizer = row.optimizer_factory(total_train_steps(timing_loader, epochs))
        timing = time_steps(
            params,
            structure,
            optimizer,
            timing_loader,
            timing_key,
            algorithm=algorithm,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
        )
        memory = memory_snapshot().bytes_in_use
        probe_batch = next(iter(timing_loader))
        regime_at_init = epc_regime(params, structure, probe_batch, timing_key)
        diag_init = (
            diagnose(params, structure, probe_batch, timing_key, row.algorithm)
            if diagnostics
            else None
        )
        train_loader, test_loader = loaders or row.loader_factory(seed)

        compute = count_compute(
            params, structure, batch_size=row.batch_size, algorithm=row.algorithm
        )
        # tflops = flops per update / seconds per update / 1e12
        achieved_tflops = (
            compute.flops_per_update / (timing.step_time_ms / 1000.0) / 1e12
        )

        config = {**row.train_config, "num_epochs": epochs}
        t0 = time.perf_counter()
        controller = None
        train_structure = structure
        if row.rate_control is not None:
            # The probe batch is the same fixed batch the regime is read on.
            controller = InferenceRateController(
                structure,
                build_clamps(convert_batch(probe_batch), structure, clamp_target=True),
                target=row.rate_control.target,
                every=row.rate_control.every,
                window=row.rate_control.window,
                key=timing_key,
            )
            train_structure = controller.start(params)
        curve = _Curve(
            train_structure,
            test_loader,
            config,
            eval_key,
            algorithm,
            curve_batches,
            metrics=row.metrics_for(algorithm),
        )
        trained = train(
            params,
            train_structure,
            train_loader,
            optimizer,
            config,
            train_key,
            algorithm=algorithm,
            verbose=False,
            epoch_callback=curve.on_epoch if curve.enabled else None,
            structure_callback=controller.on_iter if controller else None,
        )
        # The curve's evaluations are not training time. The rate probes
        # are: a followed rate costs them on every run.
        train_time_s = time.perf_counter() - t0 - curve.seconds
        if controller is not None:
            # Evaluate and read the regime at the rate the run ended on.
            structure = controller.structure

        raw = evaluate(
            trained.params,
            structure,
            test_loader,
            config,
            eval_key,
            algorithm=algorithm,
            metrics=row.metrics_for(algorithm),
        )
        metrics = {k: float(v) for k, v in raw.items()}
        metrics.update(
            _forward_metrics(
                trained.params,
                structure,
                test_loader,
                config,
                eval_key,
                algorithm,
                metrics,
                scores=row.metrics_for("backprop"),
            )
        )
        peak_memory = memory_snapshot().peak_bytes
        regime = None
        if regime_at_init is not None:
            regime = {
                "init": regime_at_init,
                "final": epc_regime(trained.params, structure, probe_batch, timing_key),
            }

        checkpoint = None
        if zoo_dir is not None:
            path = save_params(
                zoo_dir,
                row,
                trial,
                trained.params,
                meta={"seed": seed, "num_epochs": epochs, "metrics": metrics},
            )
            # Relative to the results folder, so it still points at the
            # weights after the folder is copied somewhere else.
            checkpoint = os.path.relpath(path, out_dir)

        result = TrialResult(
            row_id=row.id,
            trial=trial,
            seed=seed,
            algorithm=row.algorithm,
            n_params=n_params,
            metrics=metrics,
            compile_time_s=timing.compile_time_s,
            step_time_ms=timing.step_time_ms,
            train_time_s=train_time_s,
            memory_bytes=memory,
            peak_memory_bytes=peak_memory,
            step_memory=timing.step_memory,
            epc_regime=regime,
            diagnostics=(
                None
                if diag_init is None
                else {
                    "init": diag_init,
                    "final": diagnose(
                        trained.params,
                        structure,
                        probe_batch,
                        timing_key,
                        row.algorithm,
                    ),
                }
            ),
            rate_control=(
                None
                if controller is None
                else {"summary": controller.summary(), "history": controller.history}
            ),
            curve=curve.points if curve.enabled else None,
            num_epochs=epochs,
            compute=asdict(compute),
            achieved_tflops=achieved_tflops,
            checkpoint=checkpoint,
        )
    except Exception:  # noqa: BLE001 - a failed trial must still be recorded
        result = TrialResult(
            row_id=row.id,
            trial=trial,
            seed=seed,
            algorithm=row.algorithm,
            n_params=0,
            num_epochs=epochs,
            status="failed",
            error=traceback.format_exc(),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2))
    return result


def finished_trial(out_dir, row_id: str, trial: int, num_epochs: float) -> bool:
    """True when ``trial<i>.json`` holds a good result for this epoch count.

    Used by ``--resume``. A failed trial, a missing or broken file, or one
    trained for a different number of epochs does not count as finished.
    """
    path = Path(out_dir) / row_id / f"trial{trial}.json"
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return saved.get("status") == "ok" and saved.get("num_epochs") == float(num_epochs)
