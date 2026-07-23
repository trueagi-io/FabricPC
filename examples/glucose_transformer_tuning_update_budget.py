"""Archived update-budget Optuna tuner for the FabricPC glucose transformer.

This is the Livia phase 1-4 implementation that budgets trials by optimizer
updates (pilot/full updates, MedianPruner, search spaces
``phase1|refined|local|breakthrough``). It produced the phase-4 breakthrough
champion (~19.88 Optuna val MAE; epoch confirm ~18.78 val / 18.20 test).

The **active** default tuner is ``examples/glucose_transformer_tuning.py``
(Anton epoch-based Hyperband study). Keep this file so those configs and
CLI knobs can be reinstated without digging through git history.

Run:

```bash
uv run glucose-transformer-tune-update-budget run \
  --run-dir runs/glucose_tuning_pc_breakthrough \
  --study-name glucose_transformer_pc_breakthrough \
  --search-space breakthrough \
  --n-trials 24 --max-workers 2 \
  --full-updates 10000 --pilot-updates 800
```

See ``docs/GLUCOSE.md`` section "Two Optuna implementations".
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import optuna
import typer
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

app = typer.Typer(
    help=(
        "Tune FabricPC glucose predictive-coding (PC) dynamics only. "
        "Does not run backpropagation."
    )
)

DEFAULT_RUN_DIR = Path("runs/glucose_tuning")
DEFAULT_STUDY_NAME = "glucose_transformer_pc"
DEFAULT_SEARCH_SPACE = "refined"
TERMINAL_STATES = {
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.PRUNED,
    optuna.trial.TrialState.FAIL,
}


@dataclass(frozen=True)
class TrialSettings:
    """Fixed geometry and resource budget shared by all trials."""

    run_dir: Path
    seq_len: int = 128
    horizon: int = 12
    depth: int = 3
    embed_dim: int = 32
    num_heads: int = 4
    mlp_dim: int = 128
    batch_size: int = 64
    pilot_updates: int = 600
    full_updates: int = 6000
    validation_interval: int = 200
    seed: int = 42
    search_space: str = DEFAULT_SEARCH_SPACE
    # PC-only study: workers call fabricpc.training.train.train_step (not backprop).
    training_mode: str = "pc"


@dataclass
class WorkerProcess:
    """A managed trial worker and its output stream."""

    process: subprocess.Popen[bytes]
    log_file: Any
    log_path: Path
    started_at: float


def create_storage(journal_path: Path) -> JournalStorage:
    """Create resumable, multi-process-safe local Optuna storage.

    On Windows the default symlink lock requires elevated privileges, so use
    ``JournalFileOpenLock`` there. Elsewhere keep Optuna's default lock.
    """
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    path = str(journal_path)
    if sys.platform == "win32":
        backend = JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
    else:
        backend = JournalFileBackend(path)
    return JournalStorage(backend)


def create_study(
    journal_path: Path,
    study_name: str,
    *,
    n_startup_trials: int = 16,
    search_space: str = DEFAULT_SEARCH_SPACE,
) -> optuna.Study:
    """Create or load the shared PC dynamics study."""
    if search_space == "breakthrough":
        # Softer pruner: prior SH killed slow starters in the good η band.
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )
        n_startup_trials = min(n_startup_trials, 6)
    else:
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    return optuna.create_study(
        study_name=study_name,
        storage=create_storage(journal_path),
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=n_startup_trials,
            multivariate=True,
            group=True,
            constant_liar=True,
        ),
        pruner=pruner,
    )


PHASE1_WINNER_PARAMS: dict[str, float | int] = {
    "seq_len": 64,
    "depth": 1,
    "num_heads": 1,
    "lr": 0.002974957684513748,
    "eta_infer": 1.2809872060487701e-05,
    "infer_steps": 14,
    "max_infer_norm": 1.0,
    "grad_clip": 0.5,
    "weight_init_std": 0.016838494278391843,
}


def enqueue_local_baselines(study: optuna.Study) -> None:
    """Seed a fresh local study with the phase-1 winner and nearby PC variants."""
    if study.get_trials(deepcopy=False):
        return
    baselines: list[dict[str, float | int]] = [
        dict(PHASE1_WINNER_PARAMS),
        {
            **PHASE1_WINNER_PARAMS,
            "lr": 0.0034,
            "eta_infer": 1.1e-05,
            "infer_steps": 16,
            "grad_clip": 0.5,
        },
        {
            **PHASE1_WINNER_PARAMS,
            "lr": 0.0025,
            "eta_infer": 1.6e-05,
            "infer_steps": 12,
            "max_infer_norm": 0.5,
        },
        {
            **PHASE1_WINNER_PARAMS,
            "lr": 0.0032,
            "eta_infer": 9.5e-06,
            "infer_steps": 15,
            "grad_clip": 1.0,
            "weight_init_std": 0.018,
        },
    ]
    for params in baselines:
        study.enqueue_trial(params)


def enqueue_breakthrough_baselines(study: optuna.Study) -> None:
    """Seed phase-4 with champion params, winning seed, and lighter readouts."""
    if study.get_trials(deepcopy=False):
        return
    # Phase-1 Optuna trial 21 used seed = 42 + 21.
    champion_seed = 21
    baselines: list[dict[str, float | int]] = [
        {
            **PHASE1_WINNER_PARAMS,
            "readout": "flatten",
            "seed_offset": champion_seed,
        },
        {
            **PHASE1_WINNER_PARAMS,
            "readout": "mean_pool",
            "seed_offset": champion_seed,
        },
        {
            **PHASE1_WINNER_PARAMS,
            "readout": "last",
            "seed_offset": champion_seed,
        },
        {
            **PHASE1_WINNER_PARAMS,
            "lr": 0.0032,
            "eta_infer": 1.15e-05,
            "infer_steps": 16,
            "readout": "mean_pool",
            "seed_offset": champion_seed,
        },
        {
            **PHASE1_WINNER_PARAMS,
            "lr": 0.0026,
            "eta_infer": 1.5e-05,
            "infer_steps": 14,
            "readout": "mean_pool",
            "seed_offset": 7,
        },
        {
            **PHASE1_WINNER_PARAMS,
            "lr": 0.0035,
            "eta_infer": 1.05e-05,
            "infer_steps": 15,
            "grad_clip": 0.5,
            "readout": "last",
            "seed_offset": 15,
        },
    ]
    for params in baselines:
        study.enqueue_trial(params)


def suggest_pc_dynamics_phase1(trial: optuna.Trial) -> dict[str, float | int]:
    """Broad PC search used for the first Optuna study (best MAE ~20.38)."""
    return {
        "seq_len": trial.suggest_categorical("seq_len", [64, 128]),
        "depth": trial.suggest_int("depth", 1, 3),
        "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4]),
        "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 1e-5, 5e-4, log=True),
        "infer_steps": trial.suggest_int("infer_steps", 8, 24),
        "max_infer_norm": trial.suggest_categorical(
            "max_infer_norm", [0.5, 1.0, 5.0]
        ),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0]),
        "weight_init_std": trial.suggest_float(
            "weight_init_std", 0.01, 0.03, log=True
        ),
        "weight_decay": 0.0,
        "readout": "flatten",
    }


def suggest_pc_dynamics_refined(trial: optuna.Trial) -> dict[str, float | int]:
    """Focused PC search to beat phase-1 best (~20.38 val MAE).

    Changes vs phase1 (documented for result tracking):
    - Fix ``seq_len=64`` (128 was rarely competitive).
    - Drop depth 3 and head count 2 (weak / unstable in phase1).
    - Lower ``eta_infer`` floor below 1e-5 (winners sat on the old floor).
    - Drop aggressive ``max_infer_norm=5`` and ``grad_clip=2``.
    - Narrow LR / init around the winning band.
    - Add light AdamW ``weight_decay`` for generalization.
    """
    return {
        # Fixed to the phase-1 winning context; kept as a suggest so params log it.
        "seq_len": trial.suggest_categorical("seq_len", [64]),
        "depth": trial.suggest_int("depth", 1, 2),
        "num_heads": trial.suggest_categorical("num_heads", [1, 4]),
        "lr": trial.suggest_float("lr", 1e-3, 5e-3, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 3e-6, 8e-5, log=True),
        "infer_steps": trial.suggest_int("infer_steps", 10, 20),
        "max_infer_norm": trial.suggest_categorical(
            "max_infer_norm", [0.5, 1.0]
        ),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0]),
        "weight_init_std": trial.suggest_float(
            "weight_init_std", 0.012, 0.025, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        ),
        "readout": "flatten",
    }


def suggest_pc_dynamics_local(trial: optuna.Trial) -> dict[str, float | int]:
    """Tight PC neighborhood around phase-1 winner (trial 21, MAE ~20.38).

    Uses the same geometry (64 / depth1 / heads1) and keeps ``eta_infer`` near
    1e-5–2e-5. Optimizer stays Adam (``weight_decay=0``) so gains are from
    dynamics/init, not an AdamW protocol change.
    """
    return {
        "seq_len": trial.suggest_categorical("seq_len", [64]),
        "depth": trial.suggest_categorical("depth", [1]),
        "num_heads": trial.suggest_categorical("num_heads", [1]),
        "lr": trial.suggest_float("lr", 1.5e-3, 4.0e-3, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 8e-6, 3e-5, log=True),
        "infer_steps": trial.suggest_int("infer_steps", 12, 18),
        "max_infer_norm": trial.suggest_categorical(
            "max_infer_norm", [0.5, 1.0]
        ),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0]),
        "weight_init_std": trial.suggest_float(
            "weight_init_std", 0.014, 0.022, log=True
        ),
        "weight_decay": 0.0,
        "readout": "flatten",
    }


def suggest_pc_dynamics_breakthrough(trial: optuna.Trial) -> dict[str, float | int]:
    """Phase-4 PC search built from prior HPO conclusions to beat 20.38.

    Documented changes vs saturated dynamics-only searches:
    - Lock geometry ``64/d1/h1`` and η to the proven band (~9e-6–2.5e-5).
    - Search lighter readouts (``mean_pool`` / ``last``) to ease PC inference.
    - Search ``seed_offset`` (phase-3 showed large seed sensitivity).
    - Pair with longer update budget + MedianPruner + milder early-stop
      (configured by the coordinator / trial loop for this space).
    """
    return {
        "seq_len": trial.suggest_categorical("seq_len", [64]),
        "depth": trial.suggest_categorical("depth", [1]),
        "num_heads": trial.suggest_categorical("num_heads", [1]),
        "lr": trial.suggest_float("lr", 1.8e-3, 3.8e-3, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 9e-6, 2.5e-5, log=True),
        "infer_steps": trial.suggest_int("infer_steps", 12, 18),
        "max_infer_norm": trial.suggest_categorical(
            "max_infer_norm", [0.5, 1.0]
        ),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0]),
        "weight_init_std": trial.suggest_float(
            "weight_init_std", 0.014, 0.021, log=True
        ),
        "weight_decay": 0.0,
        "readout": trial.suggest_categorical(
            "readout", ["flatten", "mean_pool", "last"]
        ),
        "seed_offset": trial.suggest_int("seed_offset", 0, 40),
    }


def suggest_pc_dynamics(
    trial: optuna.Trial, search_space: str = DEFAULT_SEARCH_SPACE
) -> dict[str, float | int]:
    """Dispatch PC-only hyperparameter suggestions (never backprop)."""
    if search_space == "phase1":
        return suggest_pc_dynamics_phase1(trial)
    if search_space == "refined":
        return suggest_pc_dynamics_refined(trial)
    if search_space == "local":
        return suggest_pc_dynamics_local(trial)
    if search_space == "breakthrough":
        return suggest_pc_dynamics_breakthrough(trial)
    raise ValueError(
        f"Unknown search space {search_space!r}; "
        "use 'phase1', 'refined', 'local', or 'breakthrough'"
    )


def query_cuda_process_memory() -> dict[int, int]:
    """Return GPU memory in MiB keyed by compute-process PID."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return {}
    allocations: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            allocations[int(fields[0])] = int(fields[1])
        except ValueError:
            continue
    return allocations


def admitted_worker_count(
    *,
    allocations: dict[int, int],
    managed_pids: set[int],
    active_workers: int,
    gpu_memory_budget_mib: int,
    estimated_trial_memory_mib: int,
    max_workers: int,
) -> int:
    """Compute additional workers admitted by the configured memory budget."""
    external_memory = sum(
        memory for pid, memory in allocations.items() if pid not in managed_pids
    )
    observed_worker_memory = max(
        (allocations[pid] for pid in managed_pids if pid in allocations),
        default=0,
    )
    reservation_per_worker = max(
        estimated_trial_memory_mib,
        observed_worker_memory,
    )
    usable_memory = max(0, gpu_memory_budget_mib - external_memory)
    memory_capacity = usable_memory // max(reservation_per_worker, 1)
    total_capacity = min(max_workers, memory_capacity)
    return max(0, total_capacity - active_workers)


def _study_finished_trials(study: optuna.Study) -> int:
    return sum(trial.state in TERMINAL_STATES for trial in study.get_trials())


def _worker_command(
    *,
    journal_path: Path,
    study_name: str,
    settings: TrialSettings,
    trial_timeout_seconds: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "examples.glucose_transformer_tuning_update_budget",
        "worker",
        "--journal",
        str(journal_path),
        "--study-name",
        study_name,
        "--run-dir",
        str(settings.run_dir),
        "--seq-len",
        str(settings.seq_len),
        "--horizon",
        str(settings.horizon),
        "--depth",
        str(settings.depth),
        "--embed-dim",
        str(settings.embed_dim),
        "--num-heads",
        str(settings.num_heads),
        "--mlp-dim",
        str(settings.mlp_dim),
        "--batch-size",
        str(settings.batch_size),
        "--pilot-updates",
        str(settings.pilot_updates),
        "--full-updates",
        str(settings.full_updates),
        "--validation-interval",
        str(settings.validation_interval),
        "--seed",
        str(settings.seed),
        "--search-space",
        settings.search_space,
        "--trial-timeout-seconds",
        str(trial_timeout_seconds),
    ]


def _spawn_worker(
    *,
    worker_index: int,
    journal_path: Path,
    study_name: str,
    settings: TrialSettings,
    trial_timeout_seconds: int,
) -> WorkerProcess:
    workers_dir = settings.run_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    log_path = workers_dir / f"worker_{int(time.time())}_{worker_index}.log"
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        _worker_command(
            journal_path=journal_path,
            study_name=study_name,
            settings=settings,
            trial_timeout_seconds=trial_timeout_seconds,
        ),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return WorkerProcess(
        process=process,
        log_file=log_file,
        log_path=log_path,
        started_at=time.time(),
    )


def _stop_worker(worker: WorkerProcess, timeout_seconds: float = 10.0) -> None:
    """Terminate exactly one worker process (group on POSIX)."""
    if worker.process.poll() is None:
        if sys.platform == "win32":
            worker.process.terminate()
        else:
            os.killpg(worker.process.pid, signal.SIGTERM)
        try:
            worker.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                worker.process.kill()
            else:
                os.killpg(worker.process.pid, signal.SIGKILL)
            worker.process.wait()
    worker.log_file.close()


def _write_coordinator_state(
    path: Path,
    *,
    study: optuna.Study,
    workers: dict[int, WorkerProcess],
    allocations: dict[int, int],
) -> None:
    payload = {
        "updated_at": time.time(),
        "finished_trials": _study_finished_trials(study),
        "active_workers": [
            {
                "pid": pid,
                "log": str(worker.log_path),
                "gpu_memory_mib": allocations.get(pid, 0),
            }
            for pid, worker in workers.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def _fail_running_trial_for_pid(study: optuna.Study, pid: int) -> None:
    """Mark the trial claimed by a dead worker as failed."""
    for trial in study.get_trials(deepcopy=False):
        if (
            trial.state == optuna.trial.TrialState.RUNNING
            and trial.user_attrs.get("worker_pid") == pid
        ):
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            return


@app.command("run")
def run_coordinator(
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, help="Study and trial artifacts."),
    study_name: str = typer.Option(DEFAULT_STUDY_NAME, help="Optuna study name."),
    search_space: str = typer.Option(
        DEFAULT_SEARCH_SPACE,
        help=(
            "PC search space: phase1|refined|local|breakthrough "
            "(breakthrough = longer budget + readout/seed search)."
        ),
    ),
    n_trials: int = typer.Option(32, min=1, help="Total terminal trials desired."),
    max_workers: int = typer.Option(8, min=1, help="Maximum parallel workers."),
    gpu_memory_budget_mib: int = typer.Option(
        8192, min=1, help="Total GPU memory budget, including unrelated jobs."
    ),
    estimated_trial_memory_mib: int = typer.Option(
        900, min=1, help="Admission reservation per worker."
    ),
    batch_size: int = typer.Option(64, min=1),
    pilot_updates: int = typer.Option(600, min=3),
    full_updates: int = typer.Option(6000, min=3),
    validation_interval: int = typer.Option(200, min=1),
    poll_seconds: float = typer.Option(2.0, min=0.1),
    trial_timeout_seconds: int = typer.Option(
        7200, min=60, help="Hard wall-clock bound for one trial worker."
    ),
) -> None:
    """Coordinate process-isolated PC trials under a CUDA memory budget.

    This coordinator never launches backprop. Every worker uses PC ``train_step``.
    """
    allowed_spaces = {"phase1", "refined", "local", "breakthrough"}
    if search_space not in allowed_spaces:
        raise typer.BadParameter(
            "search-space must be one of: " + ", ".join(sorted(allowed_spaces))
        )
    if pilot_updates > full_updates:
        raise typer.BadParameter("pilot-updates cannot exceed full-updates")
    if pilot_updates < 3 * validation_interval:
        raise typer.BadParameter(
            "pilot-updates must include at least three validation intervals"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    journal_path = run_dir / "optuna_journal.log"
    if search_space == "breakthrough":
        startup = 6
    elif search_space == "local":
        startup = 8
    else:
        startup = 16
    study = create_study(
        journal_path,
        study_name,
        n_startup_trials=startup,
        search_space=search_space,
    )
    if search_space == "local":
        enqueue_local_baselines(study)
    elif search_space == "breakthrough":
        enqueue_breakthrough_baselines(study)
    settings = TrialSettings(
        run_dir=run_dir,
        batch_size=batch_size,
        pilot_updates=pilot_updates,
        full_updates=full_updates,
        validation_interval=validation_interval,
        search_space=search_space,
        training_mode="pc",
    )
    (run_dir / "coordinator_config.json").write_text(
        json.dumps(
            {
                **asdict(settings),
                "run_dir": str(run_dir),
                "study_name": study_name,
                "n_trials": n_trials,
                "max_workers": max_workers,
                "gpu_memory_budget_mib": gpu_memory_budget_mib,
                "estimated_trial_memory_mib": estimated_trial_memory_mib,
                "training_mode": "pc",
                "note": "Predictive coding only; backpropagation is not used.",
            },
            indent=2,
        )
    )

    workers: dict[int, WorkerProcess] = {}
    worker_index = 0
    typer.echo(
        f"Study {study_name} [PC-only, search_space={search_space}]: "
        f"target={n_trials}, max_workers={max_workers}, "
        f"GPU budget={gpu_memory_budget_mib} MiB"
    )

    try:
        while _study_finished_trials(study) < n_trials or workers:
            for pid, worker in list(workers.items()):
                return_code = worker.process.poll()
                if (
                    return_code is None
                    and time.time() - worker.started_at > trial_timeout_seconds + 30
                ):
                    typer.echo(f"Worker {pid} exceeded its trial timeout", err=True)
                    _stop_worker(worker)
                    _fail_running_trial_for_pid(study, pid)
                    del workers[pid]
                    continue
                if return_code is None:
                    continue
                worker.log_file.close()
                del workers[pid]
                if return_code != 0:
                    _fail_running_trial_for_pid(study, pid)
                    typer.echo(
                        f"Worker {pid} exited {return_code}; see {worker.log_path}",
                        err=True,
                    )

            finished = _study_finished_trials(study)
            remaining = max(0, n_trials - finished - len(workers))
            allocations = query_cuda_process_memory()
            admitted = admitted_worker_count(
                allocations=allocations,
                managed_pids=set(workers),
                active_workers=len(workers),
                gpu_memory_budget_mib=gpu_memory_budget_mib,
                estimated_trial_memory_mib=estimated_trial_memory_mib,
                max_workers=max_workers,
            )
            for _ in range(min(remaining, admitted)):
                worker_index += 1
                worker = _spawn_worker(
                    worker_index=worker_index,
                    journal_path=journal_path,
                    study_name=study_name,
                    settings=settings,
                    trial_timeout_seconds=trial_timeout_seconds,
                )
                workers[worker.process.pid] = worker
                typer.echo(
                    f"Started worker pid={worker.process.pid}; log={worker.log_path}"
                )

            _write_coordinator_state(
                run_dir / "coordinator_state.json",
                study=study,
                workers=workers,
                allocations=allocations,
            )
            if _study_finished_trials(study) >= n_trials and not workers:
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        typer.echo("Stopping managed trial process groups...")
        for worker in workers.values():
            _stop_worker(worker)
        raise typer.Exit(130)

    completed = [
        trial
        for trial in study.get_trials()
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        typer.echo("No trial completed successfully.", err=True)
        raise typer.Exit(1)
    best = study.best_trial
    summary = {
        "best_trial": best.number,
        "best_val_mae_mg_dl": best.value,
        "best_params": best.params,
        "terminal_trials": _study_finished_trials(study),
    }
    (run_dir / "best_trial.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


def _make_eval_step(
    structure: Any,
    glucose_min: float,
    glucose_max: float,
) -> Any:
    import jax
    import jax.numpy as jnp

    from fabricpc.training.train_backprop import compute_forward_pass

    glucose_range = max(glucose_max - glucose_min, 1e-8)

    @jax.jit
    def eval_step(params: Any, batch: dict[str, Any], key: Any) -> tuple[Any, Any]:
        state = compute_forward_pass(params, structure, batch, key)
        predictions = (
            state.nodes[structure.task_map["y"]].z_mu * glucose_range + glucose_min
        )
        targets = batch["y"] * glucose_range + glucose_min
        absolute_error = jnp.abs(predictions - targets)
        return jnp.sum(absolute_error), jnp.sum(
            absolute_error / jnp.maximum(jnp.abs(targets), 1e-8)
        )

    return eval_step


def _evaluate_validation(
    params: Any,
    loader: Any,
    eval_step: Any,
    rng_key: Any,
) -> dict[str, float]:
    import jax.numpy as jnp

    absolute_error = 0.0
    relative_error = 0.0
    values = 0
    for batch_np in loader:
        batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
        batch_ae, batch_are = eval_step(params, batch, rng_key)
        absolute_error += float(batch_ae)
        relative_error += float(batch_are)
        values += int(batch["y"].size)
    return {
        "mae_mg_dl": absolute_error / max(values, 1),
        "mard_percent": 100.0 * relative_error / max(values, 1),
    }


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "resource_exhausted" in text


def _own_gpu_memory_mib() -> int:
    return query_cuda_process_memory().get(os.getpid(), 0)


def _run_trial_attempt(
    trial: optuna.Trial,
    dynamics: dict[str, float | int],
    settings: TrialSettings,
    *,
    batch_size: int,
) -> float:
    from jax_setup import set_jax_flags_before_importing_jax

    set_jax_flags_before_importing_jax()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from examples.glucose_data import prepare_data
    from examples.glucose_model import create_glucose_transformer
    from fabricpc.core.inference import InferenceSGDNormClip
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.training.train import train_step

    trial_dir = settings.run_dir / "trials" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial.set_user_attr("artifact_dir", str(trial_dir))
    data = prepare_data(
        seq_len=int(dynamics["seq_len"]),
        horizon=settings.horizon,
        batch_size=batch_size,
        seed=settings.seed,
    )
    inference = InferenceSGDNormClip(
        eta_infer=float(dynamics["eta_infer"]),
        infer_steps=int(dynamics["infer_steps"]),
        max_norm=float(dynamics["max_infer_norm"]),
    )
    readout = str(dynamics.get("readout", "flatten"))
    structure = create_glucose_transformer(
        depth=int(dynamics["depth"]),
        embed_dim=settings.embed_dim,
        num_heads=int(dynamics["num_heads"]),
        mlp_dim=settings.mlp_dim,
        seq_len=int(dynamics["seq_len"]),
        horizon=settings.horizon,
        inference=inference,
        weight_init_std=float(dynamics["weight_init_std"]),
        include_output_scaling=True,
        readout=readout,
    )

    seed_offset = int(dynamics.get("seed_offset", trial.number))
    seed = settings.seed + seed_offset
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    params = initialize_params(structure, init_key)
    warmup_updates = min(200, settings.full_updates // 2)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(dynamics["lr"]),
        warmup_steps=warmup_updates,
        decay_steps=settings.full_updates,
        end_value=float(dynamics["lr"]) * 0.01,
    )
    weight_decay = float(dynamics.get("weight_decay", 0.0))
    # Match phase-1 Adam when decay is zero; use AdamW only when searching decay.
    scalar_opt = (
        optax.adamw(schedule, weight_decay=weight_decay)
        if weight_decay > 0.0
        else optax.adam(schedule)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(dynamics["grad_clip"])),
        scalar_opt,
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def step_fn(
        step_params: Any,
        step_opt_state: Any,
        batch: dict[str, Any],
        key: Any,
    ) -> tuple[Any, Any, Any, Any]:
        return train_step(
            step_params, step_opt_state, batch, structure, optimizer, key
        )

    eval_step = _make_eval_step(
        structure, float(data["g_min"]), float(data["g_max"])
    )
    loader = data["train_loader"].cycle()
    history: list[dict[str, float | int]] = []
    recent_energies: list[float] = []
    best_mae = math.inf
    significant_best_mae = math.inf
    checks_without_improvement = 0
    regression_checks = 0
    peak_gpu_memory_mib = 0
    started_at = time.time()

    resolved = {
        **asdict(settings),
        "run_dir": str(settings.run_dir),
        **dynamics,
        "batch_size": batch_size,
        "seed": seed,
        "seed_offset": seed_offset,
        "readout": readout,
        "include_output_scaling": True,
    }
    (trial_dir / "config.json").write_text(json.dumps(resolved, indent=2))
    # Breakthrough: allow slower improvement before stopping (prior HPO cut early).
    patience_checks = 6 if settings.search_space == "breakthrough" else 4

    for update in range(1, settings.full_updates + 1):
        batch_np = next(loader)
        batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
        rng, step_key = jax.random.split(rng)
        params, opt_state, energy, _ = step_fn(
            params, opt_state, batch, step_key
        )
        energy_value = float(energy)
        if not np.isfinite(energy_value):
            trial.set_user_attr("prune_reason", f"non-finite energy at {update}")
            raise optuna.TrialPruned()
        if len(recent_energies) >= 20:
            median_energy = float(np.median(recent_energies[-20:]))
            if median_energy > 0 and energy_value > 10.0 * median_energy:
                trial.set_user_attr(
                    "prune_reason",
                    f"energy explosion at {update}: {energy_value:.6g}",
                )
                raise optuna.TrialPruned()
        recent_energies.append(energy_value)

        should_validate = (
            update % settings.validation_interval == 0
            or update == settings.pilot_updates
            or update == settings.full_updates
        )
        if not should_validate:
            continue

        rng, eval_key = jax.random.split(rng)
        metrics = _evaluate_validation(
            params, data["val_loader"], eval_step, eval_key
        )
        mae = metrics["mae_mg_dl"]
        parameter_norm = float(optax.tree.norm(params))
        peak_gpu_memory_mib = max(peak_gpu_memory_mib, _own_gpu_memory_mib())
        trial.set_user_attr("peak_gpu_memory_mib", peak_gpu_memory_mib)
        history.append(
            {
                "update": update,
                "energy": energy_value,
                "parameter_norm": parameter_norm,
                **metrics,
                "elapsed_s": time.time() - started_at,
                "gpu_memory_mib": peak_gpu_memory_mib,
            }
        )
        (trial_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"trial={trial.number} update={update} val_mae={mae:.3f} "
            f"energy={energy_value:.6g} param_norm={parameter_norm:.3f}",
            flush=True,
        )

        validation_index = len(history)
        trial.report(mae, step=validation_index)
        if mae < best_mae:
            best_mae = mae
            with (trial_dir / "best_params.pkl").open("wb") as file:
                import pickle

                pickle.dump(params, file)
        if mae + 0.25 < significant_best_mae:
            significant_best_mae = mae
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        previous_best = min(
            (float(row["mae_mg_dl"]) for row in history[:-1]),
            default=math.inf,
        )
        if previous_best < math.inf and mae > previous_best * 1.10:
            regression_checks += 1
        else:
            regression_checks = 0

        if regression_checks >= 2:
            trial.set_user_attr(
                "stop_reason", "validation MAE regressed over 10% twice"
            )
            break
        if (
            validation_index >= patience_checks
            and checks_without_improvement >= patience_checks
        ):
            trial.set_user_attr(
                "stop_reason",
                f"no 0.25 mg/dL improvement over {patience_checks} checks",
            )
            break
        if trial.should_prune():
            trial.set_user_attr(
                "prune_reason",
                f"{type(trial.study.pruner).__name__} at check {validation_index}",
            )
            raise optuna.TrialPruned()

        if update == settings.pilot_updates:
            trial.set_user_attr("pilot_val_mae_mg_dl", mae)
            trial.set_user_attr("pilot_passed", True)

    trial.set_user_attr("peak_gpu_memory_mib", peak_gpu_memory_mib)
    trial.set_user_attr("elapsed_s", time.time() - started_at)
    return best_mae


def objective(trial: optuna.Trial, settings: TrialSettings) -> float:
    """Run one PC trial, with a single half-batch retry on CUDA OOM."""
    if settings.training_mode != "pc":
        raise RuntimeError(
            "glucose_transformer_tuning only supports predictive coding "
            f"(got training_mode={settings.training_mode!r})"
        )
    trial.set_user_attr("worker_pid", os.getpid())
    trial.set_user_attr("training_mode", "pc")
    trial.set_user_attr("search_space", settings.search_space)
    dynamics = suggest_pc_dynamics(trial, settings.search_space)
    try:
        return _run_trial_attempt(
            trial, dynamics, settings, batch_size=settings.batch_size
        )
    except optuna.TrialPruned:
        raise
    except Exception as error:
        if not _is_oom(error) or settings.batch_size == 1:
            trial.set_user_attr("failure_reason", repr(error))
            raise

    import gc
    import jax

    jax.clear_caches()
    gc.collect()
    retry_batch_size = max(1, settings.batch_size // 2)
    trial.set_user_attr("oom_retry_batch_size", retry_batch_size)
    return _run_trial_attempt(
        trial, dynamics, settings, batch_size=retry_batch_size
    )


@app.command("worker", hidden=True)
def run_worker(
    journal: Path = typer.Option(...),
    study_name: str = typer.Option(DEFAULT_STUDY_NAME),
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR),
    seq_len: int = typer.Option(128),
    horizon: int = typer.Option(12),
    depth: int = typer.Option(3),
    embed_dim: int = typer.Option(32),
    num_heads: int = typer.Option(4),
    mlp_dim: int = typer.Option(128),
    batch_size: int = typer.Option(64),
    pilot_updates: int = typer.Option(600),
    full_updates: int = typer.Option(6000),
    validation_interval: int = typer.Option(200),
    seed: int = typer.Option(42),
    search_space: str = typer.Option(DEFAULT_SEARCH_SPACE),
    trial_timeout_seconds: int = typer.Option(7200),
) -> None:
    """Claim and execute one PC trial from the shared journal (no backprop)."""
    allowed_spaces = {"phase1", "refined", "local", "breakthrough"}
    if search_space not in allowed_spaces:
        raise typer.BadParameter(
            "search-space must be one of: " + ", ".join(sorted(allowed_spaces))
        )
    settings = TrialSettings(
        run_dir=run_dir,
        seq_len=seq_len,
        horizon=horizon,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        batch_size=batch_size,
        pilot_updates=pilot_updates,
        full_updates=full_updates,
        validation_interval=validation_interval,
        seed=seed,
        search_space=search_space,
        training_mode="pc",
    )
    study = create_study(journal, study_name, search_space=search_space)
    # Parallel workers otherwise construct identically seeded TPE samplers and
    # propose duplicate startup trials.
    study.sampler.reseed_rng()

    # SIGALRM is POSIX-only; on Windows the coordinator enforces the wall clock.
    use_alarm = hasattr(signal, "SIGALRM")
    previous_handler = None
    if use_alarm:

        def timeout_handler(_signum: int, _frame: Any) -> None:
            raise TimeoutError(
                f"trial exceeded {trial_timeout_seconds} seconds"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(trial_timeout_seconds)
    try:
        study.optimize(
            lambda trial: objective(trial, settings),
            n_trials=1,
            catch=(Exception,),
        )
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)


def main() -> None:
    """Console entry point."""
    app()


if __name__ == "__main__":
    main()
