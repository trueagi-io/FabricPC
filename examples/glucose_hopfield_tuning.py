"""Epoch-based Optuna tuning for FabricPC glucose Hopfield variants.

Mirrors the active transformer tuner (``examples/glucose_transformer_tuning.py``):
complete-epoch trials, Hyperband pruning, process-isolated workers, GPU admission.

Search is informed by the phase-4 transformer breakthrough (~19.876 Optuna val MAE):
locked geometry ``64 / depth 1 / heads 1``, η band ``9e-6–2.5e-5``, and
``seed_offset``. Trials also search Storkey placement / strength.

Uses the same ``prepare_data`` split as the transformer champion so MAE is
comparable. Goal: beat **19.876** Optuna validation MAE.

```bash
uv run glucose-hopfield-tune run \\
  --run-dir runs/glucose_hopfield_tuning_v1 \\
  --study-name glucose_hopfield_pc_v1 \\
  --n-trials 24 --max-workers 3 \\
  --max-epochs 15 --min-pruning-epochs 3 --patience 4
```
"""
from __future__ import annotations

import json
import math
import os
import pickle
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import optuna
import typer
from dotenv import load_dotenv
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

load_dotenv()

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# Prefer CUDA when nvidia-smi is present (WSL/Linux). Windows has no JAX CUDA
# wheels, so leave unset there and fall back to CPU.
if "JAX_PLATFORMS" not in os.environ:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            os.environ["JAX_PLATFORMS"] = "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

app = typer.Typer(
    help=(
        "Tune FabricPC glucose Hopfield + PC dynamics. "
        "Target: beat transformer phase-4 Optuna champion 19.876 MAE."
    )
)

DEFAULT_RUN_DIR = Path("runs/glucose_hopfield_tuning_v1")
DEFAULT_STUDY_NAME = "glucose_hopfield_pc_v1"
TARGET_OPTUNA_MAE = 19.876
TERMINAL_STATES = {
    optuna.trial.TrialState.COMPLETE,
    optuna.trial.TrialState.PRUNED,
    optuna.trial.TrialState.FAIL,
}

# Phase-4 transformer champion (trial 7) — used as enqueued baselines.
CHAMPION_PC_PARAMS: dict[str, float | int | str] = {
    "seq_len": 64,
    "depth": 1,
    "num_heads": 1,
    "lr": 0.0036910070447662953,
    "eta_infer": 1.67967291868957e-05,
    "infer_steps": 12,
    "max_infer_norm": 1.0,
    "grad_clip": 0.5,
    "lr_decay_epochs": 10,
    "weight_init_std": 0.01531315474155434,
    "seed_offset": 19,  # seed 42 + 19 = 61
}


@dataclass(frozen=True)
class TrialSettings:
    """Fixed resource budget shared by all Hopfield trials."""

    run_dir: Path
    seq_len: int = 64
    horizon: int = 12
    depth: int = 1
    embed_dim: int = 32
    num_heads: int = 1
    mlp_dim: int = 128
    batch_size: int = 64
    max_epochs: int = 15
    min_pruning_epochs: int = 3
    patience: int = 4
    warmup_steps: int = 200
    seed: int = 42
    max_batches_per_epoch: int | None = None
    max_validation_batches: int | None = None


@dataclass
class WorkerProcess:
    """A managed trial worker and its output stream."""

    process: subprocess.Popen[bytes]
    log_file: Any
    log_path: Path
    started_at: float


def create_storage(journal_path: Path) -> JournalStorage:
    """Create resumable, multi-process-safe local Optuna storage."""
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
    max_epochs: int = 15,
    min_pruning_epochs: int = 3,
) -> optuna.Study:
    """Create or load the shared Hopfield PC study."""
    return optuna.create_study(
        study_name=study_name,
        storage=create_storage(journal_path),
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=12,
            multivariate=True,
            group=True,
            constant_liar=True,
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min_pruning_epochs,
            max_resource=max_epochs,
            reduction_factor=3,
        ),
    )


def enqueue_hopfield_baselines(study: optuna.Study) -> None:
    """Seed with transformer champion + matched Hopfield arms."""
    if study.get_trials(deepcopy=False):
        return
    baselines: list[dict[str, float | int | str]] = [
        {
            **CHAMPION_PC_PARAMS,
            "variant": "baseline",
            "hopfield_strength": "1.0",
        },
        {
            **CHAMPION_PC_PARAMS,
            "variant": "projection",
            "hopfield_strength": "1.0",
        },
        {
            **CHAMPION_PC_PARAMS,
            "variant": "embed-storkey",
            "hopfield_strength": "1.0",
        },
        {
            **CHAMPION_PC_PARAMS,
            "variant": "embed-storkey",
            "hopfield_strength": "2.0",
        },
        {
            **CHAMPION_PC_PARAMS,
            "variant": "embed-storkey",
            "hopfield_strength": "learnable",
        },
        {
            **CHAMPION_PC_PARAMS,
            "variant": "forecast-storkey",
            "hopfield_strength": "1.0",
        },
        {
            **CHAMPION_PC_PARAMS,
            "lr": 0.003533,
            "eta_infer": 2.207e-05,
            "seed_offset": 21,
            "variant": "embed-storkey",
            "hopfield_strength": "1.5",
        },
        {
            **CHAMPION_PC_PARAMS,
            "lr": 0.002667,
            "eta_infer": 9.998e-06,
            "infer_steps": 14,
            "seed_offset": 21,
            "variant": "embed-storkey",
            "hopfield_strength": "2.0",
        },
    ]
    for params in baselines:
        study.enqueue_trial(params)


def suggest_hopfield_dynamics(trial: optuna.Trial) -> dict[str, float | int | str]:
    """Search Hopfield placement/strength plus breakthrough-informed PC knobs."""
    from examples.glucose_tuning_spaces import HOPFIELD_SPACE, suggest_from_spec

    return suggest_from_spec(trial, HOPFIELD_SPACE)


def _parse_hopfield_strength(raw: str | float | int) -> float | None:
    if isinstance(raw, str) and raw == "learnable":
        return None
    return float(raw)


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
        "examples.glucose_hopfield_tuning",
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
        "--max-epochs",
        str(settings.max_epochs),
        "--min-pruning-epochs",
        str(settings.min_pruning_epochs),
        "--patience",
        str(settings.patience),
        "--warmup-steps",
        str(settings.warmup_steps),
        "--seed",
        str(settings.seed),
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
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
    n_trials: int = typer.Option(24, min=1, help="Total terminal trials desired."),
    max_workers: int = typer.Option(3, min=1, help="Maximum parallel workers."),
    gpu_memory_budget_mib: int = typer.Option(
        12000, min=1, help="Total GPU memory budget, including unrelated jobs."
    ),
    estimated_trial_memory_mib: int = typer.Option(
        2500, min=1, help="Admission reservation per worker."
    ),
    batch_size: int = typer.Option(64, min=1),
    max_epochs: int = typer.Option(15, min=1),
    min_pruning_epochs: int = typer.Option(3, min=1),
    patience: int = typer.Option(4, min=1),
    warmup_steps: int = typer.Option(200, min=0),
    poll_seconds: float = typer.Option(2.0, min=0.1),
    trial_timeout_seconds: int = typer.Option(
        7200, min=60, help="Hard wall-clock bound for one trial worker."
    ),
    continue_best: bool = typer.Option(
        True, help="Continue training the best trial to MAX_EPOCHS after tuning."
    ),
) -> None:
    """Coordinate process-isolated Hopfield PC trials under a CUDA budget."""
    if min_pruning_epochs > max_epochs:
        raise typer.BadParameter("min-pruning-epochs cannot exceed max-epochs")

    run_dir.mkdir(parents=True, exist_ok=True)
    journal_path = run_dir / "optuna_journal.log"
    study = create_study(
        journal_path,
        study_name,
        max_epochs=max_epochs,
        min_pruning_epochs=min_pruning_epochs,
    )
    enqueue_hopfield_baselines(study)
    settings = TrialSettings(
        run_dir=run_dir,
        batch_size=batch_size,
        max_epochs=max_epochs,
        min_pruning_epochs=min_pruning_epochs,
        patience=patience,
        warmup_steps=warmup_steps,
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
                "target_optuna_mae": TARGET_OPTUNA_MAE,
                "protocol": "epoch-based Hyperband (mirrors glucose-transformer-tune)",
                "search_notes": (
                    "Locked geometry 64/d1/h1; Hopfield variant+strength; "
                    "PC knobs in breakthrough band; same prepare_data split"
                ),
            },
            indent=2,
        )
    )

    workers: dict[int, WorkerProcess] = {}
    worker_index = 0
    typer.echo(
        f"Study {study_name}: target={n_trials}, max_workers={max_workers}, "
        f"GPU budget={gpu_memory_budget_mib} MiB, beat_mae={TARGET_OPTUNA_MAE}"
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
        "target_optuna_mae": TARGET_OPTUNA_MAE,
        "beat_target": bool(best.value is not None and best.value < TARGET_OPTUNA_MAE),
        "delta_vs_target": (
            None if best.value is None else float(best.value) - TARGET_OPTUNA_MAE
        ),
    }
    (run_dir / "best_trial.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))

    if continue_best:
        max_epochs_env = int(os.environ.get("MAX_EPOCHS", "30"))
        if max_epochs_env > max_epochs:
            typer.echo(
                f"\nContinuing best trial to {max_epochs_env} epochs..."
            )
            cmd = [
                sys.executable,
                "-m",
                "examples.glucose_hopfield_tuning",
                "continue-best",
                "--run-dir",
                str(run_dir),
                "--max-epochs",
                str(max_epochs_env),
                "--patience",
                str(patience),
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                typer.echo("Continuation training failed", err=True)
        else:
            typer.echo(
                f"Skipping continuation: MAX_EPOCHS={max_epochs_env} "
                f"<= tuning max_epochs={max_epochs}"
            )


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
    max_batches: int | None = None,
) -> dict[str, float]:
    import jax.numpy as jnp

    absolute_error = 0.0
    relative_error = 0.0
    values = 0
    for batch_index, batch_np in enumerate(loader, start=1):
        batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
        batch_ae, batch_are = eval_step(params, batch, rng_key)
        absolute_error += float(batch_ae)
        relative_error += float(batch_are)
        values += int(batch["y"].size)
        if max_batches is not None and batch_index >= max_batches:
            break
    return {
        "mae_mg_dl": absolute_error / max(values, 1),
        "mard_percent": 100.0 * relative_error / max(values, 1),
    }


def _run_best_continuation(
    run_dir: Path,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    """Continue training the best Optuna trial's model to max_epochs."""
    from jax_setup import set_jax_flags_before_importing_jax

    set_jax_flags_before_importing_jax()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from examples.glucose_data import prepare_data
    from examples.glucose_hopfield_model import create_glucose_hopfield_transformer
    from fabricpc.core.inference import InferenceSGDNormClip
    from fabricpc.training.train import train_step
    from fabricpc.training.train_backprop import compute_forward_pass

    best_info = json.loads((run_dir / "best_trial.json").read_text())
    trial_number = best_info["best_trial"]
    trial_dir = run_dir / "trials" / f"trial_{trial_number:04d}"
    config = json.loads((trial_dir / "config.json").read_text())

    checkpoint_path = trial_dir / "checkpoint.pkl"
    if not checkpoint_path.is_file():
        print(f"No checkpoint for trial {trial_number}, skipping continuation")
        return {"skipped": True, "reason": "no_checkpoint"}

    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)

    start_epoch = checkpoint["epoch"] + 1
    if start_epoch > max_epochs:
        print(
            f"Trial {trial_number} reached epoch {checkpoint['epoch']}, "
            f"already >= {max_epochs}"
        )
        return {"skipped": True, "reason": "already_at_max_epochs"}

    cont_dir = run_dir / "best_continued"
    cont_dir.mkdir(parents=True, exist_ok=True)
    trial_best = trial_dir / "best_params.pkl"
    if trial_best.is_file():
        shutil.copy2(trial_best, cont_dir / "best_params.pkl")

    data = prepare_data(
        seq_len=int(config["seq_len"]),
        horizon=int(config["horizon"]),
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
    )
    inference = InferenceSGDNormClip(
        eta_infer=float(config["eta_infer"]),
        infer_steps=int(config["infer_steps"]),
        max_norm=float(config["max_infer_norm"]),
    )
    variant = str(config["variant"])
    hopfield_strength = _parse_hopfield_strength(config["hopfield_strength"])
    structure = create_glucose_hopfield_transformer(
        variant=variant,  # type: ignore[arg-type]
        hopfield_strength=hopfield_strength,
        depth=int(config["depth"]),
        embed_dim=int(config["embed_dim"]),
        num_heads=int(config["num_heads"]),
        mlp_dim=int(config["mlp_dim"]),
        seq_len=int(config["seq_len"]),
        horizon=int(config["horizon"]),
        inference=inference,
        weight_init_std=float(config["weight_init_std"]),
        include_output_scaling=config.get("include_output_scaling", True),
    )

    params = checkpoint["params"]
    rng = checkpoint["rng"]
    best_mae = checkpoint["best_mae"]
    tuning_best_mae = best_mae
    global_step = checkpoint["global_step"]

    batches_per_epoch = len(data["train_loader"])
    remaining_epochs = max_epochs - start_epoch + 1
    remaining_steps = remaining_epochs * batches_per_epoch
    cont_lr = float(config["lr"]) * 0.1
    warmup = min(50, remaining_steps // 4)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cont_lr,
        warmup_steps=warmup,
        decay_steps=remaining_steps,
        end_value=cont_lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(config["grad_clip"])),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(params)
    glucose_range = max(float(data["g_max"]) - float(data["g_min"]), 1e-8)
    glucose_min = float(data["g_min"])

    @jax.jit
    def step_fn(step_params, step_opt_state, batch, train_key, metric_key):
        up_params, up_opt, energy_val, final_state = train_step(
            step_params, step_opt_state, batch, structure, optimizer, train_key,
        )
        fwd = compute_forward_pass(up_params, structure, batch, metric_key)
        preds = (
            fwd.nodes[structure.task_map["y"]].z_mu * glucose_range + glucose_min
        )
        targets = batch["y"] * glucose_range + glucose_min
        t_mae = jnp.mean(jnp.abs(preds - targets))
        return up_params, up_opt, energy_val, final_state, t_mae

    eval_step = _make_eval_step(
        structure, float(data["g_min"]), float(data["g_max"])
    )
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0
    regression_checks = 0
    started_at = time.time()
    final_epoch = start_epoch - 1

    print(
        f"\nContinuation: trial {trial_number} ({variant}/"
        f"{config['hopfield_strength']}), epochs {start_epoch}-{max_epochs}, "
        f"lr={cont_lr:.2e}, patience={patience}, "
        f"tuning_best_mae={tuning_best_mae:.3f}"
    )

    unstable = False
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_energy = 0.0
        epoch_batches = 0
        batch_train_maes: list[float] = []
        final_epoch = epoch

        for batch_np in data["train_loader"]:
            batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
            rng, step_key, metric_key = jax.random.split(rng, 3)
            params, opt_state, energy_val, _, train_mae = step_fn(
                params, opt_state, batch, step_key, metric_key,
            )
            energy_value = float(energy_val)
            batch_train_maes.append(float(train_mae))
            global_step += 1
            epoch_batches += 1
            epoch_energy += energy_value
            if not np.isfinite(energy_value):
                print(
                    f"Non-finite energy at epoch {epoch}, step {global_step}"
                )
                unstable = True
                break

        if unstable:
            break

        avg_energy = epoch_energy / max(epoch_batches, 1)
        train_mae_mean = float(np.mean(batch_train_maes))
        rng, eval_key = jax.random.split(rng)
        metrics = _evaluate_validation(
            params, data["val_loader"], eval_step, eval_key,
        )
        mae = metrics["mae_mg_dl"]
        is_best = mae < best_mae
        best_tag = " *" if is_best else ""
        history.append(
            {
                "epoch": epoch,
                "step": global_step,
                "avg_energy": avg_energy,
                "train_mae_mg_dl": train_mae_mean,
                **metrics,
                "elapsed_s": time.time() - started_at,
            }
        )
        (cont_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"cont epoch={epoch}/{max_epochs} step={global_step} "
            f"val_mae={mae:.3f} train_mae={train_mae_mean:.3f} "
            f"avg_energy={avg_energy:.6g}{best_tag}",
            flush=True,
        )

        if is_best:
            best_mae = mae
            epochs_without_improvement = 0
            with (cont_dir / "best_params.pkl").open("wb") as f:
                pickle.dump(params, f)
        else:
            epochs_without_improvement += 1

        previous_best = min(
            (float(row["mae_mg_dl"]) for row in history[:-1]),
            default=math.inf,
        )
        if previous_best < math.inf and mae > previous_best * 1.10:
            regression_checks += 1
        else:
            regression_checks = 0

        cont_ckpt = {
            "params": params,
            "opt_state": opt_state,
            "rng": rng,
            "epoch": epoch,
            "global_step": global_step,
            "best_mae": best_mae,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        ckpt_tmp = cont_dir / "checkpoint.pkl.tmp"
        with ckpt_tmp.open("wb") as f:
            pickle.dump(cont_ckpt, f)
        ckpt_tmp.replace(cont_dir / "checkpoint.pkl")

        if regression_checks >= 2:
            print("Validation MAE regressed >10% twice — stopping")
            break
        if epochs_without_improvement >= patience:
            print(f"No improvement for {patience} epochs — stopping")
            break

    summary: dict[str, Any] = {
        "source_trial": trial_number,
        "variant": variant,
        "hopfield_strength": str(config["hopfield_strength"]),
        "tuning_best_mae": tuning_best_mae,
        "continuation_best_mae": best_mae,
        "improved": best_mae < tuning_best_mae,
        "improvement": (
            round(tuning_best_mae - best_mae, 6)
            if best_mae < tuning_best_mae
            else 0.0
        ),
        "epochs_trained": final_epoch - start_epoch + 1,
        "epoch_range": [start_epoch, final_epoch],
        "max_epochs": max_epochs,
        "continuation_lr": cont_lr,
        "unstable": unstable,
        "elapsed_s": round(time.time() - started_at, 1),
    }
    (cont_dir / "continuation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "resource_exhausted" in text


def _own_gpu_memory_mib() -> int:
    return query_cuda_process_memory().get(os.getpid(), 0)


def _run_trial_attempt(
    trial: optuna.Trial,
    dynamics: dict[str, float | int | str],
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
    from examples.glucose_hopfield_model import create_glucose_hopfield_transformer
    from fabricpc.core.inference import InferenceSGDNormClip
    from fabricpc.graph_initialization import initialize_params
    from fabricpc.training.train import train_step
    from fabricpc.training.train_backprop import compute_forward_pass

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
    variant = str(dynamics["variant"])
    hopfield_strength = _parse_hopfield_strength(dynamics["hopfield_strength"])
    structure = create_glucose_hopfield_transformer(
        variant=variant,  # type: ignore[arg-type]
        hopfield_strength=hopfield_strength,
        depth=int(dynamics["depth"]),
        embed_dim=settings.embed_dim,
        num_heads=int(dynamics["num_heads"]),
        mlp_dim=settings.mlp_dim,
        seq_len=int(dynamics["seq_len"]),
        horizon=settings.horizon,
        inference=inference,
        weight_init_std=float(dynamics["weight_init_std"]),
        include_output_scaling=True,
    )

    seed = settings.seed + int(dynamics["seed_offset"])
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    params = initialize_params(structure, init_key)
    batches_per_epoch = len(data["train_loader"])
    if settings.max_batches_per_epoch is not None:
        batches_per_epoch = min(
            batches_per_epoch,
            settings.max_batches_per_epoch,
        )
    total_steps = settings.max_epochs * batches_per_epoch
    decay_epochs = min(
        int(dynamics["lr_decay_epochs"]),
        settings.max_epochs,
    )
    decay_steps = decay_epochs * batches_per_epoch
    warmup_updates = min(settings.warmup_steps, decay_steps // 2)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(dynamics["lr"]),
        warmup_steps=warmup_updates,
        decay_steps=decay_steps,
        end_value=float(dynamics["lr"]) * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(dynamics["grad_clip"])),
        optax.adam(schedule),
    )
    opt_state = optimizer.init(params)
    glucose_range = max(float(data["g_max"]) - float(data["g_min"]), 1e-8)
    glucose_min = float(data["g_min"])

    @jax.jit
    def step_fn(
        step_params: Any,
        step_opt_state: Any,
        batch: dict[str, Any],
        train_key: Any,
        metric_key: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        updated_params, updated_opt_state, energy, final_state = train_step(
            step_params,
            step_opt_state,
            batch,
            structure,
            optimizer,
            train_key,
        )
        forward_state = compute_forward_pass(
            updated_params,
            structure,
            batch,
            metric_key,
        )
        predictions = (
            forward_state.nodes[structure.task_map["y"]].z_mu * glucose_range
            + glucose_min
        )
        targets = batch["y"] * glucose_range + glucose_min
        train_mae = jnp.mean(jnp.abs(predictions - targets))
        return (
            updated_params,
            updated_opt_state,
            energy,
            final_state,
            train_mae,
        )

    eval_step = _make_eval_step(
        structure, float(data["g_min"]), float(data["g_max"])
    )
    history: list[dict[str, float | int | str]] = []
    recent_energies: list[float] = []
    best_mae = math.inf
    epochs_without_improvement = 0
    regression_checks = 0
    peak_gpu_memory_mib = 0
    started_at = time.time()
    global_step = 0

    resolved = {
        **asdict(settings),
        "run_dir": str(settings.run_dir),
        **dynamics,
        "hopfield_strength_resolved": hopfield_strength,
        "batch_size": batch_size,
        "seed": seed,
        "batches_per_epoch": batches_per_epoch,
        "total_steps": total_steps,
        "decay_steps": decay_steps,
        "include_output_scaling": True,
        "target_optuna_mae": TARGET_OPTUNA_MAE,
        "data_protocol": "examples.glucose_data.prepare_data (transformer-comparable)",
    }
    (trial_dir / "config.json").write_text(json.dumps(resolved, indent=2))

    for epoch in range(1, settings.max_epochs + 1):
        epoch_energy = 0.0
        epoch_batches = 0
        batch_train_maes: list[float] = []
        for batch_index, batch_np in enumerate(data["train_loader"], start=1):
            batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
            rng, step_key, metric_key = jax.random.split(rng, 3)
            params, opt_state, energy, _, train_mae = step_fn(
                params,
                opt_state,
                batch,
                step_key,
                metric_key,
            )
            energy_value = float(energy)
            batch_train_maes.append(float(train_mae))
            global_step += 1
            epoch_batches += 1
            epoch_energy += energy_value
            if not np.isfinite(energy_value):
                trial.set_user_attr(
                    "prune_reason",
                    f"non-finite energy at epoch {epoch}, step {global_step}",
                )
                raise optuna.TrialPruned()
            if len(recent_energies) >= 20:
                median_energy = float(np.median(recent_energies[-20:]))
                # Hopfield dual-energy can spike vs plain PC; keep only severe
                # explosions (absolute + relative), not mild Storkey transients.
                if (
                    median_energy > 0
                    and energy_value > 5.0
                    and energy_value > 50.0 * median_energy
                ):
                    trial.set_user_attr(
                        "prune_reason",
                        f"energy explosion at epoch {epoch}, step {global_step}: "
                        f"{energy_value:.6g}",
                    )
                    raise optuna.TrialPruned()
            recent_energies.append(energy_value)
            if (
                settings.max_batches_per_epoch is not None
                and batch_index >= settings.max_batches_per_epoch
            ):
                break

        avg_energy = epoch_energy / max(epoch_batches, 1)
        train_mae_mean = float(np.mean(batch_train_maes))
        train_mae_std = float(np.std(batch_train_maes))
        train_mae_min = float(np.min(batch_train_maes))
        train_mae_max = float(np.max(batch_train_maes))
        rng, eval_key = jax.random.split(rng)
        metrics = _evaluate_validation(
            params,
            data["val_loader"],
            eval_step,
            eval_key,
            settings.max_validation_batches,
        )
        mae = metrics["mae_mg_dl"]
        parameter_norm = float(optax.tree.norm(params))
        peak_gpu_memory_mib = max(peak_gpu_memory_mib, _own_gpu_memory_mib())
        trial.set_user_attr("peak_gpu_memory_mib", peak_gpu_memory_mib)
        history.append(
            {
                "epoch": epoch,
                "step": global_step,
                "avg_energy": avg_energy,
                "train_mae_mg_dl": train_mae_mean,
                "train_mae_std_mg_dl": train_mae_std,
                "train_mae_min_mg_dl": train_mae_min,
                "train_mae_max_mg_dl": train_mae_max,
                "parameter_norm": parameter_norm,
                **metrics,
                "elapsed_s": time.time() - started_at,
                "gpu_memory_mib": peak_gpu_memory_mib,
            }
        )
        (trial_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"trial={trial.number} epoch={epoch}/{settings.max_epochs} "
            f"variant={variant} strength={dynamics['hopfield_strength']} "
            f"step={global_step} val_mae={mae:.3f} "
            f"train_mae={train_mae_mean:.3f}+/-{train_mae_std:.3f} "
            f"avg_energy={avg_energy:.6g} param_norm={parameter_norm:.3f}",
            flush=True,
        )

        trial.report(mae, step=epoch)
        if mae < best_mae:
            best_mae = mae
            epochs_without_improvement = 0
            with (trial_dir / "best_params.pkl").open("wb") as file:
                import pickle

                pickle.dump(params, file)
        else:
            epochs_without_improvement += 1

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
        if epochs_without_improvement >= settings.patience:
            trial.set_user_attr(
                "stop_reason",
                f"no validation MAE improvement for {settings.patience} epochs",
            )
            break
        if epoch >= settings.min_pruning_epochs and trial.should_prune():
            trial.set_user_attr(
                "prune_reason",
                f"{type(trial.study.pruner).__name__} at epoch {epoch}",
            )
            raise optuna.TrialPruned()

        if epoch == settings.min_pruning_epochs:
            trial.set_user_attr("pilot_val_mae_mg_dl", mae)
            trial.set_user_attr("pilot_passed", True)

    trial.set_user_attr("peak_gpu_memory_mib", peak_gpu_memory_mib)
    trial.set_user_attr("elapsed_s", time.time() - started_at)
    trial.set_user_attr("best_val_mae_mg_dl", best_mae)
    trial.set_user_attr("beat_target", best_mae < TARGET_OPTUNA_MAE)
    trial.set_user_attr("variant", variant)
    trial.set_user_attr("hopfield_strength", str(dynamics["hopfield_strength"]))
    return best_mae


def objective(trial: optuna.Trial, settings: TrialSettings) -> float:
    """Run one Hopfield PC trial, with a single half-batch retry on CUDA OOM."""
    trial.set_user_attr("worker_pid", os.getpid())
    dynamics = suggest_hopfield_dynamics(trial)
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
    seq_len: int = typer.Option(64),
    horizon: int = typer.Option(12),
    depth: int = typer.Option(1),
    embed_dim: int = typer.Option(32),
    num_heads: int = typer.Option(1),
    mlp_dim: int = typer.Option(128),
    batch_size: int = typer.Option(64),
    max_epochs: int = typer.Option(15),
    min_pruning_epochs: int = typer.Option(3),
    patience: int = typer.Option(4),
    warmup_steps: int = typer.Option(200),
    seed: int = typer.Option(42),
    trial_timeout_seconds: int = typer.Option(7200),
) -> None:
    """Claim and execute one trial from the shared journal."""
    settings = TrialSettings(
        run_dir=run_dir,
        seq_len=seq_len,
        horizon=horizon,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        batch_size=batch_size,
        max_epochs=max_epochs,
        min_pruning_epochs=min_pruning_epochs,
        patience=patience,
        warmup_steps=warmup_steps,
        seed=seed,
    )
    study = create_study(
        journal,
        study_name,
        max_epochs=max_epochs,
        min_pruning_epochs=min_pruning_epochs,
    )
    study.sampler.reseed_rng()

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


@app.command("continue-best")
def continue_best_cmd(
    run_dir: Path = typer.Option(DEFAULT_RUN_DIR, help="Study and trial artifacts."),
    max_epochs: int = typer.Option(
        int(os.environ.get("MAX_EPOCHS", "30")),
        min=1,
        help="Target epoch count (default from MAX_EPOCHS env / .env).",
    ),
    patience: int = typer.Option(4, min=1, help="Early-stop patience."),
) -> None:
    """Continue the best trial's model to max_epochs (reads .env MAX_EPOCHS)."""
    result = _run_best_continuation(run_dir, max_epochs, patience)
    typer.echo(json.dumps(result, indent=2))


def main() -> None:
    """Console entry point."""
    app()


if __name__ == "__main__":
    main()
