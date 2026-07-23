"""Parallel Optuna tuning for the FabricPC glucose transformer.

The coordinator deliberately does not import JAX. Each Optuna trial runs in a
fresh child process so compilation caches and CUDA allocations disappear when
the trial exits.
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
from optuna.storages.journal import JournalFileBackend

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

app = typer.Typer(help="Tune FabricPC glucose predictive-coding dynamics.")

DEFAULT_RUN_DIR = Path("runs/glucose_tuning")
DEFAULT_STUDY_NAME = "glucose_transformer_pc"
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
    return JournalStorage(JournalFileBackend(str(journal_path)))


def create_study(journal_path: Path, study_name: str) -> optuna.Study:
    """Create or load the shared PC dynamics study."""
    return optuna.create_study(
        study_name=study_name,
        storage=create_storage(journal_path),
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=16,
            multivariate=True,
            group=True,
            constant_liar=True,
        ),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        ),
    )


def suggest_pc_dynamics(trial: optuna.Trial) -> dict[str, float | int]:
    """Search PC dynamics and architecture factors linked to instability."""
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
    }


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
        "examples.glucose_transformer_tuning",
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
    """Terminate exactly one worker process group."""
    if worker.process.poll() is None:
        os.killpg(worker.process.pid, signal.SIGTERM)
        try:
            worker.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
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
    """Coordinate process-isolated PC trials under a CUDA memory budget."""
    if pilot_updates > full_updates:
        raise typer.BadParameter("pilot-updates cannot exceed full-updates")
    if pilot_updates < 3 * validation_interval:
        raise typer.BadParameter(
            "pilot-updates must include at least three validation intervals"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    journal_path = run_dir / "optuna_journal.log"
    study = create_study(journal_path, study_name)
    settings = TrialSettings(
        run_dir=run_dir,
        batch_size=batch_size,
        pilot_updates=pilot_updates,
        full_updates=full_updates,
        validation_interval=validation_interval,
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
            },
            indent=2,
        )
    )

    workers: dict[int, WorkerProcess] = {}
    worker_index = 0
    typer.echo(
        f"Study {study_name}: target={n_trials}, max_workers={max_workers}, "
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
    )

    seed = settings.seed + trial.number
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
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(dynamics["grad_clip"])),
        optax.adam(schedule),
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
        "include_output_scaling": True,
    }
    (trial_dir / "config.json").write_text(json.dumps(resolved, indent=2))

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
        if mae + 0.25 < best_mae:
            best_mae = mae
            checks_without_improvement = 0
            with (trial_dir / "best_params.pkl").open("wb") as file:
                import pickle

                pickle.dump(params, file)
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
        if validation_index >= 4 and checks_without_improvement >= 4:
            trial.set_user_attr(
                "stop_reason", "no 0.25 mg/dL improvement over four checks"
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
    trial.set_user_attr("worker_pid", os.getpid())
    dynamics = suggest_pc_dynamics(trial)
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
        pilot_updates=pilot_updates,
        full_updates=full_updates,
        validation_interval=validation_interval,
        seed=seed,
    )
    study = create_study(journal, study_name)
    # Parallel workers otherwise construct identically seeded TPE samplers and
    # propose duplicate startup trials.
    study.sampler.reseed_rng()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(
            f"trial exceeded {trial_timeout_seconds} seconds"
        )

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(trial_timeout_seconds)
    try:
        study.optimize(
            lambda trial: objective(trial, settings),
            n_trials=1,
            catch=(Exception,),
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def main() -> None:
    """Console entry point."""
    app()


if __name__ == "__main__":
    main()
