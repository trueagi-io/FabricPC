"""A run queue: work through a list of benchmark jobs, one after another.

    python -m fabricpc.bench queue <plan.json> --out <dir>
        [--dry-run] [--probe [PATH]] [--only id1,id2] [--continue-on-failure]
        [--validate-plan]

This is for a shared GPU machine (a JupyterHub) where we want to start one
command on day one and let it chew through the whole campaign. A plan file
lists the jobs in the order we care about them:

    {
      "name": "gpu-campaign",
      "defaults": {"zoo": true},
      "jobs": [
        {"id": "cifar10-vgg5", "trials": 5, "note": "clean rerun"},
        {"id": "tinyshakespeare-transformer", "trials": 5, "epochs": 5}
      ]
    }

A job's ``id`` is a row (``mnist-mlp-spc``) or a family (``mnist-mlp``).
``trials`` and ``epochs`` default to the row's own, ``zoo`` (save trained
weights) to off, and ``note`` is just for people. ``defaults`` fills in
anything a job leaves out.

Each job runs through the normal command line, exactly as if it was typed
by hand, with ``--resume`` added. So killing the queue and starting it
again loses at most the trial that was running: every finished trial is
skipped. Each job writes into its own folder, ``<out>/<job id>/``, and the
weights all go to ``<out>/zoo/``.

After every job (and when one starts) the queue rewrites
``<out>/queue-status.json``: each job's status (pending, running, done,
failed, skipped, interrupted, not selected), its start and end times, and
the git commit and device the queue ran on. At the end it runs ``validate``
on every job folder and writes ``<out>/report.md``.

A job whose id is not in the registry (a row family someone is still
adding) is skipped with a message, never a crash. ``--validate-plan``
lists such ids and exits non-zero, so you can check a plan before a long
run. A job that fails stops the queue, unless ``--continue-on-failure``.
A job that trained every trial but missed its expected score (the band
FAIL, which makes the normal command exit 1) counts as done: that is a
result to look at, not a broken run.

``--dry-run`` prints the plan and, when it can find probe files
(``probe-*.json``, see ``fabricpc.bench.probe``) under ``--out`` or at
``--probe PATH``, an estimate in hours. A bare ``--probe`` times the rows
right now on this machine and saves the probe files under ``<out>/probes``.
"""

import argparse
import datetime
import json
import signal
import socket
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from fabricpc.bench.queue_plan import (
    PLANS_DIR,
    Job,
    Plan,
    PlanError,
    estimate_hours,
    job_rows,
    load_plan,
    load_probes,
    unknown_ids,
)

__all__ = [
    "PLANS_DIR",
    "Job",
    "Plan",
    "PlanError",
    "estimate_hours",
    "job_rows",
    "load_plan",
    "load_probes",
    "unknown_ids",
    "run_queue",
    "cli",
]

STATUS_FILE = "queue-status.json"
REPORT_FILE = "report.md"
LIVE_PROBE = "live"  # what a bare --probe means

RunMain = Callable[[List[str]], int]


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _device() -> Dict[str, object]:
    """The device JAX sees, so a CPU run is never mistaken for a GPU one."""
    try:
        import jax

        d = jax.devices()[0]
        return {
            "platform": d.platform,
            "kind": d.device_kind,
            "count": len(jax.devices()),
        }
    except Exception as e:  # the status file must still be written
        return {"platform": "unknown", "kind": str(e), "count": 0}


def _git_sha() -> Optional[str]:
    from fabricpc.bench.manifest import _git_sha as sha

    return sha()


def job_argv(
    job: Job,
    out,
    *,
    in_process: bool = False,
    warmup: Optional[int] = None,
    timed: Optional[int] = None,
) -> List[str]:
    """The normal command line for one job (without ``python -m ...``)."""
    argv = [job.id, "--out", str(Path(out) / job.id), "--resume"]
    if job.trials is not None:
        argv += ["--trials", str(job.trials)]
    if job.epochs is not None:
        argv += ["--epochs", f"{job.epochs:g}"]
    if job.zoo:
        argv += ["--zoo", str(Path(out) / "zoo")]
    if warmup is not None:
        argv += ["--warmup", str(warmup)]
    if timed is not None:
        argv += ["--timed", str(timed)]
    if in_process:
        argv.append("--in-process")
    return argv


def _all_trials_finished(job: Job, out) -> bool:
    """True when every row of the job has every trial finished and good."""
    from fabricpc.bench.registry import ROWS
    from fabricpc.bench.runner import finished_trial

    folder = Path(out) / job.id
    for row_id in job_rows(job.id) or ():
        row = ROWS[row_id]
        trials = job.trials if job.trials is not None else row.n_trials
        epochs = (
            job.epochs if job.epochs is not None else row.train_config["num_epochs"]
        )
        if not all(finished_trial(folder, row_id, t, epochs) for t in range(trials)):
            return False
    return True


class _Status:
    """The status file, kept in memory and rewritten on every change."""

    def __init__(self, plan: Plan, out, plan_path, only):
        self.path = Path(out) / STATUS_FILE
        restarts = 0
        try:
            restarts = json.loads(self.path.read_text()).get("restarts", 0) + 1
        except (OSError, ValueError):
            pass
        self.data = {
            "plan": str(plan_path) if plan_path else None,
            "plan_name": plan.name,
            "out": str(out),
            "state": "running",
            "git_sha": _git_sha(),
            "device": _device(),
            "host": socket.gethostname(),
            "started": _now(),
            "updated": None,
            "ended": None,
            "restarts": restarts,
            "jobs": [],
        }
        for job in plan.jobs:
            rows = job_rows(job.id)
            self.data["jobs"].append(
                {
                    **asdict(job),
                    "rows": list(rows) if rows else [],
                    "folder": str(Path(out) / job.id),
                    "status": (
                        "pending" if not only or job.id in only else "not selected"
                    ),
                    "started": None,
                    "ended": None,
                    "exit_code": None,
                    "message": "",
                }
            )

    def job(self, i: int) -> dict:
        return self.data["jobs"][i]

    def write(self) -> None:
        self.data["updated"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)  # never leave a half-written file behind


def _run_one(job: Job, entry: dict, out, run_main: RunMain, argv_opts) -> bool:
    """Run one job and fill in its status entry. Returns True if it is ok."""
    argv = job_argv(job, out, **argv_opts)
    print(f"queue: starting {job.id}: {' '.join(argv)}", flush=True)
    try:
        code = run_main(argv)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # one broken job must not take the queue down
        code = None
        entry["message"] = f"crashed: {type(e).__name__}: {e}"
    entry["exit_code"] = code
    entry["ended"] = _now()
    if code == 0:
        entry["status"] = "done"
    elif code is not None and _all_trials_finished(job, out):
        entry["status"] = "done"
        entry["message"] = (
            f"exit code {code}, but every trial finished; most likely a "
            "band FAIL (missed the expected score), see the row summaries"
        )
    else:
        entry["status"] = "failed"
        if not entry["message"]:
            entry["message"] = (
                f"exit code {code}; see the trial files in {entry['folder']}"
            )
    print(f"queue: {job.id}: {entry['status']} {entry['message']}".rstrip(), flush=True)
    return entry["status"] == "done"


def _finish(status: _Status, out) -> bool:
    """Validate every job folder and write the report. True if all is well."""
    from fabricpc.bench.report import run_report
    from fabricpc.bench.writer import validate

    problems: List[str] = []
    for entry in status.data["jobs"]:
        folder = Path(entry["folder"])
        if entry["status"] in ("done", "failed") and folder.is_dir():
            problems += validate(folder)
    for p in problems:
        print(f"validate: {p}", file=sys.stderr)
    status.data["validate"] = {"ok": not problems, "problems": problems}
    report = Path(out) / REPORT_FILE
    try:
        run_report(out, out=report)
        status.data["report"] = str(report)
        print(f"queue: report written to {report}")
    except Exception as e:  # the results matter more than the page
        status.data["report"] = None
        print(f"queue: could not write the report: {e}", file=sys.stderr)
    return not problems


def run_queue(
    plan: Plan,
    out,
    *,
    run_main: RunMain,
    only: Optional[Sequence[str]] = None,
    continue_on_failure: bool = False,
    plan_path=None,
    finish: bool = True,
    in_process: bool = False,
    warmup: Optional[int] = None,
    timed: Optional[int] = None,
) -> int:
    """Run the plan's jobs in order. Returns 0 if every selected job is ok.

    ``run_main`` is the normal command line (``fabricpc.bench.__main__.main``);
    tests pass a fake. A KeyboardInterrupt marks the running job as
    interrupted, saves the status and is raised again.
    """
    only = list(only or [])
    status = _Status(plan, out, plan_path, only)
    status.write()
    argv_opts = {"in_process": in_process, "warmup": warmup, "timed": timed}
    ok = True
    for i, job in enumerate(plan.jobs):
        entry = status.job(i)
        if entry["status"] != "pending":
            continue
        if job_rows(job.id) is None:
            entry["status"] = "skipped"
            entry["message"] = (
                f"'{job.id}' is not in the registry (not a row or family)"
            )
            print(f"queue: skipping {job.id}: {entry['message']}", flush=True)
            status.write()
            continue
        entry["status"] = "running"
        entry["started"] = _now()
        status.write()
        try:
            job_ok = _run_one(job, entry, out, run_main, argv_opts)
        except (KeyboardInterrupt, SystemExit):
            entry["status"] = "interrupted"
            entry["ended"] = _now()
            status.data["state"] = "interrupted"
            status.write()
            raise
        status.write()
        if not job_ok:
            ok = False
            if not continue_on_failure:
                print("queue: stopping (use --continue-on-failure to go on)")
                status.data["state"] = "stopped"
                break
    if status.data["state"] == "running":
        status.data["state"] = "finished"
    if finish:
        ok = _finish(status, out) and ok
    status.data["ended"] = _now()
    status.write()
    return 0 if ok else 1


# --- command line -----------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fabricpc.bench queue")
    p.add_argument("plan", help="the plan file (JSON)")
    p.add_argument("--out", help="results folder for the whole queue")
    p.add_argument("--dry-run", action="store_true", help="print the plan only")
    p.add_argument(
        "--probe",
        nargs="?",
        const=LIVE_PROBE,
        help="with --dry-run: a probe-*.json file or folder to estimate from; "
        "on its own, time every row now on this machine",
    )
    p.add_argument(
        "--only",
        action="append",
        help="run only these job ids (comma separated, may repeat)",
    )
    p.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="go on to the next job when one fails",
    )
    p.add_argument(
        "--validate-plan",
        action="store_true",
        help="check the plan and list ids not in the registry",
    )
    p.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--warmup", type=int, help=argparse.SUPPRESS)
    p.add_argument("--timed", type=int, help=argparse.SUPPRESS)
    return p


def _hours(h: float) -> str:
    return f"{h:.1f} h" if h >= 1 else f"{h * 60:.0f} min"


def _job_line(n: int, job: Job, est: Optional[float]) -> str:
    rows = job_rows(job.id)
    if rows is None:
        return f"{n:2d}. {job.id}: unknown id, will be skipped"
    kind = "row" if rows == (job.id,) else f"family of {len(rows)} rows"
    from fabricpc.bench.registry import ROWS

    row = ROWS[rows[0]]
    trials = job.trials if job.trials is not None else f"{row.n_trials} (row's own)"
    epochs = (
        f"{job.epochs:g}"
        if job.epochs is not None
        else f"{row.train_config['num_epochs']:g} (row's own)"
    )
    hours = f", about {_hours(est)}" if est is not None else ", no estimate"
    note = f"  ({job.note})" if job.note else ""
    return (
        f"{n:2d}. {job.id}: {kind}, trials {trials}, epochs {epochs}, "
        f"zoo {'on' if job.zoo else 'off'}{hours}{note}"
    )


def _live_probes(plan: Plan, out) -> List[Path]:
    """Time every known row now, save probe files, return where they went."""
    from fabricpc.bench import probe

    folder = Path(out or ".") / "probes"
    written = []
    for job in plan.jobs:
        for row_id in job_rows(job.id) or ():
            report = probe.probe(row_id, trials=job.trials, epochs=job.epochs)
            print(probe.format_probe(report))
            written.append(probe.write_probe(folder, report))
    return written


def dry_run(plan: Plan, out, probe_path: Optional[str]) -> str:
    """The plan as readable lines, with estimates where probes exist."""
    paths: List[Path] = []
    if out and Path(out).is_dir():
        paths.append(Path(out))
    if probe_path == LIVE_PROBE:
        paths += _live_probes(plan, out)
    elif probe_path:
        paths.append(Path(probe_path))
    probes = load_probes(paths)
    lines = [f"Plan '{plan.name}': {len(plan.jobs)} jobs, in this order"]
    total, missing = 0.0, 0
    for n, job in enumerate(plan.jobs, 1):
        est = estimate_hours(job, probes)
        lines.append(_job_line(n, job, est))
        if est is None:
            missing += 1
        else:
            total += est
    devices = sorted({str(p.get("device")) for p in probes.values()})
    if probes:
        lines.append(
            f"Estimated total: {_hours(total)} for the jobs with a probe "
            f"({missing} without one), measured on {', '.join(devices)}"
        )
    else:
        lines.append("No probe files found, so no time estimate (see --probe).")
    unknown = unknown_ids(plan)
    if unknown:
        lines.append("Not in the registry yet (skipped): " + ", ".join(unknown))
    return "\n".join(lines)


def _split_only(values) -> List[str]:
    return [v for value in values or [] for v in value.split(",") if v]


def cli(argv: Sequence[str], run_main: RunMain) -> int:
    """``python -m fabricpc.bench queue ...``. Returns the exit code."""
    args = _parser().parse_args(list(argv))
    try:
        plan = load_plan(args.plan)
    except PlanError as e:
        print(f"queue: {e}", file=sys.stderr)
        return 2
    if args.validate_plan:
        unknown = unknown_ids(plan)
        for job_id in unknown:
            print(f"unknown id: {job_id} (not a row or family in the registry)")
        print(f"plan '{plan.name}': {len(plan.jobs)} jobs, {len(unknown)} unknown")
        return 1 if unknown else 0
    if args.dry_run:
        print(dry_run(plan, args.out, args.probe))
        return 0
    if not args.out:
        print("queue: --out is required (where results go)", file=sys.stderr)
        return 2
    only = _split_only(args.only)
    missing = [i for i in only if i not in {j.id for j in plan.jobs}]
    if missing:
        print(f"queue: --only names ids not in the plan: {missing}", file=sys.stderr)
        return 2
    return _run_with_signals(
        lambda: run_queue(
            plan,
            args.out,
            run_main=run_main,
            only=only,
            continue_on_failure=args.continue_on_failure,
            plan_path=args.plan,
            in_process=args.in_process,
            warmup=args.warmup,
            timed=args.timed,
        )
    )


def _run_with_signals(go: Callable[[], int]) -> int:
    """Treat a plain ``kill`` (SIGTERM) like Ctrl-C, so the status file
    records the interruption. A restart then picks up where it stopped."""

    def on_term(signum, frame):
        raise KeyboardInterrupt

    old = None
    if threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGTERM, on_term)
    try:
        return go()
    except KeyboardInterrupt:
        print(
            "queue: interrupted; run the same command again to resume", file=sys.stderr
        )
        return 130
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
