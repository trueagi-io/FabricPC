"""Run each trial in its own Python process.

A fresh process gets a fresh JAX memory pool, so a trial's peak memory is
its own and not left over from the trial before it. It also means a trial
that crashes hard (out of memory, a segfault in a device driver) cannot
take the other trials down with it.

The child is just the normal command line with ``--trial i --in-process``:
it trains one trial and writes ``trial<i>.json`` like any other run. The
parent only has to start it and read that file back. If the child dies
before writing it, the parent writes a failed result with the exit code
and the end of the child's error output, so the run still records what
happened.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from fabricpc.bench.definitions.registry import BenchmarkRow
from fabricpc.bench.run.runner import TrialResult, seed_for_trial

_STDERR_KEPT = 4000  # characters from the end of a dead child's stderr


def child_command(
    row_id: str,
    trial: int,
    out_dir,
    *,
    num_epochs: Optional[float] = None,
    warmup_steps: int = 5,
    timed_steps: int = 30,
    zoo_dir=None,
    curve_batches: Optional[int] = None,
    diagnostics: bool = True,
) -> List[str]:
    """The command that runs one trial of one row in a new process."""
    cmd = [sys.executable, "-m", "fabricpc.bench", row_id]
    cmd += ["--trial", str(trial), "--in-process", "--out", str(out_dir)]
    cmd += ["--warmup", str(warmup_steps), "--timed", str(timed_steps)]
    if num_epochs is not None:
        cmd += ["--epochs", str(float(num_epochs))]
    if zoo_dir is not None:
        cmd += ["--zoo", str(zoo_dir)]
    if curve_batches is not None:
        cmd += ["--curve-batches", str(curve_batches)]
    if not diagnostics:
        cmd += ["--no-diagnostics"]
    return cmd


def run_trial_in_child(
    row: BenchmarkRow,
    trial: int,
    out_dir,
    *,
    num_epochs: Optional[float] = None,
    warmup_steps: int = 5,
    timed_steps: int = 30,
    zoo_dir=None,
    curve_batches: Optional[int] = None,
    diagnostics: bool = True,
) -> TrialResult:
    """Run one trial in a new process and return what it wrote."""
    path = Path(out_dir) / row.id / f"trial{trial}.json"
    # A file left from an earlier run must not pass for this one's result.
    path.unlink(missing_ok=True)

    cmd = child_command(
        row.id,
        trial,
        out_dir,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        zoo_dir=zoo_dir,
        curve_batches=curve_batches,
        diagnostics=diagnostics,
    )
    # stdout passes straight through so training progress still shows.
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)

    if path.exists():
        return TrialResult(**json.loads(path.read_text()))

    epochs = num_epochs if num_epochs is not None else row.train_config["num_epochs"]
    result = TrialResult(
        row_id=row.id,
        trial=trial,
        seed=seed_for_trial(trial),
        algorithm=row.algorithm,
        n_params=0,
        num_epochs=float(epochs),
        status="failed",
        error=(
            f"the trial's process ended with exit code {proc.returncode} "
            f"before writing a result. End of its error output:\n"
            f"{(proc.stderr or '')[-_STDERR_KEPT:]}"
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(result), indent=2))
    return result
