"""Generate reports for ALL Optuna run directories under runs/.

Auto-discovers directories with ``optuna_journal.log``, detects whether each
is a transformer or Hopfield study, generates reports, and continues on
failure with a warning.

Usage::

    uv run python scripts/generate_all_glucose_reports.py
    uv run python scripts/generate_all_glucose_reports.py --format html
    uv run python scripts/generate_all_glucose_reports.py --runs-root runs
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Generate reports for all Optuna run directories.")

RUNS_ROOT = Path("runs")

TRANSFORMER_SCRIPT = "scripts/generate_glucose_tuning_report.py"
HOPFIELD_SCRIPT = "scripts/generate_glucose_hopfield_tuning_report.py"


def _detect_study_type(run_dir: Path) -> str:
    """Return 'transformer' or 'hopfield' based on study name or dir name."""
    config_path = run_dir / "coordinator_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            study_name = config.get("study_name", "")
            if "hopfield" in study_name.lower():
                return "hopfield"
            if "transformer" in study_name.lower():
                return "transformer"
        except (json.JSONDecodeError, OSError):
            pass
    dirname = run_dir.name.lower()
    if "hopfield" in dirname:
        return "hopfield"
    return "transformer"


def _discover_run_dirs(runs_root: Path) -> list[Path]:
    """Find all subdirectories with an Optuna journal."""
    if not runs_root.is_dir():
        return []
    dirs = sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and (d / "optuna_journal.log").exists()
    )
    return dirs


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    runs_root: Path = typer.Option(RUNS_ROOT, help="Parent directory containing run subdirectories."),
    format: str = typer.Option("all", "--format", help="Output format: md, html, json, or all."),
    timeout: int = typer.Option(300, help="Timeout per report in seconds."),
) -> None:
    """Discover all Optuna run directories and generate reports for each."""
    if ctx.invoked_subcommand is not None:
        return

    run_dirs = _discover_run_dirs(runs_root)
    if not run_dirs:
        typer.echo(f"No run directories with optuna_journal.log found under {runs_root}/")
        raise typer.Exit(1)

    typer.echo(f"Found {len(run_dirs)} Optuna run directories:")
    for d in run_dirs:
        study_type = _detect_study_type(d)
        config_path = d / "coordinator_config.json"
        study_name = "?"
        if config_path.exists():
            try:
                study_name = json.loads(config_path.read_text()).get("study_name", "?")
            except (json.JSONDecodeError, OSError):
                pass
        typer.echo(f"  {d.name:40s}  type={study_type:12s}  study={study_name}")
    typer.echo("")

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for run_dir in run_dirs:
        study_type = _detect_study_type(run_dir)
        script = HOPFIELD_SCRIPT if study_type == "hopfield" else TRANSFORMER_SCRIPT

        config_path = run_dir / "coordinator_config.json"
        study_name = None
        if config_path.exists():
            try:
                study_name = json.loads(config_path.read_text()).get("study_name")
            except (json.JSONDecodeError, OSError):
                pass

        cmd = [
            sys.executable, script,
            "--run-dir", str(run_dir),
            "--format", format,
        ]
        if study_name:
            cmd.extend(["--study-name", study_name])

        typer.echo(f"{'='*70}")
        typer.echo(f"Generating {study_type} report for {run_dir.name}")
        typer.echo(f"  command: {' '.join(cmd)}")
        typer.echo(f"{'='*70}")

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd(),
            )
            elapsed = time.monotonic() - t0

            if result.stdout:
                typer.echo(result.stdout.rstrip())
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else f"exit code {result.returncode}"
                typer.secho(
                    f"FAILED ({elapsed:.1f}s): {run_dir.name} — {error_msg[:200]}",
                    fg=typer.colors.RED,
                )
                failures.append((run_dir.name, error_msg[:200]))
            else:
                typer.secho(
                    f"OK ({elapsed:.1f}s): {run_dir.name}",
                    fg=typer.colors.GREEN,
                )
                successes.append(run_dir.name)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - t0
            msg = f"Timed out after {timeout}s"
            typer.secho(
                f"FAILED ({elapsed:.1f}s): {run_dir.name} — {msg}",
                fg=typer.colors.RED,
            )
            failures.append((run_dir.name, msg))
        except Exception as exc:
            elapsed = time.monotonic() - t0
            msg = f"{type(exc).__name__}: {exc}"
            typer.secho(
                f"FAILED ({elapsed:.1f}s): {run_dir.name} — {msg}",
                fg=typer.colors.RED,
            )
            failures.append((run_dir.name, msg))

        typer.echo("")

    typer.echo("=" * 70)
    typer.echo("SUMMARY")
    typer.echo("=" * 70)
    typer.secho(f"  Succeeded: {len(successes)}/{len(run_dirs)}", fg=typer.colors.GREEN)
    for name in successes:
        typer.echo(f"    ✓ {name}")
    if failures:
        typer.secho(f"  Failed:    {len(failures)}/{len(run_dirs)}", fg=typer.colors.RED)
        for name, reason in failures:
            typer.echo(f"    ✗ {name}: {reason}")
    typer.echo("")

    if failures:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
