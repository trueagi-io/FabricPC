"""Command line for the benchmark suite.

    python -m fabricpc.bench list
    python -m fabricpc.bench <row-id> [--trials N] [--epochs E] [--out DIR]
                                      [--resume] [--zoo DIR]
    python -m fabricpc.bench <row-id> --dry-run
    python -m fabricpc.bench <family>             (all three rows, then compare)
    python -m fabricpc.bench compare <row-a> <row-b> [--metric accuracy]
    python -m fabricpc.bench compare <family>     (e.g. mnist-mlp)
    python -m fabricpc.bench smoke [--floor 0.85]
    python -m fabricpc.bench regress [row-id ...] [--out DIR]
    python -m fabricpc.bench validate [DIR]
    python -m fabricpc.bench refresh <old-dir> --out <new-dir> [--zoo DIR]
    python -m fabricpc.bench probe <row|family> [--price-per-hour P]
    python -m fabricpc.bench report <results_root> [--out report.md]
                                                   [--format md|html]
    python -m fabricpc.bench queue <plan.json> --out DIR [--dry-run]
                                   [--only IDS] [--continue-on-failure]

Running a row writes one JSON file per trial, a manifest, ``trials.csv``
(every trial on one line), and (with two or more good trials) a summary
with the mean and standard error. ``validate`` checks that a results
folder is complete and consistent. ``probe`` times a few steps and
estimates a full run's time and cost (see ``fabricpc.bench.probe``).
``report`` writes one readable page for a whole results folder (see
``fabricpc.bench.report``). ``queue`` works through a plan file of rows
and families, one after another, and can be restarted (see
``fabricpc.bench.queue``). ``smoke`` is the short run CI uses: a small
row, two seeds, half an epoch, and a check that accuracy is still above a
floor. ``regress`` is the stronger CI check: tiny MLP, autoencoder and
Hopfield runs with all three algorithms, each held to a fixed line (see
``fabricpc.bench.regress``).

A full run (the row's own epochs and seed count) is also checked against
the row's expected score, when it has one, and prints PASS or FAIL. A FAIL
makes the command exit non-zero.

A run replaces that row's earlier results in the same ``--out`` folder, so
old trials never mix with new ones; use another folder to keep them.
``--resume`` skips trials that already finished with the same epoch count,
so a run cut off by a cloud session limit can pick up where it stopped.
``--zoo DIR`` saves each trial's trained weights there.

``refresh`` fixes numbers that older code worked out wrong (the FLOP count,
cloud checkpoint paths) in a copy of a results folder, without retraining;
see ``fabricpc.bench.refresh``.

Each trial runs in its own Python process (see ``fabricpc.bench.isolate``).
``--in-process`` runs them all in this one instead, which is handy for
debugging; ``--trial i`` runs a single trial and is what each child does.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from fabricpc.bench.stats.band import attach_band
from fabricpc.bench.run.isolate import run_trial_in_child
from fabricpc.bench.run.manifest import describe_row, write_manifest
from fabricpc.bench.stats.compare import compare_family
from fabricpc.bench.results.refresh import refresh_results
from fabricpc.bench.definitions.registry import COMPARISONS, ROWS
from fabricpc.bench.run.runner import finished_trial, run_trial, seed_for_trial
from fabricpc.bench.stats.summary import MIN_TRIALS, compare_rows, summarize_row
from fabricpc.bench.results.writer import validate, write_trials_csv

SMOKE_ROW = "mnist-mlp-spc"
SMOKE_FLOOR = 0.85  # half an epoch of MNIST lands well above this


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fabricpc.bench")
    p.add_argument(
        "target",
        help="'list', 'compare', 'smoke', 'regress', 'validate', 'refresh', "
        "'probe', 'queue', 'report', or a row id like mnist-mlp-spc",
    )
    p.add_argument(
        "rows",
        nargs="*",
        help="for 'compare': the two row ids; for 'validate' and 'refresh': "
        "the results folder",
    )
    p.add_argument("--dry-run", action="store_true", help="show the row, do not train")
    p.add_argument("--out", default="results", help="where result files go")
    p.add_argument("--trials", type=int, help="how many seeds to run")
    p.add_argument("--epochs", type=float, help="override the row's epoch count")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps")
    p.add_argument("--timed", type=int, default=30, help="timed steps")
    p.add_argument(
        "--metric",
        help="metric for 'compare' (default: the family's own, e.g. perplexity "
        "for text models, else accuracy)",
    )
    p.add_argument(
        "--format", choices=["md", "html"], help="for 'report' (default: md)"
    )
    p.add_argument("--row", default=SMOKE_ROW, help="row for 'smoke'")
    p.add_argument("--floor", type=float, default=SMOKE_FLOOR, help="min accuracy")
    p.add_argument(
        "--resume", action="store_true", help="skip trials that already finished"
    )
    p.add_argument(
        "--zoo",
        help="save each trial's trained weights in this folder "
        "(for 'refresh': where the old run's zoo is, if not next to it)",
    )
    p.add_argument("--trial", type=int, help="run only this one trial")
    p.add_argument(
        "--curve-batches",
        type=int,
        help="test batches scored after every epoch for the learning curve "
        "(default 10; 0 turns the curve off)",
    )
    p.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="skip the backprop-likeness and stiffness checks at start and end",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="quicker runs: no learning curve and no diagnostics "
        "(same training, same scores)",
    )
    p.add_argument(
        "--in-process",
        action="store_true",
        help="run trials in this process instead of one process each",
    )
    p.add_argument(
        "--price-per-hour",
        type=float,
        help="for 'probe': machine price in USD per hour (default 1.06, a T4)",
    )
    return p


def _check_numbers(args) -> list:
    """Return one message per bad number flag, so all of them show at once.

    Runs before any command, so a bad value never starts training or clears
    old results. A plain ``x <= 0`` check lets nan through, so each rule says
    what a good value is, and float flags also need ``math.isfinite``.
    """
    problems = []

    def need(flag, value, ok, rule):
        if value is not None and not ok(value):
            problems.append(f"{flag} must be {rule}, got {value}")

    need("--trials", args.trials, lambda v: v >= 1, "1 or more")
    need("--epochs", args.epochs, lambda v: v > 0 and math.isfinite(v), "above 0")
    need("--warmup", args.warmup, lambda v: v >= 0, "0 or more")
    need("--timed", args.timed, lambda v: v >= 1, "1 or more")
    need("--trial", args.trial, lambda v: v >= 0, "0 or more")
    need("--curve-batches", args.curve_batches, lambda v: v >= 0, "0 or more")
    need("--floor", args.floor, lambda v: 0 <= v <= 1, "between 0 and 1")
    need(
        "--price-per-hour",
        args.price_per_hour,
        lambda v: v > 0 and math.isfinite(v),
        "above 0",
    )
    return problems


def _unknown_row(row_id: str) -> int:
    print(
        f"Unknown row '{row_id}'. Run 'python -m fabricpc.bench list' "
        "to see the available rows.",
        file=sys.stderr,
    )
    return 2


def _clear_old_results(out, row_id, *, keep_trials):
    """Remove a row's earlier results, except trial numbers in ``keep_trials``.

    Without this, a 2-seed run into a folder that held a 5-seed run would be
    summarized with 3 stale trials mixed in. Other rows are not touched.
    """
    row_dir = Path(out) / row_id
    if not row_dir.is_dir():
        return
    for path in row_dir.glob("trial*.json"):
        if int(path.stem[5:]) not in keep_trials:
            path.unlink()
    for name in ("summary.json", "trials.csv"):
        (row_dir / name).unlink(missing_ok=True)


def _run_row(
    row,
    out,
    *,
    n_trials,
    epochs,
    warmup,
    timed,
    command,
    resume=False,
    zoo=None,
    in_process=False,
    curve_batches=None,
    diagnostics=True,
):
    """Run every trial of one row, then summarize. Returns (failed, summary)."""
    run_one = run_trial if in_process else run_trial_in_child
    extra = {} if curve_batches is None else {"curve_batches": curve_batches}
    if not diagnostics:
        extra["diagnostics"] = False
    _clear_old_results(out, row.id, keep_trials=range(n_trials) if resume else ())
    write_manifest(
        out,
        row,
        seeds=[seed_for_trial(t) for t in range(n_trials)],
        command=command,
    )
    wanted_epochs = epochs if epochs is not None else row.train_config["num_epochs"]
    failed = 0
    for trial in range(n_trials):
        if resume and finished_trial(out, row.id, trial, wanted_epochs):
            print(f"{row.id} trial {trial}: already done, skipping")
            continue
        result = run_one(
            row,
            trial,
            out,
            num_epochs=epochs,
            warmup_steps=warmup,
            timed_steps=timed,
            zoo_dir=zoo,
            **extra,
        )
        if result.status == "ok":
            acc = result.metrics.get("accuracy")
            shown = f"accuracy={acc:.4f}" if acc is not None else ""
            print(
                f"{row.id} trial {trial}: {shown} step={result.step_time_ms:.2f}ms "
                f"train={result.train_time_s:.1f}s"
            )
        else:
            failed += 1
            print(f"{row.id} trial {trial}: FAILED (see {out})", file=sys.stderr)
        # After every trial, so a run cut off by a time limit is still tidy.
        write_trials_csv(out, row.id)

    write_trials_csv(out, row.id)  # also covers a run where every trial was skipped
    summary = None
    if n_trials - failed >= MIN_TRIALS:
        summary = attach_band(out, row, summarize_row(out, row.id))
        acc = summary.metrics.get("accuracy")
        if acc:
            print(
                f"{row.id}: accuracy {acc['mean']:.4f} ± {acc['se']:.4f} SE, "
                f"95% CI [{acc['ci95_low']:.4f}, {acc['ci95_high']:.4f}] "
                f"(n={acc['n']}), step {summary.timing['step_time_ms']['mean']:.2f}ms"
            )
        band = summary.band
        label = {"pass": "PASS", "fail": "FAIL"}.get(band["status"], "no verdict")
        pending = (
            " (band rule pending sponsor sign-off)" if band["pending_signoff"] else ""
        )
        print(f"{row.id}: band {label}: {band['reason']}{pending}")
    return failed, summary


def cmd_list() -> int:
    for row_id in ROWS:
        print(row_id)
    print("\nFamilies (run all three rows, then spc vs backprop, epc vs backprop,")
    print("epc vs spc):")
    for family_id in COMPARISONS:
        print(family_id)
    return 0


def _print_contrast(a, b, metric, c) -> None:
    sig = "yes" if c["significant_at_05"] else "no"
    print(
        f"{a} - {b} on {metric}: {c['mean_diff']:+.4f} "
        f"(se {c['se_diff']:.4f}, "
        f"95% CI [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}], "
        f"p={c['p_value']:.3g}, d={c['cohens_d']:.2f}, "
        f"n={c['n']}, significant at 5%: {sig})"
    )


def _compare_family(family_id, out, metric) -> int:
    result = compare_family(out, COMPARISONS[family_id], metric=metric)
    for c in result["contrasts"]:
        _print_contrast(c["arm_a"], c["arm_b"], metric, c)
    return 0


def cmd_compare(args) -> int:
    if len(args.rows) == 1 and args.rows[0] in COMPARISONS:
        family = COMPARISONS[args.rows[0]]
        return _compare_family(family.id, args.out, args.metric or family.metric)
    if len(args.rows) != 2:
        print(
            "usage: compare <row-a> <row-b> | compare <family> [--metric NAME]",
            file=sys.stderr,
        )
        return 2
    row_a, row_b = args.rows
    for row_id in (row_a, row_b):
        if row_id not in ROWS:
            return _unknown_row(row_id)
    metric = args.metric or ROWS[row_a].metric
    c = compare_rows(args.out, row_a, row_b, metric=metric)
    _print_contrast(row_a, row_b, metric, asdict(c))
    return 0


def cmd_validate(args) -> int:
    folder = args.rows[0] if args.rows else args.out
    problems = validate(folder)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"validate: {len(problems)} problem(s) in {folder}", file=sys.stderr)
        return 1
    print(f"validate: {folder} is complete and consistent")
    return 0


def cmd_refresh(args, command) -> int:
    """Write a corrected copy of an old results folder, then validate it."""
    # --out defaults to "results" for the run commands. Refresh must be told
    # where to write, or it could land on a real run in ./results.
    typed_out = any(a == "--out" or a.startswith("--out=") for a in command[3:])
    if len(args.rows) != 1 or not typed_out:
        print("usage: refresh <old-results-dir> --out <new-dir>", file=sys.stderr)
        return 2
    try:
        reports = refresh_results(args.rows[0], args.out, zoo=args.zoo, command=command)
    except ValueError as e:
        print(f"refresh: {e}", file=sys.stderr)
        return 1
    for r in reports:
        old, new = r["achieved_tflops_old"], r["achieved_tflops_new"]
        if new is not None:
            print(f"{r['row_id']}: {old:.3f} -> {new:.3f} TFLOP/s (n={r['n_ok']})")
    args.rows = [args.out]
    return cmd_validate(args)


def cmd_report(args, argv) -> int:
    """Write one page about a results root. Without --out, print it."""
    from fabricpc.bench.results.report import run_report

    if len(args.rows) != 1:
        print(
            "usage: report <results_root> [--out FILE] [--format md|html]",
            file=sys.stderr,
        )
        return 2
    # --out has a default ("results") for the run commands; only use it here
    # when the user actually typed it.
    typed_out = any(a == "--out" or a.startswith("--out=") for a in argv)
    out = args.out if typed_out else None
    text = run_report(args.rows[0], out=out, fmt=args.format)
    print(f"report: wrote {out}" if out else text, end="\n" if out else "")
    return 0


def cmd_smoke(args, command) -> int:
    """A short real run that proves the whole pipeline still works."""
    row = ROWS.get(args.row)
    if row is None:
        return _unknown_row(args.row)
    failed, summary = _run_row(
        row,
        args.out,
        n_trials=2,
        epochs=args.epochs if args.epochs is not None else 0.5,
        warmup=2,
        timed=5,
        command=command,
        in_process=args.in_process,
    )
    if failed or summary is None:
        print("smoke: a trial failed", file=sys.stderr)
        return 1
    mean = summary.metrics["accuracy"]["mean"]
    if mean < args.floor:
        print(
            f"smoke: accuracy {mean:.4f} is below the floor {args.floor}",
            file=sys.stderr,
        )
        return 1
    print(f"smoke: ok (accuracy {mean:.4f} >= {args.floor})")
    return 0


def cmd_regress(args, command) -> int:
    """Tiny fixed-seed runs held to fixed lines (fabricpc.bench.regress)."""
    from fabricpc.bench.cli import regress

    try:
        checks = regress.select(regress.CI_CHECKS, args.rows)
    except KeyError as e:
        return _unknown_row(e.args[0])

    def run_row(check):
        return _run_row(
            ROWS[check.row_id],
            args.out,
            n_trials=check.trials,
            epochs=check.epochs,
            warmup=2,
            timed=5,
            command=command,
            in_process=args.in_process,
        )

    return regress.run_regress(args.out, run_row, checks)


def cmd_one_trial(row, args) -> int:
    """Run a single trial here and write its file. This is what a child runs."""
    result = run_trial(
        row,
        args.trial,
        args.out,
        num_epochs=args.epochs,
        warmup_steps=args.warmup,
        timed_steps=args.timed,
        zoo_dir=args.zoo,
        diagnostics=not args.no_diagnostics,
        **({} if args.curve_batches is None else {"curve_batches": args.curve_batches}),
    )
    return 0 if result.status == "ok" else 1


def cmd_family(args, command) -> int:
    """Run every row of a family, then compare them."""
    family = COMPARISONS[args.target]
    codes = [_run_one_row(ROWS[row_id], args, command) for row_id in family.rows]
    try:
        _compare_family(family.id, args.out, args.metric or family.metric)
    except ValueError as e:  # too few good trials to pair
        print(f"{family.id}: no comparison: {e}", file=sys.stderr)
        return 1
    return max(codes)


def cmd_run(args, command) -> int:
    if args.target in COMPARISONS and args.target not in ROWS:
        return cmd_family(args, command)
    row = ROWS.get(args.target)
    if row is None:
        return _unknown_row(args.target)
    if args.dry_run:
        print(json.dumps(describe_row(row), indent=2))
        return 0
    if args.trial is not None:
        return cmd_one_trial(row, args)
    return _run_one_row(row, args, command)


def _run_one_row(row, args, command) -> int:
    n_trials = args.trials if args.trials is not None else row.n_trials
    failed, summary = _run_row(
        row,
        args.out,
        n_trials=n_trials,
        epochs=args.epochs,
        warmup=args.warmup,
        timed=args.timed,
        command=command,
        resume=args.resume,
        zoo=args.zoo,
        in_process=args.in_process,
        curve_batches=args.curve_batches,
        diagnostics=not args.no_diagnostics,
    )
    band_failed = summary is not None and summary.band["status"] == "fail"
    return 1 if failed or band_failed else 0


def cmd_probe(args, argv) -> int:
    """Estimate a row's or family's run time and cost from a few steps."""
    from fabricpc.bench.run import probe

    target = args.rows[0] if args.rows else None
    if target not in ROWS and target not in COMPARISONS:
        return _unknown_row(target)
    # --warmup, --timed and --out have defaults meant for real runs; the
    # probe only uses them when they were typed.
    typed = {a.split("=")[0] for a in argv}
    kwargs = {"trials": args.trials, "epochs": args.epochs}
    if "--warmup" in typed:
        kwargs["warmup"] = args.warmup
    if "--timed" in typed:
        kwargs["timed"] = args.timed
    if args.price_per_hour is not None:
        kwargs["price_per_hour"] = args.price_per_hour
    report = probe.probe(target, **kwargs)
    print(probe.format_probe(report))
    if "--out" in typed:
        print(f"wrote {probe.write_probe(args.out, report)}")
    return 0


def main(argv=None) -> int:
    if (sys.argv[1:] if argv is None else list(argv))[:1] == ["queue"]:
        from fabricpc.bench.cli.queue import cli as queue_cli  # has its own flags

        return queue_cli((sys.argv[1:] if argv is None else argv)[1:], run_main=main)
    parser = _parser()
    args = parser.parse_args(argv)
    problems = _check_numbers(args)
    if problems:
        parser.error("\n  ".join(["bad values, nothing was run:", *problems]))
    if args.fast:
        # Same training and scores; only the extras that cost time are skipped.
        args.no_diagnostics = True
        if args.curve_batches is None:
            args.curve_batches = 0
    command = [sys.executable, "-m", "fabricpc.bench", *(argv or sys.argv[1:])]
    if args.target == "list":
        return cmd_list()
    if args.target == "compare":
        return cmd_compare(args)
    if args.target == "smoke":
        return cmd_smoke(args, command)
    if args.target == "regress":
        return cmd_regress(args, command)
    if args.target == "validate":
        return cmd_validate(args)
    if args.target == "refresh":
        return cmd_refresh(args, command)
    if args.target == "probe":
        return cmd_probe(args, argv if argv is not None else sys.argv[1:])
    if args.target == "report":
        return cmd_report(args, command[3:])
    return cmd_run(args, command)


if __name__ == "__main__":
    sys.exit(main())
