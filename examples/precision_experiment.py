"""
Precision-weighting ablation driver for the ResNet-18 CIFAR-10 PC demo.

Runs the demo across seeds for a control arm ("regular" PC, --precision none) and
one or more treatment arms (e.g. diag_probe), captures each run's test accuracy and
final train energy, and builds a *paired* statistical comparison of every treatment
against the control.

Why paired: each seed fixes both the model init and the train-batch order, so the
per-seed delta (treatment - control at the SAME seed) cancels that shared variance
and isolates the precision effect. The summary reports mean +/- std per arm, the
mean paired delta, and a paired t-test p-value (when scipy is available).

Each cell is checkpointed to the CSV as soon as it finishes, so a crash partway
through a long sweep loses nothing; rerun with --resume to skip completed cells.

Usage (from the repo root):
    PYTHONPATH=. python examples/precision_experiment.py \
        --seeds 0 1 2 --num_epochs 30 --activation tanh --eval_every 5

    # quick end-to-end smoke test of the harness itself (2 epochs):
    PYTHONPATH=. python examples/precision_experiment.py --seeds 0 1 --num_epochs 2 --no_augment

    # resume an interrupted sweep (skips cells already in the CSV):
    PYTHONPATH=. python examples/precision_experiment.py --seeds 0 1 2 --num_epochs 30 --activation tanh --resume

Outputs:
    precision_results.csv  — one row per run (raw, append-only, resume source)
    precision_results.md   — paired summary table, ready to paste into the PR / report
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time

# Matches the demo's periodic eval line, e.g. "  Epoch 5: accuracy=42.13%"
_EVAL_RE = re.compile(r"Epoch\s+(\d+):\s*accuracy=([\d.]+)%")

try:
    from scipy import stats as scipy_stats
except Exception:  # scipy optional
    scipy_stats = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(REPO_ROOT, "examples", "resnet18_cifar10_demo.py")

CSV_FIELDS = [
    "seed",
    "precision",
    "optimizer",
    "momentum",
    "activation",
    "num_epochs",
    "batch_size",
    "accuracy",  # fraction in [0, 1]
    "final_train_energy",
    "train_time_s",
]


# ---------------------------------------------------------------------------
# Running one cell (one seed x one arm) as an isolated subprocess
# ---------------------------------------------------------------------------


def _run_once(cmd, env):
    """Run the demo subprocess once, streaming output live.

    Returns (result|None, exit_code, evals) where `evals` is a list of
    (epoch:int, accuracy_pct:float) parsed from the demo's periodic eval lines, so a
    crash mid-cell still yields the accuracy trajectory captured up to the crash."""
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    result = None
    evals = []
    for line in proc.stdout:
        sys.stdout.write(line)  # live progress (tqdm, prints)
        sys.stdout.flush()
        if line.startswith("[RESULT] "):
            try:
                result = json.loads(line[len("[RESULT] ") :])
            except json.JSONDecodeError:
                pass
        else:
            m = _EVAL_RE.search(line)
            if m:
                evals.append((int(m.group(1)), float(m.group(2))))
    proc.wait()
    if proc.stdout is not None:
        proc.stdout.close()  # release the pipe fd promptly (this runs in a retry loop)
    return result, proc.returncode, evals


def append_progress(path, seed, precision, optimizer, momentum, attempt, evals):
    """Append per-epoch eval rows to a progress CSV so the within-cell accuracy
    trajectory survives a crash. The `attempt` column disambiguates retries (a retried
    cell re-emits the same epochs); `optimizer`/`momentum` keep distinct sweeps' (e.g.
    momentum 0.0 vs 0.9) trajectories separable if they share the same --out."""
    if not evals:
        return
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "seed",
                    "precision",
                    "optimizer",
                    "momentum",
                    "attempt",
                    "epoch",
                    "accuracy_pct",
                ]
            )
        for epoch, acc in evals:
            w.writerow([seed, precision, optimizer, momentum, attempt, epoch, acc])


def run_cell(seed, precision, args):
    """Run one cell with up to `args.retries` attempts; return the parsed [RESULT] dict
    or None if every attempt failed. Retries guard against TRANSIENT failures (e.g. the
    intermittent cuDNN sublibrary mismatch / a momentarily busy GPU) — a systematic error
    will still fail all attempts and be recorded as failed."""
    cmd = [
        sys.executable,
        DEMO,
        "--seed",
        str(seed),
        "--precision",
        precision,
        "--optimizer",
        args.optimizer,
        "--momentum",
        str(args.momentum),
        "--num_epochs",
        str(args.num_epochs),
        "--activation",
        args.activation,
        "--batch_size",
        str(args.batch_size),
        "--infer_steps",
        str(args.infer_steps),
        "--eta_infer",
        str(args.eta_infer),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--eval_every",
        str(args.eval_every),
    ]
    if args.no_augment:
        cmd.append("--no_augment")

    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("\n" + "=" * 70)
    print(
        f">>> RUN  seed={seed}  precision={precision}  "
        f"epochs={args.num_epochs}  act={args.activation}"
    )
    print("=" * 70, flush=True)

    progress_path = os.path.splitext(args.out)[0] + ".progress.csv"

    attempts = max(1, args.retries)
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            print(
                f">>> retry {attempt}/{attempts} for seed={seed} precision={precision} "
                f"(sleeping {args.retry_sleep}s first)",
                flush=True,
            )
            if args.retry_sleep > 0:
                time.sleep(args.retry_sleep)
        result, rc, evals = _run_once(cmd, env)
        # Persist the per-epoch trajectory even if the cell ultimately fails, so a
        # mid-cell crash still leaves the accuracy curve up to the crash on disk.
        append_progress(
            progress_path,
            seed,
            precision,
            args.optimizer,
            args.momentum,
            attempt,
            evals,
        )
        if result is not None:
            return result
        print(
            f"!!! attempt {attempt}/{attempts} seed={seed} precision={precision} "
            f"produced no [RESULT] (exit {rc})",
            file=sys.stderr,
        )

    print(
        f"!!! seed={seed} precision={precision} FAILED after {attempts} attempt(s) "
        f"— recording as failed.",
        file=sys.stderr,
    )
    return None


# ---------------------------------------------------------------------------
# CSV checkpointing
# ---------------------------------------------------------------------------


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def already_done(rows, seed, precision, args):
    for r in rows:
        # Old CSVs predate the optimizer/momentum columns -> treat missing as adamw / 0.0.
        if (
            int(r["seed"]) == seed
            and r["precision"] == precision
            and r.get("optimizer", "adamw") == args.optimizer
            and float(r.get("momentum") or 0.0) == args.momentum
            and int(r["num_epochs"]) == args.num_epochs
            and r["activation"] == args.activation
            and r["accuracy"] not in ("", "nan")
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    # sample standard deviation (ddof=1)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_ttest(treat, ctrl):
    """Return (mean_delta, std_delta, t_stat, p_value, n). p/t are nan without scipy."""
    deltas = [t - c for t, c in zip(treat, ctrl)]
    n = len(deltas)
    md, sd = mean(deltas), std(deltas)
    t = p = float("nan")
    if scipy_stats is not None and n >= 2:
        t, p = scipy_stats.ttest_rel(treat, ctrl)
    elif n >= 2 and sd > 0:
        t = md / (sd / math.sqrt(n))  # p left nan (no t-distribution without scipy)
    return md, sd, float(t), float(p), n


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------


def build_report(rows, args):
    """Build the paired markdown report from CSV rows matching this config."""
    # index: (seed, precision) -> {accuracy, final_train_energy}
    cells = {}
    for r in rows:
        # Filter to this sweep's config; old CSVs lack optimizer/momentum -> adamw / 0.0.
        if (
            int(r["num_epochs"]) != args.num_epochs
            or r["activation"] != args.activation
            or r.get("optimizer", "adamw") != args.optimizer
            or float(r.get("momentum") or 0.0) != args.momentum
        ):
            continue
        if r["accuracy"] in ("", "nan"):
            continue
        cells[(int(r["seed"]), r["precision"])] = {
            "acc": float(r["accuracy"]) * 100.0,
            "energy": float(r["final_train_energy"]),
        }

    control = args.control
    treatments = [a for a in args.arms if a != control]
    seeds = sorted({s for (s, p) in cells})

    lines = []
    lines.append(f"# Precision weighting ablation — ResNet-18 / CIFAR-10 (PC + muPC)\n")
    opt_desc = args.optimizer + (
        f" (momentum {args.momentum})"
        if args.optimizer == "ngd" and args.momentum
        else ""
    )
    lines.append(
        f"- Config: **{args.num_epochs} epochs**, optimizer **{opt_desc}**, "
        f"activation **{args.activation}**, batch {args.batch_size}, "
        f"infer_steps {args.infer_steps}, lr {args.lr}, "
        f"augment={'off' if args.no_augment else 'on'}"
    )
    lines.append(f"- Control arm: `{control}` (regular PC, isotropic precision Pi=1)")
    lines.append(f"- Seeds: {seeds}  (paired: each seed fixes init + batch order)\n")

    for treat in treatments:
        paired_seeds = [
            s for s in seeds if (s, control) in cells and (s, treat) in cells
        ]
        if not paired_seeds:
            lines.append(f"## `{treat}` vs `{control}` — no paired seeds yet\n")
            continue

        ctrl_acc = [cells[(s, control)]["acc"] for s in paired_seeds]
        treat_acc = [cells[(s, treat)]["acc"] for s in paired_seeds]
        ctrl_en = [cells[(s, control)]["energy"] for s in paired_seeds]
        treat_en = [cells[(s, treat)]["energy"] for s in paired_seeds]

        lines.append(f"## `{treat}` vs `{control}`\n")
        lines.append(
            "| seed | acc {ctrl} (%) | acc {tr} (%) | Δacc (pts) | "
            "energy {ctrl} | energy {tr} | Δenergy |".format(ctrl=control, tr=treat)
        )
        lines.append("|---|---|---|---|---|---|---|")
        for s in paired_seeds:
            a_c, a_t = cells[(s, control)]["acc"], cells[(s, treat)]["acc"]
            e_c, e_t = cells[(s, control)]["energy"], cells[(s, treat)]["energy"]
            lines.append(
                f"| {s} | {a_c:.2f} | {a_t:.2f} | {a_t - a_c:+.2f} | "
                f"{e_c:.4f} | {e_t:.4f} | {e_t - e_c:+.4f} |"
            )

        md_a, sd_a, t_a, p_a, n = paired_ttest(treat_acc, ctrl_acc)
        md_e, sd_e, _, _, _ = paired_ttest(treat_en, ctrl_en)

        lines.append("")
        lines.append(
            f"**Accuracy** — {control}: {mean(ctrl_acc):.2f} ± {std(ctrl_acc):.2f} | "
            f"{treat}: {mean(treat_acc):.2f} ± {std(treat_acc):.2f}"
        )
        lines.append(
            f"**Paired Δacc**: {md_a:+.2f} ± {sd_a:.2f} pts over n={n} seeds"
            + (
                f"  (paired t={t_a:.2f}, p={p_a:.4f})"
                if not math.isnan(p_a)
                else f"  (t={t_a:.2f}; install scipy for p-value)"
            )
        )
        lines.append(
            f"**Paired Δenergy**: {md_e:+.4f} ± {sd_e:.4f} "
            f"(negative = lower/better energy)\n"
        )

        verdict = "FAVORS precision" if md_a > 0 else "favors control / no gain"
        sig = (
            ""
            if math.isnan(p_a)
            else (
                " — significant at 0.05" if p_a < 0.05 else " — NOT significant at 0.05"
            )
        )
        lines.append(
            f"> Verdict: mean paired Δacc {md_a:+.2f} pts ({verdict}){sig}. "
            f"n={n} is small — treat p-values as indicative, not definitive.\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Precision-weighting paired ablation driver"
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=["none", "diag_probe"],
        help="Arms to run; the control must be among them.",
    )
    p.add_argument(
        "--control",
        type=str,
        default="none",
        help="Control arm every treatment is compared against (default: none)",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "ngd"],
        help="Optimizer for ALL arms in this sweep. 'ngd' = SGD so "
        "precision acts as the per-channel adaptive LR (use with "
        "--arms none online for the faithful NGD test). Default adamw. "
        "Recorded per-row; the report keys on it so adamw/ngd rows in "
        "the same CSV are never conflated.",
    )
    p.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="SGD momentum passed to every cell (only affects --optimizer ngd; "
        "0.0 = plain SGD, try 0.9). Default 0.0.",
    )
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument(
        "--activation",
        type=str,
        default="tanh",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--infer_steps", type=int, default=80)
    p.add_argument("--eta_infer", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument(
        "--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES for child runs"
    )
    p.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max attempts per cell; retries guard transient failures "
        "(e.g. intermittent cuDNN mismatch / busy GPU). Default 2.",
    )
    p.add_argument(
        "--retry_sleep",
        type=float,
        default=10.0,
        help="Seconds to wait before each retry (lets a transient GPU/cuDNN "
        "state clear). Default 10.",
    )
    p.add_argument("--out", type=str, default="precision_results.csv")
    p.add_argument("--md", type=str, default="precision_results.md")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (seed, arm) cells already present in --out",
    )
    p.add_argument(
        "--report_only",
        action="store_true",
        help="Skip running; just rebuild the markdown report from --out",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.control not in args.arms:
        sys.exit(f"--control '{args.control}' must be one of --arms {args.arms}")

    if not args.report_only:
        existing = load_csv(args.out)
        # Run control first per seed, then treatments (so partial reports are paired).
        ordered_arms = [args.control] + [a for a in args.arms if a != args.control]
        for seed in args.seeds:
            for arm in ordered_arms:
                if args.resume and already_done(existing, seed, arm, args):
                    print(
                        f"--resume: skipping seed={seed} precision={arm} (already done)"
                    )
                    continue
                result = run_cell(seed, arm, args)
                if result is None:
                    result = {  # record the failure so it's visible / resumable
                        "seed": seed,
                        "precision": arm,
                        "optimizer": args.optimizer,
                        "momentum": args.momentum,
                        "activation": args.activation,
                        "num_epochs": args.num_epochs,
                        "batch_size": args.batch_size,
                        "accuracy": "nan",
                        "final_train_energy": "nan",
                        "train_time_s": "nan",
                    }
                append_csv(args.out, result)

    rows = load_csv(args.out)
    report = build_report(rows, args)
    with open(args.md, "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nRaw results: {args.out}   Summary table: {args.md}")


if __name__ == "__main__":
    main()
