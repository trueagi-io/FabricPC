"""Fix results that older code wrote, without training anything again.

Some numbers in a trial file are worked out from the run, not measured,
so when the code that works them out gets a bug fix the old files can be
corrected on a laptop in a few seconds:

* ``compute`` and ``achieved_tflops``. The FLOP count is rebuilt from the
  row's graph with today's ``fabricpc.bench.run.compute`` (the graph is code,
  so building it again costs nothing), and the TFLOP/s is that count over
  the step time the run measured. The settling steps and batch size the
  run used are kept, so the count describes the run that happened.
* ``checkpoint``. Older runs saved the absolute path on the cloud machine.
  When the zoo is on disk, the path is rewritten relative to the new
  results folder, the way a fresh run writes it.
* ``peak_memory_bytes``. Older runs only had the process-wide peak, which
  on GPU is mostly compile scratch space and comes out the same for every
  algorithm. The value is kept but marked as not comparable. We do not
  make up a ``step_memory``; that needs the run itself.

Everything the run measured (metrics, times, seeds) is copied unchanged.
The input folder is only read. Every trial and manifest gets a ``refresh``
list saying what changed, from what, to what, and with which commit, and
the manifest keeps the commit that did the training. ``summary.json``,
``trials.csv`` and every comparison the old folder had are rebuilt with the
normal code. A compare file that cannot be rebuilt is copied as it was and
listed in the new ``NOTES.md``, and the old ``NOTES.md`` is kept under a
line saying what was corrected. Every row is checked (known to the
registry, same weighted edges and parameter count, nothing already in the
output folder) before any file is written.
"""

import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import jax

import fabricpc
from fabricpc.bench.stats.band import attach_band
from fabricpc.bench.stats.compare import compare_family
from fabricpc.bench.run.compute import count_compute
from fabricpc.bench.results.manifest import _git_sha
from fabricpc.bench.definitions.registry import COMPARISONS, ROWS
from fabricpc.bench.stats.summary import MIN_TRIALS, compare_rows, summarize_row
from fabricpc.bench.results.writer import _trial_files, write_trials_csv

PEAK_MEMORY_NOTE = (
    "peak_memory_bytes is the process-wide peak, mostly GPU compile scratch "
    "space, and comes out about the same for every algorithm; do not compare "
    "methods on it. This run did not record step_memory."
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _find_rows(source: Path):
    """``(results root, row folders)`` for a results folder or one row folder."""
    if not source.is_dir():
        raise ValueError(f"{source}: not a folder")
    if any(source.glob("trial*.json")):
        return source.parent, [source]
    rows = sorted(
        d for d in source.iterdir() if d.is_dir() and any(d.glob("trial*.json"))
    )
    if not rows:
        raise ValueError(f"{source}: no benchmark results (no trial files) found")
    return source, rows


def _inside(a: Path, b: Path) -> bool:
    """True when ``a`` is ``b`` or somewhere under it."""
    return a == b or b in a.parents


def _refuse_overlap(source: Path, out: Path) -> None:
    src, dst = source.resolve(), out.resolve()
    if _inside(dst, src) or _inside(src, dst):
        raise ValueError(
            f"refresh writes to a new folder and never edits the old one; "
            f"{out} overlaps {source}"
        )


class _Recounter:
    """Builds each row's graph once and counts its FLOPs with today's code."""

    def __init__(self):
        self._built: Dict[str, tuple] = {}

    def count(
        self,
        row_id: str,
        algorithm: str,
        saved: dict,
        batch_size: int,
        n_params: Optional[int] = None,
    ):
        if row_id not in ROWS:
            raise ValueError(f"{row_id}: not a row this code knows, cannot recount")
        if row_id not in self._built:
            # Any key will do: only the shapes of the weights matter here.
            self._built[row_id] = ROWS[row_id].model_factory(jax.random.PRNGKey(0))
        params, structure = self._built[row_id]
        # The edge count below misses a change of widths or kernel sizes, and
        # those change the FLOPs. The parameter count catches most of them.
        today = int(sum(p.size for p in jax.tree_util.tree_leaves(params)))
        if n_params is not None and today != int(n_params):
            raise ValueError(
                f"{row_id}: the graph today has {today} parameters but the saved "
                f"run had {n_params}; the row has changed since, so its FLOPs "
                "cannot be recounted"
            )
        compute = count_compute(
            params,
            structure,
            batch_size=batch_size,
            algorithm=algorithm,
            infer_steps=int(saved["infer_steps"]),
        )
        if compute.weighted_edges != saved.get("weighted_edges"):
            raise ValueError(
                f"{row_id}: the graph today has {compute.weighted_edges} weighted "
                f"edges but the saved run had {saved.get('weighted_edges')}; the "
                "row has changed since, so its FLOPs cannot be recounted"
            )
        return compute


def _find_checkpoint(saved, row_id, trial, src_root, zoo) -> Optional[Path]:
    """Where a trial's weights are on this disk, or None if we cannot tell."""
    if not saved:
        return None
    path = Path(saved)
    if not path.is_absolute() and (src_root / path).exists():
        return src_root / path
    # Only paths that end in <row id>/trial<i> point into a zoo folder.
    if path.parts[-2:] != (row_id, f"trial{trial}"):
        return None
    for zoo_dir in ([Path(zoo)] if zoo else []) + [src_root / "zoo"]:
        candidate = zoo_dir / row_id / f"trial{trial}"
        if candidate.exists():
            return candidate
    return None


def _change(field, old, new, why) -> dict:
    return {"field": field, "old": old, "new": new, "why": why}


def refresh_trial(
    trial: dict,
    *,
    recounter: _Recounter,
    batch_size: int,
    src_path: Path,
    src_root: Path,
    out_root: Path,
    zoo=None,
    sha: Optional[str] = None,
) -> dict:
    """A corrected copy of one trial, with a record of what changed."""
    new = dict(trial)
    changes: List[dict] = []
    notes: List[str] = []
    row_id, i = trial["row_id"], trial["trial"]

    saved = trial.get("compute") or {}
    if trial.get("status") == "ok" and "infer_steps" in saved:
        compute = asdict(
            recounter.count(
                row_id, trial["algorithm"], saved, batch_size, trial.get("n_params")
            )
        )
        if compute != saved:
            new["compute"] = compute
            changes.append(
                _change("compute", saved, compute, "recounted with today's counter")
            )
        step_s = float(trial.get("step_time_ms") or 0.0) / 1000.0
        if step_s > 0:
            tflops = compute["flops_per_update"] / step_s / 1e12
            if tflops != trial.get("achieved_tflops"):
                new["achieved_tflops"] = tflops
                changes.append(
                    _change(
                        "achieved_tflops",
                        trial.get("achieved_tflops"),
                        tflops,
                        "recounted FLOPs over the step time the run measured",
                    )
                )

    old_ckpt = trial.get("checkpoint")
    found = _find_checkpoint(old_ckpt, row_id, i, src_root, zoo)
    if found is not None:
        rel = os.path.relpath(found.resolve(), out_root.resolve())
        if rel != old_ckpt:
            new["checkpoint"] = rel
            changes.append(
                _change(
                    "checkpoint",
                    old_ckpt,
                    rel,
                    "made relative to the results folder; the zoo is not copied",
                )
            )
    elif old_ckpt and Path(old_ckpt).is_absolute():
        notes.append(f"checkpoint {old_ckpt} was not found on this disk; kept as is")

    if trial.get("peak_memory_bytes") is not None and not trial.get("step_memory"):
        if new.get("peak_memory_comparable") is not False:
            new["peak_memory_comparable"] = False
            changes.append(
                _change(
                    "peak_memory_comparable",
                    trial.get("peak_memory_comparable"),
                    False,
                    PEAK_MEMORY_NOTE,
                )
            )
        notes.append(PEAK_MEMORY_NOTE)

    record = {
        "refreshed_utc": _now(),
        "git_sha": sha,
        "fabricpc": fabricpc.__version__,
        "source": str(src_path.resolve()),
        "changes": changes,
        "notes": notes,
    }
    new["refresh"] = list(trial.get("refresh") or []) + [record]
    return new


def _tflops_mean(trials: Sequence[dict]) -> Optional[float]:
    values = [t["achieved_tflops"] for t in trials if t.get("status") == "ok"]
    return sum(values) / len(values) if values else None


def _needs_registry(row_id, manifest, trials) -> bool:
    """True when refreshing this row needs today's registry entry for it."""
    has_batch = ((manifest or {}).get("row") or {}).get("batch_size") is not None
    recounts = any(
        t.get("status") == "ok" and "infer_steps" in (t.get("compute") or {})
        for t in trials
    )
    return not has_batch or recounts


def _read_row(src_row):
    """The manifest (or None) and every trial of one old row folder."""
    manifest_path = src_row / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    files = _trial_files(src_row)
    return manifest, [json.loads(p.read_text()) for p in files], files


def _check_rows(row_dirs, out_root) -> None:
    """Refuse the whole folder up front, before anything is written."""
    unknown, taken = [], []
    for src_row in row_dirs:
        manifest, trials, _ = _read_row(src_row)
        if src_row.name not in ROWS and _needs_registry(src_row.name, manifest, trials):
            unknown.append(src_row.name)
        if any((out_root / src_row.name).glob("trial*.json")):
            taken.append(src_row.name)
    if unknown:
        raise ValueError(
            "these rows are not in the registry, so they cannot be refreshed: "
            + ", ".join(unknown)
            + ". Refresh the other row folders one at a time instead."
        )
    if taken:
        raise ValueError(
            f"{out_root} already holds results for "
            + ", ".join(taken)
            + "; refresh writes to a new folder, pick one that is empty"
        )


def _plan_row(src_row, src_root, out_root, *, recounter, zoo, sha):
    """Work out the refreshed trials for one row without writing anything."""
    row_id = src_row.name
    manifest, old, files = _read_row(src_row)
    batch_size = ((manifest or {}).get("row") or {}).get("batch_size")
    if batch_size is None:
        batch_size = ROWS[row_id].batch_size
    new = [
        refresh_trial(
            t,
            recounter=recounter,
            batch_size=batch_size,
            src_path=p,
            src_root=src_root,
            out_root=out_root,
            zoo=zoo,
            sha=sha,
        )
        for t, p in zip(old, files)
    ]
    return {"src_row": src_row, "manifest": manifest, "old": old, "new": new}


def _old_band(src_row) -> Optional[str]:
    path = src_row / "summary.json"
    if not path.exists():
        return None
    return (json.loads(path.read_text()).get("band") or {}).get("status")


def _write_row(plan, out_root, *, sha, command):
    src_row, manifest, old, new = (
        plan["src_row"],
        plan["manifest"],
        plan["old"],
        plan["new"],
    )
    row_id = src_row.name
    out_row = out_root / row_id
    out_row.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "trials.csv"):
        (out_row / name).unlink(missing_ok=True)
    for t in new:
        (out_row / f"trial{t['trial']}.json").write_text(json.dumps(t, indent=2))

    write_trials_csv(out_root, row_id)
    n_ok = sum(t.get("status") == "ok" for t in new)
    band_notes = []
    if n_ok >= MIN_TRIALS and row_id in ROWS:
        judged = attach_band(out_root, ROWS[row_id], summarize_row(out_root, row_id))
        before, after = _old_band(src_row), (judged.band or {}).get("status")
        if before != after:
            band_notes.append(
                f"band re-judged against today's reference: {before} -> {after}"
            )

    if manifest is not None:
        fields = {c["field"] for t in new for c in t["refresh"][-1]["changes"]}
        if band_notes:
            fields.add("band")
        notes = sorted({n for t in new for n in t["refresh"][-1]["notes"]})
        record = {
            "refreshed_utc": _now(),
            "git_sha": sha,
            "fabricpc": fabricpc.__version__,
            "command": " ".join(command) if command else None,
            "source": str(src_row.resolve()),
            "changed_fields": sorted(fields),
            "notes": notes + band_notes,
        }
        manifest["refresh"] = list(manifest.get("refresh") or []) + [record]
        (out_row / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "row_id": row_id,
        "n_ok": n_ok,
        "achieved_tflops_old": _tflops_mean(old),
        "achieved_tflops_new": _tflops_mean(new),
    }


def _rebuild_compare(path: Path, out_root: Path, done) -> bool:
    """Rebuild one old compare file in the new folder. False if we cannot."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    try:
        if "comparison" in data:
            family = COMPARISONS.get(data["comparison"])
            if family is None or not set(family.rows) <= done:
                return False
            compare_family(out_root, family, metric=data.get("metric", family.metric))
            return True
        if "row_a" in data and "row_b" in data:
            if not {data["row_a"], data["row_b"]} <= done:
                return False
            compare_rows(
                out_root,
                data["row_a"],
                data["row_b"],
                metric=data.get("metric", "accuracy"),
            )
            return True
    except (KeyError, ValueError):  # a missing metric, or too few shared seeds
        return False
    return False


def _carry_over_compares_and_notes(src_root, out_root, done, sha) -> None:
    """Rebuild every comparison the old folder had, and keep its NOTES.md.

    A compare file we cannot rebuild (a row is missing, or it is in a shape
    we do not know) is copied as it was and listed in NOTES.md, so nothing
    is dropped without a word.
    """
    copied = []
    for path in sorted(src_root.glob("compare-*.json")):
        if not _rebuild_compare(path, out_root, done):
            shutil.copyfile(path, out_root / path.name)
            copied.append(path.name)

    old_notes = src_root / "NOTES.md"
    if not (copied or old_notes.exists()):
        return
    lines = [
        f"> This folder is a refresh copy of `{src_root.resolve()}`, made "
        f"{_now()} (commit {sha or 'unknown'}). FLOP counts, TFLOP/s, "
        "checkpoint paths and memory flags were corrected; see the `refresh` "
        "record in each trial file. Any such numbers quoted below describe "
        "the old files.",
        "",
    ]
    for name in copied:
        lines.append(
            f"- `{name}` was copied unchanged from the old folder; refresh "
            "could not rebuild it (a row it needs is missing, or its metric "
            "is not in the trials), so its numbers are not refreshed."
        )
    if copied:
        lines.append("")
    if old_notes.exists():
        lines.append(old_notes.read_text())
    (out_root / "NOTES.md").write_text("\n".join(lines))


def refresh_results(source, out, *, zoo=None, command=None) -> List[dict]:
    """Write a corrected copy of ``source`` (a results or row folder) to ``out``.

    Every row is checked and worked out before any file is written, so a
    row that cannot be refreshed leaves no half-made folder behind. Returns
    one small report per row with the old and new mean TFLOP/s.
    """
    source, out_root = Path(source), Path(out)
    _refuse_overlap(source, out_root)
    src_root, row_dirs = _find_rows(source)
    _check_rows(row_dirs, out_root)
    recounter = _Recounter()
    sha = _git_sha()
    plans = [
        _plan_row(d, src_root, out_root, recounter=recounter, zoo=zoo, sha=sha)
        for d in row_dirs
    ]
    reports = [_write_row(p, out_root, sha=sha, command=command) for p in plans]

    done = {r["row_id"] for r in reports}
    for family in COMPARISONS.values():
        if set(family.rows) <= done:
            try:
                compare_family(out_root, family, metric=family.metric)
            except ValueError:  # too few shared seeds to pair
                pass
    # Only a whole results folder has compare files and notes next to it.
    if source.resolve() == src_root.resolve():
        _carry_over_compares_and_notes(src_root, out_root, done, sha)
    return reports
