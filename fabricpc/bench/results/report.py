"""One readable page for a whole results folder.

    python -m fabricpc.bench report <results_root> [--out report.md]
                                                   [--format md|html]

A results root holds one folder per run, often nested (a Lightning job
folder with several families inside). A "family folder" is any folder with
row folders in it (``<row>/summary.json``, ``manifest.json``, ``trials.csv``
or ``trial*.json``) or ``compare-*.json`` files next to them. For each one
the page shows:

- a table of rows: algorithm, seeds, the family's metric as mean ± SE (and
  the 95% interval when the summary has one), step time, slowdown against
  the backprop row, and the band verdict,
- the paired comparisons, with the p-value and a plain yes/no,
- where the numbers came from (git sha, hardware, date), and
- any NOTES.md in that folder, quoted.

Real results folders are messy: old compare files, rows with one trial and
no summary, logs from debugging sessions. The report never stops on those.
It reads what it can and says what it skipped.
"""

import csv
import datetime
import html
import json
import re
import statistics
from pathlib import Path

ROW_FILES = ("summary.json", "manifest.json", "trials.csv")
SKIP_DIRS = {"zoo", "__pycache__"}
ALGORITHMS = ("backprop", "spc", "epc")
NOTES_MAX_LINES = 60
BAND_WORDS = {
    "pass": "PASS",
    "fail": "FAIL",
    "no_reference": "no reference",
    "not_comparable": "not comparable (short run)",
}


# ---------------------------------------------------------------- scanning


def _is_row_dir(path: Path) -> bool:
    if path.name in SKIP_DIRS or not path.is_dir():
        return False
    return any((path / f).exists() for f in ROW_FILES) or any(path.glob("trial*.json"))


def _subdirs(path: Path):
    try:
        kids = sorted(p for p in path.iterdir() if p.is_dir())
    except OSError:
        return []
    return [p for p in kids if p.name not in SKIP_DIRS and not p.name.startswith(".")]


def _read_json(path: Path, notes: list):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        notes.append(f"could not read `{path.name}` ({type(e).__name__})")
        return None


def _algorithm_of(row_id: str) -> str:
    """Guess the algorithm from the row id when no file says it."""
    for part in reversed(row_id.split("-")):
        if part in ALGORITHMS:
            return part
    return "?"


def _family_of(row_id: str) -> str:
    """The part of a row id before its algorithm: mnist-mlp-spc -> mnist-mlp."""
    parts = row_id.split("-")
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] in ALGORITHMS:
            return "-".join(parts[:i])
    return row_id


def _registry_metric(row_id: str):
    try:
        from fabricpc.bench.registry import ROWS
    except Exception:  # the report should work even if the registry cannot load
        return None
    row = ROWS.get(row_id)
    return row.metric if row is not None else None


def _ci(stats: dict):
    """A 95% interval from a metric's stats, in any of the shapes we have used."""
    pairs = []
    for key in ("ci95", "ci"):
        value = stats.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            pairs.append((value[0], value[1]))
        if isinstance(value, dict):
            pairs.append((value.get("low"), value.get("high")))
    for low, high in (("ci_low", "ci_high"), ("ci95_low", "ci95_high")):
        pairs.append((stats.get(low), stats.get(high)))
    # With one seed there is no interval and summary.py writes null for it.
    for low, high in pairs:
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return float(low), float(high)
    return None


def _mean_se(values):
    if not values:
        return None, None
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None
    return mean, se


def _from_trials(row_dir: Path, notes: list):
    """(n, n_failed, {metric: [values]}, [step ms]) from trials.csv or trial files."""
    records = []
    csv_path = row_dir / "trials.csv"
    if csv_path.exists():
        try:
            with csv_path.open(newline="") as f:
                records = list(csv.DictReader(f))
        except (OSError, csv.Error) as e:
            notes.append(f"could not read `trials.csv` ({type(e).__name__})")
    if not records:
        for path in sorted(row_dir.glob("trial*.json")):
            data = _read_json(path, notes)
            if isinstance(data, dict):
                flat = dict(data.get("metrics") or {})
                flat.update(status=data.get("status"))
                flat.update(step_time_ms=data.get("step_time_ms"))
                records.append(flat)
    ok = [r for r in records if r.get("status") == "ok"]
    values = {}
    for r in ok:
        for key, value in r.items():
            try:
                values.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                continue
    steps = values.pop("step_time_ms", [])
    return len(ok), len(records) - len(ok), values, steps


def load_row(row_dir: Path, metric=None) -> dict:
    """Everything the report needs about one row, never raising."""
    notes = []
    row = {"id": row_dir.name, "notes": notes, "stats": {}, "step_ms": None}
    manifest = (
        _read_json(row_dir / "manifest.json", notes)
        if (row_dir / "manifest.json").exists()
        else None
    )
    row["manifest"] = manifest if isinstance(manifest, dict) else {}
    summary = (
        _read_json(row_dir / "summary.json", notes)
        if (row_dir / "summary.json").exists()
        else None
    )
    info = row["manifest"].get("row") or {}
    if isinstance(summary, dict) and summary.get("metrics"):
        row["algorithm"] = summary.get("algorithm") or _algorithm_of(row_dir.name)
        row["n"] = summary.get("n_ok")
        row["n_failed"] = summary.get("n_failed", 0)
        row["stats"] = {
            k: v for k, v in summary["metrics"].items() if isinstance(v, dict)
        }
        step = (summary.get("timing") or {}).get("step_time_ms") or {}
        row["step_ms"] = step.get("mean")
        row["band"] = summary.get("band") or {}
    else:
        n, n_failed, values, steps = _from_trials(row_dir, notes)
        notes.append("no summary.json; numbers are from the trial files")
        row["algorithm"] = info.get("algorithm") or _algorithm_of(row_dir.name)
        row["n"], row["n_failed"], row["band"] = n, n_failed, {}
        for name, vals in values.items():
            if name in ("trial", "seed", "num_epochs", "n_params"):
                continue
            mean, se = _mean_se(vals)
            row["stats"][name] = {"mean": mean, "se": se, "n": len(vals)}
        row["step_ms"] = _mean_se(steps)[0]
    row["metric"] = _pick_metric(row, metric)
    return row


def _pick_metric(row: dict, wanted):
    choices = [
        _registry_metric(row["id"]),
        wanted,
        (row.get("band") or {}).get("metric"),
        "accuracy",
    ]
    for name in choices:
        if name and name in row["stats"]:
            return name
    return next(iter(sorted(row["stats"])), None)


def load_compares(family_dir: Path, notes: list) -> list:
    """Every contrast in the folder's compare-*.json files, old or new shape."""
    found = []
    for path in sorted(family_dir.glob("compare-*.json")):
        data = _read_json(path, notes)
        if not isinstance(data, dict):
            continue
        if "contrasts" in data:  # a whole family: several contrasts
            for c in data.get("contrasts") or []:
                if isinstance(c, dict):
                    found.append({**c, "metric": data.get("metric"), "file": path.name})
        elif "row_a" in data:  # older, one pair per file
            found.append(
                {
                    **data,
                    "arm_a": data["row_a"],
                    "arm_b": data.get("row_b"),
                    "file": path.name,
                }
            )
        else:
            notes.append(f"`{path.name}` is not a compare file this report knows")
    return found


def _scan(folder: Path, root: Path, families: list, others: list) -> bool:
    """Find family folders under ``folder``. True if any were found."""
    kids = _subdirs(folder)
    rows = [k for k in kids if _is_row_dir(k)]
    has_compare = any(folder.glob("compare-*.json"))
    found = False
    if rows or has_compare:
        families.append((folder, rows))
        found = True
    for kid in kids:
        if kid in rows:
            continue
        if _scan(kid, root, families, others):
            found = True
        elif folder == root:
            n = sum(1 for p in kid.rglob("*") if p.is_file())
            others.append(f"`{kid.relative_to(root).as_posix()}/`: {n} files, skipped")
    if found and not (rows or has_compare) and folder != root:
        # e.g. a job folder with a stray result file next to its families
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            rel = path.relative_to(root).as_posix()
            others.append(f"`{rel}`: loose file, not read")
    return found


# ---------------------------------------------------------------- building


def _fmt(x, digits=4):
    if x is None:
        return "-"
    return f"{x:.{digits}f}" if abs(x) < 1000 else f"{x:,.1f}"


def _metric_cell(stats: dict):
    if not stats or stats.get("mean") is None:
        return "-"
    text = _fmt(stats["mean"])
    if stats.get("se") is not None:
        text += f" ± {_fmt(stats['se'])}"
    ci = _ci(stats)
    if ci:
        text += f" [{_fmt(ci[0])}, {_fmt(ci[1])}]"
    return text


def _short(arm: str, rows: dict) -> str:
    row = rows.get(arm)
    return row["algorithm"] if row else _algorithm_of(arm) if arm else "?"


def _family_blocks(folder: Path, row_dirs: list, root: Path) -> list:
    notes = []
    contrasts = load_compares(folder, notes)
    metric = next((c.get("metric") for c in contrasts if c.get("metric")), None)
    rows = {d.name: load_row(d, metric) for d in row_dirs}
    title = folder.relative_to(root).as_posix() if folder != root else folder.name
    blocks = [("h", 2, title)]

    # A folder can hold several model families (say mnist-mlp and
    # fashionmnist-mlp), so each row is timed against its own backprop row.
    bp_ms = {
        _family_of(r["id"]): r["step_ms"]
        for r in rows.values()
        if r["algorithm"] == "backprop"
    }
    table = []
    for row in rows.values():
        m = row["metric"]
        base = bp_ms.get(_family_of(row["id"]))
        slow = f"{row['step_ms'] / base:.2f}x" if row["step_ms"] and base else "-"
        n = row["n"] if row["n"] is not None else "-"
        if row.get("n_failed"):
            n = f"{n} (+{row['n_failed']} failed)"
        band = row.get("band") or {}
        table.append(
            [
                row["algorithm"],
                row["id"],
                str(n),
                m or "-",
                _metric_cell(row["stats"].get(m) if m else None),
                _fmt(row["step_ms"], 2),
                slow,
                BAND_WORDS.get(band.get("status"), band.get("status") or "-"),
            ]
        )
    if table:
        head = ["algorithm", "row", "seeds", "metric", "mean ± SE [95% CI]"]
        head += ["step ms", "vs backprop", "band"]
        blocks.append(("table", head, table))
    if contrasts:
        blocks.append(("h", 3, "Paired comparisons"))
        ctable = []
        arms = [c.get("arm_a") or "" for c in contrasts] + list(rows)
        mixed = len({_family_of(a) for a in arms}) > 1
        for c in contrasts:
            a, b = _short(c.get("arm_a"), rows), _short(c.get("arm_b"), rows)
            if mixed:
                b += f" ({_family_of(c.get('arm_a') or '')})"
            sig = c.get("significant_at_05")
            # A handful of seeds and several contrasts with no correction:
            # say what the test says, not more.
            words = {
                True: "yes (p < 0.05, uncorrected)",
                False: "no, could be seed noise",
            }
            p = c.get("p_value")
            ci = _ci(c)
            ctable.append(
                [
                    f"{a} - {b}",
                    c.get("metric") or "-",
                    _sign(c.get("mean_diff")),
                    f"[{_sign(ci[0])}, {_sign(ci[1])}]" if ci else "-",
                    f"{p:.3g}" if isinstance(p, (int, float)) else "-",
                    str(c.get("n", "-")),
                    words.get(sig, "-"),
                ]
            )
        head = ["difference (a - b)", "metric", "mean diff", "95% CI of diff"]
        head += ["p-value", "pairs"]
        blocks.append(("table", head + ["significant at 5%?"], ctable))
    blocks += _provenance_blocks(rows)
    blocks += _notes_blocks(folder, root)
    all_notes = notes + [f"{r['id']}: {n}" for r in rows.values() for n in r["notes"]]
    if all_notes:
        blocks.append(("p", "Skipped or partial:"))
        blocks.append(("list", all_notes))
    return blocks


def _sign(x):
    if not isinstance(x, (int, float)):
        return "-"
    return f"{x:+.4f}" if abs(x) < 1000 else f"{x:+,.1f}"


def _provenance_blocks(rows: dict) -> list:
    shas, hardware, dates = [], [], []
    for row in rows.values():
        m = row["manifest"]
        if m.get("git_sha"):
            shas.append(str(m["git_sha"])[:7])
        plat = m.get("platform") or {}
        where = ", ".join(m.get("devices") or []) or "?"
        if plat:
            where += f" ({plat.get('system', '?')} {plat.get('machine', '')})".rstrip()
        if m:
            hardware.append(where.replace(" )", ")"))
        if m.get("created_utc"):
            dates.append(str(m["created_utc"])[:10])
    if not (shas or hardware or dates):
        return [("p", "Provenance: no manifest.json found.")]

    def join(items):
        return ", ".join(dict.fromkeys(items)) or "?"

    text = f"Provenance: git `{join(shas)}` · hardware {join(hardware)} · run {join(dates)}"
    return [("p", text)]


def _notes_blocks(folder: Path, root: Path) -> list:
    blocks = []
    for path in sorted(folder.glob("*.md")):
        if path.name.upper().startswith("REPORT"):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            lines = path.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            blocks.append(("p", f"Notes: [{rel}]({rel}) (could not read it)"))
            continue
        blocks.append(("p", f"Notes from [{rel}]({rel}):"))
        shown = lines[:NOTES_MAX_LINES]
        if len(lines) > NOTES_MAX_LINES:
            shown.append(f"... ({len(lines) - NOTES_MAX_LINES} more lines in the file)")
        blocks.append(("quote", shown))
    return blocks


def _depth_blocks(families, root) -> list:
    """One table of score against depth for rows swept over depth
    (fabricpc.bench.deep), across all their family folders. Each row is
    tagged with the results folder its family sits in, so the same row
    run in two places shows up twice instead of one silently winning."""
    try:
        from fabricpc.bench.deep import depth_table

        loaded = []
        for folder, dirs in families:
            try:
                source = str(Path(folder).parent.relative_to(root))
            except ValueError:  # the root is itself a family folder
                source = "."
            for d in dirs:
                loaded.append({**load_row(d), "source": source})
        table = depth_table(loaded)
    except Exception as e:  # an extra view; never sink the page over it
        return [("p", f"Accuracy against depth: skipped ({e})")]
    if table is None:
        return []
    return [("h", 2, "Accuracy against depth"), ("table", *table)]


def build_report(root) -> list:
    """The whole page as a list of simple blocks, ready for ``render``."""
    root = Path(root)
    today = datetime.date.today().isoformat()
    blocks = [("h", 1, "FabricPC benchmark report")]
    blocks.append(("p", f"Results root: `{root}` · made {today}"))
    if not root.is_dir():
        return blocks + [("p", f"Results root not found: `{root}`")]
    blocks.append(
        (
            "p",
            "Each metric is the mean over seeds ± its standard error (SE). "
            "'vs backprop' is step time divided by the backprop row's step "
            "time. A comparison is paired seed by seed; 'significant' means "
            "p < 0.05.",
        )
    )
    families, others = [], []
    _scan(root, root, families, others)
    if not families:
        blocks.append(("p", "No benchmark results were found here."))
    for folder, row_dirs in families:
        try:
            blocks += _family_blocks(folder, row_dirs, root)
        except Exception as e:  # one odd folder must not sink the whole page
            blocks.append(("p", f"Skipped `{folder.name}`: could not read it ({e})"))
    blocks += _depth_blocks(families, root)
    if others:
        blocks.append(("h", 2, "Other files and folders (not benchmark results)"))
        blocks.append(("list", others))
    return blocks


# ---------------------------------------------------------------- rendering


def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def _render_md(blocks) -> str:
    out = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            out.append("#" * block[1] + " " + block[2])
        elif kind == "p":
            out.append(block[1])
        elif kind == "list":
            out.append("\n".join(f"- {item}" for item in block[1]))
        elif kind == "quote":
            out.append("\n".join(f"> {line}".rstrip() for line in block[1]))
        elif kind == "table":
            head, rows = block[1], block[2]
            lines = ["| " + " | ".join(head) + " |"]
            lines.append("|" + "|".join("---" for _ in head) + "|")
            lines += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in rows]
            out.append("\n".join(lines))
    return "\n\n".join(out) + "\n"


def _inline_html(text: str) -> str:
    text = html.escape(str(text))
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)


_CSS = (
    "body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;"
    "padding:0 16px;line-height:1.45}table{border-collapse:collapse;margin:1em 0}"
    "td,th{border:1px solid #bbb;padding:4px 8px;text-align:left}"
    "th{background:#eee}blockquote{border-left:3px solid #bbb;margin:1em 0;"
    "padding:0 1em;color:#444;white-space:pre-wrap}"
)


def _render_html(blocks) -> str:
    out = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            out.append(f"<h{block[1]}>{_inline_html(block[2])}</h{block[1]}>")
        elif kind == "p":
            out.append(f"<p>{_inline_html(block[1])}</p>")
        elif kind == "list":
            items = "".join(f"<li>{_inline_html(i)}</li>" for i in block[1])
            out.append(f"<ul>{items}</ul>")
        elif kind == "quote":
            text = "\n".join(html.escape(line) for line in block[1])
            out.append(f"<blockquote>{text}</blockquote>")
        elif kind == "table":
            head = "".join(f"<th>{html.escape(h)}</th>" for h in block[1])
            body = "".join(
                "<tr>" + "".join(f"<td>{_inline_html(c)}</td>" for c in r) + "</tr>"
                for r in block[2]
            )
            out.append(f"<table><tr>{head}</tr>{body}</table>")
    return (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        "<title>FabricPC benchmark report</title>"
        f"<style>{_CSS}</style></head><body>\n" + "\n".join(out) + "\n</body></html>\n"
    )


def render(blocks, fmt: str = "md") -> str:
    return _render_html(blocks) if fmt == "html" else _render_md(blocks)


def run_report(root, out=None, fmt=None) -> str:
    """Build the page and write it to ``out`` (or just return it)."""
    if fmt is None:
        fmt = "html" if out and str(out).endswith((".html", ".htm")) else "md"
    text = render(build_report(root), fmt)
    if out:
        Path(out).write_text(text)
    return text
