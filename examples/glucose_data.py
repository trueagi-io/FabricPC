"""Self-contained glucose data pipeline for the FabricPC glucose transformer example.

Uses the prepared Livia dataset (``examples/data/livia_sugar_one_ready.csv``)
which contains contiguous 5-minute glucose sequences with ``sequence_id`` and
``Recommended Split`` columns.  Falls back to downloading and preparing the
raw ``livia_mini.csv`` from Hugging Face if the bundled file is absent.

No glucose-forecasting imports — everything lives here.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

BUNDLED_CSV = Path(__file__).parent / "data" / "livia_sugar_one_ready.csv"
LIVIA_URL = (
    "https://huggingface.co/datasets/Livia-Zaharia/glucose_processed"
    "/resolve/main/livia_mini.csv"
)
DEFAULT_CACHE_DIR = Path("data") / "glucose"


def _load_bundled() -> pd.DataFrame:
    """Load the bundled prepared CSV, normalizing the glucose column name."""
    df = pd.read_csv(BUNDLED_CSV)
    gl_candidates = ["Glucose (mg/dL)", "Glucose Value (mg/dL)", "glucose"]
    for col in gl_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "glucose"})
            break
    return df


def _download_and_prepare(cache_dir: Path) -> pd.DataFrame:
    """Fallback: download livia_mini.csv and build sequence_id / glucose."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = cache_dir / "livia_mini.csv"
    if not raw.exists():
        print(f"Downloading Livia dataset to {raw} ...")
        urllib.request.urlretrieve(LIVIA_URL, raw)
        sha = hashlib.sha256(raw.read_bytes()).hexdigest()[:16]
        print(f"  saved ({raw.stat().st_size:,} bytes, sha256={sha}...)")

    df = pd.read_csv(raw)
    gl_candidates = ["gl", "Glucose Value (mg/dL)", "Glucose (mg/dL)", "glucose"]
    gl_col = next((c for c in gl_candidates if c in df.columns), None)
    if gl_col is None:
        raise ValueError(f"No glucose column found; available: {list(df.columns)}")
    df = df.rename(columns={gl_col: "glucose"})

    id_col = next((c for c in ["id", "patient_id", "subject_id"] if c in df.columns), None)
    time_col = next(
        (c for c in ["time", "timestamp", "Timestamp (YYYY-MM-DDThh:mm:ss)"] if c in df.columns),
        None,
    )
    if id_col is None or time_col is None:
        raise ValueError(f"Need id and time columns; available: {list(df.columns)}")

    df = df[[id_col, time_col, "glucose"]].dropna()
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")
    df = df.dropna()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)

    rows = []
    seq_id = 0
    for _, group in df.groupby(id_col, sort=False):
        group = group.sort_values(time_col).reset_index(drop=True)
        times = group[time_col].values
        glucose = group["glucose"].values
        diffs = np.diff(times.astype("datetime64[m]").astype(np.int64))
        breaks = np.where(diffs != 5)[0]
        segments = np.split(np.arange(len(glucose)), breaks + 1)
        for seg in segments:
            if len(seg) < 2:
                continue
            for idx in seg:
                rows.append({"sequence_id": seq_id, "glucose": glucose[idx]})
            seq_id += 1

    out = pd.DataFrame(rows)
    print(f"Prepared {seq_id} sequences, {len(out)} rows from {raw.name}")
    return out


def load_glucose_df(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Load glucose data — bundled prepared CSV preferred, HF fallback."""
    if BUNDLED_CSV.exists():
        return _load_bundled()
    return _download_and_prepare(cache_dir)


def split_by_sequence(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leak-free split by ``sequence_id``.

    Uses the ``Recommended Split`` column if present and non-empty,
    otherwise splits randomly by sequence_id.
    """
    split_col = "Recommended Split"
    if (
        split_col in df.columns
        and df[split_col].notna().any()
        and (df[split_col] != "").any()
    ):
        train = df[df[split_col] == "train"]
        val = df[df[split_col] == "val"]
        test = df[df[split_col] == "test"]
        if len(train) > 0 and len(val) > 0 and len(test) > 0:
            return train, val, test

    seq_ids = df["sequence_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(seq_ids)
    n = len(seq_ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_ids = set(seq_ids[:n_test])
    val_ids = set(seq_ids[n_test : n_test + n_val])
    train_ids = set(seq_ids[n_test + n_val :])
    return (
        df[df["sequence_id"].isin(train_ids)],
        df[df["sequence_id"].isin(val_ids)],
        df[df["sequence_id"].isin(test_ids)],
    )


def build_sliding_windows(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create ``(X, Y)`` sliding-window arrays grouped by sequence_id.

    Returns:
        X: ``(N, seq_len, 1)`` float32 input windows.
        Y: ``(N, horizon)`` float32 target horizons.
    """
    xs, ys = [], []
    for _, group in df.groupby("sequence_id", sort=False):
        glucose = group["glucose"].values.astype(np.float32)
        total = seq_len + horizon
        for start in range(0, len(glucose) - total + 1, stride):
            xs.append(glucose[start : start + seq_len, None])
            ys.append(glucose[start + seq_len : start + total])
    if not xs:
        raise ValueError(
            f"No valid windows (need {seq_len + horizon} contiguous points; "
            f"max sequence length {df.groupby('sequence_id').size().max()})"
        )
    return np.stack(xs), np.stack(ys)


def normalize_glucose(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    val_X: np.ndarray | None = None,
    val_Y: np.ndarray | None = None,
    test_X: np.ndarray | None = None,
    test_Y: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Min-max normalize to [0, 1] using train statistics only."""
    g_min = float(np.nanmin(train_X))
    g_max = float(np.nanmax(train_X))
    spread = g_max - g_min if g_max > g_min else 1.0

    def norm(arr: np.ndarray) -> np.ndarray:
        return ((arr - g_min) / spread).astype(np.float32)

    result: dict[str, np.ndarray | float] = {
        "train_X": norm(train_X),
        "train_Y": norm(train_Y),
        "g_min": g_min,
        "g_max": g_max,
    }
    if val_X is not None:
        result["val_X"] = norm(val_X)
    if val_Y is not None:
        result["val_Y"] = norm(val_Y)
    if test_X is not None:
        result["test_X"] = norm(test_X)
    if test_Y is not None:
        result["test_Y"] = norm(test_Y)
    return result


class GlucoseWindowLoader:
    """Yields batches of ``{"x": (B, S, 1), "y": (B, H)}`` dicts.

    Supports both iteration (one pass) and cycling (infinite steps).
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        idx = np.arange(len(self.X))
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield {"x": self.X[batch_idx], "y": self.Y[batch_idx]}

    def cycle(self) -> Iterator[dict[str, np.ndarray]]:
        """Infinite iterator that reshuffles each pass."""
        while True:
            yield from self


def prepare_data(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seq_len: int = 128,
    horizon: int = 12,
    stride: int = 1,
    batch_size: int = 64,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """End-to-end: load → split → window → normalize → loaders.

    Returns a dict with keys: train_loader, val_loader, test_loader,
    g_min, g_max, n_train, n_val, n_test.
    """
    df = load_glucose_df(cache_dir)
    train_df, val_df, test_df = split_by_sequence(
        df, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    train_X, train_Y = build_sliding_windows(train_df, seq_len, horizon, stride)
    val_X, val_Y = build_sliding_windows(val_df, seq_len, horizon, stride)
    test_X, test_Y = build_sliding_windows(test_df, seq_len, horizon, stride)

    norm = normalize_glucose(train_X, train_Y, val_X, val_Y, test_X, test_Y)

    train_loader = GlucoseWindowLoader(
        norm["train_X"], norm["train_Y"], batch_size, shuffle=True, seed=seed
    )
    val_loader = GlucoseWindowLoader(
        norm["val_X"], norm["val_Y"], batch_size, shuffle=False
    )
    test_loader = GlucoseWindowLoader(
        norm["test_X"], norm["test_Y"], batch_size, shuffle=False
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "g_min": norm["g_min"],
        "g_max": norm["g_max"],
        "n_train": len(norm["train_X"]),
        "n_val": len(norm["val_X"]),
        "n_test": len(norm["test_X"]),
    }
