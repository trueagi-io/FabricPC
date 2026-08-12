"""Data preparation for the separate glucose/Hopfield experiments.

Livia is split chronologically by complete contiguous sequences.  External
datasets such as ``loop_ai_ready_joined2_dev.csv`` keep their published
``Recommended Split`` labels when available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from examples.glucose_data import (
    BUNDLED_CSV,
    GlucoseWindowLoader,
    build_sliding_windows,
    normalize_glucose,
)

SplitStrategy = Literal["auto", "chronological", "recommended"]
GLUCOSE_COLUMNS = ("Glucose (mg/dL)", "Glucose Value (mg/dL)", "glucose")


def load_experiment_csv(path: Path) -> pd.DataFrame:
    """Load a prepared glucose CSV and normalize its glucose column."""
    dataframe = pd.read_csv(path)
    glucose_column = next(
        (column for column in GLUCOSE_COLUMNS if column in dataframe.columns),
        None,
    )
    if glucose_column is None:
        raise ValueError(
            f"No glucose column found in {path}; tried {GLUCOSE_COLUMNS}"
        )
    if "sequence_id" not in dataframe.columns:
        raise ValueError(f"Missing sequence_id column in {path}")
    return dataframe.rename(columns={glucose_column: "glucose"})


def _has_recommended_split(dataframe: pd.DataFrame) -> bool:
    if "Recommended Split" not in dataframe.columns:
        return False
    labels = set(
        dataframe["Recommended Split"].dropna().astype(str).str.strip().str.lower()
    )
    return {"train", "val", "test"}.issubset(labels)


def split_recommended(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use complete train/val/test labels from the prepared dataset."""
    if not _has_recommended_split(dataframe):
        raise ValueError(
            "Recommended Split must contain non-empty train, val, and test labels"
        )
    labels = (
        dataframe["Recommended Split"].fillna("").astype(str).str.strip().str.lower()
    )
    return tuple(
        dataframe.loc[labels == split].copy()
        for split in ("train", "val", "test")
    )


def split_chronological(
    dataframe: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    timestamp_column: str = "Timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split complete sequences by their first timestamp.

    Keeping a sequence wholly within one partition avoids overlapping-window
    leakage while ensuring validation and test represent later periods.
    """
    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("Validation and test fractions must be between 0 and 1")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("Validation and test fractions must sum to less than 1")
    if timestamp_column not in dataframe.columns:
        raise ValueError(
            f"Chronological splitting requires {timestamp_column!r} in the CSV"
        )

    timestamps = pd.to_datetime(dataframe[timestamp_column], errors="raise")
    starts = (
        dataframe.assign(_timestamp=timestamps)
        .groupby("sequence_id", sort=False)["_timestamp"]
        .min()
        .sort_values()
    )
    sequence_ids = starts.index.to_list()
    if len(sequence_ids) < 3:
        raise ValueError("Chronological splitting requires at least three sequences")

    count = len(sequence_ids)
    test_count = max(1, int(count * test_fraction))
    val_count = max(1, int(count * val_fraction))
    train_count = count - val_count - test_count
    if train_count < 1:
        raise ValueError("Split fractions leave no training sequences")

    partitions = (
        set(sequence_ids[:train_count]),
        set(sequence_ids[train_count : train_count + val_count]),
        set(sequence_ids[train_count + val_count :]),
    )
    return tuple(
        dataframe.loc[dataframe["sequence_id"].isin(ids)].copy()
        for ids in partitions
    )


def prepare_hopfield_data(
    *,
    data_path: Path = BUNDLED_CSV,
    split_strategy: SplitStrategy = "auto",
    seq_len: int = 64,
    horizon: int = 12,
    stride: int = 1,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_eval_windows: int | None = None,
    seed: int = 42,
) -> dict[str, GlucoseWindowLoader | float | int | str]:
    """Prepare normalized windows for Livia or a prepared external dataset."""
    dataframe = load_experiment_csv(data_path)
    strategy = split_strategy
    if strategy == "auto":
        strategy = "recommended" if _has_recommended_split(dataframe) else "chronological"

    if strategy == "recommended":
        train, val, test = split_recommended(dataframe)
    else:
        train, val, test = split_chronological(
            dataframe,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

    train_x, train_y = build_sliding_windows(train, seq_len, horizon, stride)
    val_x, val_y = build_sliding_windows(val, seq_len, horizon, stride)
    test_x, test_y = build_sliding_windows(test, seq_len, horizon, stride)
    normalized = normalize_glucose(
        train_x, train_y, val_x, val_y, test_x, test_y
    )

    def evenly_spaced_subset(
        features: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_eval_windows is None or len(features) <= max_eval_windows:
            return features, targets
        indices = np.linspace(
            0, len(features) - 1, num=max_eval_windows, dtype=np.int64
        )
        return features[indices], targets[indices]

    val_features, val_targets = evenly_spaced_subset(
        normalized["val_X"], normalized["val_Y"]
    )
    test_features, test_targets = evenly_spaced_subset(
        normalized["test_X"], normalized["test_Y"]
    )

    return {
        "train_loader": GlucoseWindowLoader(
            normalized["train_X"],
            normalized["train_Y"],
            batch_size,
            shuffle=True,
            seed=seed,
        ),
        "val_loader": GlucoseWindowLoader(
            val_features, val_targets, batch_size, shuffle=False
        ),
        "test_loader": GlucoseWindowLoader(
            test_features, test_targets, batch_size, shuffle=False
        ),
        "g_min": normalized["g_min"],
        "g_max": normalized["g_max"],
        "n_train": len(normalized["train_X"]),
        "n_val": len(val_features),
        "n_test": len(test_features),
        "n_val_full": len(normalized["val_X"]),
        "n_test_full": len(normalized["test_X"]),
        "split_strategy": strategy,
        "data_path": str(data_path),
    }
