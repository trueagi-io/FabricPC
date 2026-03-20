"""Synthetic periodic fluid datasets for predictive coding benchmarks."""

from __future__ import annotations

import math
from typing import Dict, Iterator

import numpy as np


def generate_taylor_green_vortex_dataset(
    num_samples: int,
    grid_size: int,
    seed: int = 0,
    amplitude_range: tuple[float, float] = (0.7, 1.3),
) -> tuple[np.ndarray, float]:
    """
    Generate periodic Taylor-Green-like vortex fields in `(u, v, p)` form.

    Returns:
        Tuple of `(fields, dx)` where `fields.shape == (num_samples, H, W, 3)`.
    """
    rng = np.random.default_rng(seed)
    coords = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False, dtype=np.float32)
    dx = float(2.0 * np.pi / grid_size)
    x_grid, y_grid = np.meshgrid(coords, coords, indexing="xy")

    fields = []
    for _ in range(num_samples):
        amplitude = rng.uniform(*amplitude_range)
        phase_x = rng.uniform(0.0, 2.0 * np.pi)
        phase_y = rng.uniform(0.0, 2.0 * np.pi)

        u = amplitude * np.sin(x_grid + phase_x) * np.cos(y_grid + phase_y)
        v = -amplitude * np.cos(x_grid + phase_x) * np.sin(y_grid + phase_y)
        p = 0.25 * amplitude**2 * (
            np.cos(2.0 * (x_grid + phase_x)) + np.cos(2.0 * (y_grid + phase_y))
        )
        fields.append(np.stack([u, v, p], axis=-1).astype(np.float32))

    return np.stack(fields), dx


def make_observation_mask(
    grid_size: int,
    observed_fraction: float,
    seed: int = 0,
    channels: int = 3,
) -> np.ndarray:
    """Create a fixed spatial observation mask shared across all channels."""
    rng = np.random.default_rng(seed)
    base = (rng.random((grid_size, grid_size, 1)) < observed_fraction).astype(np.float32)
    return np.repeat(base, channels, axis=-1)


def apply_observation_model(
    fields: np.ndarray,
    mask: np.ndarray,
    noise_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Mask fields and optionally add observation noise on observed entries only."""
    observations = fields * mask[None, ...]
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(fields.shape).astype(np.float32)
        observations = observations + noise_std * noise * mask[None, ...]
    return observations.astype(np.float32)


class ArrayBatchLoader:
    """Simple numpy-backed batch loader yielding dict batches for `train_pcn`."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        mask: np.ndarray | None = None,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.mask = None if mask is None else np.broadcast_to(mask, y.shape).astype(np.float32)

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        indices = np.arange(len(self.x))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch = {
                "x": self.x[batch_idx],
                "y": self.y[batch_idx],
            }
            if self.mask is not None:
                batch["mask"] = self.mask[batch_idx]
            yield batch
