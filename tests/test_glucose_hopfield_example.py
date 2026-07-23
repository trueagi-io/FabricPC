"""Behavioral coverage for the separate glucose/Hopfield experiment."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest

from examples.glucose_hopfield_data import (
    prepare_hopfield_data,
    split_chronological,
)
from examples.glucose_hopfield_model import create_glucose_hopfield_transformer
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.training.train import train_step
from fabricpc.training.train_backprop import compute_forward_pass


def _write_glucose_csv(
    path: Path,
    *,
    split_labels: tuple[str, ...],
    points_per_sequence: int = 8,
) -> None:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-01")
    for sequence_id, split_label in enumerate(split_labels):
        sequence_start = start + pd.Timedelta(days=sequence_id)
        for offset in range(points_per_sequence):
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "Timestamp": sequence_start + pd.Timedelta(minutes=5 * offset),
                    "Glucose (mg/dL)": 80.0 + sequence_id * 10 + offset,
                    "Recommended Split": split_label,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_chronological_split_keeps_complete_future_sequences(tmp_path: Path) -> None:
    csv_path = tmp_path / "livia_like.csv"
    _write_glucose_csv(csv_path, split_labels=("",) * 10)
    dataframe = pd.read_csv(csv_path)

    train, val, test = split_chronological(
        dataframe, val_fraction=0.2, test_fraction=0.2
    )

    split_ids = [
        set(split_frame["sequence_id"].unique())
        for split_frame in (train, val, test)
    ]
    assert len(split_ids[0]) == 6
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
    assert pd.to_datetime(train["Timestamp"]).max() < pd.to_datetime(
        val["Timestamp"]
    ).min()
    assert pd.to_datetime(val["Timestamp"]).max() < pd.to_datetime(
        test["Timestamp"]
    ).min()


def test_prepared_labels_drive_external_dataset_windows(tmp_path: Path) -> None:
    csv_path = tmp_path / "loop_dev_like.csv"
    _write_glucose_csv(
        csv_path,
        split_labels=("train", "train", "val", "test"),
    )

    prepared = prepare_hopfield_data(
        data_path=csv_path,
        split_strategy="auto",
        seq_len=4,
        horizon=2,
        batch_size=2,
        max_eval_windows=2,
    )

    assert prepared["split_strategy"] == "recommended"
    assert prepared["n_train"] == 6
    assert prepared["n_val"] == 2
    assert prepared["n_test"] == 2
    assert prepared["n_val_full"] == 3
    assert prepared["n_test_full"] == 3
    train_batch = next(iter(prepared["train_loader"]))
    assert train_batch["x"].shape == (2, 4, 1)
    assert train_batch["y"].shape == (2, 2)
    assert np.isfinite(train_batch["x"]).all()


@pytest.mark.parametrize(
    "variant",
    ["baseline", "projection", "embed-storkey", "forecast-storkey"],
)
def test_graph_variants_complete_pc_training_step(variant: str) -> None:
    inference = InferenceSGDNormClip(
        eta_infer=1e-4,
        infer_steps=2,
        max_norm=1.0,
    )
    structure = create_glucose_hopfield_transformer(
        variant=variant,
        hopfield_strength=1.0,
        depth=1,
        embed_dim=4,
        num_heads=1,
        mlp_dim=8,
        seq_len=4,
        horizon=2,
        inference=inference,
        include_output_scaling=True,
    )
    key = jax.random.PRNGKey(7)
    params = initialize_params(structure, key)
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(params)
    batch = {
        "x": jnp.linspace(0.1, 0.8, 8).reshape(2, 4, 1),
        "y": jnp.asarray([[0.7, 0.8], [0.8, 0.9]]),
    }

    params, optimizer_state, energy, _ = train_step(
        params,
        optimizer_state,
        batch,
        structure,
        optimizer,
        key,
    )
    state = compute_forward_pass(params, structure, batch, key)
    output = state.nodes[structure.task_map["y"]].z_mu

    assert jnp.isfinite(energy)
    assert output.shape == (2, 2)
    assert jnp.isfinite(output).all()
