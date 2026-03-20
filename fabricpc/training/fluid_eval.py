"""Fluid-specific evaluation utilities for predictive coding graphs."""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.core.inference import run_inference
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.utils.fluid import compute_fluid_metrics


def _to_batch(batch_data: Any) -> Dict[str, jnp.ndarray]:
    if isinstance(batch_data, (list, tuple)):
        return {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
    if isinstance(batch_data, dict):
        return {key: jnp.array(value) for key, value in batch_data.items()}
    raise ValueError(f"Unsupported batch format: {type(batch_data)}")


def predict_fluid_batch(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    output_node_name: str | None = None,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Run inference with only `x` clamped and return the output-node prediction."""
    if "x" not in structure.task_map:
        raise ValueError("structure.task_map must contain 'x' for fluid evaluation")

    output_node_name = output_node_name or structure.task_map.get("y")
    if output_node_name is None:
        raise ValueError(
            "structure.task_map must contain 'y' or output_node_name must be provided"
        )

    clamps = {structure.task_map["x"]: batch["x"]}
    state = initialize_graph_state(
        structure,
        batch_size=batch["x"].shape[0],
        rng_key=rng_key,
        clamps=clamps,
        params=params,
    )
    final_state = run_inference(params, state, clamps, structure)
    return final_state.nodes[output_node_name].z_mu, batch


def evaluate_fluid_reconstruction(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    rng_key: jax.Array,
    viscosity: float,
    dx: float = 1.0,
    dy: float = 1.0,
    channel_map: Dict[str, int] | None = None,
    output_node_name: str | None = None,
) -> Dict[str, float]:
    """Evaluate a graph on fluid reconstruction metrics instead of accuracy."""
    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000

    batch_keys = jax.random.split(rng_key, num_batches)
    totals: Dict[str, float] = {}
    total_samples = 0

    for batch_idx, batch_data in enumerate(test_loader):
        batch = _to_batch(batch_data)
        predictions, batch = predict_fluid_batch(
            params,
            structure,
            batch,
            batch_keys[batch_idx],
            output_node_name=output_node_name,
        )
        metrics = compute_fluid_metrics(
            predictions,
            batch["y"],
            viscosity=viscosity,
            dx=dx,
            dy=dy,
            channel_map=channel_map,
            mask=batch.get("mask"),
        )
        batch_size = int(batch["x"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        total_samples += batch_size

    return {key: value / total_samples for key, value in totals.items()}
