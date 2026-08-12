"""Run separate Livia-first glucose/Hopfield feasibility experiments.

Examples:
    uv run glucose-hopfield
    uv run glucose-hopfield --mode compare --variant embed-storkey
    uv run glucose-hopfield --variant embed-storkey --hopfield-strength 0 \
        --hopfield-strength 1 --hopfield-strength 2
    uv run glucose-hopfield \
        --data ../glucose-forecasting/data/input/loop_ai_ready_joined2_dev.csv
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from examples import glucose_transformer as base_training

import jax
import jax.numpy as jnp
import numpy as np
import typer

from examples.glucose_data import BUNDLED_CSV, GlucoseWindowLoader
from examples.glucose_hopfield_data import prepare_hopfield_data
from examples.glucose_hopfield_model import create_glucose_hopfield_transformer
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state

app = typer.Typer(
    add_completion=False,
    help="Compare matched glucose transformer and Hopfield-memory variants.",
)


class TrainingMode(StrEnum):
    PC = "pc"
    BACKPROP = "backprop"
    COMPARE = "compare"


class ModelVariant(StrEnum):
    BASELINE = "baseline"
    PROJECTION = "projection"
    EMBED_STORKEY = "embed-storkey"
    FORECAST_STORKEY = "forecast-storkey"


class SplitStrategy(StrEnum):
    AUTO = "auto"
    CHRONOLOGICAL = "chronological"
    RECOMMENDED = "recommended"


@dataclass(frozen=True)
class TrainingArguments:
    """Arguments consumed by the shared stable glucose training loop."""

    depth: int
    embed_dim: int
    num_heads: int
    mlp_dim: int
    seq_len: int
    horizon: int
    epochs: int
    max_updates: int | None
    lr: float
    lr_backprop: float
    warmup_steps: int
    decay_steps: int | None
    decay_epochs: int | None
    batch_size: int
    seed: int
    eta_infer: float
    infer_steps: int
    max_infer_norm: float
    weight_init_std: float
    grad_clip: float
    include_output_scaling: bool
    patience: int
    log_every: int
    resume: bool


def evaluate_with_settling(
    params,
    structure,
    loader: GlucoseWindowLoader,
    rng_key: jax.Array,
    glucose_min: float,
    glucose_max: float,
) -> dict[str, float]:
    """Evaluate after unclamped PC relaxation of internal memory states."""
    glucose_range = max(glucose_max - glucose_min, 1e-8)

    @jax.jit
    def evaluate_batch(
        graph_params,
        batch: dict[str, jax.Array],
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
        input_name = structure.task_map["x"]
        output_name = structure.task_map["y"]
        clamps = {input_name: batch["x"]}
        state = initialize_graph_state(
            structure,
            batch_size=batch["x"].shape[0],
            rng_key=key,
            clamps=clamps,
            params=graph_params,
        )
        final_state = run_inference(
            graph_params, state, clamps=clamps, structure=structure
        )
        predictions = (
            final_state.nodes[output_name].z_mu * glucose_range + glucose_min
        )
        targets = batch["y"] * glucose_range + glucose_min
        absolute_error = jnp.abs(predictions - targets)
        return (
            jnp.sum((predictions - targets) ** 2),
            jnp.sum(absolute_error),
            jnp.sum(
                absolute_error / jnp.maximum(jnp.abs(targets), 1e-8)
            ),
            predictions.size,
        )

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_absolute_relative_error = 0.0
    count = 0
    for numpy_batch in loader:
        rng_key, batch_key = jax.random.split(rng_key)
        batch = {name: jnp.asarray(value) for name, value in numpy_batch.items()}
        squared_error, absolute_error, relative_error, batch_count = (
            evaluate_batch(params, batch, batch_key)
        )
        total_squared_error += float(squared_error)
        total_absolute_error += float(absolute_error)
        total_absolute_relative_error += float(relative_error)
        count += int(batch_count)

    return {
        "rmse_mg_dl": float(np.sqrt(total_squared_error / max(count, 1))),
        "mae_mg_dl": total_absolute_error / max(count, 1),
        "mard_percent": (
            100.0 * total_absolute_relative_error / max(count, 1)
        ),
    }


def _run_name(
    mode: TrainingMode,
    variant: ModelVariant,
    strength: float | None,
) -> str:
    if "storkey" not in variant.value:
        return f"{variant.value}_{mode.value}"
    strength_name = "learnable" if strength is None else f"{strength:g}"
    return f"{variant.value}_strength-{strength_name}_{mode.value}"


def _variants_with_strengths(
    variants: list[ModelVariant],
    strengths: list[float],
    learnable_strength: bool,
) -> list[tuple[ModelVariant, float | None]]:
    runs: list[tuple[ModelVariant, float | None]] = []
    for variant in variants:
        if "storkey" not in variant.value:
            runs.append((variant, 0.0))
            continue
        runs.extend((variant, strength) for strength in strengths)
        if learnable_strength:
            runs.append((variant, None))
    return runs


def _write_run_metadata(
    run_dir: Path,
    *,
    variant: ModelVariant,
    strength: float | None,
    data: dict[str, object],
    settled_evaluation: bool,
) -> dict[str, object]:
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        {
            "variant": variant.value,
            "hopfield_strength": strength,
            "data_path": data["data_path"],
            "split_strategy": data["split_strategy"],
            "settled_evaluation": settled_evaluation,
        }
    )
    config_path.write_text(json.dumps(config, indent=2))
    return config


@app.command()
def main(
    data: Path = typer.Option(
        BUNDLED_CSV,
        exists=True,
        dir_okay=False,
        help="Prepared Livia or Loop+AI-READI CSV.",
    ),
    out_dir: Path = typer.Option(
        Path("runs/glucose_hopfield"),
        help="Experiment output directory.",
    ),
    mode: TrainingMode = typer.Option(TrainingMode.PC),
    variant: list[ModelVariant] = typer.Option(
        [
            ModelVariant.BASELINE,
            ModelVariant.PROJECTION,
            ModelVariant.EMBED_STORKEY,
        ],
        "--variant",
        help="Repeat to select matched experimental arms.",
    ),
    hopfield_strength: list[float] = typer.Option(
        [1.0],
        "--hopfield-strength",
        help="Repeat to sweep fixed Storkey strengths.",
    ),
    learnable_strength: bool = typer.Option(
        False,
        help="Also run Storkey arms with learnable strength.",
    ),
    split: SplitStrategy = typer.Option(
        SplitStrategy.AUTO,
        help="Auto uses published labels, otherwise chronological sequences.",
    ),
    seq_len: int = typer.Option(64, min=4),
    horizon: int = typer.Option(12, min=1),
    stride: int = typer.Option(1, min=1),
    depth: int = typer.Option(2, min=1),
    embed_dim: int = typer.Option(32, min=1),
    num_heads: int = typer.Option(1, min=1),
    mlp_dim: int = typer.Option(128, min=1),
    epochs: int = typer.Option(30, min=1),
    max_updates: int | None = typer.Option(
        None,
        min=1,
        help="Stop after this many optimizer updates.",
    ),
    batch_size: int = typer.Option(64, min=1),
    max_eval_windows: int | None = typer.Option(
        None,
        min=1,
        help="Evenly subsample validation/test windows for pilot runs.",
    ),
    seed: int = typer.Option(42),
    lr: float = typer.Option(3.2753170973521557e-3, min=0.0),
    lr_backprop: float = typer.Option(1e-3, min=0.0),
    warmup_steps: int = typer.Option(200, min=0),
    decay_steps: int | None = typer.Option(None, min=1),
    decay_epochs: int | None = typer.Option(
        None,
        min=1,
        help="LR decay horizon in epochs.",
    ),
    eta_infer: float = typer.Option(1.4435783212385837e-5, min=0.0),
    infer_steps: int = typer.Option(19, min=1),
    max_infer_norm: float = typer.Option(1.0, min=0.0),
    weight_init_std: float = typer.Option(0.02186191083483616, min=0.0),
    grad_clip: float = typer.Option(0.5, min=0.0),
    include_output_scaling: bool = typer.Option(True),
    patience: int = typer.Option(4, min=1),
    log_every: int = typer.Option(20, min=1),
    resume: bool = typer.Option(False),
    platform: str | None = typer.Option(
        None,
        help="JAX platform override; auto-detects CUDA when omitted.",
    ),
) -> None:
    """Run matched Livia-first memory experiments."""
    del platform  # Read before JAX import by examples.glucose_transformer.
    if embed_dim % num_heads:
        raise typer.BadParameter("embed-dim must be divisible by num-heads")
    if seq_len % 4:
        raise typer.BadParameter("seq-len must be divisible by 4")
    if not hopfield_strength and any(
        "storkey" in selected.value for selected in variant
    ) and not learnable_strength:
        raise typer.BadParameter(
            "Storkey variants require a fixed or learnable Hopfield strength"
        )

    prepared = prepare_hopfield_data(
        data_path=data,
        split_strategy=split.value,
        seq_len=seq_len,
        horizon=horizon,
        stride=stride,
        batch_size=batch_size,
        max_eval_windows=max_eval_windows,
        seed=seed,
    )
    typer.echo(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
    typer.echo(
        f"Data: {prepared['n_train']} train, {prepared['n_val']} val, "
        f"{prepared['n_test']} test windows; split={prepared['split_strategy']}"
    )
    if max_eval_windows is not None:
        typer.echo(
            f"Pilot evaluation subset: val {prepared['n_val']}/"
            f"{prepared['n_val_full']}, test {prepared['n_test']}/"
            f"{prepared['n_test_full']}"
        )

    arguments = TrainingArguments(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        horizon=horizon,
        epochs=epochs,
        max_updates=max_updates,
        lr=lr,
        lr_backprop=lr_backprop,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_epochs=decay_epochs,
        batch_size=batch_size,
        seed=seed,
        eta_infer=eta_infer,
        infer_steps=infer_steps,
        max_infer_norm=max_infer_norm,
        weight_init_std=weight_init_std,
        grad_clip=grad_clip,
        include_output_scaling=include_output_scaling,
        patience=patience,
        log_every=log_every,
        resume=resume,
    )
    inference_modes = (
        [TrainingMode.PC, TrainingMode.BACKPROP]
        if mode is TrainingMode.COMPARE
        else [mode]
    )
    summaries: list[dict[str, object]] = []

    for selected_variant, strength in _variants_with_strengths(
        variant, hopfield_strength, learnable_strength
    ):
        builder: Callable = partial(
            create_glucose_hopfield_transformer,
            variant=selected_variant.value,
            hopfield_strength=strength,
        )
        for selected_mode in inference_modes:
            run_name = _run_name(selected_mode, selected_variant, strength)
            run_dir = out_dir / run_name
            typer.echo(f"\nRunning {run_name}")
            result = base_training.train_single(
                selected_mode.value,
                prepared,
                arguments,
                out_dir=run_dir,
                lr_override=(
                    lr if selected_mode is TrainingMode.PC else lr_backprop
                ),
                structure_builder=builder,
                evaluate_fn=base_training.evaluate,
            )
            config = _write_run_metadata(
                run_dir,
                variant=selected_variant,
                strength=strength,
                data=prepared,
                settled_evaluation=False,
            )
            if selected_mode is TrainingMode.PC:
                with (run_dir / "best_params.pkl").open("rb") as file:
                    best_params = pickle.load(file)
                settle_started = time.perf_counter()
                settled_test = evaluate_with_settling(
                    best_params,
                    builder(
                        depth=depth,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        seq_len=seq_len,
                        horizon=horizon,
                        inference=base_training.InferenceSGDNormClip(
                            eta_infer=eta_infer,
                            infer_steps=infer_steps,
                            max_norm=max_infer_norm,
                        ),
                        weight_init_std=weight_init_std,
                        include_output_scaling=include_output_scaling,
                    ),
                    prepared["test_loader"],
                    jax.random.PRNGKey(seed),
                    prepared["g_min"],
                    prepared["g_max"],
                )
                config.update(
                    {
                        "settled_test_mae_mg_dl": settled_test["mae_mg_dl"],
                        "settled_test_rmse_mg_dl": settled_test["rmse_mg_dl"],
                        "settled_test_mard_percent": settled_test["mard_percent"],
                        "settled_test_elapsed_s": round(
                            time.perf_counter() - settle_started, 1
                        ),
                    }
                )
                (run_dir / "config.json").write_text(
                    json.dumps(config, indent=2)
                )
            summaries.append(config)
            del result["params"]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    typer.echo(f"\nWrote {summary_path}")


if __name__ == "__main__":
    app()
