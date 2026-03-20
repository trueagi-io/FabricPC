"""
Navier-Stokes Energy Acceptance Benchmark
========================================

Benchmark `NavierStokesEnergy` against `GaussianEnergy` on a synthetic periodic
fluid reconstruction task. The benchmark measures:
- reconstruction quality via MSE
- physics consistency via divergence and momentum residual norms

Artifacts are written to a temporary directory so the script can be run without
polluting the repository.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.energy import GaussianEnergy, NavierStokesEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph import initialize_params
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.training import evaluate_fluid_reconstruction, predict_fluid_batch, train_pcn
from fabricpc.utils.data import (
    ArrayBatchLoader,
    apply_observation_model,
    generate_taylor_green_vortex_dataset,
    make_observation_mask,
)
from fabricpc.utils.fluid import compute_navier_stokes_diagnostics

jax.config.update("jax_default_prng_impl", "threefry2x32")

GRID_SIZE = 12
HIDDEN_UNITS = 48
BATCH_SIZE = 12
NUM_EPOCHS = 4
OBSERVATION_MASK_SEED = 7
DATASET_SEED = 123
RUN_SEEDS = (0, 1, 2)
OUTPUT_DIR = Path(
    os.environ.get(
        "FABRICPC_FLUID_ACCEPTANCE_DIR",
        os.path.join(tempfile.gettempdir(), "fabricpc_navier_stokes_acceptance"),
    )
)
NS_WEIGHT = 0.27
NS_VISCOSITY = 0.0
FIELD_MSE_TOLERANCE = 1.05
PHYSICS_IMPROVEMENT_TARGET = 0.75

SCENARIOS = (
    {"name": "full_field_clean", "observed_fraction": 1.0, "noise_std": 0.0},
    {"name": "partial_field_clean", "observed_fraction": 0.25, "noise_std": 0.0},
    {"name": "partial_field_noisy", "observed_fraction": 0.25, "noise_std": 0.03},
)


def create_structure(energy, grid_size: int):
    input_node = IdentityNode(shape=(grid_size, grid_size, 3), name="input")
    hidden_node = Linear(
        shape=(HIDDEN_UNITS,),
        name="hidden",
        activation=TanhActivation(),
        flatten_input=True,
    )
    output_node = Linear(
        shape=(grid_size, grid_size, 3),
        name="field",
        activation=IdentityActivation(),
        flatten_input=True,
        energy=energy,
    )
    return graph(
        nodes=[input_node, hidden_node, output_node],
        edges=[
            Edge(source=input_node, target=hidden_node.slot("in")),
            Edge(source=hidden_node, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
        inference=InferenceSGD(eta_infer=0.02, infer_steps=5),
    )


def split_fields(fields: np.ndarray):
    return fields[:72], fields[72:96], fields[96:]


def build_loaders(
    fields: np.ndarray,
    observed_fraction: float,
    noise_std: float,
    seed: int,
):
    train_y, val_y, test_y = split_fields(fields)
    mask = make_observation_mask(
        grid_size=GRID_SIZE,
        observed_fraction=observed_fraction,
        seed=OBSERVATION_MASK_SEED,
    )
    train_x = apply_observation_model(train_y, mask, noise_std=noise_std, seed=seed + 10)
    val_x = apply_observation_model(val_y, mask, noise_std=noise_std, seed=seed + 20)
    test_x = apply_observation_model(test_y, mask, noise_std=noise_std, seed=seed + 30)

    return {
        "train": ArrayBatchLoader(
            train_x,
            train_y,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=seed,
            mask=mask,
        ),
        "val": ArrayBatchLoader(
            val_x,
            val_y,
            batch_size=BATCH_SIZE,
            shuffle=False,
            mask=mask,
        ),
        "test": ArrayBatchLoader(
            test_x,
            test_y,
            batch_size=BATCH_SIZE,
            shuffle=False,
            mask=mask,
        ),
        "mask": mask,
    }


def build_energy_configs(dx: float):
    return {
        "gaussian": GaussianEnergy(),
        "navier_stokes": NavierStokesEnergy(
            viscosity=NS_VISCOSITY,
            dx=dx,
            dy=dx,
            latent_ns_weight=NS_WEIGHT,
            prediction_ns_weight=NS_WEIGHT,
        ),
    }


def save_pgm(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


def normalize_panel_image(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    field_min = float(np.min(field))
    field_max = float(np.max(field))
    if np.isclose(field_min, field_max):
        return np.full(field.shape, 127, dtype=np.uint8)
    scaled = (field - field_min) / (field_max - field_min)
    return np.clip(255.0 * scaled, 0.0, 255.0).astype(np.uint8)


def build_panel(arrays: list[np.ndarray], rows: int, cols: int, pad: int = 2) -> np.ndarray:
    height, width = arrays[0].shape
    canvas = np.full(
        (rows * height + (rows - 1) * pad, cols * width + (cols - 1) * pad),
        255,
        dtype=np.uint8,
    )
    for idx, array in enumerate(arrays):
        row = idx // cols
        col = idx % cols
        y0 = row * (height + pad)
        x0 = col * (width + pad)
        canvas[y0 : y0 + height, x0 : x0 + width] = normalize_panel_image(array)
    return canvas


def save_visualizations(
    output_dir: Path,
    scenario_name: str,
    model_name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    dx: float,
) -> None:
    pred_sample = prediction[0]
    target_sample = target[0]

    field_panel = build_panel(
        [
            target_sample[..., 0],
            target_sample[..., 1],
            target_sample[..., 2],
            pred_sample[..., 0],
            pred_sample[..., 1],
            pred_sample[..., 2],
        ],
        rows=2,
        cols=3,
    )
    save_pgm(output_dir / f"{scenario_name}_{model_name}_fields.pgm", field_panel)

    pred_diag = compute_navier_stokes_diagnostics(
        jnp.array(prediction[:1]),
        viscosity=NS_VISCOSITY,
        dx=dx,
        dy=dx,
    )
    target_diag = compute_navier_stokes_diagnostics(
        jnp.array(target[:1]),
        viscosity=NS_VISCOSITY,
        dx=dx,
        dy=dx,
    )
    physics_panel = build_panel(
        [
            np.asarray(target_diag["divergence"][0]),
            np.asarray(pred_diag["divergence"][0]),
            np.asarray(target_diag["momentum_residual"][0]),
            np.asarray(pred_diag["momentum_residual"][0]),
        ],
        rows=2,
        cols=2,
    )
    save_pgm(output_dir / f"{scenario_name}_{model_name}_physics.pgm", physics_panel)


def aggregate_metrics(runs: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    split_names = runs[0].keys()
    for split_name in split_names:
        metric_names = runs[0][split_name].keys()
        summary[split_name] = {}
        for metric_name in metric_names:
            values = np.array([run[split_name][metric_name] for run in runs], dtype=np.float64)
            summary[split_name][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return summary


def evaluate_split(params, structure, loader, dx: float, seed: int):
    return evaluate_fluid_reconstruction(
        params=params,
        structure=structure,
        test_loader=loader,
        rng_key=jax.random.PRNGKey(seed),
        viscosity=NS_VISCOSITY,
        dx=dx,
        dy=dx,
    )


def run_model_for_scenario(
    model_name: str,
    energy,
    dx: float,
    loaders: dict[str, object],
    seed: int,
):
    structure = create_structure(energy, grid_size=GRID_SIZE)
    params = initialize_params(structure, jax.random.PRNGKey(seed))
    params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=loaders["train"],
        optimizer=optax.adam(1e-3),
        config={"num_epochs": NUM_EPOCHS},
        rng_key=jax.random.PRNGKey(seed + 100),
        verbose=False,
    )

    metrics = {
        "train": evaluate_split(params, structure, loaders["train"], dx, seed + 200),
        "val": evaluate_split(params, structure, loaders["val"], dx, seed + 300),
        "test": evaluate_split(params, structure, loaders["test"], dx, seed + 400),
    }

    example_batch = next(iter(loaders["test"]))
    prediction, batch = predict_fluid_batch(
        params,
        structure,
        {"x": jnp.array(example_batch["x"]), "y": jnp.array(example_batch["y"])},
        rng_key=jax.random.PRNGKey(seed + 500),
    )
    return params, metrics, np.asarray(prediction), np.asarray(batch["y"])


def acceptance_from_summary(summary):
    baseline = summary["gaussian"]["test"]
    candidate = summary["navier_stokes"]["test"]
    checks = {
        "field_mse_within_5pct": candidate["field_mse"]["mean"]
        <= FIELD_MSE_TOLERANCE * baseline["field_mse"]["mean"],
        "divergence_improves_25pct": candidate["divergence_norm"]["mean"]
        <= PHYSICS_IMPROVEMENT_TARGET * baseline["divergence_norm"]["mean"],
        "momentum_residual_improves_25pct": candidate["momentum_residual_norm"]["mean"]
        <= PHYSICS_IMPROVEMENT_TARGET * baseline["momentum_residual_norm"]["mean"],
        "three_seed_stability": summary["gaussian"]["num_seeds"] >= 3
        and summary["navier_stokes"]["num_seeds"] >= 3,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fields, dx = generate_taylor_green_vortex_dataset(
        num_samples=120,
        grid_size=GRID_SIZE,
        seed=DATASET_SEED,
    )
    model_energies = build_energy_configs(dx)
    benchmark_summary = {
        "config": {
            "grid_size": GRID_SIZE,
            "hidden_units": HIDDEN_UNITS,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "run_seeds": list(RUN_SEEDS),
            "dx": dx,
            "navier_stokes_weight": NS_WEIGHT,
            "viscosity": NS_VISCOSITY,
        },
        "scenarios": {},
    }

    print("Navier-Stokes acceptance benchmark")
    print(f"Artifacts: {OUTPUT_DIR}")

    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
        print(f"\nScenario: {scenario_name}")
        scenario_results = {}
        seed_metrics: dict[str, list[dict[str, dict[str, float]]]] = {
            "gaussian": [],
            "navier_stokes": [],
        }

        for seed in RUN_SEEDS:
            loaders = build_loaders(
                fields=fields,
                observed_fraction=scenario["observed_fraction"],
                noise_std=scenario["noise_std"],
                seed=seed,
            )
            for model_name, energy in model_energies.items():
                _, metrics, prediction, target = run_model_for_scenario(
                    model_name=model_name,
                    energy=energy,
                    dx=dx,
                    loaders=loaders,
                    seed=seed,
                )
                seed_metrics[model_name].append(metrics)
                if seed == RUN_SEEDS[0]:
                    save_visualizations(
                        OUTPUT_DIR,
                        scenario_name=scenario_name,
                        model_name=model_name,
                        prediction=prediction,
                        target=target,
                        dx=dx,
                    )

        for model_name, runs in seed_metrics.items():
            scenario_results[model_name] = aggregate_metrics(runs)
            scenario_results[model_name]["num_seeds"] = len(runs)

        scenario_results["acceptance"] = acceptance_from_summary(scenario_results)
        benchmark_summary["scenarios"][scenario_name] = scenario_results

        gaussian_test = scenario_results["gaussian"]["test"]
        ns_test = scenario_results["navier_stokes"]["test"]
        print(
            "  Gaussian test:",
            f"field_mse={gaussian_test['field_mse']['mean']:.6f}",
            f"div={gaussian_test['divergence_norm']['mean']:.6f}",
            f"mom={gaussian_test['momentum_residual_norm']['mean']:.6f}",
        )
        print(
            "  Navier-Stokes test:",
            f"field_mse={ns_test['field_mse']['mean']:.6f}",
            f"div={ns_test['divergence_norm']['mean']:.6f}",
            f"mom={ns_test['momentum_residual_norm']['mean']:.6f}",
        )
        print("  Acceptance:", scenario_results["acceptance"]["passed"])

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(benchmark_summary, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
