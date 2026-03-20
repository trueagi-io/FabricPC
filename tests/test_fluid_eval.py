"""Tests for synthetic fluid utilities and fluid-specific evaluation."""

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph import initialize_params
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training.fluid_eval import evaluate_fluid_reconstruction
from fabricpc.utils.data.synthetic_fluid import (
    ArrayBatchLoader,
    apply_observation_model,
    generate_taylor_green_vortex_dataset,
    make_observation_mask,
)
from fabricpc.utils.fluid import compute_fluid_metrics


class TestSyntheticFluidData:
    def test_taylor_green_dataset_shape_and_low_mismatch(self):
        fields, dx = generate_taylor_green_vortex_dataset(
            num_samples=4,
            grid_size=12,
            seed=0,
        )

        assert fields.shape == (4, 12, 12, 3)
        metrics = compute_fluid_metrics(
            jnp.array(fields),
            jnp.array(fields),
            viscosity=0.0,
            dx=dx,
            dy=dx,
        )

        assert metrics["field_mse"] == 0.0
        assert metrics["divergence_norm"] < 0.1
        assert metrics["momentum_residual_norm"] < 0.2

    def test_observation_mask_and_loader(self):
        fields, _ = generate_taylor_green_vortex_dataset(
            num_samples=3,
            grid_size=12,
            seed=1,
        )
        mask = make_observation_mask(12, observed_fraction=0.25, seed=2)
        observed = apply_observation_model(fields, mask, noise_std=0.0, seed=3)

        assert observed.shape == fields.shape
        assert np.allclose(observed * mask[None, ...], fields * mask[None, ...])
        assert np.allclose(observed * (1.0 - mask[None, ...]), 0.0)

        loader = ArrayBatchLoader(observed, fields, batch_size=2, mask=mask)
        batch = next(iter(loader))

        assert set(batch.keys()) == {"x", "y", "mask"}
        assert batch["x"].shape[-1] == 3
        assert batch["mask"].shape == batch["y"].shape


class TestFluidEvaluation:
    def test_evaluate_fluid_reconstruction_for_identity_graph(self):
        fields, dx = generate_taylor_green_vortex_dataset(
            num_samples=6,
            grid_size=12,
            seed=4,
        )
        loader = ArrayBatchLoader(fields, fields, batch_size=3)

        input_node = IdentityNode(shape=(12, 12, 3), name="input")
        output_node = IdentityNode(
            shape=(12, 12, 3),
            name="output",
            energy=GaussianEnergy(),
        )
        structure = graph(
            nodes=[input_node, output_node],
            edges=[Edge(source=input_node, target=output_node.slot("in"))],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(eta_infer=0.01, infer_steps=2),
        )
        params = initialize_params(structure, jax.random.PRNGKey(0))

        metrics = evaluate_fluid_reconstruction(
            params=params,
            structure=structure,
            test_loader=loader,
            rng_key=jax.random.PRNGKey(1),
            viscosity=0.0,
            dx=dx,
            dy=dx,
        )

        assert metrics["field_mse"] < 1e-6
        assert metrics["velocity_mse"] < 1e-6
        assert metrics["pressure_mse"] < 1e-6
        assert metrics["divergence_norm"] < 0.1
        assert metrics["momentum_residual_norm"] < 0.2
