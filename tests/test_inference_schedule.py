"""Tests for composable inference schedules and segmented tracking."""

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.core.inference import InferenceSGD, InferenceSchedule, run_inference
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.training.train import train_step
from fabricpc.utils.dashboarding.inference_tracking import (
    run_inference_with_full_history,
    run_inference_with_history,
)


def _model(inference):
    x = IdentityNode(shape=(2,), name="x")
    h = Linear(shape=(3,), name="h")
    y = Linear(shape=(1,), name="y")
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=inference,
    )


def _setup(structure, rng_key):
    params = initialize_params(structure, rng_key)
    clamps = {
        "x": jnp.asarray([[0.2, -0.1], [0.4, 0.7]], dtype=jnp.float32),
        "y": jnp.asarray([[0.8], [-0.2]], dtype=jnp.float32),
    }
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    return params, clamps, state


def _assert_states_allclose(left, right, atol=1e-6):
    assert left.nodes.keys() == right.nodes.keys()
    for name in left.nodes:
        for field in left.nodes[name]._fields:
            assert jnp.allclose(
                getattr(left.nodes[name], field),
                getattr(right.nodes[name], field),
                atol=atol,
            ), f"{name}.{field} differs"


def _total_energy(state, structure):
    return sum(
        jnp.sum(state.nodes[name].energy)
        for name, node in structure.nodes.items()
        if node.node_info.in_degree > 0
    )


def _normalize_for_energy(params, state, clamps, structure):
    state = EPCInference.begin_segment(params, state, clamps, structure)
    return EPCInference.finalize_state(params, state, clamps, structure)


def test_segments_flatten_nested_schedules():
    first = EPCInference(infer_steps=2)
    second = InferenceSGD(infer_steps=3)
    third = InferenceSGD(infer_steps=4)
    schedule = InferenceSchedule(first, InferenceSchedule(second, third))
    assert schedule.segments() == ((first, 2), (second, 3), (third, 4))


def test_schedule_validates_entries():
    with pytest.raises(ValueError, match="at least one"):
        InferenceSchedule()
    with pytest.raises(TypeError, match="InferenceBase"):
        InferenceSchedule(InferenceSGD(), object())


def test_single_solver_schedule_matches_plain_solver(rng_key):
    solver = InferenceSGD(eta_infer=0.05, infer_steps=4)
    plain = _model(solver)
    scheduled = plain._replace(
        config={**plain.config, "inference": InferenceSchedule(solver)}
    )
    params, clamps, state = _setup(plain, rng_key)
    plain_state = run_inference(params, state, clamps, plain)
    scheduled_state = run_inference(params, state, clamps, scheduled)
    _assert_states_allclose(plain_state, scheduled_state)


def test_epc_then_spc_matches_manual_sequential_calls(rng_key):
    epc = EPCInference(eta_infer=0.05, infer_steps=3)
    spc = InferenceSGD(eta_infer=0.02, infer_steps=4)
    schedule = InferenceSchedule(epc, spc)
    structure = _model(schedule)
    params, clamps, state = _setup(structure, rng_key)

    scheduled_state = run_inference(params, state, clamps, structure)
    manual_state = epc.run_inference(params, state, clamps, structure)
    manual_state = spc.run_inference(params, manual_state, clamps, structure)
    _assert_states_allclose(scheduled_state, manual_state)


def test_schedule_runs_inside_jit(rng_key):
    structure = _model(
        InferenceSchedule(
            EPCInference(eta_infer=0.05, infer_steps=2),
            InferenceSGD(eta_infer=0.02, infer_steps=2),
        )
    )
    params, clamps, state = _setup(structure, rng_key)
    compiled = jax.jit(
        lambda graph_params, graph_state: run_inference(
            graph_params, graph_state, clamps, structure
        )
    )
    final_state = compiled(params, state)
    assert jnp.all(jnp.isfinite(final_state.nodes["h"].z_latent))


def test_schedule_runs_inside_jitted_train_step(rng_key):
    structure = _model(
        InferenceSchedule(
            EPCInference(eta_infer=0.02, infer_steps=2),
            InferenceSGD(eta_infer=0.01, infer_steps=2),
        )
    )
    params, clamps, _ = _setup(structure, rng_key)
    optimizer = optax.sgd(1e-3)
    opt_state = optimizer.init(params)
    batch = {"x": clamps["x"], "y": clamps["y"]}
    compiled = jax.jit(
        lambda graph_params, optimizer_state: train_step(
            graph_params,
            optimizer_state,
            batch,
            structure,
            optimizer,
            rng_key,
        )
    )

    updated_params, _, energy, final_state = compiled(params, opt_state)
    assert jnp.isfinite(energy)
    assert jnp.all(jnp.isfinite(final_state.nodes["h"].z_latent))
    changed = [
        not jnp.allclose(before, after)
        for before, after in zip(
            jax.tree_util.tree_leaves(params),
            jax.tree_util.tree_leaves(updated_params),
        )
    ]
    assert any(changed)


def test_round_trip_solver_handoffs_do_not_increase_true_energy(rng_key):
    epc = EPCInference(eta_infer=0.01, infer_steps=5)
    spc = InferenceSGD(eta_infer=0.01, infer_steps=5)
    structure = _model(InferenceSchedule(epc, spc, epc))
    params, clamps, state = _setup(structure, rng_key)
    state = _normalize_for_energy(params, state, clamps, structure)
    energies = [_total_energy(state, structure)]

    for solver in (epc, spc, epc):
        state = solver.run_inference(params, state, clamps, structure)
        state = _normalize_for_energy(params, state, clamps, structure)
        energies.append(_total_energy(state, structure))

    for before, after in zip(energies, energies[1:]):
        assert after <= before + 1e-6


def test_segmented_tracking_matches_run_and_concatenates_rows(rng_key):
    structure = _model(
        InferenceSchedule(
            EPCInference(eta_infer=0.05, infer_steps=2),
            InferenceSGD(eta_infer=0.02, infer_steps=3),
        )
    )
    params, clamps, state = _setup(structure, rng_key)
    expected = run_inference(params, state, clamps, structure)
    tracked, metrics = run_inference_with_history(params, state, clamps, structure)
    _assert_states_allclose(expected, tracked)
    assert metrics["h"]["energy"].shape == (5,)
    assert metrics["y"]["error_norm"].shape == (5,)

    full_state, full_history = run_inference_with_full_history(
        params, state, clamps, structure
    )
    _assert_states_allclose(expected, full_state)
    assert len(full_history) == 5


def test_schedule_step_stubs_fail_loudly():
    schedule = InferenceSchedule(InferenceSGD(infer_steps=1))
    with pytest.raises(NotImplementedError, match=r"segments\(\)"):
        schedule.compute_new_latent("node", None, {})
    with pytest.raises(NotImplementedError, match=r"segments\(\)"):
        type(schedule).inference_step(None, None, {}, None, schedule.config)
