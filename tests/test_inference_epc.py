"""Correctness and integration tests for error-based PC inference."""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.activations import IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphParams
from fabricpc.graph_assembly import (
    GraphCycleError,
    TaskMap,
    UnrolledCycleScheduler,
    graph,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import IdentityNode, Linear, StorkeyHopfield
from fabricpc.nodes.transformer_v2 import EmbeddingNode


def _chain(inference, insertion_order=("x", "h", "y"), scaling=None):
    x = IdentityNode(shape=(2,), name="x")
    h = Linear(shape=(3,), name="h", activation=IdentityActivation())
    y = Linear(shape=(1,), name="y", activation=IdentityActivation())
    by_name = {node.name: node for node in (x, h, y)}
    return graph(
        nodes=[by_name[name] for name in insertion_order],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=inference,
        scaling=scaling,
    )


def _data():
    return {
        "x": jnp.asarray([[0.2, -0.4], [0.7, 0.1]], dtype=jnp.float32),
        "y": jnp.asarray([[0.8], [-0.3]], dtype=jnp.float32),
    }


def _total_energy(state, structure):
    return sum(
        jnp.sum(state.nodes[name].energy)
        for name, node in structure.nodes.items()
        if node.node_info.in_degree > 0
    )


def test_feedforward_initialization_is_zero_error_parameterization(rng_key):
    structure = _chain(EPCInference(infer_steps=1))
    params = initialize_params(structure, rng_key)
    clamps = _data()
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    initial_hidden = state.nodes["h"].z_latent

    state = EPCInference.begin_segment(params, state, clamps, structure)
    assert jnp.allclose(state.nodes["h"].error, 0.0, atol=1e-7)

    derived = EPCInference.derive_states(params, state, clamps, structure)
    assert jnp.allclose(derived.nodes["h"].z_latent, initial_hidden, atol=1e-7)


def test_global_error_gradient_matches_closed_form_linear_chain(rng_key):
    structure = _chain(EPCInference(infer_steps=1))
    params = initialize_params(structure, rng_key)
    clamps = _data()
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    epsilon = jnp.asarray(
        [[0.04, -0.03, 0.02], [-0.01, 0.05, -0.02]], dtype=jnp.float32
    )
    state = update_node_in_state(state, "h", error=epsilon)
    state = EPCInference.zero_grads(params, state, clamps, structure)
    result = EPCInference.forward_value_and_grad(params, state, clamps, structure)

    output_residual = result.nodes["y"].z_mu - clamps["y"]
    output_weight = params.nodes["y"].weights["h->y:in"]
    expected = epsilon + output_residual @ output_weight.T
    assert jnp.allclose(result.nodes["h"].latent_grad, expected, atol=1e-6)


def test_global_error_gradient_matches_independent_jax_grad_oracle(rng_key):
    structure = _chain(EPCInference(infer_steps=1))
    params = initialize_params(structure, rng_key)
    clamps = _data()
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    epsilon = jnp.asarray(
        [[0.04, -0.03, 0.02], [-0.01, 0.05, -0.02]], dtype=jnp.float32
    )
    state = update_node_in_state(state, "h", error=epsilon)
    state = EPCInference.zero_grads(params, state, clamps, structure)
    result = EPCInference.forward_value_and_grad(params, state, clamps, structure)

    hidden_params = params.nodes["h"]
    output_params = params.nodes["y"]

    def hand_rolled_energy(relaxed_error):
        hidden_mu = (
            clamps["x"] @ hidden_params.weights["x->h:in"] + hidden_params.biases["b"]
        )
        hidden_latent = hidden_mu + relaxed_error
        output_mu = (
            hidden_latent @ output_params.weights["h->y:in"] + output_params.biases["b"]
        )
        hidden_energy = 0.5 * jnp.sum(relaxed_error**2)
        output_energy = 0.5 * jnp.sum((clamps["y"] - output_mu) ** 2)
        return hidden_energy + output_energy

    expected = jax.grad(hand_rolled_energy)(epsilon)
    assert jnp.allclose(result.nodes["h"].latent_grad, expected, atol=1e-6)


def test_epc_energy_decreases_over_steps(rng_key):
    structure = _chain(EPCInference(eta_infer=0.05, infer_steps=1))
    params = initialize_params(structure, rng_key)
    clamps = _data()
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    after_one = run_inference(params, state, clamps, structure)

    long_structure = structure._replace(
        config={
            **structure.config,
            "inference": EPCInference(eta_infer=0.05, infer_steps=80),
        }
    )
    after_many = run_inference(params, state, clamps, long_structure)
    assert _total_energy(after_many, long_structure) < _total_energy(
        after_one, structure
    )


def test_epc_and_spc_converge_to_same_state_and_weight_gradients(rng_key):
    epc_structure = _chain(EPCInference(eta_infer=0.05, infer_steps=400))
    spc_structure = epc_structure._replace(
        config={
            **epc_structure.config,
            "inference": InferenceSGD(eta_infer=0.05, infer_steps=400),
        }
    )
    params = initialize_params(epc_structure, rng_key)
    clamps = _data()
    initial = initialize_graph_state(
        epc_structure, 2, rng_key, clamps=clamps, params=params
    )

    epc_state = run_inference(params, initial, clamps, epc_structure)
    spc_state = run_inference(params, initial, clamps, spc_structure)
    for node_name in epc_structure.nodes:
        for field in ("z_latent", "z_mu", "energy"):
            assert jnp.allclose(
                getattr(epc_state.nodes[node_name], field),
                getattr(spc_state.nodes[node_name], field),
                atol=2e-4,
            ), f"{node_name}.{field} differs at the fixed point"

    epc_grads = compute_local_weight_gradients(params, epc_state, epc_structure)
    spc_grads = compute_local_weight_gradients(params, spc_state, spc_structure)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda left, right: jnp.allclose(left, right, atol=2e-4),
            epc_grads,
            spc_grads,
        )
    )


def test_free_readout_tracks_prediction_with_zero_energy(rng_key):
    structure = _chain(EPCInference(eta_infer=0.05, infer_steps=3))
    params = initialize_params(structure, rng_key)
    clamps = {"x": _data()["x"]}
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    final_state = run_inference(params, state, clamps, structure)
    readout = final_state.nodes["y"]
    assert jnp.allclose(readout.z_latent, readout.z_mu)
    assert jnp.allclose(readout.error, 0.0)
    assert jnp.allclose(readout.energy, 0.0)


def test_unclamped_source_latent_is_relaxed(rng_key):
    structure = _chain(EPCInference(eta_infer=0.05, infer_steps=3))
    params = initialize_params(structure, rng_key)
    clamps = {"y": _data()["y"]}
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    initial_source = state.nodes["x"].z_latent
    final_state = run_inference(params, state, clamps, structure)
    assert not jnp.allclose(final_state.nodes["x"].z_latent, initial_source)
    assert jnp.all(jnp.isfinite(final_state.nodes["x"].z_latent))


def test_integer_token_source_stays_out_of_autodiff(rng_key):
    tokens = Linear(shape=(3,), name="tokens")
    embedding = EmbeddingNode(
        shape=(3, 4), name="embedding", vocab_size=11, embed_dim=4
    )
    output = Linear(shape=(3, 2), name="output")
    structure = graph(
        nodes=[tokens, embedding, output],
        edges=[
            Edge(source=tokens, target=embedding.slot("in")),
            Edge(source=embedding, target=output.slot("in")),
        ],
        task_map=TaskMap(x=tokens, y=output),
        inference=EPCInference(eta_infer=0.05, infer_steps=2),
    )
    params = initialize_params(structure, rng_key)
    clamps = {
        "tokens": jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
        "output": jnp.zeros((2, 3, 2), dtype=jnp.float32),
    }
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    final_state = jax.jit(
        lambda graph_params, graph_state: run_inference(
            graph_params, graph_state, clamps, structure
        )
    )(params, state)
    assert final_state.nodes["tokens"].z_latent.dtype == jnp.int32
    assert jnp.all(jnp.isfinite(final_state.nodes["embedding"].z_latent))


@pytest.mark.parametrize("num_unrolls, message", [(1, "back edges"), (2, "repeated")])
def test_epc_rejects_all_cyclic_schedules(rng_key, num_unrolls, message):
    x = IdentityNode(shape=(2,), name="x")
    a = Linear(shape=(2,), name="a")
    b = Linear(shape=(2,), name="b")
    y = Linear(shape=(1,), name="y")
    structure = graph(
        nodes=[x, a, b, y],
        edges=[
            Edge(source=x, target=a.slot("in")),
            Edge(source=a, target=b.slot("in")),
            Edge(source=b, target=a.slot("in")),
            Edge(source=b, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=EPCInference(infer_steps=1),
        topology_scheduler=UnrolledCycleScheduler(num_unrolls=num_unrolls),
    )
    params = initialize_params(structure, rng_key)
    clamps = {"x": _data()["x"], "y": _data()["y"]}
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    with pytest.raises(GraphCycleError, match=message):
        run_inference(params, state, clamps, structure)


def test_final_state_satisfies_error_parameterization(rng_key):
    structure = _chain(EPCInference(eta_infer=0.05, infer_steps=5))
    params = initialize_params(structure, rng_key)
    clamps = _data()
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    final_state = run_inference(params, state, clamps, structure)
    hidden = final_state.nodes["h"]
    assert jnp.allclose(hidden.z_latent, hidden.z_mu + hidden.error, atol=1e-7)


def test_cross_entropy_clamped_output_has_finite_global_gradients(rng_key):
    x = IdentityNode(shape=(2,), name="x")
    h = Linear(shape=(3,), name="h", activation=IdentityActivation())
    y = Linear(
        shape=(2,),
        name="y",
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
    )
    structure = graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=EPCInference(eta_infer=0.02, infer_steps=2),
    )
    params = initialize_params(structure, rng_key)
    clamps = {
        "x": _data()["x"],
        "y": jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
    }
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    final_state = run_inference(params, state, clamps, structure)
    assert jnp.all(jnp.isfinite(final_state.nodes["h"].latent_grad))
    assert jnp.all(jnp.isfinite(final_state.nodes["y"].energy))


def test_mupc_forward_scaling_is_inside_derived_program(rng_key):
    structure = _chain(
        EPCInference(infer_steps=1), scaling=MuPCConfig(include_output=True)
    )
    params = initialize_params(structure, rng_key)
    clamps = _data()
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    state = EPCInference.begin_segment(params, state, clamps, structure)
    derived = EPCInference.derive_states(params, state, clamps, structure)

    edge_key = "x->h:in"
    scale = structure.nodes["h"].node_info.scaling_config.forward_scale[edge_key]
    expected = clamps["x"] * scale @ params.nodes["h"].weights[edge_key]
    expected = expected + params.nodes["h"].biases["b"]
    assert jnp.allclose(derived.nodes["h"].z_mu, expected, atol=1e-6)


def test_one_step_gradient_is_independent_of_node_insertion_order(rng_key):
    forward = _chain(EPCInference(infer_steps=1), ("x", "h", "y"))
    reverse = _chain(EPCInference(infer_steps=1), ("y", "h", "x"))
    params = initialize_params(forward, rng_key)
    reverse_params = GraphParams(
        nodes={name: params.nodes[name] for name in reverse.nodes}
    )
    clamps = _data()
    forward_state = initialize_graph_state(
        forward, 2, rng_key, clamps=clamps, params=params
    )
    reverse_state = initialize_graph_state(
        reverse, 2, rng_key, clamps=clamps, params=reverse_params
    )
    forward_state = EPCInference.inference_step(
        params,
        forward_state,
        clamps,
        forward,
        forward.config["inference"].config,
    )
    reverse_state = EPCInference.inference_step(
        reverse_params,
        reverse_state,
        clamps,
        reverse,
        reverse.config["inference"].config,
    )
    for name in forward.nodes:
        assert jnp.allclose(
            forward_state.nodes[name].latent_grad,
            reverse_state.nodes[name].latent_grad,
            atol=1e-6,
        )


def test_hopfield_in_forward_energy_is_differentiated(rng_key):
    x = IdentityNode(shape=(4,), name="x")
    memory = StorkeyHopfield(shape=(4,), name="memory", hopfield_strength=0.5)
    y = Linear(shape=(2,), name="y", activation=IdentityActivation())
    structure = graph(
        nodes=[x, memory, y],
        edges=[
            Edge(source=x, target=memory.slot("in")),
            Edge(source=memory, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=EPCInference(eta_infer=0.01, infer_steps=1),
    )
    params = initialize_params(structure, rng_key)
    clamps = {
        "x": jnp.ones((2, 4), dtype=jnp.float32),
        "y": jnp.zeros((2, 2), dtype=jnp.float32),
    }
    state = initialize_graph_state(structure, 2, rng_key, clamps=clamps, params=params)
    state = EPCInference.begin_segment(params, state, clamps, structure)
    epsilon = jnp.full_like(state.nodes["memory"].error, 0.1)
    state = update_node_in_state(state, "memory", error=epsilon)
    state = EPCInference.zero_grads(params, state, clamps, structure)
    result = EPCInference.forward_value_and_grad(params, state, clamps, structure)

    memory_state = result.nodes["memory"]
    pc_energy = 0.5 * jnp.sum(memory_state.error**2, axis=-1)
    assert jnp.any(jnp.abs(memory_state.energy - pc_energy) > 1e-7)
    assert jnp.all(jnp.isfinite(memory_state.latent_grad))
