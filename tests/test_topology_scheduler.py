"""Tests for graph topology scheduling and explicit cyclic behavior."""

import math

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.inference import InferenceSGD, gather_inputs
from fabricpc.core.mupc import (
    MuPCConfig,
    _count_skip_connections_depth,
    compute_mupc_scalings,
)
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import (
    DAGScheduler,
    GraphCycleError,
    TaskMap,
    TopologySchedulerBase,
    UnrolledCycleScheduler,
    graph,
)
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import IdentityNode, Linear, SkipConnection


def _chain(node_order=("x", "h", "y"), topology_scheduler=None):
    x = IdentityNode(shape=(3,), name="x")
    h = Linear(shape=(4,), name="h")
    y = Linear(shape=(2,), name="y")
    nodes = {node.name: node for node in (x, h, y)}
    return graph(
        nodes=[nodes[name] for name in node_order],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(infer_steps=1),
        topology_scheduler=topology_scheduler,
    )


def _cycle(topology_scheduler=None, scaling=None):
    x = IdentityNode(shape=(3,), name="x")
    a = Linear(shape=(4,), name="a")
    b = Linear(shape=(4,), name="b")
    y = Linear(shape=(2,), name="y")
    return graph(
        nodes=[x, a, b, y],
        edges=[
            Edge(source=x, target=a.slot("in")),
            Edge(source=a, target=b.slot("in")),
            Edge(source=b, target=a.slot("in")),
            Edge(source=b, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(infer_steps=1),
        topology_scheduler=topology_scheduler,
        scaling=scaling,
    )


def _residual_cycle(num_unrolls):
    """Cycle with two residual merges and a two-source feedback target."""
    x = IdentityNode(shape=(4,), name="x")
    a = Linear(shape=(4,), name="a")
    merge1 = SkipConnection(shape=(4,), name="merge1")
    b = Linear(shape=(4,), name="b")
    merge2 = SkipConnection(shape=(4,), name="merge2")
    feedback = Linear(shape=(4,), name="feedback")
    y = Linear(shape=(2,), name="y")
    return graph(
        nodes=[x, a, merge1, b, merge2, feedback, y],
        edges=[
            Edge(source=x, target=a.slot("in")),
            Edge(source=feedback, target=a.slot("in")),
            Edge(source=a, target=merge1.slot("in")),
            Edge(source=x, target=merge1.slot("skip")),
            Edge(source=merge1, target=b.slot("in")),
            Edge(source=b, target=merge2.slot("in")),
            Edge(source=merge1, target=merge2.slot("skip")),
            Edge(source=merge2, target=feedback.slot("in")),
            Edge(source=merge2, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(infer_steps=1),
        topology_scheduler=UnrolledCycleScheduler(num_unrolls=num_unrolls),
        scaling=MuPCConfig(include_output=True),
    )


@pytest.mark.parametrize(
    "insertion_order",
    [("x", "h", "y"), ("y", "h", "x"), ("h", "x", "y")],
)
def test_dag_scheduler_preserves_legacy_kahn_order(insertion_order):
    structure = _chain(insertion_order)
    assert structure.schedule == ("x", "h", "y")
    assert structure.node_order == structure.schedule
    assert isinstance(structure.config["topology_scheduler"], DAGScheduler)


def test_default_scheduler_rejects_cycles_and_names_unordered_nodes():
    with pytest.raises(GraphCycleError, match=r"a.*b.*y"):
        _cycle()


def test_unrolled_cycle_schedule_and_unique_order():
    structure = _cycle(UnrolledCycleScheduler(num_unrolls=2))
    assert structure.schedule == ("x", "a", "b", "a", "b", "y")
    assert structure.node_order == ("x", "a", "b", "y")


def test_single_unroll_is_explicit_degenerate_cycle_schedule():
    structure = _cycle(UnrolledCycleScheduler(num_unrolls=1))
    assert structure.schedule == ("x", "a", "b", "y")
    assert structure.node_order == structure.schedule


@pytest.mark.parametrize("num_unrolls", [1, 2, 5])
def test_cycle_scheduler_matches_dag_scheduler_on_dag(num_unrolls):
    dag = _chain()
    scheduled = _chain(
        topology_scheduler=UnrolledCycleScheduler(num_unrolls=num_unrolls)
    )
    assert scheduled.schedule == dag.schedule


@pytest.mark.parametrize("value", [0, -1])
def test_unroll_count_must_be_positive(value):
    with pytest.raises(ValueError, match=">= 1"):
        UnrolledCycleScheduler(num_unrolls=value)


@pytest.mark.parametrize("value", [1.5, True])
def test_unroll_count_must_be_an_integer(value):
    with pytest.raises(TypeError, match="integer"):
        UnrolledCycleScheduler(num_unrolls=value)


class _FixedScheduler(TopologySchedulerBase):
    @staticmethod
    def compute_schedule(nodes, edges, config):
        del nodes, edges
        return tuple(config["schedule"])


@pytest.mark.parametrize(
    "schedule, expected",
    [(("x", "h"), "Missing"), (("x", "h", "y", "ghost"), "unknown")],
)
def test_graph_validates_custom_scheduler_coverage(schedule, expected):
    with pytest.raises(ValueError, match=expected):
        _chain(topology_scheduler=_FixedScheduler(schedule=schedule))


def test_unrolled_cycle_schedule_is_deterministic():
    first = _cycle(UnrolledCycleScheduler(num_unrolls=3))
    second = _cycle(UnrolledCycleScheduler(num_unrolls=3))
    assert first.schedule == second.schedule


def test_cycle_feedforward_initialization_replays_repeated_visits():
    once = _cycle(UnrolledCycleScheduler(num_unrolls=1))
    twice = _cycle(UnrolledCycleScheduler(num_unrolls=2))
    rng_key = jax.random.PRNGKey(7)
    params = initialize_params(once, rng_key)
    clamps = {
        "x": jnp.asarray([[0.2, -0.1, 0.4]], dtype=jnp.float32),
        "y": jnp.asarray([[0.3, -0.2]], dtype=jnp.float32),
    }
    once_state = initialize_graph_state(once, 1, rng_key, clamps=clamps, params=params)
    twice_state = initialize_graph_state(
        twice, 1, rng_key, clamps=clamps, params=params
    )

    manual = once_state
    for name in ("a", "b", "y"):
        info = once.nodes[name].node_info
        inputs = scale_inputs(gather_inputs(info, once, manual), info.scaling_config)
        projected = info.node_class.forward(
            params.nodes[name], inputs, manual.nodes[name], info
        )
        if name in clamps:
            node_state = manual.nodes[name]._replace(
                z_mu=projected.z_mu,
                error=projected.error,
                energy=projected.energy,
            )
        else:
            node_state = manual.nodes[name]._replace(
                z_latent=projected.z_mu, z_mu=projected.z_mu
            )
        manual = manual._replace(nodes={**manual.nodes, name: node_state})

    for name in twice.nodes:
        for field in twice_state.nodes[name]._fields:
            assert jnp.allclose(
                getattr(twice_state.nodes[name], field),
                getattr(manual.nodes[name], field),
            )
    assert not jnp.allclose(
        once_state.nodes["a"].z_latent, twice_state.nodes["a"].z_latent
    )


def test_mupc_uses_unique_cycle_order_independent_of_unroll_count():
    one = _cycle(
        UnrolledCycleScheduler(num_unrolls=1),
        scaling=MuPCConfig(include_output=True),
    )
    five = _cycle(
        UnrolledCycleScheduler(num_unrolls=5),
        scaling=MuPCConfig(include_output=True),
    )
    assert one.node_order == five.node_order
    for name in ("a", "b", "y"):
        assert one.nodes[name].node_info.scaling_config is not None
        assert (
            one.nodes[name].node_info.scaling_config
            == five.nodes[name].node_info.scaling_config
        )


def test_cyclic_mupc_uses_expected_slot_degree_and_residual_depth():
    one = _residual_cycle(num_unrolls=1)
    five = _residual_cycle(num_unrolls=5)

    assert one.node_order == five.node_order
    assert _count_skip_connections_depth(one.nodes, one.edges, one.node_order) == 2
    assert _count_skip_connections_depth(five.nodes, five.edges, five.node_order) == 2

    for name in one.nodes:
        if name != "x":
            assert one.nodes[name].node_info.scaling_config is not None
            assert (
                one.nodes[name].node_info.scaling_config
                == five.nodes[name].node_info.scaling_config
            )

    a_scaling = one.nodes["a"].node_info.scaling_config
    expected_two_source_scale = 1.0 / math.sqrt(4.0 * 2.0)
    assert a_scaling.forward_scale["x->a:in"] == pytest.approx(
        expected_two_source_scale
    )
    assert a_scaling.forward_scale["feedback->a:in"] == pytest.approx(
        expected_two_source_scale
    )

    expected_merge_scale = 1.0 / math.sqrt(2.0)
    assert one.nodes["merge1"].node_info.scaling_config.forward_scale[
        "a->merge1:in"
    ] == pytest.approx(expected_merge_scale)
    assert one.nodes["merge2"].node_info.scaling_config.forward_scale[
        "b->merge2:in"
    ] == pytest.approx(expected_merge_scale)


def test_mupc_rejects_duplicate_node_order():
    structure = _chain()
    with pytest.raises(ValueError, match="duplicate"):
        compute_mupc_scalings(
            structure.nodes,
            structure.edges,
            MuPCConfig(),
            list(structure.node_order) + ["h"],
        )
