"""
Test suite for muPC (Maximal Update Parameterization for Predictive Coding).

Covers:
- Depth metric computation (shortest path, longest path, fixed)
- MuPCInitializer weight initialization
- MuPCScalingFactors computation from graph topology
- Activation gain and jacobian gain integration
- End-to-end training with muPC scaling
"""

import math
import warnings
import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.core.activations import IdentityActivation, ReLUActivation, TanhActivation
from fabricpc.core.mupc import MuPCConfig, MuPCScalingFactors
from fabricpc.core.depth_metric import (
    ShortestPathDepth,
    LongestPathDepth,
    FixedDepth,
)
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.graph.graph_net import (
    compute_local_weight_gradients,
    set_latents_to_clamps,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def linear_chain_with_mupc():
    """Linear chain graph built with muPC scaling: x(10) -> h(20,ReLU) -> y(5)."""
    x = IdentityNode(shape=(10,), name="x")
    h = Linear(
        shape=(20,),
        name="h",
        activation=ReLUActivation(),
        weight_init=MuPCInitializer(),
    )
    y = Linear(
        shape=(5,),
        name="y",
        activation=IdentityActivation(),
        weight_init=MuPCInitializer(),
    )
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=5),
        scaling=MuPCConfig(),
    )


@pytest.fixture
def skip_connection_structure():
    """4-node graph with a skip connection: x -> h1 -> h2 -> y, x -> h2."""
    x = IdentityNode(shape=(10,), name="x")
    h1 = Linear(shape=(20,), name="h1", weight_init=MuPCInitializer())
    h2 = Linear(shape=(20,), name="h2", weight_init=MuPCInitializer())
    y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
    return graph(
        nodes=[x, h1, h2, y],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=h2.slot("in")),
            Edge(source=x, target=h2.slot("in")),  # skip
            Edge(source=h2, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=5),
        scaling=MuPCConfig(),
    )


# ============================================================================
# Depth Metric Tests
# ============================================================================


class TestDepthMetrics:
    """Test depth computation for various graph topologies."""

    def test_shortest_path(self, linear_chain_with_mupc, skip_connection_structure):
        """Shortest path depths: linear chain and skip connection."""
        metric = ShortestPathDepth()

        # Linear chain: 0, 1, 2
        depths = metric.compute(
            linear_chain_with_mupc.nodes, linear_chain_with_mupc.edges
        )
        assert depths == {"x": 0, "h": 1, "y": 2}

        # Skip: h2 gets depth 1 via direct x->h2 edge
        depths = metric.compute(
            skip_connection_structure.nodes, skip_connection_structure.edges
        )
        assert depths["h2"] == 1

    def test_longest_path(self, linear_chain_with_mupc, skip_connection_structure):
        """Longest path depths: linear chain and skip connection."""
        metric = LongestPathDepth()

        # Linear chain: same as shortest
        depths = metric.compute(
            linear_chain_with_mupc.nodes, linear_chain_with_mupc.edges
        )
        assert depths == {"x": 0, "h": 1, "y": 2}

        # Skip: h2 gets depth 2 via x->h1->h2
        depths = metric.compute(
            skip_connection_structure.nodes, skip_connection_structure.edges
        )
        assert depths["h2"] == 2
        assert depths["y"] == 3

    def test_fixed_depth(self, linear_chain_with_mupc):
        """Fixed depth assigns same depth to all non-source nodes."""
        metric = FixedDepth(depth=5)
        depths = metric.compute(
            linear_chain_with_mupc.nodes, linear_chain_with_mupc.edges
        )
        assert depths["x"] == 0
        assert depths["h"] == 5
        assert depths["y"] == 5


# ============================================================================
# MuPCInitializer Tests
# ============================================================================


class TestMuPCInitializer:
    """Test MuPCInitializer produces correctly scaled weights."""

    def test_standard_normal(self, rng_key):
        """Weights should be drawn from N(0, 1)."""
        init = MuPCInitializer()
        W = MuPCInitializer.initialize(rng_key, (500, 500), init.config)
        assert W.shape == (500, 500)
        assert abs(float(jnp.mean(W))) < 0.1
        assert abs(float(jnp.std(W)) - 1.0) < 0.1

    def test_gain_scaling(self, rng_key):
        """Gain should scale the standard deviation."""
        init = MuPCInitializer(gain=0.5)
        W = MuPCInitializer.initialize(rng_key, (500, 500), init.config)
        assert abs(float(jnp.std(W)) - 0.5) < 0.1


# ============================================================================
# MuPC Scaling Computation Tests
# ============================================================================


class TestMuPCScaling:
    """Test scaling factor computation from graph topology."""

    def test_scaling_attachment_by_node_role(self, linear_chain_with_mupc):
        """Source=None, hidden=MuPCScalingFactors, output=None."""
        s = linear_chain_with_mupc
        assert s.nodes["x"].node_info.scaling_config is None
        assert isinstance(s.nodes["h"].node_info.scaling_config, MuPCScalingFactors)
        assert s.nodes["y"].node_info.scaling_config is None

    def test_hidden_forward_scale_formula(self, linear_chain_with_mupc):
        """Hidden node forward scale = gain/sqrt(fan_in * K)."""
        h_info = linear_chain_with_mupc.nodes["h"].node_info
        scaling = h_info.scaling_config

        # h: fan_in=10, K=1, ReLU gain=sqrt(2)
        expected_a = math.sqrt(2.0) / math.sqrt(10 * 1)
        edge_key = h_info.in_edges[0]
        assert abs(scaling.forward_scale[edge_key] - expected_a) < 1e-10

    def test_skip_connection_per_edge_scales(self, skip_connection_structure):
        """Node with multiple inputs should have separate scales per edge."""
        h2_info = skip_connection_structure.nodes["h2"].node_info
        scaling = h2_info.scaling_config
        assert len(scaling.forward_scale) == 2
        assert len(scaling.topdown_grad_scale) == 2

    def test_no_scaling_without_config(self):
        """Without scaling parameter, all nodes have scaling_config=None."""
        x = IdentityNode(shape=(10,), name="x")
        h = Linear(shape=(20,), name="h")
        structure = graph(
            nodes=[x, h],
            edges=[Edge(source=x, target=h.slot("in"))],
            task_map=TaskMap(x=x, y=h),
            inference=InferenceSGD(),
        )
        assert structure.nodes["x"].node_info.scaling_config is None
        assert structure.nodes["h"].node_info.scaling_config is None

    def test_invalid_scaling_type_raises(self):
        """Passing non-MuPCConfig should raise TypeError."""
        x = IdentityNode(shape=(10,), name="x")
        h = Linear(shape=(20,), name="h")
        with pytest.raises(TypeError, match="MuPCConfig"):
            graph(
                nodes=[x, h],
                edges=[Edge(source=x, target=h.slot("in"))],
                task_map=TaskMap(x=x, y=h),
                inference=InferenceSGD(),
                scaling="not_a_config",
            )

    def test_include_output_with_formula(self):
        """With include_output=True, output gets scaling a = 1/(fan_in * sqrt(K))."""
        x = IdentityNode(shape=(10,), name="x")
        h = Linear(shape=(20,), name="h", weight_init=MuPCInitializer())
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(include_output=True),
        )
        scaling = structure.nodes["y"].node_info.scaling_config
        assert scaling is not None
        assert isinstance(scaling, MuPCScalingFactors)
        # y: fan_in=20, K=1 -> a = 1/(20 * 1) = 0.05
        edge_key = structure.nodes["y"].node_info.in_edges[0]
        assert abs(scaling.forward_scale[edge_key] - 1.0 / 20) < 1e-10


# ============================================================================
# IdentityNode Scaling Tests
# ============================================================================


class TestIdentityNodeScaling:
    """Test that IdentityNode sum junctions get correct muPC scaling."""

    def test_identity_junction_scaling(self):
        """IdentityNode with K=2 in-edges gets a=1/sqrt(K) per edge (fan_in=1)."""
        x = IdentityNode(shape=(10,), name="x")
        h1 = Linear(shape=(20,), name="h1", weight_init=MuPCInitializer())
        h2 = Linear(shape=(20,), name="h2", weight_init=MuPCInitializer())
        sum_node = IdentityNode(shape=(20,), name="sum")
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h1, h2, sum_node, y],
            edges=[
                Edge(source=x, target=h1.slot("in")),
                Edge(source=h1, target=h2.slot("in")),
                Edge(source=h2, target=sum_node.slot("in")),
                Edge(source=h1, target=sum_node.slot("in")),  # skip
                Edge(source=sum_node, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=5),
            scaling=MuPCConfig(),
        )
        scaling = structure.nodes["sum"].node_info.scaling_config
        assert scaling is not None
        expected_a = 1.0 / math.sqrt(2)  # fan_in=1, K=2
        for a in scaling.forward_scale.values():
            assert abs(a - expected_a) < 1e-10


# ============================================================================
# Activation and Jacobian Gain Tests
# ============================================================================


class TestActivationAndGradientGain:
    """Test activation gain and jacobian gain integration with scaling factors."""

    def test_gain_included_in_forward_scale(self):
        """Forward scale includes activation gain: a = gain/sqrt(fan_in*K)."""
        x = IdentityNode(shape=(10,), name="x")
        h = Linear(
            shape=(20,),
            name="h",
            activation=TanhActivation(),
            weight_init=MuPCInitializer(),
        )
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        h_info = structure.nodes["h"].node_info
        edge_key = h_info.in_edges[0]
        actual_a = h_info.scaling_config.forward_scale[edge_key]
        # tanh gain = sqrt(5/3), fan_in=10, K=1
        expected_a = math.sqrt(5.0 / 3.0) / math.sqrt(10)
        assert abs(actual_a - expected_a) < 1e-10

    def test_jacobian_gain_in_topdown_scale(self):
        """For tanh, topdown_grad_scale = forward_scale * jacobian_gain."""
        x = IdentityNode(shape=(10,), name="x")
        h = Linear(
            shape=(20,),
            name="h",
            activation=TanhActivation(),
            weight_init=MuPCInitializer(),
        )
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        h_info = structure.nodes["h"].node_info
        edge_key = h_info.in_edges[0]
        fwd = h_info.scaling_config.forward_scale[edge_key]
        td = h_info.scaling_config.topdown_grad_scale[edge_key]
        expected_td = fwd * TanhActivation.jacobian_gain()
        assert abs(td - expected_td) < 1e-10
        # tanh jacobian_gain != 1.0, so topdown != forward
        assert td != fwd

    def test_deep_tanh_chain_no_activation_collapse(self):
        """100-layer tanh chain with muPC gain should maintain O(1) activations."""
        width = 32
        num_hidden = 100
        x = IdentityNode(shape=(width,), name="x")
        layers = [
            Linear(
                shape=(width,),
                name=f"h{i}",
                activation=TanhActivation(),
                weight_init=MuPCInitializer(),
            )
            for i in range(num_hidden)
        ]
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())

        all_nodes = [x] + layers + [y]
        all_edges = []
        prev = x
        for h in layers:
            all_edges.append(Edge(source=prev, target=h.slot("in")))
            prev = h
        all_edges.append(Edge(source=prev, target=y.slot("in")))

        structure = graph(
            nodes=all_nodes,
            edges=all_edges,
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=10),
            scaling=MuPCConfig(),
        )

        rng_key = jax.random.PRNGKey(42)
        params = initialize_params(structure, rng_key)
        state = initialize_graph_state(structure, 32, rng_key, params=params)

        last_hidden = f"h{num_hidden - 1}"
        var_last = float(jnp.var(state.nodes[last_hidden].z_latent))
        assert var_last > 0.01, f"Activations collapsed: var={var_last}"
        assert var_last < 100.0, f"Activations exploded: var={var_last}"


# ============================================================================
# Variance Propagation Tests
# ============================================================================


class TestVariancePropagation:
    """Test the unified muPC scaling formula a=1/sqrt(fan_in*K)."""

    def test_chain_scaling_independent_of_depth(self):
        """Same-width hidden nodes get identical scaling regardless of depth."""
        x = IdentityNode(shape=(10,), name="x")
        h1 = Linear(shape=(20,), name="h1", weight_init=MuPCInitializer())
        h2 = Linear(shape=(20,), name="h2", weight_init=MuPCInitializer())
        h3 = Linear(shape=(20,), name="h3", weight_init=MuPCInitializer())
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h1, h2, h3, y],
            edges=[
                Edge(source=x, target=h1.slot("in")),
                Edge(source=h1, target=h2.slot("in")),
                Edge(source=h2, target=h3.slot("in")),
                Edge(source=h3, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        # h1: fan_in=10 -> a=1/sqrt(10)
        h1_edge = structure.nodes["h1"].node_info.in_edges[0]
        assert (
            abs(
                structure.nodes["h1"].node_info.scaling_config.forward_scale[h1_edge]
                - 1.0 / math.sqrt(10)
            )
            < 1e-10
        )

        # h2, h3: fan_in=20 -> a=1/sqrt(20), both identical
        h2_edge = structure.nodes["h2"].node_info.in_edges[0]
        h3_edge = structure.nodes["h3"].node_info.in_edges[0]
        a_h2 = structure.nodes["h2"].node_info.scaling_config.forward_scale[h2_edge]
        a_h3 = structure.nodes["h3"].node_info.scaling_config.forward_scale[h3_edge]
        assert abs(a_h2 - 1.0 / math.sqrt(20)) < 1e-10
        assert abs(a_h2 - a_h3) < 1e-10


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Test deprecated MuPCConfig parameters."""

    def test_deprecated_params_emit_warnings(self):
        """depth_metric and min_depth emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="depth_metric.*deprecated"):
            MuPCConfig(depth_metric=ShortestPathDepth())
        with pytest.warns(DeprecationWarning, match="min_depth.*deprecated"):
            MuPCConfig(min_depth=3)

    def test_deprecated_config_still_works(self):
        """Graph builds correctly even with deprecated parameters."""
        x = IdentityNode(shape=(10,), name="x")
        h = Linear(shape=(20,), name="h", weight_init=MuPCInitializer())
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            structure = graph(
                nodes=[x, h, y],
                edges=[
                    Edge(source=x, target=h.slot("in")),
                    Edge(source=h, target=y.slot("in")),
                ],
                task_map=TaskMap(x=x, y=y),
                inference=InferenceSGD(),
                scaling=MuPCConfig(depth_metric=ShortestPathDepth()),
            )
        assert structure.nodes["h"].node_info.scaling_config is not None


# ============================================================================
# End-to-End Tests
# ============================================================================


class TestEndToEnd:
    """Test full inference and training with muPC scaling."""

    def test_inference_and_gradients_valid(self, rng_key, linear_chain_with_mupc):
        """Inference produces valid state; weight gradients are finite and nonzero."""
        structure = linear_chain_with_mupc
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x_data = jax.random.normal(rng_key, (batch_size, 10))
        y_data = jax.random.normal(rng_key, (batch_size, 5))

        state = initialize_graph_state(structure, batch_size, rng_key, params=params)
        clamps = {"x": x_data, "y": y_data}
        state = set_latents_to_clamps(state, clamps)

        final_state = run_inference(params, state, clamps, structure)

        # All nodes have valid (non-NaN) state
        for node_name in structure.nodes:
            ns = final_state.nodes[node_name]
            assert not jnp.any(jnp.isnan(ns.z_latent)), f"NaN in {node_name}.z_latent"
            assert not jnp.any(jnp.isnan(ns.z_mu)), f"NaN in {node_name}.z_mu"

        # Weight gradients are finite and nonzero
        grad_params = compute_local_weight_gradients(params, final_state, structure)
        for node_name in ["h", "y"]:
            node_grads = grad_params.nodes[node_name]
            for edge_key, wg in node_grads.weights.items():
                assert not jnp.any(jnp.isnan(wg)), f"NaN in grad {node_name}/{edge_key}"
                assert jnp.any(wg != 0), f"All-zero grad {node_name}/{edge_key}"

    def test_train_step_reduces_energy(self, rng_key):
        """Training steps should reduce total energy."""
        import optax
        from fabricpc.training import train_step

        x = IdentityNode(shape=(10,), name="x")
        h = Linear(
            shape=(20,),
            name="h",
            activation=ReLUActivation(),
            weight_init=MuPCInitializer(),
        )
        y = Linear(
            shape=(5,),
            name="y",
            activation=IdentityActivation(),
            weight_init=MuPCInitializer(),
        )
        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=10),
            scaling=MuPCConfig(),
        )

        params = initialize_params(structure, rng_key)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        batch_size = 8
        k1, k2, k3 = jax.random.split(rng_key, 3)
        x_data = jax.random.normal(k1, (batch_size, 10))
        y_data = jax.random.normal(k2, (batch_size, 5))
        batch = {"x": x_data, "y": y_data}

        # Initial energy
        state0 = initialize_graph_state(structure, batch_size, rng_key, params=params)
        state0 = set_latents_to_clamps(state0, batch)
        state0 = run_inference(params, state0, batch, structure)
        energy_0 = sum(float(jnp.mean(state0.nodes[n].energy)) for n in structure.nodes)

        # Train for a few steps
        for i in range(5):
            step_key = jax.random.fold_in(k3, i)
            params, opt_state, loss, _ = train_step(
                params, opt_state, batch, structure, optimizer, step_key
            )

        # Final energy
        state_f = initialize_graph_state(structure, batch_size, rng_key, params=params)
        state_f = set_latents_to_clamps(state_f, batch)
        state_f = run_inference(params, state_f, batch, structure)
        energy_f = sum(
            float(jnp.mean(state_f.nodes[n].energy)) for n in structure.nodes
        )

        assert (
            energy_f < energy_0
        ), f"Energy did not decrease: {energy_0:.6f} -> {energy_f:.6f}"
