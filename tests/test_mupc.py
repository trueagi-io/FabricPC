"""
Test suite for muPC (Maximal Update Parameterization for Predictive Coding).

Covers:
- Depth metric computation (shortest path, longest path, fixed)
- MuPCInitializer weight initialization
- MuPCScalingFactors computation from graph topology
- Graph builder integration with scaling parameter
- Forward scaling application in forward_inference and forward_learning
- End-to-end training with muPC scaling
"""

import math
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import MuPCInitializer, NormalInitializer
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.depth_metric import (
    ShortestPathDepth,
    LongestPathDepth,
    FixedDepth,
)
from fabricpc.core.mupc import MuPCConfig, MuPCScalingFactors, compute_mupc_scalings
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.graph.graph_net import (
    compute_local_weight_gradients,
    set_latents_to_clamps,
)
from conftest import with_inference

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def linear_chain_structure():
    """A simple 3-node linear chain: input -> hidden -> output."""
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
    return x, h, y


@pytest.fixture
def linear_chain_with_mupc(linear_chain_structure):
    """Linear chain graph built with muPC scaling."""
    x, h, y = linear_chain_structure
    structure = graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=5),
        scaling=MuPCConfig(),
    )
    return structure


@pytest.fixture
def skip_connection_structure():
    """4-node graph with a skip connection: x -> h1 -> h2 -> y, x -> h2."""
    x = IdentityNode(shape=(10,), name="x")
    h1 = Linear(shape=(20,), name="h1", weight_init=MuPCInitializer())
    h2 = Linear(shape=(20,), name="h2", weight_init=MuPCInitializer())
    y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
    structure = graph(
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
    return structure


# ============================================================================
# Depth Metric Tests
# ============================================================================


class TestDepthMetrics:
    """Test depth computation for various graph topologies."""

    def test_shortest_path_linear_chain(self, linear_chain_with_mupc):
        """Shortest path depths for a linear chain: 0, 1, 2."""
        structure = linear_chain_with_mupc
        metric = ShortestPathDepth()
        depths = metric.compute(structure.nodes, structure.edges)
        assert depths["x"] == 0
        assert depths["h"] == 1
        assert depths["y"] == 2

    def test_shortest_path_with_skip(self, skip_connection_structure):
        """Skip connection gives h2 a shorter path (1 instead of 2)."""
        structure = skip_connection_structure
        metric = ShortestPathDepth()
        depths = metric.compute(structure.nodes, structure.edges)
        assert depths["x"] == 0
        assert depths["h1"] == 1
        assert depths["h2"] == 1  # shortest via skip: x -> h2
        assert depths["y"] == 2

    def test_longest_path_linear_chain(self, linear_chain_with_mupc):
        """Longest path same as shortest for linear chain."""
        structure = linear_chain_with_mupc
        metric = LongestPathDepth()
        depths = metric.compute(structure.nodes, structure.edges)
        assert depths["x"] == 0
        assert depths["h"] == 1
        assert depths["y"] == 2

    def test_longest_path_with_skip(self, skip_connection_structure):
        """Longest path gives h2 depth 2 (via h1), not 1."""
        structure = skip_connection_structure
        metric = LongestPathDepth()
        depths = metric.compute(structure.nodes, structure.edges)
        assert depths["x"] == 0
        assert depths["h1"] == 1
        assert depths["h2"] == 2  # longest via h1: x -> h1 -> h2
        assert depths["y"] == 3

    def test_fixed_depth(self, linear_chain_with_mupc):
        """Fixed depth assigns same depth to all nodes with in_degree > 0."""
        structure = linear_chain_with_mupc
        metric = FixedDepth(depth=5)
        depths = metric.compute(structure.nodes, structure.edges)
        assert depths["x"] == 0
        assert depths["h"] == 5
        assert depths["y"] == 5

    def test_fixed_depth_validation(self):
        """FixedDepth rejects depth < 1."""
        with pytest.raises(ValueError, match="must be >= 1"):
            FixedDepth(depth=0)


# ============================================================================
# MuPCInitializer Tests
# ============================================================================


class TestMuPCInitializer:
    """Test MuPCInitializer produces unit-variance weights."""

    def test_standard_normal(self, rng_key):
        """Weights should be drawn from N(0, 1)."""
        init = MuPCInitializer()
        W = MuPCInitializer.initialize(rng_key, (500, 500), init.config)
        assert W.shape == (500, 500)
        # Mean near 0, std near 1
        assert abs(float(jnp.mean(W))) < 0.1
        assert abs(float(jnp.std(W)) - 1.0) < 0.1

    def test_gain_scaling(self, rng_key):
        """Gain should scale the standard deviation."""
        init = MuPCInitializer(gain=0.5)
        W = MuPCInitializer.initialize(rng_key, (500, 500), init.config)
        assert abs(float(jnp.std(W)) - 0.5) < 0.1

    def test_different_from_normal_init(self, rng_key):
        """MuPC init should have different variance than NormalInitializer (std=0.05)."""
        mupc_init = MuPCInitializer()
        normal_init = NormalInitializer()
        W_mupc = MuPCInitializer.initialize(rng_key, (200, 200), mupc_init.config)
        W_normal = NormalInitializer.initialize(rng_key, (200, 200), normal_init.config)
        # MuPC std ≈ 1.0, Normal std ≈ 0.05
        assert float(jnp.std(W_mupc)) > 5 * float(jnp.std(W_normal))


# ============================================================================
# MuPC Scaling Computation Tests
# ============================================================================


class TestMuPCScalingComputation:
    """Test that scaling factors are computed correctly from graph topology."""

    def test_source_node_gets_no_scaling(self, linear_chain_with_mupc):
        """Source nodes (in_degree=0) should have no scaling."""
        structure = linear_chain_with_mupc
        x_info = structure.nodes["x"].node_info
        assert x_info.scaling_config is None

    def test_hidden_node_has_scaling(self, linear_chain_with_mupc):
        """Hidden nodes should have MuPCScalingFactors attached."""
        structure = linear_chain_with_mupc
        h_info = structure.nodes["h"].node_info
        assert h_info.scaling_config is not None
        assert isinstance(h_info.scaling_config, MuPCScalingFactors)

    def test_output_node_has_no_scaling(self, linear_chain_with_mupc):
        """Output nodes (out_degree=0) use standard init, no muPC forward scaling."""
        structure = linear_chain_with_mupc
        y_info = structure.nodes["y"].node_info
        assert y_info.scaling_config is None

    def test_hidden_forward_scale_formula(self, linear_chain_with_mupc):
        """Hidden node forward scale = 1/sqrt(fan_in * K) where K=in_degree."""
        structure = linear_chain_with_mupc
        h_info = structure.nodes["h"].node_info
        scaling = h_info.scaling_config

        # h has one input from x (shape=(10,)), K=1 (in_degree)
        fan_in = 10  # x.shape[-1]
        K = 1
        expected_a = 1.0 / math.sqrt(fan_in * K)

        edge_key = h_info.in_edges[0]
        actual_a = scaling.forward_scale[edge_key]
        assert abs(actual_a - expected_a) < 1e-10

    def test_topdown_grad_scale_disabled(self, linear_chain_with_mupc):
        """Top-down gradient scale is 1.0 (disabled, matches jpc reference)."""
        structure = linear_chain_with_mupc
        h_info = structure.nodes["h"].node_info
        scaling = h_info.scaling_config

        edge_key = h_info.in_edges[0]
        assert scaling.topdown_grad_scale[edge_key] == 1.0

    def test_self_grad_scale_default(self, linear_chain_with_mupc):
        """Self-gradient scale defaults to 1.0."""
        structure = linear_chain_with_mupc
        h_info = structure.nodes["h"].node_info
        assert h_info.scaling_config.self_grad_scale == 1.0

    def test_weight_grad_scale_default(self, linear_chain_with_mupc):
        """Weight gradient scale defaults to 1.0."""
        structure = linear_chain_with_mupc
        h_info = structure.nodes["h"].node_info
        edge_key = h_info.in_edges[0]
        assert h_info.scaling_config.weight_grad_scale[edge_key] == 1.0

    def test_multi_input_node_has_per_edge_scales(self, skip_connection_structure):
        """Node with multiple inputs should have separate scales per edge."""
        structure = skip_connection_structure
        h2_info = structure.nodes["h2"].node_info
        scaling = h2_info.scaling_config
        # h2 has 2 incoming edges (from h1 and from x)
        assert len(scaling.forward_scale) == 2
        assert len(scaling.topdown_grad_scale) == 2


# ============================================================================
# Graph Builder Integration Tests
# ============================================================================


class TestGraphBuilderIntegration:
    """Test that graph() correctly attaches muPC scalings."""

    def test_no_scaling_by_default(self):
        """Without scaling parameter, nodes should have scaling_config=None."""
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

    def test_scaling_attached_with_mupc_config(self, linear_chain_with_mupc):
        """With MuPCConfig, hidden nodes should have scaling; source/output should not."""
        structure = linear_chain_with_mupc
        assert structure.nodes["x"].node_info.scaling_config is None  # source
        assert structure.nodes["h"].node_info.scaling_config is not None  # hidden
        assert structure.nodes["y"].node_info.scaling_config is None  # output

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


# ============================================================================
# Forward Scaling Application Tests
# ============================================================================


class TestForwardScalingApplication:
    """Test that scaling is correctly applied during inference and learning."""

    def test_mupc_produces_smaller_activations(self, rng_key):
        """muPC scaling should produce smaller pre-activation magnitudes
        compared to no scaling, since inputs are divided by sqrt(fan_in*L)."""
        x = IdentityNode(shape=(100,), name="x")
        h = Linear(
            shape=(100,),
            name="h",
            activation=IdentityActivation(),
            weight_init=MuPCInitializer(),
        )
        y = Linear(
            shape=(10,),
            name="y",
            activation=IdentityActivation(),
            weight_init=MuPCInitializer(),
        )

        # Build with muPC scaling
        structure_mupc = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
            scaling=MuPCConfig(),
        )

        # Build without muPC scaling (same graph)
        x2 = IdentityNode(shape=(100,), name="x")
        h2 = Linear(
            shape=(100,),
            name="h",
            activation=IdentityActivation(),
            weight_init=MuPCInitializer(),
        )
        y2 = Linear(
            shape=(10,),
            name="y",
            activation=IdentityActivation(),
            weight_init=MuPCInitializer(),
        )
        structure_sp = graph(
            nodes=[x2, h2, y2],
            edges=[
                Edge(source=x2, target=h2.slot("in")),
                Edge(source=h2, target=y2.slot("in")),
            ],
            task_map=TaskMap(x=x2, y=y2),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
        )

        # Use same random key for both
        params_mupc = initialize_params(structure_mupc, rng_key)
        params_sp = initialize_params(structure_sp, rng_key)

        batch_size = 4
        x_data = jax.random.normal(rng_key, (batch_size, 100))

        # Initialize states
        state_mupc = initialize_graph_state(
            structure_mupc, batch_size, rng_key, params=params_mupc
        )
        state_sp = initialize_graph_state(
            structure_sp, batch_size, rng_key, params=params_sp
        )

        # Clamp inputs
        clamps = {"x": x_data}
        state_mupc = set_latents_to_clamps(state_mupc, clamps)
        state_sp = set_latents_to_clamps(state_sp, clamps)

        # Run inference
        final_mupc = run_inference(params_mupc, state_mupc, clamps, structure_mupc)
        final_sp = run_inference(params_sp, state_sp, clamps, structure_sp)

        # muPC z_mu at hidden node should have smaller magnitude (scaled down)
        h_mu_mupc = jnp.mean(jnp.abs(final_mupc.nodes["h"].z_mu))
        h_mu_sp = jnp.mean(jnp.abs(final_sp.nodes["h"].z_mu))
        assert float(h_mu_mupc) < float(h_mu_sp), (
            f"muPC hidden z_mu ({float(h_mu_mupc):.4f}) should be smaller "
            f"than SP hidden z_mu ({float(h_mu_sp):.4f})"
        )

    def test_inference_runs_without_error(self, rng_key, linear_chain_with_mupc):
        """Full inference loop should run without errors with muPC scaling."""
        structure = linear_chain_with_mupc
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x_data = jax.random.normal(rng_key, (batch_size, 10))
        y_data = jax.random.normal(rng_key, (batch_size, 5))

        state = initialize_graph_state(structure, batch_size, rng_key, params=params)
        clamps = {"x": x_data, "y": y_data}
        state = set_latents_to_clamps(state, clamps)

        final_state = run_inference(params, state, clamps, structure)

        # Check all nodes have valid state
        for node_name in structure.nodes:
            ns = final_state.nodes[node_name]
            assert not jnp.any(jnp.isnan(ns.z_latent)), f"NaN in {node_name}.z_latent"
            assert not jnp.any(jnp.isnan(ns.z_mu)), f"NaN in {node_name}.z_mu"

    def test_weight_gradients_computed_correctly(self, rng_key, linear_chain_with_mupc):
        """Weight gradients should be computed without errors with muPC."""
        structure = linear_chain_with_mupc
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x_data = jax.random.normal(rng_key, (batch_size, 10))
        y_data = jax.random.normal(rng_key, (batch_size, 5))

        state = initialize_graph_state(structure, batch_size, rng_key, params=params)
        clamps = {"x": x_data, "y": y_data}
        state = set_latents_to_clamps(state, clamps)

        # Run inference to convergence
        final_state = run_inference(params, state, clamps, structure)

        # Compute weight gradients
        grad_params = compute_local_weight_gradients(params, final_state, structure)

        # Check gradients exist and are finite
        for node_name in ["h", "y"]:
            node_grads = grad_params.nodes[node_name]
            for edge_key, wg in node_grads.weights.items():
                assert not jnp.any(
                    jnp.isnan(wg)
                ), f"NaN in weight gradient for {node_name}/{edge_key}"
                assert jnp.any(
                    wg != 0
                ), f"All-zero weight gradient for {node_name}/{edge_key}"


# ============================================================================
# End-to-End Training Tests
# ============================================================================


class TestEndToEndTraining:
    """Test that a full train step works with muPC scaling."""

    def test_train_step_reduces_energy(self, rng_key):
        """A training step should reduce total energy."""
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

        # Compute initial energy
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

        # Compute final energy
        state_f = initialize_graph_state(structure, batch_size, rng_key, params=params)
        state_f = set_latents_to_clamps(state_f, batch)
        state_f = run_inference(params, state_f, batch, structure)
        energy_f = sum(
            float(jnp.mean(state_f.nodes[n].energy)) for n in structure.nodes
        )

        assert (
            energy_f < energy_0
        ), f"Energy did not decrease: {energy_0:.6f} -> {energy_f:.6f}"


# ============================================================================
# IdentityNode Scaling and include_output Tests
# ============================================================================


class TestIdentityNodeScaling:
    """Test that IdentityNode gets muPC scaling based on in-degree."""

    def test_identity_node_fan_in_is_one(self):
        """IdentityNode.get_weight_fan_in() should return 1 (no weight matrix)."""
        assert IdentityNode.get_weight_fan_in((10,), {}) == 1
        assert IdentityNode.get_weight_fan_in((784,), {}) == 1
        assert IdentityNode.get_weight_fan_in((3, 32, 32), {}) == 1

    def test_identity_junction_gets_scaling(self):
        """IdentityNode used as a sum junction should get scaling a=1/sqrt(K)."""
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
        # IdentityNode sum junction: fan_in=1, K=2 -> a=1/sqrt(2) per edge
        scaling = structure.nodes["sum"].node_info.scaling_config
        assert scaling is not None
        expected_a = 1.0 / math.sqrt(2)  # 1/sqrt(fan_in * K) = 1/sqrt(1 * 2)
        for edge_key, a in scaling.forward_scale.items():
            assert (
                abs(a - expected_a) < 1e-10
            ), f"Expected a={expected_a:.6f}, got {a:.6f} for edge {edge_key}"
        # Linear nodes should still have scaling
        assert structure.nodes["h1"].node_info.scaling_config is not None
        assert structure.nodes["h2"].node_info.scaling_config is not None


class TestIncludeOutput:
    """Test the include_output flag on MuPCConfig."""

    def test_include_output_false_by_default(self):
        """Default include_output=False excludes output from scaling."""
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
            scaling=MuPCConfig(),
        )
        assert structure.nodes["y"].node_info.scaling_config is None

    def test_include_output_true(self):
        """With include_output=True, output node gets scaling."""
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

    def test_output_forward_scale_formula(self):
        """Output forward scale = 1/(fan_in * sqrt(K)) for K=1: 1/fan_in."""
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
        # y has input from h (shape=(20,)), so fan_in=20, K=1
        # Output formula: a = 1/(fan_in * sqrt(K)) = 1/(20 * 1) = 0.05
        edge_key = structure.nodes["y"].node_info.in_edges[0]
        expected_a = 1.0 / 20
        actual_a = scaling.forward_scale[edge_key]
        assert (
            abs(actual_a - expected_a) < 1e-10
        ), f"Output forward_scale: expected {expected_a}, got {actual_a}"

    def test_hidden_scaling_unchanged_with_include_output(self):
        """Hidden node scaling should be the same regardless of include_output."""
        x = IdentityNode(shape=(10,), name="x")
        h = Linear(shape=(20,), name="h", weight_init=MuPCInitializer())
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())

        structure_incl = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(include_output=True),
        )
        x2 = IdentityNode(shape=(10,), name="x")
        h2 = Linear(shape=(20,), name="h", weight_init=MuPCInitializer())
        y2 = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure_excl = graph(
            nodes=[x2, h2, y2],
            edges=[
                Edge(source=x2, target=h2.slot("in")),
                Edge(source=h2, target=y2.slot("in")),
            ],
            task_map=TaskMap(x=x2, y=y2),
            inference=InferenceSGD(),
            scaling=MuPCConfig(include_output=False),
        )

        edge_incl = structure_incl.nodes["h"].node_info.in_edges[0]
        edge_excl = structure_excl.nodes["h"].node_info.in_edges[0]
        a_incl = structure_incl.nodes["h"].node_info.scaling_config.forward_scale[
            edge_incl
        ]
        a_excl = structure_excl.nodes["h"].node_info.scaling_config.forward_scale[
            edge_excl
        ]
        assert abs(a_incl - a_excl) < 1e-10


# ============================================================================
# Variance Propagation Tests (unified a=1/sqrt(fan_in*K) formula)
# ============================================================================


class TestVariancePropagation:
    """Test the unified muPC scaling formula a=1/sqrt(fan_in*K)."""

    def test_plain_chain_no_depth_factor(self):
        """Plain chain: all hidden nodes get a=1/sqrt(fan_in), independent of depth."""
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
        # h1: fan_in=10 (from x), K=1 -> a=1/sqrt(10)
        h1_edge = structure.nodes["h1"].node_info.in_edges[0]
        assert (
            abs(
                structure.nodes["h1"].node_info.scaling_config.forward_scale[h1_edge]
                - 1.0 / math.sqrt(10)
            )
            < 1e-10
        )

        # h2: fan_in=20 (from h1), K=1 -> a=1/sqrt(20)
        h2_edge = structure.nodes["h2"].node_info.in_edges[0]
        assert (
            abs(
                structure.nodes["h2"].node_info.scaling_config.forward_scale[h2_edge]
                - 1.0 / math.sqrt(20)
            )
            < 1e-10
        )

        # h3: fan_in=20 (from h2), K=1 -> a=1/sqrt(20)
        h3_edge = structure.nodes["h3"].node_info.in_edges[0]
        assert (
            abs(
                structure.nodes["h3"].node_info.scaling_config.forward_scale[h3_edge]
                - 1.0 / math.sqrt(20)
            )
            < 1e-10
        )

        # h2 and h3 should have identical scaling (no depth dependency)
        a_h2 = structure.nodes["h2"].node_info.scaling_config.forward_scale[h2_edge]
        a_h3 = structure.nodes["h3"].node_info.scaling_config.forward_scale[h3_edge]
        assert abs(a_h2 - a_h3) < 1e-10

    def test_identity_node_gets_scaling_k2(self):
        """IdentityNode with K=2 in-edges gets a=1/sqrt(2) per edge."""
        x = IdentityNode(shape=(10,), name="x")
        h1 = Linear(shape=(20,), name="h1", weight_init=MuPCInitializer())
        h2 = Linear(shape=(20,), name="h2", weight_init=MuPCInitializer())
        junction = IdentityNode(shape=(20,), name="junction")
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h1, h2, junction, y],
            edges=[
                Edge(source=x, target=h1.slot("in")),
                Edge(source=h1, target=h2.slot("in")),
                Edge(source=h1, target=junction.slot("in")),
                Edge(source=h2, target=junction.slot("in")),
                Edge(source=junction, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        scaling = structure.nodes["junction"].node_info.scaling_config
        assert scaling is not None
        expected_a = 1.0 / math.sqrt(2)  # fan_in=1, K=2
        for a in scaling.forward_scale.values():
            assert abs(a - expected_a) < 1e-10

    def test_multi_edge_linear(self):
        """Linear with K=2 edges: a_k=1/sqrt(fan_in*2) per edge."""
        x = IdentityNode(shape=(10,), name="x")
        h1 = Linear(shape=(20,), name="h1", weight_init=MuPCInitializer())
        h2 = Linear(shape=(20,), name="h2", weight_init=MuPCInitializer())
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h1, h2, y],
            edges=[
                Edge(source=x, target=h1.slot("in")),
                Edge(source=h1, target=h2.slot("in")),
                Edge(source=x, target=h2.slot("in")),  # skip -> h2 has K=2
                Edge(source=h2, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        h2_scaling = structure.nodes["h2"].node_info.scaling_config
        # h2 has 2 inputs: from h1 (shape=20) and from x (shape=10)
        # K=2, fan_in varies by edge
        for edge_key, a in h2_scaling.forward_scale.items():
            if "h1->" in edge_key:
                expected = 1.0 / math.sqrt(20 * 2)  # fan_in=20, K=2
            else:
                expected = 1.0 / math.sqrt(10 * 2)  # fan_in=10, K=2
            assert (
                abs(a - expected) < 1e-10
            ), f"Edge {edge_key}: expected {expected:.6f}, got {a:.6f}"

    def test_deep_chain_uniform_scaling(self):
        """20-layer chain: all same-width hidden nodes get same a=1/sqrt(fan_in)."""
        width = 32
        x = IdentityNode(shape=(width,), name="x")
        layers = [
            Linear(shape=(width,), name=f"h{i}", weight_init=MuPCInitializer())
            for i in range(20)
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
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )

        # All hidden nodes have K=1, fan_in=width -> a=1/sqrt(width)
        expected_a = 1.0 / math.sqrt(width)
        for i in range(20):
            node_name = f"h{i}"
            scaling = structure.nodes[node_name].node_info.scaling_config
            assert scaling is not None, f"{node_name} should have scaling"
            edge_key = structure.nodes[node_name].node_info.in_edges[0]
            actual_a = scaling.forward_scale[edge_key]
            assert (
                abs(actual_a - expected_a) < 1e-10
            ), f"{node_name}: expected {expected_a:.6f}, got {actual_a:.6f}"

    def test_output_with_include_output_k1(self):
        """Output node with K=1: a = 1/(fan_in * sqrt(1)) = 1/fan_in."""
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
        y_scaling = structure.nodes["y"].node_info.scaling_config
        edge_key = structure.nodes["y"].node_info.in_edges[0]
        expected_a = 1.0 / 20  # 1/(fan_in * sqrt(K)) = 1/(20 * 1)
        assert abs(y_scaling.forward_scale[edge_key] - expected_a) < 1e-10


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Test deprecated MuPCConfig parameters."""

    def test_depth_metric_emits_deprecation_warning(self):
        """MuPCConfig(depth_metric=...) should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="depth_metric.*deprecated"):
            MuPCConfig(depth_metric=ShortestPathDepth())

    def test_min_depth_emits_deprecation_warning(self):
        """MuPCConfig(min_depth=...) should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="min_depth.*deprecated"):
            MuPCConfig(min_depth=3)

    def test_deprecated_config_still_works(self):
        """Graph should build correctly even with deprecated parameters."""
        x = IdentityNode(shape=(10,), name="x")
        h = Linear(shape=(20,), name="h", weight_init=MuPCInitializer())
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        import warnings

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
        # Should still compute scaling correctly
        assert structure.nodes["h"].node_info.scaling_config is not None
