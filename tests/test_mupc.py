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
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.core.activations import IdentityActivation, ReLUActivation, TanhActivation
from fabricpc.core.mupc import MuPCConfig, MuPCScalingFactors
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.state_ops import set_latents_to_clamps
from fabricpc.core.learning import compute_local_weight_gradients

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
# SkipConnection Scaling Tests
# ============================================================================


class TestSkipConnectionScaling:
    """Test SkipConnection node and depth-dependent scaling."""

    def test_skip_connection_unscaled(self):
        """SkipConnection node gets scale 1.0 on all edges."""
        from fabricpc.nodes.skip_connection import SkipConnection

        x = IdentityNode(shape=(10,), name="x")
        h = Linear(shape=(10,), name="h", weight_init=MuPCInitializer())
        skip = SkipConnection(shape=(10,), name="skip")
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h, skip, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=skip.slot("in")),
                Edge(source=x, target=skip.slot("in")),  # skip path
                Edge(source=skip, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        scaling = structure.nodes["skip"].node_info.scaling_config
        assert scaling is not None
        for a in scaling.forward_scale.values():
            assert a == 1.0
        for td in scaling.topdown_grad_scale.values():
            assert td == 1.0

    def test_skip_depth_affects_compute_scaling(self):
        """Compute nodes get depth factor L = number of SkipConnection nodes."""
        from fabricpc.nodes.skip_connection import SkipConnection

        x = IdentityNode(shape=(10,), name="x")
        h1 = Linear(shape=(10,), name="h1", weight_init=MuPCInitializer())
        s1 = SkipConnection(shape=(10,), name="s1")
        h2 = Linear(shape=(10,), name="h2", weight_init=MuPCInitializer())
        s2 = SkipConnection(shape=(10,), name="s2")
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h1, s1, h2, s2, y],
            edges=[
                Edge(source=x, target=h1.slot("in")),
                Edge(source=x, target=s1.slot("in")),  # skip
                Edge(source=h1, target=s1.slot("in")),  # compute -> merge
                Edge(source=s1, target=h2.slot("in")),
                Edge(source=s1, target=s2.slot("in")),  # skip
                Edge(source=h2, target=s2.slot("in")),  # compute -> merge
                Edge(source=s2, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        # L = 2 (two SkipConnection nodes: s1, s2)
        # h1: fan_in=10, K=1, L=2
        # expected a = 1/sqrt(10 * 1 * 2)  (identity activation, gain=1)
        h1_edge = structure.nodes["h1"].node_info.in_edges[0]
        a_h1 = structure.nodes["h1"].node_info.scaling_config.forward_scale[h1_edge]
        expected_a = 1.0 / math.sqrt(10 * 1 * 2)
        assert abs(a_h1 - expected_a) < 1e-10

    def test_no_skip_connections_degenerates_to_old_formula(self):
        """Without SkipConnection nodes, L=1 and formula = gain/sqrt(fan_in*K)."""
        x = IdentityNode(shape=(10,), name="x")
        h1 = Linear(shape=(20,), name="h1", weight_init=MuPCInitializer())
        h2 = Linear(shape=(20,), name="h2", weight_init=MuPCInitializer())
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, h1, h2, y],
            edges=[
                Edge(source=x, target=h1.slot("in")),
                Edge(source=h1, target=h2.slot("in")),
                Edge(source=h2, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        # L=1 (no SkipConnections), K=1
        # h1: gain=1 (identity), fan_in=10 -> a = 1/sqrt(10)
        h1_edge = structure.nodes["h1"].node_info.in_edges[0]
        a_h1 = structure.nodes["h1"].node_info.scaling_config.forward_scale[h1_edge]
        assert abs(a_h1 - 1.0 / math.sqrt(10)) < 1e-10

    def test_slot_is_variance_scalable_property(self):
        """Linear slots are scalable, SkipConnection slots are not."""
        from fabricpc.nodes.skip_connection import SkipConnection

        linear_slots = Linear.get_slots()
        assert linear_slots["in"].is_variance_scalable is True
        assert linear_slots["in"].is_skip_connection is False

        identity_slots = IdentityNode.get_slots()
        assert identity_slots["in"].is_variance_scalable is True
        assert identity_slots["in"].is_skip_connection is False

        skip_slots = SkipConnection.get_slots()
        assert skip_slots["in"].is_variance_scalable is False
        assert skip_slots["in"].is_skip_connection is True

    def test_is_skip_connection_forces_unscalable(self):
        """is_skip_connection=True requires is_variance_scalable=False."""
        from fabricpc.nodes.base import SlotSpec

        slot = SlotSpec(
            name="skip",
            is_multi_input=True,
            is_skip_connection=True,
            is_variance_scalable=False,
        )
        assert slot.is_skip_connection is True
        assert slot.is_variance_scalable is False

        # is_skip_connection=True with default is_variance_scalable (True) raises
        with pytest.raises(ValueError):
            SlotSpec(name="skip", is_multi_input=True, is_skip_connection=True)

        # is_skip_connection=True with explicit is_variance_scalable=True raises
        with pytest.raises(ValueError):
            SlotSpec(
                name="skip",
                is_multi_input=True,
                is_variance_scalable=True,
                is_skip_connection=True,
            )

    def test_metadata_slot_does_not_inflate_depth(self):
        """A non-scalable metadata slot (is_skip_connection=False) should not inflate L."""
        from fabricpc.nodes.base import NodeBase, SlotSpec

        class MetadataNode(NodeBase):
            """Node with a metadata slot (like a mask) that is not a skip connection."""

            @staticmethod
            def get_slots():
                return {
                    "in": SlotSpec(name="in", is_multi_input=False),
                    "meta": SlotSpec(
                        name="meta", is_multi_input=False, is_variance_scalable=False
                    ),
                }

            @staticmethod
            def get_weight_fan_in(source_shape, config):
                return source_shape[-1]

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, weight_init, config):
                from fabricpc.core.types import NodeParams
                import jax

                weights = {}
                for edge_key, in_shape in input_shapes.items():
                    if ":in" in edge_key:
                        weights[edge_key] = jax.random.normal(
                            key, (in_shape[-1], node_shape[-1])
                        )
                return NodeParams(weights, {})

            @staticmethod
            def forward(params, inputs, state, node_info):
                import jax.numpy as jnp

                x = inputs[next(k for k in inputs if k.endswith(":in"))]
                edge_key = next(k for k in params.weights if ":in" in k)
                z_mu = jnp.matmul(x, params.weights[edge_key])
                error = state.z_latent - z_mu
                state = state._replace(z_mu=z_mu, error=error)
                state = node_info.node_class.energy_functional(state, node_info)
                return jnp.sum(state.energy), state

        # Build graph: x -> meta_node (with meta slot) -> y
        # Also connect a "meta" source to meta_node's meta slot
        x = IdentityNode(shape=(10,), name="x")
        meta_src = IdentityNode(shape=(10,), name="meta_src")
        mn = MetadataNode(
            shape=(10,),
            name="mn",
            weight_init=MuPCInitializer(),
        )
        y = Linear(shape=(5,), name="y", weight_init=MuPCInitializer())
        structure = graph(
            nodes=[x, meta_src, mn, y],
            edges=[
                Edge(source=x, target=mn.slot("in")),
                Edge(source=meta_src, target=mn.slot("meta")),
                Edge(source=mn, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(),
            scaling=MuPCConfig(),
        )
        # L should be 1 (no skip connections), NOT 2
        # mn: fan_in=10, K=1, L=1 -> a = 1/sqrt(10)
        mn_in_edge = next(
            e for e in structure.nodes["mn"].node_info.in_edges if ":in" in e
        )
        a_mn = structure.nodes["mn"].node_info.scaling_config.forward_scale[mn_in_edge]
        expected_a = 1.0 / math.sqrt(10)
        assert abs(a_mn - expected_a) < 1e-10

        # Meta edge passes through unscaled — non-scalable slots are absent
        # from the per-edge dicts (callsites treat missing keys as no-op).
        mn_meta_edge = next(
            e for e in structure.nodes["mn"].node_info.in_edges if ":meta" in e
        )
        scaling = structure.nodes["mn"].node_info.scaling_config
        assert mn_meta_edge not in scaling.forward_scale
        assert mn_meta_edge not in scaling.topdown_grad_scale
        assert mn_meta_edge not in scaling.weight_grad_scale


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Test deprecated MuPCConfig parameters."""

    def test_deprecated_params_emit_warnings(self):
        """depth_metric and min_depth emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="depth_metric.*deprecated"):
            MuPCConfig(depth_metric="ignored")
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
                scaling=MuPCConfig(depth_metric="ignored"),
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
