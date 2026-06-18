"""
Tests for MaxPool and AvgPool.
"""

import pytest
import jax
import jax.numpy as jnp
import fabricpc.nodes as nodes
from fabricpc.nodes.pooling import MaxPool, AvgPool, _PoolBase
from fabricpc.nodes.base import SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy

jax.config.update("jax_default_prng_impl", "threefry2x32")


# ── Validation ──────────────────────────────────────────────────────────


class TestMaxPoolValidation:
    """Windowed validation is deferred to initialize_params (input shapes are
    only known there), so these construct then trigger it."""

    @staticmethod
    def _init(node, in_shape):
        return MaxPool.initialize_params(
            jax.random.PRNGKey(0),
            node._shape,
            {"e:in": in_shape},
            None,
            node._extra_config,
        )

    def test_bad_spatial_rank_raises(self):
        node = MaxPool(shape=(10,), name="bad", window_shape=(2,), stride=(2,))
        with pytest.raises(ValueError, match="spatial_rank"):
            self._init(node, (10,))

    def test_window_shape_mismatch_raises(self):
        node = MaxPool(shape=(14, 14, 16), name="bad", window_shape=(2,), stride=(2, 2))
        with pytest.raises(ValueError, match="length"):
            self._init(node, (28, 28, 16))

    def test_stride_mismatch_raises(self):
        node = MaxPool(shape=(14, 14, 16), name="bad", window_shape=(2, 2), stride=(2,))
        with pytest.raises(ValueError, match="stride"):
            self._init(node, (28, 28, 16))

    def test_channel_mismatch_raises(self):
        """Pooling preserves channels: declared C must match the input's C."""
        node = MaxPool(
            shape=(14, 14, 8), name="bad", window_shape=(2, 2), stride=(2, 2)
        )
        with pytest.raises(ValueError, match="channels|preserves channels"):
            self._init(node, (28, 28, 16))  # 16 in, 8 declared

    def test_explicit_padding_ge_window_raises(self):
        """Explicit padding >= window would let a max window fill with -inf."""
        node = MaxPool(
            shape=(8, 8, 4),
            name="bad",
            window_shape=(2, 2),
            stride=(2, 2),
            padding=((2, 2), (0, 0)),  # lo=hi=2 >= window 2
        )
        with pytest.raises(ValueError, match="window"):
            self._init(node, (14, 14, 4))

    def test_valid_2d_construction(self):
        node = MaxPool(
            shape=(14, 14, 16), name="pool", window_shape=(2, 2), stride=(2, 2)
        )
        assert node._shape == (14, 14, 16)

    def test_valid_1d_construction(self):
        node = MaxPool(shape=(14, 8), name="pool1d", window_shape=(2,), stride=(2,))
        assert node._shape == (14, 8)

    def test_default_activation_is_identity(self):
        node = MaxPool(
            shape=(14, 14, 16), name="pool", window_shape=(2, 2), stride=(2, 2)
        )
        assert isinstance(node._activation, IdentityActivation)

    def test_stride_defaults_to_window_shape(self):
        node = MaxPool(shape=(14, 14, 16), name="pool", window_shape=(2, 2))
        assert node._extra_config["stride"] == (2, 2)


# ── Slots ───────────────────────────────────────────────────────────────


class TestMaxPoolSlots:
    def test_get_slots_returns_default(self):
        slots = MaxPool.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True

    def test_slot_is_variance_scalable(self):
        """MaxPool is a weightless *transformation*, not a skip path —
        its slot should remain variance-scalable. The muPC formula degenerates
        to a = gain/sqrt(K*L) via get_weight_fan_in() returning 1."""
        slots = MaxPool.get_slots()
        assert slots["in"].is_variance_scalable is True
        assert slots["in"].is_skip_connection is False


# ── muPC fan_in ─────────────────────────────────────────────────────────


class TestMaxPoolMuPC:
    def test_fan_in_is_one(self):
        """Weightless nodes return fan_in=1 (IdentityNode convention).

        Without this override, the base default returns source_shape[-1] (the
        upstream channel count), which silently attenuates activations and
        gradients through every pool by 1/sqrt(C_in).
        """
        assert MaxPool.get_weight_fan_in((28, 28, 32), {}) == 1
        assert MaxPool.get_weight_fan_in((14, 14, 64), {}) == 1
        assert MaxPool.get_weight_fan_in((20, 8), {}) == 1


# ── Params ──────────────────────────────────────────────────────────────


class TestMaxPoolParams:
    def test_wrong_declared_output_shape_raises(self):
        """Windowed pooling fails fast when the declared output spatial shape
        disagrees with window/stride/padding (28 /2 -> 14, not 13)."""
        node = MaxPool(
            shape=(13, 13, 16),
            name="p",
            window_shape=(2, 2),
            stride=(2, 2),
            padding="VALID",
        )
        key = jax.random.PRNGKey(0)
        with pytest.raises(ValueError, match="declared output spatial shape"):
            MaxPool.initialize_params(
                key,
                (13, 13, 16),
                {"src→pool:in": (28, 28, 16)},
                None,
                node._extra_config,
            )

    def test_no_weights_or_biases(self):
        key = jax.random.PRNGKey(0)
        # A valid windowed config (28x28 -> 14x14) so validation passes; the
        # point of the test is that pooling allocates no params.
        config = {"window_shape": (2, 2), "stride": (2, 2), "padding": "VALID"}
        params = MaxPool.initialize_params(
            key,
            node_shape=(14, 14, 16),
            input_shapes={"src→pool:in": (28, 28, 16)},
            config=config,
        )
        assert params.weights == {}
        assert params.biases == {}


# ── Forward ─────────────────────────────────────────────────────────────


class TestMaxPoolForward:
    @staticmethod
    def _build_node_info(node, node_shape, config):
        return NodeInfo(
            name=node._name,
            shape=node_shape,
            node_type=type(node).__name__,
            node_class=type(node),
            node_config=config,
            activation=node._activation,
            energy=node._energy,
            latent_init=node._latent_init,
            weight_init=None,
            slots={},
            in_degree=1,
            out_degree=1,
            in_edges=("src→pool:in",),
            out_edges=(),
        )

    def test_2d_forward_shape(self):
        B, H_in, W_in, C = 4, 28, 28, 16
        node = MaxPool(
            shape=(14, 14, C),
            name="pool2d",
            window_shape=(2, 2),
            stride=(2, 2),
            padding="VALID",
        )
        key = jax.random.PRNGKey(0)
        params = NodeParams(weights={}, biases={})
        x = jax.random.normal(key, (B, H_in, W_in, C))
        z_latent = jax.random.normal(key, (B, 14, 14, C))
        state = NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros_like(z_latent),
            pre_activation=jnp.zeros_like(z_latent),
            error=jnp.zeros_like(z_latent),
            energy=jnp.zeros(()),
            latent_grad=jnp.zeros_like(z_latent),
        )
        config = {"window_shape": (2, 2), "stride": (2, 2), "padding": "VALID"}
        node_info = self._build_node_info(node, (14, 14, C), config)
        energy, new_state = MaxPool.forward(
            params, {"src→pool:in": x}, state, node_info
        )
        assert new_state.z_mu.shape == (B, 14, 14, C)
        assert energy.shape == ()

    def test_1d_forward_shape(self):
        B, L_in, C = 4, 20, 8
        node = MaxPool(
            shape=(10, C),
            name="pool1d",
            window_shape=(2,),
            stride=(2,),
            padding="VALID",
        )
        key = jax.random.PRNGKey(1)
        params = NodeParams(weights={}, biases={})
        x = jax.random.normal(key, (B, L_in, C))
        z_latent = jax.random.normal(key, (B, 10, C))
        state = NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros_like(z_latent),
            pre_activation=jnp.zeros_like(z_latent),
            error=jnp.zeros_like(z_latent),
            energy=jnp.zeros(()),
            latent_grad=jnp.zeros_like(z_latent),
        )
        config = {"window_shape": (2,), "stride": (2,), "padding": "VALID"}
        node_info = self._build_node_info(node, (10, C), config)
        energy, new_state = MaxPool.forward(
            params, {"src→pool:in": x}, state, node_info
        )
        assert new_state.z_mu.shape == (B, 10, C)

    def test_forward_energy_nonnegative(self):
        B, H_in, W_in, C = 2, 8, 8, 4
        node = MaxPool(
            shape=(4, 4, C),
            name="pool_e",
            window_shape=(2, 2),
            stride=(2, 2),
            padding="VALID",
        )
        key = jax.random.PRNGKey(2)
        params = NodeParams(weights={}, biases={})
        x = jax.random.normal(key, (B, H_in, W_in, C))
        z_latent = jax.random.normal(key, (B, 4, 4, C))
        state = NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros_like(z_latent),
            pre_activation=jnp.zeros_like(z_latent),
            error=jnp.zeros_like(z_latent),
            energy=jnp.zeros(()),
            latent_grad=jnp.zeros_like(z_latent),
        )
        config = {"window_shape": (2, 2), "stride": (2, 2), "padding": "VALID"}
        node_info = self._build_node_info(node, (4, 4, C), config)
        energy, _ = MaxPool.forward(params, {"src→pool:in": x}, state, node_info)
        assert energy >= 0.0

    def test_max_pool_selects_max(self):
        """Verify that the pooling output actually contains the max values."""
        # Create a simple 4x4 input with known values
        x = jnp.array(
            [
                [
                    [[1.0], [2.0], [3.0], [4.0]],
                    [[5.0], [6.0], [7.0], [8.0]],
                    [[9.0], [10.0], [11.0], [12.0]],
                    [[13.0], [14.0], [15.0], [16.0]],
                ]
            ]
        )  # (1, 4, 4, 1)
        node = MaxPool(
            shape=(2, 2, 1),
            name="pool_max",
            window_shape=(2, 2),
            stride=(2, 2),
            padding="VALID",
        )
        params = NodeParams(weights={}, biases={})
        z_latent = jnp.zeros((1, 2, 2, 1))
        state = NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros_like(z_latent),
            pre_activation=jnp.zeros_like(z_latent),
            error=jnp.zeros_like(z_latent),
            energy=jnp.zeros(()),
            latent_grad=jnp.zeros_like(z_latent),
        )
        config = {"window_shape": (2, 2), "stride": (2, 2), "padding": "VALID"}
        node_info = self._build_node_info(node, (2, 2, 1), config)
        _, new_state = MaxPool.forward(params, {"src→pool:in": x}, state, node_info)
        # Max of each 2x2 block: [[6, 8], [14, 16]]
        expected = jnp.array([[[[6.0], [8.0]], [[14.0], [16.0]]]])
        assert jnp.allclose(new_state.z_mu, expected)


# ── Shared helpers for forward tests ─────────────────────────────────────


def _make_node_info(node, node_shape, config):
    return NodeInfo(
        name=node._name,
        shape=node_shape,
        node_type=type(node).__name__,
        node_class=type(node),
        node_config=config,
        activation=node._activation,
        energy=node._energy,
        latent_init=node._latent_init,
        weight_init=None,
        slots={},
        in_degree=1,
        out_degree=1,
        in_edges=("src→pool:in",),
        out_edges=(),
    )


def _make_state(z_latent):
    return NodeState(
        z_latent=z_latent,
        z_mu=jnp.zeros_like(z_latent),
        pre_activation=jnp.zeros_like(z_latent),
        error=jnp.zeros_like(z_latent),
        energy=jnp.zeros(()),
        latent_grad=jnp.zeros_like(z_latent),
    )


# ── AvgPool: windowed ─────────────────────────────────────────────────


class TestAvgPoolWindowed:
    def test_window_shape_required_when_not_global(self):
        with pytest.raises(ValueError, match="window_shape is required"):
            AvgPool(shape=(14, 14, 16), name="bad")

    def test_stride_defaults_to_window_shape(self):
        node = AvgPool(shape=(14, 14, 16), name="avg", window_shape=(2, 2))
        assert node._extra_config["stride"] == (2, 2)

    def test_fan_in_is_one(self):
        assert AvgPool.get_weight_fan_in((28, 28, 32), {}) == 1

    def test_slot_is_variance_scalable(self):
        slots = AvgPool.get_slots()
        assert slots["in"].is_variance_scalable is True
        assert slots["in"].is_skip_connection is False

    def test_2d_forward_shape(self):
        B, H_in, W_in, C = 4, 28, 28, 16
        node = AvgPool(
            shape=(14, 14, C), name="avg2d", window_shape=(2, 2), padding="VALID"
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (B, H_in, W_in, C))
        state = _make_state(jnp.zeros((B, 14, 14, C)))
        config = {
            "window_shape": (2, 2),
            "stride": (2, 2),
            "padding": "VALID",
            "global_pool": False,
        }
        node_info = _make_node_info(node, (14, 14, C), config)
        energy, new_state = AvgPool.forward(
            NodeParams(weights={}, biases={}), {"src→pool:in": x}, state, node_info
        )
        assert new_state.z_mu.shape == (B, 14, 14, C)
        assert energy.shape == ()

    def test_avg_pool_computes_mean(self):
        """Each 2x2 block should be averaged."""
        x = jnp.array(
            [
                [
                    [[1.0], [2.0], [3.0], [4.0]],
                    [[5.0], [6.0], [7.0], [8.0]],
                    [[9.0], [10.0], [11.0], [12.0]],
                    [[13.0], [14.0], [15.0], [16.0]],
                ]
            ]
        )  # (1, 4, 4, 1)
        node = AvgPool(shape=(2, 2, 1), name="avg_m", window_shape=(2, 2))
        state = _make_state(jnp.zeros((1, 2, 2, 1)))
        config = {
            "window_shape": (2, 2),
            "stride": (2, 2),
            "padding": "VALID",
            "global_pool": False,
        }
        node_info = _make_node_info(node, (2, 2, 1), config)
        _, new_state = AvgPool.forward(
            NodeParams(weights={}, biases={}), {"src→pool:in": x}, state, node_info
        )
        # Mean of each 2x2 block: [[3.5, 5.5], [11.5, 13.5]]
        expected = jnp.array([[[[3.5], [5.5]], [[11.5], [13.5]]]])
        assert jnp.allclose(new_state.z_mu, expected)

    def test_count_include_pad_false_divides_by_real_elements(self):
        """SAME padding + count_include_pad=False divides border windows by the
        number of real (non-padded) elements, not the full window volume."""
        x = jnp.array(
            [[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]]]
        )  # (1, 3, 3, 1)
        node = AvgPool(
            shape=(2, 2, 1),
            name="cip_f",
            window_shape=(2, 2),
            padding="SAME",
            count_include_pad=False,
        )
        state = _make_state(jnp.zeros((1, 2, 2, 1)))
        config = {
            "window_shape": (2, 2),
            "stride": (2, 2),
            "padding": "SAME",
            "global_pool": False,
            "count_include_pad": False,
        }
        node_info = _make_node_info(node, (2, 2, 1), config)
        _, new_state = AvgPool.forward(
            NodeParams(weights={}, biases={}), {"src→pool:in": x}, state, node_info
        )
        # SAME pads 3->4 on the high side. Real-element means per window:
        #   [1,2,4,5]/4=3.0   [3,6]/2=4.5   [7,8]/2=7.5   [9]/1=9.0
        expected = jnp.array([[[[3.0], [4.5]], [[7.5], [9.0]]]])
        assert jnp.allclose(new_state.z_mu, expected)

    def test_count_include_pad_true_divides_by_window_volume(self):
        """Default count_include_pad=True divides every window by the full volume."""
        x = jnp.array(
            [[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]]]
        )  # (1, 3, 3, 1)
        node = AvgPool(
            shape=(2, 2, 1),
            name="cip_t",
            window_shape=(2, 2),
            padding="SAME",
            count_include_pad=True,
        )
        state = _make_state(jnp.zeros((1, 2, 2, 1)))
        config = {
            "window_shape": (2, 2),
            "stride": (2, 2),
            "padding": "SAME",
            "global_pool": False,
            "count_include_pad": True,
        }
        node_info = _make_node_info(node, (2, 2, 1), config)
        _, new_state = AvgPool.forward(
            NodeParams(weights={}, biases={}), {"src→pool:in": x}, state, node_info
        )
        expected = jnp.array([[[[3.0], [2.25]], [[3.75], [2.25]]]])
        assert jnp.allclose(new_state.z_mu, expected)


# ── AvgPool: global ───────────────────────────────────────────────────


class TestAvgPoolGlobal:
    def test_global_construction_no_window(self):
        node = AvgPool(shape=(256,), name="gap", global_pool=True)
        assert node._shape == (256,)

    def test_global_pool_rejects_non_rank1_shape(self):
        """global_pool=True maps (B, Spatial..., C) -> (B, C); the declared
        output shape must be rank-1 (C,). A spatial-rank shape must fail fast."""
        with pytest.raises(ValueError, match="rank-1 shape"):
            AvgPool(shape=(4, 4, 256), name="bad_gap", global_pool=True)

    def test_global_forward_collapses_spatial(self):
        B, H, W, C = 4, 7, 7, 64
        node = AvgPool(shape=(C,), name="gap", global_pool=True)
        x = jax.random.normal(jax.random.PRNGKey(3), (B, H, W, C))
        state = _make_state(jnp.zeros((B, C)))
        config = {"global_pool": True, "padding": "VALID"}
        node_info = _make_node_info(node, (C,), config)
        energy, new_state = AvgPool.forward(
            NodeParams(weights={}, biases={}), {"src→pool:in": x}, state, node_info
        )
        assert new_state.z_mu.shape == (B, C)
        # Matches jnp.mean over spatial axes
        assert jnp.allclose(new_state.z_mu, jnp.mean(x, axis=(1, 2)))
        assert energy >= 0.0


# ── Tuple/numeric padding ─────────────────────────────────────────────────


class TestPoolPadding:
    def test_max_pool_tuple_padding(self):
        """Explicit spatial (low,high) padding must work (was a crash before
        the batch/channel wrapping helper)."""
        B, C = 1, 1
        x = jax.random.normal(jax.random.PRNGKey(4), (B, 4, 4, C))
        node = MaxPool(
            shape=(3, 3, C),
            name="pad",
            window_shape=(2, 2),
            stride=(2, 2),
            padding=((1, 1), (1, 1)),
        )
        state = _make_state(jnp.zeros((B, 3, 3, C)))
        config = {"window_shape": (2, 2), "stride": (2, 2), "padding": ((1, 1), (1, 1))}
        node_info = _make_node_info(node, (3, 3, C), config)
        _, new_state = MaxPool.forward(
            NodeParams(weights={}, biases={}), {"src→pool:in": x}, state, node_info
        )
        # padded 4->6, (6-2)/2 + 1 = 3
        assert new_state.z_mu.shape == (B, 3, 3, C)

    def test_format_padding_wraps_batch_channel(self):
        assert _PoolBase._format_pool_padding("SAME") == "SAME"
        assert _PoolBase._format_pool_padding([(1, 1), (2, 2)]) == (
            (0, 0),
            (1, 1),
            (2, 2),
            (0, 0),
        )


# ── Exports ───────────────────────────────────────────────────────────────


class TestPoolExports:
    def test_concrete_nodes_exported(self):
        assert "MaxPool" in nodes.__all__
        assert "AvgPool" in nodes.__all__

    def test_base_not_exported(self):
        assert "_PoolBase" not in nodes.__all__
        assert not hasattr(nodes, "_PoolBase")
