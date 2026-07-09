"""
Unit tests for ConvNode (unified 1D/2D/3D).

Run with:
    pytest tests/test_convolutional.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.nodes.convolutional import ConvNode
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ReLUActivation, LeakyReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    KaimingInitializer,
    NormalInitializer,
    ZerosInitializer,
    XavierInitializer,
)
from fabricpc.nodes.base import SlotSpec

# =============================================================================
# Helpers
# =============================================================================


def _make_node_info(
    node_cls,
    node_shape,
    config,
    activation,
    energy_obj,
    latent_init,
    weight_init,
    in_edges=("src->dst:in",),
):
    return NodeInfo(
        name="test_node",
        shape=node_shape,
        node_type=node_cls.__name__,
        node_class=node_cls,
        node_config=config,
        activation=activation,
        energy=energy_obj,
        latent_init=latent_init,
        weight_init=weight_init,
        slots={},
        in_degree=len(in_edges),
        out_degree=0,
        in_edges=in_edges,
        out_edges=(),
    )


def _make_state(key, batch_size, node_shape):
    z_latent = jax.random.normal(key, (batch_size, *node_shape))
    return NodeState(
        z_latent=z_latent,
        z_mu=jnp.zeros_like(z_latent),
        error=jnp.zeros_like(z_latent),
        energy=jnp.zeros(batch_size),
        pre_activation=jnp.zeros_like(z_latent),
        latent_grad=jnp.zeros_like(z_latent),
    )


# =============================================================================
# ConvNode — construction-time validation
# =============================================================================


class TestConvNodeValidation:
    """Tests for argument validation.

    Structural/shape validation is deferred to ``initialize_params`` (the only
    place input shapes are known), so these construct the node and then trigger
    the shared ``validate_windowed_output`` check via ``initialize_params``.
    """

    @staticmethod
    def _init(node, in_shape):
        return ConvNode.initialize_params(
            jax.random.PRNGKey(0),
            node._shape,
            {"e:in": in_shape},
            node._weight_init,
            node._extra_config,
        )

    def test_bad_spatial_rank_raises(self):
        node = ConvNode(shape=(32,), name="x", kernel_size=(3,))  # 0-D spatial
        with pytest.raises(ValueError, match="spatial_rank"):
            self._init(node, (32,))

    def test_shape_5d_raises(self):
        node = ConvNode(shape=(4, 4, 4, 4, 16), name="x", kernel_size=(3, 3, 3, 3))
        with pytest.raises(ValueError, match="spatial_rank"):
            self._init(node, (4, 4, 4, 4, 3))

    def test_kernel_size_rank_mismatch_raises(self):
        node = ConvNode(shape=(28, 28, 16), name="x", kernel_size=(3,))  # 2D, 1D kernel
        with pytest.raises(ValueError, match="length"):
            self._init(node, (28, 28, 3))

    def test_stride_auto_filled(self):
        node = ConvNode(shape=(28, 28, 16), name="x", kernel_size=(3, 3))
        assert node._extra_config["stride"] == (1, 1)

    def test_default_activation_is_relu(self):
        node = ConvNode(shape=(28, 28, 16), name="x", kernel_size=(3, 3))
        assert isinstance(node._activation, ReLUActivation)

    def test_default_bias_init_is_zeros(self):
        """With signature defaults, the constructor fills bias_init=ZerosInitializer()
        into config; initialize_params then produces an all-zero bias."""
        node = ConvNode(shape=(10, 8), name="x", kernel_size=(3,), use_bias=True)
        assert isinstance(node._extra_config["bias_init"], ZerosInitializer)
        key = jax.random.PRNGKey(0)
        params = ConvNode.initialize_params(
            key, (10, 8), {"e:in": (10, 4)}, node._weight_init, node._extra_config
        )
        assert jnp.all(params.biases["b"] == 0.0)

    def test_use_bias_false_no_bias_init_required(self):
        node = ConvNode(
            shape=(28, 28, 16),
            name="x",
            kernel_size=(3, 3),
            use_bias=False,
            bias_init=None,
        )
        assert node._extra_config.get("bias_init") is None

    def test_default_singletons_shared_but_weights_independent(self):
        """Signature defaults mean two ConvNodes share the same default activation
        and initializer singleton. That is safe: the configs are immutable
        (MappingProxyType) and ``initialize`` is a pure static method, so nothing
        mutates the shared object. Weights still differ per layer because
        ``initialize_params`` draws from a distinct PRNG key on each call."""
        n1 = ConvNode(shape=(8, 8, 4), name="a", kernel_size=(3, 3))
        n2 = ConvNode(shape=(8, 8, 4), name="b", kernel_size=(3, 3))
        # Shared singleton is expected and harmless.
        assert n1._activation is n2._activation
        assert n1._weight_init is n2._weight_init
        # ...yet weights drawn with different keys are not identical.
        p1 = ConvNode.initialize_params(
            jax.random.PRNGKey(1),
            (8, 8, 4),
            {"e:in": (8, 8, 3)},
            n1._weight_init,
            n1._extra_config,
        )
        p2 = ConvNode.initialize_params(
            jax.random.PRNGKey(2),
            (8, 8, 4),
            {"e:in": (8, 8, 3)},
            n2._weight_init,
            n2._extra_config,
        )
        assert not jnp.array_equal(p1.weights["e:in"], p2.weights["e:in"])


# =============================================================================
# ConvNode — slot management
# =============================================================================


class TestConvNodeSlots:
    """Tests for get_slots() from NodeBase behavior."""

    def test_get_slots_returns_default(self):
        slots = ConvNode.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True

    def test_skip_connection_variance_conflict_raises(self):
        """SlotSpec invariant: skip + variance_scalable cannot coexist."""
        with pytest.raises(ValueError):
            SlotSpec(
                name="skip",
                is_multi_input=False,
                is_skip_connection=True,
                is_variance_scalable=True,
            )


# =============================================================================
# ConvNode — get_weight_fan_in
# =============================================================================


class TestConvNodeFanIn:
    def test_fan_in_1d(self):
        fan = ConvNode.get_weight_fan_in((10, 3), {"kernel_size": (5,)})
        assert fan == 3 * 5  # C_in=3, kL=5

    def test_fan_in_2d(self):
        fan = ConvNode.get_weight_fan_in((28, 28, 3), {"kernel_size": (3, 3)})
        assert fan == 3 * 3 * 3  # C_in=3, kH=3, kW=3

    def test_fan_in_3d(self):
        fan = ConvNode.get_weight_fan_in((10, 10, 10, 4), {"kernel_size": (3, 3, 3)})
        assert fan == 4 * 27  # C_in=4, 3^3=27


# =============================================================================
# ConvNode — initialize_params
# =============================================================================


class TestConvNodeInitializeParams:
    """Tests for kernel shapes and bias shapes across 1D/2D/3D."""

    def _config(self, node):
        return node._extra_config

    def test_1d_kernel_and_bias_shape(self):
        node = ConvNode(shape=(10, 8), name="c", kernel_size=(5,))
        config = self._config(node)
        key = jax.random.PRNGKey(0)
        params = ConvNode.initialize_params(
            key, (10, 8), {"e:in": (10, 3)}, KaimingInitializer(), config
        )
        assert params.weights["e:in"].shape == (5, 3, 8)
        assert params.biases["b"].shape == (1, 1, 8)

    def test_2d_kernel_and_bias_shape(self):
        node = ConvNode(shape=(28, 28, 16), name="c", kernel_size=(3, 3))
        config = self._config(node)
        key = jax.random.PRNGKey(0)
        params = ConvNode.initialize_params(
            key, (28, 28, 16), {"e:in": (28, 28, 3)}, KaimingInitializer(), config
        )
        assert params.weights["e:in"].shape == (3, 3, 3, 16)
        assert params.biases["b"].shape == (1, 1, 1, 16)

    def test_3d_kernel_and_bias_shape(self):
        node = ConvNode(shape=(10, 10, 10, 8), name="c", kernel_size=(3, 3, 3))
        config = self._config(node)
        key = jax.random.PRNGKey(0)
        params = ConvNode.initialize_params(
            key,
            (10, 10, 10, 8),
            {"e:in": (10, 10, 10, 4)},
            KaimingInitializer(),
            config,
        )
        assert params.weights["e:in"].shape == (3, 3, 3, 4, 8)
        assert params.biases["b"].shape == (1, 1, 1, 1, 8)

    def test_no_bias(self):
        node = ConvNode(shape=(10, 8), name="c", kernel_size=(3,), use_bias=False)
        key = jax.random.PRNGKey(0)
        params = ConvNode.initialize_params(
            key, (10, 8), {"e:in": (10, 4)}, KaimingInitializer(), node._extra_config
        )
        assert params.biases == {}

    def test_use_bias_without_bias_init_raises(self):
        """use_bias=True with bias_init=None fails fast rather than passing None
        into the initializer (only reachable via a hand-built config)."""
        key = jax.random.PRNGKey(0)
        config = {"kernel_size": (3,), "use_bias": True, "bias_init": None}
        with pytest.raises(ValueError, match="bias_init is None"):
            ConvNode.initialize_params(
                key, (10, 8), {"e:in": (10, 4)}, KaimingInitializer(), config
            )

    def test_xavier_init_accepted(self):
        """Non-Kaiming initializers should be forwarded directly."""
        node = ConvNode(
            shape=(10, 8), name="c", kernel_size=(3,), weight_init=XavierInitializer()
        )
        key = jax.random.PRNGKey(0)
        params = ConvNode.initialize_params(
            key, (10, 8), {"e:in": (10, 4)}, XavierInitializer(), node._extra_config
        )
        assert params.weights["e:in"].shape == (3, 4, 8)

    def test_leaky_relu_kaiming(self):
        node = ConvNode(
            shape=(10, 8),
            name="c",
            kernel_size=(3,),
            activation=LeakyReLUActivation(alpha=0.1),
        )
        key = jax.random.PRNGKey(0)
        params = ConvNode.initialize_params(
            key, (10, 8), {"e:in": (10, 4)}, KaimingInitializer(), node._extra_config
        )
        assert params.weights["e:in"].shape == (3, 4, 8)

    def test_multi_input_allocates_multiple_kernels(self):
        node = ConvNode(shape=(10, 8), name="c", kernel_size=(3,))
        key = jax.random.PRNGKey(0)
        input_shapes = {"a->c:in": (10, 3), "b->c:in": (10, 5)}
        params = ConvNode.initialize_params(
            key, (10, 8), input_shapes, KaimingInitializer(), node._extra_config
        )
        assert params.weights["a->c:in"].shape == (3, 3, 8)
        assert params.weights["b->c:in"].shape == (3, 5, 8)

    def test_wrong_declared_output_shape_raises(self):
        """initialize_params fails fast when the declared output spatial shape
        disagrees with what the conv params produce (VALID shrinks 10 -> 8)."""
        node = ConvNode(shape=(8, 8), name="c", kernel_size=(3,), padding="VALID")
        key = jax.random.PRNGKey(0)
        # Declare (10, 8) but VALID k=3 on input length 10 yields length 8.
        with pytest.raises(ValueError, match="declared output spatial shape"):
            ConvNode.initialize_params(
                key,
                (10, 8),
                {"e:in": (10, 4)},
                KaimingInitializer(),
                node._extra_config,
            )


# =============================================================================
# ConvNode — forward pass
# =============================================================================


class TestConvNodeForward:
    """Shape and value tests for the forward pass."""

    def test_2d_known_value_cross_correlation(self):
        """Hand-computed cross-correlation on a *rectangular* input with an
        asymmetric kernel. Shape/energy-only tests pass even with transposed
        dimension numbers; this pins the actual arithmetic and H/W ordering.

        Input is 3x4 with values 0..11 row-major; kernel (kH=2, kW=1) = [1, -1]
        computes out[h, w] = x[h, w] - x[h+1, w]. Each row is +4 over the one
        below, so every output element is -4, on a (2, 4) grid. A transposed
        layout would instead produce a (3, 3) grid and fail the shape assert.
        """
        from fabricpc.core.activations import IdentityActivation

        x = jnp.arange(12.0).reshape(1, 3, 4, 1)
        kernel = jnp.array([1.0, -1.0]).reshape(2, 1, 1, 1)  # (kH, kW, C_in, C_out)
        config = {
            "kernel_size": (2, 1),
            "stride": (1, 1),
            "padding": "VALID",
            "use_bias": False,
        }
        node_info = _make_node_info(
            ConvNode,
            (2, 4, 1),
            config,
            IdentityActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
            in_edges=("e:in",),
        )
        params = NodeParams(weights={"e:in": kernel}, biases={})
        state = _make_state(jax.random.PRNGKey(0), 1, (2, 4, 1))
        _, new_state = ConvNode.forward(params, {"e:in": x}, state, node_info)
        assert new_state.z_mu.shape == (1, 2, 4, 1)
        assert jnp.allclose(new_state.z_mu, -4.0)

    @pytest.fixture(params=["SAME", "VALID"])
    def setup_1d(self, request):
        padding = request.param
        batch, seq_len, C_in, C_out, k = 4, 10, 3, 8, 3
        key = jax.random.PRNGKey(42)
        activation = ReLUActivation()
        energy_obj = GaussianEnergy()

        # Output length depends on padding: SAME preserves, VALID shrinks by k-1.
        out_len = seq_len if padding == "SAME" else seq_len - k + 1

        node = ConvNode(
            shape=(out_len, C_out), name="c1d", kernel_size=(k,), padding=padding
        )
        config = node._extra_config
        input_shapes = {"e:in": (seq_len, C_in)}
        params = ConvNode.initialize_params(
            key, (out_len, C_out), input_shapes, KaimingInitializer(), config
        )
        node_info = _make_node_info(
            ConvNode,
            (out_len, C_out),
            config,
            activation,
            energy_obj,
            NormalInitializer(),
            KaimingInitializer(),
            in_edges=("e:in",),
        )
        state = _make_state(key, batch, (out_len, C_out))

        inputs = {"e:in": jax.random.normal(key, (batch, seq_len, C_in))}
        return params, state, node_info, inputs, padding

    def test_1d_forward_shape(self, setup_1d):
        params, state, node_info, inputs, padding = setup_1d
        energy, new_state = ConvNode.forward(params, inputs, state, node_info)
        assert energy.shape == ()
        assert new_state.z_mu.shape == state.z_latent.shape
        assert new_state.error.shape == state.z_latent.shape

    def test_2d_forward_shape(self):
        batch, H, W, C_in, C_out = 4, 28, 28, 3, 16
        key = jax.random.PRNGKey(0)
        activation = ReLUActivation()
        energy_obj = GaussianEnergy()

        node = ConvNode(shape=(H, W, C_out), name="c2d", kernel_size=(3, 3))
        config = node._extra_config
        input_shapes = {"e:in": (H, W, C_in)}
        params = ConvNode.initialize_params(
            key, (H, W, C_out), input_shapes, KaimingInitializer(), config
        )
        node_info = _make_node_info(
            ConvNode,
            (H, W, C_out),
            config,
            activation,
            energy_obj,
            NormalInitializer(),
            KaimingInitializer(),
        )
        state = _make_state(key, batch, (H, W, C_out))
        inputs = {"e:in": jax.random.normal(key, (batch, H, W, C_in))}

        energy, new_state = ConvNode.forward(params, inputs, state, node_info)
        assert energy.shape == ()
        assert new_state.z_mu.shape == (batch, H, W, C_out)

    def test_3d_forward_shape(self):
        batch, D, H, W, C_in, C_out = 2, 8, 8, 8, 2, 4
        key = jax.random.PRNGKey(0)
        activation = ReLUActivation()
        energy_obj = GaussianEnergy()

        node = ConvNode(shape=(D, H, W, C_out), name="c3d", kernel_size=(3, 3, 3))
        config = node._extra_config
        input_shapes = {"e:in": (D, H, W, C_in)}
        params = ConvNode.initialize_params(
            key, (D, H, W, C_out), input_shapes, KaimingInitializer(), config
        )
        node_info = _make_node_info(
            ConvNode,
            (D, H, W, C_out),
            config,
            activation,
            energy_obj,
            NormalInitializer(),
            KaimingInitializer(),
        )
        state = _make_state(key, batch, (D, H, W, C_out))
        inputs = {"e:in": jax.random.normal(key, (batch, D, H, W, C_in))}

        energy, new_state = ConvNode.forward(params, inputs, state, node_info)
        assert energy.shape == ()
        assert new_state.z_mu.shape == (batch, D, H, W, C_out)

    def test_forward_energy_nonnegative_gaussian(self):
        """Gaussian energy is always non-negative."""
        batch, H, W, C_in, C_out = 4, 8, 8, 3, 8
        key = jax.random.PRNGKey(0)
        node = ConvNode(shape=(H, W, C_out), name="c", kernel_size=(3, 3))
        config = node._extra_config
        params = ConvNode.initialize_params(
            key, (H, W, C_out), {"e:in": (H, W, C_in)}, KaimingInitializer(), config
        )
        node_info = _make_node_info(
            ConvNode,
            (H, W, C_out),
            config,
            ReLUActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        state = _make_state(key, batch, (H, W, C_out))
        inputs = {"e:in": jax.random.normal(key, (batch, H, W, C_in))}

        energy, new_state = ConvNode.forward(params, inputs, state, node_info)
        assert float(energy) >= 0.0

    def test_forward_multi_input(self):
        """Two edges with different C_in each get their own kernel."""
        batch, seq, C1, C2, C_out = 2, 12, 3, 5, 8
        key = jax.random.PRNGKey(7)
        node = ConvNode(shape=(seq, C_out), name="c", kernel_size=(3,))
        config = node._extra_config
        input_shapes = {"a:in": (seq, C1), "b:in": (seq, C2)}
        params = ConvNode.initialize_params(
            key, (seq, C_out), input_shapes, KaimingInitializer(), config
        )
        node_info = _make_node_info(
            ConvNode,
            (seq, C_out),
            config,
            ReLUActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
            in_edges=("a:in", "b:in"),
        )
        state = _make_state(key, batch, (seq, C_out))
        inputs = {
            "a:in": jax.random.normal(key, (batch, seq, C1)),
            "b:in": jax.random.normal(key, (batch, seq, C2)),
        }
        energy, new_state = ConvNode.forward(params, inputs, state, node_info)
        assert new_state.z_mu.shape == (batch, seq, C_out)

    def test_latent_grads(self):
        batch, H, W, C_in, C_out = 2, 8, 8, 2, 4
        key = jax.random.PRNGKey(0)
        node = ConvNode(shape=(H, W, C_out), name="c", kernel_size=(3, 3))
        config = node._extra_config
        params = ConvNode.initialize_params(
            key, (H, W, C_out), {"e:in": (H, W, C_in)}, KaimingInitializer(), config
        )
        node_info = _make_node_info(
            ConvNode,
            (H, W, C_out),
            config,
            ReLUActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        state = _make_state(key, batch, (H, W, C_out))
        inputs = {"e:in": jax.random.normal(key, (batch, H, W, C_in))}

        new_state, input_grads, self_grad = ConvNode.forward_and_latent_grads(
            params, inputs, state, node_info, is_clamped=False
        )
        assert "e:in" in input_grads
        assert input_grads["e:in"].shape == inputs["e:in"].shape
        assert self_grad.shape == state.z_latent.shape

    def test_weight_grads(self):
        batch, H, W, C_in, C_out = 2, 8, 8, 2, 4
        key = jax.random.PRNGKey(0)
        node = ConvNode(shape=(H, W, C_out), name="c", kernel_size=(3, 3))
        config = node._extra_config
        params = ConvNode.initialize_params(
            key, (H, W, C_out), {"e:in": (H, W, C_in)}, KaimingInitializer(), config
        )
        node_info = _make_node_info(
            ConvNode,
            (H, W, C_out),
            config,
            ReLUActivation(),
            GaussianEnergy(),
            NormalInitializer(),
            KaimingInitializer(),
        )
        state = _make_state(key, batch, (H, W, C_out))
        inputs = {"e:in": jax.random.normal(key, (batch, H, W, C_in))}

        new_state, param_grads = ConvNode.forward_and_weight_grads(
            params, inputs, state, node_info
        )
        assert param_grads.weights["e:in"].shape == params.weights["e:in"].shape
