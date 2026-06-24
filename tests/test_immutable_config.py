"""
Tests for the immutable config-object guarantee behind Flax-style signature
defaults.

Node ``__init__`` signatures place stateless value objects
(``ReLUActivation()``, ``GaussianEnergy()``, ``KaimingInitializer()``, ...)
directly as parameter defaults. A signature default is evaluated once at import
and shared by every defaulted call, so that singleton is only safe if it cannot
be mutated. These tests assert the three config base classes
(``ActivationBase`` / ``EnergyFunctional`` / ``InitializerBase``) are frozen
after construction, that nodes do share the one default instance, and that the
graph's copy-on-finalize path still works when nodes hold these frozen objects.
"""

import copy

import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import ConvNode, MaxPool, Linear, IdentityNode
from fabricpc.nodes.transformer_v2 import (
    MhaResidualNode,
    LnMlp1Node,
    Mlp2ResidualNode,
)
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference, InferenceSGD
from fabricpc.core.activations import (
    ReLUActivation,
    SoftmaxActivation,
    GeluActivation,
)
from fabricpc.core.energy import GaussianEnergy, CrossEntropyEnergy
from fabricpc.core.initializers import KaimingInitializer
from fabricpc.core._frozen import FrozenConfig
from fabricpc.nodes.transformer import TransformerBlock

# One representative from each of the three frozen base-class families.
CONFIG_OBJECTS = [
    pytest.param(ReLUActivation(), id="activation"),
    pytest.param(GaussianEnergy(), id="energy"),
    pytest.param(KaimingInitializer(), id="initializer"),
]


# =============================================================================
# 1. Immutability is enforced (safe by construction)
# =============================================================================


class TestImmutability:
    @pytest.mark.parametrize("obj", CONFIG_OBJECTS)
    def test_config_mapping_is_readonly(self, obj):
        """``config`` is a MappingProxyType: item assignment raises TypeError."""
        with pytest.raises(TypeError):
            obj.config["new_key"] = 1

    @pytest.mark.parametrize("obj", CONFIG_OBJECTS)
    def test_cannot_set_new_attribute(self, obj):
        with pytest.raises(AttributeError):
            obj.new_attr = 1

    @pytest.mark.parametrize("obj", CONFIG_OBJECTS)
    def test_cannot_reassign_config(self, obj):
        with pytest.raises(AttributeError):
            obj.config = {}

    @pytest.mark.parametrize("obj", CONFIG_OBJECTS)
    def test_cannot_delete_attribute(self, obj):
        with pytest.raises(AttributeError):
            del obj.config

    def test_config_value_survives_freeze(self):
        """Freezing does not break ordinary construction-time configuration."""
        assert GaussianEnergy(precision=2.0).config["precision"] == 2.0
        assert KaimingInitializer(mode="fan_out").config["mode"] == "fan_out"


# =============================================================================
# 2. The shared signature-default singleton is safe
# =============================================================================


class TestSharedDefaultIsSafe:
    def test_conv_nodes_share_one_immutable_default(self):
        """Two ConvNodes built without overrides share the same default objects,
        and those objects cannot be mutated — so the singleton cannot leak
        state from one node to another."""
        a = ConvNode(shape=(8, 8, 4), name="a", kernel_size=(3, 3))
        b = ConvNode(shape=(8, 8, 4), name="b", kernel_size=(3, 3))

        assert a._activation is b._activation
        assert a._energy is b._energy
        assert a._weight_init is b._weight_init

        with pytest.raises(AttributeError):
            a._activation.leaked = 1
        with pytest.raises(TypeError):
            a._weight_init.config["mode"] = "fan_out"

    def test_transformer_nodes_share_one_immutable_default(self):
        a = MhaResidualNode(shape=(5, 8), name="a", embed_dim=8, num_heads=2)
        b = MhaResidualNode(shape=(5, 8), name="b", embed_dim=8, num_heads=2)

        assert a._weight_init is b._weight_init
        assert a._energy is b._energy
        assert a._latent_init is b._latent_init

        with pytest.raises(AttributeError):
            a._weight_init.leaked = 1


# =============================================================================
# 3. Copy-on-finalize still works with frozen-object-holding nodes
# =============================================================================


def _build_conv_pool_graph(use_bias=True):
    """input(8,8,1) -> conv(8,8,4) -> maxpool(4,4,4) -> linear(3)."""
    pixels = IdentityNode(shape=(8, 8, 1), name="pixels")
    conv = ConvNode(
        shape=(8, 8, 4),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        use_bias=use_bias,
        name="conv",
    )
    pool = MaxPool(
        shape=(4, 4, 4),
        window_shape=(2, 2),
        stride=(2, 2),
        padding="VALID",
        name="pool",
    )
    out = Linear(
        shape=(3,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="out",
    )
    return graph(
        nodes=[pixels, conv, pool, out],
        edges=[
            Edge(source=pixels, target=conv.slot("in")),
            Edge(source=conv, target=pool.slot("in")),
            Edge(source=pool, target=out.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
    )


class TestCopyOnFinalize:
    def test_node_copy_shares_frozen_objects_by_reference(self):
        """``_with_graph_info`` does ``copy.copy`` on the node; the shallow copy
        shares the (frozen) activation/energy/init objects by reference, which is
        exactly why freezing them is safe."""
        conv = ConvNode(shape=(8, 8, 4), name="conv", kernel_size=(3, 3))
        clone = copy.copy(conv)
        assert clone._activation is conv._activation
        assert clone._energy is conv._energy
        assert clone._weight_init is conv._weight_init
        with pytest.raises(AttributeError):
            clone._activation.leaked = 1

    def test_conv_pool_graph_finalizes_and_runs(self, rng_key):
        structure = _build_conv_pool_graph()
        params = initialize_params(structure, rng_key)

        batch = 2
        clamps = {
            "pixels": jax.random.normal(rng_key, (batch, 8, 8, 1)),
            "out": jax.nn.one_hot(jnp.zeros((batch,), dtype=jnp.int32), 3),
        }
        state = initialize_graph_state(
            structure, batch, rng_key, clamps=clamps, params=params
        )
        final = run_inference(params, state, clamps, structure)

        assert final.nodes["conv"].z_latent.shape == (batch, 8, 8, 4)
        assert bool(jnp.all(jnp.isfinite(final.nodes["conv"].z_latent)))
        # The finalized node still holds the frozen default, immutable as ever.
        with pytest.raises(AttributeError):
            structure.nodes["conv"].node_info.activation.leaked = 1

    def test_transformer_block_finalizes_and_runs(self, rng_key):
        seq, dim, ff = 6, 8, 16
        inp = Linear(shape=(seq, dim), name="input")
        mha = MhaResidualNode(shape=(seq, dim), name="mha", embed_dim=dim, num_heads=2)
        mlp1 = LnMlp1Node(shape=(seq, ff), name="mlp1", embed_dim=dim, ff_dim=ff)
        mlp2 = Mlp2ResidualNode(shape=(seq, dim), name="mlp2", embed_dim=dim, ff_dim=ff)
        out = Linear(shape=(seq, dim), name="output")

        structure = graph(
            nodes=[inp, mha, mlp1, mlp2, out],
            edges=[
                Edge(source=inp, target=mha.slot("in")),
                Edge(source=mha, target=mlp1.slot("in")),
                Edge(source=mlp1, target=mlp2.slot("in")),
                Edge(source=mha, target=mlp2.slot("residual")),
                Edge(source=mlp2, target=out.slot("in")),
            ],
            task_map=TaskMap(x=inp, y=out),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
        )
        params = initialize_params(structure, rng_key)

        batch = 2
        x = jax.random.normal(rng_key, (batch, seq, dim))
        clamps = {"input": x, "output": jnp.zeros_like(x)}
        state = initialize_graph_state(
            structure, batch, rng_key, clamps=clamps, params=params
        )
        final = run_inference(params, state, clamps, structure)

        assert final.nodes["mlp2"].z_latent.shape == (batch, seq, dim)
        with pytest.raises(AttributeError):
            structure.nodes["mha"].node_info.weight_init.leaked = 1


# =============================================================================
# 4. ConvNode bias behavior is unchanged by the migration
# =============================================================================


def test_conv_bias_parity(rng_key):
    """use_bias=True yields a bias param; use_bias=False yields none — the same
    parameter sets as before the signature-default migration."""
    p_bias = initialize_params(_build_conv_pool_graph(use_bias=True), rng_key)
    p_nobias = initialize_params(_build_conv_pool_graph(use_bias=False), rng_key)

    assert "b" in p_bias.nodes["conv"].biases
    assert p_nobias.nodes["conv"].biases == {}


# =============================================================================
# 5. The freeze is defined once (shared FrozenConfig mixin)
# =============================================================================


def test_freeze_is_single_source():
    """All three families inherit one freeze implementation, so a future change
    to the freeze cannot drift across copies."""
    for obj in (ReLUActivation(), GaussianEnergy(), KaimingInitializer()):
        assert type(obj).__setattr__ is FrozenConfig.__setattr__
        assert type(obj).__delattr__ is FrozenConfig.__delattr__


# =============================================================================
# 6. TransformerBlock internal_activation: default vs explicit None
# =============================================================================


def test_transformer_block_internal_activation_default_and_none():
    """The signature default is a shared, frozen GELU; passing None selects the
    identity path that forward() implements (previously unreachable because the
    old ``internal_activation or GeluActivation()`` idiom forced a non-None
    value)."""
    a = TransformerBlock(shape=(4, 8), name="a", num_heads=2)
    b = TransformerBlock(shape=(4, 8), name="b", num_heads=2)

    assert isinstance(a._extra_config["internal_activation"], GeluActivation)
    # One shared, frozen singleton across defaulted nodes.
    assert (
        a._extra_config["internal_activation"] is b._extra_config["internal_activation"]
    )
    with pytest.raises(AttributeError):
        a._extra_config["internal_activation"].leaked = 1

    identity_block = TransformerBlock(
        shape=(4, 8), name="c", num_heads=2, internal_activation=None
    )
    assert identity_block._extra_config["internal_activation"] is None
