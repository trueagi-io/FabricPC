"""
Tests for TransformerBlock muPC variance control.

Verifies:
- get_weight_fan_in returns embed_dim
- forward() with muPC scaling produces z_mu with Var ~ 1.0
- forward() without muPC scaling preserves original behavior
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import (
    initialize_graph_state,
    FeedforwardStateInit,
)


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


# ============================================================================
# get_weight_fan_in tests
# ============================================================================


class TestTransformerFanIn:

    def test_returns_embed_dim(self):
        """get_weight_fan_in should return the last dimension (embed_dim)."""
        assert TransformerBlock.get_weight_fan_in((128, 512), {}) == 512
        assert TransformerBlock.get_weight_fan_in((64, 256), {}) == 256
        assert TransformerBlock.get_weight_fan_in((32, 768), {}) == 768

    def test_ignores_flatten_input(self):
        """flatten_input should not affect transformer fan_in (always last dim)."""
        assert (
            TransformerBlock.get_weight_fan_in((128, 512), {"flatten_input": True})
            == 512
        )


# ============================================================================
# Variance propagation tests
# ============================================================================


class TestTransformerMuPCVariance:

    @pytest.fixture
    def transformer_graph_mupc(self, rng_key):
        """TransformerBlock graph with muPC scaling."""
        seq_len, embed_dim = 32, 64
        num_heads = 8

        x = IdentityNode(shape=(seq_len, embed_dim), name="x")
        t = TransformerBlock(
            shape=(seq_len, embed_dim),
            name="transformer",
            num_heads=num_heads,
        )
        y = Linear(shape=(seq_len, 10), name="y")

        structure = graph(
            nodes=[x, t, y],
            edges=[
                Edge(source=x, target=t.slot("in")),
                Edge(source=t, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
            scaling=MuPCConfig(),
        )
        return structure

    @pytest.fixture
    def transformer_graph_no_mupc(self, rng_key):
        """TransformerBlock graph without muPC scaling."""
        seq_len, embed_dim = 32, 64
        num_heads = 8

        x = IdentityNode(shape=(seq_len, embed_dim), name="x")
        t = TransformerBlock(
            shape=(seq_len, embed_dim),
            name="transformer",
            num_heads=num_heads,
        )
        y = Linear(shape=(seq_len, 10), name="y")

        structure = graph(
            nodes=[x, t, y],
            edges=[
                Edge(source=x, target=t.slot("in")),
                Edge(source=t, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
        )
        return structure

    def test_mupc_scaling_attached(self, transformer_graph_mupc):
        """TransformerBlock should have muPC scaling_config when MuPCConfig is used."""
        t_node = transformer_graph_mupc.nodes["transformer"]
        assert t_node.node_info.scaling_config is not None

    def test_no_mupc_scaling_when_absent(self, transformer_graph_no_mupc):
        """TransformerBlock should have no scaling_config without MuPCConfig."""
        t_node = transformer_graph_no_mupc.nodes["transformer"]
        assert t_node.node_info.scaling_config is None

    def test_z_mu_variance_near_unity_with_mupc(self, transformer_graph_mupc, rng_key):
        """With muPC scaling, z_mu should have Var ~ 1.0 at init."""
        structure = transformer_graph_mupc
        params = initialize_params(structure, rng_key)

        # Use multiple batches for stable variance estimate
        batch_size = 64
        k1, k2 = jax.random.split(rng_key)

        # Generate unit-variance input
        seq_len, embed_dim = 32, 64
        x_data = jax.random.normal(k1, (batch_size, seq_len, embed_dim))
        y_data = jax.random.normal(k2, (batch_size, seq_len, 10))
        clamps = {"x": x_data, "y": y_data}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps=clamps,
            params=params,
            state_init=FeedforwardStateInit(),
        )

        z_mu = state.nodes["transformer"].z_mu
        var_z_mu = float(jnp.var(z_mu))

        # Variance should be near unity. Allow generous tolerance for
        # finite-size effects (softmax averaging approximation, etc.)
        assert 0.3 < var_z_mu < 3.0, (
            f"z_mu variance {var_z_mu:.4f} is outside acceptable range [0.3, 3.0] "
            f"for muPC-scaled TransformerBlock"
        )

    def test_z_mu_variance_different_seq_lens(self, rng_key):
        """Variance should remain near unity across different sequence lengths."""
        embed_dim = 64
        num_heads = 8
        batch_size = 64

        for seq_len in [16, 64, 128]:
            x = IdentityNode(shape=(seq_len, embed_dim), name="x")
            t = TransformerBlock(
                shape=(seq_len, embed_dim),
                name="transformer",
                num_heads=num_heads,
            )
            y = Linear(shape=(seq_len, 10), name="y")

            structure = graph(
                nodes=[x, t, y],
                edges=[
                    Edge(source=x, target=t.slot("in")),
                    Edge(source=t, target=y.slot("in")),
                ],
                task_map=TaskMap(x=x, y=y),
                inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
                scaling=MuPCConfig(),
            )

            k1, k2 = jax.random.split(jax.random.fold_in(rng_key, seq_len))
            params = initialize_params(structure, k1)
            x_data = jax.random.normal(k1, (batch_size, seq_len, embed_dim))
            y_data = jax.random.normal(k2, (batch_size, seq_len, 10))

            state = initialize_graph_state(
                structure,
                batch_size,
                k2,
                clamps={"x": x_data, "y": y_data},
                params=params,
                state_init=FeedforwardStateInit(),
            )

            var_z_mu = float(jnp.var(state.nodes["transformer"].z_mu))
            assert (
                0.2 < var_z_mu < 5.0
            ), f"seq_len={seq_len}: z_mu variance {var_z_mu:.4f} outside [0.2, 5.0]"

    def test_no_mupc_variance_still_controlled(
        self, transformer_graph_no_mupc, rng_key
    ):
        """Without muPC, internal scaling (1/sqrt(2) residuals, sqrt-eff-ctx) still applies."""
        structure = transformer_graph_no_mupc
        params = initialize_params(structure, rng_key)

        batch_size = 16
        seq_len, embed_dim = 32, 64
        k1, k2 = jax.random.split(rng_key)
        x_data = jax.random.normal(k1, (batch_size, seq_len, embed_dim))
        y_data = jax.random.normal(k2, (batch_size, seq_len, 10))

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps={"x": x_data, "y": y_data},
            params=params,
            state_init=FeedforwardStateInit(),
        )

        # Internal variance control is always on — z_mu should be near unity
        # even without muPC inter-node scaling.
        var_z_mu = float(jnp.var(state.nodes["transformer"].z_mu))
        assert (
            0.2 < var_z_mu < 5.0
        ), f"Non-muPC z_mu variance {var_z_mu:.4f} outside [0.2, 5.0]"
