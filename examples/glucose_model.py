"""Glucose transformer graph builder and custom nodes.

Implements the GluMind-Uni architecture as a FabricPC predictive-coding graph:

    glucose (batch, 128, 1)
      → ContinuousEmbed (batch, 128, d_model)
      → [MultiScaleMha(+skip) → LnMlp1 → Mlp2Residual(+skip)] × depth
      → RegressionOutput (batch, horizon)

Multi-scale self-attention operates at DS=1, DS=2, DS=4 with RoPE.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams
from fabricpc.core.initializers import (
    NormalInitializer,
    XavierInitializer,
    initialize,
)
from fabricpc.core.positional import precompute_freqs_cis, apply_rotary_emb
from fabricpc.utils.helpers import layernorm
from fabricpc.core.activations import IdentityActivation, GeluActivation
from fabricpc.core.energy import EnergyFunctional, GaussianEnergy
from fabricpc.nodes import LnMlp1Node, Mlp2ResidualNode, Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.inference import InferenceBase, InferenceSGDNormClip
from fabricpc.core.mupc import MuPCConfig
from fabricpc.graph_initialization import FeedforwardStateInit


class ContinuousEmbeddingNode(NodeBase):
    """Project a scalar time series to d_model via a learned linear layer."""

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(self, shape, name, embed_dim, in_channels=1,
                 weight_init=XavierInitializer(),
                 latent_init=NormalInitializer(),
                 energy=GaussianEnergy(), **kwargs):
        super().__init__(shape=shape, name=name, embed_dim=embed_dim,
                         in_channels=in_channels, weight_init=weight_init,
                         latent_init=latent_init, energy=energy, **kwargs)

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        embed_dim = config["embed_dim"]
        in_channels = config.get("in_channels", 1)
        weights = {"W_embed": initialize(key, (in_channels, embed_dim), weight_init)}
        biases = {"b_embed": jnp.zeros((embed_dim,))}
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = inputs[list(inputs.keys())[0]]
        z_mu = jnp.dot(x, params.weights["W_embed"]) + params.biases["b_embed"]
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        return node_info.node_class.energy_functional(state, node_info)


def _avg_pool_1d(x, factor):
    B, S, D = x.shape
    S_out = S // factor
    return x[:, :S_out * factor, :].reshape(B, S_out, factor, D).mean(axis=2)


def _upsample_nearest_1d(x, target_len):
    indices = jnp.arange(target_len) * x.shape[1] // target_len
    return x[:, indices, :]


class MultiScaleMhaResidualNode(NodeBase):
    """Multi-scale self-attention at DS=1, DS=2, DS=4 with RoPE and residual."""

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(self, shape, name, embed_dim, num_heads,
                 use_rope=True, rope_theta=10000.0,
                 weight_init=XavierInitializer(),
                 latent_init=NormalInitializer(),
                 energy=GaussianEnergy(), **kwargs):
        super().__init__(shape=shape, name=name, embed_dim=embed_dim,
                         num_heads=num_heads, use_rope=use_rope,
                         rope_theta=rope_theta, weight_init=weight_init,
                         latent_init=latent_init, energy=energy, **kwargs)

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec("in", False),
            "skip": SlotSpec("skip", False,
                             is_variance_scalable=False,
                             is_skip_connection=True),
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        dim = config["embed_dim"]
        keys = jax.random.split(key, 14)

        def iw(k, s):
            return initialize(k, s, weight_init)

        weights = {"ln_gamma": jnp.ones((dim,))}
        biases = {"ln_beta": jnp.zeros((dim,))}
        for suffix in ("1", "2", "4"):
            k_base = {"1": 0, "2": 4, "4": 8}[suffix]
            for i, p in enumerate(("q", "k", "v", "o")):
                weights[f"W_{p}{suffix}"] = iw(keys[k_base + i], (dim, dim))
                biases[f"b_{p}{suffix}"] = jnp.zeros((dim,))
        return NodeParams(weights, biases)

    @staticmethod
    def _self_attention(x, W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o,
                        num_heads, use_rope, rope_theta):
        B, L, D = x.shape
        hd = D // num_heads
        Q = (jnp.dot(x, W_q) + b_q).reshape(B, L, num_heads, hd)
        K = (jnp.dot(x, W_k) + b_k).reshape(B, L, num_heads, hd)
        V = (jnp.dot(x, W_v) + b_v).reshape(B, L, num_heads, hd)
        if use_rope:
            freqs = precompute_freqs_cis(hd, L, theta=rope_theta)
            Q, K = apply_rotary_emb(Q, K, freqs)
        Q, K, V = (t.transpose(0, 2, 1, 3) for t in (Q, K, V))
        scores = jnp.matmul(Q, K.swapaxes(-1, -2)) / jnp.sqrt(hd)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, L, D)
        return jnp.dot(out, W_o) + b_o

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = inputs[next(k for k in inputs if k.endswith(":in"))]
        skip_key = next((k for k in inputs if k.endswith(":skip")), None)
        skip = inputs[skip_key] if skip_key else x

        cfg = node_info.node_config
        nh = cfg["num_heads"]
        rope = cfg.get("use_rope", True)
        theta = cfg.get("rope_theta", 10000.0)
        seq_len = x.shape[1]
        w, b = params.weights, params.biases

        x_n = layernorm(x, w["ln_gamma"], b["ln_beta"])

        sa = MultiScaleMhaResidualNode._self_attention
        high = sa(x_n, w["W_q1"], w["W_k1"], w["W_v1"], w["W_o1"],
                  b["b_q1"], b["b_k1"], b["b_v1"], b["b_o1"], nh, rope, theta)
        low2 = sa(_avg_pool_1d(x_n, 2), w["W_q2"], w["W_k2"], w["W_v2"], w["W_o2"],
                  b["b_q2"], b["b_k2"], b["b_v2"], b["b_o2"], nh, rope, theta)
        low4 = sa(_avg_pool_1d(x_n, 4), w["W_q4"], w["W_k4"], w["W_v4"], w["W_o4"],
                  b["b_q4"], b["b_k4"], b["b_v4"], b["b_o4"], nh, rope, theta)

        fused = high + _upsample_nearest_1d(low2, seq_len) + _upsample_nearest_1d(low4, seq_len)
        z_mu = skip + fused
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        return node_info.node_class.energy_functional(state, node_info)


class RegressionOutputNode(NodeBase):
    """Sequence → pooled/flat features → Linear → GELU → Linear regression head.

    ``readout`` modes:
    - ``flatten``: full ``seq_len * embed_dim`` projection (GluMind-style default)
    - ``mean_pool``: mean over time, then ``embed_dim`` projection (lighter PC head)
    - ``last``: last timestep only (lighter PC head)
    """

    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(self, shape, name, seq_len, embed_dim, horizon,
                 readout: str = "flatten",
                 weight_init=XavierInitializer(),
                 latent_init=NormalInitializer(),
                 energy=GaussianEnergy(), **kwargs):
        if readout not in {"flatten", "mean_pool", "last"}:
            raise ValueError(
                f"readout must be flatten|mean_pool|last, got {readout!r}"
            )
        super().__init__(shape=shape, name=name, seq_len=seq_len,
                         embed_dim=embed_dim, horizon=horizon,
                         readout=readout,
                         weight_init=weight_init, latent_init=latent_init,
                         energy=energy, **kwargs)

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        readout = config.get("readout", "flatten")
        embed_dim, horizon = config["embed_dim"], config["horizon"]
        if readout == "flatten":
            feature_dim = config["seq_len"] * embed_dim
        else:
            feature_dim = embed_dim
        keys = jax.random.split(key, 2)
        weights = {
            "W_flat": initialize(keys[0], (feature_dim, embed_dim), weight_init),
            "W_out": initialize(keys[1], (embed_dim, horizon), weight_init),
        }
        biases = {"b_flat": jnp.zeros((embed_dim,)), "b_out": jnp.zeros((horizon,))}
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        x = inputs[list(inputs.keys())[0]]
        readout = node_info.node_config.get("readout", "flatten")
        if readout == "mean_pool":
            features = jnp.mean(x, axis=1)
        elif readout == "last":
            features = x[:, -1, :]
        else:
            features = x.reshape(x.shape[0], -1)
        h = jax.nn.gelu(
            jnp.dot(features, params.weights["W_flat"]) + params.biases["b_flat"]
        )
        z_mu = jnp.dot(h, params.weights["W_out"]) + params.biases["b_out"]
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        return node_info.node_class.energy_functional(state, node_info)


def create_glucose_transformer(
    depth: int = 3,
    embed_dim: int = 32,
    num_heads: int = 4,
    mlp_dim: int = 128,
    seq_len: int = 128,
    horizon: int = 12,
    in_channels: int = 1,
    inference: InferenceBase | None = None,
    weight_init_std: float = 0.02,
    use_rope: bool = True,
    include_output_scaling: bool = False,
    readout: str = "flatten",
    energy: EnergyFunctional | None = None,
):
    """Build a glucose transformer graph for forecasting.

    Default geometry matches GluMind-Uni: seq_len=128 (10.67 h at 5-min),
    depth=3, embed_dim=32, num_heads=4, mlp_dim=128, horizon=12 (60 min).
    Set ``include_output_scaling=True`` for Gaussian/MSE regression muPC
    scaling; the default remains false for checkpoint compatibility.
    ``readout`` selects the regression head pooling mode (see
    ``RegressionOutputNode``).
    """
    assert seq_len % 4 == 0, f"seq_len must be divisible by 4, got {seq_len}"

    if inference is None:
        inference = InferenceSGDNormClip(
            eta_infer=5e-5, infer_steps=12, max_norm=1.0
        )

    if energy is None:
        energy = GaussianEnergy()

    w_init = NormalInitializer(std=weight_init_std)

    nodes, edges = [], []

    input_node = Linear(
        shape=(seq_len, in_channels),
        activation=IdentityActivation(),
        name="glucose_input",
        energy=energy,
    )
    nodes.append(input_node)

    embed = ContinuousEmbeddingNode(
        name="embed", shape=(seq_len, embed_dim),
        embed_dim=embed_dim, in_channels=in_channels,
        weight_init=XavierInitializer(),
        energy=energy,
    )
    nodes.append(embed)
    edges.append(Edge(source=input_node, target=embed.slot("in")))

    prev = embed
    for i in range(depth):
        msha = MultiScaleMhaResidualNode(
            name=f"L{i}_msha", shape=(seq_len, embed_dim),
            embed_dim=embed_dim, num_heads=num_heads,
            use_rope=use_rope, weight_init=w_init,
            energy=energy,
        )
        nodes.append(msha)
        edges.append(Edge(source=prev, target=msha.slot("in")))
        edges.append(Edge(source=prev, target=msha.slot("skip")))

        mlp1 = LnMlp1Node(
            name=f"L{i}_mlp1", shape=(seq_len, mlp_dim),
            embed_dim=embed_dim, ff_dim=mlp_dim,
            activation=GeluActivation(), weight_init=w_init,
            energy=energy,
        )
        nodes.append(mlp1)
        edges.append(Edge(source=msha, target=mlp1.slot("in")))

        mlp2 = Mlp2ResidualNode(
            name=f"L{i}_mlp2", shape=(seq_len, embed_dim),
            embed_dim=embed_dim, ff_dim=mlp_dim, weight_init=w_init,
            energy=energy,
        )
        nodes.append(mlp2)
        edges.append(Edge(source=mlp1, target=mlp2.slot("in")))
        edges.append(Edge(source=msha, target=mlp2.slot("residual")))

        prev = mlp2

    output = RegressionOutputNode(
        name="output", shape=(horizon,), seq_len=seq_len,
        embed_dim=embed_dim, horizon=horizon,
        readout=readout,
        weight_init=NormalInitializer(std=float(jnp.sqrt(1.0 / embed_dim))),
        energy=energy,
    )
    nodes.append(output)
    edges.append(Edge(source=prev, target=output.slot("in")))

    return graph(
        nodes=nodes, edges=edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=inference,
        scaling=MuPCConfig(include_output=include_output_scaling),
        graph_state_initializer=FeedforwardStateInit(),
    )
