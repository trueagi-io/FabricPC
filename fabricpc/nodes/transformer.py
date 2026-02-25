"""Transformer block node and components for building transformer architectures."""

from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, NodeParams, SlotSpec
from fabricpc.core.activations import (
    ActivationBase,
    IdentityActivation,
    GeluActivation,
)
from fabricpc.core.energy import EnergyFunctional, GaussianEnergy
from fabricpc.core.initializers import InitializerBase
from fabricpc.core.types import NodeState, NodeInfo


def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    dim_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    freqs = 1.0 / (theta ** (dim_indices / head_dim))
    positions = jnp.arange(max_seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, freqs)

    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary_emb(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    seq_len = x.shape[2]

    cos = cos[:seq_len, :]
    sin = sin[:seq_len, :]

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    x_rot = jnp.stack([x_even_rot, x_odd_rot], axis=-1)
    x_rot = x_rot.reshape(x.shape)

    return x_rot


class TransformerBlockNode(NodeBase):
    """Complete Transformer Block with attention and FFN."""

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        activation: ActivationBase | None = None,
        energy: EnergyFunctional | None = None,
        latent_init: InitializerBase | None = None,
        num_heads: int = 8,
        ff_dim: int | None = None,
        internal_activation: ActivationBase | None = None,
        dropout_rate: float = 0.0,
        pre_norm: bool = True,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        weight_init: InitializerBase | None = None,
    ):
        super().__init__(
            name,
            shape,
            activation=activation or IdentityActivation(),
            energy=energy or GaussianEnergy(),
            latent_init=latent_init,
            num_heads=num_heads,
            ff_dim=ff_dim,
            internal_activation=internal_activation or GeluActivation(),
            dropout_rate=dropout_rate,
            pre_norm=pre_norm,
            use_rope=use_rope,
            rope_theta=rope_theta,
            weight_init=weight_init,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec(name="in", is_multi_input=False),
            "mask": SlotSpec(name="mask", is_multi_input=False),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        num_heads = int(config.get("num_heads", 8))
        embed_dim = node_shape[-1]
        ff_dim = config.get("ff_dim") or (4 * embed_dim)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        keys = jax.random.split(key, 8)
        std = 1.0 / jnp.sqrt(embed_dim)

        return NodeParams(
            weights={
                "W_q": jax.random.normal(keys[0], (embed_dim, embed_dim)) * std,
                "W_k": jax.random.normal(keys[1], (embed_dim, embed_dim)) * std,
                "W_v": jax.random.normal(keys[2], (embed_dim, embed_dim)) * std,
                "W_o": jax.random.normal(keys[3], (embed_dim, embed_dim)) * std,
                "W_ff1": jax.random.normal(keys[4], (embed_dim, ff_dim))
                * std
                * jnp.sqrt(ff_dim / embed_dim),
                "W_ff2": jax.random.normal(keys[5], (ff_dim, embed_dim))
                * std
                * jnp.sqrt(ff_dim / embed_dim),
                "ln1_gamma": jnp.ones((1, 1, embed_dim)),
                "ln2_gamma": jnp.ones((1, 1, embed_dim)),
            },
            biases={
                "b_q": jnp.zeros((1, 1, embed_dim)),
                "b_k": jnp.zeros((1, 1, embed_dim)),
                "b_v": jnp.zeros((1, 1, embed_dim)),
                "b_o": jnp.zeros((1, 1, embed_dim)),
                "b_ff1": jnp.zeros((1, 1, ff_dim)),
                "b_ff2": jnp.zeros((1, 1, embed_dim)),
                "ln1_beta": jnp.zeros((1, 1, embed_dim)),
                "ln2_beta": jnp.zeros((1, 1, embed_dim)),
            },
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        config = node_info.node_config
        num_heads = int(config.get("num_heads", 8))

        in_edge_key = next(iter(k for k in inputs.keys() if k.endswith(":in")))
        input_tensor = inputs[in_edge_key]

        batch_size, seq_len, embed_dim = input_tensor.shape
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        mask_edge_key = next((k for k in inputs.keys() if k.endswith(":mask")), None)
        mask = inputs[mask_edge_key] if mask_edge_key else None
        internal_activation = config.get("internal_activation", GeluActivation())

        x_norm1 = TransformerBlockNode._layernorm(
            input_tensor, params.weights["ln1_gamma"], params.biases["ln1_beta"]
        )

        attn_output, substructure_attn = TransformerBlockNode._mha(
            x_norm1,
            mask,
            num_heads,
            params.weights["W_q"],
            params.weights["W_k"],
            params.weights["W_v"],
            params.weights["W_o"],
            params.biases["b_q"],
            params.biases["b_k"],
            params.biases["b_v"],
            params.biases["b_o"],
            lambda x: x,
            use_rope=bool(config.get("use_rope", True)),
            rope_theta=float(config.get("rope_theta", 10000.0)),
        )

        x_res1 = input_tensor + attn_output

        x_norm2 = TransformerBlockNode._layernorm(
            x_res1, params.weights["ln2_gamma"], params.biases["ln2_beta"]
        )

        ff_intermediate = (
            jnp.matmul(x_norm2, params.weights["W_ff1"]) + params.biases["b_ff1"]
        )
        ff_activated = internal_activation.forward(ff_intermediate)
        ff_output = (
            jnp.matmul(ff_activated, params.weights["W_ff2"]) + params.biases["b_ff2"]
        )

        z_mu = x_res1 + ff_output

        pre_activation = z_mu
        error = state.z_latent - z_mu

        state = state._replace(
            z_mu=z_mu,
            pre_activation=pre_activation,
            error=error,
            substructure={**state.substructure, **substructure_attn},
        )

        state = node_info.node.__class__.energy_functional(state, node_info)

        total_energy = jnp.sum(state.energy)
        return total_energy, state

    @staticmethod
    def _layernorm(
        x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5
    ) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / jnp.sqrt(variance + eps)
        return gamma * x_normalized + beta

    @staticmethod
    def _mha(
        x: jnp.ndarray,
        mask: jnp.ndarray,
        num_heads: int,
        W_q: jnp.ndarray,
        W_k: jnp.ndarray,
        W_v: jnp.ndarray,
        W_o: jnp.ndarray,
        b_q: jnp.ndarray,
        b_k: jnp.ndarray,
        b_v: jnp.ndarray,
        b_o: jnp.ndarray,
        activation_fn,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
    ) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_dim = embed_dim // num_heads

        Q = jnp.matmul(x, W_q) + b_q
        K = jnp.matmul(x, W_k) + b_k
        V = jnp.matmul(x, W_v) + b_v

        Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

        if use_rope:
            cos, sin = precompute_freqs_cis(head_dim, seq_len, theta=rope_theta)
            Q = apply_rotary_emb(Q, cos, sin)
            K = apply_rotary_emb(K, cos, sin)

        scale = jnp.sqrt(head_dim)
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            scores = jnp.where(mask == 0, -1e9, scores)

        attn_matrix = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_matrix, V)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, embed_dim
        )

        pre_activation = jnp.matmul(attn_output, W_o) + b_o
        projection = activation_fn(pre_activation)

        substructure = {
            "attn_matrix": attn_matrix,
            "Q": Q,
            "K": K,
            "V": V,
        }

        return projection, substructure
