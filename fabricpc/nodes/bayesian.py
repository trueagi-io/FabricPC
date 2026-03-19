"""
Partially Bayesian transformer nodes with deterministic moment propagation.

These nodes replace the corresponding transformer_v2 nodes with mean-field
Gaussian weight posteriors. Uncertainty is propagated analytically (no sampling):

  - Linear layers: exact moment propagation
      mu_out  = mu_in @ W_mu
      var_out = var_in @ W_mu^2  +  mu_in^2 @ sigma_W^2

  - GELU: first-order linearisation
      mu_out  = GELU(mu_h)
      var_out = GELU'(mu_h)^2 * var_h

  - Attention softmax: variance-as-temperature
      effective_scale = sqrt(head_dim) * (1 + mean(var_Q + var_K))
      A = softmax(Q_mu @ K_mu^T / effective_scale)

  - Residual connections: conservative (drop covariance), then smooth-normalise
      var_z = (var_x + var_branch) / (1 + var_x + var_branch)

Each Bayesian node:
  - Stores W_mu and W_rho (sigma = softplus(W_rho)) for every weight matrix.
  - Adds KL(q(W) || N(0,1)) to its energy output (scaled by kl_beta).
  - Writes propagated variance to state.z_var for downstream nodes.
  - Uses HeteroscedasticGaussianEnergy: precision-weighted PC errors.

Embedding and VocabProjection nodes remain deterministic (imported from
transformer_v2).
"""

from typing import Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf as norm_cdf, pdf as norm_pdf

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.initializers import (
    InitializerBase,
    initialize,
    XavierInitializer,
    KaimingInitializer,
    NormalInitializer,
)
from fabricpc.core.positional import precompute_freqs_cis, apply_rotary_emb
from fabricpc.core.activations import GeluActivation, IdentityActivation, SoftmaxActivation
from fabricpc.core.energy import HeteroscedasticGaussianEnergy, KLDivergenceEnergy
from fabricpc.utils.helpers import layernorm
from fabricpc.core.inference import InferenceBase

# Builder imports
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.transformer_v2 import EmbeddingNode, VocabProjectionNode
from fabricpc.builder import Edge, TaskMap, graph

# ==============================================================================
# HELPERS
# ==============================================================================

# W_rho initialisation: softplus_inverse(0.01) ≈ -4.6
# => sigma = softplus(-4.6) ≈ 0.01  (tiny initial uncertainty)
_RHO_INIT = -4.6


def kl_gaussian(W_mu: jnp.ndarray, W_rho: jnp.ndarray) -> jnp.ndarray:
    """
    KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Summed over all elements of the weight tensor.
    """
    sigma = jax.nn.softplus(W_rho) + 1e-6
    return -0.5 * jnp.sum(1.0 + jnp.log(sigma ** 2) - W_mu ** 2 - sigma ** 2)


def linear_moment(
    mu_in: jnp.ndarray,
    var_in: jnp.ndarray,
    W_mu: jnp.ndarray,
    W_rho: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Exact moment propagation through y = x @ W, W ~ N(W_mu, softplus(W_rho)^2).

    Drops the second-order cross term var_in @ sigma_W^2 (small when both are small).

    Works for any leading batch/sequence dimensions:
        mu_in:  (..., in_features)
        W_mu:   (in_features, out_features)
        Returns (mu_out, var_out) both shaped (..., out_features)
    """
    sigma_W = jax.nn.softplus(W_rho) + 1e-6
    mu_out = mu_in @ W_mu
    var_out = (var_in @ (W_mu ** 2)) + (mu_in ** 2 @ (sigma_W ** 2))
    return mu_out, var_out


def smooth_normalize(var: jnp.ndarray) -> jnp.ndarray:
    """
    Map variance from [0, inf) to [0, 1) smoothly.
    f(v) = v / (1 + v)  — monotone, differentiable, no kinks.
    """
    return var / (1.0 + var)


def _hetero_energy_and_grad(
    z_latent: jnp.ndarray,
    z_mu: jnp.ndarray,
    z_var: jnp.ndarray,
    sigma_noise: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Inline heteroscedastic Gaussian energy and its gradient w.r.t. z_latent.

    E   = 0.5 * (z - mu)^2 / (sigma_noise^2 + var) + 0.5 * log(sigma_noise^2 + var)
    dE/dz = (z - mu) / (sigma_noise^2 + var)
    """
    total_var = sigma_noise ** 2 + z_var
    diff = z_latent - z_mu
    axes = tuple(range(1, len(diff.shape)))
    energy = 0.5 * jnp.sum(diff ** 2 / total_var, axis=axes) + \
             0.5 * jnp.sum(jnp.log(total_var), axis=axes)
    grad = diff / total_var
    return energy, grad


def _update_state_hetero(
    state: NodeState,
    z_mu: jnp.ndarray,
    z_var: jnp.ndarray,
    sigma_noise: float,
) -> NodeState:
    """Compute heteroscedastic energy, update state fields."""
    error = state.z_latent - z_mu
    energy, grad = _hetero_energy_and_grad(state.z_latent, z_mu, z_var, sigma_noise)
    latent_grad = state.latent_grad + grad
    return state._replace(
        z_mu=z_mu,
        z_var=z_var,
        error=error,
        energy=energy,
        latent_grad=latent_grad,
    )


# ==============================================================================
# BAYESIAN MHA + RESIDUAL NODE
# ==============================================================================


class BayesianMhaResidualNode(NodeBase):
    """
    Multi-head attention with residual connection and Bayesian weight posteriors.

    Replaces MhaResidualNode with:
      - Mean-field Gaussian weights: W_{q,k,v,o} ~ N(W_mu, softplus(W_rho)^2)
      - Moment propagation through Q, K, V projections
      - Uncertainty-scaled attention temperature
      - KL divergence added to node energy
      - Propagated variance written to state.z_var

    LayerNorm parameters (gamma, beta) remain deterministic.
    """

    DEFAULT_ENERGY = HeteroscedasticGaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        num_heads,
        use_rope=True,
        rope_theta=10000.0,
        is_causal=True,
        kl_beta=1e-4,
        sigma_noise=1.0,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            is_causal=is_causal,
            kl_beta=kl_beta,
            sigma_noise=sigma_noise,
            weight_init=weight_init or XavierInitializer(),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or HeteroscedasticGaussianEnergy(sigma_noise=sigma_noise),
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False), "mask": SlotSpec("mask", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        dim = config["embed_dim"]
        keys = jax.random.split(key, 8)

        def init_mu(k):
            return initialize(k, (dim, dim), weight_init)

        weights = {
            "ln_gamma": jnp.ones((dim,)),
            # Q
            "W_q_mu":  init_mu(keys[0]),
            "W_q_rho": jnp.full((dim, dim), _RHO_INIT),
            # K
            "W_k_mu":  init_mu(keys[1]),
            "W_k_rho": jnp.full((dim, dim), _RHO_INIT),
            # V
            "W_v_mu":  init_mu(keys[2]),
            "W_v_rho": jnp.full((dim, dim), _RHO_INIT),
            # O
            "W_o_mu":  init_mu(keys[3]),
            "W_o_rho": jnp.full((dim, dim), _RHO_INIT),
        }
        biases = {
            "ln_beta": jnp.zeros((dim,)),
            "b_q": jnp.zeros((dim,)),
            "b_k": jnp.zeros((dim,)),
            "b_v": jnp.zeros((dim,)),
            "b_o": jnp.zeros((dim,)),
        }
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        """
        Mean-weight forward pass used by the learning phase (forward_learning).
        Reads state.z_var (set by forward_inference) for heteroscedastic energy.
        Adds KL term to total energy.
        """
        cfg = node_info.node_config
        x = inputs[next(k for k in inputs if k.endswith(":in"))]
        mask_key = next((k for k in inputs if k.endswith(":mask")), None)
        external_mask = inputs[mask_key] if mask_key else None

        B, L, D = x.shape
        num_heads = cfg["num_heads"]
        head_dim = D // num_heads

        # LayerNorm (deterministic)
        x_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])

        # Projections with mean weights
        Q = (x_norm @ params.weights["W_q_mu"] + params.biases["b_q"]).reshape(B, L, num_heads, head_dim)
        K = (x_norm @ params.weights["W_k_mu"] + params.biases["b_k"]).reshape(B, L, num_heads, head_dim)
        V = (x_norm @ params.weights["W_v_mu"] + params.biases["b_v"]).reshape(B, L, num_heads, head_dim)

        if cfg.get("use_rope"):
            freqs_cis = precompute_freqs_cis(head_dim, L, theta=cfg.get("rope_theta", 10000.0))
            Q, K = apply_rotary_emb(Q, K, freqs_cis)

        Q, K, V = Q.transpose(0, 2, 1, 3), K.transpose(0, 2, 1, 3), V.transpose(0, 2, 1, 3)
        scores = jnp.matmul(Q, K.swapaxes(-1, -2)) / jnp.sqrt(head_dim)

        if cfg.get("is_causal", True):
            causal_mask = jnp.tril(jnp.ones((L, L)))
            scores = jnp.where(causal_mask == 0, -1e9, scores)
        if external_mask is not None:
            scores = jnp.where(external_mask == 0, -1e9, scores)

        attn = jax.nn.softmax(scores, axis=-1)
        mha = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, L, D)
        mha = mha @ params.weights["W_o_mu"] + params.biases["b_o"]

        z_mu = x + mha  # residual

        # Use z_var stored from previous forward_inference (zeros on first step)
        z_var = state.z_var
        sigma_noise = cfg.get("sigma_noise", 1.0)
        new_state = _update_state_hetero(state, z_mu, z_var, sigma_noise)

        # KL regularisation
        kl_beta = cfg.get("kl_beta", 1e-4)
        kl = (
            kl_gaussian(params.weights["W_q_mu"], params.weights["W_q_rho"])
            + kl_gaussian(params.weights["W_k_mu"], params.weights["W_k_rho"])
            + kl_gaussian(params.weights["W_v_mu"], params.weights["W_v_rho"])
            + kl_gaussian(params.weights["W_o_mu"], params.weights["W_o_rho"])
        )
        total_energy = jnp.sum(new_state.energy) + kl_beta * kl
        return total_energy, new_state

    @staticmethod
    def forward_inference(params, inputs, state, node_info, is_clamped, var_inputs=None):
        """
        Moment-propagated forward pass for the inference phase.
        Computes z_mu and z_var via full moment propagation, writes z_var to state,
        then uses JAX autodiff through forward() for input gradients.
        """
        cfg = node_info.node_config
        node_class = node_info.node_class

        in_key = next(k for k in inputs if k.endswith(":in"))
        x = inputs[in_key]
        var_in = var_inputs.get(in_key, jnp.zeros_like(x)) if var_inputs else jnp.zeros_like(x)

        B, L, D = x.shape
        num_heads = cfg["num_heads"]
        head_dim = D // num_heads

        # --- LayerNorm moment propagation ---
        mu_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])
        # var_norm = gamma^2 * var_in / (spatial_var_of_x + eps)
        # LN normalises along feature dim, so variance is attenuated by 1/Var(x_features)
        spatial_var = jnp.var(x, axis=-1, keepdims=True) + 1e-5  # (B, L, 1)
        var_norm = params.weights["ln_gamma"] ** 2 * var_in / spatial_var  # (B, L, D)

        # --- Q, K, V moment propagation ---
        Q_mu, var_Q = linear_moment(mu_norm, var_norm, params.weights["W_q_mu"], params.weights["W_q_rho"])
        K_mu, var_K = linear_moment(mu_norm, var_norm, params.weights["W_k_mu"], params.weights["W_k_rho"])
        V_mu, var_V = linear_moment(mu_norm, var_norm, params.weights["W_v_mu"], params.weights["W_v_rho"])

        Q_mu = (Q_mu + params.biases["b_q"]).reshape(B, L, num_heads, head_dim)
        K_mu = (K_mu + params.biases["b_k"]).reshape(B, L, num_heads, head_dim)
        V_mu = (V_mu + params.biases["b_v"]).reshape(B, L, num_heads, head_dim)
        # Variance doesn't shift with bias — bias is deterministic
        var_Q = var_Q.reshape(B, L, num_heads, head_dim)
        var_K = var_K.reshape(B, L, num_heads, head_dim)
        var_V = var_V.reshape(B, L, num_heads, head_dim)

        if cfg.get("use_rope"):
            freqs_cis = precompute_freqs_cis(head_dim, L, theta=cfg.get("rope_theta", 10000.0))
            Q_mu, K_mu = apply_rotary_emb(Q_mu, K_mu, freqs_cis)

        Q_mu = Q_mu.transpose(0, 2, 1, 3)   # (B, heads, L, head_dim)
        K_mu = K_mu.transpose(0, 2, 1, 3)
        V_mu = V_mu.transpose(0, 2, 1, 3)
        var_Q = var_Q.transpose(0, 2, 1, 3)
        var_K = var_K.transpose(0, 2, 1, 3)
        var_V = var_V.transpose(0, 2, 1, 3)

        # --- Uncertainty-scaled attention temperature ---
        scores = jnp.matmul(Q_mu, K_mu.swapaxes(-1, -2))       # (B, heads, L, L)
        uncertainty = jnp.mean(var_Q + var_K)
        effective_scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32)) * (1.0 + uncertainty)
        scores = scores / effective_scale

        if cfg.get("is_causal", True):
            causal_mask = jnp.tril(jnp.ones((L, L)))
            scores = jnp.where(causal_mask == 0, -1e9, scores)
        mask_key = next((k for k in inputs if k.endswith(":mask")), None)
        if mask_key and mask_key in inputs:
            scores = jnp.where(inputs[mask_key] == 0, -1e9, scores)

        A = jax.nn.softmax(scores, axis=-1)  # (B, heads, L, L)

        # --- MHA output moment propagation ---
        mha_mu = jnp.matmul(A, V_mu)                            # (B, heads, L, head_dim)
        var_mha = jnp.matmul(A ** 2, var_V)                     # treat A as deterministic

        mha_mu = mha_mu.transpose(0, 2, 1, 3).reshape(B, L, D)
        var_mha = var_mha.transpose(0, 2, 1, 3).reshape(B, L, D)

        # Output projection moment propagation
        mha_mu_proj, var_mha_proj = linear_moment(
            mha_mu, var_mha, params.weights["W_o_mu"], params.weights["W_o_rho"]
        )
        mha_mu_proj = mha_mu_proj + params.biases["b_o"]

        # --- Residual + smooth normalisation ---
        z_var_raw = var_in + var_mha_proj
        z_var = smooth_normalize(z_var_raw)

        # Write z_var into state so forward() picks it up as a constant
        state_with_var = state._replace(z_var=z_var)

        # --- Autodiff through forward() for input gradients ---
        if node_info.in_degree == 0:
            z_mu = state_with_var.z_latent
            new_state = _update_state_hetero(
                state_with_var, z_mu, z_var, cfg.get("sigma_noise", 1.0)
            )
            new_state = new_state._replace(error=jnp.zeros_like(new_state.error))
            input_grads = {k: jnp.zeros_like(v) for k, v in inputs.items()}

        elif node_info.out_degree == 0 and not is_clamped:
            _, new_state = node_class.forward(params, inputs, state_with_var, node_info)
            new_state = new_state._replace(
                z_latent=new_state.z_mu,
                error=jnp.zeros_like(new_state.error),
                energy=jnp.zeros_like(new_state.energy),
                latent_grad=jnp.zeros_like(new_state.latent_grad),
            )
            input_grads = {k: jnp.zeros_like(v) for k, v in inputs.items()}

        else:
            (_, new_state), input_grads = jax.value_and_grad(
                node_class.forward, argnums=1, has_aux=True
            )(params, inputs, state_with_var, node_info)

        return new_state, input_grads


# ==============================================================================
# BAYESIAN LN + MLP1 NODE
# ==============================================================================


class BayesianLnMlp1Node(NodeBase):
    """
    LayerNorm + first FFN linear + GELU with Bayesian weight posterior on W_ff1.

    Replaces LnMlp1Node with moment propagation through:
      LN → linear(W_ff1_mu, W_ff1_rho) → GELU
    """

    DEFAULT_ENERGY = HeteroscedasticGaussianEnergy
    DEFAULT_ACTIVATION = GeluActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        ff_dim,
        kl_beta=1e-4,
        sigma_noise=1.0,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            kl_beta=kl_beta,
            sigma_noise=sigma_noise,
            weight_init=weight_init or KaimingInitializer(),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or HeteroscedasticGaussianEnergy(sigma_noise=sigma_noise),
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        embed_dim, ff_dim = config["embed_dim"], config["ff_dim"]
        key_mu, = jax.random.split(key, 1)

        weights = {
            "ln_gamma":  jnp.ones((embed_dim,)),
            "W_ff1_mu":  initialize(key_mu, (embed_dim, ff_dim), weight_init),
            "W_ff1_rho": jnp.full((embed_dim, ff_dim), _RHO_INIT),
        }
        biases = {
            "ln_beta": jnp.zeros((embed_dim,)),
            "b_ff1":   jnp.zeros((ff_dim,)),
        }
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        cfg = node_info.node_config
        x = inputs[list(inputs.keys())[0]]

        # LN
        x_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])

        # Linear with mean weights + GELU
        pre_act = x_norm @ params.weights["W_ff1_mu"] + params.biases["b_ff1"]
        z_mu = jax.nn.gelu(pre_act)

        z_var = state.z_var  # set by forward_inference
        sigma_noise = cfg.get("sigma_noise", 1.0)
        new_state = _update_state_hetero(state, z_mu, z_var, sigma_noise)
        new_state = new_state._replace(pre_activation=pre_act)

        kl_beta = cfg.get("kl_beta", 1e-4)
        kl = kl_gaussian(params.weights["W_ff1_mu"], params.weights["W_ff1_rho"])
        total_energy = jnp.sum(new_state.energy) + kl_beta * kl
        return total_energy, new_state

    @staticmethod
    def forward_inference(params, inputs, state, node_info, is_clamped, var_inputs=None):
        cfg = node_info.node_config
        node_class = node_info.node_class

        in_key = list(inputs.keys())[0]
        x = inputs[in_key]
        var_in = var_inputs.get(in_key, jnp.zeros_like(x)) if var_inputs else jnp.zeros_like(x)

        # --- LN moment propagation ---
        mu_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])
        spatial_var = jnp.var(x, axis=-1, keepdims=True) + 1e-5
        var_norm = params.weights["ln_gamma"] ** 2 * var_in / spatial_var

        # --- Linear moment propagation ---
        mu_h, var_h = linear_moment(mu_norm, var_norm, params.weights["W_ff1_mu"], params.weights["W_ff1_rho"])
        mu_h = mu_h + params.biases["b_ff1"]

        # --- GELU moment propagation (first-order) ---
        gelu_prime = norm_cdf(mu_h) + mu_h * norm_pdf(mu_h)
        var_out = (gelu_prime ** 2) * var_h
        var_out = smooth_normalize(var_out)

        state_with_var = state._replace(z_var=var_out)

        if node_info.in_degree == 0:
            z_mu = jax.nn.gelu(mu_h)
            new_state = _update_state_hetero(state_with_var, z_mu, var_out, cfg.get("sigma_noise", 1.0))
            new_state = new_state._replace(error=jnp.zeros_like(new_state.error))
            input_grads = {k: jnp.zeros_like(v) for k, v in inputs.items()}
        elif node_info.out_degree == 0 and not is_clamped:
            _, new_state = node_class.forward(params, inputs, state_with_var, node_info)
            new_state = new_state._replace(
                z_latent=new_state.z_mu,
                error=jnp.zeros_like(new_state.error),
                energy=jnp.zeros_like(new_state.energy),
                latent_grad=jnp.zeros_like(new_state.latent_grad),
            )
            input_grads = {k: jnp.zeros_like(v) for k, v in inputs.items()}
        else:
            (_, new_state), input_grads = jax.value_and_grad(
                node_class.forward, argnums=1, has_aux=True
            )(params, inputs, state_with_var, node_info)

        return new_state, input_grads


# ==============================================================================
# BAYESIAN MLP2 + RESIDUAL NODE
# ==============================================================================


class BayesianMlp2ResidualNode(NodeBase):
    """
    Second FFN linear + residual connection with Bayesian weight posterior on W_ff2.

    Slots:
      "in"       — output of BayesianLnMlp1Node (carries z_var via moment prop)
      "residual" — output of BayesianMhaResidualNode (carries z_var)
    """

    DEFAULT_ENERGY = HeteroscedasticGaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        ff_dim,
        kl_beta=1e-4,
        sigma_noise=1.0,
        weight_init=None,
        latent_init=None,
        energy=None,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            kl_beta=kl_beta,
            sigma_noise=sigma_noise,
            weight_init=weight_init or XavierInitializer(),
            latent_init=latent_init or NormalInitializer(),
            energy=energy or HeteroscedasticGaussianEnergy(sigma_noise=sigma_noise),
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {
            "in":       SlotSpec("in", False),
            "residual": SlotSpec("residual", False),
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        embed_dim, ff_dim = config["embed_dim"], config["ff_dim"]

        weights = {
            "W_ff2_mu":  initialize(key, (ff_dim, embed_dim), weight_init),
            "W_ff2_rho": jnp.full((ff_dim, embed_dim), _RHO_INIT),
        }
        biases = {"b_ff2": jnp.zeros((embed_dim,))}
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        cfg = node_info.node_config
        mlp1_in  = next(v for k, v in inputs.items() if k.endswith(":in"))
        res_in   = next(v for k, v in inputs.items() if k.endswith(":residual"))

        ff2 = mlp1_in @ params.weights["W_ff2_mu"] + params.biases["b_ff2"]
        z_mu = res_in + ff2

        z_var = state.z_var  # set by forward_inference
        sigma_noise = cfg.get("sigma_noise", 1.0)
        new_state = _update_state_hetero(state, z_mu, z_var, sigma_noise)

        kl_beta = cfg.get("kl_beta", 1e-4)
        kl = kl_gaussian(params.weights["W_ff2_mu"], params.weights["W_ff2_rho"])
        total_energy = jnp.sum(new_state.energy) + kl_beta * kl
        return total_energy, new_state

    @staticmethod
    def forward_inference(params, inputs, state, node_info, is_clamped, var_inputs=None):
        cfg = node_info.node_config
        node_class = node_info.node_class

        in_key  = next(k for k in inputs if k.endswith(":in"))
        res_key = next(k for k in inputs if k.endswith(":residual"))

        mu_mlp1  = inputs[in_key]
        mu_res   = inputs[res_key]
        var_mlp1 = var_inputs.get(in_key,  jnp.zeros_like(mu_mlp1)) if var_inputs else jnp.zeros_like(mu_mlp1)
        var_res  = var_inputs.get(res_key, jnp.zeros_like(mu_res))   if var_inputs else jnp.zeros_like(mu_res)

        # --- Linear moment propagation ---
        mu_ff2, var_ff2 = linear_moment(mu_mlp1, var_mlp1, params.weights["W_ff2_mu"], params.weights["W_ff2_rho"])
        mu_ff2 = mu_ff2 + params.biases["b_ff2"]

        # --- Residual: drop covariance (conservative), then smooth-normalise ---
        z_var_raw = var_res + var_ff2
        z_var = smooth_normalize(z_var_raw)

        state_with_var = state._replace(z_var=z_var)

        if node_info.in_degree == 0:
            z_mu = mu_res + mu_ff2
            new_state = _update_state_hetero(state_with_var, z_mu, z_var, cfg.get("sigma_noise", 1.0))
            new_state = new_state._replace(error=jnp.zeros_like(new_state.error))
            input_grads = {k: jnp.zeros_like(v) for k, v in inputs.items()}
        elif node_info.out_degree == 0 and not is_clamped:
            _, new_state = node_class.forward(params, inputs, state_with_var, node_info)
            new_state = new_state._replace(
                z_latent=new_state.z_mu,
                error=jnp.zeros_like(new_state.error),
                energy=jnp.zeros_like(new_state.energy),
                latent_grad=jnp.zeros_like(new_state.latent_grad),
            )
            input_grads = {k: jnp.zeros_like(v) for k, v in inputs.items()}
        else:
            (_, new_state), input_grads = jax.value_and_grad(
                node_class.forward, argnums=1, has_aux=True
            )(params, inputs, state_with_var, node_info)

        return new_state, input_grads


# ==============================================================================
# MODEL BUILDER
# ==============================================================================


def create_partial_bayesian_transformer(
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    seq_len: int,
    vocab_size: int,
    inference: InferenceBase,
    kl_beta: float = 1e-4,
    w_rho_init: float = _RHO_INIT,
    sigma_noise: float = 1.0,
):
    """
    Build a partially Bayesian PC transformer.

    Deterministic nodes : EmbeddingNode, VocabProjectionNode
    Bayesian nodes      : BayesianMhaResidualNode, BayesianLnMlp1Node,
                          BayesianMlp2ResidualNode

    Graph structure per layer:
        embed → mha:in
        mha   → mlp1:in
        mlp1  → mlp2:in
        mha   → mlp2:residual   (skip for residual addition)
        mlp2  → next_mha:in  (or → logits for last layer)

    Args:
        depth:       Number of transformer layers.
        embed_dim:   Model dimension.
        num_heads:   Attention heads.
        mlp_dim:     FFN hidden dimension.
        seq_len:     Sequence length.
        vocab_size:  Vocabulary size.
        inference:   InferenceBase instance (e.g. InferenceSGDNormClip).
        kl_beta:     KL weight. Rule of thumb: batch_size / dataset_size.
                     Use KL annealing in training — start near 0, ramp up.
        w_rho_init:  Initial W_rho value. softplus(w_rho_init) = initial sigma.
                     Default -4.6 → sigma ≈ 0.01.
        sigma_noise: Observation noise std for heteroscedastic energy.
    """
    nodes = []
    edges = []

    # Input node (deterministic — just passes token IDs through)
    input_node = Linear(shape=(seq_len,), activation=IdentityActivation(), name="input_ids")
    nodes.append(input_node)

    # Embedding (deterministic)
    embed_node = EmbeddingNode(
        name="embed",
        shape=(seq_len, embed_dim),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    nodes.append(embed_node)
    edges.append(Edge(source=input_node, target=embed_node.slot("in")))

    previous = embed_node

    for i in range(depth):
        mha = BayesianMhaResidualNode(
            name=f"L{i}_mha",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            num_heads=num_heads,
            kl_beta=kl_beta,
            sigma_noise=sigma_noise,
        )
        nodes.append(mha)
        edges.append(Edge(source=previous, target=mha.slot("in")))

        mlp1 = BayesianLnMlp1Node(
            name=f"L{i}_mlp1",
            shape=(seq_len, mlp_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            kl_beta=kl_beta,
            sigma_noise=sigma_noise,
        )
        nodes.append(mlp1)
        edges.append(Edge(source=mha, target=mlp1.slot("in")))

        mlp2 = BayesianMlp2ResidualNode(
            name=f"L{i}_mlp2",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            kl_beta=kl_beta,
            sigma_noise=sigma_noise,
        )
        nodes.append(mlp2)
        edges.append(Edge(source=mlp1, target=mlp2.slot("in")))
        edges.append(Edge(source=mha,  target=mlp2.slot("residual")))

        previous = mlp2

    # Output projection (deterministic)
    logits = VocabProjectionNode(
        name="logits",
        shape=(seq_len, vocab_size),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    nodes.append(logits)
    edges.append(Edge(source=previous, target=logits.slot("in")))

    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=logits),
        inference=inference,
    )
