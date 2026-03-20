
"""
Bayesian Predictive Coding Transformer using fabricPC.

A minimal uncertainty-aware sequence model where:
- Representations are probabilistic (mean + per-token variance)
- Computation uses predictive coding (iterative inference with error nodes)
- Attention scores are modulated by uncertainty: score / sqrt(var_q + var_k + eps)
- MLP outputs are gated by certainty: delta * sigmoid(1 / (var + eps))
- Output layer is deterministic

Variance is tracked externally and injected via clamped source nodes.
Learning uses fabricPC's native local weight updates (not backpropagation).

Note: This is a heuristic uncertainty-aware system, not formally Bayesian.
Variance tracks prediction error magnitude as an interpretable confidence proxy.
"""

import jax
import jax.numpy as jnp
import optax
from typing import cast

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.transformer_v2 import EmbeddingNode, VocabProjectionNode
from fabricpc.core.types import NodeParams, GraphParams
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import initialize, NormalInitializer, XavierInitializer
from fabricpc.core.inference import InferenceBase, InferenceSGD
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph.graph_net import initialize_params, compute_local_weight_gradients
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.utils.helpers import layernorm, update_node_in_state


# ============================================================================
# Helper: Rotary Position Embeddings (real arithmetic, no complex numbers)
# ============================================================================


def _apply_rope(x, seq_len, head_dim, theta=10000.0):
    """
    Apply rotary position embeddings.

    Args:
        x: (batch, num_heads, seq_len, head_dim)
        seq_len: sequence length
        head_dim: dimension per head (must be even)
        theta: RoPE base frequency

    Returns:
        Rotated tensor, same shape as input.
    """
    dim_indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    freqs = 1.0 / (theta ** (dim_indices / head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(positions, freqs)  # (seq_len, head_dim/2)
    cos = jnp.cos(angles)[None, None, :, :]  # (1, 1, seq_len, head_dim/2)
    sin = jnp.sin(angles)[None, None, :, :]

    x_even = x[..., 0::2]  # (batch, heads, seq, head_dim/2)
    x_odd = x[..., 1::2]
    x_rot = jnp.stack(
        [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], axis=-1
    ).reshape(x.shape)
    return x_rot


# ============================================================================
# Section 1: Custom Node Types
# ============================================================================


class BayesianAttentionNode(NodeBase):
    """
    Uncertainty-modulated multi-head attention with residual connection.

    Attention scores are divided by sqrt(var_q + var_k + eps), dampening
    interactions between uncertain tokens. Uses pre-norm architecture
    with RoPE positional encoding and causal masking.
    """

    DEFAULT_ACTIVATION = IdentityActivation
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_LATENT_INIT = NormalInitializer

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        num_heads,
        is_causal=True,
        rope_theta=10000.0,
        weight_init=None,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=weight_init or XavierInitializer(),
            embed_dim=embed_dim,
            num_heads=num_heads,
            is_causal=is_causal,
            rope_theta=rope_theta,
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec("in", False),  # Token representations
            "var": SlotSpec("var", False),  # Per-token variance
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        dim = config["embed_dim"]
        weight_init = config.get("weight_init", weight_init)
        keys = jax.random.split(key, 5)

        weights = {
            "ln_gamma": jnp.ones((dim,)),
            "W_q": initialize(keys[0], (dim, dim), weight_init),
            "W_k": initialize(keys[1], (dim, dim), weight_init),
            "W_v": initialize(keys[2], (dim, dim), weight_init),
            "W_o": initialize(keys[3], (dim, dim), weight_init),
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
        cfg = node_info.node_config
        num_heads = cfg["num_heads"]
        is_causal = cfg.get("is_causal", True)
        rope_theta = cfg.get("rope_theta", 10000.0)
        eps = 1e-6

        # Get inputs by slot suffix
        x = inputs[next(k for k in inputs if k.endswith(":in"))]
        var = inputs[next(k for k in inputs if k.endswith(":var"))]

        B, L, D = x.shape
        head_dim = D // num_heads

        # Pre-norm
        x_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])

        # Q, K, V projections -> (batch, seq, embed_dim)
        Q = jnp.dot(x_norm, params.weights["W_q"]) + params.biases["b_q"]
        K = jnp.dot(x_norm, params.weights["W_k"]) + params.biases["b_k"]
        V = jnp.dot(x_norm, params.weights["W_v"]) + params.biases["b_v"]

        # Reshape to (batch, heads, seq, head_dim)
        Q = Q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply rotary position embeddings
        Q = _apply_rope(Q, L, head_dim, rope_theta)
        K = _apply_rope(K, L, head_dim, rope_theta)

        # Attention scores: (batch, heads, seq_q, seq_k)
        scores = jnp.matmul(Q, K.swapaxes(-1, -2)) / jnp.sqrt(head_dim)

        # --- Uncertainty modulation ---
        # var: (batch, seq, 1) -> per-token scalar
        var_flat = var[..., 0]  # (batch, seq)
        var_q = var_flat[:, None, :, None]  # (batch, 1, seq_q, 1)
        var_k = var_flat[:, None, None, :]  # (batch, 1, 1, seq_k)
        combined_var = var_q + var_k  # (batch, 1, seq_q, seq_k) broadcasts over heads
        scores = scores / jnp.sqrt(combined_var + eps)

        # Causal mask
        if is_causal:
            causal_mask = jnp.tril(jnp.ones((L, L)))
            scores = jnp.where(causal_mask == 0, -1e9, scores)

        # Softmax + weighted sum
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn, V)  # (batch, heads, seq, head_dim)

        # Reshape back and output projection
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        mha = jnp.dot(out, params.weights["W_o"]) + params.biases["b_o"]

        # Residual connection
        z_mu = x + mha

        # Error and energy
        error = state.z_latent - z_mu
        state = state._replace(
            z_mu=z_mu,
            error=error,
            pre_activation=z_mu,
        )
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class BayesianMLPNode(NodeBase):
    """
    Uncertainty-gated feedforward network with residual connection.

    MLP output is scaled by a certainty gate: sigmoid(1 / (var + eps)).
    Uncertain tokens receive dampened updates, keeping them closer to
    the residual. Uses pre-norm with GELU activation.
    """

    DEFAULT_ACTIVATION = IdentityActivation
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_LATENT_INIT = NormalInitializer

    def __init__(self, shape, name, embed_dim, ff_dim, weight_init=None, **kwargs):
        super().__init__(
            shape=shape,
            name=name,
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=weight_init or XavierInitializer(),
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec("in", False),  # Input representations
            "var": SlotSpec("var", False),  # Per-token variance
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        embed_dim = config["embed_dim"]
        ff_dim = config["ff_dim"]
        weight_init = config.get("weight_init", weight_init)
        keys = jax.random.split(key, 3)

        weights = {
            "ln_gamma": jnp.ones((embed_dim,)),
            "W_ff1": initialize(keys[0], (embed_dim, ff_dim), weight_init),
            "W_ff2": initialize(keys[1], (ff_dim, embed_dim), weight_init),
        }
        biases = {
            "ln_beta": jnp.zeros((embed_dim,)),
            "b_ff1": jnp.zeros((ff_dim,)),
            "b_ff2": jnp.zeros((embed_dim,)),
        }
        return NodeParams(weights, biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        eps = 1e-6

        # Get inputs
        x = inputs[next(k for k in inputs if k.endswith(":in"))]
        var = inputs[next(k for k in inputs if k.endswith(":var"))]

        # Pre-norm
        x_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])

        # Two-layer FFN with GELU
        h = jnp.dot(x_norm, params.weights["W_ff1"]) + params.biases["b_ff1"]
        h = jax.nn.gelu(h)
        delta = jnp.dot(h, params.weights["W_ff2"]) + params.biases["b_ff2"]

        # Certainty gate: dampen updates for uncertain tokens
        # var: (batch, seq, 1), certainty broadcasts over embed_dim
        certainty = jax.nn.sigmoid(1.0 / (var + eps))
        delta = delta * certainty

        # Residual
        z_mu = x + delta

        # Error and energy
        error = state.z_latent - z_mu
        state = state._replace(
            z_mu=z_mu,
            error=error,
            pre_activation=z_mu,
        )
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


# ============================================================================
# Section 2: Model Builder
# ============================================================================


def create_bayesian_pc_transformer(
    depth, embed_dim, num_heads, ff_dim, seq_len, vocab_size, inference=None
):
    """
    Build a Bayesian PC Transformer graph.

    Architecture per layer:
        var_i (clamped) -> BayesianAttentionNode -> BayesianMLPNode
                       |-> BayesianMLPNode (var input)

    Args:
        depth: Number of transformer layers
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: FFN hidden dimension
        seq_len: Maximum sequence length
        vocab_size: Vocabulary size

    Returns:
        structure: GraphStructure
        layer_info: List of dicts with var_node/attn_node/error_node names per layer
    """
    nodes = []
    edges = []

    # Input node (clamped to integer token indices)
    input_node = Linear(
        shape=(seq_len,), activation=IdentityActivation(), name="input_ids"
    )
    nodes.append(input_node)

    # Embedding lookup
    embed_node = EmbeddingNode(
        name="embed",
        shape=(seq_len, embed_dim),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    nodes.append(embed_node)
    edges.append(Edge(source=input_node, target=embed_node.slot("in")))

    layer_info = []
    prev_output = embed_node

    for i in range(depth):
        # Variance source node (no learnable params, clamped externally)
        var_node = IdentityNode(shape=(seq_len, 1), name=f"var_{i}")
        nodes.append(var_node)

        # Bayesian attention block
        attn_node = BayesianAttentionNode(
            shape=(seq_len, embed_dim),
            name=f"bay_attn_{i}",
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        nodes.append(attn_node)
        edges.append(Edge(source=prev_output, target=attn_node.slot("in")))
        edges.append(Edge(source=var_node, target=attn_node.slot("var")))

        # Bayesian MLP block
        mlp_node = BayesianMLPNode(
            shape=(seq_len, embed_dim),
            name=f"bay_mlp_{i}",
            embed_dim=embed_dim,
            ff_dim=ff_dim,
        )
        nodes.append(mlp_node)
        edges.append(Edge(source=attn_node, target=mlp_node.slot("in")))
        edges.append(Edge(source=var_node, target=mlp_node.slot("var")))

        layer_info.append(
            {
                "var_node": var_node.name,
                "attn_node": attn_node.name,
                "error_node": mlp_node.name,  # variance updated from MLP output error
            }
        )
        prev_output = mlp_node

    # Deterministic output projection
    output_node = VocabProjectionNode(
        name="output",
        shape=(seq_len, vocab_size),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    nodes.append(output_node)
    edges.append(Edge(source=prev_output, target=output_node.slot("in")))

    if inference is None:
        inference = InferenceSGD(eta_infer=0.1, infer_steps=20)

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=output_node),
        inference=inference,
    )

    return structure, layer_info


# ============================================================================
# Section 3: Custom Inference Loop with Uncertainty Tracking
# ============================================================================


def run_bayesian_inference(
    params,
    initial_state,
    clamps,
    structure,
    infer_steps,
    eta_infer,
    layer_info,
    variance_config,
):
    """
    Predictive coding inference with external variance tracking.

    Each step:
    1. Inject current variance into graph via clamped source nodes
    2. Run one PC inference step (updates latent states via gradient descent)
    3. Update variance from prediction errors: var = EMA(error^2) + clip

    Args:
        params: GraphParams
        initial_state: GraphState (from initialize_graph_state)
        clamps: Dict of clamped values (input data + output targets)
        structure: GraphStructure
        infer_steps: Number of inference iterations
        eta_infer: Inference learning rate
        layer_info: Layer metadata from create_bayesian_pc_transformer
        variance_config: Dict with keys: decay, min_var, max_var, init_var

    Returns:
        final_state: GraphState after convergence
        variances: Dict mapping var node names to final variance arrays
        energy_history: List of total energy per step
        variance_history: List of dicts (mean variance per layer per step)
    """
    decay = variance_config.get("decay", 0.8)
    min_var = variance_config.get("min_var", 1e-4)
    max_var = variance_config.get("max_var", 10.0)
    init_var = variance_config.get("init_var", 0.1)

    batch_size = initial_state.batch_size
    seq_len = structure.nodes[layer_info[0]["var_node"]].node_info.shape[0]

    # Initialize per-layer variance
    variances = {}
    for info in layer_info:
        variances[info["var_node"]] = jnp.full((batch_size, seq_len, 1), init_var)

    state = initial_state
    energy_history = []
    variance_history = []

    for t in range(infer_steps):
        # 1. Inject current variance into state so forward pass sees it
        for info in layer_info:
            state = update_node_in_state(
                state, info["var_node"], z_latent=variances[info["var_node"]]
            )

        # 2. Build clamps with current variance (so clamping step preserves them)
        step_clamps = dict(clamps)
        for info in layer_info:
            step_clamps[info["var_node"]] = variances[info["var_node"]]

        # 3. Run one PC inference step (with decaying learning rate)
        eta_t = eta_infer * (0.98 ** t)
        step_config = dict(structure.config["inference"].config)
        step_config["eta_infer"] = eta_t
        state = InferenceBase.inference_step(params, state, step_clamps, structure, step_config)

        # 4. Record energy (skip source nodes with in_degree=0)
        total_energy = 0.0
        for n in structure.nodes:
            if structure.nodes[n].node_info.in_degree > 0:
                total_energy += float(jnp.sum(state.nodes[n].energy))
        energy_history.append(total_energy)

        # 5. Update variance: EMA of squared prediction error, clipped
        step_var_means = {}
        for info in layer_info:
            error = state.nodes[info["error_node"]].error  # (batch, seq, embed_dim)
            token_error_sq = jnp.mean(
                error**2, axis=-1, keepdims=True
            )  # (batch, seq, 1)
            old_var = variances[info["var_node"]]
            new_var = decay * old_var + (1.0 - decay) * token_error_sq
            new_var = jnp.clip(new_var, min_var, max_var)
            variances[info["var_node"]] = new_var
            step_var_means[info["var_node"]] = float(jnp.mean(new_var))

        variance_history.append(step_var_means)

    return state, variances, energy_history, variance_history


# ============================================================================
# Section 4: Training Integration
# ============================================================================


def bayesian_get_param_gradient(
    params, batch, structure, rng_key, infer_steps, eta_infer, layer_info, variance_config
):
    """Compute local weight gradients using Bayesian PC inference."""
    batch_size = next(iter(batch.values())).shape[0]
    seq_len = structure.nodes[layer_info[0]["var_node"]].node_info.shape[0]
    init_var = variance_config.get("init_var", 0.1)

    # Map task names to node-level clamps
    clamps = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map:
            clamps[structure.task_map[task_name]] = task_value

    # Include initial variance in clamps (needed by FeedforwardStateInit)
    for info in layer_info:
        clamps[info["var_node"]] = jnp.full((batch_size, seq_len, 1), init_var)

    # Initialize graph state
    init_state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )

    # Run Bayesian inference to convergence
    final_state, final_variances, energy_hist, var_hist = run_bayesian_inference(
        params,
        init_state,
        clamps,
        structure,
        infer_steps,
        eta_infer,
        layer_info,
        variance_config,
    )

    # Total energy at convergence
    energy = 0.0
    for n in structure.nodes:
        if structure.nodes[n].node_info.in_degree > 0:
            energy += float(jnp.sum(final_state.nodes[n].energy))

    # Compute local weight gradients (Hebbian learning)
    grads = compute_local_weight_gradients(params, final_state, structure)

    return grads, energy, final_state, final_variances


def bayesian_train_step(
    params,
    opt_state,
    batch,
    structure,
    optimizer,
    rng_key,
    infer_steps,
    eta_infer,
    layer_info,
    variance_config,
):
    """Single training step: Bayesian PC inference + local weight update."""
    grads, energy, final_state, final_variances = bayesian_get_param_gradient(
        params, batch, structure, rng_key, infer_steps, eta_infer, layer_info, variance_config
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, energy, final_variances


def train_bayesian_pcn(
    params, structure, train_loader, config, rng_key, layer_info, verbose=True
):
    """
    Full training loop for Bayesian PC Transformer.
    Mirrors fabricpc.training.train.train_pcn with Bayesian inference.
    """
    opt_config = config.get("optimizer", {"type": "adam", "lr": 1e-3})
    lr = opt_config.get("lr", 1e-3)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    infer_steps = config.get("infer_steps", 10)
    eta_infer = config.get("eta_infer", 0.1)
    num_epochs = config.get("num_epochs", 5)
    variance_config = config.get("variance", {})

    energy_history = []

    for epoch in range(num_epochs):
        epoch_energies = []
        epoch_rng, rng_key = jax.random.split(rng_key)
        num_batches = len(train_loader)
        batch_keys = jax.random.split(epoch_rng, num_batches)

        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, (list, tuple)):
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            elif isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            params, opt_state, energy, _ = bayesian_train_step(
                params,
                opt_state,
                batch,
                structure,
                optimizer,
                batch_keys[batch_idx],
                infer_steps,
                eta_infer,
                layer_info,
                variance_config,
            )
            bs = next(iter(batch.values())).shape[0]
            epoch_energies.append(energy / bs)

        avg_energy = sum(epoch_energies) / len(epoch_energies) if epoch_energies else 0.0
        energy_history.append(avg_energy)

        if verbose:
            print(f"  Epoch {epoch + 1}/{num_epochs}, avg energy: {avg_energy:.4f}")

    return params, energy_history


# ============================================================================
# Section 5: Visualization
# ============================================================================


def plot_convergence(energy_history, save_path="convergence.png"):
    """Plot prediction error (energy) over inference iterations."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(energy_history) + 1), energy_history, "b-o", markersize=4)
    plt.xlabel("Inference Step")
    plt.ylabel("Total Energy (Prediction Error)")
    plt.title("Predictive Coding Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_uncertainty_heatmap(
    variances, layer_info, noisy_positions=None, sample_idx=0, save_path="uncertainty.png"
):
    """
    Plot per-token uncertainty for each layer.

    Shows a single sample (not batch-averaged) so position-level uncertainty
    is not washed out. Noisy positions are highlighted in red.

    Args:
        variances: dict of variance arrays from inference
        layer_info: layer metadata
        noisy_positions: list of token positions where noise was injected
        sample_idx: which sample in the batch to visualize
        save_path: output file path
    """
    import matplotlib.pyplot as plt

    n_layers = len(layer_info)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    noisy_set = set(noisy_positions) if noisy_positions else set()

    for idx, info in enumerate(layer_info):
        var = variances[info["var_node"]]  # (batch, seq, 1)
        sample_var = var[sample_idx, :, 0]  # (seq,) single sample

        n_tokens = len(sample_var)
        colors = ["tomato" if p in noisy_set else "steelblue" for p in range(n_tokens)]

        axes[idx].bar(range(n_tokens), sample_var, color=colors, alpha=0.8)
        axes[idx].set_xlabel("Token Position")
        axes[idx].set_ylabel("Variance")
        axes[idx].set_title(f"Layer {idx} Per-Token Uncertainty (sample {sample_idx})")
        axes[idx].grid(True, alpha=0.3, axis="y")

        # Legend
        if noisy_set:
            from matplotlib.patches import Patch
            axes[idx].legend(
                handles=[
                    Patch(color="steelblue", label="Clean"),
                    Patch(color="tomato", label="Noisy"),
                ],
                loc="upper right",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# Section 6: Synthetic Data
# ============================================================================


def generate_synthetic_data(rng_key, num_samples, seq_len, vocab_size, noisy_positions=None):
    """
    Generate next-token prediction data.
    Pattern: token[t+1] = token[t] + 1 (mod vocab_size).
    Fixed positions have random targets (noise) to test uncertainty.

    Args:
        rng_key: JAX PRNG key
        num_samples: number of sequences
        seq_len: sequence length
        vocab_size: vocabulary size
        noisy_positions: list of token positions that always get random targets.
            Defaults to every 4th position starting at 3: [3, 7, 11, 15, ...].

    Returns:
        x: integer token indices, shape (num_samples, seq_len)
        y_onehot: one-hot targets, shape (num_samples, seq_len, vocab_size)
        noisy_positions: the positions where noise was injected
    """
    k1, k2 = jax.random.split(rng_key, 2)

    if noisy_positions is None:
        noisy_positions = list(range(3, seq_len, 4))  # e.g. [3, 7, 11, 15, ...]

    starts = jax.random.randint(k1, (num_samples, 1), 0, vocab_size)
    offsets = jnp.arange(seq_len)[None, :]
    x = (starts + offsets) % vocab_size

    # Target: next token in sequence
    y = (x + 1) % vocab_size

    # Inject noise at fixed positions (consistent across all samples)
    noise_mask = jnp.zeros((seq_len,), dtype=bool)
    noise_mask = noise_mask.at[jnp.array(noisy_positions)].set(True)
    noise_tokens = jax.random.randint(k2, (num_samples, seq_len), 0, vocab_size)
    y = jnp.where(noise_mask[None, :], noise_tokens, y)

    y_onehot = jax.nn.one_hot(y, vocab_size)
    return x, y_onehot, noisy_positions


class SimpleDataLoader:
    """Minimal data loader yielding dict batches."""

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.n = x.shape[0]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(0, self.n, self.batch_size):
            yield {
                "x": self.x[i : i + self.batch_size],
                "y": self.y[i : i + self.batch_size],
            }


# ============================================================================
# Section 7: Demo
# ============================================================================


def main():
    print("=" * 60)
    print("Bayesian Predictive Coding Transformer Demo")
    print("=" * 60)

    # --- Model config ---
    seq_len = 32
    embed_dim = 64
    num_heads = 8
    ff_dim = 64
    vocab_size = 50
    depth = 3

    # --- Inference config ---
    infer_steps = 100
    eta_infer = 0.02
    variance_config = {
        "decay": 0.9,
        "min_var": 1e-4,
        "max_var": 10.0,
        "init_var": 0.1,
    }

    rng = jax.random.PRNGKey(42)
    rng, model_key, data_key, infer_key = jax.random.split(rng, 4)

    # ---- Build model ----
    print("\n[1] Building model...")
    structure, layer_info = create_bayesian_pc_transformer(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )
    params = initialize_params(structure, model_key)
    print(f"    Nodes: {len(structure.nodes)}")
    print(f"    Edges: {len(structure.edges)}")
    print(f"    {params}")
    for info in layer_info:
        print(f"    Layer: {info}")

    # ---- Inference demo (untrained) ----
    print(f"\n[2] Running inference ({infer_steps} steps)...")
    batch_size = 4
    dk1, dk2 = jax.random.split(data_key)
    x_demo = jax.random.randint(dk1, (batch_size, seq_len), 0, vocab_size)
    y_demo = jax.nn.one_hot(
        jax.random.randint(dk2, (batch_size, seq_len), 0, vocab_size), vocab_size
    )

    # Clamps: input + output + initial variance
    clamps = {
        structure.task_map["x"]: x_demo,
        structure.task_map["y"]: y_demo,
    }
    for info in layer_info:
        clamps[info["var_node"]] = jnp.full(
            (batch_size, seq_len, 1), variance_config["init_var"]
        )

    init_state = initialize_graph_state(
        structure, batch_size, infer_key, clamps=clamps, params=params
    )

    final_state, variances, energy_history, variance_history = run_bayesian_inference(
        params,
        init_state,
        clamps,
        structure,
        infer_steps,
        eta_infer,
        layer_info,
        variance_config,
    )

    print("\n    Energy convergence:")
    for t, e in enumerate(energy_history):
        print(f"      Step {t + 1:2d}: {e:.4f}")

    print("\n    Final uncertainty (avg over batch):")
    for info in layer_info:
        v = variances[info["var_node"]]
        avg = jnp.mean(v, axis=0)[..., 0]
        print(
            f"      {info['var_node']}: mean={float(jnp.mean(avg)):.6f}, "
            f"min={float(jnp.min(avg)):.6f}, max={float(jnp.max(avg)):.6f}"
        )

    output_node_name = structure.task_map["y"]
    predictions = jnp.argmax(final_state.nodes[output_node_name].z_mu, axis=-1)
    print(f"\n    Input  (sample 0): {x_demo[0].tolist()}")
    print(f"    Predicted tokens:  {predictions[0].tolist()}")

    # ---- Training demo ----
    print(f"\n[3] Training on synthetic data...")
    rng, train_key, gen_key = jax.random.split(rng, 3)

    x_train, y_train, noisy_positions = generate_synthetic_data(
        gen_key, num_samples=256, seq_len=seq_len, vocab_size=vocab_size
    )
    train_loader = SimpleDataLoader(x_train, y_train, batch_size=16)

    train_config = {
        "optimizer": {"type": "adam", "lr": 1e-3},
        "num_epochs": 20,
        "infer_steps": infer_steps,
        "eta_infer": eta_infer,
        "variance": variance_config,
    }

    trained_params, train_energy = train_bayesian_pcn(
        params, structure, train_loader, train_config, train_key, layer_info, verbose=True
    )

    # ---- Post-training inference ----
    print("\n[4] Post-training inference...")
    rng, post_key = jax.random.split(rng)

    clamps_post = {
        structure.task_map["x"]: x_train[:batch_size],
        structure.task_map["y"]: y_train[:batch_size],
    }
    for info in layer_info:
        clamps_post[info["var_node"]] = jnp.full(
            (batch_size, seq_len, 1), variance_config["init_var"]
        )

    post_state = initialize_graph_state(
        structure, batch_size, post_key, clamps=clamps_post, params=trained_params
    )

    post_final, post_var, post_energy, post_var_hist = run_bayesian_inference(
        trained_params,
        post_state,
        clamps_post,
        structure,
        infer_steps,
        eta_infer,
        layer_info,
        variance_config,
    )

    print(f"    Final energy: {post_energy[-1]:.4f}")
    for info in layer_info:
        print(
            f"    {info['var_node']} uncertainty: "
            f"mean={float(jnp.mean(post_var[info['var_node']])):.6f}"
        )

    # ---- Visualization ----
    print("\n[5] Generating plots...")
    try:
        plot_convergence(post_energy, save_path="convergence.png")
        plot_uncertainty_heatmap(post_var, layer_info, noisy_positions=noisy_positions, save_path="uncertainty.png")
    except ImportError:
        print("  matplotlib not available, skipping plots")

    print("\nDone!")


if __name__ == "__main__":
    main()