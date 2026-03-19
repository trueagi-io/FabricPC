"""
Partially Bayesian Predictive Coding Language Model Demo

Trains a PC transformer where attention and MLP layers have mean-field Gaussian
weight posteriors with full deterministic moment propagation.  Embedding and
output layers remain deterministic.

Key features demonstrated:
  - BayesianMhaResidualNode : uncertainty-scaled attention temperature
  - BayesianLnMlp1Node      : GELU moment propagation
  - BayesianMlp2ResidualNode: residual variance accumulation + smooth normalisation
  - KL annealing            : beta ramps from 0 to kl_beta_max over 20% of training
  - Per-step logging        : reconstruction energy vs KL contribution
  - Post-training analysis  : layer-wise uncertainty (mean softplus(W_rho))

USAGE:
    $ PYTHONPATH=. python examples/bayesian_pc_lm_demo.py

The demo uses Tiny Shakespeare (loaded via CharDataLoader) at a small model
scale so it runs in reasonable time on CPU / a single GPU.
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import jax
import jax.numpy as jnp
import optax

from fabricpc.graph import initialize_params
from fabricpc.graph.state_initializer import initialize_graph_state, FeedforwardStateInit
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.nodes.bayesian import create_partial_bayesian_transformer
from fabricpc.utils.data import CharDataLoader


# =============================================================================
# PYTHON-LOOP INFERENCE (avoids slow lax.fori_loop JIT for Bayesian nodes)
# =============================================================================

@jax.jit
def _inference_step(params, state, clamps, structure):
    """Single JIT-compiled inference step (compiled once, cached)."""
    inference_obj = structure.config["inference"]
    cls = type(inference_obj)
    config = inference_obj.config
    return cls.inference_step(params, state, clamps, structure, config)


def run_inference_python_loop(params, state, clamps, structure):
    """Run inference using a Python loop so each step is JIT-compiled independently."""
    inference_obj = structure.config["inference"]
    infer_steps = inference_obj.config["infer_steps"]
    for _ in range(infer_steps):
        state = _inference_step(params, state, clamps, structure)
    return state


# =============================================================================
# CONFIG
# =============================================================================

SEQ_LEN    = 32
BATCH_SIZE = 16
MAX_SAMPLES = 16000  # cap dataset size for demo (~1000 batches/epoch)

DEPTH      = 2
EMBED_DIM  = 64
NUM_HEADS  = 4
MLP_DIM    = 128

NUM_EPOCHS    = 5
LR            = 1e-5     # must be small — deterministic demo uses 1e-5
KL_BETA_MAX   = 1e-4     # balanced: ~10% of recon energy per Bayesian node
KL_WARMUP_FRAC = 0.20    # ramp kl_beta over the first 20 % of steps

ETA_INFER  = 0.033   # matched to working deterministic demo
INFER_STEPS = 20     # slightly more steps to ensure convergence

SEED = 42

# =============================================================================
# DATA
# =============================================================================

train_loader = CharDataLoader(
    "train", seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True, seed=SEED,
    max_samples=MAX_SAMPLES,
)
val_loader = CharDataLoader(
    "validation", seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=False
)

vocab_size  = train_loader.vocab_size
char_to_idx = train_loader.char_to_idx
idx_to_char = train_loader.idx_to_char

steps_per_epoch = len(train_loader)  # len() returns num_batches directly
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = max(1, int(total_steps * KL_WARMUP_FRAC))

print(f"Vocab size: {vocab_size} | Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

# =============================================================================
# MODEL
# =============================================================================

inference = InferenceSGDNormClip(
    eta_infer=ETA_INFER,
    infer_steps=INFER_STEPS,
    max_norm=1.0,
)

structure = create_partial_bayesian_transformer(
    depth=DEPTH,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM,
    seq_len=SEQ_LEN,
    vocab_size=vocab_size,
    inference=inference,
    kl_beta=KL_BETA_MAX,   # will be overridden per-step during training
    sigma_noise=1.0,
)

rng = jax.random.PRNGKey(SEED)
rng, init_key = jax.random.split(rng)
params = initialize_params(structure, init_key)
print(f"Parameters: {params}")

# =============================================================================
# TRAINING LOOP
# =============================================================================

optimizer = optax.adam(LR)
opt_state = optimizer.init(params)

# JIT-compile the learning step
@jax.jit
def learning_step(params, final_state, structure):
    """Compute local weight gradients from converged inference state."""
    from fabricpc.graph.graph_net import compute_local_weight_gradients
    return compute_local_weight_gradients(params, final_state, structure)


def kl_schedule(step: int) -> float:
    """Linear ramp from 0 to KL_BETA_MAX over warmup_steps."""
    return float(jnp.minimum(step / warmup_steps, 1.0)) * KL_BETA_MAX


def run_epoch(params, opt_state, loader, epoch, global_step):
    total_energy = 0.0
    total_kl     = 0.0
    n_batches    = 0

    for batch_x, batch_y in loader:
        rng_key = jax.random.PRNGKey(global_step)  # deterministic per step

        # KL annealing
        kl_beta = kl_schedule(global_step)

        # Clamp input (x) and target (y)
        x_arr = jnp.array(batch_x, dtype=jnp.float32)
        y_arr = jnp.array(batch_y, dtype=jnp.float32)
        clamps = {"input_ids": x_arr, "logits": y_arr}

        # Initialise state via feedforward pass
        state = initialize_graph_state(
            structure, BATCH_SIZE, rng_key, clamps=clamps,
            state_init=FeedforwardStateInit(), params=params,
        )

        # Inference: minimise free energy over latents
        final_state = run_inference_python_loop(params, state, clamps, structure)

        # Compute per-node energy for logging
        recon_energy = float(jnp.mean(jnp.stack(
            [jnp.mean(ns.energy) for ns in final_state.nodes.values()]
        )))

        # Collect KL contributions from Bayesian nodes
        kl_total = 0.0
        for node_name, node in structure.nodes.items():
            node_class = node.node_info.node_class.__name__
            if "Bayesian" in node_class:
                np_ = params.nodes[node_name]
                for wname, wval in np_.weights.items():
                    if wname.endswith("_mu"):
                        rho_name = wname[:-3] + "_rho"
                        if rho_name in np_.weights:
                            sigma = jax.nn.softplus(np_.weights[rho_name]) + 1e-6
                            kl = -0.5 * jnp.sum(
                                1.0 + jnp.log(sigma ** 2) - wval ** 2 - sigma ** 2
                            )
                            kl_total += float(kl)

        # Weight update
        grads = learning_step(params, final_state, structure)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)

        total_energy += recon_energy
        total_kl     += kl_total * kl_beta
        n_batches    += 1
        global_step  += 1
        opt_state     = opt_state_new
        params        = params_new

        if global_step % 25 == 0:
            print(
                f"  Epoch {epoch+1} | Step {global_step:5d} | "
                f"Recon energy: {recon_energy:.4f} | "
                f"KL (weighted): {kl_total * kl_beta:.4f} | "
                f"kl_beta: {kl_beta:.2e}"
            )

    avg_energy = total_energy / max(n_batches, 1)
    avg_kl     = total_kl / max(n_batches, 1)
    return params, opt_state, avg_energy, avg_kl, global_step


print("\n=== Training ===")
global_step = 0
for epoch in range(NUM_EPOCHS):
    params, opt_state, avg_e, avg_kl, global_step = run_epoch(
        params, opt_state, train_loader, epoch, global_step
    )
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} done | "
        f"Avg recon energy: {avg_e:.4f} | Avg weighted KL: {avg_kl:.4f}"
    )

# =============================================================================
# POST-TRAINING: LAYER-WISE UNCERTAINTY ANALYSIS
# =============================================================================

print("\n=== Uncertainty Analysis (mean sigma per weight tensor) ===")
print(f"{'Node':<20} {'Weight':<14} {'mean(sigma)':<14} {'mean(|mu|)':<12}")
print("-" * 62)

for node_name, node in structure.nodes.items():
    node_class = node.node_info.node_class.__name__
    if "Bayesian" not in node_class:
        continue
    np_ = params.nodes[node_name]
    for wname, wval in np_.weights.items():
        if not wname.endswith("_mu"):
            continue
        rho_name = wname[:-3] + "_rho"
        if rho_name not in np_.weights:
            continue
        sigma = float(jnp.mean(jax.nn.softplus(np_.weights[rho_name]) + 1e-6))
        mu_abs = float(jnp.mean(jnp.abs(wval)))
        print(f"{node_name:<20} {wname:<14} {sigma:<14.6f} {mu_abs:<12.6f}")

# =============================================================================
# TEXT GENERATION
# =============================================================================


def generate(params, structure, start_text="ROMEO: ", length=100, temperature=0.8):
    seed_indices = [char_to_idx.get(c, 0) for c in start_text]
    if len(seed_indices) < SEQ_LEN:
        current_indices = [0] * (SEQ_LEN - len(seed_indices)) + seed_indices
    else:
        current_indices = seed_indices[-SEQ_LEN:]

    result = start_text
    gen_key = jax.random.PRNGKey(99)

    for _ in range(length):
        x_batch = jnp.array([current_indices], dtype=jnp.float32)
        clamps  = {"input_ids": x_batch}

        state = initialize_graph_state(
            structure, 1, gen_key, clamps=clamps,
            state_init=FeedforwardStateInit(), params=params,
        )
        final_state = run_inference_python_loop(params, state, clamps, structure)

        logits_state = final_state.nodes["logits"]
        probs = logits_state.z_latent[0, -1, :]
        log_probs = jnp.log(probs + 1e-8)

        gen_key, sample_key = jax.random.split(gen_key)
        next_idx = int(jax.random.categorical(sample_key, log_probs / temperature))
        result  += idx_to_char[next_idx]
        current_indices = current_indices[1:] + [next_idx]

    return result


print("\n=== Generation ===")
print(generate(params, structure, start_text="ROMEO: ", length=120, temperature=0.9))
