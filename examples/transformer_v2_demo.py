"""
FabricPC Transformer Execution Script

This script trains the decomposed PC Transformer model on the Tiny
Shakespeare dataset using JAX's multi-GPU capabilities (`pmap`),
evaluates its performance, and generates sample text using temperature
sampling to prevent repetitive loops.

USAGE:
Run this script from the root of the project directory. You must set
the PYTHONPATH so Python can locate the `fabricpc` package.

    $ PYTHONPATH=. python examples/transformer_v2_demo.py

"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import jax
import jax.numpy as jnp
from fabricpc.graph import initialize_params
from fabricpc.training.multi_gpu import (
    train_pcn_multi_gpu,
    evaluate_transformer_multi_gpu,
)
from fabricpc.core.inference import run_inference, InferenceSGD
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.utils.data import CharDataLoader
import optax
import time

# --- Data ---

seq_len = 32

n_devices = jax.device_count()
base_batch_size = 32
batch_size = base_batch_size * n_devices
print(f"Running on {n_devices} device(s). Total batch size: {batch_size}")

train_loader = CharDataLoader(
    "train", seq_len=seq_len, batch_size=batch_size, shuffle=True, seed=42
)
val_loader = CharDataLoader(
    "validation", seq_len=seq_len, batch_size=batch_size, shuffle=False
)
test_loader = CharDataLoader(
    "test", seq_len=seq_len, batch_size=batch_size, shuffle=False
)
vocab_size = train_loader.vocab_size
char_to_ix = train_loader.char_to_idx
ix_to_char = train_loader.idx_to_char

# --- Model ---

structure = create_deep_transformer(
    depth=4,
    embed_dim=64,
    num_heads=4,
    mlp_dim=128,
    seq_len=seq_len,
    vocab_size=vocab_size,
    inference=InferenceSGD(eta_infer=0.033195052120243505, infer_steps=17),
    weight_init={"type": "normal", "std": 0.04402197307582635},
)

# --- Train & Evaluate ---

master_rng_key = jax.random.PRNGKey(42)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

params = initialize_params(structure, graph_key)

train_config = {
    "num_epochs": 5,
}
optimizer = optax.adam(1e-5)

print(f"Vocab Size: {vocab_size}")
start = time.time()
trained_params = train_pcn_multi_gpu(
    params, structure, train_loader, optimizer, train_config, train_key, verbose=True
)
print(f"Training completed in {time.time() - start:.1f}s")

# Evaluate
metrics = evaluate_transformer_multi_gpu(
    trained_params, structure, test_loader, train_config, eval_key
)

print(f"Test Accuracy:   {metrics['accuracy'] * 100:.2f}%")
print(f"Test CE Loss:  {metrics['cross_entropy']:.4f}")
print(f"Test Perplexity: {metrics['perplexity']:.2f}")
print(f"Test Energy:     {metrics['energy']:.4f}")


# --- Text Generation ---


def generate(
    trained_params, structure, start_text="ROMEO: ", length=50, temperature=0.8
):
    seed_indices = [char_to_ix.get(c, 0) for c in start_text]
    if len(seed_indices) < seq_len:
        current_indices = [0] * (seq_len - len(seed_indices)) + seed_indices
    else:
        current_indices = seed_indices[-seq_len:]

    result_text = start_text
    gen_key = jax.random.PRNGKey(99)

    print(f"--- Generating ---")
    for _ in range(length):
        input_batch = jnp.array([current_indices], dtype=jnp.float32)
        inputs = {"input_ids": input_batch}
        batch_size = input_batch.shape[0]

        state = initialize_graph_state(
            structure, batch_size, gen_key, clamps=inputs, params=trained_params
        )

        final_state = run_inference(trained_params, state, inputs, structure)

        logits_node_state = final_state.nodes["logits"]
        last_step_logits = logits_node_state.z_latent[0, -1, :]

        gen_key, sample_key = jax.random.split(gen_key)
        scaled_logits = last_step_logits / temperature
        next_idx = int(jax.random.categorical(sample_key, scaled_logits))

        next_char = ix_to_char[next_idx]
        result_text += next_char
        current_indices = current_indices[1:] + [next_idx]

    print(result_text)


generate(trained_params, structure, start_text="ROMEO: ", length=100, temperature=0.8)
