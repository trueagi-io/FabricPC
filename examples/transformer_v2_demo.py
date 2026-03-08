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

import os
import jax
import jax.numpy as jnp
import torch
from torch.utils.data import DataLoader, Dataset
from fabricpc.graph import create_pc_graph
from fabricpc.training.multi_gpu import train_pcn_multi_gpu, evaluate_pcn_multi_gpu
from fabricpc.core.inference import run_inference
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.nodes.transformer_v2 import create_deep_transformer


# ----------------------------------------------------------------------
# LOAD LOCAL DATA
# ----------------------------------------------------------------------
def load_data(path="data/tiny_shakespeare.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [char_to_ix[ch] for ch in text]

    return data, vocab_size, char_to_ix, ix_to_char


def split_data(data, train_frac=0.8, val_frac=0.1):
    N = len(data)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    return train_data, val_data, test_data


class TextDataset(Dataset):
    def __init__(self, data, seq_len, vocab_size):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, i):
        x = torch.tensor(self.data[i : i + self.seq_len], dtype=torch.float32)
        y = torch.tensor(self.data[i + 1 : i + self.seq_len + 1], dtype=torch.long)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.vocab_size).float()
        return x, y_one_hot


# ----------------------------------------------------------------------
# INITIALIZE DATA
# ----------------------------------------------------------------------
seq_len = 32

# Scale batch size by device count
n_devices = jax.device_count()
base_batch_size = 32
batch_size = base_batch_size * n_devices
print(f"Running on {n_devices} device(s). Total batch size: {batch_size}")

data, vocab_size, char_to_ix, ix_to_char = load_data()
train_data, val_data, test_data = split_data(data)

train_dataset = TextDataset(train_data, seq_len, vocab_size)
val_dataset = TextDataset(val_data, seq_len, vocab_size)
test_dataset = TextDataset(test_data, seq_len, vocab_size)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------------------------------------------------
# MODEL ARCHITECTURE
# ----------------------------------------------------------------------
config = create_deep_transformer(
    depth=1,
    embed_dim=64,
    num_heads=4,
    mlp_dim=128,
    seq_len=seq_len,
    vocab_size=vocab_size,
    weight_init={"type": "normal", "std": 0.02},
)

# ----------------------------------------------------------------------
# CREATE PC GRAPH & TRAIN
# ----------------------------------------------------------------------
master_rng_key = jax.random.PRNGKey(42)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)
params, structure = create_pc_graph(config, graph_key)

train_config = {
    "num_epochs": 5,
    "infer_steps": 20,
    "eta_infer": 0.01,
    "optimizer": {"type": "adam", "lr": 1e-4},
}

print(f"Vocab Size: {vocab_size} | Training on local tiny_shakespeare.txt...")
trained_params = train_pcn_multi_gpu(
    params, structure, train_loader, train_config, train_key, verbose=True
)
metrics = evaluate_pcn_multi_gpu(
    trained_params, structure, test_loader, train_config, eval_key
)

print(f"Test Accuracy:   {metrics['accuracy'] * 100:.2f}%")
print(f"Test CE Loss:  {metrics['cross_entropy']:.4f}")
print(f"Test Perplexity: {metrics['perplexity']:.2f}")
print(f"Test Energy:     {metrics['energy']:.4f}")


# ----------------------------------------------------------------------
# TEXT GENERATION
# ----------------------------------------------------------------------
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

        # Initialize state
        state = initialize_graph_state(
            structure, batch_size, gen_key, clamps=inputs, params=trained_params
        )

        # Run PC inference
        final_state = run_inference(
            trained_params,
            state,
            clamps=inputs,
            structure=structure,
            infer_steps=train_config["infer_steps"],
            eta_infer=train_config["eta_infer"],
        )

        logits_node_state = final_state.nodes["logits"]
        last_step_logits = logits_node_state.z_latent[0, -1, :]

        # Temperature Sampling
        gen_key, sample_key = jax.random.split(gen_key)
        scaled_logits = last_step_logits / temperature
        next_idx = int(jax.random.categorical(sample_key, scaled_logits))

        next_char = ix_to_char[next_idx]
        result_text += next_char
        current_indices = current_indices[1:] + [next_idx]

    print(result_text)


generate(trained_params, structure, start_text="ROMEO: ", length=100, temperature=0.8)
