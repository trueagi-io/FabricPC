"""
Transformer Predictive Coding Demo
==================================

A quick experiment demonstrating transformer blocks with predictive coding
on character-level language modeling using the TinyShakespeare dataset.

This demo:
1. Downloads TinyShakespeare (~1MB of text)
2. Trains a small transformer for next-character prediction
3. Generates sample text

Expected runtime: ~5-10 minutes on a consumer GPU (RTX 3080/4080 class)

Training needs tuning, eval metrics, and AIM tensor board - treat this as a starting point!
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
import os
import time
import urllib.request
from typing import Tuple, Iterator, Dict, List

from fabricpc.graph import create_pc_graph
from fabricpc.training.train_autoregressive import train_autoregressive, generate_autoregressive
from fabricpc.training.train_backprop import train_backprop_autoregressive

# Reproducibility
jax.config.update('jax_default_prng_impl', 'threefry2x32')
np.random.seed(42)


# ==============================================================================
# DATA LOADING: TinyShakespeare
# ==============================================================================

def download_tiny_shakespeare(data_dir: str = "./data") -> str:
    """Download TinyShakespeare dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "tiny_shakespeare.txt")

    if not os.path.exists(filepath):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading TinyShakespeare from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


class CharDataset:
    """Character-level dataset for language modeling."""

    def __init__(self, text: str, seq_len: int = 64):
        self.text = text
        self.seq_len = seq_len

        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode full text
        self.data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Dataset size: {len(self.data)} characters")
        print(f"Sample chars: {''.join(self.chars[:20])}...")

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get input sequence and target (next character for each position)."""
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]  # Shifted by 1
        return x, y

    def decode(self, indices: np.ndarray) -> str:
        """Convert indices back to text."""
        return ''.join([self.idx_to_char[int(i)] for i in indices])


def create_dataloader(
    dataset: CharDataset,
    batch_size: int,
    shuffle: bool = True
) -> Iterator[Dict[str, np.ndarray]]:
    """Simple dataloader for character sequences."""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        if len(batch_indices) < batch_size:
            continue  # Skip incomplete batches

        batch_x = []
        batch_y = []
        for idx in batch_indices:
            x, y = dataset[idx]
            batch_x.append(x)
            batch_y.append(y)

        # Convert to one-hot for input embedding
        x_array = np.array(batch_x)  # (batch, seq_len)
        y_array = np.array(batch_y)  # (batch, seq_len)

        # One-hot encode: (batch, seq_len, vocab_size)
        x_onehot = np.eye(dataset.vocab_size)[x_array]
        y_onehot = np.eye(dataset.vocab_size)[y_array]

        yield {"x": x_onehot, "y": y_onehot}


class DataLoaderWrapper:
    """Wrapper to make dataloader compatible with train_pcn."""

    def __init__(self, dataset: CharDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._len = len(dataset) // batch_size

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        return create_dataloader(self.dataset, self.batch_size, self.shuffle)


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

def create_transformer_config(
    vocab_size: int,
    seq_len: int = 64,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_blocks: int = 2,
    ff_dim: int = 256,
) -> dict:
    """
    Create a transformer language model configuration.

    Architecture:
        Input (one-hot) -> Embedding -> [Transformer Block] x N -> Output projection

    For predictive coding, each node maintains its own latent state.
    """
    node_list = [
        # Input embedding: linear projection from one-hot to embed_dim
        {
            "name": "input",
            "shape": (seq_len, vocab_size),
            "type": "linear",
            "activation": {"type": "identity"},
        },
        {
            "name": "embed",
            "shape": (seq_len, embed_dim),
            "type": "linear",
            "activation": {"type": "identity"},
            "weight_init": {"type": "kaiming", "mode": "fan_out"},
        },
        {
            "name": "mask",
            "shape": (1, seq_len, seq_len),
            "type": "linear",
            "activation": {"type": "identity"},
        }
    ]

    edge_list = [
        {"source_name": "input", "target_name": "embed", "slot": "in"},
    ]

    # Add transformer blocks
    prev_name = "embed"
    for i in range(num_blocks):
        block_name = f"transformer_{i}"
        node_list.append({
            "name": block_name,
            "shape": (seq_len, embed_dim),
            "type": "transformer_block",
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "internal_activation": {"type": "gelu"},
        })
        edge_list.append({"source_name": prev_name, "target_name": block_name, "slot": "in"})  # wire one block to the next
        edge_list.append({"source_name": "mask", "target_name": block_name, "slot": "mask"})  # wire mask to the block
        prev_name = block_name

    # Output projection back to vocabulary
    node_list.append({
        "name": "output",
        "shape": (seq_len, vocab_size),
        "type": "linear",
        "activation": {"type": "softmax"},
        "energy": {"type": "cross_entropy"},
        "weight_init": {"type": "kaiming", "mode": "fan_out"},
    })
    edge_list.append({"source_name": prev_name, "target_name": "output", "slot": "in"})

    return {
        "node_list": node_list,
        "edge_list": edge_list,
        "task_map": {"x": "input", "y": "output", "causal_mask": "mask"},
    }


# ==============================================================================
# TEXT GENERATION
# ==============================================================================

def generate_text(
    params,
    structure,
    dataset: CharDataset,
    prompts: List[str],
    max_new_tokens: int = 100,
    rng_key: jax.Array = None,
    temperature: float = 0.8,
    infer_steps: int = 30,
    eta_infer: float = 0.01,
    top_k: int = None,
    top_p: float = None,
) -> List[str]:
    """
    Generate text autoregressively from batched prompts.

    Uses the autoregressive generation function from train_autoregressive module.
    Supports top-k and top-p (nucleus) sampling for better text quality.

    Args:
        params: Trained model parameters
        structure: Graph structure
        dataset: CharDataset with vocabulary mappings
        prompts: List of prompt strings to generate from
        max_new_tokens: Number of new tokens to generate per prompt
        rng_key: JAX random key
        temperature: Sampling temperature (higher = more random)
        infer_steps: Number of inference steps
        eta_infer: Inference learning rate
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling with this probability threshold

    Returns:
        List of generated strings (one per prompt)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    seq_len = structure.nodes["input"].shape[0]
    pad_char = dataset.char_to_idx.get(' ', 0)

    # Encode all prompts and pad to seq_len
    batch_indices = []
    for prompt in prompts:
        prompt_indices = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
        if len(prompt_indices) > seq_len:
            prompt_indices = prompt_indices[-seq_len:]
        elif len(prompt_indices) < seq_len:
            prompt_indices = [pad_char] * (seq_len - len(prompt_indices)) + prompt_indices
        batch_indices.append(prompt_indices)

    # Convert to JAX array
    prompt_tokens = jnp.array(batch_indices)  # (batch_size, seq_len)

    # Use the autoregressive generation function
    generated_tokens = generate_autoregressive(
        params=params,
        structure=structure,
        prompt=prompt_tokens,
        max_new_tokens=max_new_tokens,
        rng_key=rng_key,
        temperature=temperature,
        infer_steps=infer_steps,
        eta_infer=eta_infer,
        top_k=top_k,
        top_p=top_p,
    )

    # Decode the generated tokens back to strings
    generated_texts = []
    for i, prompt in enumerate(prompts):
        # Get the tokens for this batch element
        tokens = np.array(generated_tokens[i])

        # Decode only the non-padded part (skip initial padding)
        pad_len = seq_len - len(prompt)
        if pad_len > 0:
            tokens = tokens[pad_len:]

        text = dataset.decode(tokens)
        generated_texts.append(text)

    return generated_texts


# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================


def main():
    print("=" * 70)
    print("Transformer Predictive Coding Demo")
    print("Character-level language modeling on TinyShakespeare")
    print("=" * 70)

    # Choose training mode (PC or backprop)
    use_pcn = True  # Set to True to use predictive coding training

    # Configuration
    SEQ_LEN = 128        # Sequence length
    EMBED_DIM = 64     # Embedding dimension
    NUM_HEADS = 8       # Attention heads
    NUM_BLOCKS = 3      # Transformer blocks
    FF_DIM = 256        # Feedforward hidden dimension
    BATCH_SIZE = 128     # Batch size
    NUM_EPOCHS = 10      # Training epochs
    INFER_STEPS = 20    # Inference iterations per step
    ETA_INFER = 0.05    # Inference learning rate
    LR = 1e-5           # Weight learning rate

    # Random keys
    master_key = jax.random.PRNGKey(42)
    graph_key, train_key, gen_key = jax.random.split(master_key, 3)

    # Load data
    print("\n[1/5] Loading TinyShakespeare dataset...")
    text = download_tiny_shakespeare()

    # Use a subset for faster training (first 100k characters)
    dataset_cutoff_len = 100000
    print(f"Using first {dataset_cutoff_len} of {len(text)} total characters for training...")
    text = text[:dataset_cutoff_len]
    dataset = CharDataset(text, seq_len=SEQ_LEN)

    train_loader = DataLoaderWrapper(dataset, BATCH_SIZE, shuffle=True)
    print(f"Training batches per epoch: {len(train_loader)}")

    # Create model
    print("\n[2/5] Creating transformer model...")
    config = create_transformer_config(
        vocab_size=dataset.vocab_size,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        ff_dim=FF_DIM,
    )

    params, structure = create_pc_graph(config, graph_key)

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model created: {len(config['node_list'])} nodes, {len(config['edge_list'])} edges")
    print(f"Total parameters: {total_params:,}")

    # Training config for autoregressive training
    train_config = {
        "num_epochs": NUM_EPOCHS,
        "infer_steps": INFER_STEPS,
        "eta_infer": ETA_INFER,
        "optimizer": {"type": "adam", "lr": LR, "weight_decay": 0.001},
        "use_causal_mask": True,  # Enable causal masking for autoregressive
    }

    # Train using autoregressive trainer
    print("\n[3/5] Training with autoregressive trainer (JIT compilation on first batch)...")
    print(f"Config: {NUM_EPOCHS} epochs, {INFER_STEPS} inference steps, lr={LR}")

    start_time = time.time()

    ### Choose one of the training methods below
    if use_pcn:
        # Train with predictive coding (autoregressive)
        trained_params, energy_history, _ = train_autoregressive(
            params=params,
            structure=structure,
            train_loader=train_loader,
            config=train_config,
            rng_key=train_key,
            verbose=True,
        )
    else:
        # Backprop (autoregressive)
        trained_params, energy_history, _ = train_backprop_autoregressive(
            params, structure, train_loader,
            {"num_epochs": NUM_EPOCHS, "optimizer": {"type": "adam", "lr": 1e-3}, "use_causal_mask": True},
            train_key
        )

    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.1f}s ({train_time/NUM_EPOCHS:.1f}s per epoch)")

    # Generate samples
    print("\n[4/5] Generating sample text...")
    prompts = [
        "Know, Rome, that",
        "MENENIUS:",
        "the more virtuous",
        "by his looks",
        "ROMEO: ",
        "To be or not to be",
        "The king",
    ]

    # Batch all prompts together for efficient generation
    generated_texts = generate_text(
        trained_params,
        structure,
        dataset,
        prompts=prompts,
        max_new_tokens=20,
        rng_key=gen_key,
        temperature=0.8,
        infer_steps=INFER_STEPS,
        eta_infer=ETA_INFER,
    )

    for prompt, generated in zip(prompts, generated_texts):
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        print(generated)
        print("-" * 40)

    # Summary
    print("\n[5/5] Summary")
    print("=" * 70)
    print(f"Dataset: TinyShakespeare ({len(text):,} characters, {dataset.vocab_size} unique)")
    print(f"Model: {NUM_BLOCKS} transformer blocks, {EMBED_DIM}d embeddings, {NUM_HEADS} heads")
    print(f"Parameters: {total_params:,}")
    print(f"Training: {NUM_EPOCHS} epochs in {train_time:.1f}s")
    print(f"Final energy: {energy_history[-1][-1]:.4f}")
    print("=" * 70)

    return trained_params, structure, dataset


if __name__ == "__main__":
    main()