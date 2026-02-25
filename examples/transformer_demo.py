"""
Transformer Predictive Coding Demo
==================================

A quick experiment demonstrating transformer blocks with predictive coding versus backprop training
on character-level language modeling using the TinyShakespeare dataset.

Works in Backprop mode out of the box, but predictive coding training is not yet tuned - treat this as a starting point for experimentation!

This demo:
1. Downloads TinyShakespeare (~1MB of text)
2. Trains a small transformer for next-character prediction
3. Generates sample text
4. Backprop training option, set use_pcn = False

Expected runtime: ~5-20 minutes on a consumer GPU (RTX 3080/4080 class)
"""

use_pcn = False  # Set to True to use predictive coding training, False for backprop

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import numpy as np
import time
import urllib.request
from typing import Tuple, Iterator, Dict, List, Optional, Any
from tqdm.auto import tqdm

from fabricpc.nodes import Linear, TransformerBlock
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params, FeedforwardStateInit
from fabricpc.core.activations import (
    IdentityActivation,
    SoftmaxActivation,
    GeluActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import NormalInitializer, KaimingInitializer
from fabricpc.training.train_autoregressive import (
    train_autoregressive,
    generate_autoregressive,
    evaluate_autoregressive,
)
from fabricpc.training.train_backprop import (
    train_backprop_autoregressive,
    evaluate_backprop_autoregressive,
)

# Reproducibility
jax.config.update("jax_default_prng_impl", "threefry2x32")
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

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    return text


class CharDataset:
    """Character-level dataset for language modeling."""

    def __init__(self, text: str, seq_len: int, vocab: dict = None):
        self.text = text
        self.seq_len = seq_len

        # Build vocabulary or use provided one
        if vocab is None:
            self.chars = sorted(list(set(text)))
            self.vocab_size = len(self.chars)
            self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        else:
            self.char_to_idx = vocab["char_to_idx"]
            self.idx_to_char = vocab["idx_to_char"]
            self.chars = vocab["chars"]
            self.vocab_size = vocab["vocab_size"]

        # Encode full text
        self.data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Dataset size: {len(self.data)} characters")
        print(f"Sample chars: {''.join(self.chars[:20])}...")

    def get_vocab(self) -> dict:
        """Return vocabulary for sharing with other datasets."""
        return {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "chars": self.chars,
            "vocab_size": self.vocab_size,
        }

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get input sequence and target (next character for each position)."""
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]  # Shifted by 1
        return x, y

    def decode(self, indices: np.ndarray) -> str:
        """Convert indices back to text."""
        return "".join([self.idx_to_char[int(i)] for i in indices])


def create_dataloader(
    dataset: CharDataset, batch_size: int, shuffle: bool = True
) -> Iterator[Dict[str, np.ndarray]]:
    """Simple dataloader for character sequences."""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        batch_indices = indices[start_idx : start_idx + batch_size]
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


def create_transformer_model(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    ff_dim: int,
    rope_theta: float,
    rng_key: jax.Array,
) -> Tuple:
    """
    Create a transformer language model using the new object-oriented API.

    Architecture:
        Input (one-hot) -> Embedding -> [Transformer Block] x N -> Output projection

    For predictive coding, each node maintains its own latent state.

    Args:
        vocab_size: Size of the vocabulary
        seq_len: Sequence length
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_blocks: Number of transformer blocks
        ff_dim: Feedforward hidden dimension (should be 4x embed_dim)
        rope_theta: RoPE frequency parameter (longer than sequence length)
        rng_key: JAX random key for parameter initialization

    Returns:
        Tuple of (structure, params)
    """
    input_node = Linear(
        shape=(seq_len, vocab_size), activation=IdentityActivation(), name="input"
    )
    embed = Linear(
        shape=(seq_len, embed_dim),
        activation=IdentityActivation(),
        weight_init=KaimingInitializer(mode="fan_out"),
        name="embed",
    )
    mask_node = Linear(
        shape=(1, seq_len, seq_len), activation=IdentityActivation(), name="mask"
    )

    nodes = [input_node, embed, mask_node]
    edges = [Edge(source=input_node, target=embed.slot("in"))]

    prev_node = embed
    for i in range(num_blocks):
        block = TransformerBlock(
            shape=(seq_len, embed_dim),
            num_heads=num_heads,
            ff_dim=ff_dim,
            internal_activation=GeluActivation(),
            rope_theta=rope_theta,
            name=f"transformer_{i}",
        )
        nodes.append(block)
        edges.append(Edge(source=prev_node, target=block.slot("in")))
        edges.append(Edge(source=mask_node, target=block.slot("mask")))
        prev_node = block

    output_node = Linear(
        shape=(seq_len, vocab_size),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=NormalInitializer(mean=0.0, std=0.02),
        name="output",
    )
    nodes.append(output_node)
    edges.append(Edge(source=prev_node, target=output_node.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=output_node, causal_mask=mask_node),
        graph_state_initializer=FeedforwardStateInit(),
    )
    params = initialize_params(structure, rng_key)
    return structure, params


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

    seq_len = structure.nodes["input"].node_info.shape[0]
    pad_char = dataset.char_to_idx.get(" ", 0)

    # Encode all prompts and pad to seq_len
    batch_indices = []
    for prompt in prompts:
        prompt_indices = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
        if len(prompt_indices) > seq_len:
            prompt_indices = prompt_indices[-seq_len:]
        elif len(prompt_indices) < seq_len:
            prompt_indices = [pad_char] * (
                seq_len - len(prompt_indices)
            ) + prompt_indices
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


class TrainingProgressBar:
    """Manage per-epoch tqdm bars during training."""

    def __init__(self, total_batches: int, num_epochs: int, mode_label: str):
        self.total_batches = total_batches
        self.num_epochs = num_epochs
        self.mode_label = mode_label
        self.current_epoch: Optional[int] = None
        self._bar: Optional[Any] = None

    def _open_epoch_bar(self, epoch_idx: int):
        self.close()
        self.current_epoch = epoch_idx
        self._bar = tqdm(
            total=self.total_batches,
            desc=f"{self.mode_label} Epoch {epoch_idx + 1}/{self.num_epochs}",
            dynamic_ncols=True,
            leave=False,
        )

    def update(self, epoch_idx: int, metrics: Dict[str, float]):
        if self.current_epoch != epoch_idx:
            self._open_epoch_bar(epoch_idx)

        if self._bar is None:
            return

        self._bar.update(1)
        formatted_metrics = {
            key: f"{value:.2f}" if key == "ppl" else f"{value:.4f}"
            for key, value in metrics.items()
        }
        self._bar.set_postfix(formatted_metrics, refresh=False)

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def main():
    print("=" * 70)
    print("Transformer Predictive Coding Demo")
    print("Character-level language modeling on TinyShakespeare")
    print("")
    print(
        "Not yet tuned in hyperparams and weight initialization for PC training - treat this as a starting point!"
    )
    print("=" * 70)

    # fmt: off
    # Configuration
    SEQ_LEN = 128        # Sequence length
    EMBED_DIM = 128      # Embedding dimension
    NUM_HEADS = 8       # Attention heads
    NUM_BLOCKS = 3      # Transformer blocks
    FF_DIM = 512        # Feedforward hidden dimension
    ROPE_THETA = 500.0    # RoPE frequency
    BATCH_SIZE = 128     # Batch size
    NUM_EPOCHS = 1      # Training epochs
    INFER_STEPS = 10    # Inference iterations per step
    ETA_INFER = 0.05    # Inference learning rate
    LR = 1e-3           # Weight learning rate

    # fmt: on
    # Random keys
    master_key = jax.random.PRNGKey(42)
    graph_key, train_key, gen_key = jax.random.split(master_key, 3)

    # Load data
    print("\n[1/5] Loading TinyShakespeare dataset...")
    full_text = download_tiny_shakespeare()

    # Optionally use a subset for faster training (e.g. first 100k characters)
    dataset_cutoff_len = len(full_text)
    print(f"Using first {dataset_cutoff_len} of {len(full_text)} total characters...")
    full_text = full_text[:dataset_cutoff_len]

    # Split into train (90%) and test (10%) - last 10% reserved for test
    split_idx = int(len(full_text) * 0.9)
    train_text = full_text[:split_idx]
    test_text = full_text[split_idx:]
    print(
        f"Train/test split: {len(train_text):,} train / {len(test_text):,} test characters"
    )

    # Create train dataset (builds vocabulary)
    print("\nTrain dataset:")
    train_dataset = CharDataset(train_text, seq_len=SEQ_LEN)

    # Create test dataset (uses train vocabulary)
    print("\nTest dataset:")
    test_dataset = CharDataset(
        test_text, seq_len=SEQ_LEN, vocab=train_dataset.get_vocab()
    )

    train_loader = DataLoaderWrapper(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoaderWrapper(test_dataset, BATCH_SIZE, shuffle=False)
    print(f"\nTraining batches per epoch: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n[2/5] Creating transformer model...")
    structure, params = create_transformer_model(
        vocab_size=train_dataset.vocab_size,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        ff_dim=FF_DIM,
        rope_theta=ROPE_THETA,
        rng_key=graph_key,
    )

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model created: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    print(f"Total parameters: {total_params:,}")

    # Training config for autoregressive training
    train_config = {
        "num_epochs": NUM_EPOCHS,
        "infer_steps": INFER_STEPS,
        "eta_infer": ETA_INFER,
        "optimizer": {"type": "adam", "lr": LR, "weight_decay": 0.001},
        "use_causal_mask": True,  # Enable causal masking for autoregressive
    }

    # Create evaluation callback for test set
    def create_eval_callback(use_pc: bool):
        """Create appropriate eval callback based on training method."""
        if use_pc:

            def eval_callback(epoch_idx, params, structure, config, rng_key):
                eval_rng = jax.random.fold_in(rng_key, epoch_idx)
                metrics = evaluate_autoregressive(
                    params,
                    structure,
                    test_loader,
                    {
                        "infer_steps": 0,
                        "eta_infer": ETA_INFER,
                        "use_causal_mask": True,
                    },  # No inference steps for eval because model predicts feedforward
                    eval_rng,
                    debug=(epoch_idx == 0),  # Debug first epoch only
                )
                tqdm.write(
                    f"  Test - Loss: {metrics['loss']:.4f}, Perplexity: {metrics['perplexity']:.2f}, Acc: {metrics['accuracy']:.4f}"
                )
                return metrics

        else:

            def eval_callback(epoch_idx, params, structure, config, rng_key):
                eval_rng = jax.random.fold_in(rng_key, epoch_idx)
                # Enable debug on first epoch to diagnose metrics
                metrics = evaluate_backprop_autoregressive(
                    params,
                    structure,
                    test_loader,
                    {"use_causal_mask": True},
                    eval_rng,
                    debug=(epoch_idx == 0),  # Debug first epoch only
                )
                tqdm.write(
                    f"  Test - Loss: {metrics['loss']:.4f}, Perplexity: {metrics['perplexity']:.2f}, Acc: {metrics['accuracy']:.4f}"
                )
                return metrics

        return eval_callback

    eval_callback = create_eval_callback(use_pcn)
    progress_bar = TrainingProgressBar(
        total_batches=len(train_loader),
        num_epochs=NUM_EPOCHS,
        mode_label="PC" if use_pcn else "BP",
    )

    def create_iter_callback(use_pc: bool):
        if use_pc:

            def iter_callback(epoch_idx, batch_idx, energy):
                del batch_idx
                energy_value = float(energy)
                progress_bar.update(epoch_idx, {"energy": energy_value})
                return energy_value

        else:

            def iter_callback(epoch_idx, batch_idx, loss):
                del batch_idx
                loss_value = float(loss)
                perplexity = float(np.exp(loss_value))
                progress_bar.update(epoch_idx, {"loss": loss_value, "ppl": perplexity})
                return loss_value

        return iter_callback

    iter_callback = create_iter_callback(use_pcn)

    # Train using autoregressive trainer
    print(
        "\n[3/5] Training with autoregressive trainer (JIT compilation on first batch)..."
    )
    print(f"Config: {NUM_EPOCHS} epochs, {INFER_STEPS} inference steps, lr={LR}")
    print(
        f"Using {'Predictive Coding' if use_pcn else 'Backpropagation'} training method"
    )

    start_time = time.time()

    ### Choose one of the training methods below
    try:
        if use_pcn:
            # Train with predictive coding (autoregressive)
            trained_params, energy_history, eval_results = train_autoregressive(
                params=params,
                structure=structure,
                train_loader=train_loader,
                config=train_config,
                rng_key=train_key,
                verbose=True,
                epoch_callback=eval_callback,
                iter_callback=iter_callback,
            )
        else:
            # Backprop (autoregressive)
            trained_params, energy_history, eval_results = (
                train_backprop_autoregressive(
                    params,
                    structure,
                    train_loader,
                    {
                        "num_epochs": NUM_EPOCHS,
                        "optimizer": {"type": "adam", "lr": 1e-3},
                        "use_causal_mask": True,
                    },
                    train_key,
                    verbose=True,
                    epoch_callback=eval_callback,
                    iter_callback=iter_callback,
                )
            )
    finally:
        progress_bar.close()

    train_time = time.time() - start_time

    print(
        f"\nTraining completed in {train_time:.1f}s ({train_time/NUM_EPOCHS:.1f}s per epoch)"
    )

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
        train_dataset,
        prompts=prompts,
        max_new_tokens=20,
        rng_key=gen_key,
        temperature=0.8,
        infer_steps=0,  # No inference steps needed for generation
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
    print(
        f"Dataset: TinyShakespeare ({len(full_text):,} total, {len(train_text):,} train, {len(test_text):,} test)"
    )
    print(f"Vocabulary: {train_dataset.vocab_size} unique characters")
    print(
        f"Model: {NUM_BLOCKS} transformer blocks, {EMBED_DIM}d embeddings, {NUM_HEADS} heads"
    )
    print(f"Parameters: {total_params:,}")
    print(f"Training: {NUM_EPOCHS} epochs in {train_time:.1f}s")
    print(f"Final train loss: {energy_history[-1][-1]:.4f}")
    if eval_results and eval_results[-1]:
        final_eval = eval_results[-1]
        print(
            f"Final test loss: {final_eval['loss']:.4f}, Perplexity: {final_eval['perplexity']:.2f}"
        )
    print("=" * 70)

    return trained_params, structure, train_dataset, test_dataset


if __name__ == "__main__":
    main()
