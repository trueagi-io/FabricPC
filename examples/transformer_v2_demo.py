"""
FabricPC Transformer V2 Execution Script

Trains the decomposed PC Transformer model on the Tiny Shakespeare dataset,
evaluates its performance, and generates sample text using temperature sampling.

Architecture (each transformer block decomposed into separate PC nodes)::

    input ──→ Embedding ──→ MhaResidual_0 ──→ LnMlp1_0 ──→ Mlp2Residual_0 ──→ ... ──→ VocabProjection
                                  ↑                              ↑
                             mask │                       (skip from MhaResidual)

    Each block: MhaResidual(in + mask) ──→ LnMlp1 ──→ Mlp2Residual(in + skip)

Supports two training modes:
  - Predictive Coding (PC): Local Hebbian learning with multi-GPU pmap support
  - Backpropagation: Standard end-to-end gradient training (single device)

Usage:
    PYTHONPATH=. python examples/transformer_v2_demo.py
    PYTHONPATH=. python examples/transformer_v2_demo.py --mode backprop --lr 1e-3
    PYTHONPATH=. python examples/transformer_v2_demo.py --mode pc --depth 6 --num_epochs 10
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import jax
import jax.numpy as jnp
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import (
    train_autoregressive,
    evaluate_autoregressive,
    train_backprop_autoregressive,
    evaluate_backprop_autoregressive,
    generate_autoregressive,
)
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.utils.data import CharDataLoader, BpeDataLoader
import optax
import time

BPE_DEFAULTS = {
    "embed_dim": 256, "num_heads": 8, "mlp_dim": 512, "depth": 4,
    "seq_len": 64, "batch_size": 16, "num_epochs": 5, "infer_steps": 30,
    "lr": 4.8336867874408474e-05, "eta_infer": 0.087354491301969,
    "weight_init_std": 0.019440512955251017,
}

CHAR_DEFAULTS = {
    "embed_dim": 64, "num_heads": 4, "mlp_dim": 512, "depth": 3,
    "seq_len": 64, "batch_size": 16, "num_epochs": 5, "infer_steps": 18,
    "lr": 6.710357156410781e-05, "eta_infer": 0.08895631378177452,
    "weight_init_std": 0.043898823650793964,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Decomposed PC Transformer on Tiny Shakespeare"
    )
    parser.add_argument(
        "--mode",
        choices=["pc", "backprop"],
        default="pc",
        help="Training mode: predictive coding or backpropagation (default: pc)",
    )
    parser.add_argument(
        "--depth", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--mlp_dim", type=int, default=256, help="MLP hidden dimension")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (per-device for PC mode, total for backprop)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument(
        "--infer_steps", type=int, default=23, help="PC inference steps"
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=0.06701833916050529,
        help="PC inference step size",
    )
    parser.add_argument(
        "--weight_init_std",
        type=float,
        default=0.013123252658288186,
        help="Weight init std",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char", help="Tokenizer to use",
)
    return parser.parse_args(), parser

def main(args=None):
    if args is None:
        args, parser = parse_args()

    use_pc = args.mode == "pc"

    # --- Batch size ---
    if use_pc:
        n_devices = jax.device_count()
        batch_size = args.batch_size * n_devices
        print(f"PC mode: {n_devices} device(s), total batch_size={batch_size}")
    else:
        batch_size = args.batch_size
        print(f"Backprop mode: single device, batch_size={batch_size}")

    # --- Data ---
    use_bpe = args.tokenizer == "bpe"
    defaults = BPE_DEFAULTS if use_bpe else CHAR_DEFAULTS

    for key, val in defaults.items():
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, val)
            
    if use_bpe:
        train_loader = BpeDataLoader("train", seq_len=args.seq_len, batch_size=batch_size, shuffle=True, seed=args.seed)
        test_loader = BpeDataLoader("test", seq_len=args.seq_len, batch_size=batch_size, shuffle=False)
    else:
        train_loader = CharDataLoader("train", seq_len=args.seq_len, batch_size=batch_size, shuffle=True, seed=args.seed)
        test_loader = CharDataLoader("test", seq_len=args.seq_len, batch_size=batch_size, shuffle=False)

    vocab_size = train_loader.vocab_size
    char_to_ix = train_loader.token_to_idx if use_bpe else train_loader.char_to_idx
    ix_to_char = train_loader.idx_to_token if use_bpe else train_loader.idx_to_char

    # --- Model ---
    structure = create_deep_transformer(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        inference=InferenceSGDNormClip(
            eta_infer=args.eta_infer,
            infer_steps=args.infer_steps,
            max_norm=5.0,
            latent_decay=0.0,
        ),
        weight_init={"type": "normal", "std": args.weight_init_std},
    )

    # --- Train & Evaluate ---
    master_rng_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,}")

    train_config = {
        "num_epochs": args.num_epochs,
        "use_causal_mask": True,
    }
    steps_per_epoch = train_loader.num_sequences // batch_size
    schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=args.num_epochs * steps_per_epoch,
        alpha=0.1,
    )
    optimizer = optax.adam(schedule)

    print(f"Vocab Size: {vocab_size}")
    start = time.time()

    def iter_callback(epoch_idx, batch_idx, energy):
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch {epoch_idx + 1} | Batch {batch_idx + 1} | Energy: {energy:.4f}")
        return energy

    if use_pc:
        trained_params, _, _ = train_autoregressive(
            params, structure, train_loader, optimizer, train_config, train_key, verbose=True, 
            iter_callback=iter_callback,
        )
    else:
        trained_params, _, _ = train_backprop_autoregressive(
            params, structure, train_loader, optimizer, train_config, train_key, verbose=True,
        )

    print(f"Training completed in {time.time() - start:.1f}s")

    # --- Evaluate ---
    eval_start = time.time()
    if use_pc:
        metrics = evaluate_autoregressive(
            trained_params, structure, test_loader, train_config, eval_key
        )
    else:
        metrics = evaluate_backprop_autoregressive(
            trained_params, structure, test_loader, train_config, eval_key
        )
    print(f"Evaluation completed in {time.time() - eval_start:.1f}s")

    print(f"Test Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"Test CE Loss:    {metrics['loss']:.4f}")
    print(f"Test Perplexity: {metrics['perplexity']:.2f}")

    # --- Text Generation ---
    gen_key = jax.random.PRNGKey(99)
    prompt_text = "ROMEO: "
    if use_bpe:
        seed_indices = train_loader._tok.encode(prompt_text).ids
    else:
        seed_indices = [char_to_ix.get(c, 0) for c in prompt_text]

    current_indices = ([0] * (args.seq_len - len(seed_indices)) + seed_indices)[-args.seq_len:]
    prompt = jnp.array(current_indices, dtype=jnp.int32)

    print("--- Generating ---")
    generated = generate_autoregressive(
        trained_params,
        structure,
        prompt,
        max_new_tokens=200,
        rng_key=gen_key,
        temperature=0.8,
    )

    # Decode - skip the prompt padding, decode only from where prompt starts
    generated_ids = generated[args.seq_len:]
    print(prompt_text + train_loader.decode(generated_ids))

if __name__ == "__main__":
    main()
