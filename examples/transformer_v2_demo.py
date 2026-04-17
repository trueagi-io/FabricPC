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
from fabricpc.graph import initialize_params
from fabricpc.training import (
    train_pcn,
    evaluate_transformer,
    train_backprop,
    evaluate_backprop,
)
from fabricpc.core.inference import run_inference, InferenceSGD
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.utils.data import CharDataLoader
import optax
import time


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
    parser.add_argument("--mlp_dim", type=int, default=128, help="MLP hidden dimension")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (per-device for PC mode, total for backprop)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--infer_steps", type=int, default=17, help="PC inference steps"
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=0.033195052120243505,
        help="PC inference step size",
    )
    parser.add_argument(
        "--weight_init_std",
        type=float,
        default=0.04402197307582635,
        help="Weight init std",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def generate(
    trained_params,
    structure,
    char_to_ix,
    ix_to_char,
    seq_len,
    start_text="ROMEO: ",
    length=50,
    temperature=0.8,
    use_inference=True,
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

        if use_inference:
            final_state = run_inference(trained_params, state, inputs, structure)
        else:
            final_state = state

        logits_node_state = final_state.nodes["logits"]
        last_step_logits = logits_node_state.z_latent[0, -1, :]

        gen_key, sample_key = jax.random.split(gen_key)
        scaled_logits = last_step_logits / temperature
        next_idx = int(jax.random.categorical(sample_key, scaled_logits))

        next_char = ix_to_char[next_idx]
        result_text += next_char
        current_indices = current_indices[1:] + [next_idx]

    print(result_text)


def main(args=None):
    if args is None:
        args = parse_args()

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
    train_loader = CharDataLoader(
        "train",
        seq_len=args.seq_len,
        batch_size=batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = CharDataLoader(
        "validation", seq_len=args.seq_len, batch_size=batch_size, shuffle=False
    )
    test_loader = CharDataLoader(
        "test", seq_len=args.seq_len, batch_size=batch_size, shuffle=False
    )
    vocab_size = train_loader.vocab_size
    char_to_ix = train_loader.char_to_idx
    ix_to_char = train_loader.idx_to_char

    # --- Model ---
    structure = create_deep_transformer(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        inference=InferenceSGD(eta_infer=args.eta_infer, infer_steps=args.infer_steps),
        weight_init={"type": "normal", "std": args.weight_init_std},
    )

    # --- Train & Evaluate ---
    master_rng_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)

    train_config = {
        "num_epochs": args.num_epochs,
    }
    optimizer = optax.adam(args.lr)

    print(f"Vocab Size: {vocab_size}")
    start = time.time()

    if use_pc:
        trained_params, _, _ = train_pcn(
            params,
            structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=True,
        )
    else:
        trained_params, _, _ = train_backprop(
            params,
            structure,
            train_loader,
            optimizer,
            train_config,
            train_key,
            verbose=True,
        )

    print(f"Training completed in {time.time() - start:.1f}s")

    # --- Evaluate ---
    if use_pc:
        metrics = evaluate_transformer(
            trained_params, structure, test_loader, train_config, eval_key
        )
        print(f"Test Accuracy:   {metrics['accuracy'] * 100:.2f}%")
        print(f"Test CE Loss:    {metrics['cross_entropy']:.4f}")
        print(f"Test Perplexity: {metrics['perplexity']:.2f}")
        print(f"Test Energy:     {metrics['energy']:.4f}")
    else:
        metrics = evaluate_backprop(
            trained_params, structure, test_loader, train_config, eval_key
        )
        print(f"Test Accuracy:   {metrics['accuracy'] * 100:.2f}%")
        print(f"Test CE Loss:    {metrics['loss']:.4f}")
        print(f"Test Perplexity: {metrics['perplexity']:.2f}")

    # --- Text Generation ---
    generate(
        trained_params,
        structure,
        char_to_ix,
        ix_to_char,
        args.seq_len,
        start_text="ROMEO: ",
        length=100,
        temperature=0.8,
        use_inference=use_pc,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
