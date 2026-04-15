"""
Transformer Predictive Coding Demo

Character-level language modeling on TinyShakespeare with PC or backprop training.
PC training is not yet tuned — treat as a starting point for experimentation.
"""

use_pcn = True  # Set to True to use predictive coding training, False for backprop
use_extra_skip_connections = True  # Add extra skip connections from embedding to all transformer blocks (can help PC inference convergence)

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import math
import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Tuple, Dict, List, Optional, Any
from tqdm.auto import tqdm

from fabricpc.nodes import Linear, TransformerBlock, IdentityNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params, FeedforwardStateInit
from fabricpc.core.activations import (
    IdentityActivation,
    SoftmaxActivation,
    GeluActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import (
    NormalInitializer,
    KaimingInitializer,
    XavierInitializer,
)
from fabricpc.core.inference import InferenceSGDNormClip, InferenceSGD
import optax
from fabricpc.training.train_autoregressive import (
    train_step_autoregressive,
    generate_autoregressive,
    evaluate_autoregressive,
    create_causal_mask,
)
from fabricpc.graph import initialize_graph_state
from fabricpc.utils.dashboarding.inference_tracking import (
    run_inference_with_full_history,
)
from fabricpc.training.train_backprop import (
    train_step_backprop_autoregressive,
    evaluate_backprop_autoregressive,
)
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    is_aim_available,
)
from fabricpc.utils.data import CharDataLoader

jax.config.update("jax_default_prng_impl", "threefry2x32")

TRACKED_NODES = ["embed", "transformer_0"]


# --- Model Configuration ---


def create_transformer_model(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    ff_dim: int,
    rope_theta: float,
    rng_key: jax.Array,
    infer_steps: int = 10,
    eta_infer: float = 0.01,
) -> Tuple:
    """Create a transformer language model. Returns (structure, params)."""
    input_node = IdentityNode(shape=(seq_len, vocab_size), name="input")
    embed = Linear(
        shape=(seq_len, embed_dim),
        activation=IdentityActivation(),
        weight_init=NormalInitializer(std=1.0 / jnp.sqrt(vocab_size)),
        name="embed",
    )
    mask_node = IdentityNode(shape=(1, seq_len, seq_len), name="mask")

    nodes = [input_node, embed, mask_node]
    edges = [Edge(source=input_node, target=embed.slot("in"))]

    xmfr_blocks = []
    block_skip_nodes = []
    summing_nodes = []
    for i in range(num_blocks):
        xmfr_blocks.append(
            TransformerBlock(
                shape=(seq_len, embed_dim),
                num_heads=num_heads,
                ff_dim=ff_dim,
                internal_activation=GeluActivation(),
                rope_theta=rope_theta,
                name=f"transformer_{i}",
            )
        )
        block_skip_nodes.append(
            IdentityNode(
                shape=(seq_len, embed_dim),
                name=f"block_skip_{i}",
                scale=(1.0 / num_blocks),
            )
        )
        summing_nodes.append(
            IdentityNode(
                shape=(seq_len, embed_dim),
                name=f"summing_skip_{i}",
                scale=(0.1 / (1 + num_blocks)),
            )
        )
    nodes = nodes + xmfr_blocks
    if use_extra_skip_connections:
        nodes = nodes + summing_nodes  # + block_skip_nodes

    prev_node = embed
    for i in range(num_blocks):
        edges.append(Edge(source=prev_node, target=xmfr_blocks[i].slot("in")))
        edges.append(Edge(source=mask_node, target=xmfr_blocks[i].slot("mask")))

        if use_extra_skip_connections:
            edges.append(
                Edge(source=xmfr_blocks[i], target=summing_nodes[i].slot("in"))
            )
            for j in range(i, num_blocks):
                edges.append(Edge(source=prev_node, target=summing_nodes[j].slot("in")))
            prev_node = summing_nodes[i]
        else:
            prev_node = xmfr_blocks[i]

    output_node = Linear(
        shape=(seq_len, vocab_size),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=NormalInitializer(std=1.0 / jnp.sqrt(embed_dim)),
        name="output",
    )
    nodes.append(output_node)
    edges.append(Edge(source=prev_node, target=output_node.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=output_node, causal_mask=mask_node),
        graph_state_initializer=FeedforwardStateInit(),
        inference=InferenceSGDNormClip(
            eta_infer=eta_infer, infer_steps=infer_steps, max_norm=0.5, latent_decay=0.0
        ),
    )
    params = initialize_params(structure, rng_key)
    return structure, params


# --- Text Generation ---


def generate_text(
    params,
    structure,
    dataset: CharDataLoader,
    prompts: List[str],
    max_new_tokens: int = 100,
    rng_key: jax.Array = None,
    temperature: float = 0.8,
    top_k: int = None,
    top_p: float = None,
) -> List[str]:
    """Generate text autoregressively from batched prompts."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    seq_len = structure.nodes["input"].node_info.shape[0]
    pad_char = dataset.char_to_idx.get(" ", 0)

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

    prompt_tokens = jnp.array(batch_indices)  # (batch_size, seq_len)

    generated_tokens = generate_autoregressive(
        params=params,
        structure=structure,
        prompt=prompt_tokens,
        max_new_tokens=max_new_tokens,
        rng_key=rng_key,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    generated_texts = []
    for i, prompt in enumerate(prompts):
        tokens = np.array(generated_tokens[i])

        pad_len = seq_len - len(prompt)
        if pad_len > 0:
            tokens = tokens[pad_len:]

        text = dataset.decode(tokens)
        generated_texts.append(text)

    return generated_texts


# --- Main Experiment ---


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
    SEQ_LEN = 128
    EMBED_DIM = 128
    NUM_HEADS = 8
    NUM_BLOCKS = 1
    FF_DIM = 512
    ROPE_THETA = 500.0
    BATCH_SIZE = 128
    NUM_EPOCHS = 1.0
    INFER_STEPS = 11
    ETA_INFER = 0.05
    LR = 1e-3

    master_key = jax.random.PRNGKey(42)
    graph_key, train_key, gen_key = jax.random.split(master_key, 3)

    # Data
    train_loader = CharDataLoader(
        "train", seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=True, seed=0
    )
    test_loader = CharDataLoader(
        "test", seq_len=SEQ_LEN, batch_size=BATCH_SIZE, shuffle=False
    )

    # Linear embedding requires one-hot x; wrap loaders accordingly.
    vocab_size = train_loader.vocab_size
    eye = np.eye(vocab_size, dtype=np.float32)

    class _OneHotLoader:
        """Thin wrapper that one-hot encodes x from CharDataLoader."""

        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __iter__(self):
            for x_idx, y_oh in self.base:
                yield {"x": eye[x_idx], "y": y_oh}

    train_loader_oh = _OneHotLoader(train_loader)
    test_loader_oh = _OneHotLoader(test_loader)

    print(
        f"Vocab: {vocab_size}, Train batches: {len(train_loader)}, Test batches: {len(test_loader)}"
    )

    # Model
    structure, params = create_transformer_model(
        vocab_size=train_loader.vocab_size,
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        ff_dim=FF_DIM,
        rope_theta=ROPE_THETA,
        rng_key=graph_key,
        infer_steps=INFER_STEPS,
        eta_infer=ETA_INFER,
    )

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model created: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    print(f"Total parameters: {total_params:,}")

    # Aim tracking (optional)
    if is_aim_available():
        tracking_config = TrackingConfig(
            experiment_name="transformer_pc_shakespeare",
            run_name=f"{'PC' if use_pcn else 'BP'}_{NUM_BLOCKS}blk_{EMBED_DIM}d",
            track_energy=True,
            track_weight_distributions=True,
            track_state_distributions=True,
            nodes_to_track=TRACKED_NODES,
            tracking_every_n_batches=50,
            state_tracking_every_n_infer_steps=5,
        )
        tracker = AimExperimentTracker(config=tracking_config)
        tracker.log_hyperparams(
            {
                "model_config": {
                    "seq_len": SEQ_LEN,
                    "embed_dim": EMBED_DIM,
                    "num_heads": NUM_HEADS,
                    "num_blocks": NUM_BLOCKS,
                    "ff_dim": FF_DIM,
                    "rope_theta": ROPE_THETA,
                    "total_params": total_params,
                    "use_extra_skip_connections": use_extra_skip_connections,
                },
                "training_method": "PC" if use_pcn else "Backprop",
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "infer_steps": INFER_STEPS,
                "eta_infer": ETA_INFER,
                "lr": LR,
            }
        )
        tracker.log_graph_structure(structure)
    else:
        tracker = None

    # Training
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(LR, weight_decay=0.1),
    )
    train_config = {
        "num_epochs": NUM_EPOCHS,
        "use_causal_mask": True,  # Enable causal masking for autoregressive
    }

    def create_eval_callback(use_pc: bool):
        """Create appropriate eval callback based on training method."""
        if use_pc:

            def eval_callback(epoch_idx, params, structure, config, rng_key):
                eval_rng = jax.random.fold_in(rng_key, epoch_idx)
                metrics = evaluate_autoregressive(
                    params,
                    structure,
                    test_loader_oh,
                    {
                        "use_causal_mask": True,
                    },  # No inference steps for eval because model predicts feedforward
                    eval_rng,
                    debug=(epoch_idx == 0),
                )
                tqdm.write(
                    f"  Test - Loss: {metrics['loss']:.4f}, Perplexity: {metrics['perplexity']:.2f}, Acc: {metrics['accuracy']:.4f}"
                )
                return metrics

        else:

            def eval_callback(epoch_idx, params, structure, config, rng_key):
                eval_rng = jax.random.fold_in(rng_key, epoch_idx)
                metrics = evaluate_backprop_autoregressive(
                    params,
                    structure,
                    test_loader_oh,
                    {"use_causal_mask": True},
                    eval_rng,
                    debug=(epoch_idx == 0),
                )
                tqdm.write(
                    f"  Test - Loss: {metrics['loss']:.4f}, Perplexity: {metrics['perplexity']:.2f}, Acc: {metrics['accuracy']:.4f}"
                )
                return metrics

        return eval_callback

    eval_callback = create_eval_callback(use_pcn)
    progress_bar = TrainingProgressBar(
        total_batches=len(train_loader_oh),
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

    print(
        f"\nTraining ({'PC' if use_pcn else 'Backprop'}, {NUM_EPOCHS} epochs, lr={LR})..."
    )

    start_time = time.time()

    opt_state = optimizer.init(params)

    num_epochs = train_config["num_epochs"]
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)
    use_causal_mask = train_config.get("use_causal_mask", True)

    if use_pcn:
        jit_train_step = jax.jit(
            lambda p, o, b, k: train_step_autoregressive(
                p,
                o,
                b,
                structure,
                optimizer,
                k,
                use_causal_mask,
            )
        )
    else:
        jit_train_step = jax.jit(
            lambda p, o, b, k: train_step_backprop_autoregressive(
                p, o, b, structure, optimizer, k, use_causal_mask
            )
        )

    energy_history = []
    eval_results = []

    try:
        for epoch in range(total_epochs):
            num_batches = len(train_loader_oh)
            is_last = epoch == total_epochs - 1
            max_batches = (
                round(frac * num_batches) if (is_last and frac > 0) else num_batches
            )

            epoch_rng, train_key = jax.random.split(train_key)
            batch_keys = jax.random.split(epoch_rng, max_batches)

            batch_energies = []
            for batch_idx, batch_data in enumerate(train_loader_oh):
                if batch_idx >= max_batches:
                    break

                batch = {k: jnp.array(v) for k, v in batch_data.items()}

                if use_pcn:
                    params, opt_state, energy, ce_loss, final_state = jit_train_step(
                        params, opt_state, batch, batch_keys[batch_idx]
                    )
                    loss_val = float(energy)
                else:
                    params, opt_state, loss, _predictions = jit_train_step(
                        params, opt_state, batch, batch_keys[batch_idx]
                    )
                    loss_val = float(loss)
                    final_state = None

                iter_callback(epoch, batch_idx, loss_val)
                batch_energies.append(loss_val)

                if tracker is not None:
                    tracker.track_batch_energy(loss_val, epoch=epoch, batch=batch_idx)
                    tracker.track_weight_distributions(
                        params,
                        structure,
                        epoch=epoch,
                        batch=batch_idx,
                        nodes=TRACKED_NODES,
                    )

                    should_track_state = (
                        final_state is not None
                        and batch_idx % tracker.config.tracking_every_n_batches == 0
                    )
                    if should_track_state:
                        track_clamps = {}
                        for task_name, task_value in batch.items():
                            if task_name in structure.task_map:
                                track_clamps[structure.task_map[task_name]] = task_value
                        if use_causal_mask:
                            seq_len = batch["x"].shape[1]
                            cm = create_causal_mask(seq_len)[None, None, :, :]
                            cm = jnp.broadcast_to(
                                cm, (batch["x"].shape[0], 1, seq_len, seq_len)
                            )
                            track_clamps[structure.task_map["causal_mask"]] = cm

                        track_init_state = initialize_graph_state(
                            structure,
                            batch["x"].shape[0],
                            batch_keys[batch_idx],
                            clamps=track_clamps,
                            params=params,
                        )
                        _, state_history = run_inference_with_full_history(
                            params, track_init_state, track_clamps, structure
                        )
                        for infer_step_idx, step_state in enumerate(state_history):
                            tracker.track_state(
                                step_state,
                                epoch=epoch,
                                batch=batch_idx,
                                infer_step=infer_step_idx,
                                nodes=TRACKED_NODES,
                            )

            energy_history.append(batch_energies)

            eval_results.append(
                eval_callback(epoch, params, structure, train_config, train_key)
            )

            if batch_energies:
                avg_loss = sum(batch_energies) / len(batch_energies)
                tqdm.write(
                    f"  Train Epoch {epoch + 1}/{total_epochs}, Avg loss: {avg_loss:.4f}"
                )
    finally:
        progress_bar.close()

    trained_params = params
    train_time = time.time() - start_time

    print(
        f"\nTraining completed in {train_time:.1f}s ({train_time/NUM_EPOCHS:.1f}s per epoch)"
    )

    # Generate samples
    prompts = [
        "Know, Rome, that",
        "MENENIUS:",
        "the more virtuous",
        "by his looks",
        "ROMEO: ",
        "To be or not to be",
        "The king",
    ]

    generated_texts = generate_text(
        trained_params,
        structure,
        train_loader,
        prompts=prompts,
        max_new_tokens=20,
        rng_key=gen_key,
        temperature=0.8,
    )

    for prompt, generated in zip(prompts, generated_texts):
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        print(generated)
        print("-" * 40)

    if tracker is not None:
        tracker.close()

    # Results
    print(f"\nFinal train loss: {energy_history[-1][-1]:.4f}")
    if eval_results and eval_results[-1]:
        final_eval = eval_results[-1]
        print(
            f"Final test loss: {final_eval['loss']:.4f}, Perplexity: {final_eval['perplexity']:.2f}"
        )

    return trained_params, structure, train_loader, test_loader


if __name__ == "__main__":
    main()
