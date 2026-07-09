"""
ResNet-18 — CIFAR-10 Demo (Predictive Coding)
===============================================

Demonstrates a ResNet-18 architecture built as a predictive coding graph
using muPC scaling on CIFAR-10.

Architecture (CIFAR-10 variant — no 7x7 conv or maxpool):
    input(32,32,3) -> stem(32,32,32, 3x3)
    -> Stage 1: 2 residual blocks (32,32,32)
    -> Stage 2: 2 residual blocks (16,16,64)
    -> Stage 3: 2 residual blocks (8,8,128)
    -> Stage 4: 2 residual blocks (4,4,256)
    -> GlobalAvgPool -> Linear(10, softmax+CE)

Each residual block:
    x -> conv_a(3x3, act) -> conv_b(3x3, act) -> skip(sum)

Skip connections use SkipConnection (same dims) or 1x1 conv (downsample).

Supports multiple activation functions (--activation):
    relu      — baseline, fast but dead neurons during PC inference
    tanh      — bounded, non-zero gradients everywhere, best for PC
    gelu      — smooth middle ground between relu and tanh
    leaky_relu — avoids dead neurons with minimal change from relu

Includes cosine LR schedule with warmup and optional data augmentation
(random horizontal flip + random crop with padding).

Usage:
    python examples/resnet18_cifar10_demo.py --quick                 # 2-epoch smoke test
    python examples/resnet18_cifar10_demo.py --num_epochs 30 --activation tanh --eval_every 5

    Precision (--precision): none (Pi=1, baseline) | diag_probe (frozen per-channel Pi=1/Var
        from a one-pass residual probe) | online (Pi learned each step via EMA -- the NGD scheme).
    Optimizer (--optimizer): adamw (default) | ngd (plain SGD, so precision acts as the
        per-channel adaptive LR). --momentum M (ngd only; ~10x's the effective step, use lower lr).
    --seed S (independent run, for error bars); --probe_latents (per-node latent/error/move RMS).

    # NGD arm (precision as the adaptive LR): online precision + SGD + momentum.
    python examples/resnet18_cifar10_demo.py --precision online --optimizer ngd \\
        --momentum 0.9 --lr 0.02 --activation tanh --num_epochs 30

Multi-seed paired sweeps + significance stats: examples/precision_experiment.py.

Results (CIFAR-10, muPC, tanh; diag_probe vs none Pi=1 baseline, AdamW):
    - Early training (2 epochs): diag_probe CONSISTENTLY beats the baseline across seeds
      (+1.6 / +2.8 / +3.1 pts, lower train energy) -- a convergence-speed / conditioning gain.
    - At convergence (30 epochs, 3 seeds): the gap WASHES OUT -- none 42.03 +/- 0.23% vs
      diag_probe 42.01 +/- 1.15%; paired Delta_acc = -0.02 pts (p=0.98, not significant);
      energy marginally lower for diag_probe. Precision speeds early convergence but does NOT
      change the converged accuracy ceiling.
    - NGD (online precision + plain SGD) trains far slower than AdamW and does not reach the
      AdamW baseline in this setup.
    Precision weighting is a conditioning/robustness lever, not a path to backprop accuracy.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import jax
import jax.numpy as jnp
import numpy as np
import optax
import argparse
import json
import time
from fabricpc.nodes import ConvNode, Linear, IdentityNode, SkipConnection, AvgPool
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    TanhActivation,
    GeluActivation,
    LeakyReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import GaussianEnergy, CrossEntropyEnergy
from fabricpc.core.initializers import (
    NormalInitializer,
    MuPCInitializer,
    XavierInitializer,
)
from fabricpc.core.mupc import MuPCConfig
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.training.train import _convert_batch
from fabricpc.training.ngd_trainer import train_ngd, probe_latent_propagation
from fabricpc.core.precision import probe_residual_precision
from fabricpc.utils.data.dataloader import Cifar10Loader

jax.config.update("jax_default_prng_impl", "threefry2x32")


# =============================================================================
# Activation Factory
# =============================================================================


def get_activation(name):
    factories = {
        "relu": ReLUActivation,
        "tanh": TanhActivation,
        "gelu": GeluActivation,
        "leaky_relu": lambda: LeakyReLUActivation(alpha=0.1),
    }
    if name not in factories:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(factories)}")
    return factories[name]()


# =============================================================================
# Data Augmentation
# =============================================================================


class AugmentedCifar10Loader:
    """Wraps Cifar10Loader with random horizontal flip and random crop+pad."""

    def __init__(self, base_loader, seed=42, pad=4):
        self.base_loader = base_loader
        self.seed = seed
        self.pad = pad
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        pad = self.pad
        for images, labels in self.base_loader:
            flip_mask = rng.random(images.shape[0]) > 0.5
            images[flip_mask] = images[flip_mask, :, ::-1, :]

            padded = np.pad(
                images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect"
            )
            B, H, W, C = images.shape
            crop_y = rng.integers(0, 2 * pad + 1, size=B)
            crop_x = rng.integers(0, 2 * pad + 1, size=B)
            for i in range(B):
                images[i] = padded[
                    i, crop_y[i] : crop_y[i] + H, crop_x[i] : crop_x[i] + W, :
                ]

            yield images, labels

    def __len__(self):
        return len(self.base_loader)


# =============================================================================
# Custom Nodes
# =============================================================================


# AvgPool now lives in fabricpc.nodes.pooling (imported above). Use
# global_pool=True for the (B, H, W, C) -> (B, C) global average pooling here.


# =============================================================================
# ResNet-18 Graph Builder
# =============================================================================


def _gauss_energy(name, precision_map):
    """GaussianEnergy for `name`, with diagonal precision from the probe if present."""
    if precision_map is not None and name in precision_map:
        return GaussianEnergy(precision=precision_map[name])
    return GaussianEnergy()


def make_residual_block(
    prev_node,
    channels,
    stride,
    block_name,
    weight_init,
    activation=None,
    precision_map=None,
):
    """
    Create one residual block: conv_a -> conv_b(act) -> skip(sum).

    Activation is applied on the main path before summation. The skip path
    passes through without activation, preserving gradient flow.

    Returns:
        (nodes_list, edges_list, skip_node) where skip_node is the block output.
    """
    if activation is None:
        activation = ReLUActivation()

    in_h, in_w, in_channels = prev_node._shape

    if stride == 1:
        out_h, out_w = in_h, in_w
    else:
        out_h, out_w = in_h // stride, in_w // stride

    nodes = []
    edges = []

    conv_a = ConvNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        energy=_gauss_energy(f"{block_name}_conv_a", precision_map),
        name=f"{block_name}_conv_a",
    )

    conv_b = ConvNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        energy=_gauss_energy(f"{block_name}_conv_b", precision_map),
        name=f"{block_name}_conv_b",
    )

    skip_node = SkipConnection(
        shape=(out_h, out_w, channels),
        energy=_gauss_energy(f"{block_name}_skip_sum", precision_map),
        name=f"{block_name}_skip_sum",
    )

    nodes.extend([conv_a, conv_b, skip_node])

    # Main path edges
    edges.append(Edge(source=prev_node, target=conv_a.slot("in")))
    edges.append(Edge(source=conv_a, target=conv_b.slot("in")))
    edges.append(Edge(source=conv_b, target=skip_node.slot("in")))

    # Skip connection
    needs_downsample = (stride != 1) or (in_channels != channels)
    if needs_downsample:
        conv_skip = ConvNode(
            shape=(out_h, out_w, channels),
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="SAME",
            activation=IdentityActivation(),
            weight_init=weight_init,
            energy=_gauss_energy(f"{block_name}_skip", precision_map),
            name=f"{block_name}_skip",
        )
        nodes.append(conv_skip)
        edges.append(Edge(source=prev_node, target=conv_skip.slot("in")))
        edges.append(Edge(source=conv_skip, target=skip_node.slot("in")))
    else:
        edges.append(Edge(source=prev_node, target=skip_node.slot("in")))

    return nodes, edges, skip_node


def build_resnet18(
    weight_init,
    scaling=None,
    output_weight_init=None,
    activation=None,
    precision_map=None,
    *,
    infer_steps,
    eta_infer,
):
    """
    Build ResNet-18 for CIFAR-10 as a predictive coding graph.

    Args:
        weight_init: InitializerBase for conv/linear weights.
        scaling: Optional MuPCConfig for muPC parameterization.
        output_weight_init: Optional InitializerBase for the output layer.
        activation: Activation for hidden conv layers (default: ReLU).
        infer_steps: Number of PC inference steps.
        eta_infer: Inference rate.

    Returns:
        GraphStructure ready for initialize_params().
    """
    if output_weight_init is None:
        output_weight_init = XavierInitializer()
    if activation is None:
        activation = ReLUActivation()

    # Input
    input_node = IdentityNode(shape=(32, 32, 3), name="input")

    # Stem convolution: 3x3, 32 channels, no maxpool (CIFAR is 32x32)
    stem = ConvNode(
        shape=(32, 32, 32),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        energy=_gauss_energy("stem", precision_map),
        name="stem",
    )

    all_nodes = [input_node, stem]
    all_edges = [Edge(source=input_node, target=stem.slot("in"))]

    # Build 4 stages with [2, 2, 2, 2] blocks
    stage_configs = [
        (32, 1, 2),  # (channels, first_stride, num_blocks)
        (64, 2, 2),
        (128, 2, 2),
        (256, 2, 2),
    ]

    prev = stem
    for stage_idx, (channels, first_stride, num_blocks) in enumerate(stage_configs, 1):
        for block_idx in range(num_blocks):
            stride = first_stride if block_idx == 0 else 1
            block_name = f"s{stage_idx}b{block_idx + 1}"

            nodes, edges, add_node = make_residual_block(
                prev_node=prev,
                channels=channels,
                stride=stride,
                block_name=block_name,
                weight_init=weight_init,
                activation=activation,
                precision_map=precision_map,
            )
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            prev = add_node

    # Global average pooling: (B, 4, 4, 256) -> (B, 256)
    avg_pool = AvgPool(
        shape=(256,),
        name="avgpool",
        global_pool=True,
        energy=_gauss_energy("avgpool", precision_map),
    )
    all_nodes.append(avg_pool)
    all_edges.append(Edge(source=prev, target=avg_pool.slot("in")))

    # Output: Linear(10) with softmax + cross-entropy
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=output_weight_init,
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=avg_pool, target=output.slot("in")))

    # Build graph
    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGDNormClip(
            eta_infer=eta_infer, infer_steps=infer_steps, max_norm=1.0
        ),
        scaling=scaling,
    )

    return structure


# =============================================================================
# Model Factory
# =============================================================================


def _create_mupc_model(
    rng_key,
    *,
    infer_steps,
    eta_infer,
    activation=None,
    precision_mode="none",
    probe_batch=None,
    probe_key=None,
):
    """Create ResNet-18 with muPC parameterization.

    precision_mode: "none" (Pi=1) | "diag_probe" (frozen per-channel Pi=1/Var from one
    clamped inference pass) | "online" (built as none; Pi learned in the NGD trainer).
    """
    build = lambda precision_map=None: build_resnet18(
        weight_init=MuPCInitializer(),
        scaling=MuPCConfig(include_output=False),
        output_weight_init=XavierInitializer(),
        activation=activation,
        precision_map=precision_map,
        infer_steps=infer_steps,
        eta_infer=eta_infer,
    )
    structure = build()
    params = initialize_params(structure, rng_key)
    if precision_mode == "diag_probe":
        if probe_batch is None or probe_key is None:
            raise ValueError("diag_probe requires probe_batch and probe_key")
        precision_map = probe_residual_precision(
            params, structure, probe_batch, probe_key
        )
        structure = build(precision_map)
        print(
            f"Precision probe: diagonal Pi estimated for {len(precision_map)} "
            f"Gaussian-energy nodes (per-channel 1/Var, mean-normalized)."
        )
    return params, structure


# =============================================================================
# Training
# =============================================================================


def run_single_mupc(args):
    """Default mode: single muPC training run with progress bar."""
    activation = get_activation(args.activation)

    print("=" * 60)
    print("ResNet-18 on CIFAR-10 (Predictive Coding + muPC)")
    print("=" * 60)
    print(
        f"Activation: {args.activation}  |  Epochs: {args.num_epochs}  |  "
        f"LR: {args.lr}  |  Augment: {not args.no_augment}"
    )

    master_rng_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key, probe_key = jax.random.split(master_rng_key, 4)

    # Data (built before the model so the precision probe can use a real batch); the
    # train-shuffle seed is tied to --seed so each seed is a fully independent run.
    base_train_loader = Cifar10Loader(
        "train", batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    if args.no_augment:
        train_loader = base_train_loader
    else:
        train_loader = AugmentedCifar10Loader(base_train_loader, seed=args.seed)
    test_loader = Cifar10Loader("test", batch_size=args.batch_size, shuffle=False)

    probe_batch = None
    if args.precision == "diag_probe":
        probe_batch = _convert_batch(next(iter(train_loader)))

    # Build model
    print(f"Precision weighting: {args.precision}")
    params, structure = _create_mupc_model(
        graph_key,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
        activation=activation,
        precision_mode=args.precision,
        probe_batch=probe_batch,
        probe_key=probe_key,
    )

    # Print model summary
    print(f"\nModel: {len(structure.nodes)} nodes, {len(structure.edges)} edges")

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Cosine LR schedule with warmup
    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    if args.optimizer == "ngd":
        # plain SGD so the per-channel precision acts as the adaptive LR; momentum (0.0 ->
        # None = plain SGD) makes the narrow stable-LR regime trainable.
        optimizer = optax.sgd(schedule, momentum=(args.momentum or None))
        if args.momentum:
            print(f"  ngd: SGD with momentum={args.momentum}")
    else:
        optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    print(f"Optimizer: {args.optimizer}")
    train_config = {"num_epochs": args.num_epochs}

    # Periodic evaluation callback
    eval_every = args.eval_every

    def epoch_callback(epoch_idx, params, structure, config, rng_key):
        epoch_num = epoch_idx + 1
        if eval_every > 0 and (
            epoch_num % eval_every == 0 or epoch_num == args.num_epochs
        ):
            metrics = evaluate_pcn(
                params, structure, test_loader, train_config, eval_key
            )
            print(f"  Epoch {epoch_num}: accuracy={metrics['accuracy'] * 100:.2f}%")
            return metrics
        return None

    print(
        f"\nTraining for {args.num_epochs} epochs "
        f"(JIT compilation on first batch)..."
    )
    start_time = time.time()

    learned_precision = None
    if args.precision == "online":
        trained_params, learned_precision, energy_history = train_ngd(
            params=params,
            structure=structure,
            train_loader=train_loader,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            rng_key=train_key,
            lam=args.precision_lam,
            epoch_callback=epoch_callback,
            verbose=False,
        )
    else:
        trained_params, energy_history, _ = train_pcn(
            params=params,
            structure=structure,
            train_loader=train_loader,
            optimizer=optimizer,
            config=train_config,
            rng_key=train_key,
            verbose=False,
            epoch_callback=epoch_callback,
        )

    elapsed = time.time() - start_time
    print(
        f"\nTraining time: {elapsed:.1f}s ({elapsed / args.num_epochs:.1f}s per epoch)"
    )

    # Final evaluation
    print("Final evaluation...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

    if args.probe_latents:
        probe_b = _convert_batch(next(iter(test_loader)))
        report = probe_latent_propagation(
            trained_params,
            structure,
            probe_b,
            eval_key,
            precision_map=learned_precision,
        )
        print("\nLatent propagation (per node, topological order):")
        print(
            f"  {'node':<18}{'in_deg':>7}{'z_rms':>10}{'err_rms':>10}{'move_rms':>10}"
        )
        for name, m in report.items():
            print(
                f"  {name:<18}{m['in_degree']:>7}{m['z_rms']:>10.4f}"
                f"{m['err_rms']:>10.4f}{m['move_rms']:>10.4f}"
            )

    final_energy = float("nan")
    if energy_history and energy_history[-1]:
        last_epoch = energy_history[-1]
        final_energy = float(sum(last_epoch) / len(last_epoch))
    result = {
        "seed": args.seed,
        "precision": args.precision,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "activation": args.activation,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "accuracy": float(metrics["accuracy"]),
        "final_train_energy": final_energy,
        "train_time_s": float(elapsed),
    }
    print("[RESULT] " + json.dumps(result))
    return result


# =============================================================================
# CLI and Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResNet-18 on CIFAR-10 (Predictive Coding)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--infer_steps", type=int, default=80, help="Inference steps (default: 80)"
    )
    parser.add_argument(
        "--eta_infer", type=float, default=0.1, help="Inference rate (default: 0.1)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
        help="Activation function for hidden layers (default: relu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="none",
        choices=["none", "diag_probe", "online"],
        help="Precision weighting: none (Pi=1) | diag_probe (frozen per-channel 1/Var) | "
        "online (per-step EMA, the NGD scheme). Default none",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "ngd"],
        help="adamw (default) | ngd (plain SGD; precision acts as the adaptive LR)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="SGD momentum for --optimizer ngd (0.0=plain SGD; try 0.9). Ignored for adamw.",
    )
    parser.add_argument(
        "--precision_lam",
        type=float,
        default=0.05,
        help="EMA rate for online precision (only with --precision online; default 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for model init AND train-shuffle order (default: 42)",
    )
    parser.add_argument(
        "--probe_latents",
        action="store_true",
        help="After training, print a per-node latent-propagation table.",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable data augmentation (random crop + horizontal flip)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="Evaluate on test set every N epochs (0 to disable; default: 10)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 2 epochs, no augmentation",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch output")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 <= args.momentum < 1.0):
        raise SystemExit(f"--momentum must be in [0.0, 1.0); got {args.momentum}")
    if args.quick:
        args.num_epochs = 2
        args.no_augment = True
        args.eval_every = 0
    run_single_mupc(args)


if __name__ == "__main__":
    main()
