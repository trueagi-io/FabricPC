"""
ResNet-18 — CIFAR-10 Demo (Predictive Coding)
===============================================

Demonstrates a ResNet-18 architecture built as a predictive coding graph
using muPC scaling on CIFAR-10.

Architecture (CIFAR-10 variant — no 7x7 conv or maxpool):
    input(32,32,3) -> stem(32,32,16, 3x3)
    -> Stage 1: 2 residual blocks (32,32,16)
    -> Stage 2: 2 residual blocks (16,16,32)
    -> Stage 3: 2 residual blocks (8,8,64)
    -> Stage 4: 2 residual blocks (4,4,128)
    -> GlobalAvgPool -> Linear(10, softmax+CE)

Each residual block:
    x -> conv_a(3x3, ReLU) -> conv_b(3x3, Identity) -> add(+skip, ReLU)

Skip connections use IdentityNode (same dims) or 1x1 conv (downsample).

Usage:
    python examples/resnet18_cifar10_demo.py
    python examples/resnet18_cifar10_demo.py --num_epochs 10 --verbose
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import time

from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax

from custom_node import Conv2DNode
from fabricpc.nodes import Linear, IdentityNode
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import GaussianEnergy, CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import (
    NormalInitializer,
    MuPCInitializer,
    XavierInitializer,
)
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import Cifar10Loader

jax.config.update("jax_default_prng_impl", "threefry2x32")


# =============================================================================
# Custom Nodes
# =============================================================================


class AddActivationNode(NodeBase):
    """
    Summation node with activation: sums all inputs, applies activation.

    Used for residual addition + ReLU in ResNet blocks. Like IdentityNode
    but applies the configured activation function to the sum.
    """

    def __init__(
        self,
        shape,
        name,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape, config):
        return 1  # Weightless: a = gain/sqrt(K)

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(params, inputs, state, node_info):
        # Sum all inputs
        pre_activation = None
        for edge_key, x in inputs.items():
            if pre_activation is None:
                pre_activation = x
            else:
                pre_activation = pre_activation + x

        # Apply activation (e.g. ReLU)
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        return total_energy, state


class AvgPoolNode(NodeBase):
    """
    Global average pooling: (B, H, W, C) -> (B, C).

    Averages over spatial dimensions. No learnable parameters.
    """

    def __init__(
        self,
        shape,
        name,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape, config):
        return 1

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(params, inputs, state, node_info):
        # Sum all inputs (typically just one)
        spatial = None
        for edge_key, x in inputs.items():
            if spatial is None:
                spatial = x
            else:
                spatial = spatial + x

        # Global average pooling: (B, H, W, C) -> (B, C)
        z_mu = jnp.mean(spatial, axis=(1, 2))

        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        return total_energy, state


# =============================================================================
# ResNet-18 Graph Builder
# =============================================================================


def make_residual_block(
    prev_node,
    channels,
    stride,
    block_name,
    weight_init,
):
    """
    Create one residual block: conv_a -> conv_b -> add(+skip) -> ReLU.

    Args:
        prev_node: The node feeding into this block.
        channels: Output channels for this block.
        stride: Stride for the first conv (1=identity, 2=downsample).
        block_name: Name prefix (e.g., "s1b1" for stage 1, block 1).
        weight_init: Weight initializer for conv layers.

    Returns:
        (nodes_list, edges_list, add_node) where add_node is the block output.
    """
    in_h, in_w, in_channels = prev_node._shape

    if stride == 1:
        out_h, out_w = in_h, in_w
    else:
        out_h, out_w = in_h // stride, in_w // stride

    nodes = []
    edges = []

    # Main path: conv_a (3x3, stride, ReLU) -> conv_b (3x3, stride=1, Identity)
    conv_a = Conv2DNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=weight_init,
        name=f"{block_name}_conv_a",
    )

    conv_b = Conv2DNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=IdentityActivation(),
        weight_init=weight_init,
        name=f"{block_name}_conv_b",
    )

    # Addition node with ReLU
    add_node = AddActivationNode(
        shape=(out_h, out_w, channels),
        activation=ReLUActivation(),
        name=f"{block_name}_add",
    )

    nodes.extend([conv_a, conv_b, add_node])

    # Main path edges
    edges.append(Edge(source=prev_node, target=conv_a.slot("in")))
    edges.append(Edge(source=conv_a, target=conv_b.slot("in")))
    edges.append(Edge(source=conv_b, target=add_node.slot("in")))

    # Skip connection
    needs_downsample = (stride != 1) or (in_channels != channels)
    if needs_downsample:
        # 1x1 conv to match dimensions
        conv_skip = Conv2DNode(
            shape=(out_h, out_w, channels),
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="SAME",
            activation=IdentityActivation(),
            weight_init=weight_init,
            name=f"{block_name}_skip",
        )
        nodes.append(conv_skip)
        edges.append(Edge(source=prev_node, target=conv_skip.slot("in")))
        edges.append(Edge(source=conv_skip, target=add_node.slot("in")))
    else:
        # Direct skip connection (identity)
        edges.append(Edge(source=prev_node, target=add_node.slot("in")))

    return nodes, edges, add_node


def build_resnet18(
    weight_init, scaling=None, output_weight_init=None, *, infer_steps, eta_infer
):
    """
    Build ResNet-18 for CIFAR-10 as a predictive coding graph.

    Architecture:
        input(32,32,3) -> stem(32,32,16)
        -> Stage 1: 2x ResBlock(16)
        -> Stage 2: 2x ResBlock(32, stride=2 on first)
        -> Stage 3: 2x ResBlock(64, stride=2 on first)
        -> Stage 4: 2x ResBlock(128, stride=2 on first)
        -> AvgPool -> Linear(10, softmax+CE)

    Args:
        weight_init: InitializerBase for conv/linear weights.
        scaling: Optional MuPCConfig for muPC parameterization.
        output_weight_init: Optional InitializerBase for the output layer.
            Defaults to XavierInitializer.
        infer_steps: Number of PC inference steps.
        eta_infer: Inference rate.

    Returns:
        GraphStructure ready for initialize_params().
    """
    if output_weight_init is None:
        output_weight_init = XavierInitializer()

    # Input
    input_node = IdentityNode(shape=(32, 32, 3), name="input")

    # Stem convolution: 3x3, 16 channels, no maxpool (CIFAR is 32x32)
    stem = Conv2DNode(
        shape=(32, 32, 16),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=weight_init,
        name="stem",
    )

    all_nodes = [input_node, stem]
    all_edges = [Edge(source=input_node, target=stem.slot("in"))]

    # Build 4 stages with [2, 2, 2, 2] blocks
    stage_configs = [
        (16, 1, 2),  # (channels, first_stride, num_blocks)
        (32, 2, 2),
        (64, 2, 2),
        (128, 2, 2),
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
            )
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            prev = add_node

    # Global average pooling: (B, 4, 4, 128) -> (B, 128)
    avg_pool = AvgPoolNode(shape=(128,), name="avgpool")
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
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
        scaling=scaling,
    )

    return structure


# =============================================================================
# Model Factory
# =============================================================================


def _create_mupc_model(rng_key, *, infer_steps, eta_infer):
    """Create ResNet-18 with muPC parameterization."""
    structure = build_resnet18(
        weight_init=MuPCInitializer(),
        scaling=MuPCConfig(include_output=False),
        output_weight_init=XavierInitializer(),
        infer_steps=infer_steps,
        eta_infer=eta_infer,
    )
    params = initialize_params(structure, rng_key)
    return params, structure


# =============================================================================
# Training
# =============================================================================


def run_single_mupc(args):
    """Default mode: single muPC training run with progress bar."""
    print("=" * 60)
    print("ResNet-18 on CIFAR-10 (Predictive Coding + muPC)")
    print("=" * 60)

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    params, structure = _create_mupc_model(
        graph_key,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )

    # Print model summary
    print(f"\nModel: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    for name, node in structure.nodes.items():
        ni = node.node_info
        if ni.scaling_config is not None:
            fwd_scales = list(ni.scaling_config.forward_scale.values())
            tag = f"fwd_scale={fwd_scales[0]:.6f}" if fwd_scales else ""
        else:
            tag = "no scaling"
        print(f"  {name}: shape={ni.shape}, {tag}")

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data
    train_loader = Cifar10Loader(
        "train", batch_size=args.batch_size, shuffle=True, seed=42
    )
    test_loader = Cifar10Loader("test", batch_size=args.batch_size, shuffle=False)

    # Train
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    train_config = {"num_epochs": args.num_epochs}

    print(
        f"\nTraining for {args.num_epochs} epochs "
        f"(JIT compilation on first batch)..."
    )
    start_time = time.time()

    num_batches = len(train_loader)
    progress = tqdm(total=num_batches * args.num_epochs, desc="Training", unit="batch")

    def iter_cb(epoch_idx, batch_idx, energy):
        norm = float(energy)  # already per-sample
        progress.update(1)
        progress.set_postfix(
            epoch=f"{epoch_idx + 1}/{args.num_epochs}", energy=f"{norm:.4f}"
        )
        return norm

    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=False,
        iter_callback=iter_cb,
    )
    progress.close()

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.1f}s ({elapsed / args.num_epochs:.1f}s per epoch)")

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")


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
        "--eta_infer", type=float, default=0.003, help="Inference rate (default: 0.003)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.002, help="Learning rate (default: 0.002)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch output")
    return parser.parse_args()


def main():
    args = parse_args()
    run_single_mupc(args)


if __name__ == "__main__":
    main()
