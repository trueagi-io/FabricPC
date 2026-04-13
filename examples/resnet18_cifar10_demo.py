"""
ResNet-18 — CIFAR-10 Demo (Predictive Coding)
===============================================

Demonstrates a ResNet-18 architecture built as a predictive coding graph
using muPC scaling on CIFAR-10.

Architecture (CIFAR-10 variant — no 7x7 conv or maxpool):
    input(32,32,3) -> conv0(32,32,64, 3x3)
    -> Stage 1: 2 residual blocks (32,32,64)
    -> Stage 2: 2 residual blocks (16,16,128)
    -> Stage 3: 2 residual blocks (8,8,256)
    -> Stage 4: 2 residual blocks (4,4,512)
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

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import argparse
import time
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
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
    initialize,
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


def build_resnet18(infer_steps=30, eta_infer=0.1):
    """
    Build ResNet-18 for CIFAR-10 with muPC scaling.

    Architecture:
        input(32,32,3) -> conv0(32,32,64)
        -> Stage 1: 2× ResBlock(64)
        -> Stage 2: 2× ResBlock(128, stride=2 on first)
        -> Stage 3: 2× ResBlock(256, stride=2 on first)
        -> Stage 4: 2× ResBlock(512, stride=2 on first)
        -> AvgPool -> Linear(10, softmax+CE)

    Returns:
        GraphStructure with muPC scaling.
    """
    weight_init = MuPCInitializer()

    # Input
    input_node = IdentityNode(shape=(32, 32, 3), name="input")

    # Initial convolution: 3x3, 64 channels, no maxpool (CIFAR is 32x32)
    conv0 = Conv2DNode(
        shape=(32, 32, 64),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=weight_init,
        name="conv0",
    )

    all_nodes = [input_node, conv0]
    all_edges = [Edge(source=input_node, target=conv0.slot("in"))]

    # Build 4 stages with [2, 2, 2, 2] blocks
    stage_configs = [
        (64, 1, 2),  # (channels, first_stride, num_blocks)
        (128, 2, 2),
        (256, 2, 2),
        (512, 2, 2),
    ]

    prev = conv0
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

    # Global average pooling: (B, 4, 4, 512) -> (B, 512)
    avg_pool = AvgPoolNode(shape=(512,), name="avgpool")
    all_nodes.append(avg_pool)
    all_edges.append(Edge(source=prev, target=avg_pool.slot("in")))

    # Output: Linear(10) with softmax + cross-entropy
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=XavierInitializer(),
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=avg_pool, target=output.slot("in")))

    # Build graph with muPC scaling
    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
        scaling=MuPCConfig(include_output=False),
    )

    return structure


# =============================================================================
# CLI and Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResNet-18 on CIFAR-10 (Predictive Coding)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--infer_steps", type=int, default=30, help="Inference steps (default: 30)"
    )
    parser.add_argument(
        "--eta_infer", type=float, default=0.1, help="Inference LR (default: 0.1)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch output")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("ResNet-18 on CIFAR-10 (Predictive Coding + muPC)")
    print("=" * 60)

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    structure = build_resnet18(
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )
    params = initialize_params(structure, graph_key)

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
    optimizer = optax.adamw(args.lr, weight_decay=0.01)
    train_config = {"num_epochs": args.num_epochs}

    print(
        f"\nTraining for {args.num_epochs} epochs "
        f"(JIT compilation on first batch)..."
    )
    start_time = time.time()

    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=args.verbose,
    )

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.1f}s ({elapsed / args.num_epochs:.1f}s per epoch)")

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
