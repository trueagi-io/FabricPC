"""
jpc FC-ResNet Comparison Demo
==============================

Replicates the FC-ResNet architecture from the jpc library (thebuckleylab/jpc)
using FabricPC's predictive coding mechanics, enabling a fair comparison of the
two implementations.

Architecture (matching jpc mupc.ipynb):
    input(784) -> FCInput(width) -> PreActResBlock(width) x (depth-2) -> Readout(10)

Each PreActResBlock computes:
    z_mu = hidden_scale * (W @ act(x)) + x     (pre-activation + identity skip)

This is a fully-connected ResNet — no convolutions, no spatial structure. The
skip connection is internal to each block (one z_latent per layer, matching jpc).

Scaling modes (--scaling flag):
    jpc:      in=1/√D, hidden=1/√(N*L), out=1/N          (includes depth L)
    fabricpc: in=1/√D, hidden=gain/√N,   out=1/N          (no depth factor)

Key difference from jpc: FabricPC uses fixed-step SGD inference with gradient
norm clipping (InferenceSGDNormClip), while jpc uses an adaptive ODE solver
(diffrax Heun + PID controller). The ODE solver handles stiff dynamics from
muPC scaling natively. For FabricPC's SGD, we need gradient clipping and a
lower parameter learning rate to compensate.

Results (3 epochs, MNIST, jpc scaling, width=128):
    depth=5:  ~93% (matches jpc reference)
    depth=10: ~93%
    depth=30: ~86% (limited by SGD inference convergence)

Usage:
    # Default: depth=10 FC-ResNet with jpc scaling
    python examples/jpc_fc_resnet_compare.py

    # Compare scaling modes
    python examples/jpc_fc_resnet_compare.py --scaling jpc --depth 10
    python examples/jpc_fc_resnet_compare.py --scaling fabricpc --depth 10

    # Test depth scaling
    for d in 5 10 30 50; do
      python examples/jpc_fc_resnet_compare.py --scaling jpc --depth $d
    done

    # Deep network with tuned hyperparams
    python examples/jpc_fc_resnet_compare.py --depth 30 --param_lr 0.0003 \\
        --eta_infer 0.1 --max_norm 0.1 --num_epochs 5

References:
    - jpc: https://github.com/thebuckleylab/jpc
    - Innocenti et al., "muPC: scaling for predictive coding networks"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import math
import time
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.nodes import IdentityNode
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.initializers import NormalInitializer, initialize
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader

jax.config.update("jax_default_prng_impl", "threefry2x32")


# =============================================================================
# Scaling Factor Computation
# =============================================================================


def compute_scaling_factors(
    input_dim: int, width: int, depth: int, mode: str
) -> Tuple[float, float, float]:
    """
    Compute per-layer-type scaling factors.

    Args:
        input_dim: Input dimensionality (784 for MNIST).
        width: Hidden layer width.
        depth: Total number of parameterized layers.
        mode: 'jpc' or 'fabricpc'.

    Returns:
        (in_scale, hidden_scale, out_scale)
    """
    if mode == "jpc":
        # jpc mupc.ipynb: exact formulas from Innocenti et al.
        in_scale = 1.0 / math.sqrt(input_dim)
        hidden_scale = 1.0 / math.sqrt(width * depth)
        out_scale = 1.0 / width
    elif mode == "fabricpc":
        # FabricPC muPC: a = gain / sqrt(fan_in), no depth factor
        # Input layer: identity activation, gain=1.0
        in_scale = 1.0 / math.sqrt(input_dim)
        # Hidden layers: ReLU activation, gain=sqrt(2)
        relu_gain = math.sqrt(2.0)
        hidden_scale = relu_gain / math.sqrt(width)
        # Output layer: identity activation, gain=1.0
        # FabricPC output formula: a = gain / (fan_in * sqrt(K)) for K=1
        out_scale = 1.0 / width
    else:
        raise ValueError(f"Unknown scaling mode: {mode!r}. Use 'jpc' or 'fabricpc'.")

    return in_scale, hidden_scale, out_scale


# =============================================================================
# Custom Nodes — Pre-activation FC-ResNet blocks
# =============================================================================


class FCInputNode(NodeBase):
    """
    First layer of the FC-ResNet: z_mu = in_scale * (W @ x).

    No activation, no skip connection. Matches jpc's ScaledLinear input layer.
    """

    def __init__(
        self,
        shape,
        name,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=1.0),
        scale=1.0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            scale=scale,
            use_bias=False,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def get_weight_fan_in(source_shape, config):
        return int(np.prod(source_shape))

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=1.0)

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes))

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_features = int(np.prod(in_shape))
            out_features = int(np.prod(node_shape))
            weight_shape = (in_features, out_features)
            weights_dict[edge_key] = initialize(keys[i], weight_shape, weight_init)

        return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def forward(params, inputs, state, node_info):
        config = node_info.node_config
        scale = config.get("scale", 1.0)
        batch_size = state.z_latent.shape[0]

        # Single input, flatten to (batch, input_dim)
        edge_key, x = next(iter(inputs.items()))
        x_flat = x.reshape(batch_size, -1)

        # z_mu = scale * (x @ W)
        W = params.weights[edge_key]
        z_mu = scale * jnp.matmul(x_flat, W)

        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        return total_energy, state


class PreActResBlock(NodeBase):
    """
    Pre-activation residual block: z_mu = hidden_scale * (W @ act(x)) + x.

    Matches jpc's ResNetBlock exactly:
    - Activation applied BEFORE linear transformation (pre-activation)
    - Identity skip connection adds raw input (NOT the activated/scaled version)
    - Scaling applied only to the linear path

    The skip connection is internal to this node — it does NOT appear as a
    graph edge. This means there is one z_latent per block, matching jpc.
    """

    def __init__(
        self,
        shape,
        name,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=1.0),
        scale=1.0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            scale=scale,
            use_bias=False,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def get_weight_fan_in(source_shape, config):
        return source_shape[-1]

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=1.0)

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes))

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_features = in_shape[-1]
            out_features = node_shape[-1]
            weight_shape = (in_features, out_features)
            weights_dict[edge_key] = initialize(keys[i], weight_shape, weight_init)

        return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def forward(params, inputs, state, node_info):
        config = node_info.node_config
        scale = config.get("scale", 1.0)
        activation = node_info.activation

        # Single input
        edge_key, x = next(iter(inputs.items()))
        W = params.weights[edge_key]

        # Pre-activation: apply activation BEFORE linear transform
        act_x = type(activation).forward(x, activation.config)

        # z_mu = scale * (act(x) @ W) + x   (skip adds raw input)
        z_mu = scale * jnp.matmul(act_x, W) + x

        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        return total_energy, state


class PreActReadout(NodeBase):
    """
    Pre-activation output layer: z_mu = out_scale * (W @ act(x)).

    Matches jpc's Readout layer: pre-activation linear with scaling, no skip.
    """

    def __init__(
        self,
        shape,
        name,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=1.0),
        scale=1.0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            scale=scale,
            use_bias=False,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def get_weight_fan_in(source_shape, config):
        return source_shape[-1]

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=1.0)

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes))

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_features = in_shape[-1]
            out_features = int(np.prod(node_shape))
            weight_shape = (in_features, out_features)
            weights_dict[edge_key] = initialize(keys[i], weight_shape, weight_init)

        return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def forward(params, inputs, state, node_info):
        config = node_info.node_config
        scale = config.get("scale", 1.0)
        activation = node_info.activation

        # Single input
        edge_key, x = next(iter(inputs.items()))
        W = params.weights[edge_key]

        # Pre-activation: apply activation BEFORE linear transform
        act_x = type(activation).forward(x, activation.config)

        # z_mu = scale * (act(x) @ W)   (no skip connection)
        z_mu = scale * jnp.matmul(act_x, W)

        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)
        return total_energy, state


# =============================================================================
# FC-ResNet Graph Builder
# =============================================================================


def build_fc_resnet(
    input_dim=784,
    width=128,
    depth=30,
    output_dim=10,
    scaling_mode="jpc",
    eta_infer=0.5,
    infer_steps=None,
    max_norm=0.5,
):
    """
    Build a fully-connected ResNet matching jpc's mupc.ipynb architecture.

    Architecture:
        input(input_dim) -> FCInput(width) -> PreActResBlock(width) x (depth-2)
                         -> PreActReadout(output_dim)

    Uses InferenceSGDNormClip for stable convergence with muPC scaling.
    jpc uses an adaptive ODE solver; FabricPC's fixed-step SGD needs gradient
    clipping to handle the stiff dynamics introduced by small scaling factors.

    Args:
        input_dim: Input dimensionality (784 for MNIST).
        width: Hidden layer width.
        depth: Total parameterized layers (input_linear + resblocks + readout).
        output_dim: Output dimensionality (10 for MNIST).
        scaling_mode: 'jpc' or 'fabricpc'.
        eta_infer: Inference learning rate.
        infer_steps: Inference steps (default: max(100, 3*depth)).
        max_norm: Maximum gradient norm for inference clipping.

    Returns:
        GraphStructure (no MuPCConfig — scaling is internal to nodes).
    """
    if infer_steps is None:
        infer_steps = max(100, 3 * depth)

    in_scale, hidden_scale, out_scale = compute_scaling_factors(
        input_dim, width, depth, scaling_mode
    )

    weight_init = NormalInitializer(mean=0.0, std=1.0)

    # Input identity node (clamped to data)
    input_node = IdentityNode(shape=(input_dim,), name="input")

    # Layer 0: FCInputNode (input_dim -> width, no activation, no skip)
    layer_0 = FCInputNode(
        shape=(width,),
        name="layer_0",
        weight_init=weight_init,
        scale=in_scale,
    )

    all_nodes = [input_node, layer_0]
    all_edges = [Edge(source=input_node, target=layer_0.slot("in"))]

    # Layers 1 to depth-2: PreActResBlock (width -> width, ReLU, skip)
    prev = layer_0
    for i in range(1, depth - 1):
        block = PreActResBlock(
            shape=(width,),
            name=f"layer_{i}",
            weight_init=weight_init,
            scale=hidden_scale,
        )
        all_nodes.append(block)
        all_edges.append(Edge(source=prev, target=block.slot("in")))
        prev = block

    # Layer depth-1: PreActReadout (width -> output_dim, ReLU pre-act, no skip)
    readout = PreActReadout(
        shape=(output_dim,),
        name="output",
        weight_init=weight_init,
        scale=out_scale,
    )
    all_nodes.append(readout)
    all_edges.append(Edge(source=prev, target=readout.slot("in")))

    # Build graph — NO MuPCConfig (scaling is internal to nodes)
    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=readout),
        inference=InferenceSGDNormClip(
            eta_infer=eta_infer, infer_steps=infer_steps, max_norm=max_norm
        ),
    )

    return structure


# =============================================================================
# CLI and Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="jpc FC-ResNet comparison on MNIST")
    parser.add_argument(
        "--scaling",
        type=str,
        default="jpc",
        choices=["jpc", "fabricpc"],
        help="Scaling formula: 'jpc' (1/sqrt(N*L)) or 'fabricpc' (gain/sqrt(N))",
    )
    parser.add_argument(
        "--depth", type=int, default=10, help="Total parameterized layers (default: 10)"
    )
    parser.add_argument(
        "--width", type=int, default=128, help="Hidden layer width (default: 128)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Training epochs (default: 3)"
    )
    parser.add_argument(
        "--eta_infer", type=float, default=0.2, help="Inference LR (default: 0.2)"
    )
    parser.add_argument(
        "--param_lr", type=float, default=0.001, help="Parameter LR (default: 0.001)"
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="Inference steps (default: max(100, 3*depth))",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=0.2,
        help="Gradient norm clipping for inference (default: 0.2)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch output")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("jpc FC-ResNet Comparison on MNIST")
    print("=" * 60)
    print(f"Scaling mode: {args.scaling}")
    print(f"Architecture: FC-ResNet, depth={args.depth}, width={args.width}")
    default_steps = args.infer_steps or max(100, 3 * args.depth)
    print(
        f"Inference: eta={args.eta_infer}, steps={default_steps}, "
        f"max_norm={args.max_norm}"
    )
    print(
        f"Training: lr={args.param_lr}, batch_size={args.batch_size}, "
        f"epochs={args.num_epochs}"
    )

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    structure = build_fc_resnet(
        input_dim=784,
        width=args.width,
        depth=args.depth,
        output_dim=10,
        scaling_mode=args.scaling,
        eta_infer=args.eta_infer,
        infer_steps=args.infer_steps,
        max_norm=args.max_norm,
    )
    params = initialize_params(structure, graph_key)

    # Print scaling factors
    in_s, hid_s, out_s = compute_scaling_factors(
        784, args.width, args.depth, args.scaling
    )
    print(f"\nScaling factors ({args.scaling}):")
    print(f"  input layer:  {in_s:.6f}")
    print(f"  hidden layers: {hid_s:.6f}")
    print(f"  output layer: {out_s:.6f}")

    # Print model summary
    print(f"\nModel: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data — MNIST, flattened, with one-hot labels for MSE
    train_loader = MnistLoader(
        "train",
        batch_size=args.batch_size,
        tensor_format="flat",
        shuffle=True,
        seed=42,
    )
    test_loader = MnistLoader(
        "test",
        batch_size=args.batch_size,
        tensor_format="flat",
        shuffle=False,
    )

    # Train — Adam (not AdamW) matching jpc
    optimizer = optax.adam(args.param_lr)
    train_config = {"num_epochs": args.num_epochs}

    print(
        f"\nTraining for {args.num_epochs} epoch(s) "
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
    print(f"Training time: {elapsed:.1f}s")

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )
    accuracy = metrics["accuracy"] * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Comparison
    print(f"\n--- Comparison ---")
    print(f"jpc reference:   ~93% (depth=30, width=128, ODE solver)")
    print(
        f"This run:        {accuracy:.2f}% (depth={args.depth}, "
        f"width={args.width}, scaling={args.scaling})"
    )
    print(f"\nNote: jpc uses an adaptive ODE solver for inference while")
    print(f"FabricPC uses fixed-step SGD with gradient clipping. At high")
    print(f"depths (>20), this gap widens due to stiff inference dynamics.")


if __name__ == "__main__":
    main()
