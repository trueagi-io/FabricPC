"""
muPC vs Standard Parameterization — ResNet-18 A/B Experiment
=============================================================

Compares two parameterization strategies on a ResNet-18-style predictive
coding network trained on CIFAR-100:

- **muPC arm**: MuPCInitializer (W ~ N(0,1)) + MuPCConfig scaling.
  Forward scaling maintains O(1) activation variance; gradient scaling
  maintains O(1) gradient variance across depth and width.

- **Standard arm**: XavierInitializer, no runtime scaling.
  Traditional fan-in/fan-out weight initialization without forward
  scaling correction.

Both arms share the same architecture and PC hyperparameters to
isolate the effect of the parameterization.

Architecture (CIFAR-100 ResNet-18 variant):
    Stem: input(32,32,3) -> Conv3x3(32,32,64)
    Stage 1 (32x32, 64ch):  2 residual blocks
    Stage 2 (16x16, 128ch): 2 residual blocks (stride-2 downsample)
    Stage 3 (8x8, 256ch):   2 residual blocks (stride-2 downsample)
    Stage 4 (4x4, 512ch):   2 residual blocks (stride-2 downsample)
    Head: flatten -> Linear(100, softmax, CE)

Residual connections use IdentityNode as summation junctions.
Dimension-change blocks use 1x1 Conv projection skips (standard
ResNet Option B).

Usage:
    python examples/mupc_ab_experiment.py
    python examples/mupc_ab_experiment.py --n_trials 10 --num_epochs 5
    python examples/mupc_ab_experiment.py --verbose
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cuda")

import argparse
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode
from custom_node import Conv2DNode
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.depth_metric import ShortestPathDepth
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.utils.data.dataloader import Cifar100Loader

jax.config.update("jax_default_prng_impl", "threefry2x32")

# Shared hyperparameters
optimizer = optax.adamw(0.001, weight_decay=0.01)
train_config = {"num_epochs": 3}
batch_size = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description="A/B comparison: muPC vs Standard parameterization on CIFAR-100 ResNet-18"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of independent paired trials (default: 5)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Training epochs per trial (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# ResNet-18 building blocks
# ---------------------------------------------------------------------------


def _make_resnet_block(block_in, channels, spatial, name_prefix, weight_init, stride=1):
    """
    Build one residual block: two 3x3 convolutions + skip connection.

    Same-dimension block (stride=1, matching channels):
        block_in -> conv_a -> conv_b -> sum_node
               \\________________________/  (direct skip)

    Dimension-change block (stride>1 or different channels):
        block_in -> conv_a(stride) -> conv_b -> sum_node
               \\-> proj(1x1, stride) __________/  (projection skip)

    Args:
        block_in: Source node for this block's input.
        channels: Number of output channels.
        spatial: Output spatial dimension (assumes square: spatial x spatial).
        name_prefix: Prefix for node names (e.g., "s1b1" for stage 1 block 1).
        weight_init: InitializerBase for all conv weights in this block.
        stride: Stride for the first conv (1 for same-dim, 2 for downsample).

    Returns:
        Tuple of (nodes_list, edges_list, sum_node) where sum_node is the
        block output to chain into the next block.
    """
    conv_a = Conv2DNode(
        shape=(spatial, spatial, channels),
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=weight_init,
        name=f"{name_prefix}_a",
    )
    conv_b = Conv2DNode(
        shape=(spatial, spatial, channels),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=weight_init,
        name=f"{name_prefix}_b",
    )
    sum_node = IdentityNode(
        shape=(spatial, spatial, channels),
        name=f"{name_prefix}_sum",
    )

    nodes = [conv_a, conv_b, sum_node]
    edges = [
        Edge(source=block_in, target=conv_a.slot("in")),
        Edge(source=conv_a, target=conv_b.slot("in")),
        Edge(source=conv_b, target=sum_node.slot("in")),
    ]

    # Skip connection
    needs_projection = (stride != 1) or (block_in.shape[-1] != channels)
    if needs_projection:
        proj = Conv2DNode(
            shape=(spatial, spatial, channels),
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="SAME",
            activation=ReLUActivation(),
            weight_init=weight_init,
            name=f"{name_prefix}_proj",
        )
        nodes.append(proj)
        edges.extend(
            [
                Edge(source=block_in, target=proj.slot("in")),
                Edge(source=proj, target=sum_node.slot("in")),
            ]
        )
    else:
        # Direct skip: same spatial size and channels
        edges.append(Edge(source=block_in, target=sum_node.slot("in")))

    return nodes, edges, sum_node


def _build_resnet18(weight_init, scaling=None, output_weight_init=None):
    """
    Build a CIFAR-100 ResNet-18 predictive coding graph.

    Args:
        weight_init: InitializerBase for all Conv/Linear weights.
        scaling: Optional MuPCConfig for muPC parameterization.
        output_weight_init: Optional InitializerBase for output layer.
            Defaults to weight_init. For muPC, use XavierInitializer
            since the output layer is excluded from forward scaling.

    Returns:
        GraphStructure ready for initialize_params().
    """
    if output_weight_init is None:
        output_weight_init = weight_init
    all_nodes = []
    all_edges = []

    # --- Stem ---
    input_node = IdentityNode(shape=(32, 32, 3), name="input")
    stem = Conv2DNode(
        shape=(32, 32, 64),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=weight_init,
        name="stem",
    )
    all_nodes.extend([input_node, stem])
    all_edges.append(Edge(source=input_node, target=stem.slot("in")))

    prev = stem  # Tracks the output of the previous block

    # --- Stages ---
    # (channels, spatial, num_blocks, first_block_stride)
    stages = [
        (64, 32, 2, 1),  # Stage 1: 32x32, 64ch, no downsample
        (128, 16, 2, 2),  # Stage 2: 16x16, 128ch, stride-2 downsample
        (256, 8, 2, 2),  # Stage 3: 8x8, 256ch
        (512, 4, 2, 2),  # Stage 4: 4x4, 512ch
    ]

    for stage_idx, (channels, spatial, num_blocks, first_stride) in enumerate(
        stages, 1
    ):
        for block_idx in range(num_blocks):
            stride = first_stride if block_idx == 0 else 1
            prefix = f"s{stage_idx}b{block_idx + 1}"

            nodes, edges, sum_node = _make_resnet_block(
                block_in=prev,
                channels=channels,
                spatial=spatial,
                name_prefix=prefix,
                weight_init=weight_init,
                stride=stride,
            )
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            prev = sum_node

    # --- Classification head ---
    output = Linear(
        shape=(100,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=output_weight_init,
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=prev, target=output.slot("in")))

    # --- Build graph ---
    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=150),
        scaling=scaling,
    )

    return structure


# ---------------------------------------------------------------------------
# Model factories for A/B experiment
# ---------------------------------------------------------------------------


def create_mupc_resnet18(rng_key):
    """ResNet-18 with muPC parameterization."""
    structure = _build_resnet18(
        weight_init=MuPCInitializer(),
        scaling=MuPCConfig(depth_metric=ShortestPathDepth()),
        output_weight_init=XavierInitializer(),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def create_standard_resnet18(rng_key):
    """ResNet-18 with standard Xavier parameterization."""
    structure = _build_resnet18(
        weight_init=XavierInitializer(),
        scaling=None,
    )
    params = initialize_params(structure, rng_key)
    return params, structure


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    train_config["num_epochs"] = args.num_epochs
    global batch_size
    batch_size = args.batch_size

    # Print architecture summary from a throwaway build
    summary_struct = _build_resnet18(weight_init=XavierInitializer())
    n_nodes = len(summary_struct.nodes)
    n_edges = len(summary_struct.edges)

    print("=" * 70)
    print("A/B Experiment: muPC vs Standard Parameterization")
    print("=" * 70)
    print(f"Architecture: CIFAR-100 ResNet-18 ({n_nodes} nodes, {n_edges} edges)")
    print(f"muPC arm:     MuPCInitializer + MuPCConfig(ShortestPathDepth)")
    print(f"Standard arm: XavierInitializer, no scaling")
    print(f"Epochs per trial: {args.num_epochs}")
    print(f"Trials: {args.n_trials}")
    print(f"Batch size: {args.batch_size}")
    print()

    arm_mupc = ExperimentArm(
        name="muPC",
        model_factory=create_mupc_resnet18,
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=train_config,
    )

    arm_standard = ExperimentArm(
        name="Standard",
        model_factory=create_standard_resnet18,
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=train_config,
    )

    experiment = ABExperiment(
        arm_a=arm_mupc,
        arm_b=arm_standard,
        metric="accuracy",
        data_loader_factory=lambda seed: (
            Cifar100Loader(
                "train",
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
            ),
            Cifar100Loader(
                "test",
                batch_size=batch_size,
                shuffle=False,
            ),
        ),
        n_trials=args.n_trials,
        verbose=args.verbose,
    )

    results = experiment.run()
    results.print_summary()


if __name__ == "__main__":
    main()
