"""
Example: Custom Conv2D Node
=====================================

This example demonstrates:
1. Creating a custom node type using @register_node
2. Implementing forward pass using JAX's lax.conv
3. Using CONFIG_SCHEMA for node-specific parameters
4. Training on MNIST using the standard train_pcn/evaluate_pcn methods

Run with: python examples/custom_node.py
"""

import os  # set environment variables before importing JAX

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault(
    "JAX_PLATFORMS", "cuda"
)  # options: "cpu", "cuda" or "tpu" if available
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress XLA warnings

import time
from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from fabricpc.utils.data.dataloader import MnistLoader

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.registry import register_node
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import get_activation
from fabricpc.core.initializers import initialize
from fabricpc.graph import create_pc_graph
from fabricpc.training import train_pcn, evaluate_pcn


# ==============================================================================
# CUSTOM NODE DEFINITION
# ==============================================================================


@register_node("conv2d")
class Conv2DNode(NodeBase):
    """
    2D Convolutional node using JAX's lax.conv_general_dilated.

    Expects inputs in NHWC format (batch, height, width, channels).
    Output shape should be specified as (H_out, W_out, C_out).

    Config parameters:
        kernel_size: Tuple[int, int] - Kernel dimensions (kH, kW)
        stride: Tuple[int, int] - Stride (default: (1, 1))
        padding: str - "VALID" or "SAME" (default: "SAME")
    """

    CONFIG_SCHEMA = {
        "kernel_size": {
            "type": tuple,
            "required": True,
            "description": "Convolution kernel size (kH, kW)",
        },
        "stride": {
            "type": tuple,
            "default": (1, 1),
            "description": "Stride for convolution",
        },
        "padding": {
            "type": str,
            "default": "SAME",
            "choices": ["SAME", "VALID"],
            "description": "Padding mode",
        },
    }

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv2D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Initialize convolution kernels and biases.

        Kernel shape: (kH, kW, C_in, C_out)
        Bias shape: (1, 1, 1, C_out) for NHWC broadcasting
        """
        kernel_size = config["kernel_size"]
        out_channels = node_shape[-1]  # Last dim is channels (NHWC)

        # Get weight initialization config
        default_cfg = {"type": "normal", "mean": 0.0, "std": 0.05}
        weight_init_config = config.get("weight_init", default_cfg)

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes) + 1)

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]  # Input channels from source

            kernel_param_shape = (
                kernel_size[0],
                kernel_size[1],
                in_channels,
                out_channels,
            )

            weights_dict[edge_key] = initialize(
                keys[i], kernel_param_shape, weight_init_config
            )

        # Initialize bias
        use_bias = config.get("use_bias", True)
        if use_bias:
            bias = jnp.zeros((1, 1, 1, out_channels))
        else:
            bias = jnp.array([])

        return NodeParams(weights=weights_dict, biases={"b": bias} if use_bias else {})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass using JAX conv2d.

        Computes: conv2d(x, kernel) + bias -> activation -> error -> energy
        """
        from fabricpc.nodes import get_node_class

        node_class = get_node_class(node_info.node_type)

        config = node_info.node_config
        stride = config.get("stride", (1, 1))
        padding = config.get("padding", "SAME")

        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape

        if node_info.in_degree == 0:
            # Source node
            z_mu = state.z_latent
            pre_activation = jnp.zeros_like(state.z_latent)
            error = jnp.zeros_like(state.z_latent)
        else:
            # Accumulate convolution outputs from all inputs
            pre_activation = jnp.zeros((batch_size, *out_shape))

            for edge_key, x in inputs.items():
                kernel = params.weights[edge_key]
                # Use JAX's lax.conv_general_dilated for the convolution
                conv_out = jax.lax.conv_general_dilated(
                    x,  # input: NHWC
                    kernel,  # kernel: HWIO
                    window_strides=stride,
                    padding=padding,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                )
                pre_activation = pre_activation + conv_out

            # Add bias if present
            if "b" in params.biases and params.biases["b"].size > 0:
                pre_activation = pre_activation + params.biases["b"]

            # Apply activation
            activation_fn, activation_deriv = get_activation(
                node_info.node_config["activation"]
            )
            z_mu = activation_fn(pre_activation)

            # Compute error
            error = state.z_latent - z_mu

        # Update state
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error,
        )

        # Compute energy
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


# ==============================================================================
# NETWORK CONFIGURATION
# ==============================================================================


def create_conv_mnist_config():
    """
    Create a convolutional MNIST classifier using Conv2D and Linear nodes.

    Architecture:
        input (28, 28, 1)
        -> conv1 (26, 26, 16) with 3x3 kernel, ReLU
        -> conv2 (24, 24, 32) with 3x3 kernel, ReLU
        -> flatten -> linear (10) output

    Note: Using smaller channel counts for faster training in this demo.
    """
    return {
        "node_list": [
            {
                "name": "input",
                "shape": (28, 28, 1),
                "type": "linear",  # Input node, identity
                "activation": {"type": "identity"},
            },
            {
                "name": "conv1",
                "shape": (26, 26, 16),  # VALID padding: 28-3+1=26
                "type": "conv2d",
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": "VALID",
                "activation": {"type": "relu"},
            },
            {
                "name": "conv2",
                "shape": (24, 24, 32),  # 26-3+1=24
                "type": "conv2d",
                "kernel_size": (3, 3),
                "stride": (1, 1),
                "padding": "VALID",
                "activation": {"type": "relu"},
            },
            {
                "name": "output",
                "shape": (10,),  # 10 classes
                "type": "linear",
                "flatten_input": True,
                "activation": {"type": "sigmoid"},
            },
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "conv1", "slot": "in"},
            {"source_name": "conv1", "target_name": "conv2", "slot": "in"},
            {"source_name": "conv2", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
    }


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    print("=" * 70)
    print("Custom Node Example: Conv2D on MNIST")
    print("=" * 70)

    # Set random seeds for reproducibility
    jax.config.update("jax_default_prng_impl", "threefry2x32")
    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Show registered node types (should include our custom conv2d)
    from fabricpc.nodes import list_node_types

    print(f"\nRegistered node types: {list_node_types()}")

    # Create model
    print("\nCreating convolutional MNIST classifier...")
    config = create_conv_mnist_config()
    params, structure = create_pc_graph(config, graph_key)

    print(
        f"Model created: {len(config['node_list'])} nodes, {len(config['edge_list'])} edges"
    )
    for name, node_info in structure.nodes.items():
        print(f"  {name}: shape={node_info.shape}, type={node_info.node_type}")

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Training config (fewer epochs for demo)
    train_config = {
        "num_epochs": 3,  # Fewer epochs for demo
        "infer_steps": 10,  # Inference steps
        "eta_infer": 0.05,  # Inference learning rate
        "optimizer": {"type": "adam", "lr": 0.001},
    }
    batch_size = 64  # Smaller batch for conv nets

    train_loader = MnistLoader("train", batch_size=batch_size, shuffle=True, seed=42)
    test_loader = MnistLoader("test", batch_size=batch_size, shuffle=False)

    # Train
    print("\nTraining (JIT compilation on first batch, may take a moment)...")
    print("Note: Conv2D gradient computation is slower than Linear nodes.")
    start_time = time.time()

    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        config=train_config,
        rng_key=train_key,
        verbose=True,
    )

    train_time = time.time() - start_time
    print(
        f"Training time: {train_time:.1f}s ({train_time / train_config['num_epochs']:.1f}s per epoch)"
    )

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )

    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Energy: {metrics['energy']:.4f}")

    print("\n" + "=" * 70)
    print("Custom node example complete!")
    print("\nKey takeaways:")
    print("  1. Create custom nodes by subclassing NodeBase")
    print("  2. Use @register_node decorator to register with the library")
    print("  3. Implement get_slots(), initialize_params(), and forward()")
    print("  4. Use train_pcn/evaluate_pcn for standard training workflow")
    print("=" * 70)


if __name__ == "__main__":
    main()
