"""
Custom Conv2D Node — MNIST Demo

Create a custom node type by subclassing NodeBase, implementing a
forward pass with JAX's lax.conv, and training on MNIST.
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import time
from typing import Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
import optax
import numpy as np
from fabricpc.utils.data.dataloader import MnistLoader

from fabricpc.nodes import Linear
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SigmoidActivation,
)
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, initialize
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.training import train_pcn, evaluate_pcn

# --- Custom Node Definition ---


class Conv2DNode(NodeBase):
    """
    2D Convolutional node using JAX's lax.conv_general_dilated.

    Expects inputs in NHWC format (batch, height, width, channels).
    Output shape should be specified as (H_out, W_out, C_out).

    Parameters:
        kernel_size: Tuple[int, int] - Kernel dimensions (kH, kW)
        stride: Tuple[int, int] - Stride (default: (1, 1))
        padding: str - "VALID" or "SAME" (default: "SAME")
    """

    def __init__(
        self,
        shape,
        name,
        kernel_size,
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Conv2D has a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Conv2D fan_in = C_in * kH * kW (kernel receptive field)."""
        kernel_size = config.get("kernel_size", (1, 1))
        C_in = source_shape[-1]  # NHWC: channels last
        return C_in * int(np.prod(kernel_size))

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize convolution kernels and biases.

        Kernel shape: (kH, kW, C_in, C_out)
        Bias shape: (1, 1, 1, C_out) for NHWC broadcasting
        """
        if config is None:
            config = {}
        kernel_size = config.get("kernel_size")
        out_channels = node_shape[-1]  # Last dim is channels (NHWC)

        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=0.05)

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
                keys[i], kernel_param_shape, weight_init
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
        config = node_info.node_config
        stride = config.get("stride", (1, 1))
        padding = config.get("padding", "SAME")

        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape

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
        activation = node_info.activation  # ActivationBase instance
        z_mu = type(activation).forward(pre_activation, activation.config)

        # Compute error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(
            pre_activation=pre_activation,
            z_mu=z_mu,
            error=error,
        )

        # Compute energy
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        total_energy = jnp.sum(state.energy)

        return total_energy, state


# --- Network Configuration ---


def create_conv_mnist_structure():
    """
    Create a convolutional MNIST classifier using Conv2D and Linear nodes.

    Architecture:
        input (28, 28, 1)
        -> conv1 (26, 26, 16) with 3x3 kernel, ReLU
        -> conv2 (24, 24, 32) with 3x3 kernel, ReLU
        -> flatten -> linear (10) output

    Note: Using smaller channel counts for faster training in this demo.
    """
    input_node = Linear(
        shape=(28, 28, 1), activation=IdentityActivation(), name="input"
    )
    conv1 = Conv2DNode(
        shape=(26, 26, 16),  # VALID padding: 28-3+1=26
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="VALID",
        activation=ReLUActivation(),
        name="conv1",
    )
    conv2 = Conv2DNode(
        shape=(24, 24, 32),  # 26-3+1=24
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="VALID",
        activation=ReLUActivation(),
        name="conv2",
    )
    output_node = Linear(
        shape=(10,),  # 10 classes
        activation=SigmoidActivation(),
        flatten_input=True,
        name="output",
    )

    structure = graph(
        nodes=[input_node, conv1, conv2, output_node],
        edges=[
            Edge(source=input_node, target=conv1.slot("in")),
            Edge(source=conv1, target=conv2.slot("in")),
            Edge(source=conv2, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    )

    return structure


# --- Main ---


def main():
    print("Custom Node Example: Conv2D on MNIST\n")

    jax.config.update("jax_default_prng_impl", "threefry2x32")
    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    structure = create_conv_mnist_structure()
    params = initialize_params(structure, graph_key)

    print(f"Model created: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    for name, node in structure.nodes.items():
        print(
            f"  {name}: shape={node.node_info.shape}, type={node.node_info.node_type}"
        )

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    optimizer = optax.adam(0.001)
    train_config = {"num_epochs": 3}
    batch_size = 64

    train_loader = MnistLoader("train", batch_size=batch_size, shuffle=True, seed=42)
    test_loader = MnistLoader("test", batch_size=batch_size, shuffle=False)

    print("Training (JIT compilation on first batch)...")
    start_time = time.time()

    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
    )

    train_time = time.time() - start_time
    print(
        f"Training time: {train_time:.1f}s ({train_time / train_config['num_epochs']:.1f}s per epoch)"
    )

    print("\nEvaluating...")
    metrics = evaluate_pcn(
        trained_params, structure, test_loader, train_config, eval_key
    )

    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
