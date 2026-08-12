"""Separate glucose graph variants for associative-memory experiments."""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

from examples.glucose_model import (
    ContinuousEmbeddingNode,
    MultiScaleMhaResidualNode,
    RegressionOutputNode,
)
from fabricpc.core.activations import GeluActivation, IdentityActivation
from fabricpc.core.energy import EnergyFunctional, GaussianEnergy
from fabricpc.core.inference import InferenceBase, InferenceSGDNormClip
from fabricpc.core.initializers import NormalInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import FeedforwardStateInit
from fabricpc.nodes import (
    IdentityNode,
    Linear,
    LnMlp1Node,
    Mlp2ResidualNode,
    StorkeyHopfield,
)

HopfieldVariant = Literal[
    "baseline",
    "projection",
    "embed-storkey",
    "forecast-storkey",
]


def create_glucose_hopfield_transformer(
    *,
    variant: HopfieldVariant,
    hopfield_strength: float | None = 1.0,
    depth: int = 2,
    embed_dim: int = 32,
    num_heads: int = 1,
    mlp_dim: int = 128,
    seq_len: int = 64,
    horizon: int = 12,
    in_channels: int = 1,
    inference: InferenceBase | None = None,
    weight_init_std: float = 0.02186191083483616,
    use_rope: bool = True,
    include_output_scaling: bool = True,
    energy: EnergyFunctional | None = None,
):
    """Build a matched baseline, projection control, or Storkey graph.

    The Storkey layer uses an identity activation so comparisons measure the
    learned projection and associative energy rather than tanh compression.
    ``forecast-storkey`` keeps an identity task node after the memory node,
    because FabricPC intentionally disables energy on an unclamped terminal
    output during evaluation.
    """
    if variant not in {
        "baseline",
        "projection",
        "embed-storkey",
        "forecast-storkey",
    }:
        raise ValueError(f"Unknown glucose Hopfield variant: {variant}")
    if seq_len % 4 != 0:
        raise ValueError(f"seq_len must be divisible by 4, got {seq_len}")
    if inference is None:
        inference = InferenceSGDNormClip(
            eta_infer=1.4435783212385837e-5,
            infer_steps=19,
            max_norm=1.0,
        )

    if energy is None:
        energy = GaussianEnergy()

    weight_init = NormalInitializer(std=weight_init_std)
    nodes = []
    edges = []

    input_node = Linear(
        shape=(seq_len, in_channels),
        activation=IdentityActivation(),
        name="glucose_input",
        energy=energy,
    )
    embed = ContinuousEmbeddingNode(
        name="embed",
        shape=(seq_len, embed_dim),
        embed_dim=embed_dim,
        in_channels=in_channels,
        weight_init=XavierInitializer(),
        energy=energy,
    )
    nodes.extend((input_node, embed))
    edges.append(Edge(source=input_node, target=embed.slot("in")))

    previous = embed
    if variant == "projection":
        memory_control = Linear(
            shape=(seq_len, embed_dim),
            activation=IdentityActivation(),
            name="embed_projection_control",
            energy=energy,
        )
        nodes.append(memory_control)
        edges.append(Edge(source=previous, target=memory_control.slot("in")))
        previous = memory_control
    elif variant == "embed-storkey":
        embed_memory = StorkeyHopfield(
            shape=(seq_len, embed_dim),
            name="embed_storkey",
            activation=IdentityActivation(),
            hopfield_strength=hopfield_strength,
            enforce_symmetry=True,
            zero_diagonal=False,
            energy=energy,
        )
        nodes.append(embed_memory)
        edges.append(Edge(source=previous, target=embed_memory.slot("in")))
        previous = embed_memory

    for index in range(depth):
        attention = MultiScaleMhaResidualNode(
            name=f"L{index}_msha",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_rope=use_rope,
            weight_init=weight_init,
            energy=energy,
        )
        mlp_1 = LnMlp1Node(
            name=f"L{index}_mlp1",
            shape=(seq_len, mlp_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            activation=GeluActivation(),
            weight_init=weight_init,
            energy=energy,
        )
        mlp_2 = Mlp2ResidualNode(
            name=f"L{index}_mlp2",
            shape=(seq_len, embed_dim),
            embed_dim=embed_dim,
            ff_dim=mlp_dim,
            weight_init=weight_init,
            energy=energy,
        )
        nodes.extend((attention, mlp_1, mlp_2))
        edges.extend(
            (
                Edge(source=previous, target=attention.slot("in")),
                Edge(source=previous, target=attention.slot("skip")),
                Edge(source=attention, target=mlp_1.slot("in")),
                Edge(source=mlp_1, target=mlp_2.slot("in")),
                Edge(source=attention, target=mlp_2.slot("residual")),
            )
        )
        previous = mlp_2

    forecast = RegressionOutputNode(
        name="forecast_projection" if variant == "forecast-storkey" else "output",
        shape=(horizon,),
        seq_len=seq_len,
        embed_dim=embed_dim,
        horizon=horizon,
        weight_init=NormalInitializer(std=float(jnp.sqrt(1.0 / embed_dim))),
        energy=energy,
    )
    nodes.append(forecast)
    edges.append(Edge(source=previous, target=forecast.slot("in")))

    output = forecast
    if variant == "forecast-storkey":
        forecast_memory = StorkeyHopfield(
            shape=(horizon,),
            name="forecast_storkey",
            activation=IdentityActivation(),
            hopfield_strength=hopfield_strength,
            enforce_symmetry=True,
            zero_diagonal=False,
            energy=energy,
        )
        output = IdentityNode(shape=(horizon,), name="output", energy=energy)
        nodes.extend((forecast_memory, output))
        edges.extend(
            (
                Edge(source=forecast, target=forecast_memory.slot("in")),
                Edge(source=forecast_memory, target=output.slot("in")),
            )
        )

    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=inference,
        scaling=MuPCConfig(include_output=include_output_scaling),
        graph_state_initializer=FeedforwardStateInit(),
    )
