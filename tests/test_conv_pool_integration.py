"""
Graph-level integration test for the conv/pool nodes.

Unlike test_convolutional.py / test_pooling.py (which exercise node methods in
isolation), this builds a real predictive-coding graph
(input -> conv -> maxpool -> linear), initializes parameters, and runs a couple
of full train steps (inference to convergence + local weight update). It guards
the conv/pool/muPC wiring end-to-end — the kind of test that would have caught
the non-functional ``slots=`` parameter, since a broken slot contract surfaces
only when the graph is actually assembled and run.
"""

import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import ConvNode, MaxPool, Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import train_step


def _build_structure():
    """input(8,8,1) -> conv(8,8,4) -> maxpool(4,4,4) -> linear(3)."""
    pixels = IdentityNode(shape=(8, 8, 1), name="pixels")
    conv = ConvNode(
        shape=(8, 8, 4),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        name="conv",
    )
    pool = MaxPool(
        shape=(4, 4, 4),
        window_shape=(2, 2),
        stride=(2, 2),
        padding="VALID",
        name="pool",
    )
    out = Linear(
        shape=(3,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="out",
    )
    structure = graph(
        nodes=[pixels, conv, pool, out],
        edges=[
            Edge(source=pixels, target=conv.slot("in")),
            Edge(source=conv, target=pool.slot("in")),
            Edge(source=pool, target=out.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    return structure


def test_conv_pool_linear_param_shapes(rng_key):
    """Parameters initialize with the shapes the architecture implies."""
    structure = _build_structure()
    params = initialize_params(structure, rng_key)

    # Conv kernel: (kH, kW, C_in, C_out) = (3, 3, 1, 4); bias (1, 1, 1, 4).
    conv_w = next(iter(params.nodes["conv"].weights.values()))
    assert conv_w.shape == (3, 3, 1, 4)
    assert params.nodes["conv"].biases["b"].shape == (1, 1, 1, 4)

    # Pooling is parameter-free.
    assert params.nodes["pool"].weights == {}
    assert params.nodes["pool"].biases == {}

    # Linear flattens the (4, 4, 4) pool output -> 64 -> 3.
    out_w = next(iter(params.nodes["out"].weights.values()))
    assert out_w.shape == (64, 3)


def test_conv_pool_linear_trains(rng_key):
    """A couple of train steps run, energy stays finite, and weights update."""
    structure = _build_structure()
    params = initialize_params(structure, rng_key)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    batch_size = 8
    key_x, key_y = jax.random.split(rng_key)
    labels = jax.nn.one_hot(jax.random.randint(key_y, (batch_size,), 0, 3), 3)
    batch = {
        "x": jax.random.normal(key_x, (batch_size, 8, 8, 1)),
        "y": labels,
    }

    p, os = params, opt_state
    energies = []
    for _ in range(3):
        p, os, energy, final_state = train_step(
            p, os, batch, structure, optimizer, rng_key
        )
        energies.append(energy)

    # Energy is finite at every step (no NaN/Inf through conv/pool/inference).
    for e in energies:
        assert jnp.isfinite(e)

    # The conv weights actually moved — the learning signal reached the conv node
    # through the pooling node (this is what guards the conv/pool wiring).
    conv_before = next(iter(params.nodes["conv"].weights.values()))
    conv_after = next(iter(p.nodes["conv"].weights.values()))
    assert not jnp.allclose(conv_before, conv_after)

    # Final inference state has the right per-node latent shapes.
    assert final_state.nodes["conv"].z_latent.shape == (batch_size, 8, 8, 4)
    assert final_state.nodes["pool"].z_latent.shape == (batch_size, 4, 4, 4)
    assert final_state.nodes["out"].z_latent.shape == (batch_size, 3)
