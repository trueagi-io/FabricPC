"""
Diagnostic script for verifying muPC scaling on deep networks.

Prints per-layer forward scales for a 20-layer chain to verify:
- All hidden nodes get a = 1/sqrt(fan_in) (no depth factor)
- z_mu norms are O(1) throughout the network after state initialization
- Energy is non-zero and weight gradients are non-zero after inference
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(jax_platforms="cpu")

import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.state_ops import set_latents_to_clamps
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.core.mupc import MuPCConfig


def main():
    num_hidden = 20
    hidden_dim = 64
    batch_size = 8

    rng_key = jax.random.PRNGKey(42)

    # Build network
    weight_init = MuPCInitializer()
    x = IdentityNode(shape=(784,), name="input")
    layers = []
    for i in range(num_hidden):
        layers.append(
            Linear(
                shape=(hidden_dim,),
                activation=TanhActivation(),
                weight_init=weight_init,
                flatten_input=(i == 0),
                name=f"h{i + 1}",
            )
        )
    y = Linear(
        shape=(10,),
        activation=IdentityActivation(),
        weight_init=weight_init,
        flatten_input=True,
        name="output",
    )

    all_nodes = [x] + layers + [y]
    all_edges = []
    prev = x
    for h in layers:
        all_edges.append(Edge(source=prev, target=h.slot("in")))
        prev = h
    all_edges.append(Edge(source=prev, target=y.slot("in")))

    infer_steps = max(20, 3 * (num_hidden + 2))
    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=infer_steps),
        scaling=MuPCConfig(include_output=True),
    )

    graph_key, init_key, data_key = jax.random.split(rng_key, 3)
    params = initialize_params(structure, graph_key)

    # Print scaling info
    print(f"{'Node':<12} {'Shape':<12} {'K':>3} {'fan_in':>8} {'fwd_scale':>12}")
    print("-" * 50)
    for name in structure.node_order:
        node = structure.nodes[name]
        ni = node.node_info
        scaling = ni.scaling_config
        if scaling is not None:
            for ek, a in scaling.forward_scale.items():
                source = structure.edges[ek].source
                src_shape = structure.nodes[source].node_info.shape
                fan_in = type(node).get_weight_fan_in(src_shape, ni.node_config)
                print(
                    f"{name:<12} {str(ni.shape):<12} {ni.in_degree:>3} {fan_in:>8} {a:>12.6f}"
                )
        else:
            print(f"{name:<12} {str(ni.shape):<12}   - {'':>8} {'no scaling':>12}")

    # Initialize state with feedforward propagation
    x_data = jax.random.normal(data_key, (batch_size, 784))
    y_data = jax.random.normal(data_key, (batch_size, 10)) * 0.1

    state = initialize_graph_state(structure, batch_size, init_key, params=params)
    clamps = {"input": x_data, "output": y_data}
    state = set_latents_to_clamps(state, clamps)

    # Print z_mu norms after state init
    print(f"\n{'Node':<12} {'z_mu norm':>12} {'z_latent norm':>14} {'energy':>12}")
    print("-" * 55)
    for name in structure.node_order:
        ns = state.nodes[name]
        z_mu_norm = float(jnp.mean(jnp.abs(ns.z_mu)))
        z_lat_norm = float(jnp.mean(jnp.abs(ns.z_latent)))
        energy = float(jnp.mean(ns.energy))
        print(f"{name:<12} {z_mu_norm:>12.4f} {z_lat_norm:>14.4f} {energy:>12.4f}")

    # Run inference
    print(f"\nRunning {infer_steps} inference steps...")
    final_state = run_inference(params, state, clamps, structure)

    print(f"\n{'Node':<12} {'z_mu norm':>12} {'z_latent norm':>14} {'energy':>12}")
    print("-" * 55)
    total_energy = 0.0
    for name in structure.node_order:
        ns = final_state.nodes[name]
        z_mu_norm = float(jnp.mean(jnp.abs(ns.z_mu)))
        z_lat_norm = float(jnp.mean(jnp.abs(ns.z_latent)))
        energy = float(jnp.mean(ns.energy))
        total_energy += energy
        print(f"{name:<12} {z_mu_norm:>12.4f} {z_lat_norm:>14.4f} {energy:>12.4f}")
    print(f"{'Total':>51} {total_energy:>12.4f}")

    # Compute weight gradients
    grad_params = compute_local_weight_gradients(params, final_state, structure)
    print(f"\n{'Node':<12} {'grad W norm':>12} {'grad W max':>12}")
    print("-" * 38)
    for name in structure.node_order:
        node_grads = grad_params.nodes[name]
        for ek, wg in node_grads.weights.items():
            norm = float(jnp.mean(jnp.abs(wg)))
            mx = float(jnp.max(jnp.abs(wg)))
            print(f"{name:<12} {norm:>12.6f} {mx:>12.6f}")

    # Sanity check
    non_zero_grads = sum(
        1
        for name in structure.node_order
        for wg in grad_params.nodes[name].weights.values()
        if float(jnp.max(jnp.abs(wg))) > 1e-10
    )
    print(f"\nNon-zero weight gradient nodes: {non_zero_grads}")
    print(f"Total energy: {total_energy:.4f}")
    if total_energy > 0 and non_zero_grads > 0:
        print("PASS: Network has non-zero energy and gradients")
    else:
        print("FAIL: Network has zero energy or zero gradients")


if __name__ == "__main__":
    main()
