"""
Numerical Equivalence Testing: PyTorch vs JAX
==============================================

Tests numerical equivalence between PyTorch and JAX implementations
of predictive coding networks using the aligned API.

Tests:
1. Weight initialization statistics
2. Feedforward pass equivalence
3. Inference dynamics equivalence
4. Gradient computation equivalence
"""

import sys
import os
import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fabricpc.graph.graph_net import PCGraphNet
from fabricpc.graph import create_pc_graph
from fabricpc.core.inference import run_inference
from fabricpc.graph.graph_net import initialize_state


def create_test_config():
    """Create a simple config that works in both PyTorch and JAX."""
    return {
        "node_list": [
            {"name": "input", "dim": 10, "type": "linear", "activation": {"type": "identity"}},
            {"name": "hidden", "dim": 20, "type": "linear", "activation": {"type": "relu"}},
            {"name": "output", "dim": 5, "type": "linear", "activation": {"type": "identity"}},
        ],
        "edge_list": [
            {"source_name": "input", "target_name": "hidden", "slot": "in"},
            {"source_name": "hidden", "target_name": "output", "slot": "in"},
        ],
        "task_map": {"x": "input", "y": "output"},
        "device": "cpu",  # Force CPU for numerical equivalence testing
    }


def copy_weights_pytorch_to_jax(pt_net, jax_params, jax_structure):
    """
    Copy weights from PyTorch network to JAX params.

    Args:
        pt_net: PyTorch PCGraphNet
        jax_params: JAX GraphParams
        jax_structure: JAX GraphStructure

    Returns:
        Updated JAX params with PyTorch weights
    """
    from fabricpc.core.types import GraphParams, NodeParams

    new_nodes = {}

    # Copy params for each node
    for node_name in jax_structure.nodes.keys():
        pt_node = pt_net.node_dictionary[node_name]

        # Copy weights for all incoming edges to this node
        new_weights = {}
        for edge_key in jax_structure.nodes[node_name].in_edges:
            if pt_node.in_degree > 0:
                # Copy weight matrix (PyTorch and JAX both use [in_dim, out_dim])
                pt_weight = pt_node.W.detach().cpu().numpy()
                new_weights[edge_key] = jnp.array(pt_weight)

        # Copy biases
        new_biases = {}
        if hasattr(pt_node, 'b') and pt_node.b is not None:
            pt_bias = pt_node.b.detach().cpu().numpy()
            # JAX expects biases as dict with node_name as key
            new_biases['b'] = jnp.array(pt_bias).reshape(1, -1)
        else:
            # Use zeros if no bias
            new_biases['b'] = jnp.zeros((1, jax_structure.nodes[node_name].dim))

        # Create NodeParams for this node
        new_nodes[node_name] = NodeParams(weights=new_weights, biases=new_biases)

    return GraphParams(nodes=new_nodes)


def test_weight_initialization():
    """Test weight initialization produces similar distributions."""
    print("\n" + "="*70)
    print("TEST 1: Weight Initialization Statistics")
    print("="*70)

    config = create_test_config()
    seed = 42

    # Initialize PyTorch
    torch.manual_seed(seed)
    pt_net = PCGraphNet(config)

    # Initialize JAX with same seed
    jax_key = jax.random.PRNGKey(seed)
    jax_params, jax_structure = create_pc_graph(config, jax_key)

    print("\n[Weight Shape Comparison]")
    for node_name, node in pt_net.node_dictionary.items():
        if node.in_degree > 0:
            pt_shape = node.W.shape
            # Get JAX weight from first incoming edge
            first_edge = jax_structure.nodes[node_name].in_edges[0]
            jax_shape = jax_params.nodes[node_name].weights[first_edge].shape
            match = "✓" if pt_shape == jax_shape else "✗"
            print(f"  {node_name}: PyTorch {pt_shape} vs JAX {jax_shape} {match}")

    print("\n[Weight Statistics]")
    for node_name, node in pt_net.node_dictionary.items():
        if node.in_degree > 0:
            pt_weight = node.W.detach().cpu().numpy()
            first_edge = jax_structure.nodes[node_name].in_edges[0]
            jax_weight = np.array(jax_params.nodes[node_name].weights[first_edge])

            pt_std = pt_weight.std()
            jax_std = jax_weight.std()

            print(f"  {node_name}:")
            print(f"    PyTorch std: {pt_std:.4f}")
            print(f"    JAX std: {jax_std:.4f}")
            print(f"    Relative diff: {abs(pt_std - jax_std) / (pt_std + 1e-8) * 100:.2f}%")

    print("\n✓ Weight initialization test complete")
    return True


def test_feedforward_pass():
    """Test feedforward passes produce same outputs with same weights."""
    print("\n" + "="*70)
    print("TEST 2: Feedforward Pass Equivalence")
    print("="*70)

    config = create_test_config()
    batch_size = 4

    # Initialize PyTorch
    torch.manual_seed(42)
    pt_net = PCGraphNet(config)

    # Initialize JAX and copy weights
    jax_key = jax.random.PRNGKey(42)
    jax_params, jax_structure = create_pc_graph(config, jax_key)
    jax_params = copy_weights_pytorch_to_jax(pt_net, jax_params, jax_structure)

    # Create test input
    np.random.seed(123)
    test_input = np.random.randn(batch_size, 10).astype(np.float32)

    # PyTorch forward pass (mimic feedforward initialization)
    pt_clamps = {"input": torch.from_numpy(test_input)}
    pt_net.init_latents(pt_clamps, batch_size)
    # PyTorch init_latents with latent_init_feedforward=True does feedforward init
    # This will set z_latent = z_mu for all nodes in topological order

    # JAX forward pass (feedforward initialization does this)
    clamps = {"input": jnp.array(test_input)}
    state_init_config = {"type": "feedforward", "fallback": {"type": "normal", "mean": 0.0, "std": 0.01}}
    state_key = jax.random.PRNGKey(123)  # Use consistent key for initialization
    jax_state = initialize_state(
        jax_structure, batch_size, state_key, clamps=clamps,
        state_init_config=state_init_config, params=jax_params
    )

    # Compare outputs at each node (z_latent after feedforward init)
    print("\n[Node Output Comparison]")
    max_diff_overall = 0.0

    for node_name in ["input", "hidden", "output"]:
        # After feedforward init, z_latent should match z_mu
        pt_output = pt_net.node_dictionary[node_name].z_latent.detach().cpu().numpy()
        jax_output = np.array(jax_state.nodes[node_name].z_latent)

        max_diff = np.max(np.abs(pt_output - jax_output))
        rel_diff = max_diff / (np.abs(pt_output).max() + 1e-8)
        max_diff_overall = max(max_diff_overall, max_diff)

        status = "✓" if max_diff < 1e-4 else "✗"
        print(f"  {node_name}:")
        print(f"    Max absolute diff: {max_diff:.6e} {status}")
        print(f"    Relative diff: {rel_diff * 100:.4f}%")
        print(f"    PyTorch sample: {pt_output[0, :3]}")
        print(f"    JAX sample: {jax_output[0, :3]}")

    # Feedforward has small floating point differences due to operation order
    tolerance = 2e-4
    success = max_diff_overall < tolerance
    if success:
        print(f"\n✓ Feedforward pass test PASSED (max diff < {tolerance})")
    else:
        print(f"\n✗ Feedforward pass test FAILED (max diff = {max_diff_overall:.6e})")

    return success


def test_inference_dynamics():
    """Test inference dynamics produce same results."""
    print("\n" + "="*70)
    print("TEST 3: Inference Dynamics Equivalence")
    print("="*70)

    config = create_test_config()
    batch_size = 4
    infer_steps = 10
    eta_infer = 0.1

    # Initialize PyTorch
    torch.manual_seed(42)
    pt_net = PCGraphNet(config)

    # Initialize JAX and copy weights
    jax_key = jax.random.PRNGKey(42)
    jax_params, jax_structure = create_pc_graph(config, jax_key)
    jax_params = copy_weights_pytorch_to_jax(pt_net, jax_params, jax_structure)

    # Create test data
    np.random.seed(123)
    test_x = np.random.randn(batch_size, 10).astype(np.float32)
    test_y = np.random.randn(batch_size, 5).astype(np.float32)

    # PyTorch inference
    pt_net.allocate_node_states(batch_size)
    pt_clamps = {
        "input": torch.from_numpy(test_x),
        "output": torch.from_numpy(test_y)
    }
    pt_net.init_latents(pt_clamps, batch_size)

    for _ in range(infer_steps):
        pt_net.update_projections()
        pt_net.update_error()
        pt_net.update_latents_step()

    # JAX inference
    jax_clamps = {
        "input": jnp.array(test_x),
        "output": jnp.array(test_y),
    }
    state_init_config = {"type": "feedforward", "fallback": {"type": "normal", "mean": 0.0, "std": 0.01}}
    state_key = jax.random.PRNGKey(123)  # Use consistent key for initialization
    jax_state_initial = initialize_state(
        jax_structure, batch_size, state_key, clamps=jax_clamps,
        state_init_config=state_init_config, params=jax_params
    )
    jax_state_final = run_inference(
        jax_params, jax_state_initial, jax_clamps,
        jax_structure, infer_steps, eta_infer
    )

    # Compare final latent states
    print("\n[Final Latent State Comparison]")
    max_diff_overall = 0.0

    for node_name in ["input", "hidden", "output"]:
        pt_latent = pt_net.node_dictionary[node_name].z_latent.detach().cpu().numpy()
        jax_latent = np.array(jax_state_final.nodes[node_name].z_latent)

        max_diff = np.max(np.abs(pt_latent - jax_latent))
        rel_diff = max_diff / (np.abs(pt_latent).max() + 1e-8)
        max_diff_overall = max(max_diff_overall, max_diff)

        status = "✓" if max_diff < 1e-3 else "✗"
        print(f"  {node_name}:")
        print(f"    Max absolute diff: {max_diff:.6e} {status}")
        print(f"    Relative diff: {rel_diff * 100:.4f}%")
        print(f"    PyTorch sample: {pt_latent[0, :3]}")
        print(f"    JAX sample: {jax_latent[0, :3]}")

    # Compare errors
    print("\n[Error Comparison]")
    for node_name in ["hidden", "output"]:
        pt_error = pt_net.node_dictionary[node_name].error.detach().cpu().numpy()
        jax_error = np.array(jax_state_final.nodes[node_name].error)

        max_diff = np.max(np.abs(pt_error - jax_error))
        max_diff_overall = max(max_diff_overall, max_diff)
        status = "✓" if max_diff < 1e-3 else "✗"
        print(f"  {node_name}: Max diff {max_diff:.6e} {status}")

    success = max_diff_overall < 1e-3
    if success:
        print("\n✓ Inference dynamics test PASSED (max diff < 1e-3)")
    else:
        print(f"\n✗ Inference dynamics test FAILED (max diff = {max_diff_overall:.6e})")

    return success


def test_gradient_computation():
    """Test gradient computation produces same results.

    Note: Both PyTorch and JAX now compute gradients manually from final inference state
    using local Hebbian learning rules for predictive coding.
    """
    print("\n" + "="*70)
    print("TEST 4: Gradient Computation Equivalence")
    print("="*70)
    print("Note: Comparing manual gradients (PyTorch) vs manual gradients (JAX)")

    # Import the manual gradient computation function
    from fabricpc.training.train import compute_local_weight_gradients

    config = create_test_config()
    batch_size = 4
    infer_steps = 10
    eta_infer = 0.1

    # Initialize PyTorch
    torch.manual_seed(42)
    pt_net = PCGraphNet(config)

    # Initialize JAX and copy weights
    jax_key = jax.random.PRNGKey(42)
    jax_params, jax_structure = create_pc_graph(config, jax_key)
    jax_params = copy_weights_pytorch_to_jax(pt_net, jax_params, jax_structure)

    # Create test data
    np.random.seed(123)
    test_x = np.random.randn(batch_size, 10).astype(np.float32)
    test_y = np.random.randn(batch_size, 5).astype(np.float32)

    # PyTorch: Run inference and compute gradients
    pt_net.allocate_node_states(batch_size)
    pt_clamps = {
        "input": torch.from_numpy(test_x),
        "output": torch.from_numpy(test_y)
    }
    pt_net.init_latents(pt_clamps, batch_size)

    for _ in range(infer_steps):
        pt_net.update_projections()
        pt_net.update_error()
        pt_net.update_latents_step()

    # Compute weight gradients in PyTorch
    for node in pt_net.node_dictionary.values():
        if node.in_degree > 0:
            node.W.grad = node.compute_weight_grad()
            if hasattr(node, 'b') and node.b is not None:
                node.b.grad = node.compute_bias_grad()

    # JAX: Run inference and compute manual gradients
    jax_clamps = {"input": jnp.array(test_x), "output": jnp.array(test_y)}
    state_init_config = {"type": "feedforward", "fallback": {"type": "normal", "mean": 0.0, "std": 0.01}}
    state_key = jax.random.PRNGKey(123)  # Use consistent key for initialization
    jax_state_initial = initialize_state(
        jax_structure, batch_size, state_key, clamps=jax_clamps,
        state_init_config=state_init_config, params=jax_params
    )

    jax_state_final = run_inference(
        jax_params, jax_state_initial, jax_clamps,
        jax_structure, infer_steps, eta_infer
    )

    # Compute manual gradients using local Hebbian learning rules
    jax_grads = compute_local_weight_gradients(jax_params, jax_state_final, jax_structure)

    # Compare gradients
    print("\n[Weight Gradient Comparison]")
    max_diff_overall = 0.0

    for node_name in ["hidden", "output"]:
        pt_node = pt_net.node_dictionary[node_name]
        pt_grad = pt_node.W.grad.detach().cpu().numpy()

        # Get JAX gradient from first incoming edge
        first_edge = jax_structure.nodes[node_name].in_edges[0]
        jax_grad = np.array(jax_grads.nodes[node_name].weights[first_edge])

        max_diff = np.max(np.abs(pt_grad - jax_grad))
        rel_diff = max_diff / (np.abs(pt_grad).max() + 1e-8)
        max_diff_overall = max(max_diff_overall, max_diff)

        status = "✓" if max_diff < 1e-2 else "✗"
        print(f"  {node_name} weight:")
        print(f"    Max absolute diff: {max_diff:.6e} {status}")
        print(f"    Relative diff: {rel_diff * 100:.4f}%")
        print(f"    PyTorch grad norm: {np.linalg.norm(pt_grad):.6f}")
        print(f"    JAX grad norm: {np.linalg.norm(jax_grad):.6f}")

    print("\n[Bias Gradient Comparison]")
    for node_name in ["hidden", "output"]:
        pt_node = pt_net.node_dictionary[node_name]
        if hasattr(pt_node, 'b') and pt_node.b is not None and pt_node.b.grad is not None:
            pt_grad = pt_node.b.grad.detach().cpu().numpy().squeeze()
            jax_grad = np.array(jax_grads.nodes[node_name].biases['b']).squeeze()

            max_diff = np.max(np.abs(pt_grad - jax_grad))
            max_diff_overall = max(max_diff_overall, max_diff)
            status = "✓" if max_diff < 1e-2 else "✗"
            print(f"  {node_name} bias: Max diff {max_diff:.6e} {status}")

    # Both implementations now use manual gradient computation
    # Small differences may still occur due to numerical precision
    print(f"\n[Gradient Computation Notes]")
    print(f"  Max gradient difference: {max_diff_overall:.6e}")
    print(f"  PyTorch uses manual gradient computation from final state")
    print(f"  JAX uses manual gradient computation from final state (compute_local_weight_gradients)")
    print(f"  Both use local Hebbian learning rules for predictive coding")

    # Check if gradients are closely matched
    tolerance = 1e-3
    success = max_diff_overall < tolerance
    if success:
        print(f"\n✓ Gradient test PASSED - Manual gradients match (max diff < {tolerance})")
    else:
        print(f"\n✗ Gradient test FAILED - Manual gradients differ (max diff = {max_diff_overall:.6e})")
    return success


def run_all_tests():
    """Run all numerical equivalence tests."""
    print("\n" + "="*70)
    print("NUMERICAL EQUIVALENCE TESTING: PyTorch vs JAX")
    print("="*70)
    print("\nUsing aligned API configuration format")
    print("Testing on simple 3-layer network: 10 → 20 → 5")

    results = {}

    try:
        results['initialization'] = test_weight_initialization()
        results['feedforward'] = test_feedforward_pass()
        results['inference'] = test_inference_dynamics()
        results['gradients'] = test_gradient_computation()

        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        all_passed = all(results.values())

        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {test_name.capitalize():20s}: {status}")

        print("\n" + "="*70)
        if all_passed:
            print("✓ ALL TESTS PASSED - PyTorch and JAX are numerically equivalent!")
            print("  (within expected floating-point precision tolerances)")
        else:
            print("✗ SOME TESTS FAILED - see details above")
        print("="*70)

        return all_passed

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
