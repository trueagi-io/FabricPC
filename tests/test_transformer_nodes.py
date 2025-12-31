"""
Test suite for EmbeddingNode and TransformerBlockNode functionality in FabricPC.
"""

import os
# Set JAX to CPU to avoid potential OOM on small test runners
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.graph.graph_net import create_pc_graph, initialize_state
from fabricpc.core.inference import run_inference
from fabricpc.training import train_step, create_optimizer
from fabricpc.nodes import get_node_class

from fabricpc.nodes.transformer import EmbeddingNode

@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)

class TestEmbeddingNode:
    
    @pytest.fixture
    def embedding_config(self):
        """Creates a simple graph: Input (Indices) -> Embedding -> Output (Linear)"""
        vocab_size = 100
        embed_dim = 8
        seq_len = 5
        
        return {
            "node_list": [
                # Input node holds the integer indices
                {
                    "name": "indices", 
                    "shape": (seq_len,), 
                    "type": "linear", 
                    "activation": {"type": "identity"}
                },
                {
                    "name": "embed", 
                    "shape": (seq_len, embed_dim), 
                    "type": "embedding",
                    "vocab_size": vocab_size,
                    "embed_dim": embed_dim,
                },
                # Output node to give the embedding something to predict/connect to
                {
                    "name": "output", 
                    "shape": (10,), 
                    "type": "linear"
                },
            ],
            "edge_list": [
                {"source_name": "indices", "target_name": "embed", "slot": "in"},
                {"source_name": "embed", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "indices", "y": "output"},
        }

    def test_registration_and_creation(self, embedding_config, rng_key):
        """Test that the node registers and initializes params correctly."""
        node_cls = get_node_class("embedding")
        assert node_cls is not None
        
        params, structure = create_pc_graph(embedding_config, rng_key)
        embed_params = params.nodes["embed"]
        assert "embeddings" in embed_params.weights
        
        # Check Shape: (vocab_size, embed_dim)
        expected_shape = (100, 8) 
        assert embed_params.weights["embeddings"].shape == expected_shape
        
        # Check biases
        assert len(embed_params.biases) == 0

    def test_forward_lookup(self, embedding_config, rng_key):
        """Test that z_mu correctly retrieves embeddings."""
        params, structure = create_pc_graph(embedding_config, rng_key)
        
        batch_size = 3
        seq_len = 5
        
        input_indices = jax.random.randint(rng_key, (batch_size, seq_len), 0, 100)
        dummy_y = jnp.zeros((batch_size, 10))
        
        clamps = {"indices": input_indices.astype(jnp.float32), "output": dummy_y}
        state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)
        W = params.nodes["embed"].weights["embeddings"]
        
        # Need to cast input back to int because clamps are float arrays
        indices_int = input_indices.astype(jnp.int32)
        expected_vectors = W[indices_int] # (batch, seq, embed_dim)
        
        embed_state = state.nodes["embed"]
        
        # Verify z_mu matches lookup
        assert jnp.allclose(embed_state.z_mu, expected_vectors, atol=1e-5)
        
        # Verify shape
        assert embed_state.z_mu.shape == (batch_size, seq_len, 8)

    def test_gradient_blocking(self, embedding_config, rng_key):
        """
        Critical Test: Ensure forward_inference returns 0 gradients for inputs.
        Discrete inputs cannot receive gradients.
        """
        params, structure = create_pc_graph(embedding_config, rng_key)
        batch_size = 2
        
        input_indices = jnp.ones((batch_size, 5)) 
        clamps = {"indices": input_indices}
        
        state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)
        node_info = structure.nodes["embed"]
        node_state = state.nodes["embed"]
        node_params = params.nodes["embed"]
        
        # Inputs gathered from the "indices" node
        inputs = {"indices->embed:in": input_indices}
        
        # Call forward_inference directly
        node_cls = get_node_class("embedding")
        new_state, input_grads = node_cls.forward_inference(node_params, inputs, node_state, node_info)
        
        # Check input gradients
        grad = input_grads["indices->embed:in"]
        
        assert jnp.all(grad == 0.0), "Embedding node must return zero gradients for inputs"
        assert grad.shape == input_indices.shape

    def test_learning_updates_embeddings(self, embedding_config, rng_key):
        """
        Test that training actually updates the embedding matrix.
        PC Error (z_latent - z_mu) should drive updates to W[indices].
        """
        params, structure = create_pc_graph(embedding_config, rng_key)
        
        optimizer = create_optimizer({"type": "sgd", "lr": 1.0}) 
        opt_state = optimizer.init(params)
        
        batch_size = 1
        idx = 5
        
        # The 'indices' source node will propagate this float type to z_mu/error,
        # keeping the JAX loop types consistent.
        input_indices = jnp.full((batch_size, 5), idx, dtype=jnp.float32)
        
        batch = {
            "x": input_indices,
            "y": jnp.zeros((batch_size, 10))
        }
        
        # Snapshot old weights
        old_embeddings = params.nodes["embed"].weights["embeddings"]
        old_row_5 = old_embeddings[idx].copy()
        old_row_0 = old_embeddings[0].copy() 
        
        # Run one training step
        rng_key, step_key = jax.random.split(rng_key)
        new_params, _, loss, final_state = train_step(
            params, opt_state, batch, structure, optimizer, step_key, 
            infer_steps=5, eta_infer=0.1
        )
        
        new_embeddings = new_params.nodes["embed"].weights["embeddings"]
        new_row_5 = new_embeddings[idx]
        new_row_0 = new_embeddings[0]
        
        # The row corresponding to the input index SHOULD change
        diff = jnp.abs(new_row_5 - old_row_5)
        assert jnp.max(diff) > 1e-5, "Embedding weights for active index did not update"
        
        # The row corresponding to unused index SHOULD NOT change
        diff_unused = jnp.abs(new_row_0 - old_row_0)
        assert jnp.max(diff_unused) < 1e-6, "Embedding weights for inactive index changed unexpectedly"

    def test_forward_squeeze_logic(self, embedding_config, rng_key):
        """Test the logic handling (batch, seq, 1) inputs."""
        params, structure = create_pc_graph(embedding_config, rng_key)
        node_cls = get_node_class("embedding")
        
        # Create input with extra dimension (batch, seq, 1)
        input_expanded = jnp.zeros((2, 5, 1))
        inputs = {"mock_edge": input_expanded}
        
        # State and Params
        state = initialize_state(structure, 2, rng_key, params=params).nodes["embed"]
        node_params = params.nodes["embed"]
        node_info = structure.nodes["embed"]
        
        # Should not crash and should squeeze internally
        _, new_state = node_cls.forward(node_params, inputs, state, node_info)
        
        assert new_state.z_mu.shape == (2, 5, 8)

class TestTransformerBlock:

    @pytest.fixture
    def single_block_config(self):
        """Standard config for testing a single block node."""
        return {
            "node_list": [
                {"name": "input", "shape": (10, 32), "type": "linear"},
                {
                    "name": "block", 
                    "shape": (10, 32), 
                    "type": "transformer_block",
                    "embed_dim": 32, "num_heads": 4, "mlp_dim": 64, "use_rope": True
                },
                {"name": "output", "shape": (10, 32), "type": "linear"},
            ],
            "edge_list": [
                {"source_name": "input", "target_name": "block", "slot": "in"},
                {"source_name": "block", "target_name": "output", "slot": "in"},
            ],
            "task_map": {"x": "input", "y": "output"},
        }

    def test_block_forward_shapes(self, single_block_config, rng_key):
        """Verify output shapes are preserved through Attention and MLP."""
        params, structure = create_pc_graph(single_block_config, rng_key)
        batch_size = 2
        
        # Random inputs
        x = jax.random.normal(rng_key, (batch_size, 10, 32))
        clamps = {"input": x, "output": jnp.zeros_like(x)}
        
        state = initialize_state(structure, batch_size, rng_key, clamps=clamps, params=params)
        final_state = run_inference(params, state, clamps, structure, infer_steps=1)
        
        block_latent = final_state.nodes["block"].z_latent
        assert block_latent.shape == (batch_size, 10, 32)
        assert jnp.abs(block_latent).mean() > 0.0

    def test_causal_masking(self, single_block_config, rng_key):
        """Verify future tokens do not affect past tokens."""
        params, structure = create_pc_graph(single_block_config, rng_key)
        batch_size = 1
        
        x_base = jax.random.normal(rng_key, (batch_size, 10, 32))
        clamps_base = {"input": x_base, "output": jnp.zeros_like(x_base)}
        
        state_1 = initialize_state(structure, batch_size, rng_key, clamps=clamps_base, 
                                   state_init_config={"type": "feedforward"}, params=params)
        out_1 = state_1.nodes["block"].z_mu
        
        # Modified run: Change ONLY the last token
        x_mod = x_base.at[:, -1, :].add(5.0)
        clamps_mod = {"input": x_mod, "output": jnp.zeros_like(x_base)}
        
        state_2 = initialize_state(structure, batch_size, rng_key, clamps=clamps_mod, 
                                   state_init_config={"type": "feedforward"}, params=params)
        out_2 = state_2.nodes["block"].z_mu
        
        # Check First Token (Should be Identical - Masking Working)
        diff_first = jnp.abs(out_1[:, 0, :] - out_2[:, 0, :]).max()
        assert diff_first < 1e-5, f"Causal mask failed! Past changed by {diff_first}"
        
        # Check Last Token (Should Change - Self Attention Working)
        diff_last = jnp.abs(out_1[:, -1, :] - out_2[:, -1, :]).max()
        assert diff_last > 1e-4, "Self-attention failed! Last token ignored input change."

    def test_block_learning(self, single_block_config, rng_key):
        """Verify gradients propagate and loss decreases (overfitting test)."""
        params, structure = create_pc_graph(single_block_config, rng_key)
        optimizer = create_optimizer({"type": "adam", "lr": 0.01})
        opt_state = optimizer.init(params)
        
        target = jax.random.normal(rng_key, (4, 10, 32))
        batch = {"x": target, "y": target}
        
        losses = []
        for _ in range(5):
            rng_key, step_key = jax.random.split(rng_key)
            params, opt_state, loss, _ = train_step(
                params, opt_state, batch, structure, optimizer, step_key,
                infer_steps=5, eta_infer=0.1
            )
            losses.append(loss)
            
        assert losses[-1] < losses[0], "Transformer block failed to learn."    

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))