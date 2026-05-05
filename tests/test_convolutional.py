"""
Unit tests for Conv1D, Conv2D, and Conv3D nodes.

Run with: pytest tests/test_convolutional.py -v
"""

import jax
import jax.numpy as jnp
import pytest
from fabricpc.nodes.convolutional import Conv1DNode, Conv2DNode, Conv3DNode
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import KaimingInitializer, NormalInitializer


class TestConv1D:
    """Tests for Conv1D node."""
    
    @pytest.fixture
    def setup(self):
        batch_size = 4
        seq_len = 10
        in_channels = 3
        out_channels = 8
        kernel_size = (3,)
        
        key = jax.random.PRNGKey(0)
        key_params, key_init = jax.random.split(key)
        
        activation = ReLUActivation()
        energy_obj = GaussianEnergy()
        weight_init = KaimingInitializer()
        latent_init = NormalInitializer()
        
        node = Conv1DNode(
            shape=(seq_len, out_channels),
            name="conv1d",
            kernel_size=kernel_size,
            stride=(1,),
            padding="SAME",
            activation=activation,
            energy=energy_obj,
            weight_init=weight_init,
            latent_init=latent_init
        )
        
        config = node._extra_config
        input_shapes = {"src->dst:in": (seq_len, in_channels)}
        node_shape = (seq_len, out_channels)
        
        params = Conv1DNode.initialize_params(
            key_params, node_shape, input_shapes, weight_init, config
        )
        
        node_info = NodeInfo(
            name="conv1d",
            shape=node_shape,
            node_type="Conv1DNode",
            node_class=Conv1DNode,
            node_config=config,
            activation=activation,
            energy=energy_obj,
            latent_init=latent_init,
            weight_init=weight_init,
            slots={},
            in_degree=1,
            out_degree=1,
            in_edges=("src->dst:in",),
            out_edges=(),
        )
        
        z_latent = jax.random.normal(key_init, (batch_size, seq_len, out_channels))
        state = NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros_like(z_latent),
            error=jnp.zeros_like(z_latent),
            energy=jnp.zeros(batch_size),
            pre_activation=jnp.zeros_like(z_latent),
            latent_grad=jnp.zeros_like(z_latent),
        )
        
        inputs = {"src->dst:in": jax.random.normal(key, (batch_size, seq_len, in_channels))}
        
        return params, state, node_info, inputs
    
    def test_initialize_params(self, setup):
        """Test that parameters are initialized with correct shapes."""
        params, _, _, _ = setup
        assert params.weights["src->dst:in"].shape == (3, 3, 8)
        assert params.biases["b"].shape == (1, 1, 8)
    
    def test_forward_output_shape(self, setup):
        """Test forward pass output shapes."""
        params, state, node_info, inputs = setup
        energy, new_state = Conv1DNode.forward(params, inputs, state, node_info)
        
        assert new_state.z_mu.shape == state.z_latent.shape
        assert new_state.error.shape == state.z_latent.shape
        assert new_state.energy.shape == (4,)
        assert energy.shape == ()
    
    def test_forward_inference(self, setup):
        """Test inference pass using base class autodiff."""
        params, state, node_info, inputs = setup
        new_state, input_grads = Conv1DNode.forward_inference(
            params, inputs, state, node_info, is_clamped=False
        )
        
        assert "src->dst:in" in input_grads
        assert input_grads["src->dst:in"].shape == inputs["src->dst:in"].shape
    
    def test_forward_learning(self, setup):
        """Test learning pass using base class autodiff."""
        params, state, node_info, inputs = setup
        new_state, param_grads = Conv1DNode.forward_learning(
            params, inputs, state, node_info
        )
        
        assert param_grads.weights["src->dst:in"].shape == params.weights["src->dst:in"].shape


class TestConv2D:
    """Tests for Conv2D node."""
    
    @pytest.fixture
    def setup(self):
        batch_size = 4
        h_in, w_in = 28, 28
        in_channels = 3
        out_channels = 16
        kernel_size = (3, 3)
        
        key = jax.random.PRNGKey(0)
        key_params, key_init = jax.random.split(key)
        
        activation = ReLUActivation()
        energy_obj = GaussianEnergy()
        weight_init = KaimingInitializer()
        latent_init = NormalInitializer()
        
        node = Conv2DNode(
            shape=(h_in, w_in, out_channels),
            name="conv2d",
            kernel_size=kernel_size,
            stride=(1, 1),
            padding="SAME",
            activation=activation,
            energy=energy_obj,
            weight_init=weight_init,
            latent_init=latent_init
        )
        
        config = node._extra_config
        input_shapes = {"src->dst:in": (h_in, w_in, in_channels)}
        node_shape = (h_in, w_in, out_channels)
        
        params = Conv2DNode.initialize_params(
            key_params, node_shape, input_shapes, weight_init, config
        )
        
        node_info = NodeInfo(
            name="conv2d",
            shape=node_shape,
            node_type="Conv2DNode",
            node_class=Conv2DNode,
            node_config=config,
            activation=activation,
            energy=energy_obj,
            latent_init=latent_init,
            weight_init=weight_init,
            slots={},
            in_degree=1,
            out_degree=1,
            in_edges=("src->dst:in",),
            out_edges=(),
        )
        
        z_latent = jax.random.normal(key_init, (batch_size, h_in, w_in, out_channels))
        state = NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros_like(z_latent),
            error=jnp.zeros_like(z_latent),
            energy=jnp.zeros(batch_size),
            pre_activation=jnp.zeros_like(z_latent),
            latent_grad=jnp.zeros_like(z_latent),
        )
        
        inputs = {"src->dst:in": jax.random.normal(key, (batch_size, h_in, w_in, in_channels))}
        
        return params, state, node_info, inputs
    
    def test_initialize_params(self, setup):
        """Test that parameters are initialized with correct shapes."""
        params, _, _, _ = setup
        assert params.weights["src->dst:in"].shape == (3, 3, 3, 16)
        assert params.biases["b"].shape == (1, 1, 1, 16)
    
    def test_forward_output_shape(self, setup):
        """Test forward pass output shapes."""
        params, state, node_info, inputs = setup
        energy, new_state = Conv2DNode.forward(params, inputs, state, node_info)
        
        assert new_state.z_mu.shape == state.z_latent.shape
        assert new_state.error.shape == state.z_latent.shape
        assert new_state.energy.shape == (4,)
        assert energy.shape == ()
    
    def test_forward_inference(self, setup):
        """Test inference pass."""
        params, state, node_info, inputs = setup
        new_state, input_grads = Conv2DNode.forward_inference(
            params, inputs, state, node_info, is_clamped=False
        )
        
        assert "src->dst:in" in input_grads
        assert input_grads["src->dst:in"].shape == inputs["src->dst:in"].shape
    
    def test_forward_learning(self, setup):
        """Test learning pass."""
        params, state, node_info, inputs = setup
        new_state, param_grads = Conv2DNode.forward_learning(
            params, inputs, state, node_info
        )
        
        assert param_grads.weights["src->dst:in"].shape == params.weights["src->dst:in"].shape


class TestConv3D:
    """Tests for Conv3D node."""
    
    @pytest.fixture
    def setup(self):
        batch_size = 2
        d_in, h_in, w_in = 10, 10, 10
        in_channels = 3
        out_channels = 8
        kernel_size = (3, 3, 3)
        
        key = jax.random.PRNGKey(0)
        key_params, key_init = jax.random.split(key)
        
        activation = ReLUActivation()
        energy_obj = GaussianEnergy()
        weight_init = KaimingInitializer()
        latent_init = NormalInitializer()
        
        node = Conv3DNode(
            shape=(d_in, h_in, w_in, out_channels),
            name="conv3d",
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding="SAME",
            activation=activation,
            energy=energy_obj,
            weight_init=weight_init,
            latent_init=latent_init
        )
        
        config = node._extra_config
        input_shapes = {"src->dst:in": (d_in, h_in, w_in, in_channels)}
        node_shape = (d_in, h_in, w_in, out_channels)
        
        params = Conv3DNode.initialize_params(
            key_params, node_shape, input_shapes, weight_init, config
        )
        
        node_info = NodeInfo(
            name="conv3d",
            shape=node_shape,
            node_type="Conv3DNode",
            node_class=Conv3DNode,
            node_config=config,
            activation=activation,
            energy=energy_obj,
            latent_init=latent_init,
            weight_init=weight_init,
            slots={},
            in_degree=1,
            out_degree=1,
            in_edges=("src->dst:in",),
            out_edges=(),
        )
        
        z_latent = jax.random.normal(key_init, (batch_size, d_in, h_in, w_in, out_channels))
        state = NodeState(
            z_latent=z_latent,
            z_mu=jnp.zeros_like(z_latent),
            error=jnp.zeros_like(z_latent),
            energy=jnp.zeros(batch_size),
            pre_activation=jnp.zeros_like(z_latent),
            latent_grad=jnp.zeros_like(z_latent),
        )
        
        inputs = {"src->dst:in": jax.random.normal(key, (batch_size, d_in, h_in, w_in, in_channels))}
        
        return params, state, node_info, inputs
    
    def test_initialize_params(self, setup):
        """Test that parameters are initialized with correct shapes."""
        params, _, _, _ = setup
        assert params.weights["src->dst:in"].shape == (3, 3, 3, 3, 8)
        assert params.biases["b"].shape == (1, 1, 1, 1, 8)
    
    def test_forward_output_shape(self, setup):
        """Test forward pass output shapes."""
        params, state, node_info, inputs = setup
        energy, new_state = Conv3DNode.forward(params, inputs, state, node_info)
        
        assert new_state.z_mu.shape == state.z_latent.shape
        assert new_state.error.shape == state.z_latent.shape
        assert new_state.energy.shape == (2,)
        assert energy.shape == ()
    
    def test_forward_inference(self, setup):
        """Test inference pass."""
        params, state, node_info, inputs = setup
        new_state, input_grads = Conv3DNode.forward_inference(
            params, inputs, state, node_info, is_clamped=False
        )
        
        assert "src->dst:in" in input_grads
        assert input_grads["src->dst:in"].shape == inputs["src->dst:in"].shape
    
    def test_forward_learning(self, setup):
        """Test learning pass."""
        params, state, node_info, inputs = setup
        new_state, param_grads = Conv3DNode.forward_learning(
            params, inputs, state, node_info
        )
        
        assert param_grads.weights["src->dst:in"].shape == params.weights["src->dst:in"].shape
