import pytest
import torch

from fabricpc.models.graph_net import PCGraphNet


def test_allocate_and_tensor_shapes(small_graph_config, device):
    small_graph_config["device"] = str(device)
    model = PCGraphNet(config=small_graph_config)

    B = 5
    model.allocate_node_states(batch_size=B, device=device)

    a = model.node_dictionary["a"]
    b = model.node_dictionary["b"]

    # Each node should have allocated tensors with correct shapes
    for node in (a, b):
        assert node.z_latent.shape == (B, node.dim)
        assert node.error.shape == (B, node.dim)
        assert node.z_mu.shape == (B, node.dim)
        assert node.pre_activation_val.shape == (B, node.dim)
        assert node.gain_mod_error.shape == (B, node.dim)


def test_projection_shapes_match(small_graph_config, device):
    small_graph_config["device"] = str(device)
    model = PCGraphNet(config=small_graph_config)
    B = 3
    x = torch.randn(B, 4, device=device)
    y = torch.randn(B, 3, device=device)
    model.init_latents(clamp_dict={"a": x, "b": y}, batch_size=B, device=device)
    model.update_projections()
    # After projection, z_mu for node b should be (B, 3)
    b = model.node_dictionary["b"]
    assert b.z_mu.shape == (B, 3)
