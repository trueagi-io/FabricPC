import pytest
import torch
from hypothesis import given, strategies as st

from fabricpc.models.graph_net import PCGraphNet


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    dim_a=st.integers(min_value=2, max_value=6),
    dim_b=st.integers(min_value=2, max_value=6),
)
def test_allocate_respects_batch_and_dims(batch_size, dim_a, dim_b):
    cfg = {
        "device": "cpu",
        "node_list": [
            {
                "name": "a",
                "dim": dim_a,
                "type": "linear",
                "activation": {"type": "sigmoid"},
            },
            {
                "name": "b",
                "dim": dim_b,
                "type": "linear",
                "activation": {"type": "sigmoid"},
            },
        ],
        "edge_list": [
            {"source_name": "a", "target_name": "b", "slot": "in"},
        ],
        "task_map": {"x": "a", "y": "b"},
        "T_infer": 1,
        "eta_infer": 0.05,
        "optimizer": {"type": "adam", "lr": 1e-3},
    }
    model = PCGraphNet(config=cfg)
    model.allocate_node_states(batch_size=batch_size, device=torch.device("cpu"))
    for n in model.node_dictionary.values():
        assert n.z_latent.shape == (batch_size, n.dim)
        assert n.error.shape == (batch_size, n.dim)
        assert n.z_mu.shape == (batch_size, n.dim)
        assert n.pre_activation_val.shape == (batch_size, n.dim)
        assert n.gain_mod_error.shape == (batch_size, n.dim)
