"""Training utilities for JAX predictive coding networks.

Backprop trainers are provided for performance comparison to PC and as a reference to aid in debugging or tuning of PC training dynamics. These backprop trainers operate on the same graph models ensuring no divergence of model code. If there are cycles in the graph, don't expect backprop to learn meaningful weights in those recurrency paths.
"""

from fabricpc.training.train import train_step, train_pcn, evaluate_pcn
from fabricpc.training.multi_gpu import (
    train_pcn_multi_gpu,
    evaluate_transformer_multi_gpu,
    evaluate_pcn_multi_gpu,
    replicate_params,
    shard_batch,
)
from fabricpc.training.train_autoregressive import (
    train_autoregressive,
    train_step_autoregressive,
    evaluate_autoregressive,
    generate_autoregressive,
)
from fabricpc.training.train_backprop import (
    compute_loss,
    train_step_backprop,
    train_backprop,
    compute_loss_autoregressive,
    train_step_backprop_autoregressive,
    train_backprop_autoregressive,
    evaluate_backprop,
    evaluate_backprop_autoregressive,
)

__all__ = [
    # Predictive coding training
    "train_step",
    "train_pcn",
    "evaluate_pcn",
    # Multi-GPU
    "train_pcn_multi_gpu",
    "evaluate_pcn_multi_gpu",
    "replicate_params",
    "shard_batch",
    # PC Autoregressive
    "train_autoregressive",
    "train_step_autoregressive",
    "evaluate_autoregressive",
    "generate_autoregressive",
    # Backprop training
    "compute_loss",
    "train_step_backprop",
    "train_backprop",
    "evaluate_backprop",
    # Backprop Autoregressive
    "compute_loss_autoregressive",
    "train_step_backprop_autoregressive",
    "train_backprop_autoregressive",
    "evaluate_backprop_autoregressive",
]
