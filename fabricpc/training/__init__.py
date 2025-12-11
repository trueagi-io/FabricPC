"""
Training utilities for JAX predictive coding networks.
"""

from fabricpc.training.train import train_step, train_pcn, evaluate_pcn, compute_local_weight_gradients
from fabricpc.training.optimizers import create_optimizer
from fabricpc.training.multi_gpu import (
    train_pcn_multi_gpu,
    evaluate_pcn_multi_gpu,
    replicate_params,
    shard_batch,
)
from fabricpc.training.data_utils import (
    OneHotWrapper,
    # TODO: jax data loaders)
)
from fabricpc.training.train_backprop import (
    compute_loss,
    train_step_backprop,
    train_backprop,
    compute_loss_autoregressive,
    train_step_backprop_autoregressive,
    train_backprop_autoregressive,
    evaluate_backprop,
)


__all__ = [
    # Predictive coding training
    "train_step",
    "train_pcn",
    "evaluate_pcn",
    "compute_local_weight_gradients",
    "create_optimizer",
    # Multi-GPU
    "train_pcn_multi_gpu",
    "evaluate_pcn_multi_gpu",
    "replicate_params",
    "shard_batch",
    # Backprop training
    "compute_loss",
    "train_step_backprop",
    "train_backprop",
    "compute_loss_autoregressive",
    "train_step_backprop_autoregressive",
    "train_backprop_autoregressive",
    "evaluate_backprop",
]
