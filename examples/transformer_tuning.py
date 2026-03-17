"""
Hyperparameter Tuning for Transformer on Tiny Shakespeare (Multi-GPU Version)
=====================================================================
"""

from fabricpc.utils.helpers import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import os
import jax
import torch
import optuna
import random
from torch.utils.data import DataLoader, Dataset
from fabricpc.graph import initialize_params
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training.multi_gpu import (
    train_pcn_multi_gpu,
    evaluate_transformer_multi_gpu,
)
from fabricpc.tuning.bayesian_tuner import BayesianTuner


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
def load_data(path="data/tiny_shakespeare.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [char_to_ix[ch] for ch in text]

    return data, vocab_size, char_to_ix, ix_to_char


def split_data(data, train_frac=0.8, val_frac=0.1):
    data = random.sample(data, max(1, int(len(data) * 0.05)))
    N = len(data)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    return train_data, val_data


class TextDataset(Dataset):
    def __init__(self, data, seq_len, vocab_size):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, i):
        x = torch.tensor(self.data[i : i + self.seq_len], dtype=torch.float32)
        y = torch.tensor(self.data[i + 1 : i + self.seq_len + 1], dtype=torch.long)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.vocab_size).float()
        return x, y_one_hot


# ----------------------------------------------------------------------
# MODEL FACTORY — RETURN PARAMS + STRUCTURE FOR MULTI-GPU TRAINING
# ----------------------------------------------------------------------
def trial_model(config, rng_key):
    """
    Creates the PC transformer graph using the provided config.
    Returns (params, structure) for multi-GPU training.
    """
    embed_dim = config.get("embed_dim", 64)
    num_heads = config.get("num_heads", 4)
    mlp_dim = config.get("mlp_dim", 128)
    depth = config.get("depth", 1)
    seq_len = config.get("seq_len", 32)
    vocab_size = config.get("vocab_size", 65)

    # Weight initialization config
    weight_init_std = config.get("weight_init_std", 0.02)
    weight_init = {"type": "normal", "std": weight_init_std}

    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned(
            f"embed_dim={embed_dim} not divisible by num_heads={num_heads}"
        )

    # Create the structure directly
    structure = create_deep_transformer(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        weight_init=weight_init,
    )

    # Initialize params using the structure
    params = initialize_params(structure, rng_key)

    return params, structure


# ----------------------------------------------------------------------
# OPTUNA SEARCH SPACE
# ----------------------------------------------------------------------


# Scaled down since bigger range gives NAN values
def search_space_transformer(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64])
    mlp_dim = trial.suggest_categorical("mlp_dim", [64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])

    infer_steps = trial.suggest_int("infer_steps", 10, 30)
    eta_infer = trial.suggest_float("eta_infer", 0.01, 0.2)
    depth = trial.suggest_int("depth", 1, 12)

    # Tuning weight initialization scale
    weight_init_std = trial.suggest_float("weight_init_std", 0.005, 0.05, log=True)

    return {
        "lr": lr,
        "optimizer": {"type": "adam", "lr": lr},
        "embed_dim": embed_dim,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "depth": depth,
        "infer_steps": infer_steps,
        "eta_infer": eta_infer,
        "weight_init_std": weight_init_std,
    }


# ----------------------------------------------------------------------
# INTEGRATE MULTI-GPU TRAINING INSIDE TUNER PIPELINE
# ----------------------------------------------------------------------
def multi_gpu_train_eval(params, structure, train_loader, val_loader, config, rng_key):
    train_key, eval_key = jax.random.split(rng_key)

    trained_params = train_pcn_multi_gpu(
        params=params,
        structure=structure,
        train_loader=train_loader,
        config=config,
        rng_key=train_key,
        verbose=False,
    )

    metrics = evaluate_transformer_multi_gpu(
        trained_params, structure, val_loader, config, eval_key
    )

    alpha = 0.5
    energy = metrics.get("energy", 0.0)
    perplexity = metrics.get("perplexity", 0.0)

    combined = alpha * energy + (1 - alpha) * perplexity
    metrics["combined_loss"] = combined
    return metrics


# ----------------------------------------------------------------------
# MAIN — TUNING SETUP
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading tiny_shakespeare...")
    data, vocab_size, _, _ = load_data()

    N = len(data)

    print(f"Using all {N} characters ({100}%) of the dataset")

    train_data, val_data = split_data(data)

    seq_len = 32
    batch_size = 32

    train_loader = DataLoader(
        TextDataset(train_data, seq_len, vocab_size),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TextDataset(val_data, seq_len, vocab_size),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    base_config = {
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }

    tuner = BayesianTuner(
        train_loader=train_loader,
        val_loader=val_loader,
        trial_model=trial_model,
        base_config=base_config,
        trainer_fn=multi_gpu_train_eval,
        metric="combined_loss",
        direction="minimize",
        study_name="transformer_multi_gpu_tuning",
        log_file="transformer_multi_gpu_results.jsonl",
    )

    print("\n=== Starting Multi-GPU Hyperparameter Search ===")
    study = tuner.tune(n_trials=30, search_space=search_space_transformer)

    print("\nBest params:")
    print(study.best_params)
    print(f"Best value: {study.best_value}")
