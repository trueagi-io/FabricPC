
"""
Hyperparameter Tuning for Transformer on Tiny Shakespeare
=========================================================
"""


import os
import jax
import jax.numpy as jnp
import torch
import optuna
from torch.utils.data import DataLoader, Dataset
from fabricpc.graph import create_pc_graph
from fabricpc.nodes.transformer import create_deep_transformer
# from fabricpc.tuning import BayesianTuner
import importlib.util

spec = importlib.util.spec_from_file_location(
    "bayesian_tuner",
    "/home/actpc/FabricPC/tuning/bayesian_tuner.py"
)
bayes_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bayes_module)

BayesianTuner = bayes_module.BayesianTuner

# ----------------------------------------------------------------------
# DATA LOADING (Reused from transformer_demo.py)
# ----------------------------------------------------------------------
def load_data(path="data/tiny_shakespeare.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    else:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    data = [char_to_ix[ch] for ch in text]

    return data, vocab_size, char_to_ix, ix_to_char

def split_data(data, train_frac=0.8, val_frac=0.1):
    N = len(data)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    train_data = data[:n_train]
    val_data   = data[n_train:n_train + n_val]
    return train_data, val_data

class TextDataset(Dataset):
    def __init__(self, data, seq_len, vocab_size):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, i):
        x = torch.tensor(self.data[i:i+self.seq_len], dtype=torch.float32)
        y = torch.tensor(self.data[i+1:i+self.seq_len+1], dtype=torch.long)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.vocab_size).float()
        return x, y_one_hot

# ----------------------------------------------------------------------
# FACTORY & SEARCH SPACE
# ----------------------------------------------------------------------

def trial_model(config, rng_key):
    """
    Creates the PC graph structure and parameters from a config dict.
    """
    # Extract model architecture params from config, with defaults
    embed_dim = config.get("embed_dim", 64)
    num_heads = config.get("num_heads", 4)
    mlp_dim = config.get("mlp_dim", 128)
    depth = config.get("depth", 1)
    seq_len = config.get("seq_len", 32)
    vocab_size = config.get("vocab_size", 65) 

    transformer_config = create_deep_transformer(
        depth=depth,                
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,        
        vocab_size=vocab_size   
    )
    
    return create_pc_graph(transformer_config, rng_key)

def search_space_transformer(trial):
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64])
    mlp_dim = trial.suggest_categorical("mlp_dim", [64, 128])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    
    num_epochs = trial.suggest_int("num_epochs", 1, 3) # Keep low for demo speed
    infer_steps = trial.suggest_int("infer_steps", 10, 30)
    eta_infer = trial.suggest_float("eta_infer", 0.01, 0.2)
    
    return {
        "lr": lr,
        "optimizer": {"type": "adam", "lr": lr},
        "embed_dim": embed_dim,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "num_epochs": num_epochs,
        "infer_steps": infer_steps,
        "eta_infer": eta_infer,
    }

if __name__ == "__main__":
    print("Setting up data...")
    data, vocab_size, _, _ = load_data()
    train_data, val_data = split_data(data)
    
    seq_len = 32
    batch_size = 32
    
    train_dataset = TextDataset(train_data, seq_len, vocab_size)
    val_dataset   = TextDataset(val_data, seq_len, vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # Drop last to ensure shapes

    base_config = {
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "depth": 1,
    }

    tuner = BayesianTuner(
        train_loader=train_loader,
        val_loader=val_loader,
        trial_model=trial_model,
        base_config=base_config,
        metric="accuracy",
        direction="maximize",
        study_name="transformer_optimization",
        log_file="transformer_tuning_results.jsonl"
    )

    print("Starting Trials")
    study = tuner.tune(n_trials=2, search_space=search_space_transformer)

    print("\nBest params:")
    print(study.best_params)
    print(f"Best value: {study.best_value}")
