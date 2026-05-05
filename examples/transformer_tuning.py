"""
Hyperparameter Tuning — Transformer on Tiny Shakespeare (multi-GPU)

Architecture (auto-tuned via Optuna)::

    input ──→ Embedding ──→ [MhaResidual ──→ LnMlp1 ──→ Mlp2Residual] x depth ──→ VocabProjection

    Tunable: embed_dim, mlp_dim, num_heads, depth, infer_steps, eta_infer, lr, weight_init_std
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import jax
import optax
import optuna
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.training import train_pcn, evaluate_transformer
from fabricpc.tuning.bayesian_tuner import BayesianTuner
from fabricpc.utils.data import CharDataLoader

# --- Model Factory ---


def trial_model(config, rng_key):
    """Create the PC transformer graph; returns (params, structure)."""
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

    inference = InferenceSGD(
        eta_infer=config.get("eta_infer", 0.05),
        infer_steps=config.get("infer_steps", 20),
    )

    # Create the structure directly
    structure = create_deep_transformer(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        inference=inference,
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


# --- Multi-GPU Train/Eval ---


def multi_gpu_train_eval(params, structure, train_loader, val_loader, config, rng_key):
    train_key, eval_key = jax.random.split(rng_key)

    lr = config.get("lr", 1e-3)
    optimizer = optax.adam(lr)

    trained_params, _, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=config,
        rng_key=train_key,
        verbose=False,
        use_tqdm=False,
    )

    metrics = evaluate_transformer(
        trained_params, structure, val_loader, config, eval_key
    )

    alpha = 0.5
    energy = metrics.get("energy", 0.0)
    perplexity = metrics.get("perplexity", 0.0)

    combined = alpha * energy + (1 - alpha) * perplexity
    metrics["combined_loss"] = combined
    return metrics


# --- MAIN — TUNING SETUP ---

if __name__ == "__main__":
    seq_len = 32
    batch_size = 32
    max_tuning_samples = 50_000

    train_loader = CharDataLoader(
        "train",
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        max_samples=max_tuning_samples,
    )
    val_loader = CharDataLoader(
        "validation",
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=False,
    )
    vocab_size = train_loader.vocab_size

    print(
        f"Tuning subset: {train_loader.num_sequences} train sequences, "
        f"{val_loader.num_sequences} val sequences"
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
