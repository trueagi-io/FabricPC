"""
Hyperparameter Tuning — Transformer on Tiny Shakespeare (Two-Phase)

Phase 1 — Architecture search: explore embed_dim, mlp_dim, num_heads, depth,
           infer_steps, eta_infer, lr, weight_init_std. Prune unstable trials
           early based on energy explosion. Minimize energy.

Phase 2 — Continuous fine-tuning: fix the winning architecture, search only
           lr, eta_infer, infer_steps in a tight window. Minimize perplexity.

Architecture::
    input -> Embedding -> [MhaResidual -> LnMlp1 -> Mlp2Residual] x depth -> VocabProjection
"""

from jax_setup import set_jax_flags_before_importing_jax
set_jax_flags_before_importing_jax()

import optuna
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.nodes.transformer_v2 import create_deep_transformer
from fabricpc.tuning.bayesian_tuner import BayesianTuner
from fabricpc.utils.data import CharDataLoader, BpeDataLoader
from optuna.storages import JournalStorage, JournalFileStorage

# Model Factory
def trial_model(config, rng_key):
    """Create the PC transformer graph; returns (params, structure, train_loader, val_loader)."""
    embed_dim = config.get("embed_dim", 64)
    num_heads = config.get("num_heads", 4)
    mlp_dim = config.get("mlp_dim", 128)
    depth = config.get("depth", 1)
    seq_len = config.get("seq_len", 32)
    batch_size = config.get("batch_size", 32)
    vocab_size = config.get("vocab_size", 65)
    weight_init_std = config.get("weight_init_std", 0.02)
    use_bpe = config.get("use_bpe", False)

    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned(
            f"embed_dim={embed_dim} not divisible by num_heads={num_heads}"
        )

    inference = InferenceSGDNormClip(
            eta_infer=config.get("eta_infer", 0.1),
            infer_steps=config.get("infer_steps", 20),
            max_norm=5.0,
            latent_decay=0.0,
    )

    if use_bpe:
        train_loader = BpeDataLoader(
            "train", seq_len=seq_len, batch_size=batch_size,
            shuffle=True, seed=42, max_samples=50000
        )
        val_loader = BpeDataLoader(
            "validation", seq_len=seq_len, batch_size=batch_size, shuffle=False
        )
    else:
        train_loader = CharDataLoader(
            "train", seq_len=seq_len, batch_size=batch_size,
            shuffle=True, seed=42, max_samples=50000
        )
        val_loader = CharDataLoader(
            "validation", seq_len=seq_len, batch_size=batch_size, shuffle=False
        )

    structure = create_deep_transformer(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        seq_len=seq_len,
        vocab_size=vocab_size,
        inference=inference,
        weight_init={"type": "normal", "std": weight_init_std},
    )

    params = initialize_params(structure, rng_key)
    return params, structure, train_loader, val_loader

# ==============================================================================
# PHASE 1: Architecture search
# ==============================================================================

def phase1_search_space(trial):
    """
    Full search space for architecture and training hyperparameters.
    Ranges are conservative to avoid NaN/explosion.
    """
    embed_dim = trial.suggest_categorical("embed_dim", [64, 128])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    mlp_dim = trial.suggest_categorical("mlp_dim", [256, 512])
    depth = trial.suggest_int("depth", 1, 4)
    seq_len = trial.suggest_categorical("seq_len", [64, 128])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    min_infer_steps = depth * 3 + 2
    infer_steps = trial.suggest_int("infer_steps", min_infer_steps, min_infer_steps + 10)
    eta_infer = trial.suggest_float("eta_infer", 0.01, 0.15)
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    weight_init_std = trial.suggest_float("weight_init_std", 0.01, 0.05, log=True)

    return {
        "embed_dim": embed_dim, "num_heads": num_heads, "mlp_dim": mlp_dim,
        "depth": depth, "seq_len": seq_len, "batch_size": batch_size,
        "infer_steps": infer_steps, "eta_infer": eta_infer,
        "lr": lr, "weight_init_std": weight_init_std,
    }

# ==============================================================================
# PHASE 2 SEARCH SPACE — Continuous fine-tuning around Phase 1 winner
# ==============================================================================

def phase2_search_space(trial, best_params):
    """
    Fine-tune only continuous training params around the Phase 1 best values.
    Architecture (embed_dim, num_heads, mlp_dim, depth, weight_init_std) is fixed.
    """
    lr = best_params.get("lr", 1e-4)
    eta_infer = best_params.get("eta_infer", 0.1)
    infer_steps = best_params.get("infer_steps", 20)
    depth = best_params.get("depth", 4)
    min_infer_steps = depth * 3 + 2

    return {
        "lr": trial.suggest_float("lr", max(1e-5, lr * 0.5), min(1e-3, lr * 2.0), log=True),
        "eta_infer": trial.suggest_float("eta_infer", max(0.01, eta_infer * 0.7), min(0.2, eta_infer * 1.3)),
        "infer_steps": trial.suggest_int("infer_steps", max(min_infer_steps, infer_steps - 3), infer_steps + 5),
    }

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    seq_len = 128
    batch_size = 32
    max_tuning_samples = 50000
    use_bpe = True  # set to False for character-level tokenizer

    # Used only to get vocab_size training loaders are created per trial in trial_model
    if use_bpe:
        train_loader = BpeDataLoader(
            "train", seq_len=seq_len, batch_size=batch_size,
            shuffle=True, seed=42, max_samples=max_tuning_samples
        )
        val_loader = BpeDataLoader(
            "validation", seq_len=seq_len, batch_size=batch_size, shuffle=False
        )
    else:
        train_loader = CharDataLoader(
            "train", seq_len=seq_len, batch_size=batch_size,
            shuffle=True, seed=42, max_samples=max_tuning_samples
        )
        val_loader = CharDataLoader(
            "validation", seq_len=seq_len, batch_size=batch_size, shuffle=False
        )
    vocab_size = train_loader.vocab_size

    print(
        f"Tuning subset: {train_loader.num_sequences} train sequences, "
        f"{val_loader.num_sequences} val sequences"
    )

    base_config = {
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "num_epochs": 5,
        "use_bpe": use_bpe
    }

    storage = JournalStorage(JournalFileStorage("fabricpc/tuning/optuna_journal.log"))

    tuner = BayesianTuner(
        train_loader=train_loader,
        val_loader=val_loader,
        trial_model=trial_model,
        base_config=base_config,
        study_name="transformer_v2_tuning",
        storage=storage,
        log_file="fabricpc/tuning/transformer_v2_results.txt",
        energy_threshold=300,
    )

    print("\n=== Starting Two-Phase Hyperparameter Search ===")
    results = tuner.tune(
        phase1_search_space=phase1_search_space,
        phase2_search_space=phase2_search_space,
        n_trials_phase1=20,
        n_trials_phase2=15,
        save_best_to="fabricpc/tuning/best_hyperparameters.txt",
    )

    if results:
        print("\n" + "=" * 60)
        print("TUNING COMPLETE")
        print("=" * 60)
        print(f"Phase 1 Best Energy:     {results['phase1_best_energy']:.4f}")
        print(f"Phase 2 Best Perplexity: {results['phase2_best_ppl']:.4f}")
        print(f"\nFinal parameters:")
        for k, v in results["final_params"].items():
            print(f"  {k}: {v}")
        print(f"\nSaved to: fabricpc/tuning/best_hyperparameters.txt")