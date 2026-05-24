import jax
import optuna
import os
import random
import numpy as np
import time
from datetime import datetime
import optax as _optax
from pathlib import Path
from typing import Callable, Any, Dict, Tuple, Optional

from fabricpc.training import train_autoregressive, evaluate_autoregressive
from fabricpc.core.types import GraphParams, GraphStructure

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class BayesianTuner:
    """
    Two-phase Bayesian Hyperparameter Tuner using Optuna for FabricPC models.

    Phase 1 — Architecture search: minimize energy, prune unstable trials early.
    Phase 2 — Continuous fine-tuning: fix architecture, minimize perplexity.
    """

    def __init__(
        self,
        train_loader: Any,
        val_loader: Any,
        trial_model: Callable[[Dict[str, Any], jax.Array], Tuple[GraphParams, GraphStructure]],
        base_config: Dict[str, Any],
        study_name: str = "fabricpc_tuning",
        storage=None,
        log_file: Optional[str] = "tuning_results.txt",
        energy_threshold: float = 300,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trial_model = trial_model
        self.base_config = base_config
        self.study_name = study_name
        self.storage = storage
        self.log_file = log_file
        self.energy_threshold = energy_threshold

        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_trial(
        self,
        trial: optuna.Trial,
        config: Dict[str, Any],
        phase: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Run a single train + eval pass. Prunes if energy is unstable.
        Returns (score, metrics) where score is energy (phase 1) or perplexity (phase 2).
        """
        current_seed = 42 + trial.number
        set_seed(current_seed)
        rng_key = jax.random.PRNGKey(current_seed)
        model_key, train_key, eval_key = jax.random.split(rng_key, 3)

        try:
            result = self.trial_model(config, model_key)
            if len(result) == 4:
                params, structure, train_loader, val_loader = result
            else:
                params, structure = result
                train_loader = self.train_loader
                val_loader = self.val_loader
        except Exception as e:
            print(f"  Trial {trial.number} pruned, model creation failed: {e}")
            raise optuna.TrialPruned()
  
        optimizer = _optax.adam(config.get("lr", 1e-3))
        train_config = {**config, "use_causal_mask": True}

        def iter_callback(epoch_idx, batch_idx, energy):
            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  [Phase {phase}] Trial {trial.number} | "
                    f"Epoch {epoch_idx + 1} | Batch {batch_idx + 1} | Energy: {energy:.4f}"
                )
            return energy  

        try:
            trained_params, iter_results, _ = train_autoregressive(
                params, structure, train_loader,
                optimizer, train_config, train_key,
                verbose=False,
                iter_callback=iter_callback,
            )
        except Exception as e:
            print(f"  Trial {trial.number} failed during training: {e}")
            raise optuna.TrialPruned()

        # Check energy stability from last epoch
        last_epoch_energies = [e for e in (iter_results[-1] if iter_results else []) if e is not None]
        if last_epoch_energies:
            avg_energy = sum(last_epoch_energies) / len(last_epoch_energies)
            if avg_energy != avg_energy or avg_energy > self.energy_threshold:
                reason = f"Unstable energy: {avg_energy:.4f}"
                trial.set_user_attr("prune_reason", reason)
                print(f"  Trial {trial.number} pruned — {reason}")
                raise optuna.TrialPruned()
        else:
            avg_energy = float("inf")

        try:
            metrics = evaluate_autoregressive(
                trained_params, structure, val_loader, train_config, eval_key
            )
        except Exception as e:
            print(f"  Trial {trial.number} failed during eval: {e}")
            raise optuna.TrialPruned()

        metrics["energy"] = avg_energy
        score = avg_energy if phase == 1 else metrics.get("perplexity", float("inf"))
        return score, metrics

    def _log(self, phase: int, trial_number: int, duration: float, score: float, metrics: Dict, config: Dict):
        if not self.log_file:
            return
        line = (
            f"[Phase {phase}] Trial {trial_number:<4} | {duration:<7.1f}s | "
            f"Energy: {metrics.get('energy', 0.0):<10.4f} | "
            f"PPL: {metrics.get('perplexity', 0.0):<10.4f} | "
            f"Loss: {metrics.get('loss', 0.0):<8.4f} | "
            f"LR: {config.get('lr', 'N/A')} | "
            f"Embed: {config.get('embed_dim', 'N/A')} | "
            f"MLP: {config.get('mlp_dim', 'N/A')} | "
            f"Heads: {config.get('num_heads', 'N/A')} | "
            f"Depth: {config.get('depth', 'N/A')} | "
            f"Infer: {config.get('infer_steps', 'N/A')} | "
            f"Eta: {config.get('eta_infer', 'N/A')}\n"
        )
        with open(self.log_file, "a") as f:
            f.write(line)
        print(f"  → Score: {score:.4f} | PPL: {metrics.get('perplexity', 0.0):.4f} | Energy: {metrics.get('energy', 0.0):.4f}")

    # ------------------------------------------------------------------
    # Phase 1 — architecture search, minimize energy
    # ------------------------------------------------------------------

    def tune_phase1(
        self,
        n_trials: int,
        search_space: Callable[[optuna.Trial], Dict[str, Any]],
    ) -> optuna.Study:
        study = optuna.create_study(
            study_name=f"{self.study_name}_phase1_energy",
            storage=self.storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=15, reduction_factor=2),
        )

        def objective(trial):
            sampled = search_space(trial)
            config = {**self.base_config, **sampled}
            print(f"\n[Phase 1] Trial {trial.number} | {sampled}")
            start = time.time()
            score, metrics = self._run_trial(trial, config, phase=1)
            self._log(1, trial.number, time.time() - start, score, metrics, config)
            return score

        study.optimize(objective, n_trials=n_trials)
        return study

    # ------------------------------------------------------------------
    # Phase 2 — continuous fine-tuning, minimize perplexity
    # ------------------------------------------------------------------

    def tune_phase2(
        self,
        n_trials: int,
        best_params: Dict[str, Any],
        search_space: Callable[[optuna.Trial, Dict], Dict[str, Any]],
    ) -> optuna.Study:
        study = optuna.create_study(
            study_name=f"{self.study_name}_phase2_ppl",
            storage=self.storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=3,
                multivariate=True,
                group=True,
            ),
        )

        def objective(trial):
            continuous = search_space(trial, best_params)
            config = {**self.base_config, **best_params, **continuous}
            print(f"\n[Phase 2] Trial {trial.number} | fine-tuning: {continuous}")
            start = time.time()
            score, metrics = self._run_trial(trial, config, phase=2)
            self._log(2, trial.number, time.time() - start, score, metrics, config)
            return score

        study.optimize(objective, n_trials=n_trials)
        return study

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def tune(
        self,
        phase1_search_space: Callable[[optuna.Trial], Dict[str, Any]],
        phase2_search_space: Callable[[optuna.Trial, Dict], Dict[str, Any]],
        n_trials_phase1: int = 30,
        n_trials_phase2: int = 20,
        save_best_to: Optional[str] = "tuning/best_hyperparameters.txt",
    ) -> Dict[str, Any]:
        """
        Run the full two-phase tuning pipeline.

        Phase 1: Search architecture + all params, minimize energy, prune unstable trials.
        Phase 2: Fix architecture, fine-tune lr/eta_infer/infer_steps, minimize perplexity.

        Returns a summary dict with best params from both phases.
        """
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("=" * 120 + "\n")
                f.write(f"RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("PHASE 1 — Architecture Search (minimize energy)\n")
                f.write("=" * 120 + "\n")

        print("\n" + "=" * 60)
        print("PHASE 1: Architecture search — minimizing energy")
        print("=" * 60)

        study1 = self.tune_phase1(n_trials_phase1, phase1_search_space)

        if not study1.best_trial:
            print("Phase 1 produced no successful trials.")
            return {}

        best_energy = study1.best_value
        best_params = study1.best_params
        print(f"\nPhase 1 complete — Best energy: {best_energy:.4f}")
        print(f"Best architecture: {best_params}")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("\n" + "=" * 120 + "\n")
                f.write("PHASE 2 — Continuous Fine-tuning (minimize perplexity)\n")
                f.write(f"Fixed architecture: {best_params}\n")
                f.write("=" * 120 + "\n")

        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning — minimizing perplexity")
        print(f"Architecture fixed: {best_params}")
        print("=" * 60)

        study2 = self.tune_phase2(n_trials_phase2, best_params, phase2_search_space)

        if not study2.best_trial:
            print("Phase 2 produced no successful trials.")
            return {"phase1_best_energy": best_energy, "phase1_best_params": best_params}

        best_ppl = study2.best_value
        best_continuous = study2.best_params
        final_params = {**best_params, **best_continuous}

        print(f"\nPhase 2 complete — Best perplexity: {best_ppl:.4f}")

        if save_best_to:
            Path(save_best_to).parent.mkdir(exist_ok=True)
            with open(save_best_to, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("BEST HYPERPARAMETERS\n")
                f.write("=" * 60 + "\n\n")
                f.write("PHASE 1 — Best for Energy\n")
                f.write(f"Best Energy: {best_energy:.6f}\n")
                f.write("-" * 40 + "\n")
                for k, v in best_params.items():
                    f.write(f"{k} = {v}\n")
                f.write("\nPHASE 2 — Best for Perplexity\n")
                f.write(f"Best PPL: {best_ppl:.6f}\n")
                f.write("-" * 40 + "\n")
                for k, v in final_params.items():
                    f.write(f"{k} = {v}\n")
            print(f"Best hyperparameters saved to: {save_best_to}")

        return {
            "phase1_best_energy": best_energy,
            "phase1_best_params": best_params,
            "phase2_best_ppl": best_ppl,
            "phase2_best_params": best_continuous,
            "final_params": final_params,
        }