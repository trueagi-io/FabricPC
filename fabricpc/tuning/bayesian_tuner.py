import jax
import jax.numpy as jnp
import optuna
import os
import random
import numpy as np
import json
import time
from typing import Callable, Any, Dict, Tuple, Optional, Union
from fabricpc.training.train import train_pcn, evaluate_pcn
from fabricpc.core.types import GraphParams, GraphStructure

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class BayesianTuner:
    """
    Bayesian Hyperparameter Tuner using Optuna for FabricPC models.
    """
    def __init__(
        self,
        train_loader: Any,
        val_loader: Any,
        trial_model: Callable[[Dict[str, Any], jax.Array], Tuple[GraphParams, GraphStructure]],
        base_config: Dict[str, Any],
        metric: str = "combined_loss",
        study_name: str = "fabricpc_tuning",
        storage: Optional[str] = None,
        direction: str = "minimize",
        log_file: Optional[str] = "tuning_results.jsonl"
    ):
        """
        Args:
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation.
            trial_model: Function that takes (config_dict, rng_key) and returns (params, structure).
            base_config: Base configuration dictionary. Tuning updates this.
            metric: Metric to minimize/maximize. 
            study_name: Name of the Optuna study.
            storage: Database URL for Optuna storage.
            direction: 'minimize' or 'maximize'.
            log_file: Path to save detailed trial logs.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trial_model = trial_model
        self.base_config = base_config
        self.metric = metric
        self.direction = direction
        self.log_file = log_file
        
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True
        )

    def _suggest_from_config(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate suggestions based on a dictionary configuration.
        
        Format:
        {
            "param_name": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
            "param_int": {"type": "int", "low": 1, "high": 10},
            "param_cat": {"type": "categorical", "choices": ["a", "b", "c"]}
        }
        """
        params = {}
        for name, config in search_space.items():
            param_type = config.get("type")
            if param_type == "float":
                params[name] = trial.suggest_float(
                    name, 
                    config["low"], 
                    config["high"], 
                    log=config.get("log", False)
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(
                    name, 
                    config["low"], 
                    config["high"], 
                    step=config.get("step", 1),
                    log=config.get("log", False)
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, config["choices"])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
        return params

    def tune(
        self,
        n_trials: int,
        search_space: Union[Dict[str, Any], Callable[[optuna.Trial], Dict[str, Any]]],
        callbacks: Optional[list] = None
    ):
        """
        Run the tuning process.

        Args:
            n_trials: Number of trials to run.
            search_space: Either a dictionary defining the search space OR a callable that takes a trial 
                          and returns sampled parameters.
            callbacks: List of Optuna callbacks.
        """
        if isinstance(search_space, dict):
            search_fn = lambda trial: self._suggest_from_config(trial, search_space)
        else:
            search_fn = search_space

        self.study.optimize(
            lambda trial: self._objective(trial, search_fn),
            n_trials=n_trials,
            callbacks=callbacks
        )
        return self.study

    def _log_trial(self, trial_data: Dict[str, Any]):
        """Append trial data to local log file."""
        if self.log_file:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(json.dumps(trial_data) + "\n")

    def _objective(self, trial: optuna.Trial, search_space_fn: Callable[[optuna.Trial], Dict[str, Any]]) -> float:
        start_time = time.time()
        
        # Sample hyperparameters
        sampled_params = search_space_fn(trial)
        
        # Merge with base config
        config = self.base_config.copy()
        config.update(sampled_params)
        
        # Set global seeds
        current_seed = 42 + trial.number
        set_seed(current_seed)
        
        rng_key = jax.random.PRNGKey(current_seed)
        model_key, train_key = jax.random.split(rng_key)
        
        # Build model components
        try:
            params, structure = self.trial_model(config, model_key)
        except Exception as e:
            print(f"Trial {trial.number} pruned due to creation failure: {e}")
            raise optuna.TrialPruned()

        # Training
        try:
            trained_params, iter_results, epoch_results = train_pcn(
                params,
                structure,
                self.train_loader,
                config,
                train_key,
                verbose=False
            )
        except Exception as e:
            print(f"Trial {trial.number} failed during training: {e}")
            return float("inf") if self.direction == "minimize" else float("-inf")

        final_epoch_energies = iter_results[-1] if iter_results else []
        final_energy_mean = float(np.mean(final_epoch_energies)) if final_epoch_energies else float("inf")
        final_energy_std = float(np.std(final_epoch_energies)) if final_epoch_energies else 0.0

        # Evaluation
        eval_metrics = evaluate_pcn(
            trained_params,
            structure,
            self.val_loader,
            config,
            jax.random.PRNGKey(0) 
        )
        
        if self.metric == "combined_loss":
            val_score = eval_metrics.get("loss", float("inf"))
        elif self.metric in eval_metrics:
            val_score = eval_metrics[self.metric]
        else:
             # Fallback
             val_score = eval_metrics.get("loss", float("inf"))

        duration = time.time() - start_time

        # Structured Logging
        log_data = {
            "trial_id": trial.number,
            "params": sampled_params,
            "val_metrics": eval_metrics,
            "train_energy_mean": final_energy_mean,
            "train_energy_std": final_energy_std,
            "duration": duration,
            "status": "COMPLETE"
        }
        self._log_trial(log_data)
             
        return val_score