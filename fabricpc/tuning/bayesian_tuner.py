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
    os.environ["PYTHONHASHSEED"] = str(seed)


class BayesianTuner:
    """
    Bayesian Hyperparameter Tuner using Optuna for FabricPC models.
    """

    def __init__(
        self,
        train_loader: Any,
        val_loader: Any,
        trial_model: Callable[
            [Dict[str, Any], jax.Array], Tuple[GraphParams, GraphStructure]
        ],
        base_config: Dict[str, Any],
        metric: str = "combined_loss",
        study_name: str = "fabricpc_tuning",
        storage: Optional[str] = None,
        direction: str = "minimize",
        log_file: Optional[str] = "transformer_multi_gpu_results.jsonl",
        trainer_fn: Optional[Callable] = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trial_model = trial_model
        self.base_config = base_config
        self.metric = metric
        self.direction = direction
        self.log_file = log_file
        self.trainer_fn = trainer_fn

        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True,
        )

    def _suggest_from_config(
        self, trial: optuna.Trial, search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate suggestions based on a dictionary configuration."""
        params = {}
        for name, config in search_space.items():
            param_type = config.get("type")
            if param_type == "float":
                params[name] = trial.suggest_float(
                    name, config["low"], config["high"], log=config.get("log", False)
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    config["low"],
                    config["high"],
                    step=config.get("step", 1),
                    log=config.get("log", False),
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
        callbacks: Optional[list] = None,
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
            callbacks=callbacks,
        )
        return self.study

    def _log_trial_results(
        self,
        trial_number: int,
        duration: float,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
    ):
        """Log trial results in clean table format like the original tuner."""
        if not self.log_file:
            return

        # Extract fields
        combined = metrics.get("combined_loss", 0.0)
        perplexity = metrics.get("perplexity", 0.0)
        ce_loss = metrics.get("cross_entropy", 0.0)
        energy = metrics.get("energy", float("inf"))

        header_needed = trial_number == 0

        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "a") as f:

            # Header
            if header_needed:
                f.write(
                    f"{'Trial':<6} | {'Time(s)':<8} | {'combined':<10} | {'Perplexity':<12} | "
                    f"{'CE Loss':<10} | {'Energy':<10} | {'LR':<8} | {'Embed':<6} | {'MLP':<6} | {'Heads':<5} | "
                    f"{'Depth':<5} | {'Infer':<5} | {'Eta':<6}\n"
                )
                f.write("-" * 100 + "\n")

            # Row
            f.write(
                f"{trial_number:<6} | {duration:<8.1f} | {combined:<10.4f} | {perplexity:<12.4f} | "
                f"{ce_loss:<10.4f} | {energy:<10.4f} | {config.get('lr', 'N/A'):<8} | "
                f"{config.get('embed_dim', 'N/A'):<6} | {config.get('mlp_dim', 'N/A'):<6} | "
                f"{config.get('num_heads', 'N/A'):<5} | {config.get('depth', 'N/A'):<5} | "
                f"{config.get('infer_steps', 'N/A'):<5} | {config.get('eta_infer', 'N/A'):<6}\n"
            )

    def _objective(
        self,
        trial: optuna.Trial,
        search_space_fn: Callable[[optuna.Trial], Dict[str, Any]],
    ) -> float:
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
        model_key, train_key, eval_key = jax.random.split(rng_key, 3)

        # Build model components
        try:
            params, structure = self.trial_model(config, model_key)
        except Exception as e:
            print(f"Trial {trial.number} pruned due to creation failure: {e}")
            raise optuna.TrialPruned()

        # Execute Training and Evaluation
        try:
            if self.trainer_fn:
                result = self.trainer_fn(
                    params,
                    structure,
                    self.train_loader,
                    self.val_loader,
                    config,
                    train_key,
                )

                if isinstance(result, dict):
                    metrics = result
                    val_score = metrics.get(
                        self.metric, metrics.get("loss", float("inf"))
                    )
                else:
                    # Assume it returns the scalar score directly
                    val_score = float(result)
                    metrics = {self.metric: val_score}

            else:
                # Default training loop
                trained_params, _, _ = train_pcn(
                    params,
                    structure,
                    self.train_loader,
                    config,
                    train_key,
                    verbose=False,
                )

                metrics = evaluate_pcn(
                    trained_params,
                    structure,
                    self.val_loader,
                    config,
                    eval_key,
                )

                # Calculate combined score if needed
                if self.metric == "combined_loss":
                    energy = metrics.get("energy", 0.0)
                    perplexity = metrics.get("perplexity", 0.0)
                    metrics["combined_loss"] = 0.5 * energy + 0.5 * perplexity

                if self.metric in metrics:
                    val_score = metrics[self.metric]
                else:
                    # Fallback
                    items = list(metrics.values())
                    val_score = items[0] if items else float("inf")

        except Exception as e:
            print(f"Trial {trial.number} failed during training/eval: {e}")
            import traceback

            traceback.print_exc()
            return float("inf") if self.direction == "minimize" else float("-inf")

        duration = time.time() - start_time
        self._log_trial_results(trial.number, duration, metrics, config)

        return val_score
