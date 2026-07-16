# Experiment Framework API

## A/B Experiments

`fabricpc.experiments.ab_experiment`

### ExperimentArm

Defines one condition in an A/B experiment.

```python
from fabricpc.experiments import ExperimentArm

arm = ExperimentArm(
    name="muPC",
    model_factory=create_model,    # (rng_key) -> (params, structure)
    train_fn=train_pcn,
    eval_fn=evaluate_pcn,
    optimizer=optax.adamw(1e-3),
    train_config={"num_epochs": 5},
)
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Display name for this condition |
| `model_factory` | `Callable` | `(rng_key) -> (params, structure)` |
| `train_fn` | `Callable` | Training function (e.g., `train_pcn`) |
| `eval_fn` | `Callable` | Evaluation function (e.g., `evaluate_pcn`) |
| `optimizer` | `optax.GradientTransformation` | Optimizer |
| `train_config` | `dict` | Training configuration |

### ABExperiment

Runs two arms across multiple trials with statistical analysis.

```python
from fabricpc.experiments import ABExperiment

experiment = ABExperiment(
    arm_a=arm_mupc,
    arm_b=arm_standard,
    metric="accuracy",
    data_loader_factory=lambda seed: (train_loader, test_loader),
    n_trials=5,
    verbose=False,
)

results = experiment.run()
results.print_summary()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `arm_a` | `ExperimentArm` | First condition |
| `arm_b` | `ExperimentArm` | Second condition |
| `metric` | `str` | Metric key from eval_fn return dict (e.g., `"accuracy"`) |
| `data_loader_factory` | `Callable` | `(seed) -> (train_loader, test_loader)` |
| `n_trials` | `int` | Number of independent trials |
| `verbose` | `bool` | Print per-trial progress |

**Output** from `print_summary()`:
- Descriptive statistics (mean, std, min, max)
- Paired t-test (t-statistic, p-value)
- Cohen's d effect size
- Estimated N for significance

---

## Statistics

`fabricpc.experiments.statistics`

```python
from fabricpc.experiments.statistics import (
    descriptive_stats,
    paired_ttest,
    cohens_d,
    estimate_required_n,
)

stats_a = descriptive_stats(arm_a_metrics)   # {"mean", "std", "min", "max", "median"}
t_stat, p_val = paired_ttest(arm_a_metrics, arm_b_metrics)
d = cohens_d(arm_a_metrics, arm_b_metrics)
n = estimate_required_n(arm_a_metrics, arm_b_metrics)
```

---

## Bayesian Tuning

`fabricpc.tuning.BayesianTuner`

Two-phase Optuna search for language-model hyperparameters, both phases minimizing validation perplexity. Phase 1 searches architecture and training parameters together, with Hyperband pruning that allocates training epochs as the trial resource: unpromising trials are stopped after few epochs while strong ones train longer. Phase 2 fixes the Phase 1 winning architecture and refines the continuous training parameters (`lr`, `eta_infer`, `infer_steps`) with a multivariate TPE sampler that models correlations between them. Training energy serves only as a divergence guard: a trial is pruned when its energy becomes non-finite or rises above its best epoch by more than `divergence_rel_tol`.

```python
from fabricpc.tuning import BayesianTuner

def phase1_search_space(trial):
    return {
        "embed_dim": trial.suggest_categorical("embed_dim", [64, 128]),
        "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "depth": trial.suggest_int("depth", 1, 4),
        "lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 0.01, 0.15),
    }

def phase2_search_space(trial, best_params):
    lr = best_params["lr"]
    return {
        "lr": trial.suggest_float("lr", lr * 0.5, lr * 2.0, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 0.01, 0.2),
    }

tuner = BayesianTuner(
    train_loader=train_loader,
    val_loader=val_loader,
    trial_model=trial_model,           # (config, rng_key) -> (params, structure)
    base_config={"num_epochs": 5, "seq_len": 128, "vocab_size": vocab_size},
    study_name="transformer_v2_tuning",
)

results = tuner.tune(
    phase1_search_space=phase1_search_space,
    phase2_search_space=phase2_search_space,
    n_trials_phase1=20,
    n_trials_phase2=15,
)
print(results["phase2_best_ppl"], results["final_params"])
```

Full runnable version: `examples/transformer_tuning.py`.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_loader` | iterable | required | Training batches (int32 token-id pairs, see [Data Loaders](14_api_data.md)) |
| `val_loader` | iterable | required | Validation batches for the perplexity objective |
| `trial_model` | `Callable` | required | `(config, rng_key) -> (params, structure)`; may instead return a 4-tuple `(params, structure, train_loader, val_loader)` for trials that rebuild loaders per architecture (e.g. `seq_len` or `batch_size` in the search space) |
| `base_config` | `dict` | required | Fixed config merged under every trial's sampled parameters |
| `study_name` | `str` | `"fabricpc_tuning"` | Optuna study name |
| `storage` | Optuna storage | `None` | Persistent trial storage; `None` keeps trials in memory |
| `log_file` | `str` | `"tuning_results.txt"` | Per-trial results log |
| `divergence_rel_tol` | `float` | `0.5` | Relative energy rise over the trial's best epoch that triggers pruning |
| `verbose` | `bool` | `False` | Print per-epoch trial progress |

**Methods:**

- `tune_phase1(n_trials, search_space)` — `search_space: (trial) -> dict` of sampled parameters. Returns the Optuna `Study`.
- `tune_phase2(n_trials, best_params, search_space)` — `search_space: (trial, best_params) -> dict`, sampling around the Phase 1 winners. Returns the Optuna `Study`.
- `tune(phase1_search_space, phase2_search_space, n_trials_phase1=30, n_trials_phase2=20, save_best_to="tuning/best_hyperparameters.txt")` — runs both phases. Returns `{"phase1_best_ppl", "phase1_best_params", "phase2_best_ppl", "phase2_best_params", "final_params"}`; when Phase 2 completes no trial, the dict carries the Phase 1 keys only.
