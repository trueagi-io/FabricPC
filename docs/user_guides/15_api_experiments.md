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

`fabricpc.tuning.bayesian_tuner`

Wraps Optuna for hyperparameter optimization.

```python
from fabricpc.tuning import BayesianTuner

tuner = BayesianTuner(
    train_loader=train_loader,
    val_loader=val_loader,
    trial_model=trial_model_factory,
    base_config={"num_epochs": 3},
    metric="accuracy",
    study_name="hparam_search",
)

tuner.optimize(n_trials=50)
best_params = tuner.study.best_params
```
