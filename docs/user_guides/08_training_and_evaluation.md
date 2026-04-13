# Training and Evaluation

Tutorial-style guide covering training loops, evaluation, callbacks, multi-GPU.

## High-Level API: train_pcn()

```python
from fabricpc.training import train_pcn

trained_params, energy_history, epoch_results = train_pcn(
    params=params,
    structure=structure,
    train_loader=train_loader,
    optimizer=optimizer,
    config={"num_epochs": 20},
    rng_key=train_key,
    verbose=True,
)
```

**Arguments**:
- `params` — Initial parameters from `initialize_params()`
- `structure` — `GraphStructure` from `graph()`
- `train_loader` — Iterable yielding `(x, y)` tuples or `{"x": ..., "y": ...}` dicts
- `optimizer` — Any Optax optimizer
- `config` — Dict with `num_epochs` (supports fractional epochs, e.g., `1.5`)
- `rng_key` — JAX random key (split internally per batch)
- `verbose` — Print progress per epoch

**Returns**: `(trained_params, energy_history, epoch_results)`
- `energy_history` — 2D list `[epoch][batch]` of per-batch average energy
- `epoch_results` — List of epoch_callback return values (or None)

## Evaluation: evaluate_pcn()

```python
from fabricpc.training import evaluate_pcn

metrics = evaluate_pcn(trained_params, structure, test_loader, config, eval_key)
print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Energy:   {metrics['energy']:.4f}")
```

Returns `{"energy": float, "accuracy": float}`.

**How accuracy works**: The output node's `z_mu` prediction is compared to the target using argmax. This works for classification with one-hot targets.

**Note on energy**: For feedforward DAGs with `FeedforwardStateInit`, evaluation energy will be near zero because `z_latent` starts equal to `z_mu`. Use accuracy (or other task-specific metrics) to assess model quality.

## Understanding Training Energy

Energy is the sum of per-node energies across the batch. It decreases during training as the network learns to predict its own states. Energy is **not** directly comparable to cross-entropy loss — it measures internal prediction consistency, not task performance.

## Callbacks

**Iteration callback** — Called after each batch:

```python
def my_iter_callback(epoch_idx, batch_idx, energy):
    if batch_idx % 100 == 0:
        print(f"  batch {batch_idx}: energy={energy:.4f}")
    return float(energy)

trained_params, _, _ = train_pcn(..., iter_callback=my_iter_callback)
```

**Epoch callback** — Called after each epoch:

```python
def my_epoch_callback(epoch_idx, params, structure, config, rng_key):
    metrics = evaluate_pcn(params, structure, test_loader, config, rng_key)
    print(f"  Epoch {epoch_idx}: acc={metrics['accuracy']:.4f}")
    return metrics

trained_params, _, epoch_results = train_pcn(..., epoch_callback=my_epoch_callback)
```

## Custom Training Loops

For more control, use `train_step()` directly:

```python
from fabricpc.training.train import train_step
import jax

opt_state = optimizer.init(params)
jit_step = jax.jit(lambda p, o, b, k: train_step(p, o, b, structure, optimizer, k))

for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        batch = {"x": jnp.array(x), "y": jnp.array(y)}
        rng_key, step_key = jax.random.split(rng_key)
        params, opt_state, energy, final_state = jit_step(params, opt_state, batch, step_key)
```

## Backpropagation Comparison Mode

FabricPC can train the same graph architecture with standard backpropagation for comparison:

```python
from fabricpc.training import train_backprop, evaluate_backprop

trained_params_bp, _, _ = train_backprop(
    params=params, structure=structure, train_loader=train_loader,
    optimizer=optimizer, config=config, rng_key=train_key,
)
metrics_bp = evaluate_backprop(trained_params_bp, structure, test_loader, config, eval_key)
```

This is useful for validating that the PC network architecture is capable, independently of PC-specific dynamics.

## Multi-GPU Training

Scale to multiple GPUs with data parallelism:

```python
from fabricpc.training.multi_gpu import (
    train_pcn_multi_gpu, evaluate_pcn_multi_gpu,
    replicate_params, shard_batch
)

trained_params, _, _ = train_pcn_multi_gpu(
    params=params, structure=structure, train_loader=train_loader,
    optimizer=optimizer, config=config, rng_key=train_key,
)
metrics = evaluate_pcn_multi_gpu(trained_params, structure, test_loader, config, eval_key)
```

The total batch size must be divisible by the number of devices. Works transparently with a single GPU.

## Statistical A/B Experiments

For rigorous comparisons across multiple trials, use the experiment framework:

```python
from fabricpc.experiments import ExperimentArm, ABExperiment

arm_a = ExperimentArm(name="PC", model_factory=create_model, train_fn=train_pcn, ...)
arm_b = ExperimentArm(name="Backprop", model_factory=create_model, train_fn=train_backprop, ...)

experiment = ABExperiment(arm_a=arm_a, arm_b=arm_b, metric="accuracy",
                          data_loader_factory=loader_fn, n_trials=5)
results = experiment.run()
results.print_summary()  # Paired t-test, Cohen's d, effect sizes
```

See the [Experiment Framework API](15_api_experiments.md) for full details.
