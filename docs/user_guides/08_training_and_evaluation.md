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

Next-token models use `evaluate_autoregressive` instead, which reports perplexity (see [Autoregressive Language Modeling](#autoregressive-language-modeling)).

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

## Autoregressive Language Modeling

Next-token models train with `train_autoregressive`. The loader contract differs from `train_pcn`: both `x` and `y` are `(batch, seq_len)` int32 token ids, with `y` being `x` shifted one position left (see [Data Loaders](14_api_data.md)); one-hot encoding of the target happens inside the training step.

### train_autoregressive

```python
from fabricpc.training import train_autoregressive

trained_params, energy_history, epoch_results = train_autoregressive(
    params=params,
    structure=structure,
    train_loader=train_loader,
    optimizer=optimizer,
    config={"num_epochs": 5},
    rng_key=train_key,
)
```

Same core signature as `train_pcn` (`train_pcn` additionally accepts `use_tqdm` and `pmap_single_device`). Returns `(trained_params, energy_history, epoch_results)`.

**Config keys:**
- `num_epochs` — supports fractional values (e.g. `1.5` trains half of the second epoch)
- `use_causal_mask` (default `True`) — clamps a lower-triangular attention mask on graphs whose task map declares a `"causal_mask"` node (the v1 `TransformerBlock` pattern). The decomposed transformer masks internally via `is_causal`, so its task map has no such node and nothing is clamped.

**Epoch callback** — called as `epoch_callback(epoch_idx, params, structure, config, rng_key, energy=avg_energy, ce_loss=avg_ce_loss)`, where `energy` is the epoch's average training energy and `ce_loss` its average per-token cross-entropy:

```python
def my_epoch_callback(epoch_idx, params, structure, config, rng_key, energy=None, ce_loss=None):
    print(f"Epoch {epoch_idx}: energy={energy:.4f} ce={ce_loss:.4f}")

trained_params, _, _ = train_autoregressive(..., epoch_callback=my_epoch_callback)
```

### evaluate_autoregressive

```python
from fabricpc.training import evaluate_autoregressive

metrics = evaluate_autoregressive(trained_params, structure, val_loader, config, eval_key)
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

Returns `{"loss", "perplexity", "accuracy", "num_batches"}` — average per-token cross-entropy, perplexity, next-token accuracy, and the number of evaluated batches. `use_causal_mask` follows the same default as training.

**Perplexity** is the effective number of tokens the model chooses among at each step: a model with perplexity 20 is as uncertain as a uniform choice over 20 tokens. It is computed from the mean per-token cross-entropy `CE` over the evaluation set as `perplexity = exp(CE)`.

### generate_autoregressive

```python
from fabricpc.training import generate_autoregressive

x_indices, _ = next(iter(loader))
prompt = x_indices[0]        # one sequence: 1-D int32 token ids, shape (seq_len,)

tokens = generate_autoregressive(
    trained_params,
    structure,
    prompt=prompt,
    max_new_tokens=200,
    rng_key=gen_key,
    temperature=0.8,
    top_k=40,
)
print(loader.decode(tokens))  # tokens: shape (seq_len + 200,)
```

The prompt is 1-D `(prompt_len,)` for a single sequence or 2-D `(batch, prompt_len)` for a batch; the returned tokens match (`(prompt_len + max_new_tokens,)` or `(batch, prompt_len + max_new_tokens)`). `loader.decode` accepts one 1-D sequence, so decode a batched result row by row.

- `temperature` divides the logits before sampling: below 1.0 concentrates probability on the most likely tokens, above 1.0 flattens the distribution.
- `top_k` keeps only the k most probable tokens.
- `top_p` keeps the smallest token set whose cumulative probability reaches p.

Generation slides a context window of the model's `seq_len`: once the sequence exceeds it, the oldest tokens drop out of the context.

The backprop counterparts `train_backprop_autoregressive` and `evaluate_backprop_autoregressive` share these interfaces. End-to-end example: `examples/transformer_v2_demo.py` (`--mode pc|backprop`, `--tokenizer char|bpe`); hyperparameter search: [Experiment Framework API](15_api_experiments.md).

## Multi-GPU Training

`train_pcn` and `evaluate_pcn` automatically detect available devices and use
pmap data parallelism when multiple GPUs are present. No separate import is needed:

```python
from fabricpc.training import train_pcn, evaluate_pcn

# Same API for 1 or N devices — multi-GPU is automatic
trained_params, energies, epoch_results = train_pcn(
    params=params, structure=structure, train_loader=train_loader,
    optimizer=optimizer, config=config, rng_key=train_key,
)
metrics = evaluate_pcn(trained_params, structure, test_loader, config, eval_key)
```

The total batch size must be divisible by the number of devices (batches that
aren't divisible are skipped with a warning). To force the pmap code path on a
single device (useful for testing), pass `pmap_single_device=True`.

## Statistical A/B Experiments

For rigorous comparisons across multiple trials, use the experiment framework:

```python
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.training import train_pcn, evaluate_pcn, train_backprop, evaluate_backprop

arm_a = ExperimentArm(name="PC", model_factory=create_model, train_fn=train_pcn,
                      eval_fn=evaluate_pcn, optimizer=optimizer, train_config=config)
arm_b = ExperimentArm(name="Backprop", model_factory=create_model, train_fn=train_backprop,
                      eval_fn=evaluate_backprop, optimizer=optimizer, train_config=config)

experiment = ABExperiment(arm_a=arm_a, arm_b=arm_b, metric="accuracy",
                          data_loader_factory=loader_fn, n_trials=5)
results = experiment.run()
results.print_summary()  # Paired t-test, Cohen's d, effect sizes
```

For more than two arms, `PlannedMultiContrastExperiment` runs N arms with constructor-declared planned contrasts; `ABExperiment` is its 2-arm wrapper. See the [Experiment Framework API](15_api_experiments.md) for full details.
