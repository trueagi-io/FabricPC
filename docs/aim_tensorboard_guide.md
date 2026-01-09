# Aim How-to Guide for TensorBoards with FabricPC

FabricPC integrates with [Aim](https://aimstack.io/) for comprehensive experiment tracking and visualization. This enables detailed monitoring of training quality, batch-level debugging, and hyperparameter tuning for predictive coding networks.

## Installation

```bash
pip install fabricpc[viz]
```

Or install Aim directly:

```bash
pip install aim
```

## Quick Start

```python
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    create_tracking_callbacks,
)
from fabricpc.training import train_pcn, evaluate_pcn

# Create tracking configuration
tracking_config = TrackingConfig(
    experiment_name="my_experiment",
    track_weight_distributions=True,
    track_batch_loss=True,
)

# Create callbacks for train_pcn
tracker, iter_cb, epoch_cb = create_tracking_callbacks(
    config=tracking_config,
    structure=structure,
    eval_fn=evaluate_pcn,
    eval_loader=test_loader,
    hparams=train_config,
)

# Train with tracking
trained_params, _, _ = train_pcn(
    params, structure, train_loader, train_config, rng_key,
    iter_callback=iter_cb,
    epoch_callback=epoch_cb,
)

# Close the tracker
tracker.close()
```

Then launch the Aim UI:

```bash
aim up
```

## TrackingConfig Options

```python
@dataclass
class TrackingConfig:
    # Batch-level tracking
    track_batch_loss: bool = True
    track_batch_energy_per_node: bool = False

    # Epoch-level tracking
    track_epoch_loss: bool = True
    track_epoch_accuracy: bool = True
    track_weight_distributions: bool = True
    track_latent_distributions: bool = False
    track_preactivation_distributions: bool = False
    track_activation_distributions: bool = False
    track_error_statistics: bool = False

    # Inference dynamics tracking
    track_inference_dynamics: bool = False
    inference_nodes_to_track: List[str] = field(default_factory=list)

    # Frequency controls
    weight_distribution_every_n_epochs: int = 1
    latent_distribution_every_n_batches: int = 100

    # Naming
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
```

## Tracking Predictive Coding Metrics

### Weight Distributions

Track how weights evolve during training:

```python
config = TrackingConfig(
    track_weight_distributions=True,
    weight_distribution_every_n_epochs=5,  # Every 5 epochs
)
```

### Latent State Distributions

Track z_latent, z_mu (activations), and pre_activation distributions:

```python
config = TrackingConfig(
    track_latent_distributions=True,
    track_activation_distributions=True,
    track_preactivation_distributions=True,
    latent_distribution_every_n_batches=100,
)
```

### Per-Node Energy

Track energy contribution of each node:

```python
config = TrackingConfig(
    track_batch_energy_per_node=True,
)
```

### Inference Dynamics

Track how energy and gradients evolve during the inference phase (useful for debugging convergence):

```python
config = TrackingConfig(
    track_inference_dynamics=True,
    inference_nodes_to_track=["h1", "h2", "h3"],  # Specific nodes
)
```

## Advanced Usage: Custom Training Loop

For detailed tracking including inference dynamics, use a custom training loop:

```python
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    train_step_with_history,
    summarize_inference_convergence,
)

tracker = AimExperimentTracker(config=TrackingConfig(
    experiment_name="detailed_tracking",
    track_inference_dynamics=True,
))

# JIT compile with history collection
jit_train_step = jax.jit(
    lambda p, o, b, k: train_step_with_history(
        p, o, b, structure, optimizer, k,
        infer_steps=20, eta_infer=0.05,
        collect_every=5,  # Collect every 5th inference step
    )
)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        params, opt_state, loss, final_state, inference_history = jit_train_step(
            params, opt_state, batch, rng_key
        )

        # Track batch metrics
        tracker.track_batch_loss(loss / batch_size, epoch, batch_idx)
        tracker.track_batch_energy_per_node(final_state, structure, epoch, batch_idx)

        # Analyze inference convergence
        convergence = summarize_inference_convergence(inference_history)
        print(f"h1 converged: {convergence['h1']['converged']}")

tracker.close()
```

## Metric Extractors

Use extractors to get specific metrics from GraphState and GraphParams:

```python
from fabricpc.utils.dashboarding import (
    extract_node_energies,
    extract_weight_statistics,
    extract_latent_statistics,
    extract_error_statistics,
)

# After training step
energies = extract_node_energies(final_state)
# {'pixels': array([...]), 'h1': array([...]), ...}

weight_stats = extract_weight_statistics(params)
# {'h1': {'pixels->h1:in': {'mean': 0.01, 'std': 0.05, ...}}}

latent_stats = extract_latent_statistics(final_state)
# {'h1': {'mean': 0.5, 'std': 0.2, 'min': 0.0, 'max': 1.0}}
```

## Graceful Degradation

The dashboarding module works even when Aim is not installed:

```python
from fabricpc.utils.dashboarding import is_aim_available

if is_aim_available():
    tracker = AimExperimentTracker(config)
    # Full tracking
else:
    tracker = None
    # Training continues without tracking
```

## Best Practices for PC Debugging

1. **Track weight distributions** to detect exploding/vanishing gradients in the Hebbian learning updates.

2. **Track per-node energy** to identify which layers are contributing most to the total energy.

3. **Track inference dynamics** to verify that the inference loop converges (energy should decrease, gradient norms should approach zero).

4. **Monitor latent distributions** to ensure activations are in the expected range (e.g., [0, 1] for sigmoid).

5. **Use batch-level tracking** for debugging and epoch-level for production runs.

## Example Output

See `examples/mnist_aim_tracking.py` for a complete example that tracks:

- Batch-level loss and per-node energy
- Epoch-level accuracy and weight distributions
- Latent and pre-activation distributions
- Inference dynamics (energy convergence per step)

## Launching the Dashboard

After training, view your experiments:

```bash
aim up
```

This opens a web interface at `http://localhost:43800` where you can:

- Compare runs with different hyperparameters
- Visualize weight distribution evolution
- Explore per-node energy contributions
- Analyze inference convergence patterns