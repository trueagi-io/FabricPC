# Aim Integration Plan for FabricPC

## Summary

Integrate [Aim](https://github.com/aimhubio/aim) experiment tracking into FabricPC to enable monitoring of training quality, batch-level debugging, and hyperparameter tuning visualization.

---

## Phase 1: Dependencies & Package Structure

### 1.1 Update `pyproject.toml`

Add `aim` to `[viz]` optional dependencies:
```toml
viz = [
    "plotly>=5.0.0",
    "kaleido>=0.2.1",
    "pandas>=2.0.0",
    "aim>=3.0.0",
]
```

### 1.2 Create Package Structure
```
fabricpc/utils/dashboarding/
  __init__.py              # Package exports, lazy imports
  _aim_available.py        # Lazy import helper, graceful degradation
  extractors.py            # Pure functions for metric extraction
  trackers.py              # AimExperimentTracker, TrackingConfig, StateHistoryCollector
  callbacks.py             # Callback factories for train_pcn integration
  inference_tracking.py    # Modified inference loop for state history collection
```

---

## Phase 2: Core Components

### 2.1 Lazy Import Helper (`_aim_available.py`)
- `is_aim_available()` - Check if Aim is installed
- `get_aim()` - Get aim module with helpful error if not installed
- `@require_aim` - Decorator for functions requiring Aim

### 2.2 Metric Extractors (`extractors.py`)
Pure functions for JAX-compatible metric extraction:
- `extract_node_energies(state)` - Per-node energy arrays
- `extract_total_energy(state, structure)` - Sum of non-source node energies
- `extract_latent_statistics(state, nodes)` - mean, std, min, max per node
- `extract_weight_statistics(params, nodes)` - Weight matrix stats per edge
- `extract_error_statistics(state)` - Prediction error stats
- `flatten_for_distribution(arr)` - Flatten for Aim Distribution

### 2.3 Core Tracker (`trackers.py`)

**TrackingConfig dataclass:**
```python
@dataclass
class TrackingConfig:
    # Batch-level
    track_batch_loss: bool = True
    track_batch_energy_per_node: bool = False

    # Epoch-level
    track_epoch_loss: bool = True
    track_epoch_accuracy: bool = True
    track_weight_distributions: bool = True
    track_latent_distributions: bool = False

    # Inference dynamics
    track_inference_dynamics: bool = False
    inference_nodes_to_track: List[str] = field(default_factory=list)

    # Frequency
    weight_distribution_every_n_epochs: int = 1
    latent_distribution_every_n_batches: int = 100

    # Naming
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
```

**AimExperimentTracker class:**
- `log_hyperparams(hparams)` - Log hyperparameters
- `log_graph_structure(structure)` - Log graph topology
- `track_batch_loss(loss, epoch, batch)` - Batch-level loss
- `track_batch_energy_per_node(state, structure, epoch, batch)` - Per-node energy
- `track_epoch_metrics(metrics, epoch, subset)` - Epoch metrics
- `track_weight_distributions(params, structure, epoch)` - Weight histograms
- `track_latent_distributions(state, epoch, batch)` - z_latent, z_mu, pre_activation
- `track_inference_dynamics(state_history, epoch, batch)` - Per-step convergence
- `close()` - Close Aim run

**StateHistoryCollector class:**
- Lightweight container for inference state histories
- `add_history(history, metadata)`
- `@property latest` - Most recent history

---

## Phase 3: Training Integration

### 3.1 Callback Factories (`callbacks.py`)

**`create_iter_callback(tracker)`**
Returns callback for `train_pcn` iter_callback parameter.

**`create_epoch_callback(tracker, structure, eval_fn, eval_loader)`**
Returns callback for `train_pcn` epoch_callback parameter.

**`create_tracking_callbacks(config, structure, eval_fn, eval_loader, hparams)`**
Convenience function returning `(tracker, iter_callback, epoch_callback)`.

### 3.2 Usage Pattern
```python
from fabricpc.utils.dashboarding import create_tracking_callbacks, TrackingConfig

tracker, iter_cb, epoch_cb = create_tracking_callbacks(
    config=TrackingConfig(experiment_name="mnist"),
    structure=structure,
    eval_fn=evaluate_pcn,
    eval_loader=test_loader,
    hparams=train_config,
)

trained_params, _, _ = train_pcn(
    params, structure, train_loader, train_config, rng_key,
    iter_callback=iter_cb,
    epoch_callback=epoch_cb,
)
tracker.close()
```

---

## Phase 4: Inference Dynamics Tracking

### 4.1 Modified Inference Loop (`inference_tracking.py`)

**`run_inference_with_history(params, init_state, clamps, structure, infer_steps, eta_infer, collect_every)`**
- Uses `jax.lax.scan` to collect intermediate states
- Returns `(final_state, state_history: List[GraphState])`
- `collect_every` parameter for subsampling

**`train_step_with_history(...)`**
- Modified train_step using `run_inference_with_history`
- Returns `(params, opt_state, loss, final_state, state_history)`

### 4.2 Tracked Inference Metrics
For each inference step:
- `inference_energy` - Mean energy per node
- `inference_grad_norm` - Latent gradient norm (convergence indicator)
- Allows visualization of energy minimization during inference

---

## Phase 5: New Aim Tracking Demo

**File:** `examples/mnist_aim_tracking.py`

Create a new dedicated example file (keep mnist_advanced.py unchanged).

### Structure:
1. Import dashboarding utilities
2. Create same 5-layer network as mnist_advanced
3. Create TrackingConfig with:
   - `track_batch_loss=True`
   - `track_weight_distributions=True`
   - `track_latent_distributions=True` (every 100 batches)
   - `track_inference_dynamics=True` (include full debugging)
4. Use custom training loop with `train_step_with_history` for inference dynamics
5. Track all metrics including per-inference-step evolution
6. Print instructions to launch Aim UI (`aim up`)

### Tracked Metrics for PC Debugging:
- **Weights:** Distribution histograms per layer per epoch
- **Latents:** z_latent distribution per node
- **Pre-activations:** Distribution before activation function
- **Activations:** z_mu distribution after activation
- **Node energies:** Per-node energy values
- **Inference dynamics:** Energy/gradient evolution over all 20 inference steps (full demo)

---

## Phase 6: Documentation

**File:** `docs/aim_integration.md`

Document:
1. Installation (`pip install fabricpc[viz]`)
2. Quick start example
3. TrackingConfig options
4. Best practices for PC debugging
5. Launching Aim UI (`aim up`)
6. Example visualizations

---

## Files to Create

| File | Description |
|------|-------------|
| `fabricpc/utils/dashboarding/__init__.py` | Package exports |
| `fabricpc/utils/dashboarding/_aim_available.py` | Lazy import helper |
| `fabricpc/utils/dashboarding/extractors.py` | Metric extraction utilities |
| `fabricpc/utils/dashboarding/trackers.py` | Core tracker classes |
| `fabricpc/utils/dashboarding/callbacks.py` | Callback factories |
| `fabricpc/utils/dashboarding/inference_tracking.py` | State history collection |
| `examples/mnist_aim_tracking.py` | New example with full Aim tracking demo |
| `docs/aim_integration.md` | User-facing usage documentation |

## Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `aim>=3.0.0` to `[viz]` deps |
| `fabricpc/utils/__init__.py` | Add lazy dashboarding export |

---

## Key Design Decisions

1. **Lazy imports**: Aim only imported when used (graceful degradation)
2. **JIT boundary**: All tracking happens outside JIT-compiled code
3. **Pure extractors**: Metric extraction as pure functions for testability
4. **Callback-based**: Integrates with existing `train_pcn` callback system
5. **Optional history**: Inference dynamics tracking opt-in due to memory cost
6. **Configurable frequency**: Distribution tracking intervals configurable

---

## Implementation Order

1. `pyproject.toml` - Add dependency
2. `_aim_available.py` - Lazy import foundation
3. `extractors.py` - Pure metric extraction
4. `trackers.py` - Core tracker implementation
5. `callbacks.py` - Training integration
6. `inference_tracking.py` - State history collection
7. `dashboarding/__init__.py` - Package exports
8. `utils/__init__.py` - Update exports
9. `examples/mnist_aim_tracking.py` - New demo with full inference dynamics tracking
10. `docs/aim_integration.md` - User-facing documentation