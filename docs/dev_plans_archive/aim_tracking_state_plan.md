# Plan: Batch-interval weight tracking + inference-step-interval state tracking

## Summary

Rework tracking frequency controls and consolidate state tracking:

1. **Weight tracking**: boolean enable + batch-interval for distributions
2. **State tracking**: always collect summary stats (mean, std, norm) for `z_mu`, `z_latent`, `energy`; optionally track full distributions; gated by batch-interval + inference-step-interval

## New TrackingConfig fields

Remove:
- `track_latent_distributions`
- `track_preactivation_distributions`
- `track_activation_distributions`
- `weight_distribution_every_n_epochs`
- `latent_distribution_every_n_batches`

Add:
```python
track_state_distributions: bool = False           # distribution histograms for z_mu, z_latent, energy
weight_tracking_every_n_batches: int = 50         # replaces weight_distribution_every_n_epochs
state_tracking_every_n_batches: int = 200         # replaces latent_distribution_every_n_batches
state_tracking_every_n_infer_steps: int = 5       # NEW — within sampled batches
```

Keep unchanged: `track_weight_distributions: bool = True`

## Files to modify

| File | Change |
|------|--------|
| `fabricpc/utils/dashboarding/trackers.py` | TrackingConfig fields + rename/rewrite `track_latent_distributions` → `track_state` |
| `fabricpc/utils/dashboarding/callbacks.py` | Update callers |
| `examples/transformer_demo.py` | Update config + training loop |
| `examples/mnist_aim_tracking.py` | Update config + callsites |

---

## Step 1 — TrackingConfig in `trackers.py` (lines 20-65)

Remove fields:
```python
track_latent_distributions: bool = False
track_preactivation_distributions: bool = False
track_activation_distributions: bool = False
weight_distribution_every_n_epochs: int = 1
latent_distribution_every_n_batches: int = 100
```

Add fields:
```python
track_state_distributions: bool = False
weight_tracking_every_n_batches: int = 50
state_tracking_every_n_batches: int = 200
state_tracking_every_n_infer_steps: int = 5
```

Update docstring accordingly.

---

## Step 2 — `track_weight_distributions()` in `trackers.py` (lines 279-322)

Add `batch: int` parameter. Change guard from epoch-modulo to batch-modulo. Add `step=self._global_step` to all `self._run.track()` calls.

```python
def track_weight_distributions(
    self,
    params: GraphParams,
    structure: GraphStructure,
    epoch: int,
    batch: int,
    nodes: Optional[List[str]] = None,
) -> None:
```

Guard:
```python
if not self.config.track_weight_distributions:
    return
if batch % self.config.weight_tracking_every_n_batches != 0:
    return
```

Add `step=self._global_step` to each `self._run.track()` call for correct ordering in Aim.

---

## Step 3 — Replace `track_latent_distributions()` with `track_state()` in `trackers.py` (lines 324-384)

Delete the old method. Create new `track_state()` method that:

1. Always logs summary stats (mean, std, L2 norm) for `z_mu`, `z_latent`, `energy`
2. Optionally logs distribution histograms when `track_state_distributions=True`
3. Gates on `infer_step % state_tracking_every_n_infer_steps` (batch-level gating done by caller)

```python
def track_state(
    self,
    state: GraphState,
    epoch: int,
    batch: int,
    infer_step: int,
    nodes: Optional[List[str]] = None,
) -> None:
```

Guard:
```python
if infer_step % self.config.state_tracking_every_n_infer_steps != 0:
    return
```

For each node, for each of `("z_latent", node_state.z_latent)`, `("z_mu", node_state.z_mu)`, `("energy", node_state.energy)`:

**Always track summary stats:**
```python
data = np.asarray(var_data)
ctx = {"node": node_name, "infer_step": infer_step}
self._run.track(float(np.mean(data)), name=f"{var_name}_mean",
                step=self._global_step, epoch=epoch, context=ctx)
self._run.track(float(np.std(data)), name=f"{var_name}_std",
                step=self._global_step, epoch=epoch, context=ctx)
self._run.track(float(np.linalg.norm(data)), name=f"{var_name}_norm",
                step=self._global_step, epoch=epoch, context=ctx)
```

**Conditionally track distributions:**
```python
if self.config.track_state_distributions:
    dist = aim.Distribution(flatten_for_distribution(var_data))
    self._run.track(dist, name=var_name,
                    step=self._global_step, epoch=epoch, context=ctx)
```

---

## Step 4 — `callbacks.py` updates

### `create_epoch_callback` (line 66)

Pass `batch=0` to `track_weight_distributions`:
```python
tracker.track_weight_distributions(params, structure, epoch=epoch_idx, batch=0)
```

### `create_detailed_iter_callback` (line 180)

Replace `track_latent_distributions` call with `track_state`. Add batch-level gating here (since the method only checks infer_step). Pass `infer_step=0`:
```python
if batch_idx % tracker.config.state_tracking_every_n_batches == 0:
    tracker.track_state(
        final_state, epoch=epoch_idx, batch=batch_idx, infer_step=0
    )
```

---

## Step 5 — `transformer_demo.py` updates

### 5a: TrackingConfig (lines 529-539)

```python
tracking_config = TrackingConfig(
    experiment_name="transformer_pc_shakespeare",
    run_name=f"{'PC' if use_pcn else 'BP'}_{NUM_BLOCKS}blk_{EMBED_DIM}d",
    track_batch_energy=True,
    track_batch_energy_per_node=False,
    track_weight_distributions=True,
    track_state_distributions=True,
    weight_tracking_every_n_batches=50,
    state_tracking_every_n_batches=200,
    state_tracking_every_n_infer_steps=5,
)
```

### 5b: Per-batch — weight tracking

Move weight tracking from per-epoch (line ~741) to per-batch block (batch gating inside method):
```python
if tracker is not None:
    tracker.track_batch_energy(loss_val, epoch=epoch, batch=batch_idx)
    tracker.track_weight_distributions(
        params, structure, epoch=epoch, batch=batch_idx, nodes=TRACKED_NODES
    )
```

Remove the per-epoch `track_weight_distributions` call at line ~741.

### 5c: Per-batch — state tracking (per inference step)

Replace the current `track_latent_distributions` call (lines 730-735). Two-level gating: batch check at caller, infer-step check inside method.

```python
# State tracking — per inference step (PC only)
should_track_state = (
    final_state is not None
    and tracker is not None
    and batch_idx % tracker.config.state_tracking_every_n_batches == 0
)
if should_track_state:
    # Reconstruct clamps for this batch
    track_clamps = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map:
            track_clamps[structure.task_map[task_name]] = task_value
    if use_causal_mask:
        seq_len = batch["x"].shape[1]
        cm = create_causal_mask(seq_len)[None, None, :, :]
        cm = jnp.broadcast_to(cm, (batch["x"].shape[0], 1, seq_len, seq_len))
        track_clamps[structure.task_map["causal_mask"]] = cm

    track_init_state = initialize_graph_state(
        structure, batch["x"].shape[0], batch_keys[batch_idx],
        clamps=track_clamps, params=params,
    )
    _, state_history = run_inference_with_full_history(
        params, track_init_state, track_clamps, structure
    )
    for infer_step_idx, step_state in enumerate(state_history):
        tracker.track_state(
            step_state, epoch=epoch, batch=batch_idx,
            infer_step=infer_step_idx, nodes=TRACKED_NODES,
        )
```

### 5d: Imports

Add at top of file:

```python
from fabricpc.utils.dashboarding.inference_tracking import run_inference_with_full_history
from fabricpc.graph_initialization import initialize_graph_state  # if not already imported
```

`create_causal_mask` is already imported via `train_autoregressive`.

### 5e: Summary print (line ~804)

Update to reference the new config field names:
```python
print(f"  State tracking: every {tracker.config.state_tracking_every_n_batches} batches, "
      f"every {tracker.config.state_tracking_every_n_infer_steps} infer steps")
```

---

## Step 6 — `mnist_aim_tracking.py` updates

### TrackingConfig (lines 170-189)

Replace old fields:
```python
tracking_config = TrackingConfig(
    ...
    track_weight_distributions=True,
    track_state_distributions=True,
    # Remove: track_preactivation_distributions, track_activation_distributions
    weight_tracking_every_n_batches=50,
    state_tracking_every_n_batches=50,
    state_tracking_every_n_infer_steps=5,
)
```

### Latent tracking call (line 283-285)

Replace `track_latent_distributions` with `track_state`, add batch-level gating and `infer_step=0`:
```python
if batch_idx % tracker.config.state_tracking_every_n_batches == 0:
    tracker.track_state(
        final_state, epoch=epoch, batch=batch_idx, infer_step=0
    )
```

### Weight tracking call (line 337)

Add `batch=0`:
```python
tracker.track_weight_distributions(params, structure, epoch=epoch, batch=0)
```


# All 6 steps complete, 138 tests pass.
Here's a summary of what was changed:                                                                                    
                                                                                                                                                                 
  fabricpc/utils/dashboarding/trackers.py                                                                                                                        
  - TrackingConfig: Removed track_latent_distributions, track_preactivation_distributions, track_activation_distributions, weight_distribution_every_n_epochs, latent_distribution_every_n_batches. Added track_state_distributions, weight_tracking_every_n_batches, state_tracking_every_n_batches, state_tracking_every_n_infer_steps.                                                                                                                                                          
  - track_weight_distributions(): Added batch parameter, changed guard from epoch-modulo to batch-modulo, added step=self._global_step to all track calls.       
  - track_latent_distributions() → track_state(): New method always logs summary stats (mean, std, L2 norm) for z_latent, z_mu, energy per node. Optionally logs distribution histograms when track_state_distributions=True. Gates on infer_step % state_tracking_every_n_infer_steps.
                                                                                                                                                                 
  fabricpc/utils/dashboarding/callbacks.py                                                                                                                       
  - create_epoch_callback: passes batch=0 to track_weight_distributions.                                                                                         
  - create_detailed_iter_callback: replaced track_latent_distributions with batch-gated track_state(..., infer_step=0).                                          
                                                                                                                                                                 
  examples/transformer_demo.py                                                                                                                                   
  - New imports: create_causal_mask, initialize_graph_state, run_inference_with_full_history.                                                                    
  - Config uses new field names.                                                                                                                                 
  - Weight tracking moved from per-epoch to per-batch (batch gating inside method).                                                                              
  - State tracking reconstructs clamps, runs run_inference_with_full_history, iterates all inference steps with track_state.                                     
                                                                                                                                                                 
  examples/mnist_aim_tracking.py                                                                                                                                 
  - Config uses new field names. Latent tracking call replaced with batch-gated track_state. Weight tracking call gets batch=0.

# Revisions to initial plan (completed 3/6/26)

trackers.py — TrackingConfig simplified:                                                                                                                       
  - track_batch_energy + track_epoch_energy → track_energy                                                                                                       
  - track_epoch_accuracy → track_accuracy                                                                                                                        
  - track_batch_energy_per_node removed — nodes_to_track (non-empty) enables per-node tracking                                                                   
  - inference_nodes_to_track → nodes_to_track                                                                                                                    
  - weight_tracking_every_n_batches + state_tracking_every_n_batches → tracking_every_n_batches                                                                  
  - Added tracking_every_n_epochs                                                                                                                                
                                                                                                                                                                 
  trackers.py — method guard updates:                                                                                                                            
  - track_batch_energy(): checks track_energy                                                                                                                    
  - track_batch_energy_per_node(): checks nodes_to_track and iterates only those nodes                                                                           
  - track_epoch_metrics(): checks track_energy / track_accuracy                                                                                                  
  - track_weight_distributions(): uses tracking_every_n_batches                                                                                                  
  - track_inference_dynamics(): uses nodes_to_track                                                                                                              
                                                                                                                                                                 
  callbacks.py: Updated tracking_every_n_batches reference.                                                                                                      
                                                                                                                                                                 
  transformer_demo.py: Config uses new fields, batch-interval reference updated, summary print updated.                                                          
                                                                                                                                                                 
  mnist_aim_tracking.py: Config uses new fields. All hardcoded intervals replaced:                                                                               
  - % 100 → % tracker.config.tracking_every_n_batches                                                                                                            
  - * 5 → * COLLECT_EVERY (new top-level constant)                                                                                                               
  - collect_every=5 → collect_every=COLLECT_EVERY                                                                                                                
  - tracking_config.inference_nodes_to_track → tracker.config.nodes_to_track                                                                                     

Removed track_inference_dynamics from TrackingConfig. The track_inference_dynamics() method now gates on nodes_to_track: if the list is non-empty (or an explicit nodes arg is passed), it tracks; otherwise it's a no-op.