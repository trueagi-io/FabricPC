# Consolidate train.py and multi_gpu.py into a Unified Trainer (completed 2026-04-14)

## Summary

Merge all multi-GPU features from `multi_gpu.py` into `train.py`, producing a single trainer that auto-detects devices (JIT for 1, pmap for N>1). Add TQDM progress bar. Drop `multi_gpu.py` after migration.

---

## Design Decisions (confirmed)

| Decision | Choice |
|----------|--------|
| Return value | 3-tuple `(params, iter_results, epoch_results)` with callbacks |
| Energy normalization | Inside `get_graph_param_gradient` (per-sample) |
| Progress bar | TQDM counting total minibatches, `tqdm=True` by default |
| Device path | Auto-detect; `pmap_single_device=False` kwarg to force pmap on 1 device |
| Duplication strategy | One shared loop, two step functions (JIT vs pmap) |

---

## New `train_pcn` Signature

```python
def train_pcn(
    params, structure, train_loader, optimizer, config, rng_key,
    verbose=True,
    tqdm=True,           # NEW — progress bar
    epoch_callback=None,
    iter_callback=None,
    pmap_single_device=False,  # NEW — force pmap on 1 device
) -> Tuple[GraphParams, List[Any], List[Any]]:
```

---

## Implementation Steps

### Step 1: Modify `train.py` — add pmap utilities

Move these functions from `multi_gpu.py` into `train.py` (above existing functions):
- `replicate_params`
- `replicate_opt_state`
- `shard_batch`
- `unshard_energies`

Add shared helper:
```python
def _convert_batch(batch_data) -> Dict[str, jnp.ndarray]:
```
(Extracted from the 5 duplicate copies across both files.)

Add imports: `from tqdm.auto import tqdm as _tqdm_cls`, `import time`, `import warnings`

### Step 2: Normalize energy inside `get_graph_param_gradient`

In `train.py:get_graph_param_gradient`, add `energy = energy / batch_size` before the return. This makes energy always per-sample.

**Ripple effects to fix simultaneously:**
- `train.py:224` — remove `/ next(iter(batch.values())).shape[0]` in the default iter_callback else-branch
- `train_step_pmap` (being moved) — remove `energy = energy / batch_size` (line 139 of old multi_gpu.py)

### Step 3: Move `train_step_pmap` + `create_pmap_train_step` into `train.py`

Copy from `multi_gpu.py`, remove the energy normalization line (handled in step 2). Place after `train_step`.

### Step 4: Rewrite `train_pcn` as unified loop

**Logic flow:**
```
1. n_devices = jax.device_count()
2. use_pmap = (n_devices > 1) or pmap_single_device
3. opt_state = optimizer.init(params)
4. if use_pmap:
     replicate params + opt_state
     step_fn = create_pmap_train_step(structure, optimizer)
   else:
     step_fn = jax.jit(lambda p, o, b, k: train_step(...))
5. Compute total_batches for TQDM
6. Create tqdm bar (total=total_batches, disable=not tqdm_enabled)
7. For each epoch:
     a. Split RNG keys (device-aware)
     b. For each batch:
        - _convert_batch(batch_data)
        - if use_pmap: shard_batch, handle ValueError (skip + warn)
        - call step_fn
        - if use_pmap: unshard_energies
        - iter_callback or append raw energy
        - progress_bar.update(1), set_description
     c. epoch_callback
     d. if verbose and not tqdm: print epoch summary
8. Close progress bar
9. if use_pmap: extract params from device 0
10. return (params, iter_results, epoch_results)
```

When `tqdm=True`, suppress verbose per-epoch print (TQDM provides that info). When `tqdm=False, verbose=True`, print epoch summary as current train.py does.

Import note: `from tqdm.auto import tqdm as _tqdm_cls` to avoid shadowing the `tqdm` parameter name.

### Step 5: Rewrite `evaluate_pcn` as unified eval

Same auto-detect pattern. Single-device uses existing `eval_step` via JIT. Multi-device uses pmap'd inference (ported from `evaluate_pcn_multi_gpu`). Add `pmap_single_device=False` parameter.

### Step 6: Move `evaluate_transformer_multi_gpu` → `evaluate_transformer`

Move into `train.py`, rename. It always uses pmap (even on 1 device) since it needs device-parallel inference. Keep logic unchanged.

### Step 7: Update `__init__.py`

```python
from fabricpc.training.train import (
    train_step, train_pcn, evaluate_pcn, evaluate_transformer,
    replicate_params, shard_batch, get_graph_param_gradient,
)

# Backward-compat aliases
train_pcn_multi_gpu = train_pcn
evaluate_pcn_multi_gpu = evaluate_pcn
evaluate_transformer_multi_gpu = evaluate_transformer
```

Keep old names in `__all__` temporarily.

### Step 8: Convert `multi_gpu.py` to deprecation shim

Replace contents with thin wrappers that emit `DeprecationWarning` then delegate to `train.py`.

### Step 9: Fix energy normalization in callbacks

**`fabricpc/utils/dashboarding/callbacks.py`:**
- `create_iter_callback` (line 31): energy is now per-sample. Remove `energy / batch_size`. Keep `batch_size` param but make it a no-op (or remove).
- `create_detailed_iter_callback` (line 169): same fix.
- `create_tracking_callbacks`: batch_size param becomes vestigial.

**`examples/resnet18_cifar10_demo.py` line 470-471:**
```python
# OLD: norm = float(energy) / args.batch_size
# NEW: norm = float(energy)  # already per-sample
```

### Step 10: Update example imports and call sites

| File | Changes |
|------|---------|
| `examples/mnist_multi_gpu.py` | Import `train_pcn, evaluate_pcn` instead of `*_multi_gpu`. Destructure: `params, _, _ = train_pcn(...)` |
| `examples/transformer_v2_demo.py` | Import `train_pcn, evaluate_transformer`. Destructure return. |
| `examples/transformer_tuning.py` | Import `train_pcn, evaluate_transformer`. Destructure return. |
| `examples/resnet18_cifar10_demo.py` | Fix iter_cb energy normalization (step 9). |

### Step 11: Update tests

**`tests/test_multi_gpu.py`:**
- Update imports from `fabricpc.training.multi_gpu` → `fabricpc.training.train` or `fabricpc.training`
- `train_pcn_multi_gpu(...)` → `train_pcn(..., pmap_single_device=True)`, destructure 3-tuple
- `test_numerical_similarity` is the key correctness test: verifies JIT path ≈ pmap path on 1 device

### Step 12: Update documentation

**`docs/user_guides/08_training_and_evaluation.md`:**
- Replace multi-GPU section with unified API examples showing `train_pcn` works transparently on any device count.

### Step 13: Run tests and validate

- Run full test suite
- Verify `test_numerical_similarity` passes (JIT ≈ pmap on 1 device)
- Run `examples/mnist_demo.py` (single-device path)
- Run `examples/mnist_multi_gpu.py` (updated imports)

---

## Files Modified

| File | Action |
|------|--------|
| `fabricpc/training/train.py` | **Primary target** — receives pmap utils, unified train_pcn, unified evaluate_pcn, evaluate_transformer |
| `fabricpc/training/multi_gpu.py` | Replaced with deprecation shim |
| `fabricpc/training/__init__.py` | Update exports, add backward-compat aliases |
| `fabricpc/utils/dashboarding/callbacks.py` | Remove batch_size energy normalization |
| `examples/mnist_multi_gpu.py` | Update imports + destructure |
| `examples/transformer_v2_demo.py` | Update imports + destructure |
| `examples/transformer_tuning.py` | Update imports + destructure |
| `examples/resnet18_cifar10_demo.py` | Fix iter_cb energy double-normalization |
| `tests/test_multi_gpu.py` | Update imports + call sites |
| `docs/user_guides/08_training_and_evaluation.md` | Simplify multi-GPU section |

---

## Key Risk: verbose + tqdm interaction

When `tqdm=True`, suppress the per-epoch `print()`. TQDM provides epoch info in its description. When `tqdm=False, verbose=True`, print as before. This avoids double output.


The consolidation is complete. Here's a summary of what was accomplished:                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                           
  Unified train.py — Single trainer with auto device detection:                                                                                                                                                                                                                                            
  - JIT path for 1 device, pmap for N>1 devices                                                                                                                                                                                                                                                            
  - pmap_single_device=False kwarg to force pmap on 1 device                                                                                                                                                                                                                                               
  - Built-in TQDM progress bar (tqdm=True by default), counting total minibatches                                                                                                                                                                                                                          
  - 3-tuple return: (params, iter_results, epoch_results)                                                                                                                                                                                                                                                  
  - Energy normalization moved inside get_graph_param_gradient (per-sample)                                                                                                                                                                                                                                
  - _convert_batch helper eliminates 5 duplicate batch conversion blocks                                                                                                                                                                                                                                   
  - evaluate_transformer ported from multi_gpu.py                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                           
  multi_gpu.py — Converted to thin deprecation shim (48 lines) delegating to train.py                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                           
  Updated consumers:                                                                                                                                                                                                                                                                                       
  - __init__.py — new exports + backward-compat aliases                                                                                                                                                                                                                                                    
  - callbacks.py — removed redundant energy normalization                                                                                                                                                                                                                                                  
  - 4 examples — updated imports and destructured returns                                                                                                                                                                                                                                                  
  - test_multi_gpu.py — updated imports, all 6 tests pass                                                                                                                                                                                                                                                  
  - Docs — simplified multi-GPU section                          