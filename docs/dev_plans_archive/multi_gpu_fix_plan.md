# Multi-GPU Training Fix Plan (Completed)

## Problem Summary

When running multi-GPU training on a dual GPU system, the energy is reported in **billions** while single-GPU training reports energy of only a **few hundred**.
The Multi-GPU code diverged early in development and is missing local learning dynamics and uses a deprecated initialization config.

## Root Cause Analysis

### Issue 1: State Initialization Config Mismatch (Critical)

The multi-GPU and single-GPU training use different config keys for state initialization:

| File | Line | Config Key Used |
|------|------|-----------------|
| `fabricpc/training/train.py` | 112-113 | `structure.config["graph_state_initializer"]` |
| `fabricpc/training/multi_gpu.py` | 250 | `config.get("state_initialization", None)` |

**Impact:** When the training config lacks a `"state_initialization"` key, multi_gpu.py passes `None` to `initialize_graph_state()`. This causes:
- Different/invalid initial states
- Inference fails to converge
- Energy values explode to billions

### Issue 2: Different Gradient Computation Methods (Critical)

| Aspect | Single GPU                         | Multi GPU |
|--------|------------------------------------|-----------|
| Method | Local Hebbian learning             | Autodiff through inference |
| Implementation | `compute_local_weight_gradients()` | `jax.value_and_grad(loss_fn)` |
| Location | `graph_net.py:53`                  | `multi_gpu.py:155` |

The multi-GPU version uses backpropagation through the entire inference loop, while single-GPU uses local learning rules.

---

## Proposed Fixes

### Fix 1: Align State Initialization Config (Done)

**File:** `fabricpc/training/multi_gpu.py`

**Change line 250 from:**
```python
state_init_config = config.get("state_initialization", None)
```

**To:**
```python
state_init_config = structure.config.get("graph_state_initializer")
```

This ensures multi-GPU training uses the same state initialization as single-GPU.


### Fix 2: Apply Same Fix to Evaluation (Done)

**File:** `fabricpc/training/multi_gpu.py`

Line 348 has the same issue:
```python
state_init_config = config.get("state_initialization", None)
```

Apply the same fix as above.

---

### Fix 3: Hebbian learning in multi-GPU (Required)
- Fully align code with single GPU training by using shared local gradient code
- Shard the gradients across devices
- Use shared code at the shard level, e.g. leverage train.get_graph_param_gradient()
- Use `jax.lax.pmean` to average gradients across devices
- Ensure the optimizer is applied to the averaged gradients like in single-GPU training

### Fix 4: Consistent Signatures (Required)
- Match signature between single and multi-gpu training, e.g. they should return same variables and track both the energy and output loss


## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `fabricpc/training/multi_gpu.py` | 250 | Fix state_init_config source |
| `fabricpc/training/multi_gpu.py` | 348 | Fix state_init_config source (eval) |
| `fabricpc/training/multi_gpu.py` | 150-200 | Refactor gradient computation to use local Hebbian learning |
| `fabricpc/training/multi_gpu.py` | 220-270 | Refactor optimizer application to use averaged gradients |
---

## Testing Plan (test implementation done)

tests/test_multi_gpu.py

1. **Unit test:** Verify `state_init_config` is correctly read from `structure.config`
2. **Integration test:** Run same model on single-GPU and multi-GPU, compare energy values
3. **Regression test:** Ensure existing multi-GPU workflows still function if they were passing config correctly

### Verification

```python
# After fix, these should produce similar energy and output loss values:
# Single GPU
params, losses_single, _ = train_pcn(params, structure, loader, config, rng)

# Multi GPU
params_multi = train_pcn_multi_gpu(params, structure, loader, config, rng)
```

Expected: energy values within numerical precision between single gpu and multi-gpu with single shard.

---

## Summary
Before vs After:                                                                                                                                                                                                                         
                                                                                                                                                                                                                                           
  | Aspect            | Before                                 | After                                      |                                                                                                                              
  |-------------------|----------------------------------------|--------------------------------------------|                                                                                                                              
  | Gradient method   | jax.value_and_grad(loss_fn) (backprop) | get_graph_param_gradient() (local Hebbian) |                                                                                                                              
  | Shared code       | Duplicated logic                       | Uses same function as single-GPU           |                                                                                                                              
  | State init config | Passed as parameter                    | Read from structure.config internally      |                                                                                                                              
  ___

Test Results:                                                                                                                                                                                                                            
  - test_numerical_similarity: PASSED (was failing with 58% difference, now within 1e-5)                                                                                                                                                   
  - test_parameter_magnitudes_similar: PASSED (was failing, now within 1e-5)                                                                                                                                                               
  - Full test suite: 166 passed
