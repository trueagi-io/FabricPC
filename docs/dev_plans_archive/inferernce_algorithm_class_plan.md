# Refactor: Extensible Inference Abstraction (Completed 3/5/26)

## Goal

Encapsulate `inference.py:run_inference()` and `inference_step()` into an extensible class hierarchy. The **latent update rule** (Phase 3 of inference) is the primary extension point. Users can swap inference algorithms (SGD, Adam, natural gradient, etc.) via:

Implement gradient norm clipping in a new inference class InferenceSGDNormCLip:compute_new_latent(). Do this for each node's latent gradient before applying the update. Add a small epsilon or use safe division because grads can be zero.

```python
# This is the manual equivalent of what clip_by_global_norm does:
grads_flat, _ = jax.tree_util.tree_flatten(grads)
global_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
g_factor = jnp.minimum(1.0, max_norm / global_l2)
grads = jax.tree_util.tree_map(lambda g: g * g_factor, grads)

```python
structure = graph(
    nodes=[...], edges=[...], task_map=...,
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)
```

Both `eta_infer` and `infer_steps` move from training config to the inference object.

## Design: Template Method Pattern

The inference step has 3 phases. Phases 1-2 are universal PC mechanics (shared). Phase 3 (latent update) is what varies between algorithms. The design uses **Template Method**: a shared `inference_step()` calls an overridable `latent_update()` hook.

```
InferenceBase (ABC)
  |-- latent_update()       # REQUIRED override: Phase 3 latent update rule
  |-- inference_step()      # Optional override: full single step (Phases 1-3)
  |-- run_inference()       # Optional override: outer loop (e.g. lax.fori_loop vs scan)
  |
  +-- InferenceSGD          # Default: z -= eta * grad
```

### Before / After

```python
# BEFORE
train_config = {
    "num_epochs": 20,
    "infer_steps": 20,
    "eta_infer": 0.05,
}
# eta_infer threaded through train_step -> run_inference -> inference_step

# AFTER
structure = graph(
    nodes=[...], edges=[...], task_map=...,
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)
train_config = {
    "num_epochs": 20,
    # eta_infer and infer_steps gone - they're inference algorithm properties
}
```

---

## Class Design

### `InferenceBase` (ABC) in `fabricpc/core/inference.py`

Follows the existing `StateInitBase` pattern: `__init__(**config)` stores config, static methods for JAX purity, config dict passed explicitly.

```python
class InferenceBase(ABC):
    """Abstract base class for inference algorithms."""

    def __init__(self, **config):
        self.config = types.MappingProxyType(config)  # Immutable dictionary

    @staticmethod
    @abstractmethod
    def latent_update(
        node_name: str,
        node_state: NodeState,
        is_clamped: bool,
        clamps: Dict[str, jnp.ndarray],
        config: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Compute the updated z_latent for a single node.
        This is the primary extension point.

        Args:
            node_name: Name of the node
            node_state: Current node state (has z_latent, latent_grad, substructure)
            is_clamped: Whether this node is clamped
            clamps: All clamp values
            config: Inference algorithm config (from instance .config)

        Returns:
            Updated z_latent array
        """
        pass

    @staticmethod
    def inference_step(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """
        Single inference step: zero grads -> forward -> latent_update.
        Override for algorithms that need different phase structure.
        """
        inference_cls = ...  # resolved from structure.config["inference"]

        # Phase 1: Zero latent gradients
        for node_name in structure.nodes:
            ...

        # Phase 2: Forward inference pass (shared)
        state = _forward_phase(params, state, clamps, structure)

        # Phase 3: Latent update (dispatched to subclass)
        for node_name in structure.nodes:
            node_state = state.nodes[node_name]
            new_z_latent = inference_cls.latent_update(
                node_name, node_state, node_name in clamps, clamps, config,
            )
            state = update_node_in_state(state, node_name, z_latent=new_z_latent)

        # Reset substructure
        for node_name in structure.nodes:
            state = update_node_in_state(state, node_name, substructure={})

        return state

    @staticmethod
    def run_inference(
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """Outer inference loop. Override for scan-based or adaptive stopping.
        infer_steps comes from self.config['infer_steps']."""
        inference_obj = structure.config["inference"]
        inference_cls = type(inference_obj)
        config = inference_obj.config
        infer_steps = config["infer_steps"]

        def body_fn(t, state):
            return inference_cls.inference_step(params, state, clamps, structure, config)

        return jax.lax.fori_loop(0, infer_steps, body_fn, initial_state)
```

### `InferenceSGD` in `fabricpc/core/inference.py`

```python
class InferenceSGD(InferenceBase):
    """Standard SGD inference: z -= eta * grad."""

    def __init__(self, eta_infer=0.1, infer_steps=20):
        super().__init__(eta_infer=eta_infer, infer_steps=infer_steps)

    @staticmethod
    def latent_update(node_name, node_state, is_clamped, clamps, config):
        if is_clamped:
            return clamps[node_name]
        eta_infer = config["eta_infer"]
        return node_state.z_latent - eta_infer * node_state.latent_grad
```

### Helper: `_forward_phase()` (private, in `fabricpc/core/inference.py`)

Extract the shared Phase 1-2 logic (zero gradients + forward pass + local backward) into a reusable function. This is used by the default `inference_step()` and available to custom subclasses.

```python
def _forward_phase(params, state, clamps, structure):
    """Phases 1-2: Zero gradients, forward inference, accumulate latent grads."""
    # Phase 1: Zero gradients
    for node_name in structure.nodes:
        node_state = state.nodes[node_name]
        zero_grad = jnp.zeros_like(node_state.z_latent)
        state = update_node_in_state(state, node_name, latent_grad=zero_grad)

    # Phase 2: Forward + local backward
    for node_name in structure.nodes:
        node = structure.nodes[node_name]
        node_info = node.node_info
        node_class = node_info.node_class
        node_state = state.nodes[node_name]
        node_params = params.nodes[node_name]

        in_edges_data = gather_inputs(node_info, structure, state)
        node_state, inedge_grads = node_class.forward_inference(
            node_params, in_edges_data, node_state, node_info,
            is_clamped=(node_name in clamps),
        )
        state = state._replace(nodes={**state.nodes, node_name: node_state})

        for edge_key, grad in inedge_grads.items():
            source_name = structure.edges[edge_key].source
            latent_grad = state.nodes[source_name].latent_grad + grad
            state = update_node_in_state(state, source_name, latent_grad=latent_grad)

    return state
```

---

## Integration: `graph()` builder

### `fabricpc/builder/graph_builder.py`

Add `inference` parameter to `graph()`, store in `GraphStructure.config`:

```python
def graph(nodes, edges, task_map, graph_state_initializer=None, inference=None):
    from fabricpc.core.inference import InferenceSGD

    gs_config = {
        "graph_state_initializer": graph_state_initializer or FeedforwardStateInit(),
        "inference": inference or InferenceSGD(),
    }
    ...
```

---

## Migration: `eta_infer` and `infer_steps` move from config to inference object

Both are properties of the inference algorithm, not training hyperparameters. After this refactor:
- **Inference object owns `eta_infer`** (learning rate for the update rule)
- **Inference object owns `infer_steps`** (number of inference iterations)
- Training config retains only `num_epochs` (and loss-related settings)
- All callers of `run_inference` stop passing both parameters — they come from `structure.config["inference"].config`

---

## Backward Compatibility

Keep module-level wrapper functions that delegate to the class-based implementation:

```python
# Backward-compatible wrappers (keep existing import paths working)
def run_inference(params, initial_state, clamps, structure, infer_steps=None, eta_infer=None):
    """Backward-compatible wrapper. Prefer structure.config['inference'] for new code.
    If infer_steps/eta_infer are passed, they override the inference object's config."""
    inference_obj = structure.config.get("inference", InferenceSGD())
    # Allow call-site overrides for backward compat
    config = dict(inference_obj.config)
    if infer_steps is not None:
        config["infer_steps"] = infer_steps
    if eta_infer is not None:
        config["eta_infer"] = eta_infer
    inference_cls = type(inference_obj)
    n_steps = config["infer_steps"]

    def body_fn(t, state):
        return inference_cls.inference_step(params, state, clamps, structure, config)

    return jax.lax.fori_loop(0, n_steps, body_fn, initial_state)

def inference_step(params, state, clamps, structure, eta_infer=None):
    """Backward-compatible wrapper."""
    inference_obj = structure.config.get("inference", InferenceSGD())
    config = dict(inference_obj.config)
    if eta_infer is not None:
        config["eta_infer"] = eta_infer
    return type(inference_obj).inference_step(params, state, clamps, structure, config)
```

This means **all 31 existing call sites continue to work without changes** during migration. We then incrementally update callers to drop `eta_infer` and `infer_steps` arguments.

---

## Files to Modify

### Phase 1: Core abstraction (2 files)

1. **`fabricpc/core/inference.py`** — Add `InferenceBase` ABC, `InferenceSGD`, `_forward_phase()`. Refactor existing `inference_step()` and `run_inference()` into backward-compatible wrappers.

2. **`fabricpc/core/__init__.py`** — Export `InferenceBase` and `InferenceSGD` in `__all__`.

### Phase 2: Builder integration (1 file)

3. **`fabricpc/builder/graph_builder.py`** — Add `inference=None` parameter to `graph()`, default to `InferenceSGD()`, store in config.

### Phase 3: Remove `eta_infer` and `infer_steps` from training callers (4 files)

These files currently pass `eta_infer` and `infer_steps` from config into `run_inference()`. After the refactor, they call `run_inference(params, state, clamps, structure)` with no extra args — both come from the inference object.

4. **`fabricpc/training/train.py`** — `train_step()`, `eval_step()`, `get_graph_param_gradient()`: drop `eta_infer` and `infer_steps` arguments. Call `run_inference(params, state, clamps, structure)`. Remove `eta_infer`/`infer_steps` from function signatures and config reads.

5. **`fabricpc/training/multi_gpu.py`** — `inference_fn()`, `create_pmap_train_step()`: same pattern.

6. **`fabricpc/training/train_autoregressive.py`** — `train_step_autoregressive()`, `generate_autoregressive()`, `generate_autoregressive_with_sampling()`: same pattern. Note: `generate_autoregressive` takes `infer_steps` as a direct param — remove it, let inference object handle it.

7. **`fabricpc/utils/dashboarding/inference_tracking.py`** — `run_inference_with_history()`, `run_inference_no_scan()`: switch from direct `inference_step()` calls to using the inference object from structure config. Remove `eta_infer` and `infer_steps` parameters.

### Phase 4: Update examples (8 files)

Move `eta_infer` and `infer_steps` from `train_config` to `graph(..., inference=InferenceSGD(eta_infer=..., infer_steps=...))`:

8. **`examples/mnist_demo.py`**
9. **`examples/mnist_advanced.py`**
10. **`examples/mnist_aim_tracking.py`**
11. **`examples/mnist_multi_gpu.py`**
12. **`examples/mnist_lateral_connections.py`**
13. **`examples/PC_backprop_compare.py`**
14. **`examples/transformer_demo.py`**
15. **`examples/custom_node.py`**

### Phase 5: Update tests (5 files)

16. **`tests/test_fabricpc.py`** — Remove `eta_infer`/`infer_steps` from `run_inference` calls (or let wrapper handle it)
17. **`tests/test_auto_node_grad.py`** — Same
18. **`tests/test_ndim_shapes.py`** — Same (10 call sites)
19. **`tests/test_fabricpc_extended.py`** — Same (6 call sites)
20. **`tests/test_state_initializer.py`** — Same

### Phase 6: Run tests

21. Run full test suite to verify

---

## Order of Operations

1. Remove `substructure` from `NodeState` in `types.py`
2. Refactor `linear.py`: `compute_gain_mod_error()` returns value; `forward_inference()`/`forward_learning()` use return value directly
3. Refactor `transformer.py`: `_mha()` drops substructure return
4. Remove `substructure={}` from state initializers and all `NodeState(...)` constructors in tests
5. Add `InferenceBase`, `InferenceSGD`, `_forward_phase()` to `fabricpc/core/inference.py` — keep existing functions as backward-compatible wrappers, remove substructure reset loop
6. Export new classes from `fabricpc/core/__init__.py`
7. Add `inference` param to `graph()` in `fabricpc/builder/graph_builder.py`
8. Update training callers (4 files) to drop `eta_infer` and `infer_steps` passthrough
9. Update `inference_tracking.py` to use inference object
10. Update examples (8 files) to set `eta_infer` and `infer_steps` on inference object
11. Update tests (5 files) to use new API or rely on backward-compat wrappers
12. Run test suite

---

## Design Rationale

**Why Template Method over Minimal Override or Full Step Override:**
- Most custom inference algorithms only change the latent update rule (Phase 3). `latent_update()` is the smallest, most focused override point.
- Advanced use cases (momentum inference, adaptive stopping) can override `inference_step()` or `run_inference()` without being forced to rewrite shared Phase 1-2 logic.
- `_forward_phase()` as a shared utility prevents code duplication across subclasses.

**Why `eta_infer` and `infer_steps` move to the inference object:**
- `eta_infer` is a property of the SGD update rule, not a training loop parameter. An Adam-based inference would have `beta1`, `beta2` instead.
- `infer_steps` is the iteration count for the inference algorithm. Different algorithms may need different convergence budgets. Coupling it with the algorithm makes the inference object a self-contained unit.

**Why follow `StateInitBase` pattern (`__init__(**config)` + `@staticmethod`):**
- Matches the existing library conventions users already know.
- Static methods with explicit config dict are JAX-friendly (pure functions, no hidden state).
- Instance carries config but computation is stateless.

**Substructure removal:**
- `substructure` field is removed from `NodeState` entirely.
- Linear: `gain_mod_error` was stored in substructure as intermediate state. Since `gain_mod_error = error * f_prime(pre_activation)` and both fields are already on `NodeState`, compute it inline where needed.
- Transformer: `_mha()` returns `substructure_attn` dict but it's never stored or used — dead code, just drop the return value.
- The per-step `substructure={}` reset in `inference_step()` is removed (no field to reset).

---

## Additional Refactor: Remove `substructure` from `NodeState`

### `fabricpc/core/types.py`
Remove `substructure: Dict[str, jnp.ndarray]` field from `NodeState` NamedTuple. Remove from docstring.

### `fabricpc/nodes/linear.py`
- **`compute_gain_mod_error()`**: Instead of storing in `state.substructure`, return `gain_mod_error` directly.
- **`forward_inference()`**: Call `compute_gain_mod_error()` to get the value directly (or compute `error * f_prime(pre_activation)` inline), use it for gradient computation without touching `substructure`.
- **`forward_learning()`**: Same pattern.

### `fabricpc/nodes/transformer.py`
- **`_mha()`**: Remove the substructure dict return. Return only the projection (or keep as tuple with `_` at call site).
- **`forward()`**: Update call to `_mha()` to not expect substructure return.

### `fabricpc/core/inference.py`
- Remove the `substructure={}` reset loop at the end of `inference_step()`.

### `fabricpc/graph/state_initializer.py`
- Remove `substructure={}` from all three `NodeState(...)` constructors.

### Test files
- Remove `substructure={}` from all `NodeState(...)` constructors in tests:
  - `tests/test_fabricpc.py`
  - `tests/test_identity_node.py`
  - `tests/test_auto_node_grad.py`
  - `tests/test_state_initializer.py`


#  Completed: Extensible Inference Abstraction + Substructure Removal                                                                                
                                                                                                                                                    
  What changed                                                                                                                                      
                                                                                                                                                    
  1. Removed substructure from NodeState (types.py, linear.py, transformer.py, state initializers, tests)                                           
  - compute_gain_mod_error() now returns the value directly instead of storing in substructure                                                      
  - Transformer _mha() dropped dead substructure return                                                                                             
  - Removed substructure={} from all NodeState constructors                                                                                         
                                                                                                                                                    
  2. Added InferenceBase / InferenceSGD class hierarchy (inference.py)                                                                              
  - Template Method pattern: latent_update() is the primary extension point                                                                         
  - _forward_phase() extracts shared Phase 1-2 logic                                                                                                
  - Backward-compatible run_inference() / inference_step() wrapper functions preserved                                                              
                                                                                                                                                    
  3. Integrated with graph() builder (graph_builder.py)                                                                                             
  - New inference= parameter, defaults to InferenceSGD()                                                                                            
  - Stored in GraphStructure.config["inference"]                                                                                                    
                                                                                                                                                    
  4. Migrated eta_infer and infer_steps from training config to inference object                                                                    
  - Updated 4 training modules (train.py, multi_gpu.py, train_autoregressive.py, inference_tracking.py)                                             
  - Updated 10 example files                                                                                                                        
  - Updated 5 test files                                                                                                                            
  - Both parameters are now properties of InferenceSGD, not training hyperparameters                                                                
                                                                                                                                                    
  New API                                                                                                                                           
                                                                                                                                                    
  structure = graph(                                                                                                                                
      nodes=[...], edges=[...], task_map=...,                                                                                                       
      inference=InferenceSGD(eta_infer=0.05, infer_steps=20),                                                                                       
  )                                                                                                                                                 
  train_config = {"num_epochs": 20}  # eta_infer/infer_steps gone   