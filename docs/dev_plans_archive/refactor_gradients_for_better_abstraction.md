# Refactor: Unified Autodiff Gradient Flows + muPC Scaling Abstraction

## Context

FabricPC's gradient computation has two problems:

1. **Dual-path gradients**: Self-latent gradients (`dE/dz_latent`) are computed explicitly via `energy_cls.grad_latent()` inside `energy_functional()`, while input gradients (`dE/d(inputs)`) use `jax.value_and_grad`. These should use one uniform mechanism.

2. **muPC scaling baked into NodeBase**: Forward scaling, topdown gradient scaling, self-gradient scaling, and weight gradient scaling are all hardcoded into `forward_inference()`, `forward_learning()`, and `energy_functional()`. Scaling should be a composable layer that node methods don't need to know about.

**Outcome**: Node methods become pure autodiff. muPC scaling is applied at the callsite. Users who override `forward_inference()` / `forward_learning()` with explicit gradients don't need to handle scaling at all.

---

## Part 1: Unified Autodiff via z_latent Extraction

### Mechanism

Extract `z_latent` from `NodeState` and pass it alongside `inputs` as a separate differentiable argument. A closure bridges the gap to `forward()`:

```python
def energy_fn(input_args, z_latent):
    s = state._replace(z_latent=z_latent)
    total_energy, new_s = node_class.forward(params, input_args, s, node_info)
    return total_energy, new_s

(total_energy, new_state), (input_grads, z_latent_grad) = jax.value_and_grad(
    energy_fn, argnums=(0, 1), has_aux=True
)(inputs, state.z_latent)
```

This differentiates w.r.t. exactly two things:
- `inputs` (dict of edge_key -> array) at argnums=0
- `z_latent` (single array) at argnums=1

The closure captures `state` (concrete, not traced), `params`, and `node_info`. The `_replace` creates a new `NodeState` with the traced `z_latent` — JAX correctly traces through pytree construction.

### Changes to `fabricpc/nodes/base.py`

**A. `energy_functional()` becomes energy-only** (line 535-568):

```python
@staticmethod
def energy_functional(state: NodeState, node_info: NodeInfo) -> NodeState:
    energy_obj = node_info.energy
    energy_cls = type(energy_obj)
    config = energy_obj.config
    energy = energy_cls.energy(state.z_latent, state.z_mu, config)
    return state._replace(energy=energy)
```

Remove: `grad_latent()` call, `self_grad_scale` application, `latent_grad` accumulation.

**B. `forward_inference()` uses the closure pattern** (lines 444-465):

The internal/clamped node branch becomes:

```python
else:
    # Closure extracts z_latent as separate differentiable arg
    def energy_fn(input_args, z_latent):
        s = state._replace(z_latent=z_latent)
        total_energy, new_s = node_class.forward(params, input_args, s, node_info)
        return total_energy, new_s

    (total_energy, new_state), (input_grads, z_latent_grad) = jax.value_and_grad(
        energy_fn, argnums=(0, 1), has_aux=True
    )(inputs, state.z_latent)

    # Accumulate self-gradient into state
    new_state = new_state._replace(
        latent_grad=new_state.latent_grad + z_latent_grad
    )
```

No muPC scaling here — that moves to Part 2.

**C. `forward_learning()` becomes pure autodiff** (lines 479-532):

```python
@staticmethod
def forward_learning(params, inputs, state, node_info):
    node_class = node_info.node_class
    (total_energy, new_state), params_grad = jax.value_and_grad(
        node_class.forward, argnums=0, has_aux=True
    )(params, inputs, state, node_info)
    return new_state, params_grad
```

Remove: `_apply_forward_scaling()` call, `weight_grad_scale` application. These move to Part 2.

**D. Remove `_apply_forward_scaling()` from NodeBase** (lines 353-373):

This moves to the scaling utility in Part 2. NodeBase no longer has any muPC knowledge.

### Changes to `fabricpc/nodes/storkey_hopfield.py`

**Rename `accumulate_hopfield_energy_and_grad()` to `accumulate_hopfield_energy()`** (lines 246-275):

```python
@staticmethod
def accumulate_hopfield_energy(state, W, strength):
    z = state.z_latent
    wz = z @ W
    D = z.shape[-1]
    E_hopfield = (0.5 / D) * jnp.sum(wz * (wz - z), axis=-1)
    return state._replace(energy=state.energy + strength * E_hopfield)
```

Remove: `hopfield_grad` computation and `latent_grad` accumulation. JAX autodiff through the quadratic `(0.5/D) z^T(W^2-W)z` automatically produces `(1/D)(W^2-W)z`.

Update `forward()` (line 341) to call `accumulate_hopfield_energy` instead.

### Changes to `fabricpc/nodes/linear.py` — LinearExplicitGrad

`LinearExplicitGrad.forward_inference()` (line 265-313) bypasses autodiff. Since `energy_functional()` no longer accumulates the self-gradient, add explicit self-gradient computation:

```python
_, state = node_class.forward(params, inputs, state, node_info)

# Explicit self-gradient (override pattern)
energy_obj = node_info.energy
self_grad = type(energy_obj).grad_latent(
    state.z_latent, state.z_mu, energy_obj.config
)
state = state._replace(latent_grad=state.latent_grad + self_grad)

# ... existing explicit input gradient computation ...
```

Note: no muPC scaling here. That's applied by the callsite (Part 2), transparent to the node override.

### Changes to `fabricpc/nodes/transformer_v2.py` — EmbeddingNode

`EmbeddingNode.forward_inference()` (line 115) calls `forward()` directly and returns zero input grads. Since `energy_functional()` no longer writes `latent_grad`, embedding nodes won't get a self-gradient. This is correct because embeddings are typically clamped (latent_grad is irrelevant for clamped nodes). No change needed.

### No changes needed in these node forward() implementations

These call `energy_functional()` inside `forward()`. After the refactor, `energy_functional()` only sets energy. The autodiff in `forward_inference()` handles the self-gradient. Zero code changes:

- `fabricpc/nodes/linear.py` — `Linear.forward()`
- `fabricpc/nodes/identity.py` — `IdentityNode.forward()`
- `fabricpc/nodes/skip_connection.py` — `SkipConnection.forward()`
- `fabricpc/nodes/linear_residual.py` — `LinearResidual.forward()`
- `fabricpc/nodes/transformer.py` — `TransformerBlock.forward()`
- `fabricpc/nodes/transformer_v2.py` — all node `forward()` methods

---

## Part 2: muPC Scaling Abstraction

### Design

Move all scaling logic from NodeBase methods to utility functions. The callsites (inference loop, learning loop) compose scaling with node computation. Nodes are scaling-unaware.

### New module: `fabricpc/core/scaling.py`

```python
"""muPC scaling utilities — composable layer applied at callsites."""

def scale_inputs(inputs, scaling_config):
    """Pre-scale inputs by muPC forward scaling factors."""
    if scaling_config is None:
        return inputs
    return {
        edge_key: x * scaling_config.forward_scale[edge_key]
        for edge_key, x in inputs.items()
    }

def scale_input_grads(input_grads, scaling_config):
    """Post-scale input gradients by topdown gradient scaling factors."""
    if scaling_config is None:
        return input_grads
    return {
        edge_key: grad * scaling_config.topdown_grad_scale[edge_key]
        for edge_key, grad in input_grads.items()
    }

def scale_self_grad(z_latent_grad, scaling_config):
    """Post-scale self-latent gradient."""
    if scaling_config is None:
        return z_latent_grad
    return z_latent_grad * scaling_config.self_grad_scale

def scale_weight_grads(params_grad, scaling_config):
    """Post-scale weight gradients by muPC weight gradient scaling."""
    if scaling_config is None:
        return params_grad
    wg_scale = scaling_config.weight_grad_scale
    if all(k in wg_scale for k in params_grad.weights):
        scaled_weights = {
            k: grad * wg_scale[k] for k, grad in params_grad.weights.items()
        }
    else:
        uniform_scale = sum(wg_scale.values()) / len(wg_scale)
        scaled_weights = {
            k: grad * uniform_scale for k, grad in params_grad.weights.items()
        }
    return NodeParams(weights=scaled_weights, biases=params_grad.biases)
```

### Changes to `fabricpc/core/inference.py` — `forward_value_and_grad()`

Apply scaling at the callsite, wrapping `forward_inference()`:

```python
from fabricpc.core.scaling import scale_inputs, scale_input_grads, scale_self_grad

def forward_value_and_grad(params, state, clamps, structure):
    for node_name in structure.nodes:
        ...
        in_edges_data = gather_inputs(node_info, structure, state)
        sc = node_info.scaling_config

        # Pre-scale inputs
        scaled_inputs = scale_inputs(in_edges_data, sc)

        # Pure autodiff gradient computation (node knows nothing about scaling)
        node_state, inedge_grads = node_class.forward_inference(
            node_params, scaled_inputs, node_state, node_info,
            is_clamped=(node_name in clamps),
        )

        # Post-scale gradients
        inedge_grads = scale_input_grads(inedge_grads, sc)
        scaled_self = scale_self_grad(node_state.latent_grad, sc)
        node_state = node_state._replace(latent_grad=scaled_self)

        # Update graph state and accumulate input grads to upstream nodes
        state = state._replace(nodes={**state.nodes, node_name: node_state})
        for edge_key, grad in inedge_grads.items():
            source_name = structure.edges[edge_key].source
            latent_grad = state.nodes[source_name].latent_grad + grad
            state = update_node_in_state(state, source_name, latent_grad=latent_grad)

    return state
```

### Changes to `fabricpc/graph/graph_net.py` — `compute_local_weight_gradients()`

Apply scaling at the callsite, wrapping `forward_learning()`:

```python
from fabricpc.core.scaling import scale_inputs, scale_weight_grads

def compute_local_weight_gradients(params, final_state, structure):
    gradients = {}
    for node_name, node in structure.nodes.items():
        ...
        in_edges_data = gather_inputs(node_info, structure, final_state)
        sc = node_info.scaling_config

        # Pre-scale inputs
        scaled_inputs = scale_inputs(in_edges_data, sc)

        # Pure autodiff weight gradient computation
        node_state, grad_params = node_class.forward_learning(
            params.nodes[node_name], scaled_inputs,
            final_state.nodes[node_name], node_info,
        )

        # Post-scale weight gradients
        grad_params = scale_weight_grads(grad_params, sc)

        gradients[node_name] = grad_params
    ...
```

### Changes to `fabricpc/nodes/transformer.py` — TransformerBlock.forward_learning()

The override becomes simpler — only the LayerNorm compensation, no muPC boilerplate:

```python
@staticmethod
def forward_learning(params, inputs, state, node_info):
    node_class = node_info.node_class

    # Pure autodiff (inputs are already scaled by callsite)
    (total_energy, new_state), params_grad = jax.value_and_grad(
        node_class.forward, argnums=0, has_aux=True
    )(params, inputs, state, node_info)

    # LayerNorm compensation: LN(a*x) = LN(x) absorbs forward scaling,
    # so dE/dW is independent of a. Multiply by a to compensate.
    if node_info.scaling_config is not None:
        a = 1.0
        for edge_key, scale in node_info.scaling_config.forward_scale.items():
            if edge_key.endswith(":in"):
                a = scale
                break
        if a != 1.0:
            scaled_weights = {k: g * a for k, g in params_grad.weights.items()}
            params_grad = NodeParams(weights=scaled_weights, biases=params_grad.biases)

    return new_state, params_grad
```

Note: the callsite will also apply `scale_weight_grads()` after this returns, but `weight_grad_scale` is 1.0 so it's a no-op. If `weight_grad_scale` changes in the future, the TransformerBlock needs to account for that (or skip the callsite scaling — see verification section).

---

## Part 3: Activation and Energy Convenience APIs

For users who override `forward_inference()` / `forward_learning()` with explicit gradients.

### `ActivationBase.jacobian()` in `fabricpc/core/activations.py`

Add optional method for non-element-wise activations:

```python
# On ActivationBase:
@staticmethod
def jacobian(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
    """Full Jacobian J_{ij} = df_i/dx_j, shape (..., D, D).
    
    For element-wise activations, the Jacobian is diag(derivative(x)) —
    use derivative() directly. Override for non-element-wise activations.
    """
    raise NotImplementedError(
        "jacobian() not implemented for this activation. "
        "For element-wise activations, use derivative() as the diagonal."
    )
```

```python
# On SoftmaxActivation:
@staticmethod
def jacobian(x: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
    """Full softmax Jacobian: J_{ij} = s_i(delta_{ij} - s_j)"""
    s = SoftmaxActivation.forward(x, config)
    eye = jnp.eye(s.shape[-1])
    return jnp.expand_dims(s, -1) * (eye - jnp.expand_dims(s, -2))
```

### `EnergyFunctional.grad_latent()` — keep as-is

`grad_latent()` remains abstract on `EnergyFunctional`. It's no longer called by the framework's autodiff path, but is still the convenience method for explicit gradient implementations. Update docstring to note this.

---

## File Change Summary

| File | Change |
|------|--------|
| **`fabricpc/nodes/base.py`** | `energy_functional()` energy-only. `forward_inference()` z_latent closure. `forward_learning()` pure autodiff. Remove `_apply_forward_scaling()`. |
| **`fabricpc/core/scaling.py`** | **NEW**. `scale_inputs()`, `scale_input_grads()`, `scale_self_grad()`, `scale_weight_grads()`. |
| **`fabricpc/core/inference.py`** | `forward_value_and_grad()`: apply scaling at callsite around `forward_inference()`. |
| **`fabricpc/graph/graph_net.py`** | `compute_local_weight_gradients()`: apply scaling at callsite around `forward_learning()`. |
| **`fabricpc/nodes/storkey_hopfield.py`** | `accumulate_hopfield_energy_and_grad()` → `accumulate_hopfield_energy()` (energy only). |
| **`fabricpc/nodes/linear.py`** | `LinearExplicitGrad.forward_inference()`: add explicit self-gradient. |
| **`fabricpc/nodes/transformer.py`** | `TransformerBlock.forward_learning()`: remove muPC boilerplate, keep LN compensation. |
| **`fabricpc/core/activations.py`** | Add `jacobian()` to `ActivationBase` + `SoftmaxActivation`. |
| **`fabricpc/core/energy.py`** | Update `grad_latent()` docstring. |

**No changes needed**: `Linear.forward()`, `IdentityNode.forward()`, `SkipConnection.forward()`, `LinearResidual.forward()`, `TransformerBlock.forward()`, all `transformer_v2.py` forward() methods, `mupc.py` (scaling factors computed the same way).

---

## Verification

### Test: autodiff self-gradient matches explicit
For each energy functional, verify: `jax.grad(lambda z: jnp.sum(energy(z, mu, config)))(z) == grad_latent(z, mu, config)`

### Test: StorkeyHopfield combined gradient
Verify autodiff self-gradient for combined PC+Hopfield energy matches `grad_PC + strength * grad_Hopfield`.

### Test: LinearExplicitGrad still matches Linear
Existing `test_auto_node_grad.py` compares autodiff vs explicit. Add `latent_grad` comparison.

### Test: muPC scaling produces identical results
Run identical graph with scaling applied at callsite vs old code. Compare all gradients numerically.

### Regression: full test suite
`pytest tests/` — all existing tests should pass since mathematical results are identical.

### Benchmark
Compare compilation + runtime before/after for Linear chain, StorkeyHopfield, TransformerBlock graphs. The z_latent closure approach should have no measurable overhead vs. the old argnums=1 approach (one extra scalar differentiation target).
