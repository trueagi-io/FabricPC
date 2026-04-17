# Plan: Momentum Inference for Deep PC Networks

TODO merge this plan with spiking precision inference strategy, since they both address the same root cause of local signal propagation in deep chains.


## Status of Previous Work (Complete)

Two corrections were implemented and merged:
1. **Forward variance gain** (`variance_gain()`): `a = gain / sqrt(fan_in * K)` — preserves O(1) forward activation variance
2. **Jacobian compensation** (`jacobian_gain()`): `topdown_grad_scale = a * jac_gain` — normalizes per-hop gradient factor to ~1.0

Both are theoretically sound with passing tests (50 muPC, 146 total), but had **zero practical impact on MNIST accuracy**:

| Layers | Before corrections | After corrections |
|--------|-------------------|-------------------|
| 8      | 90.8%             | 91.6%             |
| 32     | 59.9%             | 56.4%             |
| 64     | 20.8%             | 21.0%             |

## Root Cause: Local Inference Signal Propagation

Energy distribution diagnostic for a 32-layer chain (136 inference steps):
- **8 layers**: energy spread across all layers (h1=0.13, h4=0.004, h8=0.023)
- **32 layers**: middle layers h8-h24 have **zero energy** — signal from both ends doesn't reach them

The bottleneck is **not per-hop gradient magnitude** (that's already ~1.0 per hop). It's that:
1. `FeedforwardStateInit` initializes `z_latent = z_mu` → zero error everywhere except output
2. Local PC inference propagates the output error exactly **1 hop per step**
3. The self-error restoring force (`z_latent - z_mu` in energy functional) damps the signal at each hop
4. After 136 steps (~4 chain traversals), the damped signal hasn't meaningfully reached middle layers

This is analogous to SGD in a narrow valley — the gradient points the right way but convergence is slow. The classic fix is **momentum**.

## Solution: InferenceSGDMomentum

Add a new inference class with heavy-ball momentum:

```
v(t+1) = momentum * v(t) - eta * grad
z_latent(t+1) = z_latent(t) + v(t+1)
```

Once a layer receives a gradient impulse (even small), the velocity carries that information forward in subsequent steps, countering the self-error damping. With momentum=0.9, the effective signal window is ~10 steps (`0.9^10 ≈ 0.35`), dramatically accelerating deep signal propagation.

## Implementation

### Step 1: Add `InferenceSGDMomentum` class

**File**: `fabricpc/core/inference.py` (insert after `InferenceSGDNormClip`, line 323)

Override `run_inference` (not `compute_new_latent`) to carry velocity state through the fori_loop without modifying `GraphState` or `NodeState` types:

```python
class InferenceSGDMomentum(InferenceBase):
    def __init__(self, eta_infer=0.1, infer_steps=20, latent_decay=0.0, momentum=0.9):
        super().__init__(eta_infer=eta_infer, infer_steps=infer_steps,
                         latent_decay=latent_decay, momentum=momentum)

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        # Fallback (not used in practice — momentum path overrides run_inference)
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]
        return node_state.z_latent * (1.0 - eta_infer * latent_decay) - eta_infer * node_state.latent_grad

    @staticmethod
    def run_inference(params, initial_state, clamps, structure):
        inference_obj = structure.config["inference"]
        config = inference_obj.config
        infer_steps, eta_infer = config["infer_steps"], config["eta_infer"]
        latent_decay, momentum = config["latent_decay"], config["momentum"]

        # Initialize velocity to zeros (same shape as each z_latent)
        velocity = {name: jnp.zeros_like(ns.z_latent)
                    for name, ns in initial_state.nodes.items()}

        def body_fn(t, carry):
            state, vel = carry
            # Phase 1 & 2: zero grads + forward pass (reuse base class)
            state = InferenceSGDMomentum.zero_grads(params, state, clamps, structure)
            state = InferenceSGDMomentum.forward_value_and_grad(params, state, clamps, structure)
            # Phase 3: momentum update
            new_nodes = dict(state.nodes)
            new_vel = {}
            for node_name in structure.nodes:
                if node_name not in clamps:
                    ns = state.nodes[node_name]
                    v = momentum * vel[node_name] - eta_infer * ns.latent_grad
                    new_z = ns.z_latent * (1.0 - eta_infer * latent_decay) + v
                    new_vel[node_name] = v
                    new_nodes[node_name] = ns._replace(z_latent=new_z)
                else:
                    new_vel[node_name] = vel[node_name]
            return (state._replace(nodes=new_nodes), new_vel)

        final_state, _ = jax.lax.fori_loop(0, infer_steps, body_fn, (initial_state, velocity))
        return final_state
```

**Key design decision**: Override `run_inference` (not `compute_new_latent`) because `compute_new_latent` has no access to velocity state. The velocity dict lives only inside the fori_loop carry — no changes to `NodeState`, `GraphState`, or `types.py` pytree registrations.

The loop carry `(GraphState, Dict[str, jnp.ndarray])` is a valid JAX pytree with invariant structure across iterations.

### Step 2: Export the new class

**File**: `fabricpc/core/__init__.py`

- Add `InferenceSGDMomentum` to the import from `fabricpc.core.inference` (line 46)
- Add `"InferenceSGDMomentum"` to `__all__` (line 97)

### Step 3: Update the MNIST demo

**File**: `examples/mupc_mnist_demo.py`

- Import `InferenceSGDMomentum`
- Switch from `InferenceSGD(eta_infer=0.1, infer_steps=...)` to `InferenceSGDMomentum(eta_infer=0.1, infer_steps=..., momentum=0.9)`
- Add `--momentum` CLI argument (default 0.9)

### Step 4: Add tests

**File**: `tests/test_fabricpc.py` (extend existing test file)

1. **Smoke test**: Build 4-layer chain, run inference with momentum, verify energy decreases
2. **momentum=0 equivalence**: Verify `InferenceSGDMomentum(momentum=0.0)` matches `InferenceSGD`
3. **Deep chain energy**: Build 32-layer chain, compare final total energy between SGD and SGD+momentum (momentum should yield lower energy)
4. **JIT compatibility**: Verify `jax.jit` works with the momentum inference loop

### Step 5: Verify

1. `python -m pytest tests/ -v` — all tests pass
2. `python scripts/diagnose_deep_mupc.py` — verify energy in middle layers is now non-zero
3. `python examples/mupc_mnist_demo.py --num_hidden 8` — should still be ~91%
4. `python examples/mupc_mnist_demo.py --num_hidden 32` — target: improvement over 56%
5. `python examples/mupc_mnist_demo.py --num_hidden 64` — target: improvement over 21%

## Files to Modify

| File | Change |
|---|---|
| `fabricpc/core/inference.py` | Add `InferenceSGDMomentum` class (~50 lines) after line 323 |
| `fabricpc/core/__init__.py` | Add import + `__all__` entry |
| `examples/mupc_mnist_demo.py` | Switch to momentum inference, add CLI arg |
| `tests/test_fabricpc.py` | Add momentum inference tests |

**No changes to**: `types.py`, `base.py`, `state_initializer.py`, `train.py`, `mupc.py`, `activations.py`, or any existing tests.

## Risks

1. **Momentum too high**: Could cause oscillation/instability. Mitigated: 0.9 is standard, user can tune via constructor arg.
2. **JAX tracing with dict carry**: The velocity dict must have identical keys and array shapes every iteration. Guaranteed because we always update all keys (clamped nodes keep zero velocity).
3. **Inference tracking**: `run_inference_with_history` in `inference_tracking.py` calls `inference_step` directly, so the history path won't use momentum. Acceptable limitation for initial implementation; can be addressed later if needed.
