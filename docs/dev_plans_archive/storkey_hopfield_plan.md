# Storkey Hopfield Node — Design & Implementation Plan (completed 2026-04-08)

## Overview

A Hopfield associative memory node (`StorkeyHopfield`) for FabricPC's predictive coding framework. The node combines standard PC prediction-error energy with a Hopfield attractor energy term. Learning is purely energy-based (JAX autodiff through the combined energy). The input from upstream serves as a "probe pattern" that seeds the attractor dynamics.

## Architecture

```
    probe ------+
                |
                v
              [sum] + [strength * (z_latent @ W)] + bias ----> tanh ----> z_mu
                                    ^
                              Hopfield recurrence
                              (pseudo self-connection)
```

- **Single input edge** (`is_multi_input=False`) — the input is a probe pattern
- **One W matrix** (D, D) — used only for the Hopfield recurrence, stored under the edge key in `params.weights`
- **`input_dim == D` required** — probe must match Hopfield dimension (direct addition, no projection)
- **Forward pass**: `z_mu = tanh(probe + strength * (z_latent @ W) + bias)`
- **`ZerosInitializer` default** — W starts at zero so the node initially behaves as `z_mu = tanh(probe + bias)` (pure passthrough). As W learns, attractor dynamics emerge.

The probe enters as a direct additive term (no W multiplication). W encodes only the associative memory structure.

The Hopfield recurrence (`z_latent @ W`) is internal to `forward()`, not a graph edge (graph builder forbids self-edges). During inference, the system settles to fixed points satisfying `z = tanh(probe + strength * z @ W + b)`.

## Energy Formulation

Two terms summed, both competing on z_latent:

```
E_total = E_pc + hopfield_strength * E_hop

E_pc  = 0.5 ||z - mu||^2              (Gaussian, or user-specified)
E_hop = (1/2N) z^T (W^2 - W) z        (Hopfield attractor energy)

where N = dimension of z (last axis) for scale-invariance.
```

- **E_pc** pulls z toward the upstream prediction mu (trust the upstream prediction)
- **E_hop** pulls z toward stored patterns ξ^μ (trust the internal memory)
- Equilibrium z* is the PC-optimal compromise: top-down expectation meets internal prior

### Hopfield energy on z_latent (with 1/N scaling)

`E_hop(W; z) = (1/2N)(−zᵀWz + ‖Wz‖²)` has equivalent forms:
- `E = (1/2N) zᵀ(W² − W)z`
- `E = (1/2N) (Wz)ᵀ(Wz − z)`

Gradient w.r.t. z:
- `dE_hop/dz = (1/N)(W² − W)z`
- In row-vector code: `(1/D) * (wz @ W - wz)` where `wz = z @ W`

When W's eigenvalues are in (0,1) for stored pattern directions, `W² − W` has negative eigenvalues there, creating energy minima (attractors) at stored patterns.

### Notation

- ξ^μ = stored Hopfield patterns (absolute states in z-space)
- ε = z − μ = PC prediction error (context-dependent residual)
- These are completely different objects. The Hopfield attractor operates on z (absolute state), not on ε (prediction error).

## Design Decisions

### Probe path (no W multiplication)

The probe (upstream input) is added directly to pre-activation without being multiplied by W. This means:
- W=0 at init → node is a pure passthrough: `z_mu = tanh(probe + bias)`
- W only encodes the recurrent associative memory structure
- Probe acts as a persistent external drive seeding the attractor landscape

### Energy computation pattern

The standard PC energy is computed via the normal `energy_functional()` call. The Hopfield energy is then added on top via `accumulate_hopfield_energy_and_grad()`:

```python
# In forward():
state = node_class.energy_functional(state, node_info)                       # standard PC energy
state = StorkeyHopfield.accumulate_hopfield_energy_and_grad(state, W, strength)  # add Hopfield term
total_energy = jnp.sum(state.energy)
```

### Hopfield strength

- If `hopfield_strength=None` (default): learnable scalar parameter initialized to 1.0, stored in `params.biases["hopfield_strength"]`
- If `hopfield_strength=<float>`: fixed value stored in config only (not in params)

### No `forward_inference()` or `forward_learning()` override needed

Base class autodiff on `forward()` correctly computes:
- **Input gradients** (for inference): autodiff w.r.t. inputs flows through the identity addition `probe + ...`
- **Weight gradients** (for learning): autodiff w.r.t. params captures the recurrence path and Hopfield energy
- **Latent self-gradients**: computed inside `forward()` via `energy_functional()` + `accumulate_hopfield_energy_and_grad()`, stored in `state.latent_grad`

### Symmetry and diagonal

- `enforce_symmetry` (default True): symmetrize W via `0.5*(W+W.T)` inside `forward()` and at init
- `zero_diagonal` (default False): zero W diagonal inside `forward()` and at init
- Both are differentiable operations, safe under JAX tracing

### W stored under edge key

W is stored as `params.weights[edge_key]` (not as a separate `_hopfield_W` key). This ensures JAX autodiff in `forward_learning()` (argnums=0 → params) captures W gradients through both the recurrence path and the Hopfield energy term.

## Constructor Parameters

- `shape`, `name` (required)
- `activation=TanhActivation()`
- `energy=GaussianEnergy()`
- `hopfield_strength=None` — learnable if None, fixed if float
- `use_bias=True`
- `enforce_symmetry=True`
- `zero_diagonal=False`
- `weight_init=ZerosInitializer()` — W starts at zero for clean passthrough at init
- `latent_init=NormalInitializer()`

## Gradient Flow Analysis

**Inference** (`forward_inference`, argnums=1 → inputs):
- `d(total_energy)/d(probe)` flows through the identity addition `pre_act = probe + ...` → nonzero gradient
- This gradient is accumulated to the source node's `latent_grad` in `forward_value_and_grad()`
- Source node's z_latent updates during inference

**Learning** (`forward_learning`, argnums=0 → params):
- `d(total_energy)/d(W)` captures contributions from:
  - Recurrence path: `strength * (z_latent @ W)` contributes to z_mu → E_pc → total_energy
  - Hopfield energy: `accumulate_hopfield_energy_and_grad` adds E_hop which also depends on W through `total_energy = sum(state.energy)`
  - Note: probe does NOT contribute to W gradient (no W multiplication on probe)
- `d(total_energy)/d(strength)` if learnable
- `d(total_energy)/d(bias)`

**Latent self-gradient** (manual, in `accumulate_hopfield_energy_and_grad`):
- `(strength / D) * (wz @ W - wz)` is the Hopfield contribution to the node's own `latent_grad`
- Combined with the PC latent grad from `energy_functional()`, this drives z_latent during inference
- Not double-counted with autodiff (autodiff operates on different argnums)

**Inference attractor dynamics:**
- Each inference step: z_latent updated by gradient descent → new z_latent feeds back into `z_latent @ W` next step
- The probe term acts as a persistent external drive (probe = source node's z_latent, which also evolves)
- Over many steps, z_latent converges to the nearest attractor of the combined PC + Hopfield energy landscape

## Files

| File | Change |
|------|--------|
| `fabricpc/nodes/storkey_hopfield.py` | Full node implementation |
| `fabricpc/nodes/__init__.py` | `StorkeyHopfield` import and `__all__` entry |
| `fabricpc/nodes/base.py` | Removed TODO comment on line 283 |
| `tests/test_storkey_hopfield.py` | 15 tests (shapes, energy, gradients, symmetry, config, integration) |
| `examples/storkey_hopfield_demo.py` | MNIST A/B experiment: Hopfield vs MLP baseline |

No changes to energy.py, types.py, inference.py, or train.py.

## Demo

A/B experiment comparing a Hopfield-augmented network against an MLP baseline on real-valued MNIST.

**Arm A (Hopfield)** — 4-node graph:
```
IdentityNode(784) → Linear(128, sigmoid) → StorkeyHopfield(128, tanh) → Linear(10, softmax, CE)
```

**Arm B (MLP Baseline)** — 4-node graph:
```
IdentityNode(784) → Linear(128, sigmoid) → Linear(128, sigmoid) → Linear(10, softmax, CE)
```

Uses `ABExperiment` with `ExperimentArm` for each model, `MnistLoader` with `tensor_format="flat"`, shared hyperparams: `optimizer=optax.adamw(0.001, weight_decay=0.1)`, `InferenceSGD(eta_infer=0.05, infer_steps=20)`, `batch_size=200`.

## Implementation Status

| Step | Status |
|------|--------|
| `fabricpc/nodes/storkey_hopfield.py` — full StorkeyHopfield class | Done |
| `fabricpc/nodes/base.py` — removed TODO comment | Done |
| `fabricpc/nodes/__init__.py` — added export | Done |
| `tests/test_storkey_hopfield.py` — 15 tests | Done |
| `examples/storkey_hopfield_demo.py` — A/B MNIST demo | Done |
| Full test suite — 96/96 passing | Done |
