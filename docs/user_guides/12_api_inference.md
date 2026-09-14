# Inference Algorithms API

All inference algorithms extend `InferenceBase` from `fabricpc.core.inference`.

## Overview

Inference is the inner optimization loop of predictive coding. Given fixed weights and clamped data, it iteratively updates latent states to minimize total network energy.

Each inference step has three phases:
1. **Zero gradients** — Reset accumulated latent gradients
2. **Forward pass** — Compute predictions, errors, and accumulate gradient contributions
3. **Update** — Apply the algorithm-specific update rule to the relaxed variables (`z_latent` for the state-based solvers, ε for `EPCInference`)

`run_inference` brackets the step loop with two segment hooks, `begin_segment` and `finalize_state` (identity by default) — a solver's entry adaptation and exit rebuild when its per-step state is not the consumable final state.

## InferenceSGD

Standard gradient descent inference: `z -= eta * grad`

```python
from fabricpc.core.inference import InferenceSGD

inference = InferenceSGD(eta_infer=0.05, infer_steps=20, latent_decay=0.0)
structure = graph(..., inference=inference)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `0.1` | Inference rate |
| `infer_steps` | `int` | `20` | Number of inference iterations |
| `latent_decay` | `float` | `0.0` | Weight decay on latent states |

**Update rule:**
```
z_new = z * (1 - eta * latent_decay) - eta * latent_grad
```

## InferenceSGDNormClip

SGD inference with per-node gradient norm clipping.

```python
from fabricpc.core.inference import InferenceSGDNormClip

inference = InferenceSGDNormClip(
    eta_infer=0.1, infer_steps=20,
    max_norm=1.0, latent_decay=0.0, eps=1e-8,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `0.1` | Inference rate |
| `infer_steps` | `int` | `20` | Number of inference iterations |
| `latent_decay` | `float` | `0.0` | Weight decay on latent states |
| `max_norm` | `float` | `1.0` | Maximum L2 norm per node per sample |
| `eps` | `float` | `1e-8` | Numerical stability constant |

**Update rule:**
```
grad_norm = ||latent_grad||_2  (per sample)
clip_factor = min(1.0, max_norm / (grad_norm + eps))
clipped_grad = latent_grad * clip_factor
z_new = z * (1 - eta * latent_decay) - eta * clipped_grad
```

## EPCInference

Error-parameterized predictive coding (ePC, Goemaere et al., arXiv 2505.20137). The prediction error ε is the first-class relaxed variable; each latent is derived by a forward pass along `structure.schedule` as `z_latent = z_mu + ε`. Because every node's `z_mu` depends on all upstream latents, one `jax.value_and_grad` over the ε pytree per step delivers the output-loss signal to every layer unattenuated — a few steps replace sPC's hundreds on deep DAGs. The ε ↔ z_latent map is a volume-preserving bijection: identical energies, identical equilibria, and the final derived state feeds the local weight-gradient path unchanged. (The equivalence relies on the node contract's rule that `predict` never reads `state.z_latent` values — see the custom-nodes guide.)

Symbols: η is `eta_infer`; T is `infer_steps`; H_ε is the Hessian of the total energy in error coordinates; λ_max and λ_min are its largest and smallest excited eigenvalues; g₀ = ∇_ε E is the starting gradient; f̄ is the gradient-weighted relaxed fraction after T steps.

**Backprop regime.** One ePC step from ε = 0 leaves ε_t = −η·∂L/∂z_t exactly, the backprop activation gradient at the feedforward point. The local weight gradients are then taken at the re-derived latents, so they match backprop's to first order in η·λ_max: hidden layers scaled by η, the output layer unscaled (Goemaere et al., Theorem C.9). A layer fed only by clamped nodes matches exactly. Whether a run of T steps stays in this regime is read from the spectrum, below.

**Relaxed fraction and bands.** Gradient descent on the energy in error coordinates splits into independent modes along the eigenvectors of H_ε, and only the modes along which g₀ has a component ever move: the excited modes. After T steps a mode with eigenvalue λ has relaxed toward equilibrium by f(λ) = 1 − (1 − ηλ)^T. The regime is read on the modes that carry the gradient, each weighted by the fraction w of ‖g₀‖² it carries: f̄ = Σ w·f(λ) over the positive-curvature modes. `Regime.band` is `"backprop-like"` below f̄ = 0.1, `"near PC equilibrium"` above 0.9, `"partially relaxed"` between, and `"no positive curvature"` when no positive mode carries gradient weight. The equilibrium needs η·T·λ ≳ 3 on the modes that carry the gradient; on a linear graph they are the eig(S) modes of Innocenti et al.'s Theorem 1 (S = I + JJᵀ, J the map from the hidden errors to the output prediction), so the slowest of them sets T.

**Stability bound.** η·λ_max < 2 is required at every T; above it the top mode's distance from equilibrium grows every step (`Regime.unstable`).

**Odd-T reversal.** At odd T the output layer is damaged before the stability bound: its weight gradient follows the output residual after T steps, which along the top mode is (1 − η·(λ_max − 1))·r at T = 1 and reverses sign once η·(λ_max − 1) > 1 (for general odd T, once (1 − ηλ_max)^T < −1/(λ_max − 1)). The reversal is possible only at odd T, since (1 − ηλ)^T ≥ 0 at even T; the hidden errors keep their sign for every ηλ < 2 (`Regime.output_gradient_reverses`).

`fabricpc.core.epsilon_spectrum.epsilon_spectrum(params, state, clamps, structure, iters=30, key=None)` measures the excited spectrum on any graph (Lanczos on Hessian-vector products through `EPCInference.error_energy`, `iters` steps, three vectors of ε size in memory): `lambda_max`, `lambda_min`, the Ritz values with their gradient weights, `negative_weight` (the gradient weight on negative-curvature modes, nonzero on nonlinear graphs once the weights are large), and the Ritz residuals. `EPCInference.regime(spectrum)` returns a `Regime` (a `NamedTuple`) with `eta`, `steps`, `eta_lambda_max`, `eta_T_lambda_max`, `unstable`, `output_gradient_reverses`, `f_max` (f(λ_max)), `f_weighted` (f̄), `band`, `negative_weight`, `growth_min` (the T-step growth of the most negative mode, (1 + η·|λ_min|)^T), `lambda_max`, and `lambda_min`. `str(regime)` is a one-line label with precedence: `unstable`, then `growth_min > 1.1` (indefinite), then the band with f̄, f_max, and the reversal note. The linear oracle in `fabricpc.utils.linear_pc_oracle` gives the same spectrum exactly on linear-Gaussian graphs.

Two optimizer caveats. Under Adam the eta_infer scaling of the hidden-layer gradients is normalized away while eta_infer·|g| ≫ Adam's ε (1e-8), so 1-step ePC with Adam trains as backprop with Adam; below that the ε term damps the hidden layers. Without Adam the hidden layers learn eta_infer times slower than the output layer, a 1000× disparity at the default.

**Weight scale and growth.** λ_max is a weight-scale quantity: on a chain λ_max = 1 + σ_max(J)² grows with the product of the downstream weights' actions, so it grows during training and a fixed η can cross either threshold late in a run. The report `docs/reports/epc_regime_and_stability_report.md` measured this on the muPC ResNet-18 demo (`examples/resnet18_cifar10_demo.py`). λ_max at init sets the bound at init; the library defaults were backprop-like at init and collapsed over 100 epochs; η = 1e-2 collapsed over 100 epochs at every step count run while training at 2 epochs; only the smallest η·T survived the full schedule (Section 5.8). λ_max grew by orders of magnitude while every convolution weight's Frobenius norm fell, so weight-norm control did not bound it; the runaway occurred only in the ePC cells with the larger η·T while the backprop trainer and ePC at T = 1 drifted slowly on the same graph, and the mechanism is not established (Section 5.9). Goemaere et al.'s ResNet-18 trained at (1e-3, 5) without instability and differs from the demo in normalization, activation, weight decay, and parameterization (Section 5.11).

A fixed η therefore has no lasting margin (Section 6.2). `fabricpc.training.RegimeProbe` is the detector: a `train` callback that records the spectrum, the regime flags, and the Frobenius norm of every weight every N updates on a fixed probe batch (or the training batch), plus the test accuracy per epoch, to a CSV that `scripts/epc_analysis.py --plot_track` renders. Supplying `iter_callback` forces a device sync on every batch, probed or not. Under `InferenceSchedule`, `structure.config["inference"]` is the schedule, so pass the ePC segment as `RegimeProbe(..., inference=epc)`; the regime then describes that segment's relaxation. The remedies when a flag fires, and their trade-offs, are in [Training with ePC](17_training_with_epc.md#step-7-act-on-the-flags).

```python
from fabricpc.core import EPCInference, epsilon_spectrum
from fabricpc.training import RegimeProbe

# Measure the spectrum at init, tabulate the regime for an (eta, T) grid, and
# attach the probe: docs/user_guides/17_training_with_epc.md
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `1e-3`  | Inference rate on ε: one global rate through the whole network, bounded by 2/λ_max of the graph. The default has no regime meaning independent of the graph (it was backprop-like at init on the ResNet-18 demo, and that run collapsed); set it from the measured spectrum |
| `infer_steps` | `int` | `5`     | Number of inference iterations; with `eta_infer` it sets f̄ and the band. Set it from `regime` on the measured spectrum |
| `latent_decay` | `float` | `0.0`   | Weight decay on the relaxed errors |

**Update rule (per step):**
```
derive z_latent = z_mu + error along structure.schedule (clamped nodes keep the clamp, derive error)
latent_grad = d(total energy of in_degree > 0 nodes)/d(error)   # one global reverse pass
error_new = error * (1 - eta * latent_decay) - eta * latent_grad
```

A segment starts with `begin_segment` — one forward pass at the carried latents setting ε := z_latent − z_mu, so relaxation continues exactly from the incoming state (the initializer's output or a previous segment's latents) — and ends with `finalize_state`, one detached derive so the returned state satisfies z_latent = z_mu + ε with energies at the final point.

The ε gradient is taken through the full network's transfer function — a change in one node's ε moves every downstream derived latent — so `eta_infer` must be tuned like a weight learning rate, not like sPC's local per-node rate.

On cyclic graphs, ePC minimizes the unrolled approximation of the graph energy fixed by `graph(..., unroll=U)`; state-based solvers minimize the exact graph energy as-is. Memory: each ePC step's single reverse pass stores activations for the whole derived forward (depth × unroll), backprop-scale rather than sPC's per-node closures.

## InferenceSchedule

Composes solvers as segments per weight update — e.g. a few cheap global ePC steps to near-equilibrium, then sPC refinement on the true arbitrary-graph energy, warm-started from ePC's solution:

```python
from fabricpc.core.inference import InferenceSGD, InferenceSchedule
from fabricpc.core.inference_epc import EPCInference

inference = InferenceSchedule(
    EPCInference(eta_infer=1e-3, infer_steps=2),
    InferenceSGD(eta_infer=0.05, infer_steps=20),
)
```

Chained execution contract:
1. Node states are initialized once, by the graph's configured initializer, before the first segment; no segment re-initializes.
2. Each solver receives `z_latent` exactly as the previous segment (or the initializer) left it, and its `begin_segment` adapts the derived fields to its own parameterization without moving the latents — ePC recomputes ε := z_latent − z_mu at the carried latents, so relaxation continues from the incoming latents rather than from stale ε.
3. The next solver continues from the resulting state (after e.g. ePC's final derive rebuild).

Schedules nest, and `segments()` flattens them for per-step consumers (tracking iterates segments instead of assuming one global step count). A schedule has no single per-step rule, so `inference_step()` and `compute_new_latent()` raise. Under a composed schedule, a tracked `latent_grad_norm` series carries each segment's own gradient semantics — sPC's one-hop dE/dz_latent, ePC's full-forward ε gradient.

A few ePC steps then sPC steps is also a stability remedy: the ePC segment runs at a low η·T as a warm start that carries the loss signal to every layer, and the sPC segment finishes the settle on the exact graph energy at the per-node rate. `RegimeProbe` needs the ePC segment as `inference=`. [Training with ePC](17_training_with_epc.md#composing-epc-and-spc) shows how to measure what the warm start saves on your graph.

## Tuning Guidance

| Parameter | Typical Range | Notes |
|-----------|:-------------:|-------|
| `eta_infer` (state-based) |   0.01–0.2    | A per-node rate; lower for stability, higher for faster convergence |
| `eta_infer` (EPCInference) | below 2/λ_max at every T, below 1/(λ_max − 1) at odd T | One global rate; λ_max is measured with `epsilon_spectrum` at init and tracked with `RegimeProbe` because it grows during training. See [Training with ePC](17_training_with_epc.md) |
| `infer_steps` (state-based) |  ~5 * depth   | More steps = better convergence, slower training |
| `infer_steps` (EPCInference) | set by f̄ from `EPCInference.regime` | f̄ < 0.1 is backprop-like, f̄ > 0.9 the PC equilibrium (η·T·λ ≳ 3 on the modes that carry the gradient); the output-gradient reversal is possible only at odd T. See [Training with ePC](17_training_with_epc.md) |
| `latent_decay` |      0.0      | Rarely needed; try 0.001 if latents drift |
| `max_norm` |    0.5–2.0    | For InferenceSGDNormClip; prevents gradient explosions |

For deep networks (>10 layers) under the state-based solvers, consider:
- Increasing `infer_steps` to `max(20, 5 * num_layers)`
- Using `InferenceSGDNormClip` for stability
- Switching to `EPCInference`, whose step count does not grow with depth ([Training with ePC](17_training_with_epc.md))

## Creating Custom Inference Algorithms

Subclass `InferenceBase` and implement `compute_new_latent()`:

```python
from fabricpc.core.inference import InferenceBase
import jax.numpy as jnp

class InferenceMomentum(InferenceBase):
    def __init__(self, eta_infer=0.1, infer_steps=20, momentum=0.9):
        super().__init__(eta_infer=eta_infer, infer_steps=infer_steps, momentum=momentum)

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        eta = config["eta_infer"]
        momentum = config["momentum"]
        # Your custom update rule here
        # Example: add momentum tracking via node_state auxiliary fields
        return node_state.z_latent - eta * node_state.latent_grad
```

For more radical changes, override `inference_step()`, `forward_value_and_grad()`, or `run_inference()`. Segment-aware solvers additionally override the boundary hooks `begin_segment()` (entry adaptation, run before the first step) and `finalize_state()` (exit rebuild, run after the last step) — see `EPCInference` for a worked example of both.

## Convenience Function

```python
from fabricpc.core.inference import run_inference

# Run inference using the algorithm stored in structure.config["inference"]
final_state = run_inference(params, initial_state, clamps, structure)
```

This is a convenience wrapper that extracts the inference object from the graph structure and delegates to its `run_inference()` method.
