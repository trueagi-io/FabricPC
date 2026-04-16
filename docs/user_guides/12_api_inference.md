# Inference Algorithms API

All inference algorithms extend `InferenceBase` from `fabricpc.core.inference`.

## Overview

Inference is the inner optimization loop of predictive coding. Given fixed weights and clamped data, it iteratively updates latent states to minimize total network energy.

Each inference step has three phases:
1. **Zero gradients** — Reset accumulated latent gradients
2. **Forward pass** — Compute predictions, errors, and accumulate gradient contributions
3. **Latent update** — Apply the algorithm-specific update rule to z_latent

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

## Tuning Guidance

| Parameter | Typical Range | Notes |
|-----------|:------------:|-------|
| `eta_infer` | 0.01–0.2 | Lower for stability, higher for faster convergence |
| `infer_steps` | 10–50 | More steps = better convergence, slower training |
| `latent_decay` | 0.0 | Rarely needed; try 0.001 if latents drift |
| `max_norm` | 0.5–2.0 | For InferenceSGDNormClip; prevents gradient explosions |

For deep networks (>10 layers), consider:
- Increasing `infer_steps` to `max(20, 4 * num_layers)`
- Using `InferenceSGDNormClip` for stability

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

For more radical changes, override `inference_step()`, `forward_value_and_grad()`, or `run_inference()`.

## Convenience Function

```python
from fabricpc.core.inference import run_inference

# Run inference using the algorithm stored in structure.config["inference"]
final_state = run_inference(params, initial_state, clamps, structure)
```

This is a convenience wrapper that extracts the inference object from the graph structure and delegates to its `run_inference()` method.
