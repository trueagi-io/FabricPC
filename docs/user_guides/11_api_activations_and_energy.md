# Activations and Energy Functionals API

## Activation Functions

All activations extend `ActivationBase` from `fabricpc.core.activations`.

### Summary Table

| Activation | Formula | variance_gain | jacobian_gain | Typical Use |
|------------|---------|:------------:|:-------------:|-------------|
| `IdentityActivation` | `f(x) = x` | 1.0 | 1.0 | Input nodes, regression |
| `SigmoidActivation` | `f(x) = 1/(1+exp(-x))` | 1.0 | 1.0 | Hidden layers (classic PC) |
| `TanhActivation` | `f(x) = tanh(x)` | sqrt(5/3) ≈ 1.291 | 1.261 | Hidden layers (centered) |
| `ReLUActivation` | `f(x) = max(0, x)` | sqrt(2) ≈ 1.414 | 1.0 | Hidden layers (modern) |
| `LeakyReLUActivation` | `f(x) = max(alpha*x, x)` | sqrt(2/(1+alpha^2)) | 1.0 | Hidden layers |
| `GeluActivation` | `f(x) = x * Phi(x)` | sqrt(2) ≈ 1.414 | 1.168 | Transformer FFN |
| `SoftmaxActivation` | `f(x) = exp(x)/sum(exp(x))` | — | — | Classification output |
| `HardTanhActivation` | `f(x) = clip(x, min, max)` | sqrt(5/3) ≈ 1.291 | 1.035 | Bounded hidden layers |

### Individual Activations

**IdentityActivation** — `f(x) = x`. No parameters. Default activation if none specified.

```python
from fabricpc.core.activations import IdentityActivation

act = IdentityActivation()
```

**SigmoidActivation** — `f(x) = 1/(1+exp(-x))`. No parameters.

```python
from fabricpc.core.activations import SigmoidActivation

act = SigmoidActivation()
```

**TanhActivation** — `f(x) = tanh(x)`. No parameters. `variance_gain = sqrt(5/3)`, `jacobian_gain = 1.261`.

```python
from fabricpc.core.activations import TanhActivation

act = TanhActivation()
```

**ReLUActivation** — `f(x) = max(0, x)`. No parameters. `variance_gain = sqrt(2)`.

```python
from fabricpc.core.activations import ReLUActivation

act = ReLUActivation()
```

**LeakyReLUActivation** — `f(x) = max(alpha*x, x)`.

```python
from fabricpc.core.activations import LeakyReLUActivation

act = LeakyReLUActivation(alpha=0.01)  # default alpha
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.01` | Negative slope coefficient |

**GeluActivation** — `f(x) = x * 0.5 * (1 + erf(x/sqrt(2)))`. No parameters. `variance_gain = sqrt(2)`, `jacobian_gain = 1.168`.

```python
from fabricpc.core.activations import GeluActivation

act = GeluActivation()
```

**SoftmaxActivation** — `f(x) = exp(x) / sum(exp(x))` along last axis. No parameters. Use at classification output nodes.

```python
from fabricpc.core.activations import SoftmaxActivation

act = SoftmaxActivation()
```

**HardTanhActivation** — `f(x) = clip(x, min_val, max_val)`.

```python
from fabricpc.core.activations import HardTanhActivation

act = HardTanhActivation(min_val=-1.0, max_val=1.0)  # defaults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_val` | `float` | `-1.0` | Minimum output value |
| `max_val` | `float` | `1.0` | Maximum output value |

### Creating Custom Activations

```python
from fabricpc.core.activations import ActivationBase

class MyActivation(ActivationBase):
    def __init__(self, temperature=1.0):
        super().__init__(temperature=temperature)

    @staticmethod
    def forward(x, config=None):
        temp = config.get("temperature", 1.0) if config else 1.0
        return jnp.tanh(x / temp)

    @staticmethod
    def derivative(x, config=None):
        temp = config.get("temperature", 1.0) if config else 1.0
        t = jnp.tanh(x / temp)
        return (1 - t**2) / temp
```

Optional overrides: `variance_gain(config)` and `jacobian_gain(config)` for muPC compatibility.

---

## Energy Functionals

All energy functionals extend `EnergyFunctional` from `fabricpc.core.energy`.

### Summary Table

| Energy | Formula | grad_latent | Typical Use |
|--------|---------|-------------|-------------|
| `GaussianEnergy` | `(precision/2) * ||z - mu||^2` | `precision * (z - mu)` | General purpose (DEFAULT) |
| `BernoulliEnergy` | `-[z*log(mu) + (1-z)*log(1-mu)]` | `log((1-mu)/mu)` | Binary outputs |
| `CrossEntropyEnergy` | `-sum(z * log(mu))` | `-log(mu)` | Classification (with softmax) |
| `LaplacianEnergy` | `(1/b) * sum|z - mu|` | `(1/b) * sign(z - mu)` | Robust to outliers |
| `HuberEnergy` | Smooth L1 | `clip(z - mu, -delta, delta)` | Combines L2/L1 benefits |
| `KLDivergenceEnergy` | `sum(z * log(z/mu))` | `log(z) - log(mu) + 1` | Distribution matching |

### Individual Functionals

**GaussianEnergy** (default):

```python
from fabricpc.core.energy import GaussianEnergy

energy = GaussianEnergy(precision=1.0)  # precision = 1/sigma^2
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `precision` | `float` | `1.0` | Inverse variance (1/sigma^2) |

**BernoulliEnergy**:

```python
from fabricpc.core.energy import BernoulliEnergy

energy = BernoulliEnergy(eps=1e-7)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps` | `float` | `1e-7` | Numerical stability constant |

**CrossEntropyEnergy**:

```python
from fabricpc.core.energy import CrossEntropyEnergy

energy = CrossEntropyEnergy(eps=1e-7, axis=-1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps` | `float` | `1e-7` | Numerical stability constant |
| `axis` | `int` | `-1` | Axis for probability distributions |

**LaplacianEnergy**:

```python
from fabricpc.core.energy import LaplacianEnergy

energy = LaplacianEnergy(scale=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scale` | `float` | `1.0` | Scale parameter (b) |

**HuberEnergy**:

```python
from fabricpc.core.energy import HuberEnergy

energy = HuberEnergy(delta=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `delta` | `float` | `1.0` | Transition threshold between L2 and L1 |

**KLDivergenceEnergy**:

```python
from fabricpc.core.energy import KLDivergenceEnergy

energy = KLDivergenceEnergy(eps=1e-7, axis=-1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps` | `float` | `1e-7` | Numerical stability constant |
| `axis` | `int` | `-1` | Axis for probability distributions |

### Recommended Pairings

| Architecture | Hidden Layers | Output Layer |
|-------------|--------------|--------------|
| Classification | Any activation + `GaussianEnergy` | `SoftmaxActivation` + `CrossEntropyEnergy` |
| Regression | Any activation + `GaussianEnergy` | `IdentityActivation` + `GaussianEnergy` |
| Binary output | Any activation + `GaussianEnergy` | `SigmoidActivation` + `BernoulliEnergy` |

### Creating Custom Energy Functionals

```python
from fabricpc.core.energy import EnergyFunctional
import jax.numpy as jnp

class MyEnergy(EnergyFunctional):
    def __init__(self, temperature=1.0):
        super().__init__(temperature=temperature)

    @staticmethod
    def energy(z_latent, z_mu, config=None):
        temp = config.get("temperature", 1.0) if config else 1.0
        diff = z_latent - z_mu
        return 0.5 * jnp.sum(diff ** 2, axis=-1) / temp

    @staticmethod
    def grad_latent(z_latent, z_mu, config=None):
        temp = config.get("temperature", 1.0) if config else 1.0
        return (z_latent - z_mu) / temp
```
