# Initializers API

## Weight Initializers

All initializers extend `InitializerBase` from `fabricpc.core.initializers`.

| Initializer | Distribution | Parameters | When to Use |
|-------------|-------------|------------|-------------|
| `ZerosInitializer` | All zeros | — | Biases, initial states |
| `OnesInitializer` | All ones (scaled) | `gain=1.0` | Scaling factors |
| `NormalInitializer` | N(mean, std^2) | `mean=0.0`, `std=0.05`, `gain=1.0` | Direct control |
| `UniformInitializer` | U(min, max) | `min_val=-0.1`, `max_val=0.1` | Bounded initialization |
| `XavierInitializer` | Glorot | `distribution="normal"`, `gain=1.0` | Sigmoid/tanh networks |
| `KaimingInitializer` | He | `mode="fan_in"`, `nonlinearity="relu"`, `distribution="normal"`, `a=0.01`, `gain=1.0` | ReLU networks |
| `MuPCInitializer` | N(0, gain^2) | `gain=1.0` | Use with muPC scaling |

### Usage

```python
from fabricpc.core.initializers import XavierInitializer, MuPCInitializer

# Xavier for sigmoid/tanh networks
node = Linear(shape=(256,), name="h1", weight_init=XavierInitializer())

# Kaiming for ReLU networks
node = Linear(shape=(256,), name="h1", weight_init=KaimingInitializer())

# MuPC (recommended for deep networks)
node = Linear(shape=(256,), name="h1", weight_init=MuPCInitializer())
```

### XavierInitializer

```python
XavierInitializer(distribution="normal", gain=1.0)
```
- `"normal"`: `std = gain * sqrt(2 / (fan_in + fan_out))`
- `"uniform"`: `U(-limit, limit)` where `limit = gain * sqrt(6 / (fan_in + fan_out))`

### KaimingInitializer

```python
KaimingInitializer(mode="fan_in", nonlinearity="relu", distribution="normal", a=0.01, gain=1.0)
```
- `mode`: `"fan_in"` (default) or `"fan_out"`
- `nonlinearity`: `"relu"` (gain=sqrt(2)) or `"leaky_relu"` (gain=sqrt(2/(1+a^2)))

### MuPCInitializer

```python
MuPCInitializer(gain=1.0)
```
Draws weights from `N(0, gain^2)`. Forward-pass scaling is handled separately by `MuPCConfig`.

### Creating Custom Initializers

```python
class MyInitializer(InitializerBase):
    def __init__(self, gain=1.0):
        super().__init__(gain=gain)

    @staticmethod
    def initialize(key, shape, config=None):
        config = config or {}
        gain = config.get("gain", 1.0)
        return gain * jax.random.normal(key, shape)
```

---

## State Initializers

State initializers set the initial latent states for all nodes before inference begins.

All state initializers extend `StateInitBase` from `fabricpc.graph.state_initializer`.

### FeedforwardStateInit (Default)

Runs a forward pass through the network in topological order and sets `z_latent = z_mu`. Provides fast convergence for feedforward DAGs.

```python
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit

structure = graph(..., state_init=FeedforwardStateInit())
```

Requires `params` (used during forward pass).

### GlobalStateInit

All nodes use the graph-level initializer (typically `NormalInitializer`).

```python
from fabricpc.graph_initialization.state_initializer import GlobalStateInit

structure = graph(..., state_init=GlobalStateInit())
```

### NodeDistributionStateInit

Each node uses its own `latent_init` initializer.

```python
from fabricpc.graph_initialization.state_initializer import NodeDistributionStateInit

structure = graph(..., state_init=NodeDistributionStateInit())
```
