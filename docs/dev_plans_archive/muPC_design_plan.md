# muPC: Weight Initialization Algorithm - Research Analysis & Implementation Plan (completed 2024-04-09)

## Part 1: Research Analysis of muPC (from arXiv:2505.13124)

### Paper Overview

**Title:** muPC: Scaling Predictive Coding to 100+ Layer Networks
**Authors:** Francesco Innocenti, El Mehdi Achour, Christopher L. Buckley
**Reference code:** https://github.com/thebuckleylab/jpc

### The Problem

Standard Predictive Coding networks (PCNs) struggle to scale beyond ~10 layers. The root cause is the same pathology that plagued early deep backprop networks: **signals either explode or vanish** as they propagate through many layers. In standard parameterization (SP), weights are initialized with variance `1/fan_in` (LeCun-style) and the forward pass uses no additional scaling -- the raw `W * phi(z)` output is used directly. This works fine for shallow nets but becomes unstable at depth.

### The Core Insight: Separate Initialization from Scaling

The key innovation of muPC (borrowed from Depth-muP by Yang et al.) is to **decouple the weight initialization from the forward-pass scaling**. Instead of baking the scaling into the initial weight values (like Xavier/He do), muPC:

1. **Initializes weights from a standard normal** `W ~ N(0, 1)` -- no fan-in scaling
2. **Applies explicit multipliers** `a_l` at each layer during the forward pass: `z_l = a_l * W_l * phi(z_{l-1})`

This separation means the scaling factors can encode information about **both width AND depth**, which traditional initializers cannot do.

---

### Step-by-Step: How muPC Weight Init Works

Using a concrete example: **MNIST classification** with a 30-layer ResNet, width=128, input_dim=784, output_dim=10.

#### Step 1: Weight Initialization (All Layers)

Every weight matrix is drawn from a **standard normal distribution**:

```
W_l ~ N(0, 1)    for all layers l = 1, ..., L
```

In the JPC code (`make_mlp` in `jpc/_utils.py`):
```python
if param_type == "mupc":
    W = jax.random.normal(subkeys[i], linear.weight.shape)
    linear = tree_at(lambda l: l.weight, linear, W)
```

This replaces Equinox's default initialization (which uses `1/sqrt(fan_in)` variance). The weights are intentionally "too large" -- the scaling factors in the next step compensate.

#### Step 2: Compute Layer-Specific Scaling Factors

Three distinct scalings are computed based on network architecture (from `_get_param_scalings` in `jpc/_core/_energies.py`):

| Layer | Scaling Factor | Formula | Example Value (our net) |
|-------|---------------|---------|------------------------|
| Input (l=1) | `a_1` | `1 / sqrt(N_0)` | `1/sqrt(784) = 0.0357` |
| Hidden (l=2..L-1) | `a_l` | `1 / sqrt(N * L)` (with skip) or `1 / sqrt(N)` (no skip) | `1/sqrt(128*30) = 0.0161` |
| Output (l=L) | `a_L` | `1 / N` | `1/128 = 0.0078` |

Where:
- `N_0` = input dimension (784)
- `N` = hidden width (128)
- `L` = total depth (30)

In code:
```python
D = input.shape[1]                    # N_0 = 784
N = model[0][1].weight.shape[0]       # width = 128

a1 = 1 / sqrt(D)                                    # input
al = 1 / sqrt(N) if no_skips else 1 / sqrt(N * L)   # hidden
aL = 1 / N                                          # output (muPC specific!)
```

#### Step 3: Apply Scalings During Forward Pass

During the feedforward pass (used to initialize activities), each layer's output is **multiplied by its scaling factor**:

```
z_1 = a_1 * W_1 * x                          # input layer
z_l = a_l * W_l * phi(z_{l-1}) + z_{l-1}     # hidden (with residual skip)
z_L = a_L * W_L * phi(z_{L-1})               # output layer
```

In code (`init_activities_with_ffwd` in `jpc/_core/_init.py`):
```python
z1 = scalings[0] * vmap(model[0])(input)
activities = [z1]
for l in range(1, L):
    zl = scalings[l] * vmap(model[l])(activities[l-1])
    if skip_model[l] is not None:
        zl += vmap(skip_model[l])(activities[l-1])  # + identity skip
    activities.append(zl)
```

#### Step 4: Energy Computation Uses Same Scalings

The PC energy function (prediction errors) also incorporates these scalings:

```
F = sum_l (1/2) ||z_l - a_l * W_l * phi(z_{l-1}) - tau_l * z_{l-1}||^2
```

where `tau_l` is the residual connection (identity skip). The scaling `a_l` appears inside the prediction, ensuring error signals are properly scaled.

#### Step 5: Gradient Computation is Automatic

The gradients for both activity updates and parameter updates are computed via JAX autodiff on the scaled energy function. There is no separate scaling logic for gradients -- the scalings embedded in the energy function propagate naturally through `jax.grad`.

---

### Why Each Scaling Factor Works

**Input layer `a_1 = 1/sqrt(N_0)`**: A weight matrix `W` of shape `(N, N_0)` multiplied by input `x` of dimension `N_0` produces a sum of `N_0` terms. Each term is `O(1)` (weight is N(0,1), input is normalized). The sum is `O(sqrt(N_0))` by CLT, so dividing by `sqrt(N_0)` keeps the output `O(1)`.

**Hidden layers `a_l = 1/sqrt(N*L)`**: Same `1/sqrt(N)` logic for width, but the `1/sqrt(L)` factor is critical for depth. In a ResNet with skip connections `z_l = a_l * f(z_{l-1}) + z_{l-1}`, the activations accumulate across L layers. Without the `1/sqrt(L)` damping, after L residual additions the norm grows as `O(sqrt(L))`. The `1/sqrt(L)` factor keeps the total contribution from all residual branches `O(1)` regardless of depth.

**Output layer `a_L = 1/N`**: This is the distinguishing feature of muPC vs. NTP (neural tangent parameterization). NTP uses `1/sqrt(N)` here. muPC uses the stronger `1/N` scaling, which ensures that the **output logits remain O(1)** and, crucially, that **parameter updates at the output layer have the correct magnitude** -- they produce `O(1/N)` changes to the output (a "maximal update" in muP terminology), enabling hyperparameter transfer.

---

### Comparison: Standard PC vs muPC

| Aspect | Standard PC (SP) | muPC |
|--------|------------------|------|
| Weight init variance | `1/fan_in` (LeCun) | `1` (standard normal) |
| Forward scaling | `1` (none) | Layer-dependent `a_l` |
| Input scaling | Baked into weights | `1/sqrt(input_dim)` |
| Hidden scaling | Baked into weights | `1/sqrt(width * depth)` |
| Output scaling | Baked into weights | `1/width` |
| Max trainable depth | ~10 layers | 100+ layers |
| LR transfer across width/depth | No | Yes (zero-shot) |
| Requires skip connections | Yes (for depth) | Yes (integral to scaling) |

---

### The Skip Connection Model

muPC requires residual (identity) skip connections at hidden layers for the depth scaling to work. The `make_skip_model` function creates these:

```python
def make_skip_model(depth):
    skips = [None] * depth          # None = no skip
    for l in range(1, depth-1):     # skips at hidden layers only
        skips[l] = nn.Lambda(nn.Identity())  # identity function
    return skips
```

No skip at input layer (l=0) and no skip at output layer (l=L-1). Hidden layers get identity skips, creating the classic `z_l = a_l * f(z_{l-1}) + z_{l-1}` ResNet structure.

---

### Two Implementation Approaches in JPC

The JPC codebase offers two equivalent ways to use muPC:

**Approach A: Library-managed scaling (`param_type="mupc"`)**
- Use `jpc.make_mlp(param_type="mupc")` -- weights initialized N(0,1), stored unscaled
- Use `jpc.make_skip_model(depth)` -- creates identity skip connections
- Pass `param_type="mupc"` to all training functions
- The library internally calls `_get_param_scalings()` and multiplies outputs at each layer
- Scalings are recomputed from model structure each call

**Approach B: User-managed scaling (custom `FCResNet`)**
- Build a custom `FCResNet` with `param_type="mupc"`
- Scalings baked into `ScaledLinear` layers at construction time
- Skip connections built directly into `ResNetBlock.__call__` as `return self.scaled_linear(x) + res_path`
- Pass `param_type="sp"` to training functions (scaling already applied by model)
- More explicit/transparent; no skip_model needed separately

---

### End-to-End Walkthrough

For a 30-layer, width-128 network on MNIST:

1. **Create weights**: 30 weight matrices, all drawn from `N(0,1)` -- a 784x128 input matrix, 28 hidden 128x128 matrices, and a 128x10 output matrix.

2. **Compute scalings**: `[0.0357, 0.0161, 0.0161, ...(28 times)..., 0.0078]`

3. **Forward pass** (activity init): Input image `x` (784-dim) flows through, each layer applying `a_l * W_l * relu(z_{l-1}) + z_{l-1}`. The output is a 10-dim vector of logits.

4. **Inference phase** (T=30 steps): Activities are updated by gradient descent on the energy function, with the same scalings embedded in the energy. This is the "predictive coding" part -- layers adjust their activities to minimize prediction errors.

5. **Learning phase**: Parameters (weights) are updated by gradient descent on the same energy, evaluated at the converged activities. The scalings ensure gradients are properly sized for stable learning.

6. **Key benefit**: The learning rates `activity_lr=0.5, param_lr=0.1` work for this 30-layer net AND would work for a 128-layer net of width 256 without retuning.

---

## Part 1b: Fan-In/Fan-Out Asymmetry Analysis & Three-Way Decomposition

### The Problem muPC Leaves Open

Standard initializers (Xavier/Glorot) balance both signal directions by using `Var(W) = 2/(fan_in + fan_out)`. muPC only addresses the **forward** direction: the scaling `a_l = 1/sqrt(fan_in)` normalizes the forward sum, but the backward direction through `W^T` is not independently controlled.

In Predictive Coding, the gradient on activity `z_{l-1}` has two components from adjacent energy terms:

```
dF/dz_{l-1} = ε_{l-1}  -  a_l * W_l^T * diag(phi'(z_{l-1})) * ε_l  +  skip terms
               ↑                        ↑
          "self-gradient"         "top-down gradient"
          (prediction error       (backpropagated from
           at this node)          the postsynaptic node)
```

Where `ε_l = z_l - a_l * W_l * phi(z_{l-1}) - tau * z_{l-1}` is the prediction error.

### Variance Analysis of Each Gradient Term

**Self-gradient `ε_{l-1}`**: If muPC works correctly, this is O(1) per component.

**Top-down gradient `a_l * W_l^T * ε_l`**:
- `W_l` has shape `(fan_out, fan_in)` = `(N_l, N_{l-1})`
- `W_l^T * ε_l` sums over `N_l` entries (the **fan_out**) → variance = `N_l`
- Multiplied by `a_l = 1/sqrt(N_{l-1} * L)`: variance = `N_l / (N_{l-1} * L)`

**For square hidden layers** (`N_l = N_{l-1} = N`): variance = `1/L`, magnitude O(1/sqrt(L)).

**For non-square layers**, the ratio `N_l / N_{l-1}` creates an imbalance:

| Transition | fan_in | fan_out | Top-down var | Magnitude |
|-----------|--------|---------|-------------|-----------|
| Input→Hidden (784→128) | 784 | 128 | 128/(784×30) = 0.005 | 0.074 |
| Hidden→Hidden (128→128) | 128 | 128 | 128/(128×30) = 0.033 | 0.182 |
| Hidden→Output (128→10)* | 128 | 10 | (1/128)²×10 = 0.0006 | 0.025 |

*Output uses `a_L = 1/N = 1/128`

**Key finding**: The self-gradient is O(1) while the top-down gradient ranges from O(0.025) to O(0.18). The self-gradient dominates at every layer. The penultimate hidden layer gets almost no backward signal from the output layer.

### Weight Gradient Has Different Scaling Needs

```
dF/dW_l = a_l * ε_l ⊗ phi(z_{l-1})^T    (outer product)
```

Per-element variance: `a_l^2 * Var(ε_l) * Var(phi(z_{l-1})) ≈ 1/(fan_in × L)`

This is **independent of fan_out** — the weight gradient and the presynaptic latent gradient have fundamentally different scaling dependencies.

### Three-Way Decomposition Approach

To independently control all three gradient pathways, introduce a **precision weighting** `c_l` on each layer's energy, separate from the forward scaling `a_l`:

```
E_l = (c_l / 2) * ||z_l - a_l * W_l * phi(z_{l-1}) - tau * z_{l-1}||^2
```

The three gradients become:

| Gradient | Formula | Controlled By |
|----------|---------|---------------|
| Self (on z_l) | `c_l * ε_l` | `c_l` |
| Top-down (on z_{l-1}) | `c_l * a_l * W_l^T * phi'(z_{l-1}) * ε_l` | `c_l × a_l` |
| Weight (on W_l) | `c_l * a_l * ε_l ⊗ phi(z_{l-1})^T` | `c_l × a_l` |

**Setting `a_l` for forward pass** (variance of prediction):
```
a_l = 1 / sqrt(fan_in × L)     [same as muPC]
```

**Setting `c_l` for backward pass** (unit variance of top-down gradient on z_{l-1}):
We want `c_l * a_l * sqrt(fan_out) = O(1)`:
```
c_l = 1 / (a_l * sqrt(fan_out)) = sqrt(fan_in × L / fan_out)
```

For square layers: `c_l = sqrt(L)`. For non-square: `c_l = sqrt(fan_in × L / fan_out)`.

This changes the self-gradient to `c_l * ε_l = O(sqrt(fan_in × L / fan_out))` and the weight gradient to `c_l * a_l = 1/sqrt(fan_out)` per element.

### Implications for Arbitrary Graph Networks

In a graph PC network (not sequential):
- A node may have **multiple presynaptic inputs** with different dimensions
- A node may have **multiple postsynaptic targets** with different dimensions
- Skip connections may follow arbitrary patterns (pyramidal circuits, etc.)
- The "depth" L is not a single number but a **path length** that varies per node

The three-way decomposition naturally extends to this:
- Each **edge** (connection) gets its own `a_l` based on the source node's width (fan_in of that edge)
- Each **energy term** (per-edge prediction error) gets its own `c_l` based on the fan_out of that edge
- The depth factor `L` could be replaced by a per-node **effective depth** (longest path from input, or graph eccentricity)

### Design Decision

**Scalings stored per-node at init time** rather than recomputed dynamically. Each node stores:
- `a_l` (forward scaling) per incoming edge
- `c_l` (energy precision) per incoming edge
- Both computed once from the graph structure during network construction

---

## Part 2: Implementation Plan for FabricPC

### Design Decisions (from discussion)

1. **Three-way decoupled scaling** is a requirement: forward scaling, self-gradient scaling, and top-down gradient scaling are independently controlled
2. **Forward scaling is highest priority** — it's what makes muPC work
3. **Scalings stored in NodeInfo** (static, computed at graph build time)
4. **Separate scaling mechanism** independent of energy functional type (works with Gaussian, CrossEntropy, Huber, etc.)
5. **Effective depth**: use shortest path from input; abstract to extensible class for path computation
6. **Scalings stored per-node** at init time, not recomputed dynamically

### Architecture Overview

The implementation adds **5 new components** and modifies **4 existing files**:

```
NEW FILES:
  fabricpc/core/mupc.py              - MuPCScaling dataclass + MuPCConfig + compute_mupc_scalings()
  fabricpc/core/depth_metric.py      - DepthMetricBase, ShortestPathDepth, LongestPathDepth, FixedDepth

MODIFIED FILES:
  fabricpc/core/initializers.py      - Add MuPCInitializer class
  fabricpc/core/types.py             - Add scaling_config field to NodeInfo
  fabricpc/builder/graph_builder.py  - Compute and attach scaling factors at build time
  fabricpc/nodes/base.py             - Apply scaling in forward_and_latent_grads, forward_and_weight_grads, energy_functional
                                       (NO changes to any node's forward() method!)
```

### Step 1: Depth Metric Classes — `fabricpc/core/depth_metric.py`

New file with extensible depth computation.

```python
class DepthMetricBase(ABC):
    """Computes effective depth for each node in an arbitrary graph."""
    @abstractmethod
    def compute(self, nodes, edges) -> Dict[str, int]:
        """Returns {node_name: effective_depth}."""

class ShortestPathDepth(DepthMetricBase):
    """Effective depth = shortest path from any terminal input node (BFS)."""

class LongestPathDepth(DepthMetricBase):
    """Effective depth = longest path from any terminal input node (DAG DP)."""

class FixedDepth(DepthMetricBase):
    """User-specified fixed depth for all nodes."""
    def __init__(self, depth: int): ...
```

Implementation: BFS from all terminal input nodes (in_degree=0), tracking min/max distance. The graph builder already does topological sort, so this integrates naturally.

### Step 2: MuPC Scaling Computation — `fabricpc/core/mupc.py`

New file with the core scaling logic.

```python
@dataclass(frozen=True)
class MuPCScaling:
    """Per-node scaling factors for muPC parameterization."""
    forward_scale: Dict[str, float]    # {edge_key: a_l} — per incoming edge
    self_grad_scale: float             # c_self — scales self-gradient (dE/dz_self)
    topdown_grad_scale: Dict[str, float]  # {edge_key: c_td} — scales top-down grad per edge
    weight_grad_scale: Dict[str, float]   # {edge_key: c_w} — scales weight grad per edge

def compute_mupc_scalings(
    nodes: Dict[str, NodeInfo],
    edges: Dict[str, EdgeInfo],
    depth_metric: DepthMetricBase,
    output_nodes: Set[str],       # nodes clamped to targets (from task_map)
) -> Dict[str, MuPCScaling]:
    """Compute per-node MuPCScalingFactors from graph topology."""
```

**Scaling formulas per edge** (edge from source → target node):

```
fan_in  = product(source.shape)   # source node width
fan_out = product(target.shape)   # target node width
L       = effective_depth[target] # depth of the target node

# Forward scaling (maintains O(1) activations)
a = 1 / sqrt(fan_in * L)

# For output nodes (clamped targets): use muPC output scaling
# a_output = 1 / fan_in   (the stronger 1/N scaling)

# Self-gradient scaling: keep self-gradient at O(1)
c_self = 1.0   # no scaling needed, ε is already O(1) from forward scaling

# Top-down gradient scaling: normalize W^T * ε to O(1) per component
# Want: c_td * a * sqrt(fan_out) = O(1)
# c_td = 1 / (a * sqrt(fan_out)) = sqrt(fan_in * L) / sqrt(fan_out)
c_td = sqrt(fan_in * L / fan_out)

# Weight gradient scaling: normalize outer product to desired magnitude
# Default: same as forward (let optimizer handle the rest)
c_w = 1.0
```

### Step 3: MuPCInitializer — in `fabricpc/core/initializers.py`

Add a new initializer class:

```python
class MuPCInitializer(InitializerBase):
    """
    muPC weight initialization: W ~ N(0, 1).

    Weights are drawn from a standard normal distribution (unit variance).
    The actual scaling is applied during the forward pass via per-edge
    scaling factors, not baked into the weight values.
    """
    def __init__(self, gain=1.0):
        super().__init__(gain=gain)

    @staticmethod
    def initialize(key, shape, config=None):
        config = config or {}
        gain = config.get("gain", 1.0)
        return gain * jax.random.normal(key, shape)
```

### Step 4: Add scaling_config to NodeInfo — in `fabricpc/core/types.py`

Add an optional field to `NodeInfo`:

```python
@dataclass(frozen=True)
class NodeInfo:
    # ... existing fields ...
    scaling_config: Any  # Optional MuPCScalingFactors instance, or None
```

This is part of the static graph structure (not a pytree leaf), so it doesn't affect JAX tracing.

### Step 5: Compute and Attach Scalings at Graph Build Time — in `fabricpc/builder/graph_builder.py`

Modify the `graph()` function to optionally accept a `scaling` parameter:

```python
def graph(
    nodes, edges, task_map, inference,
    graph_state_initializer=None,
    scaling=None,          # NEW: Optional MuPCScalingFactors config or DepthMetricBase
) -> GraphStructure:
```

After building `NodeInfo` for all nodes (step 4 in current code), if `scaling` is provided:
1. Call `depth_metric.compute(finalized_nodes, edge_infos)` to get per-node depths
2. Call `compute_mupc_scalings(...)` to get per-node `MuPCScaling`
3. Attach each `MuPCScaling` to its `NodeInfo.scaling_config`

### Step 6: Apply Scaling Without Modifying Any Node's `forward()` — in `fabricpc/nodes/base.py`

**Key design**: All four scalings are applied in `NodeBase`'s `forward_and_latent_grads()`,
`forward_and_weight_grads()`, and `energy_functional()` methods. No changes to `Linear.forward()`
or any other node subclass's `forward()`.

The trick: **pre-scale the inputs before passing them to `forward()`**. Since
`W @ (a*x) = a * (W @ x)` for linear operations, scaling inputs is mathematically
equivalent to scaling the output. Because the scaling happens _inside_ the function
being differentiated by JAX, it automatically flows into both input gradients and
weight gradients through the chain rule.

#### 6a. Forward scaling via input pre-scaling in `forward_and_latent_grads()`

In `base.py`, before the `jax.value_and_grad` call at line 381, scale each
input tensor by its per-edge forward scale:

```python
# In forward_and_latent_grads(), in the else branch (line 378+):

# Apply muPC forward scaling by pre-scaling inputs
scaled_inputs = inputs
if node_info.scaling_config is not None:
    scaled_inputs = {
        edge_key: x * node_info.scaling_config.forward_scale[edge_key]
        for edge_key, x in inputs.items()
    }

# Use JAX's value_and_grad on the SCALED inputs
(total_energy, new_state), input_grads = jax.value_and_grad(
    node_class.forward, argnums=1, has_aux=True
)(params, scaled_inputs, state, node_info)
```

Because `a_l` is inside the differentiated function, `input_grads` already
includes `a_l` in the chain rule. The result: `dE/dx = a_l * W^T * f'(z) * ε`,
which is exactly the muPC top-down gradient before the `c_td` correction.

The same pattern applies to the unclamped leaf branch (line 364) and the
`forward_and_weight_grads()` method (line 423).

#### 6b. Self-gradient scaling in `energy_functional()`

In `base.py`, modify `energy_functional()` to apply `c_self`:

```python
@staticmethod
def energy_functional(state, node_info):
    energy_obj = node_info.energy
    energy_cls = type(energy_obj)
    config = energy_obj.config

    energy = energy_cls.energy(state.z_latent, state.z_mu, config)
    grad = energy_cls.grad_latent(state.z_latent, state.z_mu, config)

    # Apply muPC self-gradient scaling
    if node_info.scaling_config is not None:
        grad = grad * node_info.scaling_config.self_grad_scale

    latent_grad = state.latent_grad + grad
    state = state._replace(energy=energy, latent_grad=latent_grad)
    return state
```

#### 6c. Top-down gradient scaling in `forward_and_latent_grads()`

After the `value_and_grad` call, apply the additional `c_td` correction:

```python
# Apply muPC top-down gradient scaling per edge (AFTER autodiff)
if node_info.scaling_config is not None:
    input_grads = {
        edge_key: grad * node_info.scaling_config.topdown_grad_scale[edge_key]
        for edge_key, grad in input_grads.items()
    }
```

Note: The autodiff result already includes `a_l` from the input pre-scaling.
The `c_td` here is the **additional** correction factor. So the total top-down
gradient is: `c_td * a_l * W^T * f'(z) * ε`. For unit variance we need
`c_td = 1 / sqrt(fan_out)` (since `a_l` already handles `1/sqrt(fan_in * L)`).

#### 6d. Weight gradient scaling in `forward_and_weight_grads()`

Same input pre-scaling pattern, then additional `c_w` correction:

```python
# In forward_and_weight_grads():

# Pre-scale inputs (same as inference)
scaled_inputs = inputs
if node_info.scaling_config is not None:
    scaled_inputs = {
        edge_key: x * node_info.scaling_config.forward_scale[edge_key]
        for edge_key, x in inputs.items()
    }

(total_energy, new_state), params_grad = jax.value_and_grad(
    node_class.forward, argnums=0, has_aux=True
)(params, scaled_inputs, state, node_info)

# Apply muPC weight gradient scaling per edge
if node_info.scaling_config is not None:
    scaled_weights = {
        edge_key: grad * node_info.scaling_config.weight_grad_scale[edge_key]
        for edge_key, grad in params_grad.weights.items()
    }
    params_grad = NodeParams(weights=scaled_weights, biases=params_grad.biases)
```

### Step 7: Integration Points Summary

**All scaling applied in `base.py` only — no node subclass modifications:**

| Scaling | What it controls | Where applied | Mechanism |
|---------|-----------------|---------------|-----------|
| `forward_scale[edge]` | O(1) forward activations | `NodeBase.forward_and_latent_grads()` and `forward_and_weight_grads()` | Pre-scale inputs before calling `node_class.forward()` |
| `self_grad_scale` | Self-gradient magnitude | `NodeBase.energy_functional()` | Multiply `grad_latent` output |
| `topdown_grad_scale[edge]` | Top-down gradient to presynaptic node | `NodeBase.forward_and_latent_grads()` after autodiff | Multiply `input_grads` per edge |
| `weight_grad_scale[edge]` | Weight update magnitude | `NodeBase.forward_and_weight_grads()` after autodiff | Multiply `params_grad.weights` per edge |

**Why input pre-scaling works for any node type:**
- For linear nodes: `W @ (a*x) = a * (W @ x)` — exact equivalence
- For attention/transformer: pre-scaling Q, K, V inputs scales the output proportionally
- For any differentiable function: `f(a*x)` has gradient `a * f'(a*x)` w.r.t. x, which
  is exactly the scaled Jacobian we want for the top-down gradient
- The node's `forward()` is a black box — it never needs to know about scaling

### Step 8: User API

```python
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.depth_metric import ShortestPathDepth
from fabricpc.core.initializers import MuPCInitializer

# Create nodes with MuPCInitializer
h1 = Linear(shape=(128,), name="h1", weight_init=MuPCInitializer())
h2 = Linear(shape=(128,), name="h2", weight_init=MuPCInitializer())
out = Linear(shape=(10,), name="out", weight_init=MuPCInitializer())

# Build graph with muPC scaling
structure = graph(
    nodes=[x_node, h1, h2, out],
    edges=[...],
    task_map=TaskMap(x=x_node, y=out),
    inference=InferenceSGD(eta_infer=0.5, infer_steps=30),
    scaling=MuPCConfig(depth_metric=ShortestPathDepth()),
)
```

### Implementation Order

1. `fabricpc/core/depth_metric.py` — new file, no dependencies
2. `fabricpc/core/initializers.py` — add `MuPCInitializer` class
3. `fabricpc/core/mupc.py` — new file, depends on depth_metric
4. `fabricpc/core/types.py` — add `scaling_config` to `NodeInfo`
5. `fabricpc/builder/graph_builder.py` — compute and attach scalings
6. `fabricpc/nodes/base.py` — apply all four scalings via input pre-scaling + post-hoc gradient corrections
7. Tests and example script

Note: No changes to `fabricpc/nodes/linear.py` or any other node subclass.


Implementation Summary                                                                                                                                                                                                     
                                                                                                                                                                                                                             
  New Files (2)                                                                                                                                                                                                              
                                                                                                                                                                                                                             
  - fabricpc/core/depth_metric.py — Extensible depth computation for arbitrary graphs                                                                                                                                        
    - DepthMetricBase (abstract): base class for custom depth metrics                                                                                                                                                        
    - ShortestPathDepth: BFS-based minimum distance from terminal input nodes                                                                                                                                                        
    - LongestPathDepth: DAG DP-based maximum distance from terminal input nodes                                                                                                                                                      
    - FixedDepth: user-specified constant depth for all nodes with in_degree>0                                                                                                                                                     
  - fabricpc/core/mupc.py — Core muPC scaling computation                                                                                                                                                                    
    - MuPCScaling: frozen dataclass holding per-node scaling factors (forward_scale, self_grad_scale, topdown_grad_scale, weight_grad_scale — all per-edge except self_grad)                                                 
    - MuPCConfig: configuration passed to graph() builder                                                                                                                                                                    
    - compute_mupc_scalings(): computes scaling factors from graph topology                                                                                                                                                  
                                                                                                                                                                                                                             
  Modified Files (4)                                                                                                                                                                                                         
                                                                                                                                                                                                                             
  - fabricpc/core/initializers.py — Added MuPCInitializer (W ~ N(0, gain^2), unit-variance weights with scaling decoupled to forward pass)                                                                                   
  - fabricpc/core/types.py — Added scaling_config: Any = None field to NodeInfo                                                                                                                                              
  - fabricpc/builder/graph_builder.py — Added optional scaling parameter to graph(). When a MuPCConfig is provided, computes per-node scalings and attaches them to NodeInfo.scaling_config                                  
  - fabricpc/nodes/base.py — Applied all four scalings without modifying any node's forward():                                                                                                                               
    - _apply_forward_scaling(): pre-scales inputs by per-edge forward_scale (since W@(ax) = a(W@x))                                                                                                                          
    - forward_and_latent_grads(): pre-scales inputs + applies topdown_grad_scale to input_grads after autodiff                                                                                                                      
    - forward_and_weight_grads(): pre-scales inputs + applies weight_grad_scale to weight grads after autodiff                                                                                                                       
    - energy_functional(): applies self_grad_scale to the self-latent gradient                                                                                                                                               
                                                                                                                                                                                                                             
  Test File (1)                                                                                                                                                                                                              
                                                                                                                                                                                                                             
  - tests/test_mupc.py — 25 tests across 5 test classes covering depth metrics, initializer, scaling formulas, graph builder integration, forward scaling application, and end-to-end training   

# Revision to muPC Design Plan
## Width computation
The agent made a critical error in interpreting the fan-in and fan-out dimensions. These refer to the Weight matrix dimensions (Xavier/He), not the input and output dimensions of the node.

Comparison of our mupc_demo.py against the jpc reference implementation (https://github.com/thebuckleylab/jpc) reveals critical differences in the forward scaling formula and several hyperparameter mismatches.                                                                                         
                                                                                                                                                                                                                                                                                                           
 Critical Issue: Forward Scaling Uses fan_in Instead of Hidden Width N                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                           
 Reference (jpc) — scaling is based on hidden layer width (uniform across the network):                                                                                                                                                                                                                    
 a_1 = 1 / sqrt(D)           # D = input dimension                                                                                                                                                                                                                                                         
 a_l = 1 / sqrt(N * L)       # N = hidden width, L = total depth (with skips)                                                                                                                                                                                                                              
 a_L = 1 / N                 # output layer                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                           
 Our implementation — scaling is based on per-edge fan_in (total element count of source node):                                                                                                                                                                                                            
 a_l = 1 / sqrt(fan_in * L)  # fan_in = np.prod(source_shape), L = effective depth                                                                                                                                                                                                                         
 a_L = 1 / fan_in            # output layer                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                           
 For convolutional nodes, fan_in = np.prod(source_shape) is the total spatial element count (e.g., 32×32×32 = 32768 for conv1→conv2), not the number of features/channels. This makes the forward scaling orders of magnitude too small.                                                                   
                                                                                                                                                                                                                                                                                                           
 | Edge         | Our fan_in       | Our a    | Reference N=width      | Reference a |                                                                                                                                                                                                                     
 |--------------|------------------|----------|------------------------|-------------|                                                                                                                                                                                                                     
 | input→conv1  | 3072 (32×32×3)   | 0.018    | ~96 (sqrt of channels) | ~0.10       |                                                                                                                                                                                                                     
 | conv1→conv2  | 32768 (32×32×32) | 0.0039   | ~32 (channels)         | ~0.18       |                                                                                                                                                                                                                     
 | conv2→conv3  | 16384 (16×16×64) | 0.0045   | ~64 (channels)         | ~0.13       |                                                                                                                                                                                                                     
 | conv3→output | 8192 (8×8×128)   | 0.000122 | ~128 (channels)        | 0.0078      |                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                           
 The output layer forward scale is off by ~64x and hidden layers are off by ~30-50x. With activations scaled this aggressively, predictions are near-zero before softmax, producing near-uniform output probabilities → chance level.                                                                      
                                                                                                                                                                                                                                                                                                           
 Secondary Issues (Demo Hyperparameters)                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                           
 | Parameter                | Our Demo             | Reference (jpc) | Impact                                            |                                                                                                                                                                                 
 |--------------------------|----------------------|-----------------|---------------------------------------------------|                                                                                                                                                                                 
 | Inference LR (eta_infer) | 0.05                 | 0.5             | 10x too low — latents barely move                 |                                                                                                                                                                                 
 | Parameter LR             | 0.001 (AdamW)        | 0.1 (Adam)      | 100x too low — slow weight updates                |                                                                                                                                                                                 
 | Inference steps          | 20                   | 30 (= depth)    | Fewer steps for convergence                       |                                                                                                                                                                                 
 | Architecture             | ConvNet on CIFAR-100 | FC MLP on MNIST | Much harder task, reference not validated on conv |                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                           
 Additional Concern: Top-Down Gradient Scaling                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                           
 Formula: c_td = 1 / (a * sqrt(fan_out))                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                           
 With a already tiny (0.000122 for output) and fan_out = 100:                                                                                                                                                                                                                                              
 c_td = 1 / (0.000122 * 10) ≈ 820                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                           
 This produces a massive top-down gradient multiplier. In the reference implementation, there is NO separate top-down gradient scaling — the scaling is only applied in the forward direction within the energy function, and autodiff naturally propagates correctly-scaled gradients.                    
                                                                                                                                                                                                                                                                                                           
 ---                                                                                                                                                                                                                                                                                                       
 Plan: Fix Forward Scaling + Demo Hyperparameters                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                           
 Step 1: Fix fan_in computation for Conv2DNode in compute_mupc_scalings                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                           
 File: fabricpc/core/mupc.py:136                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                           
 The current code computes fan_in = int(np.prod(source_shape)) which for spatial tensors gives the total element count rather than the feature width. For muPC, fan_in should represent the "width" (number of features/channels) in the Xavier/Kaiming sense, not the total spatial element count.        
                                                                                                                                                                                                                                                                                                           
 For convolutional nodes, the meaningful width is the number of channels (last dimension of NHWC tensors), not H*W*C. For linear nodes with flatten_input, it's the total flattened dimension. The reference uses a single uniform N (hidden width) across the whole network, but our graph-based approach 
  needs per-edge fan_in.                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                           
 Proposed fix: The forward scaling factor a should use the weight matrix fan_in — the number of input units to the weight matrix — rather than the total source tensor size. This matches the Xavier convention:                                                                                           
 - For convolutional layers: fan_in = C_in * kH * kW (kernel receptive field × input channels). This is what JAX initializers use.                                                                                                                                                                         
 - For linear layers with flatten_input=True: fan_in = np.prod(source_shape) (current behavior, correct).                                                                                                                                                                                                  
 - For linear layers without flatten: fan_in = source_shape[-1] (last axis features).                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                           
 To implement this cleanly, add a method to retrieve the effective fan_in from each node's config and incoming edge, rather than using np.prod(source_shape) universally. The node class knows its kernel size and weight matrix structure.                                                                
                                                                                                                                                                                                                                                                                                           
 Implementation approach: Add a classmethod or static method get_weight_fan_in(source_shape, config) to NodeBase that returns the weight-matrix fan_in for a given incoming edge. Override in Conv2DNode to return C_in * kH * kW. The default implementation returns np.prod(source_shape) for backward   
 compatibility. Then compute_mupc_scalings calls this method instead of np.prod(source_shape).                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                           
 Similarly, fan_out (currently np.prod(node_info.shape)) should use the weight matrix fan_out:                                                                                                                                                                                                             
 - Conv2D: C_out * kH * kW                                                                                                                                                                                                                                                                                 
 - Linear with flatten: np.prod(node_shape)                                                                                                                                                                                                                                                                
 - Linear without flatten: node_shape[-1]                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
 Add a corresponding get_weight_fan_out(node_shape, config) method.                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                           
 Step 2: Fix demo hyperparameters in mupc_demo.py                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                           
 File: examples/mupc_demo.py                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                           
 - Change eta_infer from 0.05 to 0.5                                                                                                                                                                                                                                                                       
 - Change infer_steps from 20 to 50 (deeper network needs more steps)                                                                                                                                                                                                                                      
 - Change optimizer LR from 0.001 to 0.01 (10x increase, conservative compared to reference's 0.1)                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                           
 Step 3: Fix demo hyperparameters in mupc_ab_experiment.py                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                           
 File: examples/mupc_ab_experiment.py                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                           
 Same hyperparameter changes as Step 2 for both arms. Both arms should use the higher inference LR and steps.                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                           
 Step 4: Disable top-down gradient scaling (set to 1.0)                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                           
 File: fabricpc/core/mupc.py                                                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                           
 The reference implementation does NOT apply a separate top-down gradient scaling. The muPC paper's scaling is purely forward-direction: scale the prediction a_l * f_l(z_{l-1}), and let autodiff handle the rest naturally. Our c_td formula amplifies gradients to compensate for the forward           
 attenuation, but this creates numerical instability when a is small.                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                           
 Set topdown_grad_scale = 1.0 for all edges (same as weight_grad_scale). This aligns with the reference. Keep the infrastructure in place as a placeholder for future exploration.                                                                                                                         
                                                                                                                                                                                                                                                                                                           
 Step 5: Run the demo and verify learning                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
 After fixes, verify that the mupc_demo.py shows learning above chance (>1% for CIFAR-100) and that energies decrease over training.                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                           
 ---                                                                                                                                                                                                                                                                                                       
 Files to Modify                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                           
 | File                           | Change                                                           |                                                                                                                                                                                                     
 |--------------------------------|------------------------------------------------------------------|                                                                                                                                                                                                     
 | fabricpc/core/mupc.py          | Fix fan_in/fan_out to use weight-matrix dimensions; set c_td=1.0 |                                                                                                                                                                                                     
 | fabricpc/nodes/base.py         | Add get_weight_fan_in/get_weight_fan_out classmethods            |                                                                                                                                                                                                     
 | fabricpc/nodes/linear.py       | Override fan methods for flatten_input logic                     |                                                                                                                                                                                                     
 | fabricpc/nodes/identity.py     | Override fan methods (no weights → return source shape prod)     |                                                                                                                                                                                                     
 | examples/custom_node.py        | Override fan methods for Conv2D kernel dims                      |                                                                                                                                                                                                     
 | examples/mupc_demo.py          | Fix hyperparameters                                              |                                                                                                                                                                                                     
 | examples/mupc_ab_experiment.py | Fix hyperparameters                                              |                                                                                                                                                                                                     
 | tests/test_mupc.py             | Update expected scaling values in tests                          |      


1. Feedforward state initialization missing forward scaling (fabricpc/graph/state_initializer.py:272)                                                                                                                                                                                                    
    - FeedforwardStateInit called node_class.forward() without first applying _apply_forward_scaling, causing activation explosion during initialization (conv3 std=622 instead of 0.12)                                                                                                                   
    - Fixed in previous session                                                                                                                                                                                                                                                                            
  2. Softmax saturation killing gradient flow (newly identified this session)                                                                                                                                                                                                                              
    - With MuPCInitializer (W ~ N(0,1)) and fan_in=8192, the output layer's pre-activations had std ≈ 12.4                                                                                                                                                                                                 
    - Softmax with pre-act std > 5 produces near-one-hot outputs, where the gradient is numerically zero                                                                                                                                                                                                   
    - This meant zero gradients flowed back to all hidden layers, so no learning occurred                                                                                                                                                                                                                  
    - Confirmed: all weight gradients were exactly 0.0 for all layers                                                                                                                                                                                                                                      
  3. Output node initialization mismatch                                                                                                                                                                                                                                                                   
    - The output node should not use MuPCInitializer (std=1.0) because it's excluded from muPC forward scaling                                                                                                                                                                                             
    - Fixed by using XavierInitializer() for the output node, which produces properly scaled weights (std ≈ 0.016 for fan_in=8192)                                                                                                                                                                         
    - This keeps pre-activations in the safe range (std ≈ 0.3) where softmax gradients are non-degenerate                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
  Files Modified This Session                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                           
  | File                           | Change                                                                               |                                                                                                                                                                                
  |--------------------------------|--------------------------------------------------------------------------------------|                                                                                                                                                                                
  | fabricpc/core/mupc.py          | Output nodes (out_degree=0) excluded from forward scaling; use standard init instead |                                                                                                                                                                                
  | examples/mupc_demo.py          | Output uses XavierInitializer() instead of MuPCInitializer(); LR reduced to 0.001    |                                                                                                                                                                                
  | examples/mupc_ab_experiment.py | Output uses XavierInitializer() for muPC arm; LR reduced to 0.001                    |                                                                                                                                                                                
  | tests/test_mupc.py             | Output node scaling tests updated                                                    |                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                           
  Results                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
  - Before: Energy flat, 1.0% accuracy (chance), all weight gradients = 0.0                                                                                                                                                                                                                                
  - After: Energy decreasing (73 → 2.9 → 2.2 → 0.5 → 0.35), 4.31% accuracy (4x above chance after just 5 epochs)                                                                                                                                                                                           
  - All 120 tests pass 

# Revisions to plan
Based on new insights from debugging several demos, the muPC is a brittle heuristic tied to resnet-like architectures. Instead, we should have variance scaling responsive to arbitrary graph topology.

## Principled muPC Forward Scaling via Variance Propagation                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
 Problem                                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                           
 mupc_mnist_demo.py --num_hidden=20 produces 9.8% (chance) accuracy. With --num_hidden=2 it achieves 91.6%. The root cause: the current depth-based scaling formula a = 1/sqrt(fan_in * L) uses a heuristic depth metric that doesn't properly model why variance grows in different graph topologies.     
                                                                                                                                                                                                                                                                                                           
 Root Cause: Depth is a proxy for summation amplification                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
 The jpc reference's 1/sqrt(L) factor compensates for variance amplification from skip-connection summation in ResNets (z = a*W*phi(z) + z_skip). In a chain without skip connections, there's no summation amplification — the depth factor adds unnecessary damping that cascades, causing Var ∝ 1/L!    
 (factorial collapse).                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                           
 The principled fix: replace the heuristic depth metric with variance propagation through the graph.   