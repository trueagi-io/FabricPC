# Convolutional Node Design Retrospective & Improvement Plan

> [!NOTE]
> This document consolidates: (1) a retrospective of blunders from the first
> implementation attempt, (2) a precise mapping of every hand-rolled piece of
> code to the existing FabricPC class that should replace it, and (3) the
> minimal surface area of `ConvNode` that actually needs to be written from scratch.

---

## 0. House-Keeping

The top-level `docs/internship_plan.md` has been moved to
`docs/dev_plans_archive/internship_plan.md`; a redirect stub remains.

**Rule going forward**: All development plan documents live in
`docs/dev_plans_archive/`. Update files in-place rather than spawning siblings.

---

## 1. What `NodeBase` Already Does (Do Not Re-Implement)

Before listing mistakes, it is critical to understand what the base class
provides. Examining [`base.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/base.py)
and [`linear.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/linear.py)
as the canonical reference node:

| Responsibility | Provided by | Location |
|---|---|---|
| Activation application | `NodeBase` convention: `type(activation).forward(x, activation.config)` | All `forward()` methods |
| Energy computation | `NodeBase.energy_functional(state, node_info)` | [`base.py:L486`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/base.py#L486-L511) |
| Latent gradient (autodiff) | `NodeBase.forward_and_latent_grads()` | [`base.py:L347`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/base.py#L347-L446) |
| Weight gradient (autodiff) | `NodeBase.forward_and_weight_grads()` | [`base.py:L448`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/base.py#L448-L483) |
| Fan-in default | `NodeBase.get_weight_fan_in()` | [`base.py:L319`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/base.py#L319-L341) |
| Flatten + dense matmul | `FlattenInputMixin.compute_linear()` | [`base.py:L119`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/base.py#L119-L147) |
| Activation gain for muPC | `ActivationBase.variance_gain()` | [`activations.py:L122`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/core/activations.py#L122-L137) |
| Weight initialization dispatch | `initialize(key, shape, initializer)` | [`initializers.py:L314`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/core/initializers.py#L314-L332) |
| Bias initialization | `ZerosInitializer.initialize(key, shape)` | [`initializers.py:L86`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/core/initializers.py#L86-L101) |

**The entire gradient computation chain is handled by `NodeBase`** via
`jax.value_and_grad` wrapping the node's own `forward()`. The node only needs
to implement `forward()` correctly, and `NodeBase` will differentiate through it.
`lax.conv_general_dilated` is JAX-differentiable, so this works automatically.

---

## 2. Retrospective: Blunders & Mistakes in the First Attempt

### 2.1 `lax.conv_general_dilated` — Redundant and Dangerous

**Mistake**: Calling `jax.lax.conv_general_dilated` directly, bypassing any
higher-level JAX convenience wrappers.

**Why it matters**:
- `lax.conv_general_dilated` is a low-level XLA primitive. Its argument ordering,
  dimension number strings (`"NLC"`, `"HWIO"`, etc.) and dilation semantics are
  notoriously easy to get wrong and produce silent shape mismatches or NaN values.
- JAX provides `jax.lax.conv` and, more usefully, `stax.Conv` / `flax.linen.Conv`
  that wrap this safely, but FabricPC does not use Flax.
- **At minimum**, the dimension number strings must be validated at construction
  time, not silently inferred from a dict lookup.

**Fix**: Keep `lax.conv_general_dilated` (it is the correct primitive for FabricPC),
but encapsulate it inside a validated helper and add an assertion that
`len(kernel_size) == spatial_rank` at `__init__` time, not at first call.

---

### 2.2 `forward()` — Partial Redundancy with `NodeBase`

**Mistake**: The `forward()` method in each Conv class manually:
1. Extracts `stride` and `padding` from config
2. Runs the convolution loop
3. Applies the bias
4. Applies the activation
5. Computes `error = z_latent - z_mu`
6. Updates `state` with `_replace()`
7. Calls `energy_functional`
8. Returns `jnp.sum(state.energy)`

Steps 4–8 are **identical in every node** in the codebase (see
[`linear.py:L210-L225`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/linear.py#L210-L225)
and
[`transformer.py:L376-L389`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/transformer.py#L376-L389)).
They are not provided by `NodeBase` automatically — they are the **required
contract** that every `forward()` must follow. So these steps are correct but
should not be considered "extra work". The actual redundancy is:

- **NaN risk from silent config fallbacks** — if `stride` or `padding` are
  missing from `node_info.node_config`, silent defaults hide misconfiguration.
- **Gradients**: `NodeBase.forward_and_latent_grads()` calls `forward()` via
  `jax.value_and_grad`. No manual gradient code is needed — it is correct as-is.

**Fix**: `forward()` must remain — it is abstract in `NodeBase`. The gradient
and energy logic stays exactly as written; it is the FabricPC convention.

> **Update (shipped):** the raise-on-None config guard in `forward()` was
> *not* implemented. The constructor always sets valid `stride`/`padding`
> defaults, so by the time `forward()` runs those fields are guaranteed
> present — a runtime None check there would be dead code. (PR #19 review.)

---

### 2.3 `get_weight_fan_in` — Overrides Needed, Not Redundant

**Mistake**: The comment "this is kind of redundant" misidentifies the issue.

`NodeBase.get_weight_fan_in` (line 319–341 of `base.py`) defaults to:
```python
if config.get("flatten_input", False):
    return int(np.prod(source_shape))
return source_shape[-1]
```
This returns `C_in` only — **not** `C_in * prod(kernel_size)`. For a convolutional
node, this is **wrong**: the true fan_in for Kaiming initialisation is
`C_in * kH * kW` (every weight in the receptive field contributes to one output unit).

The override **must** exist in `ConvNode`. The blunder was the silent default
`(1,)` fallback — not the existence of the override.

**Correct implementation**:
```python
@staticmethod
def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
    kernel_size = config.get("kernel_size")
    if kernel_size is None:
        raise ValueError(
            "ConvNode requires kernel_size in config. "
            "No safe default exists — (1,) silently destroys spatial context."
        )
    C_in = source_shape[-1]
    return C_in * int(np.prod(kernel_size))
```

---

### 2.4 `initialize_params` — Biases Hardcoded Instead of Using Initializers

**Mistake** (original [`convolutional.py:L112-L115`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/convolutional.py#L112-L115)):
```python
# Original — WRONG
if use_bias:
    bias = jnp.zeros((1, 1, out_channels))   # hardcoded zeros, hardcoded shape
else:
    bias = jnp.array([])
```

There are **three separate problems**:

1. **Zeros hardcoded** — bypasses the `bias_init` parameter entirely.
   The user cannot pass a custom bias initializer.
2. **Shape hardcoded** — `(1, 1, out_channels)` is correct only for 1D inputs.
   For 2D it should be `(1, 1, 1, out_channels)`, for 3D `(1, 1, 1, 1, out_channels)`.
   The `Linear` node ([`linear.py:L153-L154`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/linear.py#L153-L154))
   derives this correctly:
   ```python
   bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
   ```
3. **No `bias_init` in `__init__` signature** — the parameter does not exist at all.

**Fix**: Add `bias_init: Optional[InitializerBase] = ZerosInitializer()` to
`ConvNode.__init__`. Pass it through `**extra_config` to `NodeBase`. In
`initialize_params`, use:
```python
bias_init = config.get("bias_init") or ZerosInitializer()
bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
bias = initialize(keys[-1], bias_shape, bias_init)
```
If the user does not pass `bias_init`, `ZerosInitializer` is the sensible
default (standard practice; biases contribute zero variance at init).

> **Update (shipped):** `bias_init` (and `weight_init`) are defaulted once, in
> the `__init__` signature (`ZerosInitializer()` / `KaimingInitializer()`).
> `initialize_params` does **not** re-default them — the constructor is the
> single source of truth, so there is no duplicated defaulting between the two.
> The original design (§2.8) called for raising when `use_bias=True` and
> `bias_init is None`; shipped code keeps that raise, but in `initialize_params`
> (the point of use), not the constructor. Normal graph construction never
> triggers it because the signature default fills `bias_init`; it only guards a
> hand-built config or an explicit `bias_init=None` paired with `use_bias=True`.

---

### 2.5 `initialize_params` — Kaiming `fan` Read from Wrong Axis

**Mistake**: The `KaimingInitializer` reads `fan` from `shape[0]` (line 261–263
of `initializers.py`):
```python
if mode == "fan_out":
    fan = shape[1] if len(shape) > 1 else shape[0]
else:  # fan_in
    fan = shape[0]
```
For a weight kernel of shape `(*kernel_size, C_in, C_out)`, `shape[0]` is the
**first spatial dimension** (e.g. `kH`), not `C_in * kH * kW`.

The initializer convention assumes `shape = (fan_in, fan_out)` (a 2D matrix).
For convolutional kernels, the kernel must be **reshaped** to `(fan_in, fan_out)` before passing to the
initializer, and then the returned flat weights reshaped back to `(*kernel_size, C_in, C_out)`.

**Fix**: make the initializers themselves shape-aware so a conv kernel can be
passed directly. `KaimingInitializer`/`XavierInitializer` now derive
`fan_in = prod(shape[:-1])` and `fan_out = prod(shape[:-2]) * shape[-1]` for any
`len(shape) >= 2`, so `initialize(key, (*kernel_size, C_in, C_out), weight_init)`
computes the right variance with no reshaping.

> **Update (shipped):** the original plan (reshape to `(fan_in, fan_out)` via an
> `_init_conv_kernel` wrapper, plus an activation-vs-initializer compatibility
> check that raised for non-ReLU activations) was **dropped** in review. The
> reshape wrapper is redundant now that the initializers are shape-aware, and
> the activation check was overreach — the user is free to pair any activation
> with any initializer. (PR #19 review.)

---

### 2.6 Silent `kernel_size` Default in `get_weight_fan_in` and `initialize_params`

**Mistake** (lines 81, 97, 215, 231, 349, 365 of the original):
```python
kernel_size = config.get("kernel_size", (1,))   # Conv1DNode
kernel_size = config.get("kernel_size")          # then used without None check
```
Two separate failure modes:
- In `get_weight_fan_in`: default `(1,)` silently miscalculates fan_in.
- In `initialize_params`: `config.get("kernel_size")` returns `None`, then
  `(*kernel_size, C_in, C_out)` raises a confusing `TypeError: cannot unpack None`.

**Fix**: `kernel_size` is a **required** constructor argument with no default,
so it is always present in config by the time these methods run.

> **Update (shipped):** the `kernel_size is None` runtime guards in
> `get_weight_fan_in` and `initialize_params` were **removed** as dead code —
> the constructor's required-argument contract makes them unreachable.
> (PR #19 review.)

---

### 2.7 Three Classes Instead of One

**Mistake**: `Conv1DNode`, `Conv2DNode`, `Conv3DNode` with copy-pasted bodies.
The only difference is the JAX dimension-number string and the default stride
tuple. Every bug must be fixed three times.

**Fix**: Single `ConvNode`. Infer spatial rank from `len(shape) - 1`. Map rank
to JAX dimension string at runtime.

---

### 2.8 No `bias_init` Exposed to User

**Mistake**: The user has no way to change bias initialisation. This contradicts
the FabricPC philosophy where every initialization decision is passed explicitly
(see all built-in nodes).

---

### 2.9 No `SlotSpec` Flexibility for Skip Connections

**Mistake**: All three classes hardcode exactly one slot with no user override.
The user cannot declare a skip-connection input without subclassing.

> **Update (shipped):** the `slots=` constructor parameter was **removed**. No
> node in FabricPC has an instance-level slot mechanism — the framework reads
> slots only via the static `get_slots()`, so `slots=` was silently ignored
> (non-functional). Skip connections are expressed through the multi-input slot
> (multiple edges into `"in"`), as the resnet-18 demo does. Per-instance slots
> are deferred to a future infrastructure PR. (PR #19 review.)

---

## 3. Code-Level Map: What to Inherit vs. What to Write

```
ConvNode inherits NodeBase
│
├── DO NOT implement:
│   ├── forward_and_latent_grads() ─── NodeBase does autodiff via jax.value_and_grad
│   ├── forward_and_weight_grads() ─── NodeBase does autodiff via jax.value_and_grad
│   ├── energy_functional()        ─── NodeBase calls node_info.energy.energy(...)
│   └── activation application     ─── convention: type(act).forward(x, act.config)
│                                      same in Linear, TransformerBlock, all nodes
│
├── MUST override (abstract in NodeBase):
│   ├── get_slots()           ─── define spatial conv input slot(s)
│   ├── initialize_params()   ─── allocate kernels + biases correctly
│   └── forward()             ─── run lax.conv_general_dilated + bias + activation
│                                  + error + energy_functional (standard contract)
│
└── SHOULD override (default is wrong for conv):
    └── get_weight_fan_in()   ─── must return C_in * prod(kernel_size), not C_in
```

---

## 4. Proposed `ConvNode` — Minimal Correct Implementation

```python
"""
Unified convolutional node for 1D, 2D, and 3D spatial data.

Inherits all gradient computation and energy logic from NodeBase.
Only initialize_params(), forward(), get_slots(), and get_weight_fan_in()
are implemented here; everything else is provided by NodeBase.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    KaimingInitializer, ZerosInitializer, NormalInitializer, initialize
)

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class ConvNode(NodeBase):
    """
    Unified convolutional node (1D, 2D, 3D) for predictive coding graphs.

    Spatial rank is inferred from the output shape:
        len(shape) == 2  →  1D conv  (L_out, C_out)
        len(shape) == 3  →  2D conv  (H_out, W_out, C_out)
        len(shape) == 4  →  3D conv  (D_out, H_out, W_out, C_out)

    Gradient computation (latent + weight), energy accumulation, and
    activation application follow the NodeBase contract — nothing is
    reimplemented here.

    Args:
        shape:       Output shape excluding batch (e.g. (28, 28, 32) for 2D).
        name:        Node name (auto-prefixed by graph namespace).
        kernel_size: Spatial kernel size, e.g. (3, 3) for 2D. REQUIRED — no default.
        stride:      Stride per spatial axis. Defaults to all-ones if None.
        padding:     "SAME" (output same size as input) or "VALID" (no padding).
        activation:  ActivationBase instance. Default: ReLUActivation.
        energy:      EnergyFunctional instance. Default: GaussianEnergy.
        use_bias:    Whether to add a learnable bias. Default: True.
        weight_init: InitializerBase for kernels. Default: KaimingInitializer.
                     NOTE: std is computed analytically from fan_in (see §2.5).
        bias_init:   InitializerBase for bias. Default: ZerosInitializer.
                     initialize_params raises if use_bias=True and bias_init
                     is None (only reachable via a hand-built config).
        latent_init: InitializerBase for latent states. Default: NormalInitializer.
        (Note: the `slots=` parameter shown in the original sketch was removed;
        skip connections use the multi-input `"in"` slot instead.)
    """

    _DIM_NUMBERS = {
        1: ("NLC",   "LIO",   "NLC"),
        2: ("NHWC",  "HWIO",  "NHWC"),
        3: ("NDHWC", "DHWIO", "NDHWC"),
    }

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        kernel_size: Tuple[int, ...],           # REQUIRED — no default
        stride: Optional[Tuple[int, ...]] = None,
        padding: str = "SAME",
        activation: Optional["ActivationBase"] = None,
        energy: Optional["EnergyFunctional"] = None,
        use_bias: bool = True,
        weight_init: Optional["InitializerBase"] = None,
        bias_init: Optional["InitializerBase"] = None,
        latent_init: Optional["InitializerBase"] = None,
        slots: Optional[Dict[str, SlotSpec]] = None,
    ):
        spatial_rank = len(shape) - 1
        if spatial_rank not in (1, 2, 3):
            raise ValueError(
                f"ConvNode shape must have 2–4 elements (spatial_rank 1/2/3). "
                f"Got shape={shape}."
            )
        if len(kernel_size) != spatial_rank:
            raise ValueError(
                f"kernel_size length {len(kernel_size)} must equal "
                f"spatial_rank {spatial_rank} inferred from shape={shape}."
            )
        if use_bias and bias_init is None:
            bias_init = ZerosInitializer()

        if stride is None:
            stride = (1,) * spatial_rank

        # Defaults match Linear node conventions
        if activation is None:
            activation = ReLUActivation()
        if energy is None:
            energy = GaussianEnergy()
        if weight_init is None:
            weight_init = KaimingInitializer()
        if latent_init is None:
            latent_init = NormalInitializer()

        self._slots = slots or {"in": SlotSpec(name="in", is_multi_input=True)}

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            bias_init=bias_init,       # stored in _extra_config by NodeBase
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        # Default single multi-input slot.
        # For skip connections, pass a custom `slots` dict at construction time.
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """
        fan_in = C_in * prod(kernel_size).

        Overrides NodeBase default (which returns only C_in) because the
        receptive field of a conv kernel spans kernel_size spatial positions,
        each contributing C_in connections per output unit.

        Called by the muPC module to compute per-edge scaling factors.
        """
        kernel_size = config.get("kernel_size")
        if kernel_size is None:
            raise ValueError(
                "ConvNode.get_weight_fan_in: kernel_size missing from config."
            )
        C_in = source_shape[-1]
        return C_in * int(np.prod(kernel_size))

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional["InitializerBase"] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize conv kernels and biases.

        Kernel shape: (*kernel_size, C_in, C_out)
        Bias shape:   (1,) * len(node_shape) + (C_out,)   ← derived dynamically, not hardcoded

        For Kaiming initialization, we pass the 2D shape (fan_in, fan_out) to the
        initializer and reshape the returned weights back. This avoids the fan-in
        axis bug since KaimingInitializer reads shape[0] as fan_in.
        """
        if config is None:
            config = {}

        kernel_size = config.get("kernel_size")
        if kernel_size is None:
            raise ValueError(
                "ConvNode.initialize_params: kernel_size must be in config."
            )

        out_channels = node_shape[-1]
        use_bias     = config.get("use_bias", True)
        bias_init    = config.get("bias_init")

        if use_bias and bias_init is None:
            raise ValueError(
                "ConvNode.initialize_params: bias_init must be specified "
                "when use_bias=True. Use ZerosInitializer() for the standard default."
            )

        if weight_init is None:
            weight_init = KaimingInitializer()

        # Split keys: one per edge + one for bias
        keys = jax.random.split(key, len(input_shapes) + 1)

        activation = config.get("activation")

        weights_dict = {}
        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)

            if isinstance(weight_init, KaimingInitializer):
                # Map activation to nonlinearity & negative slope
                from fabricpc.core.activations import ReLUActivation, LeakyReLUActivation
                if isinstance(activation, ReLUActivation):
                    nonlinearity = "relu"
                    a = 0.0
                elif isinstance(activation, LeakyReLUActivation):
                    nonlinearity = "leaky_relu"
                    a = activation.config.get("alpha", 0.01)
                else:
                    raise ValueError(
                        f"KaimingInitializer is only compatible with ReLUActivation or LeakyReLUActivation, "
                        f"but got activation: {type(activation).__name__}"
                    )
                
                # Merge weight_init config and override nonlinearity/a
                init_config = dict(weight_init.config)
                init_config["nonlinearity"] = nonlinearity
                init_config["a"] = a

                fan_in = in_channels * int(np.prod(kernel_size))
                fan_out = out_channels * int(np.prod(kernel_size))
                kaiming_shape = (fan_in, fan_out)

                flat_weight = KaimingInitializer.initialize(keys[i], kaiming_shape, init_config)
                weights_dict[edge_key] = flat_weight.reshape(kernel_shape)
            else:
                weights_dict[edge_key] = initialize(keys[i], kernel_shape, weight_init)

        # Bias — shape derived programmatically (mirrors linear.py:L153–154)
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (out_channels,)
            bias = initialize(keys[-1], bias_shape, bias_init)
            return NodeParams(weights=weights_dict, biases={"b": bias})
        else:
            return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Convolutional forward pass.

        Contract (identical to Linear and TransformerBlock):
            1. Compute pre_activation via conv sum over all input edges.
            2. Add bias if present.
            3. Apply activation via NodeBase convention.
            4. Compute error = z_latent - z_mu.
            5. Update state with _replace().
            6. Call energy_functional() (from NodeBase).
            7. Return (total_energy, state).

        NodeBase.forward_and_latent_grads() and forward_and_weight_grads()
        differentiate through this function automatically — no manual gradient
        code is needed here.
        """
        config  = node_info.node_config
        stride  = config.get("stride")
        padding = config.get("padding")

        spatial_rank = len(node_info.shape) - 1
        dim_numbers  = ConvNode._DIM_NUMBERS[spatial_rank]

        batch_size     = state.z_latent.shape[0]
        pre_activation = jnp.zeros((batch_size, *node_info.shape))

        for edge_key, x in inputs.items():
            kernel = params.weights[edge_key]
            pre_activation = pre_activation + lax.conv_general_dilated(
                lhs=x,
                rhs=kernel,
                window_strides=stride,
                padding=padding,
                dimension_numbers=dim_numbers,
            )

        # Bias
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Activation — NodeBase convention (same as Linear, TransformerBlock)
        activation = node_info.activation
        z_mu  = type(activation).forward(pre_activation, activation.config)
        error = state.z_latent - z_mu

        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Energy — delegated entirely to NodeBase.energy_functional()
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        return jnp.sum(state.energy), state
```

---

## 5. `SlotSpec` and Skip Connections

### Multi-input aggregation

`is_multi_input=True` on a slot means multiple nodes can wire into it.
The graph builder creates one **edge key** per connection.
`initialize_params` allocates one kernel per edge key.
`forward` sums the convolution outputs — this is a sum-of-convolutions,
not channel concatenation. Each source node can have different `C_in`; each
edge gets its own kernel sized `(*kernel_size, C_in_edge, C_out)`.

### Declaring a skip connection input

```python
conv = ConvNode(
    shape=(16, 16, 64), name="conv2", kernel_size=(3, 3),
    slots={
        "in": SlotSpec(name="in",   is_multi_input=True,  is_variance_scalable=True),
        "skip": SlotSpec(name="skip", is_multi_input=False, is_skip_connection=True),
    }
)
```
`SlotSpec.__post_init__` enforces that `is_skip_connection=True` and
`is_variance_scalable=True` cannot coexist (see
[`base.py:L66-L70`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/base.py#L66-L70)).

---

## 6. Why muPC Turns Off Latent Variance for Skip Connections

In muPC, forward scaling factors `a` grow with depth `L`. If latent states `z`
are initialised with nonzero variance, that noise is amplified by depth-scaled
weights downstream and overwhelms the clean signal on skip paths.
Setting `latent_init` variance to zero (or using `ZerosInitializer` for latents)
ensures the network starts from a clean state where the skip signal dominates.
Skip connections use unscaled weights (`is_variance_scalable=False`), so they
are not affected by depth — this prevents the degradation problem that plain
identity shortcuts would suffer from.

---

## 7. Concrete Action Items

| # | Action | File | Notes |
|---|---|---|---|
| **1** | Delete `Conv1DNode`, `Conv2DNode`, `Conv3DNode` | `convolutional.py` | Replace with single `ConvNode` |
| **2** | Add `bias_init` param to `ConvNode.__init__` | `convolutional.py` | Default `ZerosInitializer()` |
| **3** | Fix `initialize_params` bias shape (programmatic) | `convolutional.py` | Mirror `linear.py:L153–154` |
| **4** | Fix Kaiming fan_in for nd kernels (compute std analytically) | `convolutional.py` | Mirror `transformer.py:L240–258` |
| **5** | Guard all `config.get("kernel_size")` with `ValueError` | `convolutional.py` | Both `get_weight_fan_in` and `initialize_params` |
| **6** | Expose `slots` parameter in `ConvNode.__init__` | `convolutional.py` | Enables skip connections |
| **7** | Remove all manual gradient / energy code | `convolutional.py` | Already handled by `NodeBase` |
| **8** | Validate `kernel_size` length vs. `spatial_rank` in `__init__` | `convolutional.py` | Fail-fast, not at forward time |
| **9** | Write / extend `tests/test_convolutional.py` | `tests/` | Cover 1D/2D/3D, SAME/VALID, multi-input |
| **10** | Write minimal conv demo (MNIST patch) | `examples/` | Validate end-to-end |

> [!IMPORTANT]
> Items 1–8 are blockers for Task 1 of the internship plan. Items 9–10 follow.

> [!WARNING]
> Do **not** remove `forward()` — it is abstract in `NodeBase` and must be
> implemented. Do **not** remove `get_weight_fan_in()` — the base default
> returns the wrong fan_in for convolutional kernels.

---

## 8. File Reference

| File | Role |
|---|---|
| [`fabricpc/nodes/convolutional.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/convolutional.py) | **Target of refactor** |
| [`fabricpc/nodes/base.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/base.py) | `NodeBase`, `SlotSpec`, `FlattenInputMixin` |
| [`fabricpc/nodes/linear.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/linear.py) | Canonical simple node to mirror |
| [`fabricpc/nodes/transformer.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/nodes/transformer.py) | Reference for analytic std init pattern |
| [`fabricpc/core/initializers.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/core/initializers.py) | All initializer classes incl. `ZerosInitializer` |
| [`fabricpc/core/activations.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/core/activations.py) | All activation classes incl. `variance_gain()` |
| [`fabricpc/core/types.py`](file:///mnt/c/Users/User/Downloads/FabricPC/fabricpc/core/types.py) | `NodeParams`, `NodeState`, `NodeInfo` |
| [`docs/dev_plans_archive/internship_plan.md`](file:///mnt/c/Users/User/Downloads/FabricPC/docs/dev_plans_archive/internship_plan.md) | Internship timeline |
