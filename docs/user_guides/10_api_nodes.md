# Nodes API Reference

All node types extend `NodeBase` from `fabricpc.nodes.base`.

## Linear

`fabricpc.nodes.Linear`

Weighted projection node: `z_mu = activation(W @ x + b)`

```python
from fabricpc.nodes import Linear

node = Linear(
    shape=(256,),
    name="hidden1",
    activation=SigmoidActivation(),
    energy=GaussianEnergy(),
    use_bias=True,
    flatten_input=False,
    weight_init=KaimingInitializer(),
    latent_init=NormalInitializer(),
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch dimension |
| `name` | `str` | required | Node name (auto-prefixed with current namespace) |
| `activation` | `ActivationBase` | `IdentityActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `use_bias` | `bool` | `True` | Whether to include a bias term |
| `flatten_input` | `bool` | `False` | If True, flatten all input dims for dense behavior |
| `weight_init` | `InitializerBase` | `KaimingInitializer()` | Weight initializer |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (multi-input)

**Weight shape:**
- `flatten_input=False`: `(in_features, out_features)` — matmul on last axis
- `flatten_input=True`: `(in_numel, out_numel)` — fully-connected dense

**muPC fan_in:**
- `flatten_input=False`: `source_shape[-1]` (last axis features)
- `flatten_input=True`: `prod(source_shape)` (all dims flattened)

---

## IdentityNode

`fabricpc.nodes.IdentityNode`

Passthrough node with no learnable parameters. Sums all inputs when multiple edges connect.

```python
from fabricpc.nodes import IdentityNode

pixels = IdentityNode(shape=(784,), name="pixels")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch |
| `name` | `str` | required | Node name |
| `activation` | `ActivationBase` | `IdentityActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |
| `scale` | `float` | `1.0` | Fixed scaling factor applied to output |

**Slots:** `"in"` (multi-input)

**muPC fan_in:** Always returns `1` (weightless node).

---

## StorkeyHopfield

`fabricpc.nodes.StorkeyHopfield`

Associative memory node combining PC prediction-error energy with Hopfield attractor energy.

```python
from fabricpc.nodes import StorkeyHopfield

hopfield = StorkeyHopfield(
    shape=(128,),
    name="memory",
    hopfield_strength=1.0,
    activation=TanhActivation(),
)
# Connect with: Edge(source, hopfield.slot("in"))
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch |
| `name` | `str` | required | Node name |
| `hopfield_strength` | `float` | `None` | Initial blending strength (learnable if None, fixed if float) |
| `activation` | `ActivationBase` | `TanhActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | PC energy functional |
| `use_bias` | `bool` | `False` | Whether to include bias term |
| `enforce_symmetry` | `bool` | `True` | Symmetrize W via 0.5*(W+W.T) |
| `zero_diagonal` | `bool` | `False` | Zero W diagonal in forward pass |
| `weight_init` | `InitializerBase` | `XavierInitializer()` | Weight initializer for W matrix |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (single-input)

**Energy formulation:**
```
E_total = E_pc + hopfield_strength * E_hop
E_pc   = 0.5 ||z - mu||^2  (or user-specified energy)
E_hop  = (1/2D) z^T (W^2 - W) z
```

**Prediction:**
```
z_mu = activation(probe/(1+s) + (probe @ W) * s/(1+s) + bias)
```
where `s = softplus(raw_strength)` if `hopfield_strength=None` (learnable), otherwise `s = hopfield_strength` (fixed).

**Learnable parameters:** `W` (D x D matrix), `bias` (if `use_bias=True`), `raw_hopfield_strength` (if `hopfield_strength=None`)

---

## TransformerBlock

`fabricpc.nodes.TransformerBlock`

Multi-head self-attention + feedforward MLP in a single node. Uses Rotary Position Embeddings (RoPE).

```python
from fabricpc.nodes import TransformerBlock

block = TransformerBlock(
    shape=(128,), n_heads=8, d_model=128, d_ff=512,
    max_seq_len=256, name="transformer",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape `(features,)` |
| `name` | `str` | required | Node name |
| `activation` | `ActivationBase` | `IdentityActivation()` | Output activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `internal_activation` | `ActivationBase` | `GeluActivation()` | FFN internal activation |
| `num_heads` | `int` | `8` | Number of attention heads |
| `ff_dim` | `int` | `4 * embed_dim` | Feedforward hidden dimension |
| `dropout_rate` | `float` | `0.0` | Dropout rate (currently unused) |
| `pre_norm` | `bool` | `True` | Use pre-norm architecture |
| `use_rope` | `bool` | `True` | Use Rotary Position Embeddings |
| `rope_theta` | `float` | `10000.0` | Base frequency for RoPE |
| `weight_init` | `InitializerBase` | `KaimingInitializer()` | Weight initializer |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (token embeddings), `"mask"` (optional attention mask)

**Architecture:**
```
x → LayerNorm → MHA → + → LayerNorm → FFN → +
└─────────────────────┘ └────────────────────┘
     (residual)              (residual)
```

---

## Decomposed Transformer (v2)

Fine-grained transformer components in `fabricpc.nodes.transformer_v2`. Each component is a separate node for deeper PC inference.

- **`EmbeddingNode`** — Token embedding lookup
- **`MhaResidualNode`** — Multi-head attention with residual connection
- **`LnMlp1Node`** — LayerNorm + first MLP projection
- **`Mlp2ResidualNode`** — Second MLP projection with residual
- **`VocabProjectionNode`** — Project back to vocabulary dimension

See `fabricpc.nodes.transformer_v2` module for detailed API.
