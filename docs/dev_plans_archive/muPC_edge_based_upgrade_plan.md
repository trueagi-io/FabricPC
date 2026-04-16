# Edge-Based muPC Scaling with LinearResidual Node

## Context

The current muPC implementation uses a node-level `apply_variance_scaling` flag to handle skip connections. This requires a separate `SkipConnection` summing node per residual block, **doubling the graph depth** (each block = Linear + SkipConnection = 2 PC nodes). The K=2 in-degree problem (memory: `project_mupc_skip_scaling.md`) was solved by setting `apply_variance_scaling=False` on SkipConnection, but this is a coarse, all-or-nothing mechanism.

This plan introduces:
1. **Per-slot variance scalability** via `SlotSpec.is_variance_scalable`
2. **`LinearResidual` node** that internalizes the residual sum (halving graph depth)
3. **Edge-based muPC algorithm** where scaling factors are computed per in-edge based on the target slot's properties

## Files to Modify

| File | Change |
|------|--------|
| `fabricpc/nodes/base.py` | Add `is_variance_scalable` to `SlotSpec` |
| `fabricpc/core/types.py` | Add `is_variance_scalable` to `SlotInfo` |
| `fabricpc/builder/graph_builder.py` | Propagate `is_variance_scalable` from SlotSpec → SlotInfo |
| `fabricpc/core/mupc.py` | Redesign scaling algorithm for per-edge, per-slot scaling |
| `fabricpc/nodes/linear_residual.py` | **New file**: `LinearResidual` node |
| `fabricpc/nodes/skip_connection.py` | Update slot to `is_variance_scalable=False` |
| `fabricpc/nodes/__init__.py` | Export `LinearResidual` |
| `examples/mupc_demo.py` | Add LinearResidual variant alongside existing SkipConnection demo |

## Step 1: Add `is_variance_scalable` to SlotSpec

**File:** `fabricpc/nodes/base.py:51-56`

```python
@dataclass(frozen=True)
class SlotSpec:
    name: str
    is_multi_input: bool
    is_variance_scalable: bool = True  # NEW — False for skip/residual slots
```

Default `True` preserves backward compatibility for all existing nodes (Linear, IdentityNode, TransformerBlock, etc.).

## Step 2: Add `is_variance_scalable` to SlotInfo

**File:** `fabricpc/core/types.py:19-26`

```python
@dataclass(frozen=True)
class SlotInfo:
    name: str
    parent_node: str
    is_multi_input: bool
    is_variance_scalable: bool  # NEW — propagated from SlotSpec
    in_neighbors: Tuple[str, ...]
```

## Step 3: Propagate in graph_builder

**File:** `fabricpc/builder/graph_builder.py:32-55` — `_build_slots()`

Add `is_variance_scalable=slot_spec.is_variance_scalable` when constructing `SlotInfo` (line 48).

## Step 4: Update SkipConnection slot

**File:** `fabricpc/nodes/skip_connection.py:74-76`

```python
@staticmethod
def get_slots():
    return {"in": SlotSpec(name="in", is_multi_input=True, is_variance_scalable=False)}
```

This makes the SkipConnection's behavior explicit at the slot level, consistent with the new per-slot model. The node-level `apply_variance_scaling=False` is kept for backward compat.

## Step 5: Create LinearResidual node

**New file:** `fabricpc/nodes/linear_residual.py`

```python
class LinearResidual(FlattenInputMixin, NodeBase):
    """
    Residual node: z_mu = activation(W @ x_in + b) + x_skip
    
    Two slots:
      "in"   — variance-scalable, has weight matrix (the residual/transform path)
      "skip" — NOT variance-scalable, identity pass-through (the skip path)
    
    Combines Linear transform + residual sum in one PC node,
    halving the graph depth compared to Linear + SkipConnection.
    """
    
    apply_variance_scaling = True  # Per-slot control via SlotSpec
    
    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec(name="in", is_multi_input=True, is_variance_scalable=True),
            "skip": SlotSpec(name="skip", is_multi_input=True, is_variance_scalable=False),
        }
    
    @staticmethod
    def get_weight_fan_in(source_shape, config):
        # Fan-in for the "in" slot weight matrix (same logic as Linear).
        # Skip slot edges won't call this — they're non-scalable.
        if config.get("flatten_input", False):
            return int(np.prod(source_shape))
        return source_shape[-1]
    
    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        # Weight matrix only for "in" slot edges (skip edges are identity)
        in_slot_shapes = {k: v for k, v in input_shapes.items() if ":in" in k}
        skip_slot_shapes = {k: v for k, v in input_shapes.items() if ":skip" in k}
        
        # Initialize weights for "in" slot edges (same as Linear)
        flatten_input = config.get("flatten_input", False)
        weights = {}
        for edge_key, in_shape in in_slot_shapes.items():
            if flatten_input:
                weight_shape = (int(np.prod(in_shape)), int(np.prod(node_shape)))
            else:
                weight_shape = (in_shape[-1], node_shape[-1])
            weights[edge_key] = initialize(key, weight_shape, weight_init)
            key, = jax.random.split(key, 1)  # fresh key per edge
        
        # Bias
        use_bias = config.get("use_bias", True)
        biases = {}
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            biases["b"] = jnp.zeros(bias_shape)
        
        return NodeParams(weights=weights, biases=biases)
    
    @staticmethod
    def forward(params, inputs, state, node_info):
        # Separate inputs by slot (edge keys contain ":in" or ":skip")
        in_inputs = {k: v for k, v in inputs.items() if ":in" in k}
        skip_inputs = {k: v for k, v in inputs.items() if ":skip" in k}
        
        # Linear transform on "in" slot (same as Linear.forward)
        flatten_input = node_info.node_config.get("flatten_input", False)
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        
        if flatten_input:
            pre_activation = FlattenInputMixin.compute_linear(
                in_inputs, params.weights, batch_size, out_shape
            )
        else:
            pre_activation = jnp.zeros((batch_size,) + out_shape)
            for edge_key, x in in_inputs.items():
                pre_activation += jnp.matmul(x, params.weights[edge_key])
        
        if "b" in params.biases:
            pre_activation += params.biases["b"]
        
        # Apply activation to the transform path
        activation = node_info.activation
        transformed = type(activation).forward(pre_activation, activation.config)
        
        # Sum skip inputs (identity, no transform)
        skip_sum = None
        for x in skip_inputs.values():
            skip_sum = x if skip_sum is None else skip_sum + x
        
        # Residual sum: z_mu = activation(Wx + b) + x_skip
        z_mu = transformed + skip_sum if skip_sum is not None else transformed
        
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state
```

## Step 6: Redesign muPC algorithm

**File:** `fabricpc/core/mupc.py`

### 6a. Replace `_count_skip_depth` with `_count_unscalable_depth`

Instead of checking `issubclass(node_class, SkipConnection)`, check if the node has **any slot with `is_variance_scalable=False`**:

```python
def _count_unscalable_depth(nodes, edges, node_order):
    """Count nodes with non-scalable slots along the longest path (residual depth)."""
    skip_counts = {}
    
    for node_name in node_order:
        node = nodes[node_name]
        node_info = node.node_info
        
        if node_info.in_degree == 0:
            skip_counts[node_name] = 0
            continue
        
        max_pred = max(skip_counts.get(edges[ek].source, 0) for ek in node_info.in_edges)
        
        # Node has a non-scalable slot → it's a residual merge point
        has_unscalable = any(
            not s.is_variance_scalable for s in node_info.slots.values()
        )
        # Legacy fallback: node-level apply_variance_scaling=False
        if not getattr(node_info.node_class, 'apply_variance_scaling', True):
            has_unscalable = True
        
        skip_counts[node_name] = max_pred + (1 if has_unscalable else 0)
    
    return max(skip_counts.values()) if skip_counts else 0
```

This generalizes to LinearResidual, SkipConnection, Mlp2ResidualNode, and any future node with a non-scalable slot.

### 6b. Edge-based scaling in `compute_mupc_scalings`

The core change: per-edge scaling decisions based on the **target slot's `is_variance_scalable`** and the **slot's in-degree** (not the node's total in-degree).

```python
for edge_key in node_info.in_edges:
    edge_info = edges[edge_key]
    slot_name = edge_info.slot
    slot_info = node_info.slots[slot_name]
    
    if not slot_info.is_variance_scalable:
        # Non-scalable slot: identity pass-through
        forward_scale[edge_key] = 1.0
        topdown_grad_scale[edge_key] = 1.0
        weight_grad_scale[edge_key] = 1.0
    else:
        # Scalable slot: full muPC formula
        K_slot = len(slot_info.in_neighbors)   # in-degree of THIS slot
        source_shape = nodes[edge_info.source].node_info.shape
        fan_in = node_class.get_weight_fan_in(source_shape, node_config)
        
        if is_output:
            a = gain / (fan_in * math.sqrt(K_slot * L))
        else:
            a = gain / math.sqrt(fan_in * K_slot * L)
        
        forward_scale[edge_key] = a
        topdown_grad_scale[edge_key] = a * jac_gain
        weight_grad_scale[edge_key] = 1.0
```

Key differences from current code:
- **K_slot** replaces K (node-level in-degree) — uses `len(slot_info.in_neighbors)` for the specific slot
- **Per-edge slot check** replaces node-level `apply_variance_scaling` check
- **Legacy fallback**: if `apply_variance_scaling=False` on the node, ALL edges still get scale 1.0

### 6c. Remove SkipConnection import

The muPC module will no longer need `from fabricpc.nodes.skip_connection import SkipConnection` since depth counting uses slot metadata instead of isinstance checks.

## Step 7: Export LinearResidual

**File:** `fabricpc/nodes/__init__.py`

Add `from fabricpc.nodes.linear_residual import LinearResidual` and add to `__all__`.

## Step 8: Update mupc_demo.py

Add a `build_fc_resnet_linear_residual()` function alongside the existing `build_fc_resnet()` to demonstrate both approaches. The LinearResidual version:

```python
for i in range(num_blocks):
    with GraphNamespace(f"block{i}"):
        res = LinearResidual(
            shape=(hidden_dim,),
            activation=TanhActivation(),
            weight_init=mupc_init,
            name="res",
        )
    all_nodes.append(res)
    all_edges.extend([
        Edge(source=prev, target=res.slot("in")),    # transform path
        Edge(source=prev, target=res.slot("skip")),   # skip path
    ])
    prev = res
```

N blocks → N+2 nodes (vs 2N+2), 2N+1 edges (vs 3N+2). Graph depth halved.

## Scaling Factor Accounting Summary

| Factor | Source | Applied to |
|--------|--------|-----------|
| **fan_in** (Kaiming) | `node_class.get_weight_fan_in(source_shape, config)` | Scalable in-edges only |
| **K_slot** (in-degree) | `len(slot_info.in_neighbors)` — per-slot, not per-node | Scalable in-edges only |
| **gain** (activation variance) | `activation.variance_gain()` | Scalable in-edges only |
| **jac_gain** (Jacobian compensation) | `activation.jacobian_gain()` | Top-down grad of scalable edges |
| **L** (residual depth) | Count of nodes with any `is_variance_scalable=False` slot along longest path | Scalable in-edges only |
| Non-scalable edges | — | Scale 1.0 (forward, gradient, weight) |

## Verification

1. **Unit test**: Build a LinearResidual-based FC-ResNet, verify scaling factors:
   - "in" edges: `a = gain / sqrt(fan_in * K_slot * L)` with K_slot = 1 (one edge per "in" slot)
   - "skip" edges: `a = 1.0`
   - L = num_blocks (one LinearResidual per block, each has an unscalable slot)

2. **Backward compat**: Build the same network with existing SkipConnection pattern, verify identical scaling factors as before.

3. **Run mupc_demo.py** with both build functions, verify:
   - Both reach comparable accuracy at depth 16/32
   - LinearResidual version has half the nodes
   - Training time should improve (fewer PC nodes = fewer latent updates)

4. **Run existing tests**: `pytest tests/` to verify nothing breaks.
