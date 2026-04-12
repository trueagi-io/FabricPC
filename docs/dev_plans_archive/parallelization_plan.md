# Node Parallelization Plan for FabricPC

## Problem Statement

Nodes in predictive coding are **locally independent**, but the current implementation processes them **sequentially in Python for-loops**:

- `fabricpc/core/inference.py:56-112` - inference step iterates over nodes sequentially
- `fabricpc/graph/graph_net.py:73-93` - weight gradient computation iterates sequentially

This creates a scaling bottleneck.
---

### Issues Identified

1. **vmap over heterogeneous nodes is impossible**
   - `GraphState.nodes` is `Dict[str, NodeState]` where each node has different shapes
   - Example: `Linear(784,)`, `Linear(256,)`, `TransformerBlock(128, 512)`
   - `jax.vmap` requires homogeneous arrays - cannot vmap over a dict of differently-shaped tensors
   - We can vmap over **groups of homogeneous nodes** (same type and shape), but this requires restructuring.
   - In deep transformer models we can have many identical nodes (e.g., multiple attention blocks) that can be grouped.

2. **pmap over individual nodes creates massive overhead**
   - Each node depends on outputs from previous topological levels
   - Distributing nodes to different devices requires synchronization after every level
   - Communication overhead would dominate any parallel gains

3. **Gradient accumulation has reduction dependencies**
   - From `inference.py:79-84`:
     ```python
     for edge_key, grad in inedge_grads.items():
         source_name = structure.edges[edge_key].source
         latent_grad = state.nodes[source_name].latent_grad
         latent_grad = latent_grad + grad  # Accumulation dependency
     ```
   - Gradients flow backward locally to direct pre-synaptic nodes, requiring careful handling

---

## Recommended Approach: Group-Based Parallelization

### Core Concept

Similar nodes can be vmapped and typical graphs have chains of similar nodes, for examples transformers and resnets.

There are three phases to predictive coding:
1. Latent state initialization - random initializers be parallelized. Feedforward initializer has no way to parallelize due to sequential dependencies.
Current:    for node in nodes: process(node)           # O(n) sequential
Proposed:   
            shard the nodes and their initializers
            vmap(initializer process)(nodes)                        # O(n/p) parallel
            unshard the node states
            run feedforward initializer as is (sequential)          # O(n) sequential

2. Inference - nodes can be parallelized - they have dependencies only on inputs. Must stack the inputs and accumulate gradients before updating latent state by gradient descent.
Current:    for inference_step in range t_steps:
               for node in nodes: process(node)           # O(n) sequential
Proposed:   
            shard the nodes and their collections of inputs
            for inference_step in steps: vmap(process)(nodes)  # O(n/p) parallel
            unshard the outputs and vsum the gradients

3. Weight learning - nodes can be parallelized - no need to stack the inputs because gradient depends only on local node state. Optimizer has no dependencies between nodes.
Current:    for node in nodes: process(node)           # O(n) sequential
Proposed:   
            shard the nodes and their local states
            vmap(process)(nodes)                        # O(n/p) parallel
            apply optimizer locally per node


### Example

For a 3 block transformer with 1 multi-head attention node and 2 MLP nodes per block:
```
Group 0: [embedding node]
Group 1: [MHA node 1, MHA node 2, MHA node 3, MHA aux node 1, MHA aux node 2, MHA aux node 3]]  # auxilary nodes are identity nodes for residual connections to maintain coherent energy flow.
Group 2: [MLP node 1a, MLP node 2a, MLP node 3a, MLP aux node 1a, MLP aux node 2a, MLP aux node 3a]
Group 3: [MLP node 1b, MLP node 2b, MLP node 3b, MLP aux node 1b, MLP aux node 2b, MLP aux node 3b]
Group 4: [output node]
```


---

## Design Decisions with Tradeoff Analysis

### Decision 1: Input Gathering Strategy

**Question:** How to gather inputs when vmapping nodes with different input sources?

#### Option A: Pre-stack inputs before vmap

```python
# Gather inputs for all nodes in group BEFORE vmap
stacked_inputs = []
for node_name in group.node_names:
    node_info = structure.nodes[node_name]
    inputs = gather_inputs(node_info, structure, state)
    stacked_inputs.append(inputs)
stacked_inputs = jnp.stack(stacked_inputs)  # (num_nodes, batch, features)

# vmap receives pre-gathered inputs
vmap(process)(stacked_params, stacked_inputs, stacked_states)
```

#### Option B: Pass full state, index inside vmap

```python
# vmap receives full state, each instance looks up its own inputs
def process_with_lookup(node_idx, params, full_state, structure):
    node_name = group.node_names[node_idx]
    inputs = gather_inputs(structure.nodes[node_name], structure, full_state)
    return node_forward(params, inputs, ...)

vmap(process_with_lookup)(node_indices, stacked_params, full_state, structure)
```

#### Tradeoff Analysis

| Aspect | Pre-stack (A) | Index-based (B) |
|--------|---------------|-----------------|
| **Speed** | Faster - input gathering once outside vmap | Slower - dictionary lookups not vectorized |
| **Memory** | Lower peak - only stores inputs for current group | Higher - full state passed to each vmap instance |
| **Maintainability** | Cleaner separation of concerns | Complex - mixing gathering and computation |
| **JIT behavior** | Better - concrete shapes before vmap | Potential retracing from dynamic indexing |

**Final Decision: Pre-stack inputs (Option A)**

Rationale: The existing `gather_inputs()` function handles the logic. We call it for each node in the group and stack results before vmapping. This is faster, cleaner, and has better JIT behavior.

---

### Decision 2: vmap Scope

**Question:** Should vmap cover the entire inference loop (all T steps) or just a single step?

#### Option A: vmap entire inference loop

```python
def inference_loop_vmapped(params, initial_state, T, eta):
    def body_fn(state, _):
        new_states = vmap(node_forward)(...)  # All nodes parallel
        state = accumulate_and_update(new_states, state, eta)
        return state, None

    final_state, _ = jax.lax.scan(body_fn, initial_state, None, length=T)
    return final_state
```

#### Option B: vmap single inference step, loop outside

```python
def inference_step_parallel(params, state, clamps, structure, eta):
    # vmap over nodes for this single step
    for group in groups:
        new_states = vmap(node_forward)(...)
    state = accumulate_and_update(new_states, state, eta)
    return state

# fori_loop over T steps (matches current structure)
final_state = jax.lax.fori_loop(0, T, lambda t, s: inference_step_parallel(...), initial_state)
```

#### Tradeoff Analysis

| Aspect | Entire loop (A) | Single step (B) |
|--------|-----------------|-----------------|
| **Speed** | Similar - JAX compiles scan efficiently | Similar - fori_loop also compiles well |
| **Memory** | Higher if scan stores intermediates | Lower - only current state in memory |
| **Maintainability** | Complex - harder to debug/inspect | Simpler - can test single step, add logging |
| **Flexibility** | Harder to add early stopping | Easy to modify iteration logic |

**Final Decision: vmap single step (Option B)**

Rationale: Maintains existing `jax.lax.fori_loop` structure, easier to debug, and allows inspection of intermediate states. Performance is equivalent.

---

### Decision 3: Gradient Accumulation Strategy

**Question:** How to handle gradient accumulation when multiple nodes send gradients to the same source?

#### Option A: Sum after unsharding

```python
# vmap returns all gradients separately
all_node_states, all_inedge_grads = vmap(node_forward)(...)
# all_inedge_grads: (num_nodes, ...) - one gradient dict per node

# After vmap: accumulate gradients per pre-synaptic node
accumulated_grads = {}
for node_idx, node_name in enumerate(group.node_names):
    for edge_key, grad in all_inedge_grads[node_idx].items():
        source_name = structure.edges[edge_key].source
        if source_name not in accumulated_grads:
            accumulated_grads[source_name] = grad
        else:
            accumulated_grads[source_name] += grad
```

#### Option B: Reduce within vmap

```python
# Use JAX scatter/segment operations inside vmap
def node_forward_with_reduce(node_idx, params, inputs, grad_buffer):
    new_state, inedge_grads = forward(...)
    # Atomic-like accumulation into shared buffer
    for edge_key, grad in inedge_grads.items():
        source_idx = edge_to_source_idx[edge_key]
        grad_buffer = grad_buffer.at[source_idx].add(grad)
    return new_state, grad_buffer

vmap(node_forward_with_reduce, out_axes=(0, None))(...)  # Reduce grad_buffer
```

#### Tradeoff Analysis

| Aspect | Sum after (A) | Reduce within (B) |
|--------|---------------|-------------------|
| **Speed** | Minor Python loop overhead (num_nodes iterations, not batch_size) | Potentially faster but JAX scatter has overhead |
| **Memory** | Higher - stores all gradients before summing | Lower - accumulates in place |
| **Maintainability** | Much simpler - clear separation | Complex - scatter ops, index management |
| **Correctness** | Easy to verify | Tricky index mapping, easy to introduce bugs |

**Final Decision: Sum after unsharding (Option A)**

Rationale: Simplicity gain is significant. Performance difference is negligible because:
- Gradient accumulation is O(num_nodes) not O(batch_size)
- The heavy computation (node forward passes) is already vmapped
- Python loop over ~10-100 nodes is not a bottleneck

---

### Decision 4: Shape Handling

**Question:** How to handle nodes of the same type but different shapes?

#### Options
- **A) Require matching shapes** - Only group nodes with identical shapes
- **B) Padding/masking** - Pad to max shape, use masks for valid positions

**Final Decision: Require matching shapes (Option A)**

Rationale: Simpler implementation, no memory overhead from padding. Typical architectures (transformers, resnets) naturally have identical shapes across blocks.

---

### Design Decisions Summary:                                                                                                                                                                                                          
                                                                                                                                                                                                                                           
  | Decision              | Final Choice            | Key Rationale                                        |                                                                                                                               
  |-----------------------|-------------------------|------------------------------------------------------|                                                                                                                               
  | Input gathering       | Pre-stack before vmap   | Faster, cleaner, better JIT behavior                 |                                                                                                                               
  | vmap scope            | Single inference step   | Simpler debugging, matches existing structure        |                                                                                                                               
  | Gradient accumulation | Sum after unsharding    | Much simpler, negligible performance difference      |                                                                                                                               
  | Shape handling        | Require matching shapes | Simpler, typical architectures have identical shapes |  
___

## Implementation Plan

### Phase 1: Node Grouping Infrastructure

**New file:** `fabricpc/graph/parallel.py`

```python
@dataclass
class NodeGroup:
    """Group of homogeneous nodes for batched execution."""
    node_names: List[str]
    node_type: str
    shape: Tuple[int, ...]  # All nodes must have this exact shape

def compute_node_groups(structure: GraphStructure) -> List[NodeGroup]:
    """
    Group nodes by (type, shape) for parallel execution.
    Returns groups ordered for processing.
    """
    groups = {}
    for node_name, node_info in structure.nodes.items():
        key = (node_info.node_type, node_info.shape)
        if key not in groups:
            groups[key] = []
        groups[key].append(node_name)

    return [NodeGroup(names, ntype, shape)
            for (ntype, shape), names in groups.items()]
```

**Modify:** `fabricpc/core/types.py`
- Add `node_groups: Tuple[NodeGroup, ...]` field to `GraphStructure`
- Compute groups in `from_config()` after building nodes

### Phase 2: State Stacking Utilities

**File:** `fabricpc/graph/parallel.py`

```python
def stack_states_for_group(state: GraphState, group: NodeGroup) -> StackedNodeState:
    """Stack node states into (num_nodes, batch, *shape) arrays."""
    z_latents = jnp.stack([state.nodes[n].z_latent for n in group.node_names])
    z_mus = jnp.stack([state.nodes[n].z_mu for n in group.node_names])
    # ... other fields
    return StackedNodeState(z_latents, z_mus, ...)

def stack_params_for_group(params: GraphParams, group: NodeGroup) -> StackedNodeParams:
    """Stack node params into (num_nodes, ...) arrays."""
    # Stack weights and biases for all nodes in group
    ...

def stack_inputs_for_group(
    group: NodeGroup,
    structure: GraphStructure,
    state: GraphState
) -> jnp.ndarray:
    """Pre-gather and stack inputs for all nodes in group."""
    inputs = []
    for node_name in group.node_names:
        node_info = structure.nodes[node_name]
        node_inputs = gather_inputs(node_info, structure, state)
        inputs.append(node_inputs)
    return jnp.stack(inputs)  # (num_nodes, batch, features)

def unstack_states_to_dict(
    stacked: StackedNodeState,
    group: NodeGroup,
    state: GraphState
) -> GraphState:
    """Unpack stacked results back into GraphState dict."""
    new_nodes = dict(state.nodes)
    for i, node_name in enumerate(group.node_names):
        new_nodes[node_name] = NodeState(
            z_latent=stacked.z_latents[i],
            z_mu=stacked.z_mus[i],
            # ... other fields
        )
    return state._replace(nodes=new_nodes)
```

### Phase 3: Parallel Inference

**New file:** `fabricpc/core/inference_parallel.py`

```python
def inference_step_parallel(
    params: GraphParams,
    state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    eta_infer: float,
) -> GraphState:
    """Single inference step with group-based parallelization."""

    # 1. Zero all gradients
    state = zero_all_gradients(state, structure)

    # 2. Process each group
    all_inedge_grads = {}  # Collect for accumulation

    for group in structure.node_groups:
        if len(group.node_names) == 1:
            # Single node - use standard path
            state, grads = process_single_node(group.node_names[0], params, state, structure)
            all_inedge_grads.update(grads)
        else:
            # Multiple nodes - vmap
            state, grads = process_group_vmap(group, params, state, structure)
            all_inedge_grads.update(grads)

    # 3. Accumulate gradients (sum after unsharding)
    state = accumulate_gradients(state, all_inedge_grads, structure)

    # 4. Update latent states
    state = update_latents(state, clamps, eta_infer, structure)

    return state

def process_group_vmap(
    group: NodeGroup,
    params: GraphParams,
    state: GraphState,
    structure: GraphStructure
) -> Tuple[GraphState, Dict]:
    """Process multiple nodes of same type in parallel using vmap."""

    # Pre-stack everything (Decision 1: pre-stack inputs)
    stacked_params = stack_params_for_group(params, group)
    stacked_states = stack_states_for_group(state, group)
    stacked_inputs = stack_inputs_for_group(group, structure, state)

    # Get node class (all nodes in group have same type)
    node_class = get_node_class(group.node_type)

    # vmap over the node dimension
    vmapped_forward = jax.vmap(node_class.forward_inference, in_axes=(0, 0, 0, None))
    new_stacked_states, stacked_grads = vmapped_forward(
        stacked_params, stacked_inputs, stacked_states, node_info_template
    )

    # Unstack results
    state = unstack_states_to_dict(new_stacked_states, group, state)

    # Return gradients for later accumulation (Decision 3: sum after)
    inedge_grads = unstack_gradients(stacked_grads, group, structure)

    return state, inedge_grads

def run_inference_parallel(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    infer_steps: int,
    eta_infer: float,
) -> GraphState:
    """Run inference for T steps (Decision 2: vmap single step, loop outside)."""

    def body_fn(t, state):
        return inference_step_parallel(params, state, clamps, structure, eta_infer)

    return jax.lax.fori_loop(0, infer_steps, body_fn, initial_state)
```

### Phase 4: Parallel Weight Learning

**Modify:** `fabricpc/graph/graph_net.py`

```python
def compute_local_weight_gradients_parallel(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """Compute weight gradients with group-based parallelization."""

    gradients = {}

    for group in structure.node_groups:
        # Skip terminal input nodes (no weights)
        if structure.nodes[group.node_names[0]].in_degree == 0:
            for node_name in group.node_names:
                gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        if len(group.node_names) == 1:
            # Single node - standard path
            gradients[group.node_names[0]] = compute_single_node_gradient(...)
        else:
            # Multiple nodes - vmap
            stacked_grads = compute_group_gradients_vmap(group, params, final_state, structure)
            for i, node_name in enumerate(group.node_names):
                gradients[node_name] = unstack_gradient(stacked_grads, i)

    return GraphParams(nodes=gradients)
```

### Phase 5: Integration

**Modify:** `fabricpc/training/train.py`
- Add `use_parallel: bool = True` parameter to `get_graph_param_gradient()`
- Use parallel versions when enabled

**Modify:** `fabricpc/training/multi_gpu.py`
- Integrate group-parallel inference into pmap training step
- Structure: `pmap(data shards)` → within each device: `vmap(nodes per group)`

---

## Critical Files Summary

| File | Changes |
|------|---------|
| `fabricpc/core/types.py` | Add `node_groups` to `GraphStructure` |
| `fabricpc/core/inference.py` | Keep existing (fallback) |
| `fabricpc/graph/graph_net.py` | Add `compute_local_weight_gradients_parallel()` |
| `fabricpc/training/train.py` | Add `use_parallel` flag |
| `fabricpc/training/multi_gpu.py` | Integrate parallel inference |

**New files:**
- `fabricpc/graph/parallel.py` - Grouping, stacking/unstacking utilities
- `fabricpc/core/inference_parallel.py` - Parallel inference implementation

---

## Expected Performance

| Architecture | Expected Speedup | Notes |
|--------------|------------------|-------|
| Deep transformer (12+ blocks) | 3-6x | Many identical nodes per group |
| ResNet-style | 2-4x | Repeated residual blocks |
| Wide MLP | 1.5-2x | Depends on layer homogeneity |
| Heterogeneous graph | 1.1-1.5x | Limited grouping opportunities |

---

## Testing Strategy

1. **Unit tests** for node grouping with various graph topologies
2. **Numerical equivalence** tests: parallel vs sequential must produce identical results
3. **Performance benchmarks** on graphs of varying depth and homogeneity
4. **Memory profiling** to verify no excessive overhead from stacking
5. **Multi-GPU tests** to verify pmap integration

---

## Implementation Order

1. **Phase 1** - Node grouping infrastructure (foundation)
2. **Phase 2** - State stacking utilities
3. **Phase 3** - Parallel inference (highest impact)
4. **Phase 4** - Parallel weight learning
5. **Phase 5** - Integration and testing