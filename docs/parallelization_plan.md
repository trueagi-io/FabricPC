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
   - Example: `LinearNode(784,)`, `LinearNode(256,)`, `TransformerBlockNode(128, 512)`
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
   - Gradients flow backward locally to direct source nodes, requiring careful handling

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


## Testing Strategy

1. **Unit tests** for level computation with various graph topologies
2. **Numerical equivalence** tests: parallel vs sequential should produce identical results
3. **Performance benchmarks** on graphs of varying width/depth
4. **Memory profiling** to measure overhead from stacking/padding
5. **Multi-GPU tests** to verify pmap integration