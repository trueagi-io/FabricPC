# Principled muPC Forward Scaling via Variance Propagation (completed 2026-04-11)

## Problem

`mupc_mnist_demo.py --num_hidden=20` produces 9.8% (chance) accuracy. With `--num_hidden=2` it achieves 91.6%. The root cause: the current depth-based scaling formula `a = 1/sqrt(fan_in * L)` uses a heuristic depth metric that doesn't properly model why variance grows in different graph topologies.

## Root Cause: Depth is a proxy for summation amplification

The jpc reference's `1/sqrt(L)` factor compensates for **variance amplification from skip-connection summation** in ResNets (`z = a*W*phi(z) + z_skip`). In a chain without skip connections, there's no summation amplification — the depth factor adds unnecessary damping that cascades, causing `Var ∝ 1/L!` (factorial collapse).

The principled fix: replace the heuristic depth metric with **variance propagation** through the graph.

## The Algorithm: Cumulative Variance Propagation

### How variance propagates per node type

| Node Type | Computation | Unscaled Var(z_mu) | Needs scaling? |
|-----------|------------|---------------------|----------------|
| Source (in_degree=0) | clamped data | 1.0 (assumed) | No |
| **IdentityNode** (K edges, no weights) | `z_mu = sum_k(x_k)` | `sum_k(var_k)` — grows with K | **Yes**: `1/sqrt(K)` per edge |
| **Linear** (K edges, W~N(0,1)) | `z_mu = act(sum_k(W_k @ (a_k * x_k)))` | `sum_k(fan_in_k * a_k² * var_k)` | Yes: `1/sqrt(fan_in_k * K)` per edge |

### Design rationale: scaling at IdentityNode (not just downstream)

**Why IdentityNode inputs should be scaled, not just tracked:**

In predictive coding, every node has an energy functional `E = 0.5 * ||z_latent - z_mu||²`. The prediction error `ε = z_latent - z_mu` drives inference. If IdentityNode's `z_mu = sum(x_k)` has variance K (from K inputs each with O(1) variance), then:
- Prediction errors at IdentityNode are O(sqrt(K)), not O(1)
- The gradient signal from IdentityNode to presynaptic nodes is amplified by K
- This creates an asymmetric inference landscape — nodes near high-fan-in junctions receive disproportionately large gradients

By scaling IdentityNode inputs to `z_mu = (1/sqrt(K)) * sum(x_k)`, we maintain O(1) prediction errors at **every** node in the graph, not just at weighted nodes. This keeps the inference dynamics uniform and well-conditioned.

The alternative — tracking variance downstream and compensating at the next weighted node — produces the same net scaling on the forward signal, but leaves IdentityNode's energy function unbalanced. Since PCN inference relies on local prediction errors at every node, maintaining O(1) errors everywhere is the principled choice.

**Mathematical equivalence with downstream compensation:**

Both approaches produce the same effective scaling from source to the next weighted node. For the path `source(var=1) → IdentityNode(K=2) → Linear(fan_in=N)`:

- **Scale at IdentityNode**: id gets `a=1/sqrt(2)`, output var=1.0. Linear gets `a=1/sqrt(N*1)=1/sqrt(N)`. Net: `1/sqrt(2) * 1/sqrt(N) = 1/sqrt(2N)`.
- **Compensate downstream**: id unscaled, output var=2.0. Linear gets `a=1/sqrt(N*2)=1/sqrt(2N)`. Net: `1/sqrt(2N)`.

Same scaling factor, but the first approach has O(1) prediction errors at IdentityNode.

### Per-edge forward scale formula

For **any node** with K input edges, the forward scale per edge is:

```
a_k = 1 / sqrt(fan_in_k * K)
```

Where:
- `fan_in_k` = `node_class.get_weight_fan_in(source_shape, node_config)` — each node type returns its appropriate fan_in. Linear returns its weight-matrix fan_in (Kaiming convention). IdentityNode overrides to return 1 (no weight matrix, just summation).
- `K` = number of input edges (in_degree)

**No separate code paths.** The same formula and same `get_weight_fan_in()` call is used for every node type. The fan_in method is the single point of customization — node types encode their variance contribution through it.

For output nodes (`include_output=True`), maintaining the muPC `O(1/N)` convention:

```
a_k = 1 / (fan_in_k * sqrt(K))
```

### Algorithm

Since we scale ALL non-source nodes (including IdentityNode), every node's output has O(1) variance. This eliminates the need for cumulative variance tracking. The algorithm is simply:

```python
for each non-source node with K in-edges:
    for each edge_k:
        fan_in_k = node_class.get_weight_fan_in(source_shape, config)
        a_k = 1 / sqrt(fan_in_k * K)
```

The in-degree K is the only topology-dependent factor. This works because:
- Scaling at every node (including IdentityNode) maintains O(1) output variance everywhere
- Each node only needs to know its own fan_in and in-degree — no global graph analysis needed

### Verification: reduces to known formulas

| Topology | Node | K | fan_in | a | Matches |
|----------|------|---|--------|---|---------|
| Plain chain (single edges) | Hidden Linear | 1 | N | `1/sqrt(N)` | Kaiming |
| ResNet IdentityNode junction | IdentityNode(2 edges) | 2 | 1 | `1/sqrt(2)` | Halves signal power per branch |
| ResNet hidden after junction | Linear(1 edge) | 1 | N | `1/sqrt(N)` | Kaiming (junction already normalized) |
| Linear with 2 input edges | Linear(skip) | 2 | N | `1/sqrt(2N)` | Like jpc's `1/sqrt(N*L)` with L=2 |
| First hidden (from 784-dim input) | Linear | 1 | 784 | `1/sqrt(784)` | Matches jpc `a_1=1/sqrt(D)` |
| Output node | Linear | 1 | N | `1/N` | Matches jpc `a_L=1/N` |

---

## Implementation Steps

### Step 0: Override `get_weight_fan_in()` in `fabricpc/nodes/identity.py`

IdentityNode currently inherits `get_weight_fan_in()` from NodeBase, which returns `source_shape[-1]` (or `prod(source_shape)` when `flatten_input=True`). This is wrong for IdentityNode — it has no weight matrix, so there is no fan-in from weights.

Override to return 1, with a comment explaining the design pattern:

```python
@staticmethod
def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
    """Return fan_in for muPC scaling.

    IdentityNode has no weight matrix — inputs are summed directly.
    Returning fan_in=1 means the unified scaling formula a=1/sqrt(fan_in*K)
    reduces to a=1/sqrt(K), which compensates only for multi-edge
    summation variance amplification.
    """
    return 1
```

This lets `compute_mupc_scalings()` use one code path: `fan_in = node_class.get_weight_fan_in(...)` for every node type.

### Step 1: Rewrite `compute_mupc_scalings()` in `fabricpc/core/mupc.py`

Remove depth metric usage entirely. Remove `has_weights()` check. Use a single unified formula for all non-source nodes: `a = 1/sqrt(fan_in * K)` where `fan_in` comes from `get_weight_fan_in()` and `K = in_degree`.

```python
def compute_mupc_scalings(nodes, edges, config, node_order):
    output_nodes = {name for name, n in nodes.items() if n.node_info.out_degree == 0}
    scalings = {}

    for node_name in node_order:
        node = nodes[node_name]
        node_info = node.node_info
        node_class = node_info.node_class

        # Terminal input nodes (in_degree=0): no scaling needed
        if node_info.in_degree == 0:
            scalings[node_name] = None
            continue

        # Output nodes: skip unless include_output
        is_output = node_name in output_nodes
        if is_output and not config.include_output:
            scalings[node_name] = None
            continue

        K = node_info.in_degree
        node_config = node_info.node_config

        forward_scale = {}
        for edge_key in node_info.in_edges:
            source_shape = nodes[edges[edge_key].source].node_info.shape

            # Unified fan_in from node class. Weighted nodes return their
            # weight-matrix fan_in (Kaiming convention); weightless nodes
            # (e.g. IdentityNode) return 1.
            fan_in = node_class.get_weight_fan_in(source_shape, node_config)

            if is_output:
                a = 1.0 / (fan_in * math.sqrt(K))
            else:
                a = 1.0 / math.sqrt(fan_in * K)

            forward_scale[edge_key] = a

        scalings[node_name] = MuPCScalingFactors(
            forward_scale=forward_scale,
            self_grad_scale=1.0,
            topdown_grad_scale={ek: 1.0 for ek in node_info.in_edges},
            weight_grad_scale={ek: 1.0 for ek in node_info.in_edges},
        )

    return scalings
```

Update `MuPCConfig`:
```python
@dataclass(frozen=True)
class MuPCConfig:
    include_output: bool = False
    terminal_input_variance: float = 1.0  # Assumed variance of source node outputs
    # Deprecated: kept for backward compatibility
    depth_metric: Optional[Any] = None
    min_depth: Optional[int] = None
```

Emit `DeprecationWarning` in `__post_init__` if `depth_metric` is provided.

### Step 2: Pass `node_order` in `fabricpc/builder/graph_builder.py`

Line 189, change:
```python
mupc_scalings = compute_mupc_scalings(finalized_nodes, edge_infos, scaling)
```
to:
```python
mupc_scalings = compute_mupc_scalings(finalized_nodes, edge_infos, scaling, node_order)
```

### Step 3: Update `examples/mupc_mnist_demo.py`

- Remove `ShortestPathDepth` import and usage
- Use `MuPCConfig(include_output=True)` (no depth_metric needed)
- Add `--infer_steps` CLI arg (default: `max(20, 3 * (num_hidden + 2))`)
- Add `--eta_infer` CLI arg (default: 0.1)

### Step 4: Update tests in `tests/test_mupc.py`

**New test class `TestVariancePropagation`:**
1. `test_plain_chain_no_depth_factor` — x→h→y: a=1/sqrt(fan_in), no depth dependency
2. `test_identity_node_gets_scaling` — IdentityNode with K=2 in-edges gets a=1/sqrt(2) per edge
3. `test_multi_edge_linear` — Linear with K=2 edges: a_k=1/sqrt(fan_in*2) per edge
4. `test_deep_chain_uniform_scaling` — 20-layer chain: all hidden nodes get same a=1/sqrt(fan_in)
5. `test_output_with_include_output` — output node: a=1/fan_in for K=1

**Updated existing tests:**
- `test_hidden_forward_scale_formula` — change expected from `1/sqrt(fan_in*depth)` to `1/sqrt(fan_in*K)`
- `test_source_node_gets_no_scaling` — unchanged
- `test_output_node_has_no_scaling` — unchanged
- Remove/update depth metric references in muPC integration tests
- IdentityNode exclusion tests become IdentityNode scaling tests

**Backward compatibility test:**
- `MuPCConfig(depth_metric=ShortestPathDepth())` emits `DeprecationWarning`, works correctly

### Step 5: Create diagnostic script `scripts/diagnose_deep_mupc.py`

Print per-layer forward scales for a 20-layer chain and verify:
- All hidden nodes get `a = 1/sqrt(fan_in)` (no depth factor)
- Run one training batch, print z_latent/z_mu norms per layer
- Verify energy is non-zero and weight gradients are non-zero

### Step 6: Run and verify

1. `python -m pytest tests/test_mupc.py -v` — all tests pass
2. `python examples/mupc_mnist_demo.py --num_hidden 2 --num_epochs 10` — still ~91%
3. `python examples/mupc_mnist_demo.py --num_hidden 20 --hidden_dim 64 --num_epochs 1` — above chance
4. Delete `scripts/diagnose_deep_mupc.py` if no longer needed

---

## Files to Modify

| File | Change |
|------|--------|
| `fabricpc/nodes/identity.py` | Override `get_weight_fan_in()` to return 1 (no weight matrix) |
| `fabricpc/core/mupc.py` | Remove depth metric usage; remove `has_weights()` check; rewrite `compute_mupc_scalings()` with unified `a=1/sqrt(fan_in*K)` formula; update `MuPCConfig` (add `terminal_input_variance`, deprecate `depth_metric`/`min_depth`) |
| `fabricpc/builder/graph_builder.py` | Pass `node_order` to `compute_mupc_scalings()` (1-line change) |
| `examples/mupc_mnist_demo.py` | Remove depth_metric; add `--infer_steps`/`--eta_infer` args |
| `tests/test_mupc.py` | New `TestVariancePropagation` class; update existing formula tests; backward-compat test |
| `scripts/diagnose_deep_mupc.py` | **New file**: temporary diagnostic for verifying the fix |

No changes needed to:
- `fabricpc/nodes/base.py` — `_apply_forward_scaling()` mechanism unchanged; `get_weight_fan_in()` base default unchanged
- `fabricpc/nodes/linear.py` — inherits base `get_weight_fan_in()`, which already returns correct fan_in
- `fabricpc/core/depth_metric.py` — kept intact, just decoupled from muPC
- `fabricpc/core/inference.py` — inference loop unchanged
- `fabricpc/graph/state_initializer.py` — feedforward init unchanged

# Completed Implementation

Core formula change: a = 1/sqrt(fan_in * K)                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                           
  Replaced the heuristic depth-based scaling a = 1/sqrt(fan_in * L) with a principled formula based on graph topology:                                                                                                                                                                                     
  - fan_in: from get_weight_fan_in() per node type                                                                                                                                                                                                                                                         
  - K: in-degree (number of input edges)                                                                                                                                                                                                                                                                   
  - Output nodes: a = 1/(fan_in * sqrt(K))                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                           
  Files modified:                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                           
  | File                              | Change                                                                                                                                            |                                                                                                                
  |-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|                                                                                                                
  | fabricpc/nodes/identity.py        | Added get_weight_fan_in() override returning 1 (no weight matrix)                                                                                 |                                                                                                                
  | fabricpc/core/mupc.py             | Rewrote compute_mupc_scalings() — removed depth metric and has_weights() checks; unified formula; deprecated depth_metric/min_depth on MuPCConfig |                                                                                                                
  | fabricpc/builder/graph_builder.py | Pass node_order to compute_mupc_scalings()                                                                                                        |                                                                                                                
  | examples/mupc_mnist_demo.py       | Removed ShortestPathDepth, added --infer_steps/--eta_infer CLI args                                                                               |                                                                                                                
  | tests/test_mupc.py                | New TestVariancePropagation (5 tests), TestBackwardCompatibility (3 tests), updated TestIdentityNodeScaling                                       |                                                                                                                
  | scripts/diagnose_deep_mupc.py     | New diagnostic script                                                                                                                             |                                                                                                                
                                                                                                                                                                                                                                                                                                           
  Test results:                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                           
  - 40/40 muPC tests pass                                                                                                                                                                                                                                                                                  
  - 135/136 full suite pass (1 pre-existing failure in test_storkey_hopfield.py)                                                                                                                                                                                                                           
  - 2-layer MNIST: 91.67% (matches prior behavior)                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                           
  Deep network status:                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                           
  The scaling fix eliminated factorial variance collapse. The 20-layer network now has non-zero energy (107.4 decreasing to 22.9 over 5 epochs) and non-zero gradients at all layers. However, accuracy remains at chance — deep PCN training has additional challenges beyond forward scaling (inference convergence, ReLU dying neurons through many layers). This is a separate research problem from the scaling bug.
  
# Bug fixes and revisions

  Root cause and fix                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                           
  The top-down gradient in forward_and_latent_grads() was missing the forward scaling factor a due to applying scaling outside the value_and_grad closure. The autodiff computed dE/d(a*x) but the presynaptic node needs dE/dx = a * dE/d(a*x). Setting topdown_grad_scale = a (instead of the placeholder 1.0) restores the chain rule factor.
                                                                                                                                                                                                                                                                                                           
  Results                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
  | Config              | Before fix    | After fix |                                                                                                                                                                                                                                                      
  |---------------------|---------------|-----------|                                                                                                                                                                                                                                                      
  | 2 hidden, 10 epochs | 91.6%         | 94.1%     |                                                                                                                                                                                                                                                      
  | 20 hidden, 5 epochs | ~10% (chance) | 48.7%     |                                                                                                                                                                                                                                                      

  The 20-layer plain chain still doesn't reach 93% because ReLU halves variance at each layer (Var(ReLU(z)) = 0.5 * Var(z)), causing activations to decay exponentially through depth. 
  ReLU variance problem that would require either a gain factor in the scaling (He-style a = sqrt(2)/sqrt(fan_in))
  
  Changed demo from relu to tanh activation --> allowing greater depth

  | Config               | Accuracy |                                                                                                                                                                                                                                                      
  |----------------------|----------|                                                                                                                                                                                                                                                      
  | 8 hidden, 4 epochs   | 91.6%    |
  | 32 hidden, 4 epochs  | 56.4%    |
  | 64 hidden, 4 epochs  | 21.0%    |

 Root Cause                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                           
 The muPC forward scaling formula a = 1/sqrt(fan_in * K) compensates for fan-in variance amplification but does NOT compensate for activation-induced variance contraction. With tanh: Var(tanh(z)) ≈ 0.39 * Var(z) for z ~ N(0,1). Over L layers, forward activation variance decays as ~0.39^L, and      
 gradient magnitudes decay similarly through the backward Jacobian chain (since tanh'(z) < 1).                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                           
 The fix: add Kaiming-style activation gain to the scaling formula:                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                           
 a = gain / sqrt(fan_in * K)          

## Apply activation variance gain corrections to muPC scaling formula

Add a @staticmethod method variance_gain(config=None) -> float to ActivationBase (default returns 1.0) and override in each subclass:                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                           
     | Activation          | Gain                | Formula                    |                                                                                                                                                                                                                            
     |---------------------|---------------------|----------------------------|                                                                                                                                                                                                                            
     | IdentityActivation  | 1.0                 | No contraction             |                                                                                                                                                                                                                            
     | TanhActivation      | sqrt(5/3) ≈ 1.291   | PyTorch Kaiming convention |                                                                                                                                                                                                                            
     | ReLUActivation      | sqrt(2) ≈ 1.414     | Half the units zeroed      |                                                                                                                                                                                                                            
     | LeakyReLUActivation | sqrt(2/(1+alpha^2)) | Parametric                 |                                                                                                                                                                                                                            
     | SigmoidActivation   | 1.0                 | Conservative default       |                                                                                                                                                                                                                            
     | GeluActivation      | sqrt(2)             | Close to ReLU              |                                                                                                                                                                                                                            
     | SoftmaxActivation   | 1.0                 | Used at output only        |                                                                                                                                                                                                                            
     | HardTanhActivation  | sqrt(5/3)           | Same as tanh               |     

Change defaults for deep networks:                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                           
     | Parameter           | Current          | New                       | Rationale                             |                                                                                                                                                                                        
     |---------------------|------------------|---------------------------|---------------------------------------|                                                                                                                                                                                        
     | eta_infer           | 0.1              | 0.5                       | Match jpc activity_lr=0.5             |                                                                                                                                                                                        
     | infer_steps formula | 4*(num_hidden+2) | max(20, 4*(num_hidden+2)) | Keep current linear scaling           |                                                                                                                                                                                        
     | optimizer LR        | 0.001            | 0.01                      | Closer to jpc param_lr=0.1            |                                                                                                                                                                                        
     | weight_decay        | 0.01             | 0.1                       | Stronger regularization for deep nets |                                                                                                                                                                                        
     | batch_size          | 256              | 256                       | Keep current                          |                                                                                                                                                                                        
     | num_epochs          | 4                | 5                         | Slightly more for deeper models       |   


## Plan: Jacobian Compensation for Deep PC Networks                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                           
 Background                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                           
 Plain chain MLP with muPC scaling and tanh activation degrades with depth:                                                                                                                                                                                                                                
 - 8 layers: 91% | 32 layers: 48.6% (was 56% before forward gain, then 48.6% after)                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                           
 Forward variance gain (variance_gain()) was already implemented:                                                                                                                                                                                                                                          
 - a = gain / sqrt(fan_in * K) where gain = sqrt(5/3) for tanh                                                                                                                                                                                                                                             
 - This preserves forward variance (0.013 → 0.219 at layer 32)                                                                                                                                                                                                                                             
 - But accuracy dropped slightly (56% → 48.6%) because tanh saturation attenuates the per-hop Jacobian                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                           
 Root Cause Analysis                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                           
 The topdown gradient per hop in local PC inference is:                                                                                                                                                                                                                                                    
 grad_to_x = c_td * dE/d(a*x) = a * W^T @ diag(tanh'(pre_act)) @ error                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                           
 The effective per-hop Jacobian J = a * W^T @ diag(tanh'(z)) has:                                                                                                                                                                                                                                          
 - RMS singular value ≈ gain * rms(tanh'(z)) where gain = a * sqrt(fan_in)                                                                                                                                                                                                                                 
 - For tanh with gain=sqrt(5/3): RMS sv ≈ 0.79 → gradients shrink by 0.77x per hop                                                                                                                                                                                                                         
 - Over 32 hops: gradient magnitude is 3.7e-04 of the original (vanishing)                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                           
 For ReLU: gain * rms(relu') = sqrt(2) * sqrt(0.5) = 1.0 — perfect preservation.                                                                                                                                                                                                                           
 For tanh: gain * rms(tanh') = 1.29 * 0.61 = 0.79 — 0.79x attenuation per hop.                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                           
 Solution: Jacobian Compensation in topdown_grad_scale                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                           
 Add a jacobian_gain factor to the topdown gradient scaling:                                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                                                           
 topdown_grad_scale = a * jacobian_gain                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                           
 where jacobian_gain = 1 / (variance_gain * rms(act'(z))) for z ~ N(0, variance_gain^2).                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                           
 This normalizes the expected per-hop gradient propagation factor to 1.0.                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
 Simulation results (32 hops, fan_in=64, tanh)                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                           
 | Strategy                               | Gradient norm after 32 hops | Per-hop factor |                                                                                                                                                                                                                 
 |----------------------------------------|-----------------------------|----------------|                                                                                                                                                                                                                 
 | Current (c_td = a)                     | 3.7e-04                     | 0.774          |                                                                                                                                                                                                                 
 | Jacobian-compensated (c_td = a * 1.26) | 0.604                       | 0.976          |                                                                                                                                                                                                                 
 | Aggressive (c_td = a * 2)              | 1.86e+06 (exploding)        | 1.55           |                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                           
 The compensation brings per-hop factor from 0.774 to 0.976 — 1600x improvement at 32 layers.                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                           
 jacobian_gain values by activation                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                           
 | Activation | variance_gain | rms(act') | gain×rms | jacobian_gain |                                                                                                                                                                                                                                     
 |------------|---------------|-----------|----------|---------------|                                                                                                                                                                                                                                     
 | Identity   | 1.0           | 1.0       | 1.0      | 1.0           |                                                                                                                                                                                                                                     
 | ReLU       | √2            | √0.5      | 1.0      | 1.0           |                                                                                                                                                                                                                                     
 | LeakyReLU  | √(2/(1+α²))   | ~√0.5     | 1.0      | 1.0           |                                                                                                                                                                                                                                     
 | Tanh       | √(5/3)        | 0.614     | 0.793    | 1.261         |                                                                                                                                                                                                                                     
 | GELU       | √2            | 0.605     | 0.856    | 1.168         |                                                                                                                                                                                                                                     
 | HardTanh   | √(5/3)        | 0.749     | 0.967    | 1.035         |                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                           
 Note: for ReLU/LeakyReLU, jacobian_gain = 1.0 exactly — no change from current behavior.      
 
python examples/mupc_mnist_demo.py --num_hidden=*

  | Config              | Accuracy |                                                                                                                                                                                                                                                      
  |---------------------|----------|                                                                                                                                                                                                                                                      
  | 8 hidden, 4 epochs  | 90.8%    |
  | 16 hidden, 4 epochs | 90.9%    |
  | 32 hidden, 4 epochs | 62.0%    |
  | 64 hidden, 4 epochs | 24.4%    |


## JPC versus FabricPC comparison results
 
  Summary                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
  Demo: examples/jpc_fc_resnet_compare.py                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
  The demo replicates jpc's FC-ResNet architecture using three custom node types with internal scaling (no MuPCConfig), enabling a fair comparison of FabricPC vs jpc at different depths and scaling modes.                                                                                               
                                                                                                                                                                                                                                                                                                           
  Results                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                           
  | Depth | jpc scaling | fabricpc scaling | Hyperparams (same for both)     |                                                                                                                                                                                                                             
  |-------|-------------|------------------|---------------------------------|                                                                                                                                                                                                                             
  | 5     |   93.16%    |      88.98%      | eta=0.2, mn=0.2, plr=0.001, 3ep |                                                                                                                                                                                                                             
  | 10    |   92.75%    |      68.35%      | eta=0.2, mn=0.2, plr=0.001, 3ep |                                                                                                                                                                                                                             
  | 20    |   69.08%    |      15.86%      | eta=0.1, mn=0.1, plr=0.001, 3ep |                                                                                                                                                                                                                             
  | 30    |   64.57%    |      12.22%      | eta=0.1, mn=0.1, plr=0.001, 3ep |                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                           
  JPC improves to 85% with hyperparameters tuned at depth=30: plr=0.0003, eta=0.1, max_norm=0.1, 5 epochs                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                           
  Key Findings                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                           
  1. The depth factor is critical: jpc's 1/sqrt(width*depth) hidden scaling maintains performance much better as depth increases. FabricPC's gain/sqrt(width) (no depth factor) collapses beyond depth=5.                                                                                                  
  2. Inference method matters a lot: jpc uses an adaptive ODE solver (diffrax Heun + PID controller) which handles stiff dynamics natively. FabricPC's fixed-step SGD needs:                                                                                                                               
    - InferenceSGDNormClip (gradient norm clipping) for stability                                                                                                                                                                                                                                          
    - Lower parameter learning rate (0.001 vs jpc's 0.1) because inference doesn't fully converge                                                                                                                                                                                                          
    - At depth=30, the SGD inference gap accounts for ~8% accuracy loss vs jpc                                                                                                                                                                                                                             
  3. At moderate depths (5-10), FabricPC matches jpc's ~93% accuracy — the architectures and PC mechanics are equivalent when inference converges.                                                                                                                                                         
                                                                                                                                                                                                                                                                                                           
  Default usage                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                           
  python examples/jpc_fc_resnet_compare.py  # depth=10, jpc scaling → ~93%      
  
## Skip connections & Arbitry Graphs Upgrade (completed 2026-04-16)

  Request 1 — Diagnose and fix muPC scaling for deep ResNets:
  - Root cause: skip connections attenuated by 1/sqrt(K) causing exponential signal decay (0.707^L)
  - Added fabricpc_v2 mode to jpc_fc_resnet_compare.py with skip_scale=1.0 and depth-compensated linear scaling
  - Validated: 85.4% accuracy at depth 64 (vs 11.5% for broken fabricpc, 83.5% for JPC reference)

  Request 2 — Extend muPC to arbitrary graphs:
  - Added apply_variance_scaling = True default on NodeBase
  - Created SkipConnection node with apply_variance_scaling = False
  - Rewrote mupc.py with _count_skip_depth() counting only unscaled merge points as depth L
  - Formula: a = gain / sqrt(fan_in * K * L) where K = in-degree, L = max(skip_depth, 1)
  - Pure sequential chains degenerate to original formula (L=1), residual networks get depth compensation
