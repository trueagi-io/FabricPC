# Refactor: Object-Oriented Graph Builder API (Completed 2/25/2026)

## Goal
Replace dictionary-based model definition with direct object instantiation.
Remove the registry system entirely. Users import and instantiate classes directly.
Preserve JAX's stateless computation model (all computation methods stay `@staticmethod`).

## Target API
```python
from fabricpc.nodes import Linear
from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.core.activations import Identity, Sigmoid, Softmax, ReLU
from fabricpc.core.energy import CrossEntropy, Gaussian
from fabricpc.core.initializers import Xavier, Kaiming
from fabricpc.builder import Edge, TaskMap, graph, GraphNamespace

with GraphNamespace("blockQ"):
    layer1 = Linear(shape=(784,), activation=Identity(), name="pixels")
    layer2 = TransformerBlock(shape=(256,), num_heads=4, ff_dim=512,
                              internal_activation=ReLU(), name="hidden1")
    layer3 = Linear(shape=(10,), activation=Softmax(),
                    energy=CrossEntropy(), name="class")

    structure = graph(
        nodes=[layer1, layer2, layer3],
        edges=[
            Edge(source=layer1, target=layer2.slot("in")),
            Edge(source=layer2, target=layer3.slot("in")),
        ],
        task_map=TaskMap(x=layer1, y=layer3),
    )
    assert layer1.name == "blockQ/pixels"

params = initialize_params(structure, rng_key)
```

## Design Decisions
- **No registries**: Remove all 5 registries. Users import classes directly. Custom nodes/activations/energies extend base classes.
- **No dict config path**: `from_config()` and dict-based graph definition removed. Breaking change.
- **No wrapper classes**: No `Activation("relu")`. Users write `ReLU()` directly.
- **Slot access**: `node.slot("in")` explicit method call (avoids `in` keyword collision).
- **Same classes, dual role**: `Linear` is both the construction-time descriptor and the runtime computation class.
- **Static methods preserved**: Computation methods keep `@staticmethod` + `node_info` parameter. Call pattern: `node.forward_inference(params, inputs, state, node.node_info, is_clamped)`.
- **Copy-on-finalize**: `graph()` creates finalized copies with edge/slot info. Original user objects stay unchanged.
- **Activation/Energy instances hold config**: `LeakyReLU(alpha=0.02)` stores `.config = {"alpha": 0.02}`. Static methods still take `config` param for JAX compat.
- **`graph()` returns GraphStructure directly**: No dict intermediate.

---

## Phase 1: Activation/Energy/Initializer Classes Get Constructors

### `fabricpc/core/activations.py`

**Remove**: `_activation_registry`, `register_activation()`, `get_activation_class()`, `list_activation_types()`, `discover_external_activations()`, `validate_activation_config()`.

**Keep**: `ActivationBase` ABC, all concrete classes (Sigmoid, ReLU, etc.), `get_activation()` helper.

**Add `__init__` to each activation class** to hold config:
```python
class ActivationBase(ABC):
    """Base class. Custom activations extend this."""
    def __init__(self, **config):
        self.config = config

class Sigmoid(ActivationBase):
    def __init__(self): super().__init__()
    @staticmethod
    def forward(x, config=None): return jax.nn.sigmoid(x)
    @staticmethod
    def derivative(x, config=None): ...

class LeakyReLU(ActivationBase):
    def __init__(self, alpha=0.01): super().__init__(alpha=alpha)
    @staticmethod
    def forward(x, config=None):
        alpha = config.get("alpha", 0.01) if config else 0.01
        return jnp.where(x > 0, x, alpha * x)
```

**Rewrite `get_activation()`**: Instead of registry lookup from config dict, accept an activation instance:
```python
def get_activation(activation: ActivationBase) -> Tuple[Callable, Callable]:
    config = activation.config
    def forward_fn(x): return type(activation).forward(x, config)
    def deriv_fn(x): return type(activation).derivative(x, config)
    return forward_fn, deriv_fn
```

### `fabricpc/core/energy.py`

Same pattern. **Remove** registry. **Keep** `EnergyFunctional` ABC + concrete classes. **Add `__init__`**.

```python
class CrossEntropy(EnergyFunctional):
    def __init__(self, eps=1e-7): super().__init__(eps=eps)
    @staticmethod
    def energy(z_latent, z_mu, config=None): ...
    @staticmethod
    def grad_latent(z_latent, z_mu, config=None): ...
```

**Rewrite** `compute_energy()`, `compute_energy_gradient()`, `get_energy_and_gradient()` to accept energy instances instead of config dicts.

### `fabricpc/core/initializers.py`

Same pattern. **Remove** registry. **Keep** `InitializerBase` ABC + concrete classes. **Add `__init__`**.

```python
class Xavier(InitializerBase):
    def __init__(self, distribution="uniform"): super().__init__(distribution=distribution)
    @staticmethod
    def initialize(key, shape, config=None): ...
```

### `fabricpc/core/registry.py` — **DELETE** entire file

### `fabricpc/core/config.py` — **DELETE** or reduce to minimal utilities

Schema-based validation (`validate_config`, `transform_shorthand`, `validate_typed_config`) is no longer needed since users instantiate typed objects directly. Delete or keep only if other code depends on it.

---

## Phase 2: NodeBase Gets a Constructor, Registries Removed

### `fabricpc/nodes/base.py`

**Add `__init__` to NodeBase**:
```python
class NodeBase(ABC):
    def __init__(self, shape, name, activation=None, energy=None,
                 latent_init=None, **extra_config):
        from fabricpc.builder.namespace import _get_current_namespace
        ns = _get_current_namespace()
        self._name = f"{ns}/{name}" if ns else name
        self._shape = tuple(shape)
        self._activation = activation  # ActivationBase instance or None (uses class default)
        self._energy = energy          # EnergyFunctional instance or None (uses class default)
        self._latent_init = latent_init  # InitializerBase instance or None
        self._extra_config = extra_config
        self._node_info = None  # Set by graph builder (copy-on-finalize)
```

**Add properties and methods**:
```python
    @property
    def name(self) -> str: return self._name
    @property
    def shape(self) -> Tuple[int, ...]: return self._shape
    @property
    def node_info(self) -> NodeInfo: return self._node_info  # None until graph()

    def slot(self, slot_name: str) -> "SlotRef":
        slot_specs = type(self).get_slots()
        if slot_name not in slot_specs:
            raise KeyError(f"No slot '{slot_name}'. Available: {list(slot_specs.keys())}")
        return SlotRef(node=self, slot=slot_name)

    def _with_graph_info(self, node_info: NodeInfo) -> "NodeBase":
        """Copy-on-finalize: return copy with graph topology info."""
        import copy
        new = copy.copy(self)
        new._node_info = node_info
        return new
```

**Remove**: `from_config()`, `_resolve_energy_config()`, `_resolve_activation_config()`, `_resolve_state_init_config()`, `get_energy_functional()`, `BASE_CONFIG_SCHEMA`, `CONFIG_SCHEMA` validation logic.

**Keep**: `_build_slots()` (moved to graph builder), `get_slots()` (abstract), `forward()`, `forward_inference()`, `forward_learning()`, `energy_functional()`, `compute_gain_mod_error()`.

**Update `energy_functional()`** (base.py:395): Instead of `get_energy_and_gradient(state.z_latent, state.z_mu, energy_config_dict)`, accept the energy instance. The energy instance is stored on the node, so `node_info` needs to carry it. See Phase 4 (NodeInfo changes).

**Update `forward_inference()`** (base.py:213-295): Remove internal `get_node_class()` calls. These call `node_class.forward()` and `node_class.energy_functional()` — but since `forward_inference` is a static method on the same class, these can become `cls.forward()` calls or the node class is passed in/known from context.

### `fabricpc/nodes/linear.py`

**Add `__init__`**:
```python
@register_node("linear")  # decorator can stay temporarily or be removed
class LinearNode(FlattenInputMixin, NodeBase):
    def __init__(self, shape, name, activation=None, energy=None,
                 use_bias=True, flatten_input=False, weight_init=None, latent_init=None):
        super().__init__(shape=shape, name=name, activation=activation, energy=energy,
                         latent_init=latent_init, use_bias=use_bias,
                         flatten_input=flatten_input, weight_init=weight_init)
```

**Update `forward()`** (linear.py:~170-219): Replace `get_activation(node_info.node_config["activation"])` with accessing the activation instance from node_info. See Phase 4.

**Update `forward_learning()`** (linear.py:~330-403): Same pattern — replace `get_activation()` call.

### `fabricpc/nodes/transformer.py`

**Add `__init__`** with typed params (num_heads, ff_dim, internal_activation, rope_theta, etc.).

**Update** internal `get_activation()` and `get_node_class()` calls.

### `fabricpc/nodes/registry.py` — **DELETE** or reduce to bare minimum

Remove `_node_registry`, `register_node()`, `get_node_class()`, `validate_node_config()`, `list_node_types()`, `discover_external_nodes()`.

If `@register_node` is kept temporarily for the transition, it just sets `cls._registered_type = node_type` and nothing else. Eventually remove.

### `fabricpc/nodes/__init__.py`

Change from registry-based exports to direct class imports:
```python
from fabricpc.nodes.linear import LinearNode as Linear
from fabricpc.nodes.transformer import TransformerBlockNode as TransformerBlock
from fabricpc.nodes.identity import IdentityNode as Identity
```

---

## Phase 3: NodeInfo and GraphStructure Changes

### `fabricpc/core/types.py`

**Update NodeInfo** to carry activation/energy instances instead of config dicts:
```python
@dataclass(frozen=True)
class NodeInfo:
    name: str
    shape: Tuple[int, ...]
    node_type: str           # keep for debugging/display
    node_config: Dict        # extra config (use_bias, flatten_input, etc.)
    activation: Any          # ActivationBase instance
    energy: Any              # EnergyFunctional instance
    latent_init: Any         # InitializerBase instance or None
    slots: Dict[str, SlotInfo]
    in_degree: int
    out_degree: int
    in_edges: Tuple[str, ...]
    out_edges: Tuple[str, ...]
```

**Update GraphStructure**:
```python
class GraphStructure(NamedTuple):
    nodes: Dict[str, "NodeBase"]  # was Dict[str, NodeInfo]
    edges: Dict[str, EdgeInfo]
    task_map: Dict[str, str]
    node_order: Tuple[str, ...]
    config: Dict[str, Any]
```

**Remove** `GraphStructure.from_config()` and `GraphStructure.CONFIG_SCHEMA`.

**Update `_topological_sort()`**: Access `node.node_info.in_degree` etc.

---

## Phase 4: Builder Primitives (new files)

### `fabricpc/builder/edge.py`
```python
@dataclass(frozen=True)
class SlotRef:
    node: "NodeBase"
    slot: str

class Edge:
    def __init__(self, source, target):
        self.source = source
        if isinstance(target, SlotRef):
            self.target_node, self.target_slot = target.node, target.slot
        else:
            self.target_node, self.target_slot = target, "in"
```

### `fabricpc/builder/namespace.py`
Thread-local stack-based `GraphNamespace` context manager.

### `fabricpc/builder/graph_builder.py`

**`TaskMap`**: Accepts node objects or strings. Auto-resolves `.name`.

**`graph()` function** — the core builder. Directly returns `GraphStructure`:
```python
def graph(nodes, edges, task_map, graph_state_initializer=None) -> GraphStructure:
    # 1. Build EdgeInfo objects from Edge objects
    edge_infos = {}
    for edge in edges:
        key = f"{edge.source.name}->{edge.target_node.name}:{edge.target_slot}"
        edge_infos[key] = EdgeInfo(key=key, source=edge.source.name,
                                    target=edge.target_node.name, slot=edge.target_slot)

    # 2. For each node: resolve defaults, build slots, build NodeInfo, copy-on-finalize
    finalized_nodes = {}
    for node in nodes:
        in_edges = {k: e for k, e in edge_infos.items() if e.target == node.name}
        out_edges = {k: e for k, e in edge_infos.items() if e.source == node.name}
        slots = _build_slots(node, in_edges)  # reuse logic from old NodeBase._build_slots

        # Resolve activation/energy: use node's instance, or class default
        activation = node._activation or type(node).DEFAULT_ACTIVATION()
        energy = node._energy or type(node).DEFAULT_ENERGY()
        latent_init = node._latent_init or type(node).DEFAULT_LATENT_INIT()

        node_info = NodeInfo(
            name=node.name, shape=node.shape,
            node_type=type(node).__name__,
            node_config=node._extra_config,
            activation=activation, energy=energy, latent_init=latent_init,
            slots=slots,
            in_degree=len(in_edges), out_degree=len(out_edges),
            in_edges=tuple(in_edges.keys()), out_edges=tuple(out_edges.keys()),
        )
        finalized_nodes[node.name] = node._with_graph_info(node_info)

    # 3. Validate edges point to existing nodes
    # 4. Topological sort
    # 5. Build and return GraphStructure
    node_order = _topological_sort(finalized_nodes, edge_infos)
    task_map_dict = task_map.to_config() if isinstance(task_map, TaskMap) else task_map

    return GraphStructure(
        nodes=finalized_nodes, edges=edge_infos,
        task_map=task_map_dict, node_order=node_order,
        config={"graph_state_initializer": graph_state_initializer or {"type": "feedforward"}},
    )
```

### `fabricpc/builder/__init__.py`
```python
from fabricpc.builder.edge import Edge, SlotRef
from fabricpc.builder.namespace import GraphNamespace
from fabricpc.builder.graph_builder import graph, TaskMap
```

---

## Phase 5: Update Computation Code

All callsites change from registry lookup to direct use of node instances.

### Pattern: `get_node_class()` elimination

**Before**:
```python
node_info = structure.nodes[node_name]  # NodeInfo
node_class = get_node_class(node_info.node_type)  # registry lookup
node_class.forward_inference(params, inputs, state, node_info, is_clamped)
```

**After**:
```python
node = structure.nodes[node_name]  # NodeBase instance
node.forward_inference(params, inputs, state, node.node_info, is_clamped)
```

### Pattern: `get_activation()` elimination

**Before** (linear.py:212):
```python
activation_fn, _ = get_activation(node_info.node_config["activation"])
z_mu = activation_fn(pre_activation)
```

**After**:
```python
activation = node_info.activation  # ActivationBase instance
z_mu = type(activation).forward(pre_activation, activation.config)
```

### Pattern: `get_energy_and_gradient()` elimination

**Before** (base.py:431):
```python
energy_config = node_info.node_config.get("energy")
energy, grad = get_energy_and_gradient(z_latent, z_mu, energy_config)
```

**After**:
```python
energy_obj = node_info.energy  # EnergyFunctional instance
energy = type(energy_obj).energy(z_latent, z_mu, energy_obj.config)
grad = type(energy_obj).grad_latent(z_latent, z_mu, energy_obj.config)
```

### Files to update:

| File | What changes |
|------|-------------|
| `fabricpc/core/inference.py` | Remove `get_node_class` import/calls. `gather_inputs` takes node not NodeInfo. |
| `fabricpc/graph/graph_net.py` | `compute_local_weight_gradients`: node from structure. `initialize_params`: same. Remove `create_pc_graph(config_dict)` — replace with `initialize_params(structure, rng_key)` only. |
| `fabricpc/graph/state_initializer.py` | Remove `get_state_init_class`. Access `node.node_info` for shape/config. `FeedforwardStateInit` calls `node.forward()` directly. |
| `fabricpc/training/train.py` | `get_graph_param_gradient`, `eval_step`: access `node.node_info.in_degree`. |
| `fabricpc/training/multi_gpu.py` | Same pattern. |
| `fabricpc/training/train_autoregressive.py` | Same pattern. |
| `fabricpc/nodes/base.py` | `forward_inference()`, `energy_functional()`: use `node_info.activation`/`node_info.energy` instances. |
| `fabricpc/nodes/linear.py` | `forward()`, `forward_learning()`: use `node_info.activation` instance. |
| `fabricpc/nodes/transformer.py` | Use activation instance for internal activation. |

---

## Phase 6: Update Demos and Tests

### Examples:
- `examples/mnist_demo.py` — rewrite with new object API
- `examples/transformer_demo.py` — rewrite using GraphNamespace
- `examples/mnist_advanced.py` — rewrite (was template-based dicts)
- `examples/custom_node.py` — update to show custom node via class extension

### Tests:
- `tests/test_fabricpc.py` — rewrite dict-based configs to use object API
- `tests/test_fabricpc_extended.py` — same
- Add `tests/test_builder.py`:
  - Node construction, slot(), _with_graph_info
  - Edge construction, default slot
  - GraphNamespace nesting, thread safety
  - TaskMap with node refs
  - graph() end-to-end → initialize_params → train step
  - Copy-on-finalize: original node unchanged
  - Custom node via subclassing NodeBase

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `fabricpc/core/registry.py` | **DELETE** | Generic Registry class no longer needed |
| `fabricpc/core/config.py` | **DELETE/REDUCE** | Schema validation no longer needed |
| `fabricpc/nodes/registry.py` | **DELETE/REDUCE** | Node registry no longer needed |
| `fabricpc/builder/__init__.py` | **NEW** | Edge, SlotRef, GraphNamespace, graph, TaskMap |
| `fabricpc/builder/edge.py` | **NEW** | Edge, SlotRef |
| `fabricpc/builder/namespace.py` | **NEW** | GraphNamespace context manager |
| `fabricpc/builder/graph_builder.py` | **NEW** | graph(), TaskMap, _build_slots, _topological_sort |
| `fabricpc/core/activations.py` | **MODIFY** | Remove registry; add __init__ to each class; rewrite get_activation() |
| `fabricpc/core/energy.py` | **MODIFY** | Remove registry; add __init__; rewrite compute functions |
| `fabricpc/core/initializers.py` | **MODIFY** | Remove registry; add __init__ |
| `fabricpc/core/types.py` | **MODIFY** | NodeInfo gets activation/energy fields; GraphStructure stores NodeBase; remove from_config() |
| `fabricpc/nodes/base.py` | **MODIFY** | Add __init__, slot(), _with_graph_info; remove from_config and resolve methods; update energy_functional/forward_inference |
| `fabricpc/nodes/linear.py` | **MODIFY** | Add __init__; update forward/forward_learning to use activation instance |
| `fabricpc/nodes/transformer.py` | **MODIFY** | Add __init__; update activation usage |
| `fabricpc/nodes/identity.py` | **MODIFY** | Add __init__ |
| `fabricpc/nodes/__init__.py` | **MODIFY** | Direct class imports instead of registry exports |
| `fabricpc/core/__init__.py` | **MODIFY** | Remove registry function exports |
| `fabricpc/core/inference.py` | **MODIFY** | Remove get_node_class; use node objects |
| `fabricpc/graph/graph_net.py` | **MODIFY** | Remove create_pc_graph(dict) path; keep initialize_params(structure); remove get_node_class |
| `fabricpc/graph/__init__.py` | **MODIFY** | Remove registry exports |
| `fabricpc/graph/state_initializer.py` | **MODIFY** | Remove registry; use node objects; use activation/energy instances |
| `fabricpc/training/train.py` | **MODIFY** | Use node objects from structure |
| `fabricpc/training/multi_gpu.py` | **MODIFY** | Same |
| `fabricpc/training/train_autoregressive.py` | **MODIFY** | Same |
| `tests/test_fabricpc.py` | **MODIFY** | Rewrite to use object API |
| `tests/test_fabricpc_extended.py` | **MODIFY** | Rewrite to use object API |
| `tests/test_builder.py` | **NEW** | Builder API tests |
| `examples/mnist_demo.py` | **MODIFY** | Object API |
| `examples/transformer_demo.py` | **MODIFY** | Object API + GraphNamespace |
| `examples/mnist_advanced.py` | **MODIFY** | Object API |
| `examples/custom_node.py` | **MODIFY** | Custom node via subclassing |

---

## Implementation Order

1. **Phase 1** — Activation/Energy/Initializer: add constructors, remove registries from these files
2. **Phase 2** — NodeBase: add constructor, remove from_config/registry; update concrete nodes
3. **Phase 3** — NodeInfo/GraphStructure: add activation/energy fields to NodeInfo; store nodes in GraphStructure
4. **Phase 4** — Builder primitives: Edge, SlotRef, GraphNamespace, TaskMap, graph()
5. **Phase 5** — Update all computation code: inference, training, gradient, state init
6. **Phase 6** — Update tests and demos; verify everything works end-to-end

Note: Phases 1-2 modify the foundation classes. Phase 3 changes the core types. Phase 4 adds the new user-facing API. Phase 5 wires it all together. Phase 6 validates.



  Final results:                                                                                                                                                                                                                           
  - 147/147 tests pass                                                                                                                                                                                                                     
  - No stale old-API references remain (only intentional deprecated stubs in nodes/registry.py and the EdgeInfo.from_config utility)                                                                                                       
                                                                                                                                                                                                                                           
  Summary of what was done across all phases:                                                                                                                                                                                              
                                                                                                                                                                                                                                           
  | Phase | Description                                                                | Status |                                                                                                                                          
  |-------|----------------------------------------------------------------------------|--------|                                                                                                                                          
  | 1     | Activation/Energy/Initializer classes get constructors, registries removed | Done   |                                                                                                                                          
  | 2     | NodeBase gets constructor, node registries removed                         | Done   |                                                                                                                                          
  | 3     | NodeInfo/GraphStructure updated to carry instances                         | Done   |                                                                                                                                          
  | 4     | Builder primitives: Edge, SlotRef, GraphNamespace, TaskMap, graph()        | Done   |                                                                                                                                          
  | 5     | All computation code updated (inference, training, gradients, state init)  | Done   |                                                                                                                                          
  | 6     | All tests and examples rewritten for new object API                        | Done   |                                                                                                                                          
                                                                                                                                                                                                                                           
The codebase now uses the direct object instantiation API:
from fabricpc.nodes import Linear
from fabricpc.builder import Edge, TaskMap, graph
from fabricpc.graph import initialize_params
  
layer1 = Linear(shape=(784,), name="input")
layer2 = Linear(shape=(256,), name="hidden")
layer3 = Linear(shape=(10,), name="output")
structure = graph(
  nodes=[layer1, layer2, layer3],
  edges=[Edge(layer1, layer2.slot("in")), Edge(layer2, layer3.slot("in"))],
  task_map=TaskMap(x=layer1, y=layer3),
)
params = initialize_params(structure, rng_key)

Revisions to the refactorization:
Removed get_activation, get_activation_fn, and get_activation_deriv from the codebase. Changes:
- fabricpc/core/activations.py — Deleted the 3 functions and the "Convenience Functions" section. Removed unused Callable/Tuple imports.
- fabricpc/core/__init__.py — Removed from import and __all__.
- fabricpc/nodes/linear.py:325 — compute_gain_mod_error now calls type(activation).derivative(x, activation.config) directly.
- fabricpc/nodes/transformer.py:217 — FFN activation now calls type(internal_activation).forward(x, internal_activation.config) directly.
