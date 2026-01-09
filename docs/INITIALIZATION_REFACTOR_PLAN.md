# Latent Initialization System Refactoring Plan (Completed)

## Overview

Refactor FabricPC's initialization system to use registry-based patterns consistent with `EnergyFunctional` and `ActivationBase`. Introduces two new abstraction layers:

1. **Initializer Abstraction** - Context-agnostic tensor initializers (for weights and states)
2. **State Initialization Abstraction** - Graph-level state initialization strategies

## Design Decisions (User Confirmed)

- **Class Style**: Stateless with static methods (like EnergyFunctional)
- **Interface**: Single `initialize(key, shape, config)` method
- **Config Hierarchy**: Graph-level `graph_state_initializer` config with node-level override via `node_config["latent_init"]`

---

## Key Schema Changes

### GraphState Config Schema

Add `graph_state_initializer` parameter to the graph config schema with default:
```python
"graph_state_initializer": {
    "type": "feedforward",
}
```

The `state_init_config` argument to `initialize_state()` becomes **required** (no longer optional with default). Callers must explicitly provide this parameter from their graph config.

---

## Implementation Steps

### Step 1: Create `fabricpc/core/initializers.py` (New File)

**Base Class:**
```python
class InitializerBase(ABC):
    CONFIG_SCHEMA: Dict[str, Dict[str, Any]]

    @staticmethod
    @abstractmethod
    def initialize(key: jax.Array, shape: Tuple[int, ...], config: Dict[str, Any] = None) -> jnp.ndarray:
        pass
```

**Registry:**
- `_initializer_registry = Registry(name="initializer", entry_point_group="fabricpc.initializers", ...)`
- `@register_initializer(type)` decorator
- `get_initializer_class()`, `list_initializer_types()`, `validate_initializer_config()`

**Built-in Implementations:**
- `ZerosInitializer` - Returns zeros
- `NormalInitializer` - CONFIG: `mean` (default 0.0), `std` (default 0.05)
- `UniformInitializer` - CONFIG: `min` (default -0.1), `max` (default 0.1)
- `XavierInitializer` - CONFIG: `distribution` (normal/uniform)
- `KaimingInitializer` - CONFIG: `mode` (fan_in/fan_out), `nonlinearity`, `distribution`, `a`

**Convenience Function:**
```python
def initialize(key, shape, config=None) -> jnp.ndarray:
    """Dispatch to appropriate initializer based on config["type"]"""
```

---

### Step 2: Create `fabricpc/graph/state_initializer.py` (New File)

**Base Class:**
```python
class StateInitBase(ABC):
    CONFIG_SCHEMA: Dict[str, Dict[str, Any]]

    @staticmethod
    @abstractmethod
    def initialize_state(
        structure: GraphStructure,
        batch_size: int,
        rng_key: jax.Array,
        clamps: Dict[str, jnp.ndarray],
        config: Dict[str, Any],
        params: GraphParams = None,
    ) -> GraphState:
        pass
```

**Registry:**
- `_state_init_registry = Registry(name="state_init", entry_point_group="fabricpc.state_initializers", ...)`
- `@register_state_init(type)` decorator
- `get_state_init_class()`, `list_state_init_types()`

**Built-in Implementations:**

1. `GlobalStateInit`:
   - CONFIG: `default_initializer` (dict, default `{"type": "normal"}`)
   - Processes nodes independently

2. `FeedforwardStateInit`:
   - Requires `params` to be provided
   - Pass 1: Initialize all nodes with fallback
   - Pass 2: Process nodes in topological order, set `z_latent = z_mu` from forward pass

**Convenience Function:**
```python
def initialize_graph_state(structure, batch_size, rng_key, clamps=None, state_init_config=None, params=None) -> GraphState:
    """Dispatch to appropriate state initializer based on config["type"]"""
```

---

### Step 3: Update `fabricpc/core/initialization.py`

Refactor existing functions to delegate to new system:
And then remove them:
```python
def initialize_weights(config, key, shape) -> jnp.ndarray:
    from fabricpc.core.initializers import initialize
    return initialize(key, shape, config)

def initialize_state_values(config, key, shape) -> jnp.ndarray:
    from fabricpc.core.initializers import initialize
    return initialize(key, shape, config)
```

Remove/deprecate: `parse_state_init_config()` (logic moves to StateInitBase implementations)

---

### Step 4: Update `fabricpc/graph/graph_net.py`

Replace `initialize_state()` function body to delegate:

```python
def initialize_state(structure, batch_size, rng_key, clamps=None, state_init_config=None, params=None) -> GraphState:
    from fabricpc.graph.state_initializer import initialize_graph_state
    return initialize_graph_state(structure, batch_size, rng_key, clamps, state_init_config, params)
```

---

### Step 5: Update `fabricpc/nodes/base.py`

Add `latent_init` to `BASE_CONFIG_SCHEMA`:

```python
BASE_CONFIG_SCHEMA = {
    "name": {"type": str, "required": True, ...},
    "shape": {"type": tuple, "required": True, ...},
    "type": {"type": str, "required": True, ...},
    "latent_init": {
        "type": dict,
        "default": None,
        "description": "Node-level state initialization config (overrides graph-level default)"
    },
}
```

Add `_resolve_state_init_config()` classmethod (similar to `_resolve_energy_config`):
- Returns `None` if not specified (let StateInitBase use graph-level default)
- Validates against Initializer CONFIG_SCHEMA if specified

Update `from_config()` to call `_resolve_state_init_config()`.

---

### Step 6: Update Module Exports

**`fabricpc/core/__init__.py`:**
```python
from fabricpc.core.initializers import (
    InitializerBase, register_initializer, get_initializer_class,
    list_initializer_types, initialize, get_default_weight_init, get_default_state_init,
)
```

**`fabricpc/graph/__init__.py`:**
```python
from fabricpc.graph.state_initializer import (
    StateInitBase, register_state_init, get_state_init_class,
    list_state_init_types, initialize_graph_state,
)
```

---

### Step 7: Update Trainers and Tests to Pass Required Config

All callers of `initialize_state()` must now pass `state_init_config` explicitly (no longer optional).

**Training files to update:**
- `fabricpc/training/train.py`
- `fabricpc/training/train_backprop.py`
- `fabricpc/training/train_autoregressive.py`
- `fabricpc/training/multi_gpu.py`

Pattern: Read `graph_state_initializer` from the graph config and pass to `initialize_state()`:
```python
state_init_config = graph_config.get("graph_state_initializer", {
    "type": "feedforward",
})
state = initialize_state(structure, batch_size, key, clamps, state_init_config, params)
```

**Test files to update:**
- `tests/test_fabricpc.py`
- `tests/test_fabricpc_extended.py`
- `tests/test_auto_node_grad.py`
- `tests/test_ndim_shapes.py`

---

### Step 8: Add Tests

**`tests/test_initializers.py`:**
- Test each built-in initializer (zeros, normal, uniform, xavier, kaiming)
- Test registry registration/lookup/validation
- Test custom initializer registration
- Test config validation with defaults

**`tests/test_state_initializer.py`:**
- Test GlobalStateInit with graph-level config
- Test NodeDistributionStateInit with node-level override
- Test FeedforwardStateInit requires params
- Test FeedforwardStateInit topological propagation
- Test clamp handling in both strategies

---

## Files to Modify/Create

| File | Action |
|------|--------|
| `fabricpc/core/initializers.py` | **CREATE** - Initializer base class, registry, built-in implementations |
| `fabricpc/graph/state_initializer.py` | **CREATE** - StateInit base class, registry, implementations |
| `fabricpc/core/initialization.py` | MODIFY - Delegate to new initializers |
| `fabricpc/graph/graph_net.py` | MODIFY - Delegate to new state_initializer, make config required |
| `fabricpc/nodes/base.py` | MODIFY - Add state_initializer schema field |
| `fabricpc/core/__init__.py` | MODIFY - Export new initializers |
| `fabricpc/graph/__init__.py` | MODIFY - Export new state_initializer |
| `fabricpc/training/train.py` | MODIFY - Pass graph_state_initializer config to initialize_state |
| `fabricpc/training/train_backprop.py` | MODIFY - Pass graph_state_initializer config |
| `fabricpc/training/train_autoregressive.py` | MODIFY - Pass graph_state_initializer config |
| `fabricpc/training/multi_gpu.py` | MODIFY - Pass graph_state_initializer config |
| `tests/test_fabricpc.py` | MODIFY - Pass graph_state_initializer config |
| `tests/test_fabricpc_extended.py` | MODIFY - Pass graph_state_initializer config |
| `tests/test_auto_node_grad.py` | MODIFY - Pass graph_state_initializer config |
| `tests/test_ndim_shapes.py` | MODIFY - Pass graph_state_initializer config |
| `tests/test_initializers.py` | **CREATE** - Tests for initializer registry |
| `tests/test_state_initializer.py` | **CREATE** - Tests for state initialization |

---

## Configuration Examples

**Graph-level config:**
```python
graph_config = {
    "node_list": [...],
    "edge_list": [...],
    "graph_state_initializer": {
        "type": "distribution",
        "default_initializer": {"type": "normal", "mean": 0.0, "std": 0.01}
    }
}
```

**Node-level override:**
```python
{
    "name": "hidden1",
    "shape": (256,),
    "type": "linear",
    "latent_init": {"type": "uniform", "min": -0.5, "max": 0.5}
}
```

**Feedforward initialization:**
```python
state = initialize_state(
    structure, batch_size, key, clamps,
    {"type": "feedforward"},
    params=params  # Required for feedforward
)
```
