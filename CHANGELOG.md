# Changelog

## [0.3.1] - 2026-05-04
Internal infrastructure release: unified autodiff gradient path, muPC scaling lifted to callsites, and a package restructure that resolves circular import.

### Breaking changes — downstream migration guide
**Import path migrations.** The `builder` package is gone; topology primitives live in `core`, the assembly entry point lives in `graph_assembly`, and `graph` is renamed to `graph_initialization`. Mechanical replacements:
- `from fabricpc.builder import Edge` → `from fabricpc.core.topology import Edge`
- `from fabricpc.builder import SlotRef, GraphNamespace` → `from fabricpc.core.topology import SlotRef, GraphNamespace`
- `from fabricpc.builder import graph, TaskMap` → `from fabricpc.graph_assembly import graph, TaskMap`
- `from fabricpc.graph import initialize_params` → `from fabricpc.graph_initialization import initialize_params` (also re-exported from `fabricpc`)
- `from fabricpc.graph.state_initializer import ...` → `from fabricpc.graph_initialization.state_initializer import ...`
- `from fabricpc.graph.graph_net import compute_local_weight_gradients` → `from fabricpc.core.learning import compute_local_weight_gradients`
- `from fabricpc.utils.helpers import update_node_in_state, set_latents_to_clamps` → `from fabricpc.core.state_ops import ...` (`layernorm` stays in `utils.helpers`)

**Node API renames.** Methods on `NodeBase` (and any subclass that overrides them):
- `forward_inference(...)` → `forward_and_latent_grads(...)`. **Return signature changed** from `(NodeState, input_grads)` to `(NodeState, input_grads, self_grad)`. The third value is `dE/dz_latent` for this node only, unscaled; the inference loop scales it and accumulates into `state.latent_grad`. Subclasses that override this method must return the third value.
- `forward_learning(...)` → `forward_and_weight_grads(...)`.

**muPC scaling lifted out of nodes.** `NodeBase._apply_forward_scaling` is removed. Node forward/grad methods are now pure autodiff. Pre-scaling of inputs and post-scaling of input/self/weight grads are applied by the inference and learning loops via `fabricpc.core.scaling.{scale_inputs, scale_input_grads, scale_self_grad, scale_weight_grads}`. Custom nodes with a hand-written `forward_inference`/`forward_learning` override should drop any internal scaling and follow the new contract; see `nodes/linear_explicit_grad.py` (extracted from `linear.py`) for the reference pattern.
**muPC contract for non-variance-scalable slots changed.** Edges arriving at slots with `is_variance_scalable=False` are now **omitted** from `MuPCScalingFactors.{forward_scale, topdown_grad_scale, weight_grad_scale}` rather than populated with 1.0. Callsites treat missing keys as no-op pass-through. This preserves input dtype across the boundary (an `x * 1.0` previously promoted integer token indices to float). Forks that read these dicts directly must use `dict.get(k, 1.0)` or membership checks.
**Integer clamps now flow through to terminal source nodes.** State initializers propagate the clamp dtype onto `z_latent` for clamped nodes; other `NodeState` fields stay float. Callers feeding `EmbeddingNode` should clamp with integer dtype (e.g. `jnp.int32` token indices) — `EmbeddingNode.forward` no longer casts internally, and `train_autoregressive._generation_step` no longer casts indices to float. The `EmbeddingNode` "in" slot is now `is_variance_scalable=False`.
**`StorkeyHopfield`.** `accumulate_hopfield_energy_and_grad(...)` → `accumulate_hopfield_energy(...)`. The Hopfield latent gradient is no longer accumulated manually — autodiff in `forward_and_latent_grads` handles it.
**Removed duplicates / dead code.** `compute_local_weight_gradients_ar` (was a near-duplicate of `compute_local_weight_gradients`), `GraphStructure._topological_sort` (duplicate of the canonical implementation in `graph_assembly`), and the empty `fabricpc/graph_initialization/graph_net.py` shim are gone.
**Other.** `LinearExplicitGrad` moved from `fabricpc/nodes/linear.py` to `fabricpc/nodes/linear_explicit_grad.py` (still re-exported from `fabricpc.nodes`). Forced `float32` dtype removed from state initialization. RNG variable renamed: `node_keys` → `rng_keys`. New `ActivationBase.jacobian()` hook with `SoftmaxActivation.jacobian()` implemented for explicit-gradient overrides.
### Verification
`pytest tests/ -x`: 127 passed. Demos (`mnist_demo.py`, `transformer_v2_demo.py`, `resnet18_cifar10_demo.py`) run clean.

## [0.3.0] - 2026-04-17
- muPC scaling supports arbitrary DAG topologies with correct per-edge scaling, per-slot computation. Scaling formula is `a = gain / sqrt(fan_in * K_slot * L)` where K_slot is the per-slot in-degree and L is the residual depth (number of nodes with skip connection slots along the longest path).
- Stable training demonstrated on networks with 100+ layers with muPC scaling. 
- Associative memory is now a composable network component with `StorkeyHopfield` node: combines PC prediction-error energy with Hopfield attractor energy.
- Consolidated multi-GPU trainer into `train.py`.
- Comprehensive documentation in docs/user_guides folder.
- Added `is_variance_scalable` and `is_skip_connection` attributes to `SlotSpec` for fine-grained control over which edges receive muPC scaling.
- Added `SkipConnection` node: passthrough node with `is_variance_scalable=False` for residual/skip paths. Prevents exponential signal decay in deep residual networks.
- Added `LinearResidual` node: combines linear transform and +skip sum in one PC node with dual slots ("in" scaled, "skip" unscaled). Halves graph depth compared to Linear + SkipConnection pattern.
- Added `jacobian_gain()` to activation functions for gradient compensation in deep networks with saturating activations (tanh, GELU, HardTanh).
- Improved internal variance scaling in TransformerBlock with 1/sqrt(2) residual connections and position-dependent attention variance compensation.

## [0.2.9] - 2026-03-17
- Added transformer_v2 nodes and example decomposing transformer blocks to use PC inference at the attention and feedfordward layers. See examples/transformer_v2_demo.py for details.
- Improved training stability and inference convergence of the v1 transformer block by gradient clipping and residual connections. See examples/transformer_demo.py for details.
- Refactored optimizer integration to use Optax directly. Trainer signature is now train_pcn(..., optimizer=optax.adamw(0.001, weight_decay=0.001))
- Refactored nodes to use weight initializer objects instead of config dicts. New API is node = Linear(shape=(128,), ..., weight_init=XavierInitializer())
- Refactored inference to use algorithm abstraction. New API is structure = graph(nodes=[...], edges=[...], task_map, inference=InferenceSGD(eta_infer=0.05, infer_steps=20))
- Refactored Aim TrackingConfig parameters to improve configurable logging intervals.
- Added ABExperiment class for comparing model variants statistically.
- Added a fixed scaling factor argument to IdentityNode for better control over signal propagation.

## [0.2.8] - 2026-02-25
- Refactored model definition to be object based rather than purely config based. Existing model configs can be easily adapted to new format. See examples folder.
- Nodes now require class constructors instead of config dicts. Activation functions should be called like type(actfn_instance).forward(x, actfn_instance.config);
- Removed registry pattern for nodes, energy functionals, and other components in favor of explicit imports and class constructors. No registration decorators.

## [0.2.7] - 2026-02-18
- Add JAX-compatible MNIST data loader. Removed pytorch dependency from project.
- Enhanced documentation and comments across multiple files for clarity. Refactored inference to ignore energy of nodes that do not have energy (e.g. terminal input nodes).
- Added Aim integration for comprehensive experiment tracking and visualization. docs/user_guides/aim_tensorboard_guide.md provides instructions for setting up Aim and using it with FabricPC.

## [0.2.6] - 2026-01-06
- Fixed multi-GPU training to correctly use graph state initializer from GraphStructure config.
- Aligned gradient computation in multi-GPU training with single-GPU Hebbian learning.

## [0.2.5] - 2025-12-25
- Added v1 TransformerBlock encapsulating multi-head attention, layer normalization, and feedforward networks using Rotary Position Embeddings (RoPE)
- Refactored state initialization: renames "distribution" to "global", adds "node_distribution", and removes fallback configurations.
- Unifies output metric computation across training modules and returns both energy and cross-entropy for autoregressive training.

## [0.2.4] - 2025-12-24
- Added support for custom initializers with registry pattern. Introduced `InitializerBase` and `StateInitializerBase` classes for extensibility.
- Replaced initialize_weights() and initialize_state_values() with fabricpc.core.initializers.initialize() function.
- Added config attribute to GraphStructure class and field "graph_state_initializer".

## [0.2.3] - 2025-12-18
- Change Linear node default behavior to perform matmul on the last tensor dimension. Flattening inputs now requires flag `flatten_input=True`.
- Removed gain_mod_error from NodeState, as it was not used by anything other than explicit grad linear node.
- Added softmax and Gelu activation functions.
- Added KL Divergence energy functional.

## [0.2.2] - 2025-12-05
- Unified config validation and registry pattern across nodes, energy functionals, and activations
- Custom objects now follow a consistent extensibility pattern with `CONFIG_SCHEMA` and `@register_*` decorators
- Node construction delegated to `NodeBase.from_config()` for cleaner separation of concerns
- CONFIG_SCHEMA is now a required class variable for easier access and introspection

## [0.2.1] - 2025-12-04
- Node autograd is the default behavior now; can override by subclassing a node and implementing manual gradients
- N-dimensional tensor support: breaking changes to shape conventions
  - Linear nodes: shape=(features,) e.g., (128,) for 128-dimensional vector
  - 2D Conv nodes: shape=(H, W, C) e.g., (28, 28, 64) for 28x28 image with 64 channels (NHWC)
- Plugin architecture for custom nodes with two choices for registration: decorator or setuptools entry points