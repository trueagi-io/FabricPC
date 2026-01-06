# Changelog

## [0.2.6] - 2026-01-06
- Fixed multi-GPU training to correctly use graph state initializer from GraphStructure config.
- Aligned gradient computation in multi-GPU training with single-GPU Hebbian learning.

## [0.2.5] - 2025-12-25
- Added v1 TransformerBlockNode encapsulating multi-head attention, layer normalization, and feedforward networks using Rotary Position Embeddings (RoPE)
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