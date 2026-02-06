"""Core JAX predictive coding components."""

# Type definitions
from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeInfo,
    EdgeInfo,
    SlotInfo,
    NodeParams,
    NodeState,
)

# Activation functions
from fabricpc.core.activations import (
    ActivationBase,
    get_activation,
    get_activation_fn,
    get_activation_deriv,
    register_activation,
    get_activation_class,
    list_activation_types,
)

# Energy functions
from fabricpc.core.energy import (
    EnergyFunctional,
    compute_energy,
    compute_energy_gradient,
    get_energy_and_gradient,
    register_energy,
    get_energy_class,
    list_energy_types,
)

# Inference functions
from fabricpc.core.inference import (
    gather_inputs,
    inference_step,
    run_inference,
)

# Initializer registry
from fabricpc.core.initializers import (
    InitializerBase,
    register_initializer,
    get_initializer_class,
    list_initializer_types,
    initialize,
)

# Config utilities
from fabricpc.core.config import (
    validate_config,
    transform_shorthand,
)

__all__ = [
    # Types
    "GraphParams",
    "GraphState",
    "GraphStructure",
    "NodeInfo",
    "EdgeInfo",
    "SlotInfo",
    "NodeParams",
    "NodeState",
    # Activation functions
    "ActivationBase",
    "get_activation",
    "get_activation_fn",
    "get_activation_deriv",
    "register_activation",
    "get_activation_class",
    "list_activation_types",
    # Energy functions
    "EnergyFunctional",
    "compute_energy",
    "compute_energy_gradient",
    "get_energy_and_gradient",
    "register_energy",
    "get_energy_class",
    "list_energy_types",
    # Inference
    "gather_inputs",
    "inference_step",
    "run_inference",
    # Initializer registry
    "InitializerBase",
    "register_initializer",
    "get_initializer_class",
    "list_initializer_types",
    "initialize",
    # Config utilities
    "validate_config",
    "transform_shorthand",
]
