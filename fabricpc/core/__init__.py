"""Core JAX predictive coding components."""

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

from fabricpc.core.activations import (
    ActivationBase,
    IdentityActivation,
    SigmoidActivation,
    TanhActivation,
    ReLUActivation,
    LeakyReLUActivation,
    GeluActivation,
    SoftmaxActivation,
    HardTanhActivation,
    get_activation,
)

from fabricpc.core.energy import (
    EnergyFunctional,
    GaussianEnergy,
    BernoulliEnergy,
    CrossEntropyEnergy,
    LaplacianEnergy,
    HuberEnergy,
    KLDivergenceEnergy,
    get_energy_and_gradient,
)

from fabricpc.core.inference import gather_inputs, inference_step, run_inference

from fabricpc.core.initializers import (
    InitializerBase,
    ZerosInitializer,
    OnesInitializer,
    NormalInitializer,
    UniformInitializer,
    XavierInitializer,
    KaimingInitializer,
    initialize,
)

__all__ = [
    "GraphParams",
    "GraphState",
    "GraphStructure",
    "NodeInfo",
    "EdgeInfo",
    "SlotInfo",
    "NodeParams",
    "NodeState",
    "ActivationBase",
    "IdentityActivation",
    "SigmoidActivation",
    "TanhActivation",
    "ReLUActivation",
    "LeakyReLUActivation",
    "GeluActivation",
    "SoftmaxActivation",
    "HardTanhActivation",
    "get_activation",
    "EnergyFunctional",
    "GaussianEnergy",
    "BernoulliEnergy",
    "CrossEntropyEnergy",
    "LaplacianEnergy",
    "HuberEnergy",
    "KLDivergenceEnergy",
    "get_energy_and_gradient",
    "gather_inputs",
    "inference_step",
    "run_inference",
    "InitializerBase",
    "ZerosInitializer",
    "OnesInitializer",
    "NormalInitializer",
    "UniformInitializer",
    "XavierInitializer",
    "KaimingInitializer",
    "initialize",
]
