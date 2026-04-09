"""
Storkey Hopfield associative memory node for predictive coding networks.

Implements a Hopfield memory layer that combines standard PC prediction-error
energy with a Hopfield attractor energy term. The Hopfield energy pulls the
node's latent state z toward stored patterns (attractors in z-space), while
the PC energy pulls z toward the upstream prediction mu. The equilibrium z*
is the PC-optimal compromise between top-down expectation and internal memory.

Energy formulation:
    E_total = E_pc + hopfield_strength * E_hop

    E_pc = 0.5 ||z - mu||^2              (Gaussian, or user-specified)
    E_hop = (1/2N) z^T (W^2 - W) z       (Hopfield attractor energy)

    where N = dimension of z (last axis) for scale-invariance.

The standard PC energy path (via energy_functional()) is called normally.
The Hopfield energy is added afterward via accumulate_hopfield_energy_and_grad(),
which augments state.energy and state.latent_grad. The Hopfield recurrence
(z_latent @ W) is internal to the node and invisible to the graph topology.

The input from the upstream node serves as a "probe pattern" — it is added
directly to the pre-activation (no weight multiplication), seeding the
attractor dynamics. W encodes only the associative memory structure.

Notation:
    xi^mu = stored Hopfield patterns (absolute states in z-space)
    epsilon = z - mu = PC prediction error (context-dependent residual)
    These are completely different objects. The Hopfield attractor operates
    on z (absolute state), not on epsilon (prediction error).

Architecture (internal to forward()):

    probe ------+
                |
                v
              [sum] + [strength * (z_latent @ W)] + bias ----> tanh ----> z_mu
                                    ^
                              Hopfield recurrence
                              (pseudo self-connection)
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import ZerosInitializer, NormalInitializer

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class StorkeyHopfield(NodeBase):
    """
    Hopfield associative memory node with energy-based learning.

    Combines standard PC prediction-error energy with a Hopfield attractor
    energy term. The two energies compete on z_latent:

    - E_pc pulls z toward the upstream prediction mu
    - E_hop pulls z toward stored patterns (attractors learned via W)

    The Hopfield recurrence is internal: z_mu = tanh(probe + strength * z @ W + bias).
    W is a (D, D) weight matrix operating on the last axis of z_latent.
    The probe (input from upstream) is added directly without W multiplication.

    Args:
        shape: Output shape (excluding batch). Last dim is the Hopfield dimension D.
        name: Node name.
        activation: Activation function (default: TanhActivation).
        energy: Energy functional for PC term (default: GaussianEnergy).
        hopfield_strength: Scaling of E_hop relative to E_pc.
            If None (default), a learnable scalar initialized to 1.0.
            If a float, fixed at that value.
        use_bias: Whether to include bias (default: True).
        enforce_symmetry: Symmetrize W via 0.5*(W+W.T) in forward (default: True).
        zero_diagonal: Zero W diagonal in forward (default: False).
        weight_init: Initializer for weights (default: NormalInitializer(0, 0.01)).
        latent_init: Initializer for latent states (default: NormalInitializer()).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: Optional[ActivationBase] = TanhActivation(),
        energy: Optional[EnergyFunctional] = GaussianEnergy(),
        hopfield_strength: Optional[float] = None,
        use_bias: bool = True,
        enforce_symmetry: bool = True,
        zero_diagonal: bool = False,
        latent_init: Optional[InitializerBase] = NormalInitializer(),
        weight_init: Optional[InitializerBase] = ZerosInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            hopfield_strength=hopfield_strength,
            use_bias=use_bias,
            enforce_symmetry=enforce_symmetry,
            zero_diagonal=zero_diagonal,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """One single-input slot (probe). Input shape must match node output shape (D)."""
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize Hopfield W matrix and biases.

        Weights:
            - Hopfield W: (D, D) stored under the input edge key

        Biases:
            - "b": bias vector if use_bias
            - "hopfield_strength": learnable scalar if hopfield_strength is None
        """
        if config is None:
            config = {}
        from fabricpc.core.initializers import initialize

        if weight_init is None:
            weight_init = ZerosInitializer()

        # Weights
        weights_dict = {}
        D = node_shape[-1]
        key_hop, key_b = jax.random.split(
            key,
        )

        # Hopfield W: (D, D)
        W = initialize(key_hop, (D, D), weight_init)
        # Apply symmetry/diagonal constraints at init time too
        W = StorkeyHopfield._prepare_W(W, config)

        # get the first dictionary key and shape (there should only be one input edge for this node)
        edge_key, in_shape = next(iter(input_shapes.items()))
        if in_shape[-1] != D:
            raise ValueError(
                f"Input last dimension {in_shape[-1]} must match node output last dimension {D} for Hopfield recurrence."
            )
        weights_dict[edge_key] = (
            W  # Store W under the input edge key for gradient flow to presynaptic node.
        )

        # Biases
        biases = {}
        # Initialize bias (usually zeros)
        # Bias shape for proper broadcasting, prepending batch dimension: (1, ..., 1, out_features)
        use_bias = config.get("use_bias", True)
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            biases["b"] = jnp.zeros(bias_shape)

        # Learnable hopfield_strength if not fixed
        hopfield_strength = config.get("hopfield_strength", None)
        if hopfield_strength is None:
            biases["hopfield_strength"] = jnp.array(1.0)

        return NodeParams(weights=weights_dict, biases=biases)

    @staticmethod
    def _prepare_W(W: jnp.ndarray, config: Dict[str, Any]) -> jnp.ndarray:
        """Symmetrize and/or zero diagonal of W based on config.

        Both operations are differentiable, safe under JAX tracing.
        """
        D = W.shape[0]
        if config.get("enforce_symmetry", True):
            W = 0.5 * (W + W.T)
        if config.get("zero_diagonal", False):
            W = W * (1.0 - jnp.eye(D))
        return W

    @staticmethod
    def accumulate_hopfield_energy_and_grad(
        state: NodeState,
        W: jnp.ndarray,
        strength: jax.Array,
    ) -> NodeState:
        """Add Hopfield attractor energy E = (1/2N) z^T (W^2 - W) z to state.

        The Hopfield energy pulls z toward stored patterns (attractors in
        z-space). Combined with the PC energy (which pulls z toward the
        upstream prediction mu), the equilibrium z* is the PC-optimal
        compromise between top-down expectation and internal memory prior.

        Args:
            state: NodeState with z_latent set.
            W: Prepared (D, D) Hopfield weight matrix.
            strength: Scalar hopfield_strength (learnable jnp.array or fixed float).

        Returns:
            Updated NodeState with Hopfield energy and latent_grad added.
        """
        z = state.z_latent  # (batch, ..., D)
        wz = z @ W  # (batch, ..., D)
        D = z.shape[-1]
        E_h = (0.5 / D) * jnp.sum(wz * (wz - z), axis=-1)  # (1/2N) z^T(W^2-W)z
        hopfield_grad = (strength / D) * (wz @ W - wz)  # (1/N)(W^2-W)z
        return state._replace(
            energy=state.energy + strength * E_h,
            latent_grad=state.latent_grad + hopfield_grad,
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        """
        Forward pass: compute z_mu with Hopfield recurrence, then combined energy.

        z_mu = activation(probe + strength * (z_latent @ W) + bias)

        Energy = E_pc(z, z_mu) + hopfield_strength * E_hop(W, z)
        """
        config = node_info.node_config
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape

        edge_key, input_probe_state = next(iter(inputs.items()))

        # Prepare Hopfield W
        W = StorkeyHopfield._prepare_W(params.weights[edge_key], config)

        # Resolve hopfield_strength: learnable or fixed
        if "hopfield_strength" in params.biases:
            strength = params.biases["hopfield_strength"]
        else:
            strength = config.get("hopfield_strength", 1.0)

        pre_activation = jnp.zeros((batch_size,) + out_shape)
        # Probe pattern: added directly (no W multiplication)
        pre_activation = pre_activation + input_probe_state
        # Hopfield recurrence: strength * (z_latent @ W)
        pre_activation = pre_activation + strength * (state.z_latent @ W)

        # Add bias
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation (tanh by default)
        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)

        # Prediction error
        error = state.z_latent - z_mu

        # Update state
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)

        # Standard PC energy via energy_functional
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)

        # Add Hopfield attractor energy
        state = StorkeyHopfield.accumulate_hopfield_energy_and_grad(state, W, strength)

        total_energy = jnp.sum(state.energy)
        return total_energy, state
