"""Error-based predictive coding inference for directed acyclic graphs."""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.inference import InferenceBase, gather_inputs
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeState
from fabricpc.graph_assembly.scheduling import GraphCycleError


class EPCInference(InferenceBase):
    """Error-based predictive coding (ePC) for strict DAG topologies.

    ePC relaxes prediction errors and derives latent states through one global,
    differentiable forward program. muPC forward scaling remains inside that
    program, while sPC's per-hop gradient preconditioners are intentionally not
    applied: reverse-mode autodiff supplies the complete chain rule directly.

    The first release is deliberately DAG-only. Cycle-capable topology
    schedulers remain available for sPC initialization, but this solver rejects
    all back edges and repeated node visits.
    """

    def __init__(self, eta_infer=0.1, infer_steps=5, latent_decay=0.0):
        super().__init__(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            latent_decay=latent_decay,
        )

    @staticmethod
    def _validate_dag_topology(structure: GraphStructure) -> None:
        if len(structure.schedule) != len(structure.node_order):
            raise GraphCycleError(
                "EPCInference supports one visit per node on a DAG; the configured "
                "topology schedule contains repeated visits. Use an sPC solver for "
                "this cyclic graph."
            )

        position = {name: index for index, name in enumerate(structure.node_order)}
        offending_edges = tuple(
            edge.key
            for edge in structure.edges.values()
            if position[edge.source] >= position[edge.target]
        )
        if offending_edges:
            raise GraphCycleError(
                "EPCInference supports DAGs only; the configured order contains "
                f"back edges: {offending_edges}. Use an sPC solver for this graph."
            )

    @staticmethod
    def _relaxed_partition(
        structure: GraphStructure,
        clamps: Dict[str, jnp.ndarray],
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        error_nodes = []
        latent_nodes = []
        for node_name, node in structure.nodes.items():
            info = node.node_info
            if node_name in clamps:
                continue
            if info.in_degree == 0:
                latent_nodes.append(node_name)
            elif info.out_degree > 0:
                error_nodes.append(node_name)
        return tuple(error_nodes), tuple(latent_nodes)

    @classmethod
    def derive_states(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """Derive all node states from relaxed errors in topological order."""
        cls._validate_dag_topology(structure)
        for node_name in structure.schedule:
            node = structure.nodes[node_name]
            node_info = node.node_info
            node_state = state.nodes[node_name]
            inputs = gather_inputs(node_info, structure, state)
            inputs = scale_inputs(inputs, node_info.scaling_config)
            node_state = node_info.node_class.forward_from_error(
                params.nodes[node_name],
                inputs,
                node_state,
                node_info,
                is_clamped=(node_name in clamps),
            )
            state = state._replace(nodes={**state.nodes, node_name: node_state})
        return state

    @classmethod
    def begin_segment(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """Synchronize epsilon with an incoming latent-parameterized state."""
        cls._validate_dag_topology(structure)
        error_nodes, _ = cls._relaxed_partition(structure, clamps)
        for node_name in error_nodes:
            node_info = structure.nodes[node_name].node_info
            inputs = gather_inputs(node_info, structure, state)
            inputs = scale_inputs(inputs, node_info.scaling_config)
            node_state = node_info.node_class.forward(
                params.nodes[node_name],
                inputs,
                state.nodes[node_name],
                node_info,
            )
            state = state._replace(nodes={**state.nodes, node_name: node_state})
        return state

    @classmethod
    def forward_value_and_grad(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """Derive the graph and differentiate total energy globally over epsilon."""
        error_nodes, latent_nodes = cls._relaxed_partition(structure, clamps)
        relaxed = {
            "error": {name: state.nodes[name].error for name in error_nodes},
            "z_latent": {name: state.nodes[name].z_latent for name in latent_nodes},
        }

        def energy_from_relaxed(relaxed_values):
            working_state = state
            for name, error in relaxed_values["error"].items():
                working_state = update_node_in_state(working_state, name, error=error)
            for name, z_latent in relaxed_values["z_latent"].items():
                working_state = update_node_in_state(
                    working_state, name, z_latent=z_latent
                )

            derived_state = cls.derive_states(params, working_state, clamps, structure)
            energy_terms = [
                jnp.sum(derived_state.nodes[name].energy)
                for name, node in structure.nodes.items()
                if node.node_info.in_degree > 0
            ]
            if energy_terms:
                total_energy = energy_terms[0]
                for term in energy_terms[1:]:
                    total_energy = total_energy + term
            else:
                total_energy = jnp.asarray(0.0, dtype=jnp.float32)
            return total_energy, derived_state

        (_, derived_state), relaxed_grads = jax.value_and_grad(
            energy_from_relaxed, has_aux=True
        )(relaxed)

        for name in error_nodes:
            node_state = derived_state.nodes[name]
            derived_state = update_node_in_state(
                derived_state,
                name,
                latent_grad=node_state.latent_grad + relaxed_grads["error"][name],
            )
        for name in latent_nodes:
            node_state = derived_state.nodes[name]
            derived_state = update_node_in_state(
                derived_state,
                name,
                latent_grad=node_state.latent_grad + relaxed_grads["z_latent"][name],
            )
        return derived_state

    @staticmethod
    def compute_new_latent(
        node_name: str,
        node_state: NodeState,
        config: Dict[str, Any],
    ) -> jnp.ndarray:
        del node_name
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]
        return (
            node_state.z_latent * (1.0 - eta_infer * latent_decay)
            - eta_infer * node_state.latent_grad
        )

    @staticmethod
    def compute_new_error(
        node_name: str,
        node_state: NodeState,
        config: Dict[str, Any],
    ) -> jnp.ndarray:
        del node_name
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]
        return (
            node_state.error * (1.0 - eta_infer * latent_decay)
            - eta_infer * node_state.latent_grad
        )

    @classmethod
    def update_latents(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        del params
        error_nodes, latent_nodes = cls._relaxed_partition(structure, clamps)
        for name in error_nodes:
            new_error = cls.compute_new_error(name, state.nodes[name], config)
            state = update_node_in_state(state, name, error=new_error)
        for name in latent_nodes:
            new_latent = cls.compute_new_latent(name, state.nodes[name], config)
            state = update_node_in_state(state, name, z_latent=new_latent)
        return state

    @classmethod
    def finalize_state(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """Rebuild values and energies at the final relaxed point."""
        return cls.derive_states(params, state, clamps, structure)
