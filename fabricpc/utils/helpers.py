import jax.numpy as jnp
from fabricpc.core.types import GraphState


def update_node_in_state(state: GraphState, node_name: str, **updates) -> GraphState:
    """
    Update a node's fields in GraphState.
    Args:
        state: Current GraphState
        node_name: Name of the node to update
        **updates: kv args fields to update in the NodeState
    Returns:
        New GraphState with updated node
    Usage:
    state = update_node_in_state(state, "hidden", latent_grad=grad)
    """
    updated_node = state.nodes[node_name]._replace(**updates)
    return state._replace(nodes={**state.nodes, node_name: updated_node})


def set_jax_flags_before_importing_jax(jax_platforms: str = "cuda"):
    """
    Set JAX/XLA environment flags before importing JAX.

    This keeps example scripts self-contained and avoids repeating the same
    environment setup logic across entrypoints.
    """
    import os

    os.environ.setdefault("JAX_PLATFORMS", jax_platforms)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_deterministic_ops=true" not in xla_flags:
        xla_flags = (xla_flags + " --xla_gpu_deterministic_ops=true").strip()
    if os.environ.get("FABRICPC_DISABLE_TRITON_GEMM", "1") == "1":
        if "--xla_gpu_enable_triton_gemm=false" not in xla_flags:
            xla_flags = (xla_flags + " --xla_gpu_enable_triton_gemm=false").strip()
    xla_flags = (xla_flags + " --xla_gpu_autotune_level=1").strip()
    os.environ["XLA_FLAGS"] = xla_flags


def layernorm(
    x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5
) -> jnp.ndarray:
    """
    Layer normalization.
    Args:
        x: Input tensor
        gamma: Scale parameter
        beta: Shift parameter
        eps: Epsilon for numerical stability
    Returns:
        Normalized tensor
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(variance + eps) + beta
