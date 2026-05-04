from jax import numpy as jnp


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
