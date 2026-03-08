import jax
import jax.numpy as jnp
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> jnp.ndarray:
    """
    Precompute the frequency tensor for complex exponentials (cis).
    """
    freqs = 1.0 / (
        theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim)
    )
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs)  # (seq_len, dim/2)
    freqs_cis = jnp.exp(1j * freqs)  # (seq_len, dim/2)
    return freqs_cis


def apply_rotary_emb(
    xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply RoPE to queries and keys.
    Expects xq, xk shape: (batch, seq_len, n_head, head_dim)
    freqs_cis shape: (seq_len, head_dim/2)
    """
    # Reshape last dim to pairs: (batch, seq_len, n_head, head_dim//2, 2)
    xq_pairs = xq.reshape(xq.shape[:-1] + (xq.shape[-1] // 2, 2))
    xk_pairs = xk.reshape(xk.shape[:-1] + (xk.shape[-1] // 2, 2))

    # Convert to complex: (batch, seq_len, n_head, head_dim//2)
    xq_complex = xq_pairs[..., 0] + 1j * xq_pairs[..., 1]
    xk_complex = xk_pairs[..., 0] + 1j * xk_pairs[..., 1]

    # Reshape freqs_cis for broadcasting: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis[None, :, None, :]

    # Apply rotation
    xq_rotated = xq_complex * freqs_cis
    xk_rotated = xk_complex * freqs_cis

    # Convert back to real: (batch, seq_len, n_head, head_dim//2, 2)
    xq_real = jnp.stack([xq_rotated.real, xq_rotated.imag], axis=-1)
    xk_real = jnp.stack([xk_rotated.real, xk_rotated.imag], axis=-1)

    # Reshape back to original: (batch, seq_len, n_head, head_dim)
    xq_out = xq_real.reshape(xq.shape)
    xk_out = xk_real.reshape(xk.shape)

    return xq_out, xk_out
