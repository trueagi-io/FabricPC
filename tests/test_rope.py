"""
Test suite for Rotary Positional Embeddings (RoPE).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.positional import precompute_freqs_cis, apply_rotary_emb


class TestRoPE:

    @pytest.fixture
    def setup_data(self):
        batch_size = 2
        seq_len = 10
        n_head = 4
        head_dim = 16

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        xq = jax.random.normal(k1, (batch_size, seq_len, n_head, head_dim))
        xk = jax.random.normal(k2, (batch_size, seq_len, n_head, head_dim))

        freqs_cis = precompute_freqs_cis(head_dim, seq_len * 2)

        return xq, xk, freqs_cis, batch_size, seq_len, n_head, head_dim

    def test_shapes(self, setup_data):
        """Verify output shapes match input shapes exactly."""
        xq, xk, freqs_cis, _, _, _, _ = setup_data

        # Slice freqs to match input seq_len
        freqs_slice = freqs_cis[: xq.shape[1]]

        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_slice)

        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape

    def test_norm_preservation(self, setup_data):
        """
        RoPE is a rotation. Rotations preserve Euclidean norm.
        ||Rot(x)|| == ||x||
        """
        xq, xk, freqs_cis, _, _, _, _ = setup_data
        freqs_slice = freqs_cis[: xq.shape[1]]

        xq_out, _ = apply_rotary_emb(xq, xk, freqs_slice)

        # Compute norms along head_dim
        norm_in = jnp.linalg.norm(xq, axis=-1)
        norm_out = jnp.linalg.norm(xq_out, axis=-1)

        assert jnp.allclose(norm_in, norm_out, atol=1e-5)

    def test_zero_pos_identity(self, setup_data):
        """
        At position 0, the rotation angle should be 0.
        Therefore, the embedding should not change.
        """
        xq, xk, freqs_cis, _, _, _, _ = setup_data

        # Take just the first position (index 0)
        xq_pos0 = xq[:, 0:1, :, :]
        xk_pos0 = xk[:, 0:1, :, :]
        freqs_pos0 = freqs_cis[0:1]  # Freqs at pos 0

        xq_out, _ = apply_rotary_emb(xq_pos0, xk_pos0, freqs_pos0)

        assert jnp.allclose(
            xq_pos0, xq_out, atol=1e-6
        ), "Position 0 should not be rotated"

    def test_relative_position_invariance(self, setup_data):
        """
        The critical property of RoPE:
        Dot product depends on relative distance (k), not absolute position (t).

        q @ pos_t  dot  k @ pos_{t+k}  ==  q @ pos_{t'}  dot  k @ pos_{t'+k}
        """
        xq, xk, freqs_cis, _, _, n_head, head_dim = setup_data

        # Create a single query and single key vector
        q_vec = jnp.ones((1, 1, 1, head_dim))
        k_vec = jnp.ones((1, 1, 1, head_dim)) * 2

        offset = 5  # Relative distance

        # Case A: q at pos 0, k at pos 5
        freqs_A = jnp.stack([freqs_cis[0], freqs_cis[offset]])  # Extract pos 0 and 5
        # Feed them in as a sequence of length 2 to apply_rotary
        q_A = jnp.concatenate([q_vec, jnp.zeros_like(q_vec)], axis=1)
        k_A = jnp.concatenate([jnp.zeros_like(k_vec), k_vec], axis=1)

        q_out_A, k_out_A = apply_rotary_emb(q_A, k_A, freqs_A)

        # Dot product of q (pos 0) and k (pos 5)
        # q is at index 0, k is at index 1 (which corresponds to absolute pos 5 here)
        dot_A = jnp.sum(q_out_A[0, 0, 0] * k_out_A[0, 1, 0])

        # Case B: q at pos 10, k at pos 15 (Same offset of 5)
        start = 10
        freqs_B = jnp.stack([freqs_cis[start], freqs_cis[start + offset]])

        q_out_B, k_out_B = apply_rotary_emb(q_A, k_A, freqs_B)

        # Dot product of q (pos 10) and k (pos 15)
        dot_B = jnp.sum(q_out_B[0, 0, 0] * k_out_B[0, 1, 0])

        # The dot products should be equal
        assert jnp.allclose(
            dot_A, dot_B, atol=1e-5
        ), f"Relative position property failed. {dot_A} != {dot_B}"

    def test_pairwise_rotation_equivalence(self):
        """
        Sanity Check: Ensure RoPE actually rotates the vector at non-zero positions.
        """
        x = jnp.zeros((1, 1, 1, 4))
        x = x.at[..., 0].set(1.0)

        freqs = precompute_freqs_cis(4, 2)[1:2]
        x_rot, _ = apply_rotary_emb(x, x, freqs)

        # Check that rotation happened (vector changed)
        assert not jnp.allclose(x, x_rot)

        # Check that mass moved to the paired dimension (index 1)
        assert jnp.abs(x_rot[..., 1]) > 1e-5

        # Check pairwise independence (indices 2,3 should remain 0)
        assert jnp.allclose(x_rot[..., 2:], 0.0, atol=1e-6)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
