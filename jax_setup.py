"""JAX environment setup — must be importable without triggering JAX init."""

import os


def set_jax_flags_before_importing_jax(jax_platforms: str = "cuda"):
    """
    Set JAX flags for better performance and reproducibility.
    This should be called before importing JAX.
    """
    os.environ.setdefault("JAX_PLATFORMS", jax_platforms)  # "cpu", "cuda" or "tpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress XLA warnings

    # Keep deterministic kernels and default to disabling Triton GEMM, which can
    # trigger CUDA runtime errors on some GPUs for small/irregular matmuls.
    # Triton tiling logic fails when it encounters certain fused operations where dimension bounds are not divisible by the tile size.
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_deterministic_ops=true" not in _xla_flags:
        _xla_flags = (_xla_flags + " --xla_gpu_deterministic_ops=true").strip()
    if os.environ.get("FABRICPC_DISABLE_TRITON_GEMM", "1") == "1":
        if "--xla_gpu_enable_triton_gemm=false" not in _xla_flags:
            _xla_flags = (_xla_flags + " --xla_gpu_enable_triton_gemm=false").strip()

    # Set XLA flags for good performance and reproducibility
    _xla_flags = (_xla_flags + " --xla_gpu_autotune_level=1").strip()

    os.environ["XLA_FLAGS"] = _xla_flags
