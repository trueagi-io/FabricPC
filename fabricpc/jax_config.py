"""JAX environment configuration for FabricPC.

Every setting `setup_jax` writes is consumed at JAX *backend initialization* —
the first computation or device query — not at `import jax`, with one
exception: the `JAX_PLATFORMS` environment variable is read once, while jax is
imported. `setup_jax` therefore sets the platform two ways: in `os.environ`, so
spawned worker processes inherit it, and through `jax.config`, which selects the
platform in a process that has already imported jax.
"""

import os

# Importing this module imports jax. Deferring that into the function body would
# gain nothing: reaching `fabricpc.jax_config` executes `fabricpc/__init__`,
# which imports the package eagerly and jax with it.
import jax


def setup_jax(platform: str | None = None) -> None:
    """
    Configure JAX for performance and reproducibility.

    Call before the first JAX computation. Import order does not matter.

    Args:
        platform: Platform to use ("cpu", "cuda", or "tpu").
            If None, JAX auto-detects available hardware.
    """
    if platform is not None and not os.environ.get("JAX_PLATFORMS"):
        # A platform set in the caller's shell wins over the argument, keeping
        # the setdefault semantics the rest of this function uses.
        os.environ["JAX_PLATFORMS"] = platform  # inherited by subprocesses
        jax.config.update("jax_platforms", platform)  # applies to this process
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
