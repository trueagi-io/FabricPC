"""JAX environment configuration for FabricPC.

The settings `setup_jax` writes are consumed at JAX *backend initialization* —
the first computation or device query — not at `import jax`, with two
exceptions. `JAX_PLATFORMS` is read from the environment once, while jax is
imported, so `setup_jax` sets the platform two ways: in `os.environ`, so
spawned worker processes inherit it, and through `jax.config`, which selects the
platform in a process that has already imported jax. `TF_CPP_MIN_LOG_LEVEL` is
cached when the native runtime emits its first log message, so it suppresses
messages from backend initialization onward but not any emitted during
`import jax`.
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
    # Each flag is guarded by name, not name=value: a value already present in
    # XLA_FLAGS — from the caller's shell or a previous setup_jax call — wins.
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_deterministic_ops" not in _xla_flags:
        _xla_flags = (_xla_flags + " --xla_gpu_deterministic_ops=true").strip()
    if os.environ.get("FABRICPC_DISABLE_TRITON_GEMM", "1") == "1":
        if "--xla_gpu_enable_triton_gemm" not in _xla_flags:
            _xla_flags = (_xla_flags + " --xla_gpu_enable_triton_gemm=false").strip()

    # Autotune level 1 for reproducible kernel selection at low compile cost
    if "--xla_gpu_autotune_level" not in _xla_flags:
        _xla_flags = (_xla_flags + " --xla_gpu_autotune_level=1").strip()

    os.environ["XLA_FLAGS"] = _xla_flags
