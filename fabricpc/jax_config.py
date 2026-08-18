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

Backend initialization is therefore the deadline, and it is a silent one: once
the backend exists, every setting here is ignored and JAX reports nothing.
`setup_jax` detects that state and warns.
"""

import os
import warnings

# Importing this module imports jax. Deferring that into the function body would
# gain nothing: reaching `fabricpc.jax_config` executes `fabricpc/__init__`,
# which imports the package eagerly and jax with it.
import jax

try:
    # Private, so it is imported defensively: if a future jax drops or moves it,
    # the warning is lost but `setup_jax` keeps working.
    from jax._src.xla_bridge import backends_are_initialized
except ImportError:  # pragma: no cover - depends on installed jax internals

    def backends_are_initialized() -> bool:
        return False


def _flag_names(xla_flags: str) -> set[str]:
    """Flag names present in an `XLA_FLAGS` string, with values stripped.

    Matching on whole names rather than substrings keeps `--xla_gpu_foo` from
    being considered present because the caller set `--xla_gpu_foo_bar`.
    """
    return {token.split("=", 1)[0] for token in xla_flags.split()}


def setup_jax(platform: str | None = None) -> None:
    """
    Configure JAX for performance and reproducibility.

    Call before the first JAX computation. Import order does not matter.
    Calling after the backend has initialized warns and changes nothing.

    Args:
        platform: Platform to use ("cpu", "cuda", or "tpu").
            If None, JAX auto-detects available hardware.
    """
    if backends_are_initialized():
        warnings.warn(
            "setup_jax() ran after the JAX backend was initialized, so the "
            "platform selection, XLA flags, and memory settings it writes have "
            "no effect on this process. Call it before the first JAX "
            "computation or device query.",
            RuntimeWarning,
            stacklevel=2,
        )

    if platform is not None and "JAX_PLATFORMS" not in os.environ:
        # A platform set in the caller's shell wins over the argument, matching
        # the setdefault semantics the rest of this function uses.
        os.environ["JAX_PLATFORMS"] = platform  # inherited by subprocesses
        jax.config.update("jax_platforms", platform)  # applies to this process
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress XLA warnings

    # Each flag is guarded by name, not name=value: a value already present in
    # XLA_FLAGS — from the caller's shell or a previous setup_jax call — wins.
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    _present = _flag_names(_xla_flags)

    # `--xla_gpu_deterministic_ops` and `--xla_gpu_enable_triton_gemm` are
    # accepted by jaxlib but absent from XLA's stable flag list
    # (`XLA_FLAGS=--help`), unlike `--xla_gpu_autotune_level` below. XLA aborts
    # the process at backend initialization on an unrecognized flag
    # ("Unknown flag in XLA_FLAGS"), with no Python traceback, so a jaxlib that
    # drops either name turns every setup_jax() call into a hard abort. Pin jax
    # if that happens; see docs/user_guides/16_troubleshooting.md.
    if "--xla_gpu_deterministic_ops" not in _present:
        _xla_flags = (_xla_flags + " --xla_gpu_deterministic_ops=true").strip()
    # Triton GEMM is disabled by default because it can trigger CUDA runtime
    # errors on some GPUs for small or irregular matmuls: Triton's tiling logic
    # fails on certain fused operations whose dimension bounds are not divisible
    # by the tile size.
    if os.environ.get("FABRICPC_DISABLE_TRITON_GEMM", "1") == "1":
        if "--xla_gpu_enable_triton_gemm" not in _present:
            _xla_flags = (_xla_flags + " --xla_gpu_enable_triton_gemm=false").strip()

    # Autotune level 1 for reproducible kernel selection at low compile cost
    if "--xla_gpu_autotune_level" not in _present:
        _xla_flags = (_xla_flags + " --xla_gpu_autotune_level=1").strip()

    os.environ["XLA_FLAGS"] = _xla_flags
