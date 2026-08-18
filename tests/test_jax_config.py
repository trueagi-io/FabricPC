"""Contract test for `fabricpc.jax_config.setup_jax`.

`setup_jax` is documented as callable *after* `import jax`, any time before the
first JAX computation. Each case runs in a fresh interpreter: JAX reads
`JAX_PLATFORMS` once, while it is imported, and initializes its backend once
per process, so neither behavior can be exercised twice in one test session.
"""

import json
import os
import subprocess
import sys

# Stripped from the child environment so the parent's conftest settings and the
# developer's shell cannot decide the outcome.
_STRIPPED = ("JAX_PLATFORMS", "XLA_FLAGS", "FABRICPC_DISABLE_TRITON_GEMM")

_PROBE = """\
import json, os
import jax  # already imported when the helper runs — that is what is under test
from fabricpc.jax_config import setup_jax

for _ in range({repeat}):
    setup_jax({call_arg})

payload = {{
    "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    "xla_flags": os.environ.get("XLA_FLAGS", ""),
}}
{device_line}
print(json.dumps(payload))
"""

_QUERY_DEVICES = 'payload["platforms"] = sorted({d.platform for d in jax.devices()})'

# The deadline case: the backend is already up, so every setting is discarded.
# Devices are compared before and after rather than against a literal, so the
# case reads the same on a GPU host and on a CPU-only runner.
_LATE_PROBE = """\
import json, warnings
import jax
from fabricpc.jax_config import setup_jax

jax.numpy.ones(3).sum().block_until_ready()
before = sorted({d.platform for d in jax.devices()})

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    setup_jax("cpu")

print(json.dumps({
    "warnings": [str(w.message) for w in caught if w.category is RuntimeWarning],
    "before": before,
    "after": sorted({d.platform for d in jax.devices()}),
}))
"""

# The same call placed correctly, to pin that the warning is specific to the
# deadline and not emitted on every call.
_EARLY_PROBE = """\
import json, warnings
import jax
from fabricpc.jax_config import setup_jax

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    setup_jax("cpu")

print(json.dumps({
    "warnings": [str(w.message) for w in caught if w.category is RuntimeWarning],
    "platforms": sorted({d.platform for d in jax.devices()}),
}))
"""


def _run_source(source, **env):
    child_env = {k: v for k, v in os.environ.items() if k not in _STRIPPED}
    child_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, env=child_env
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


def _run(call_arg, query_devices=False, repeat=1, **env):
    return _run_source(
        _PROBE.format(
            call_arg=call_arg,
            repeat=repeat,
            device_line=_QUERY_DEVICES if query_devices else "",
        ),
        **env,
    )


def test_platform_selection_works_after_jax_is_imported():
    """The post-import contract: `jax.config` applies the platform choice."""
    out = _run('"cpu"', query_devices=True)
    assert out["platforms"] == ["cpu"]
    # Written to the environment as well, so spawned workers inherit the choice.
    assert out["jax_platforms"] == "cpu"


def test_xla_flags_are_set():
    out = _run("", query_devices=False)
    assert "--xla_gpu_deterministic_ops=true" in out["xla_flags"]
    assert "--xla_gpu_enable_triton_gemm=false" in out["xla_flags"]
    assert "--xla_gpu_autotune_level=1" in out["xla_flags"]


def test_no_argument_leaves_platform_auto_detected():
    out = _run("", query_devices=False)
    assert out["jax_platforms"] is None


def test_environment_platform_wins_over_the_argument():
    out = _run('"tpu"', query_devices=True, JAX_PLATFORMS="cpu")
    assert out["jax_platforms"] == "cpu"
    assert out["platforms"] == ["cpu"]


def test_triton_gemm_flag_is_opt_out():
    out = _run("", query_devices=False, FABRICPC_DISABLE_TRITON_GEMM="0")
    assert "--xla_gpu_enable_triton_gemm=false" not in out["xla_flags"]
    assert "--xla_gpu_deterministic_ops=true" in out["xla_flags"]


def test_user_preset_xla_flag_values_win():
    """Flags are guarded by name: a value already in XLA_FLAGS is kept as-is."""
    out = _run(
        "",
        XLA_FLAGS="--xla_gpu_autotune_level=3 --xla_gpu_deterministic_ops=false",
    )
    assert "--xla_gpu_autotune_level=3" in out["xla_flags"]
    assert "--xla_gpu_autotune_level=1" not in out["xla_flags"]
    assert "--xla_gpu_deterministic_ops=false" in out["xla_flags"]
    assert "--xla_gpu_deterministic_ops=true" not in out["xla_flags"]
    # Flags the user did not preset are still appended.
    assert "--xla_gpu_enable_triton_gemm=false" in out["xla_flags"]


def test_repeated_calls_leave_xla_flags_unchanged():
    once = _run("")
    twice = _run("", repeat=2)
    assert twice["xla_flags"] == once["xla_flags"]


def test_flag_guard_matches_whole_names_not_prefixes():
    """A longer flag sharing a prefix must not suppress the flag itself.

    The child never queries devices, so XLA never parses `XLA_FLAGS` and the
    synthetic sibling name below cannot abort the process.
    """
    out = _run("", XLA_FLAGS="--xla_gpu_deterministic_ops_extra=1")
    assert "--xla_gpu_deterministic_ops=true" in out["xla_flags"]
    assert "--xla_gpu_deterministic_ops_extra=1" in out["xla_flags"]


def test_call_after_backend_initialization_warns_and_changes_nothing():
    out = _run_source(_LATE_PROBE)
    assert len(out["warnings"]) == 1, out["warnings"]
    assert "after the JAX backend was initialized" in out["warnings"][0]
    assert out["after"] == out["before"]


def test_call_before_backend_initialization_is_silent():
    out = _run_source(_EARLY_PROBE)
    assert out["warnings"] == []
    assert out["platforms"] == ["cpu"]
