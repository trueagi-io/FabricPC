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

setup_jax({call_arg})

payload = {{
    "jax_platforms": os.environ.get("JAX_PLATFORMS"),
    "xla_flags": os.environ.get("XLA_FLAGS", ""),
}}
{device_line}
print(json.dumps(payload))
"""

_QUERY_DEVICES = 'payload["platforms"] = sorted({d.platform for d in jax.devices()})'


def _run(call_arg, query_devices=False, **env):
    source = _PROBE.format(
        call_arg=call_arg, device_line=_QUERY_DEVICES if query_devices else ""
    )
    child_env = {k: v for k, v in os.environ.items() if k not in _STRIPPED}
    child_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, env=child_env
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.splitlines()[-1])


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
