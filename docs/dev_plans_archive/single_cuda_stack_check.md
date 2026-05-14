# Design note: single-CUDA-stack enforcement

## Problem

FabricPC supports two JAX/CUDA wheel stacks: `[cuda12]` (pulls
`jax-cuda12-plugin` + `nvidia-*-cu12` wheels) and `[cuda13]` (pulls
`jax-cuda13-plugin` + `nvidia-*-cu13` wheels). Pip cannot express
mutual exclusivity in extras, and PEP 508 environment markers have no
CUDA-version field, so it is possible to land both stacks in a single
venv. Two ways this happens in practice:

1. `pip install -e ".[all,cuda12]"` on a CUDA-13 host. The build hook's
   auto-detection injects `[cuda13]` into `[all]`; the explicit
   `[cuda12]` is layered on top. Result: both plugins installed.
2. Two sequential pip invocations adding opposite stacks.

When both plugins are present, JAX loads whichever PJRT plugin
registers first via entry-point iteration order. That order is not
stable across Python versions or installs, so the resulting device
choice is undefined. Typical symptoms: cuDNN version-mismatch crashes,
silent CPU fallback, or the wrong GPU device selected.

## What we do

Two layers of defense, neither of which pip itself can enforce:

1. **Pre-install guard in `scripts/install.py`.** Before running pip,
   the helper calls `importlib.metadata.version()` on both
   `jax-cuda12-plugin` and `jax-cuda13-plugin`. If a plugin opposite
   to the one about to be installed is already present, the helper
   aborts with a clear error. No bypass — recreate the venv.

2. **Runtime check in `fabricpc/__init__.py`.** On `import fabricpc`,
   `_check_single_cuda_stack()` performs the same
   `importlib.metadata.version()` probe and raises `ImportError` if
   both plugins are found. This catches the conflict regardless of
   how it was created (the helper, manual pip, or a wheel produced on
   a different host).

## `FABRICPC_ALLOW_MULTIPLE_CUDA` — what it is and is not

The runtime check has an explicit, deliberately ugly opt-out env var.
Setting `FABRICPC_ALLOW_MULTIPLE_CUDA` to any non-empty value makes
`_check_single_cuda_stack()` return immediately, so `import fabricpc`
succeeds even with both plugins present.

Read at import time. Setting it via `os.environ[...] = "1"` after
`import fabricpc` is too late — the check has already run.

### When to set it

- **Debugging the check itself.** You are working on the dual-stack
  logic and the check is blocking your own tests.
- **Inspecting a broken venv.** You ended up with both stacks and want
  to `import fabricpc` long enough to enumerate what's installed
  before deciding what to do. Recommended fix is still "recreate the
  venv" (see below).
- **You have manually pinned `JAX_PLATFORMS`** and are confident JAX
  will load only the plugin you want. Rare; not recommended.

### What it does *not* do

- It does **not** make JAX work correctly with both plugins installed.
  Plugin registration order is still undefined.
- It does **not** affect `scripts/install.py`'s pre-install guard.
  That guard has no bypass — it just refuses.
- It does **not** prevent `nvidia/*` namespace collisions. If you
  later `pip uninstall` one stack's wheels, the other stack's files
  in `nvidia/cudnn/lib/` etc. can still get clobbered.

## Recovery: what to do when both stacks are installed

Do **not** try to "convert" the venv by `pip uninstall`ing the unwanted
stack. The NVIDIA PyPI wheels share the `nvidia/*` namespace package;
uninstalling `nvidia-cudnn-cu12` wipes files in `nvidia/cudnn/lib/`
that `nvidia-cudnn-cu13` still claims. This was reproduced during
plan validation — recovery required `pip install --force-reinstall
--no-deps` of every cu13/unsuffixed nvidia wheel.

The supported recovery is to **recreate the venv**:

```bash
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
python scripts/install.py            # auto-picks one stack
```

## Why an escape hatch exists at all

The check is paternalistic: it stops the user before they hit a worse
failure mode (silent miscomputation, wrong-cuDNN crashes, JAX picking
the wrong device). For nearly all users the right move when it fires
is "recreate the venv". But an unconditional refusal would drive
determined users to fork the package; a documented, deliberately
awkward env-var name (`FABRICPC_ALLOW_MULTIPLE_CUDA`) makes the
escape hatch findable without being casual.
