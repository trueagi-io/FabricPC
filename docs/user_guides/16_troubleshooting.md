# Troubleshooting

## Installation Issues

First run NVIDIA-SMI to view system driver version and cuda version.
```bash
nvidia-smi
```

**JAX/CUDA version conflict**

If you see CUDA-related errors, install the backend matching your driver and re-run with
`-U` so JAX's coupled wheels (`jax`, `jaxlib`, plugin, pjrt) upgrade together. See
[Why `-U`?](01_installation.md#why--u) for the reason.
```bash
pip install -U "jax[cuda12]"   # or "jax[cuda13]" for driver ≥580
```

**GPU install fails on Windows / macOS**

If `pip install -U -e ".[all,cuda12]"` (or `cuda13`) fails with
`No matching distribution found for jax-cuda12-plugin`, you are on a platform without
JAX CUDA wheels — JAX publishes them for Linux x86_64/aarch64 only (see the
[Requirements](01_installation.md#requirements) summary). Install CPU-only, or use WSL2
for GPU on Windows (JAX marks WSL2 GPU support experimental).
```bash
pip install -U -e ".[all]"   # CPU-only; works on Windows and macOS
```

**Triton GEMM XLA errors**

If XLA compilation fails with Triton-related errors:
```bash
export FABRICPC_DISABLE_TRITON_GEMM=1
```

**Python version**

FabricPC supports Python 3.10–3.13. Only the optional Aim experiment tracker (in `[viz]`/`[all]`) is limited to Python ≤3.12; on Python 3.13 it is skipped automatically and the rest installs normally.

---

## Training Issues

**Energy not decreasing**
- Check `eta_infer` — too high causes oscillation, too low causes slow convergence. Try 0.01–0.2.
- Check `infer_steps` — too few steps prevent convergence. Try 20–50.
- Verify activation/energy pairing — use `SoftmaxActivation` + `CrossEntropyEnergy` for classification output nodes.
- For deep networks, enable muPC: `scaling=MuPCConfig()` in `graph()`.

**NaN values during training**
- Reduce the optimizer learning rate (try 1e-4 or lower).
- Check weight initialization scale — use `MuPCInitializer` with `MuPCConfig` for deep networks.
- Try `InferenceSGDNormClip` for gradient stability.
- Ensure inputs are properly normalized.

**Low accuracy despite low energy**
- Verify that `TaskMap(x=..., y=...)` correctly maps to your input and output nodes.
- Ensure the output node uses `SoftmaxActivation` + `CrossEntropyEnergy` for classification.
- Check that labels are one-hot encoded.

---

## Performance Issues

**First batch is very slow (5–10 seconds)**

This is expected — JAX JIT-compiles the training step on first invocation. Subsequent batches run at full speed.

**JAX memory flags**

Set before importing JAX:
```python
from jax_setup import set_jax_flags_before_importing_jax
set_jax_flags_before_importing_jax()
```

Or manually:
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
```

---

## Graph Construction Issues

**Self-edges are not allowed**

A node cannot connect to itself via an Edge. The graph builder will raise an error.

See the `StorkeyHopfield` node for an example of self-feedback lateral connections; implement self-connection inside a node.

**Duplicate node names**

Each node must have a unique name. Use `GraphNamespace` for scoped naming:
```python
from fabricpc.core.topology import GraphNamespace
with GraphNamespace("encoder"):
    h1 = Linear(shape=(256,), name="h1")  # becomes "encoder/h1"
```

**Single-input slot receiving multiple edges**

If a slot has `is_multi_input=False` (e.g., StorkeyHopfield's `"in"`), it accepts only one edge. Use `IdentityNode` as a merge point if you need to combine multiple sources.

---

## muPC Issues

**When to set `include_output=True` vs `False`**

- `False` (default): For classification with softmax + CrossEntropy. Output node uses standard initialization.
- `True`: For regression with identity + Gaussian energy. Output node gets muPC scaling.

**Using MuPCInitializer without MuPCConfig**

`MuPCInitializer` alone just draws weights from `N(0, 1)`. Without `MuPCConfig` on the graph, there is no forward scaling — activations will explode. Always use both together.

---

## FAQ

**Is PC equivalent to backpropagation?**

Under certain conditions (infinite inference steps, specific energy functionals), PC converges to the same gradients as backprop. In practice, PC with finite inference steps and Hebbian learning produces similar but not identical results. FabricPC provides both modes for comparison.

**Why is PC slower than backprop?**

PC requires iterative inference (typically 3 to 5 steps per layer of depth per batch) before weight updates. Each inference step is a forward pass through the network. This overhead is inherent to the algorithm. Potential speedups include neuromorphic hardware that runs inference in parallel, or incremental PC methods that amortize inference across batches.

**Can I use FabricPC for non-classification tasks?**

Yes. Set the output node to `IdentityActivation()` + `GaussianEnergy()` for regression. The `TaskMap(x=..., y=...)` pattern works the same way — `y` is clamped to continuous targets during training.

**How do I save and load trained parameters?**

Use Orbax (included in dependencies):
```python
import orbax.checkpoint as ocp

# Save
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save("/path/to/checkpoint", params)

# Load
params = checkpointer.restore("/path/to/checkpoint")
```
