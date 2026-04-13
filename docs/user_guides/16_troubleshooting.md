# Troubleshooting

## Installation Issues

**JAX/CUDA version conflict**

If you see CUDA-related errors, ensure your CUDA 12 drivers match the JAX build:
```bash
pip install --upgrade jax[cuda12]
```

**Triton GEMM XLA errors**

If XLA compilation fails with Triton-related errors:
```bash
export FABRICPC_DISABLE_TRITON_GEMM=1
```

**Python version**

FabricPC requires Python 3.10–3.12. Python 3.13+ may not work with the Aim experiment tracking library.

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
from fabricpc.utils.helpers import set_jax_flags_before_importing_jax
set_jax_flags_before_importing_jax(jax_platforms="cuda")
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

**Duplicate node names**

Each node must have a unique name. Use `GraphNamespace` for scoped naming:
```python
from fabricpc.builder.namespace import GraphNamespace
with GraphNamespace("encoder"):
    h1 = Linear(shape=(256,), name="h1")  # becomes "encoder/h1"
```

**Single-input slot receiving multiple edges**

If a slot has `is_multi_input=False` (e.g., StorkeyHopfield's `"probe"`), it accepts only one edge. Use `IdentityNode` as a merge point if you need to combine multiple sources.

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

PC requires iterative inference (typically 10–50 steps per batch) before weight updates. Each inference step is a forward pass through the network. This overhead is inherent to the algorithm. Potential speedups include neuromorphic hardware that runs inference in parallel, or incremental PC methods that amortize inference across batches.

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
