# FabricPC

**A flexible, performant predictive coding library**

FabricPC implements predictive coding networks using a clean abstraction of:
- **Nodes**: State variables (latents), projection functions, and activations
- **Wires**: Connections (edges) between nodes in the model architecture
- **Updates**: Iterative inference and local learning algorithms

Uses JAX for GPU acceleration and local (node-level) automatic differentiation.

## About Predictive Coding
Predictive coding (PC) is a biologically-inspired framework for perception and learning in the brain. It posits that the brain continuously generates predictions about sensory inputs and updates its internal representations based on local prediction errors. 
PC performs bilevel optimization: an inner loop infers latent activations by minimizing prediction errors, while an outer loop updates weights via local Hebbian-like rules. Under certain conditions, this process is equivalent to backpropagation. While currently slower than backprop on standard hardware, PC offers:
- Potential for faster inference on neuromorphic hardware
- Natural handling of recurrent and arbitrary graph architectures
- Associative memory capabilities
- Potential novel plasticity rules for continual learning

There are various flavors of PC. FabricPC provides a graph-based implementation that focuses on principles:
- Local (Hebbian) learning rules
- Parallel processing of nodes
- Modularity of components
- Arbitrary architectures
- Scalability with JAX
- Extensibility for research
 
## Quick Start

FabricPC supports Python 3.10–3.13. (Python 3.14 is not yet
supported — TensorFlow has no 3.14 wheels, so `[tfds]` would be
unresolvable.)

> **Older Intel-based Macs are not currently supported.** FabricPC
> targets Linux and Apple Silicon (arm64) macOS. Intel (x86_64)
> Macs are not a supported platform at this time.

`[all]` auto-detects CUDA at install time. The build hook in
`setup.py` reads `nvidia-smi` and injects the matching
`[cuda12]` / `[cuda13]` into the resolved `[all]` extra; if no
NVIDIA driver is detected, `[all]` stays CPU-only. So the same
copy-paste works whether you have a CUDA-12 GPU, a CUDA-13 GPU, or
no GPU at all:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
pre-commit install
aim up                # optional, only available on Python <=3.12
python examples/mnist_demo.py
```

The detection is best-effort and runs every time `setup.py` is
re-executed (i.e. on every `pip install` / `pip install -e .`).
Editable installs re-detect on each run; built wheels freeze the
detection result, so prefer editable installs for development.

### Explicit alternatives

If you want to override the detection or skip optional pieces:

- **Force a CUDA major**: combine `[all]` (or a smaller subset) with
  `[cuda12]` or `[cuda13]` explicitly. For example:

  ```bash
  pip install -e ".[dev,experiments,viz,cuda13]"
  ```

- **CPU-only on a GPU host**: `pip install -e ".[dev,experiments,viz]"`
  (omit `[all]` so the auto-CUDA hook is bypassed).

- **Helper script**: `python scripts/install.py` does the same
  `nvidia-smi` detection as the build hook, plus a guard against
  installing a second CUDA stack into a venv that already has one.
  Use `--cuda 12` / `--cuda 13` / `--cuda none` to override
  detection; `--all` adds `[tfds]` to the helper's hardcoded extras.

### Notes

- On Python 3.13 the `aim` experiment-tracking server is skipped
  (no upstream Python 3.13 wheel). The `[viz]` extra self-degrades
  via an environment marker; use plotly/pandas/kaleido instead.
- For a CUDA 13 system (e.g. Fedora 43 with `cuda-toolkit-13-2` and
  a 5xx-series NVIDIA driver), the `[cuda13]` install pulls JAX 0.10
  with `jax-cuda13-plugin` and `nvidia-cudnn-cu13` — roughly 2 GB of
  wheels on first install. To use system CUDA libraries instead of
  the PyPI wheels, swap `cuda13` for the upstream
  `jax[cuda13-local]` extra and install matching cuDNN (on
  Fedora 43, `dnf install cuda-cudnn-13-2`).
- The `[cuda]` and `[cuda12]` extras still target CUDA 12 GPUs —
  they have not changed.

> **Only one CUDA stack per venv.** `scripts/install.py` refuses to
> add cuda12 to a venv that already has cuda13 (or vice versa).
> `import fabricpc` also raises `ImportError` if it detects both
> plugins installed (set `FABRICPC_ALLOW_MULTIPLE_CUDA=1` to bypass
> the runtime check, for debugging).

### Environment variables that affect install/runtime

- `FABRICPC_SKIP_CUDA_DETECT=1` — set during a build to skip the
  `setup.py` CUDA detection so the produced wheel's `[all]` extra is
  CPU-only. Use this when packaging wheels for redistribution to
  hosts whose CUDA driver may differ from the build host. Example:
  `FABRICPC_SKIP_CUDA_DETECT=1 python -m build`.
- `FABRICPC_ALLOW_MULTIPLE_CUDA=1` — bypass the runtime
  dual-CUDA-plugin check that runs on `import fabricpc`. Only set
  this for debugging; JAX behavior with both plugins installed is
  undefined.

## Features
- Modular node and wire abstractions for flexible model construction
- Inherently supports arbitrary architectures: feedforward, recurrent, skip connections, etc.
- Support for various node types: Linear, Conv1D/2D/3D (planned), Transfomers (in progress)
- Local automatic differentiation for efficient inference and learning
- JAX backend for GPU acceleration and scalability

## Contributions
Contributions are welcome! Please open issues or pull requests on the GitHub repository.
- Develop on a branch using convention "feature/your_feature_name"
- All demos must match baseline results or explain divergence and test suites must pass on rebased PR.
- Write unit tests and docstrings for new code
- Use the pre-commit hooks for PEP8 style and code quality (run `pre-commit install` once after cloning!)
- Follow the rebase instructions in `docs/rebasing_feature_branch.md` before opening a PR.

This is a research-first project.
- APIs may change frequently until v1.0 release.
- Any breaking changes are documented in the changelog.

## License
This project is licensed under the [MIT License](LICENSE).


## Building a model
A model consists of structure and parameters.

```python
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD

layer1 = Linear(shape=(784,), name="input")
layer2 = Linear(shape=(256,), name="hidden")
layer3 = Linear(shape=(10,), name="output")

structure = graph(
    nodes=[layer1, layer2, layer3],
    edges=[Edge(layer1, layer2.slot("in")),
           Edge(layer2, layer3.slot("in"))
           ],
    task_map=TaskMap(x=layer1, y=layer3),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)
params = initialize_params(structure, rng_key)
```

## Extending FabricPC

### Custom Nodes

Create custom node types by subclassing `NodeBase`. Implement the `forward()` and `initialize_params()` methods. Nodes have a single output. Define slots for incoming connections. Slots are named arguments of the node's transfer function and are referenced in edges when building the graph.

See `examples/custom_node.py` for a complete Conv2D implementation.

### Inference Algorithms
Create custom inference algorithms by subclassing `InferenceBase` and implementing the `compute_new_latent()` method.

### Learning Algorithms
Weight learning loop algorithm abstraction is planned for a future release. Optimizer chains are fully supported with Optax and can be used directly in the training loop.

### Custom Initializers
Create custom initializers by subclassing `StateInitializerBase` for latent state initialization and implementing the `initialize()` method.

Graph-aware weight initializers are in progress and will be added in a future release.

Node level initializers extend `InitializerBase` and implement `initialize_weights()` method. These are agnostic to acting on node state or parameters.

## Shape Conventions

 All shapes use batch-first, channels-last format (NHWC, NLC, NDHWC) and the batch size is not included in node shape definitions.

 - Consistent with JAX's default conv behavior
 - Linear: shape=(features,) - e.g., (128,) for 128-dimensional vector
 - 1D Conv: shape=(seq_len, channels) - e.g., (100, 32) for 100 timesteps, 32 channels
 - 2D Conv: shape=(H, W, C) - e.g., (28, 28, 64) for 28x28 image, 64 channels (NHWC)
 - 3D Conv: shape=(D, H, W, C) - e.g., (32, 32, 32, 16) for 3D volume

Conv Node Shape Flow (Future Reference)

 - Input:  (batch, H_in, W_in, C_in)   e.g., (32, 28, 28, 1)
 - Kernel: (kH, kW, C_in, C_out)       e.g., (3, 3, 1, 64)
 - Output: (batch, H_out, W_out, C_out) e.g., (32, 26, 26, 64)