# Predictive Coding Examples

This directory contains example scripts demonstrating FabricPC.

## Quick Start

```bash
# Install FabricPC in editable mode (from project root)
pip install -e ".[dev,tfds,viz]"

# Run MNIST demo
python examples/mnist_demo.py

# Run Navier-Stokes energy smoke test on MNIST data
python examples/mnist_navier_stokes_smoke.py

# Run the synthetic fluid acceptance benchmark
python examples/navier_stokes_acceptance.py

# Choose optimizer preset at runtime (no code edits)
python examples/mnist_demo.py --optimizer adam
# or via environment variable
FABRICPC_OPTIMIZER=ngd_layerwise python examples/mnist_demo.py
# if your GPU hits Triton GEMM XLA runtime errors, disable it explicitly
FABRICPC_DISABLE_TRITON_GEMM=1 python examples/mnist_demo.py --optimizer adam
```

## Examples

### `mnist_demo.py`

**Description**: MNIST classification example demonstrating the basic PC workflow.

**Architecture**:
- Input layer: 784 units (28x28 flattened images)
- Hidden layer 1: 256 units (sigmoid activation)
- Hidden layer 2: 64 units (sigmoid activation)
- Output layer: 10 units (class logits)

**Configuration**:
- Optimizer: Natural Gradient (diag Fisher, lr=1e-3)
- Inference: 20 steps @ eta=0.05
- Training: 20 epochs, batch size 200

**Expected Results**:
```
Epoch 1/20, energy: 0.3991
Epoch 2/20, energy: 0.0994
Epoch 3/20, energy: 0.0566
Epoch 4/20, energy: 0.0374
...
Epoch 18/20, energy: 0.0026
Epoch 19/20, energy: 0.0023
Epoch 20/20, energy: 0.0020
Avg Training time: 1.30 seconds per epoch

Evaluating...
Test Accuracy: 98.14%
```

### `mnist_navier_stokes_smoke.py`

**Description**: Smoke test the `NavierStokesEnergy` runtime on real MNIST images.

**Behavior**:
- Loads MNIST in NHWC format
- Adapts each grayscale image `(28, 28, 1)` into a pseudo-fluid field `(28, 28, 3)`
- Runs a short predictive-coding training pass and held-out inference check
- Reports finite energy values instead of classification accuracy

### `navier_stokes_acceptance.py`

**Description**: Acceptance benchmark for `NavierStokesEnergy` on a synthetic periodic fluid reconstruction task.

**Behavior**:
- Generates periodic `(u, v, p)` fields from a synthetic Taylor-Green-style family
- Compares `GaussianEnergy` and `NavierStokesEnergy` across full-field, partial-observation, and noisy-observation scenarios
- Reports train/validation/test fluid metrics instead of classification accuracy
- Saves qualitative field and physics panels plus a JSON summary to a temporary artifact directory

### `PC_backprop_compare.py`
**Description**: Compare accuracy and training time of Predictive Coding vs Backpropagation on MNIST.
PC:       97.99 +/- 0.03%  (mean +/- SE, SD=0.11%)
Backprop: 97.81 +/- 0.05%  (mean +/- SE, SD=0.23%)

--- Paired t-test ---
Mean difference (PC - BP): +0.18%
t-statistic: 3.6474
p-value: 0.0017, N = 20
Significant at p<0.05: YES
--- Training Time per Epoch ---
PC:       1.211 +/- 0.011s
Backprop: 0.761 +/- 0.008s
Ratio:    PC is 1.59x backprop time


### transformer_demo.py

Using Backpropagation training method (6-block Transformer on Shakespeare dataset)

  [DEBUG] per-token CE loss: min=-0.0000, max=11.7602, mean=1.4054

  [DEBUG] per-token intrinsic perplexity: min=1.0000, max=25.2567, mean=3.3277

  [DEBUG] prob of correct token: min=0.000008, max=1.000000, mean=0.593641

  [DEBUG] batch loss: 1.4054

Test - Loss: 1.9770, Perplexity: 7.22, Acc: 0.5202

Epoch 1/1, Loss: 1.2071, Perplexity: 3.34

Training completed in 347.4s (347.4s per epoch)

Generating sample text...

- Prompt: 'Know, Rome, that'
----------------------------------------
Know, Rome, that is not the devils;

----------------------------------------
- Prompt: 'MENENIUS:'
----------------------------------------
MENENIUS:
The good my lord.
----------------------------------------

Using Predictive Coding training method

  [DEBUG] per-token CE loss: min=-0.0000, max=23.0259, mean=22.4215

  [DEBUG] per-token intrinsic perplexity: min=1.0000, max=1.0000, mean=1.0000

  [DEBUG] prob of correct token: min=0.000000, max=1.000000, mean=0.026245

  [DEBUG] batch loss: 22.4215

Test - Loss: 22.0098, Perplexity: 3620162304.00, Acc: 0.0441

Train Epoch 1/1, Energy: 238896016419357371257585664.0000, Loss: 21.8046, Perplexity: 2948480768.00

Training completed in 815.3s (815.3s per epoch)

PC inference phase is very sensitive to weight initialization and hyperparameters. Mode collapse in PC training but trains successfully with BP.



## Notes

### Performance

First batch will be slow due to JIT compilation (~5-10 seconds). Subsequent batches are fast.

## Troubleshooting
**Import Error**: Make sure FabricPC is installed:
See quickstart section above.
