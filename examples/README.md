# Predictive Coding Examples

This directory contains example scripts demonstrating FabricPC.

## Quick Start

```bash
# Install FabricPC in editable mode (from project root)
pip install -e ".[dev,torch,viz]"

# Run MNIST demo
python examples/mnist_demo.py
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
- Optimizer: Adam (lr=1e-3)
- Inference: 20 steps @ eta=0.05
- Training: 20 epochs, batch size 200

**Expected Results**:
```
Epoch 1/20, Loss: 0.4529
Epoch 2/20, Loss: 0.1903
Epoch 3/20, Loss: 0.0757
Epoch 4/20, Loss: 0.0482
...
Epoch 18/20, Loss: 0.0052
Epoch 19/20, Loss: 0.0047
Epoch 20/20, Loss: 0.0043
Avg Training time: 2.05 seconds per epoch

Evaluating...
Test Accuracy: 97.93%
Test Loss: 0.0349
```

### transformer_demo.py

Using Backpropagation training method
  [DEBUG] per-token CE loss: min=-0.0000, max=11.7602, mean=1.4054
  [DEBUG] per-token intrinsic perplexity: min=1.0000, max=25.2567, mean=3.3277
  [DEBUG] prob of correct token: min=0.000008, max=1.000000, mean=0.593641
  [DEBUG] batch loss: 1.4054
Test - Loss: 1.9770, Perplexity: 7.22, Acc: 0.5202
Epoch 1/1, Loss: 1.2071, Perplexity: 3.34

Training completed in 347.4s (347.4s per epoch)

Generating sample text...

Prompt: 'Know, Rome, that'
----------------------------------------
Know, Rome, that is not the devils;

----------------------------------------
Prompt: 'MENENIUS:'
----------------------------------------
MENENIUS:
The good my lord.
----------------------------------------


Using Predictive Coding training method
  [DEBUG] per-token CE loss: min=0.1940, max=23.0259, mean=18.8622
  [DEBUG] per-token intrinsic perplexity: min=1.5732, max=1.9931, mean=1.7882
  [DEBUG] prob of correct token: min=0.000000, max=0.823626, mean=0.111168
  [DEBUG] batch loss: 18.8622
  Test - Loss: 18.5013, Perplexity: 108398384.00, Acc: 0.1491
Train Epoch 1/1, Energy: 2565.2107, Loss: 17.5974, Perplexity: 43896844.00


Using Predictive Coding training method
Test - Loss: 3.3824, Perplexity: 29.44, Acc: 0.1491
Train Epoch 1/1, Energy: 111.3551, Loss: 0.5658, Perplexity: 1.76

  [DEBUG] per-token CE loss: min=1.5117, max=6.1297, mean=3.6204
  [DEBUG] per-token intrinsic perplexity: min=25.1934, max=25.5652, mean=25.3975
  [DEBUG] prob of correct token: min=0.002177, max=0.220527, mean=0.052559
  [DEBUG] batch loss: 3.6204

Training completed in 1259.9s (1259.9s per epoch)


Using Predictive Coding training method
Test - Loss: 19.5872, Perplexity: 321063840.00, Acc: 0.1491
Train Epoch 1/1, Energy: 111.3551, Loss: 0.5658, Perplexity: 1.76

  [DEBUG] per-token CE loss: min=-0.0000, max=23.0259, mean=20.2949
  [DEBUG] per-token intrinsic perplexity: min=1.0000, max=1.0000, mean=1.0000
  [DEBUG] prob of correct token: min=0.000000, max=1.000000, mean=0.118408
  [DEBUG] batch loss: 20.2949

Training completed in 1340.1s (1340.1s per epoch)



## Notes

### Data Loading Warning

You may see warnings about `os.fork()` when using PyTorch DataLoader:
```
RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code,
and JAX is multithreaded, so this will likely lead to a deadlock.
```

This is harmless for now but will be addressed in future versions by switching to JAX-native data pipelines.

### Performance

First batch will be slow due to JIT compilation (~5-10 seconds). Subsequent batches are fast.

## Troubleshooting
**Import Error**: Make sure FabricPC is installed:
```bash
pip install -e ".[dev,torch,viz]"
```
