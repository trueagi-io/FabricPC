# FabricPC

**A flexible, performant predictive coding framework**

FabricPC implements predictive coding networks using a clean abstraction of:
- **Nodes**: State variables (latents), projection functions, and activations
- **Wires**: Connections (edges) between nodes in the model architecture
- **Updates**: Iterative inference and learning algorithms

## Quick Start
export PYTHONPATH=$PYTHONPATH:$(pwd)
python experiments/mnist/demo_minimal.py