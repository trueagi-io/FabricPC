"""
There will be many types of node classes for performing various computations at a node.
There must be a standard interface for all node classes.
The gradients will have to be computed by jax, 1) with respect to the weights, 2) with respect to the inputs, and 3) with respect to the biases.
Inputs arrive via edges to specific input slots of the node.
Slots may accept a single input (single-input nodes) or multiple inputs (multi-input nodes).
Nodes always have a single output.
There must be a function for each of the gradient computations that can be overridden by extending the class. - i.e. some node classes may be non-differentiable and the user creates implements the gradient explicitly rather than relying on autograd.
Here we will have a linear node class.
It has a single input slot named "in" of the multi-input type.

Managed by the node:
- Weights and biases
- the transformation function
- the gradient functions
- the input slots

The parameters (weight & biases) in the node should be pytrees to for jax and optimizer to easily manage complex transfer functions (e.g. the node comprises a transformer block)

The edges guide the flow of information through the graph in inference and training.
- Node outputs are passed to slots of post-synaptic nodes (out-neighbors)
- Node gets gradient contributions from post-synaptic nodes by querying the out_neighbors on each outgoing edge for the gradient contributions of that particula edge.
"""