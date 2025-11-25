"""
Training loop for JAX predictive coding networks with local Hebbian learning.

This module implements training with local gradient computation for each node,
as required for true predictive coding with local learning rules.
"""

from typing import Dict, Tuple, Any, cast, List
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import run_inference, gather_inputs
from fabricpc.nodes import get_node_class_from_type
from fabricpc.core.types import NodeParams


def compute_local_weight_gradients(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """
    Compute local weight gradients for each node using its own error signal.

    This implements the local Hebbian learning rule for predictive coding:
    - Weight gradient: -(input.T @ gain_mod_error)
    - Bias gradient: -sum(gain_mod_error)

    Args:
        params: Current model parameters
        final_state: Converged state after inference
        structure: Graph structure

    Returns:
        GraphParams containing gradients for the parameters
    """
    gradients = {}

    for node_name, node_info in structure.nodes.items():
        # Source nodes have no weights, but need empty gradient dict for consistency
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        in_edges_data = gather_inputs(node_info, structure, final_state)

        # Get node class
        node_class = get_node_class_from_type(node_info.node_type)

        # Compute local gradients using node's method
        grad_params = node_class.compute_params_gradient(
            params.nodes[node_name],
            in_edges_data,
            final_state.nodes[node_name],
            node_info
        )

        # Store gradients
        gradients[node_name] = grad_params

    # convert to GraphParams
    params_gradients = GraphParams(nodes=gradients)

    return params_gradients


def train_step(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    infer_steps: int,
    eta_infer: float = 0.1,
) -> Tuple[GraphParams, optax.OptState, float, GraphState]:
    """
    Single training step with local weight updates.

    This implements the full predictive coding training loop:
    1. Run inference to convergence
    2. Compute local gradients for each node
    3. Update weights using optimizer

    Args:
        params: Current model parameters
        opt_state: Optimizer state
        batch: Batch of data with task-specific keys
        structure: Graph structure
        optimizer: Optax optimizer
        rng_key: JAX random key for state initialization
        infer_steps: Number of inference steps
        eta_infer: Inference learning rate

    Returns:
        Tuple of (updated_params, updated_opt_state, loss, final_state)
    """
    from fabricpc.graph.graph_net import initialize_state

    batch_size = next(iter(batch.values())).shape[0]

    # Map task names to node names
    clamps = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map:
            node_name = structure.task_map[task_name]
            clamps[node_name] = task_value

    # Initialize state
    init_state = initialize_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )

    # Run inference to convergence
    final_state = run_inference(
        params, init_state, clamps, structure, infer_steps, eta_infer
    )

    # Compute energy (sum of squared errors)
    energy = sum([final_state.nodes[node_name].energy for node_name in structure.nodes if
                  structure.nodes[node_name].in_degree > 0])

    loss = energy

    # Compute LOCAL gradients for each node
    grads = compute_local_weight_gradients(params, final_state, structure)

    # Update parameters using optimizer
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))
    # Note: optax.apply_updates preserves the structure but loses type info

    return params, opt_state, loss, final_state

def train_pcn(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
    epoch_callback = None,
    iter_callback = None,
) -> Tuple[GraphParams, List[Any], List[Any]]:
    """
    Main training loop for predictive coding network with local learning.

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader (iterable yielding batches)
        config: Training configuration with keys:
            - optimizer: optimizer config dict
            - num_epochs: Number of training epochs
            - infer_steps: number of inference steps
            - eta_infer: inference learning rate
        rng_key: JAX random key (will be split for each batch)
        verbose: Whether to print progress
        epoch_callback: Optional function called at end of each epoch:
            (epoch_idx, params, structure, config, rng_key) -> any
        iter_callback: Optional function called at end of each batch:
            (epoch_idx, batch_idx, loss) -> any

    Returns:
        Trained parameters
        2D List of energy values per iteration (epochs, batches)
        List of evaluation metrics per epoch

    Example:
        >>> rng_key = jax.random.PRNGKey(0)
        >>> params_key, train_key = jax.random.split(rng_key, 2)
        >>> params, structure = create_pc_graph(model_config, params_key)
        >>> train_config = {
        ...     "optimizer": {"type": "adam", "lr": 1e-3},
        ...     "num_epochs": 10,
        ...     "infer_steps": 20,
        ...     "eta_infer": 0.1,
        ... }
        >>> trained_params = train_pcn(params, structure, train_loader, train_config, train_key)
    """
    from fabricpc.training.optimizers import create_optimizer

    # Create optimizer
    optimizer = create_optimizer(config.get("optimizer", {"type": "adam", "lr": 1e-3}))
    opt_state = optimizer.init(params)

    # Training hyperparameters
    infer_steps = config.get("infer_steps", 20)
    eta_infer = config.get("eta_infer", 0.1)
    num_epochs = config.get("num_epochs", 10)

    # Create JIT-compiled training step
    jit_train_step = jax.jit(
        lambda p, o, b, k: train_step(
            p, o, b, structure, optimizer, k, infer_steps, eta_infer
        )
    )

    # Training loop
    iter_results = []
    epoch_results = []
    for epoch_idx in range(num_epochs):
        # Estimate number of batches (if possible)
        try:
            num_batches = len(train_loader)
        except TypeError:
            # train_loader doesn't support len()
            raise TypeError

        # Split rng_key for this epoch's batches
        epoch_rng_key, rng_key = jax.random.split(rng_key)
        batch_keys = jax.random.split(epoch_rng_key, num_batches)

        batch_energies = []
        for batch_idx, batch_data in enumerate(train_loader):
            # Convert batch to JAX format
            if isinstance(batch_data, (list, tuple)):
                # Assume (x, y) format
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            elif isinstance(batch_data, dict):
                # Already a dictionary
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            else:
                raise ValueError(f"unsupported batch format: {type(batch_data)}")

            # Training step with unique rng_key for this batch
            params, opt_state, loss, _ = jit_train_step(params, opt_state, batch, batch_keys[batch_idx])

            if iter_callback is not None:
                batch_energies.append(iter_callback(epoch_idx, batch_idx, loss))
            else:
                batch_energies.append(float(loss) / next(iter(batch.values())).shape[0])  # nornalize by batch size

        iter_results.append(batch_energies)

        # Compute average loss for epoch
        avg_loss = sum(batch_energies) / len(batch_energies) if batch_energies else 0.0

        # Epoch callback
        epoch_results.append(epoch_callback(epoch_idx, params, structure, config, rng_key) if epoch_callback else None)

        if verbose:
            print(f"Epoch {epoch_idx + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return params, iter_results, epoch_results


def evaluate_pcn(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, float]:
    """
    Evaluate predictive coding network on test data.

    Args:
        params: Trained parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation configuration with keys:
            - infer_steps: number of inference steps
            - eta_infer: inference learning rate
        rng_key: JAX random key (will be split for each batch)

    Returns:
        Dictionary of evaluation metrics (e.g., accuracy, loss)
    """
    from fabricpc.graph.graph_net import initialize_state

    infer_steps = config.get("infer_steps", 20)
    eta_infer = config.get("eta_infer", 0.1)

    # Estimate number of batches
    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000

    # Split rng_key for all batches
    batch_keys = jax.random.split(rng_key, num_batches)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch_data in enumerate(test_loader):
        # Convert batch
        if isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        elif isinstance(batch_data, dict):
            batch = {k: jnp.array(v) for k, v in batch_data.items()}
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")

        batch_size = next(iter(batch.values())).shape[0]

        # Map batch to clamps (only clamp input during eval)
        clamps = {}
        if "x" in structure.task_map:
            x_node = structure.task_map["x"]
            clamps[x_node] = batch["x"]

        # Initialize and run inference with unique rng_key
        state = initialize_state(
            structure, batch_size, batch_keys[batch_idx], clamps=clamps, params=params
        )
        final_state = run_inference(
            params, state, clamps, structure, infer_steps, eta_infer
        )

        # Get the label for energy evaluation
        labels = {}
        if "y" in structure.task_map:
            y_node = structure.task_map["y"]
            labels[y_node] = batch["y"]

        # Compute energy
        energy = sum([final_state.nodes[node_name].energy for node_name in structure.nodes if
                      structure.nodes[node_name].in_degree > 0])

        # Add the energy from the label node if applicable
        for node_name, label in labels.items():
            node_state = final_state.nodes[node_name]
            error = node_state.z_latent - label
            energy += jnp.sum(error ** 2)  # TODO: use node's energy functional

        loss = energy / batch_size

        # Compute loss
        total_loss += float(loss) * batch_size

        # Compute accuracy (if applicable)
        if "y" in structure.task_map:
            y_node = structure.task_map["y"]
            predictions = final_state.nodes[y_node].z_latent
            targets = batch["y"]

            # Assume classification: argmax predictions
            pred_labels = jnp.argmax(predictions, axis=1)
            true_labels = jnp.argmax(targets, axis=1)
            correct = jnp.sum(pred_labels == true_labels)

            total_correct += int(correct)
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy}