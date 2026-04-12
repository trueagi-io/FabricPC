"""
Training loop for JAX predictive coding networks with local Hebbian learning.

This module implements training with local gradient computation for each node,
as required for true predictive coding with local learning rules.
"""

from typing import Dict, Tuple, Any, cast, List
import math
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import run_inference
from fabricpc.graph.graph_net import compute_local_weight_gradients


def get_graph_param_gradient(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
) -> Tuple[GraphParams, float, GraphState]:
    """
    Compute parameter gradients for a batch without updating parameters.

    Use this shared code for steps in single-gpu and multi-gpu training.

    Args:
        params: Current model parameters
        batch: Batch of data with task-specific keys
        structure: Graph structure
        rng_key: JAX random key for state initialization

    Returns:
        Tuple of (grads, energy, final_state)
    """
    from fabricpc.graph.state_initializer import initialize_graph_state

    batch_size = next(iter(batch.values())).shape[0]

    # Map task names to node names
    clamps = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map:
            node_name = structure.task_map[task_name]
            clamps[node_name] = task_value

    # Initialize state using graph config
    init_state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
    )

    # Run inference to convergence
    final_state = run_inference(params, init_state, clamps, structure)

    # Compute energy (ignore terminal input nodes)
    energy = sum(
        [
            sum(final_state.nodes[node_name].energy)
            for node_name in structure.nodes
            if structure.nodes[node_name].node_info.in_degree > 0
        ]
    )

    # Compute LOCAL gradients for each node
    grads = compute_local_weight_gradients(params, final_state, structure)

    return grads, energy, final_state


def train_step(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
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

    Returns:
        Tuple of (updated_params, updated_opt_state, energy, final_state)
    """

    # Compute gradients
    grads, energy, final_state = get_graph_param_gradient(
        params, batch, structure, rng_key
    )

    # Update parameters using optimizer
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))
    # Note: optax.apply_updates preserves the structure but loses type info

    return params, opt_state, energy, final_state


def train_pcn(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
    epoch_callback=None,
    iter_callback=None,
) -> Tuple[GraphParams, List[Any], List[Any]]:
    """
    Main training loop for predictive coding network with local learning.

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader (iterable yielding batches)
        optimizer: Optax optimizer (e.g., optax.adam(1e-3))
        config: Training configuration with keys:
            - num_epochs: Number of training epochs
        rng_key: JAX random key (will be split for each batch)
        verbose: Whether to print progress
        epoch_callback: Optional function called at end of each epoch:
            (epoch_idx, params, structure, config, rng_key) -> any
        iter_callback: Optional function called at end of each batch:
            (epoch_idx, batch_idx, energy) -> any

    Returns:
        Trained parameters
        2D List of energy values per iteration (epochs, batches)
        List of evaluation metrics per epoch

    Example:
        >>> rng_key = jax.random.PRNGKey(0)
        >>> params_key, train_key = jax.random.split(rng_key, 2)
        >>> params = initialize_params(structure, params_key)
        >>> optimizer = optax.adam(1e-3)
        >>> train_config = {
        ...     "num_epochs": 10,
        ... }
        >>> trained_params = train_pcn(params, structure, train_loader, optimizer, train_config, train_key)
    """
    opt_state = optimizer.init(params)

    # Training hyperparameters
    num_epochs = config.get("num_epochs", 10)  # supports float (e.g. 1.5)

    # Support fractional epochs: e.g. 1.5 -> 2 loop iterations, last stops at 50%
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)

    # Create JIT-compiled training step
    jit_train_step = jax.jit(
        lambda p, o, b, k: train_step(p, o, b, structure, optimizer, k)
    )

    # Training loop
    iter_results = []
    epoch_results = []
    for epoch_idx in range(total_epochs):
        # Estimate number of batches (if possible)
        try:
            num_batches = len(train_loader)
        except TypeError:
            # train_loader doesn't support len()
            raise TypeError

        # On the final epoch, truncate if fractional
        is_last_epoch = epoch_idx == total_epochs - 1
        if is_last_epoch and frac > 0:
            max_batches = round(frac * num_batches)
        else:
            max_batches = num_batches

        # Split rng_key for this epoch's batches
        epoch_rng_key, rng_key = jax.random.split(rng_key)
        batch_keys = jax.random.split(epoch_rng_key, max_batches)

        batch_energies = []
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

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
            params, opt_state, energy, _ = jit_train_step(
                params, opt_state, batch, batch_keys[batch_idx]
            )

            if iter_callback is not None:
                batch_energies.append(iter_callback(epoch_idx, batch_idx, energy))
            else:
                batch_energies.append(
                    float(energy) / next(iter(batch.values())).shape[0]
                )  # normalize by batch size

        iter_results.append(batch_energies)

        # Compute average energy for epoch
        avg_energy = (
            sum(batch_energies) / len(batch_energies) if batch_energies else 0.0
        )

        # Epoch callback
        epoch_results.append(
            epoch_callback(epoch_idx, params, structure, config, rng_key)
            if epoch_callback
            else None
        )

        if verbose:
            print(f"Epoch {epoch_idx + 1}/{total_epochs}, energy: {avg_energy:.4f}")

    return params, iter_results, epoch_results


def eval_step(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
) -> Tuple[float, int, int]:
    """
    Single evaluation step (JIT-compilable).

    Args:
        params: Model parameters
        batch: Batch data with 'x' and 'y' keys
        structure: Graph structure
        rng_key: Random key for initialization

    Returns:
        Tuple of (avg_energy, correct, batch_size)
    """
    from fabricpc.graph.state_initializer import initialize_graph_state

    batch_size = batch["x"].shape[0]

    # Map batch to clamps (only clamp input during eval)
    clamps = {}
    if "x" in structure.task_map:
        x_node = structure.task_map["x"]
        clamps[x_node] = batch["x"]

    # Initialize graph latent states
    state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
    )

    # Run inference steps
    # TODO - can skip inference in evaluation mode (no labels) if 1. initialization method is feed-forward AND 2. the graph has no cycles.
    final_state = run_inference(params, state, clamps, structure)

    # Compute total network energy
    total_energy = jnp.array(0.0)
    for node_name in structure.nodes:
        total_energy = total_energy + jnp.sum(
            final_state.nodes[node_name].energy
        )  # sum energy over the batch dimension and accumulate to total_energy

    # In evaluation mode, the output node won't be clamped to a label and doesn't contribute to energy. Energy is only from internal nodes. Use a proper metric to assess task performance, not energy.

    avg_energy = total_energy / batch_size

    # Compute accuracy
    correct = 0
    if "y" in structure.task_map:
        y_node = structure.task_map["y"]
        predictions = final_state.nodes[y_node].z_mu
        pred_labels = jnp.argmax(predictions, axis=1)
        true_labels = jnp.argmax(batch["y"], axis=1)
        correct = jnp.sum(pred_labels == true_labels)

    return avg_energy, correct, batch_size


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
        config: Evaluation configuration
        rng_key: JAX random key (will be split for each batch)

    Returns:
        Dictionary of evaluation metrics {"energy": avg_energy, "accuracy": accuracy}
        Note: energy is not a meaningful metric for evaluation, but we include internal node energy for completeness. Focus on accuracy or other task-specific metrics.
        Note: Energy will be zero in evaluation mode for graphs that are feed-forward in topology (no cycles) and use feed-forward initialization.
    """
    # Create JIT-compiled eval step
    jit_eval_step = jax.jit(lambda p, b, k: eval_step(p, b, structure, k))

    # Estimate number of batches
    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000

    # Split rng_key for all batches
    batch_keys = jax.random.split(rng_key, num_batches)

    total_energy = 0.0
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

        # Run JIT-compiled eval step
        batch_energy, correct, batch_size = jit_eval_step(
            params, batch, batch_keys[batch_idx]
        )

        total_energy += float(batch_energy) * int(batch_size)
        total_correct += int(correct)
        total_samples += int(batch_size)

    avg_energy = total_energy / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {"energy": avg_energy, "accuracy": accuracy}
