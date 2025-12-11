"""
Backpropagation training for FabricPC networks.

This module provides standard end-to-end backpropagation training as an
alternative to iterative predictive coding inference. Key differences:

1. No iterative latent inference - single forward pass
2. Latent states set to projections: z_latent = z_mu
3. End-to-end autodiff via jax.grad
4. Only input nodes are clamped (output computes freely)

Use cases:
- Baseline comparison with predictive coding
- Faster training when PC dynamics not needed
- Standard deep learning workflows
"""

from typing import Dict, Tuple, Any, List, Optional, Callable, cast
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.graph.graph_net import initialize_state
from fabricpc.training.train_autoregressive import create_causal_mask


def compute_loss(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    loss_type: str = "cross_entropy",
) -> jnp.ndarray:
    """
    Compute differentiable loss for backpropagation training.

    This runs a single forward pass through the network (no iterative
    inference) and computes loss at the output node.

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch data with keys matching task_map (e.g., 'x', 'y')
        rng_key: JAX random key for state initialization
        loss_type: Loss function type: "cross_entropy" or "mse"

    Returns:
        Scalar loss value (mean over batch)
    """
    batch_size = batch["x"].shape[0]

    # Clamp ONLY input node (not output!) - key difference from PC
    clamps = {}
    if "x" in structure.task_map:
        clamps[structure.task_map["x"]] = batch["x"]

    # Single forward pass via initialize_state with feedforward init
    state = initialize_state(
        structure, batch_size, rng_key,
        clamps=clamps,
        params=params
    )

    # Get prediction from output node
    output_node = structure.task_map["y"]
    predictions = state.nodes[output_node].z_mu
    targets = batch["y"]

    # Compute loss
    if loss_type == "cross_entropy":
        log_preds = jnp.log(predictions + 1e-10)
        loss = -jnp.sum(targets * log_preds) / batch_size
    elif loss_type == "mse":
        loss = jnp.mean((predictions - targets) ** 2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


def train_step_backprop(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    loss_type: str = "cross_entropy",
) -> Tuple[GraphParams, optax.OptState, float]:
    """
    Single backprop training step.

    Args:
        params: Current model parameters
        opt_state: Optimizer state
        batch: Batch data
        structure: Graph structure
        optimizer: Optax optimizer
        rng_key: JAX random key
        loss_type: Loss function type

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value)
    """
    def loss_fn(p):
        return compute_loss(p, structure, batch, rng_key, loss_type)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, loss


def train_backprop(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
    epoch_callback: Optional[Callable] = None,
    iter_callback: Optional[Callable] = None,
) -> Tuple[GraphParams, List[List[float]], List[Any]]:
    """
    Main backprop training loop.

    This provides standard end-to-end backpropagation training for FabricPC
    models, using the same graph structure but skipping iterative inference.

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader yielding batches
        config: Training configuration:
            - optimizer: Optimizer config dict (type, lr, weight_decay)
            - num_epochs: Number of training epochs
            - loss_type: "cross_entropy" or "mse" (default: "cross_entropy")
        rng_key: JAX random key
        verbose: Print progress
        epoch_callback: Optional (epoch, params, structure, config, rng) -> any
        iter_callback: Optional (epoch, batch_idx, loss) -> any

    Returns:
        Tuple of (trained_params, loss_history, epoch_results)

    Example:
        >>> params, structure = create_pc_graph(config, rng_key)
        >>> train_config = {
        ...     "num_epochs": 10,
        ...     "optimizer": {"type": "adam", "lr": 1e-3},
        ...     "loss_type": "cross_entropy",
        ... }
        >>> trained_params, losses, _ = train_backprop(
        ...     params, structure, train_loader, train_config, rng_key
        ... )
    """
    from fabricpc.training.optimizers import create_optimizer

    # Create optimizer
    optimizer = create_optimizer(config.get("optimizer", {"type": "adam", "lr": 1e-3}))
    opt_state = optimizer.init(params)

    # Training hyperparameters
    num_epochs = config.get("num_epochs", 10)
    loss_type = config.get("loss_type", "cross_entropy")

    # JIT compile training step
    jit_train_step = jax.jit(
        lambda p, o, b, k: train_step_backprop(
            p, o, b, structure, optimizer, k, loss_type
        )
    )

    iter_results = []
    epoch_results = []

    for epoch_idx in range(num_epochs):
        try:
            num_batches = len(train_loader)
        except TypeError:
            raise TypeError("train_loader must support len()")

        # Split keys for batches
        epoch_rng_key, rng_key = jax.random.split(rng_key)
        batch_keys = jax.random.split(epoch_rng_key, num_batches)

        batch_losses = []
        epoch_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            # Convert batch to JAX format
            if isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            elif isinstance(batch_data, (list, tuple)):
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            # Training step
            params, opt_state, loss = jit_train_step(
                params, opt_state, batch, batch_keys[batch_idx]
            )

            epoch_loss += float(loss)

            if iter_callback is not None:
                batch_losses.append(iter_callback(epoch_idx, batch_idx, loss))
            else:
                batch_losses.append(float(loss))

        iter_results.append(batch_losses)
        avg_loss = epoch_loss / num_batches

        # Epoch callback
        if epoch_callback is not None:
            epoch_results.append(
                epoch_callback(epoch_idx, params, structure, config, rng_key)
            )
        else:
            epoch_results.append(None)

        if verbose:
            print(f"Epoch {epoch_idx + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return params, iter_results, epoch_results


def compute_loss_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    use_causal_mask: bool = True,
) -> jnp.ndarray:
    """
    Compute loss for autoregressive (next-token) prediction with backprop.

    This runs a single forward pass with causal masking and computes
    cross-entropy loss over all sequence positions.

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch with "x" (input) and "y" (target) sequences
        rng_key: JAX random key
        use_causal_mask: Whether to apply causal masking

    Returns:
        Scalar loss (mean over batch)
    """
    batch_size = batch["x"].shape[0]
    seq_len = batch["x"].shape[1]

    # Clamp input only (not output)
    clamps = {structure.task_map["x"]: batch["x"]}

    # Add causal mask if enabled
    if use_causal_mask and "causal_mask" in structure.task_map:
        causal_mask = create_causal_mask(seq_len)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq, seq)
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))
        clamps[structure.task_map["causal_mask"]] = causal_mask

    # Single forward pass
    state = initialize_state(
        structure, batch_size, rng_key,
        clamps=clamps,
        params=params
    )

    # Cross-entropy over all positions
    output_node = structure.task_map["y"]
    predictions = state.nodes[output_node].z_mu
    targets = batch["y"]

    # Handle different target formats
    if targets.ndim == 2:
        # Convert indices to one-hot
        vocab_size = predictions.shape[-1]
        targets = jax.nn.one_hot(targets, vocab_size)

    log_preds = jnp.log(predictions + 1e-10)
    loss = -jnp.sum(targets * log_preds) / batch_size

    return loss


def train_step_backprop_autoregressive(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    use_causal_mask: bool = True,
) -> Tuple[GraphParams, optax.OptState, float]:
    """
    Single autoregressive backprop training step.

    Args:
        params: Current model parameters
        opt_state: Optimizer state
        batch: Batch data with 'x' and 'y' sequences
        structure: Graph structure
        optimizer: Optax optimizer
        rng_key: JAX random key
        use_causal_mask: Whether to apply causal masking

    Returns:
        Tuple of (updated_params, updated_opt_state, loss_value)
    """
    def loss_fn(p):
        return compute_loss_autoregressive(p, structure, batch, rng_key, use_causal_mask)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, loss


def train_backprop_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
    epoch_callback: Optional[Callable] = None,
    iter_callback: Optional[Callable] = None,
) -> Tuple[GraphParams, List[List[float]], List[Any]]:
    """
    Main autoregressive backprop training loop.

    This provides standard end-to-end backpropagation training for
    autoregressive models (e.g., transformers for language modeling).

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader yielding batches with 'x' and 'y' keys
        config: Training configuration:
            - optimizer: Optimizer config dict
            - num_epochs: Number of training epochs
            - use_causal_mask: Whether to use causal masking (default True)
        rng_key: JAX random key
        verbose: Print progress
        epoch_callback: Optional callback per epoch
        iter_callback: Optional callback per batch

    Returns:
        Tuple of (trained_params, loss_history, epoch_results)
    """
    from fabricpc.training.optimizers import create_optimizer

    # Create optimizer
    optimizer = create_optimizer(config.get("optimizer", {"type": "adam", "lr": 1e-3}))
    opt_state = optimizer.init(params)

    # Training hyperparameters
    num_epochs = config.get("num_epochs", 10)
    use_causal_mask = config.get("use_causal_mask", True)

    # JIT compile training step
    jit_train_step = jax.jit(
        lambda p, o, b, k: train_step_backprop_autoregressive(
            p, o, b, structure, optimizer, k, use_causal_mask
        )
    )

    iter_results = []
    epoch_results = []

    for epoch_idx in range(num_epochs):
        try:
            num_batches = len(train_loader)
        except TypeError:
            raise TypeError("train_loader must support len()")

        # Split keys for batches
        epoch_rng_key, rng_key = jax.random.split(rng_key)
        batch_keys = jax.random.split(epoch_rng_key, num_batches)

        batch_losses = []
        epoch_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            # Convert batch to JAX format
            if isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            elif isinstance(batch_data, (list, tuple)):
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            else:
                raise ValueError(f"Unsupported batch format: {type(batch_data)}")

            # Training step
            params, opt_state, loss = jit_train_step(
                params, opt_state, batch, batch_keys[batch_idx]
            )

            epoch_loss += float(loss)

            if iter_callback is not None:
                batch_losses.append(iter_callback(epoch_idx, batch_idx, loss))
            else:
                batch_losses.append(float(loss))

        iter_results.append(batch_losses)
        avg_loss = epoch_loss / num_batches

        # Epoch callback
        if epoch_callback is not None:
            epoch_results.append(
                epoch_callback(epoch_idx, params, structure, config, rng_key)
            )
        else:
            epoch_results.append(None)

        if verbose:
            print(f"Epoch {epoch_idx + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return params, iter_results, epoch_results


def eval_step_backprop(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    loss_type: str = "cross_entropy",
) -> Tuple[float, int, int]:
    """
    Single evaluation step for backprop-trained model.

    Args:
        params: Model parameters
        batch: Batch data
        structure: Graph structure
        rng_key: Random key for initialization
        loss_type: Loss function type

    Returns:
        Tuple of (loss, correct_predictions, batch_size)
    """
    batch_size = batch["x"].shape[0]

    # Forward pass (input only clamped)
    clamps = {structure.task_map["x"]: batch["x"]}
    state = initialize_state(
        structure, batch_size, rng_key,
        clamps=clamps,
        params=params
    )

    # Get predictions
    output_node = structure.task_map["y"]
    predictions = state.nodes[output_node].z_mu
    targets = batch["y"]

    # Compute loss
    if loss_type == "cross_entropy":
        log_preds = jnp.log(predictions + 1e-10)
        loss = -jnp.sum(targets * log_preds) / batch_size
    elif loss_type == "mse":
        loss = jnp.mean((predictions - targets) ** 2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Compute accuracy (for classification)
    pred_labels = jnp.argmax(predictions, axis=-1)
    if targets.ndim > 1 and targets.shape[-1] > 1:
        true_labels = jnp.argmax(targets, axis=-1)
    else:
        true_labels = targets

    # Handle sequence vs non-sequence
    correct = jnp.sum(pred_labels == true_labels)

    return float(loss), int(correct), int(jnp.prod(jnp.array(pred_labels.shape)))


def evaluate_backprop(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array = None,
) -> Dict[str, float]:
    """
    Evaluate backprop-trained model on test data.

    Args:
        params: Trained parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation config with loss_type
        rng_key: Random key (optional, uses fixed key if None)

    Returns:
        Dictionary with "loss", "accuracy", and optionally "perplexity"
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    loss_type = config.get("loss_type", "cross_entropy")

    # JIT compile eval step
    jit_eval_step = jax.jit(
        lambda p, b, k: eval_step_backprop(p, b, structure, k, loss_type)
    )

    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000

    batch_keys = jax.random.split(rng_key, num_batches)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, batch_data in enumerate(test_loader):
        # Convert batch
        if isinstance(batch_data, dict):
            batch = {k: jnp.array(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")

        loss, correct, count = jit_eval_step(params, batch, batch_keys[batch_idx])

        total_loss += loss * count
        total_correct += correct
        total_samples += count

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    result = {
        "loss": avg_loss,
        "accuracy": accuracy,
    }

    # Add perplexity for cross-entropy loss
    if loss_type == "cross_entropy":
        result["perplexity"] = float(jnp.exp(avg_loss))

    return result