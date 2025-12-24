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

from fabricpc.core.types import GraphParams, GraphStructure, GraphState
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.training.train_autoregressive import create_causal_mask, compute_loss


def compute_forward_pass(
        params: GraphParams,
        structure: GraphStructure,
        batch: Dict[str, jnp.ndarray],
        rng_key: jax.Array,
        loss_type: str = "cross_entropy",
    ) -> jnp.ndarray:
    """
    Compute forward pass for backpropagation training.

    This runs a single forward pass through the network (no iterative
    inference) and computes output at the output node.

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch data with keys matching task_map (e.g., 'x', 'y')
        rng_key: JAX random key for state initialization
        loss_type: Loss function type: "cross_entropy" or "mse"

    Returns:
        Output predictions from the output node
    """
    batch_size = batch["x"].shape[0]

    # Clamp ONLY input node (not output!) - key difference from PC
    clamps = {}
    if "x" in structure.task_map:
        clamps[structure.task_map["x"]] = batch["x"]

    # Single forward pass via initialize_state with feedforward init
    state = initialize_graph_state(
        structure, batch_size, rng_key,
        clamps=clamps,
        state_init_config=structure.config["graph_state_initializer"],
        params=params
    )
    output_node = structure.task_map["y"]
    targets = batch["y"]

    return compute_loss(state, targets, output_node, loss_type)


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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        Tuple of (loss, predictions) where:
            - loss: Scalar loss (mean over batch)
            - predictions: Output logits of shape (batch, seq_len, vocab_size)
    """
    batch_size = batch["x"].shape[0]
    seq_len = batch["x"].shape[1]

    # Clamp input only (not output)
    clamps = {structure.task_map["x"]: batch["x"]}

    # Add causal mask if enabled
    if use_causal_mask:
        if "causal_mask" not in structure.task_map:
            raise ValueError("Causal masking enabled but 'causal_mask' not in task_map")
        causal_mask = create_causal_mask(seq_len)
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq, seq)
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))
        clamps[structure.task_map["causal_mask"]] = causal_mask

    # Single forward pass
    state = initialize_graph_state(
        structure, batch_size, rng_key,
        clamps=clamps,
        state_init_config=structure.config["graph_state_initializer"],
        params=params
    )

    # Cross-entropy over all positions
    output_node = structure.task_map["y"]
    predictions = state.nodes[output_node].z_mu
    targets = batch["y"]

    loss = compute_loss(state, targets, output_node, "cross_entropy")

    return loss, predictions


def train_step_backprop_autoregressive(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    use_causal_mask: bool = True,
) -> Tuple[GraphParams, optax.OptState, float, jnp.ndarray]:
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
        Tuple of (updated_params, updated_opt_state, loss_value, predictions)
    """
    def loss_fn(p):
        return compute_loss_autoregressive(p, structure, batch, rng_key, use_causal_mask)

    # has_aux=True: loss_fn returns (loss, aux), grads computed w.r.t. loss only
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, loss, predictions


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

            # Training step (predictions discarded during training)
            params, opt_state, loss, _ = jit_train_step(
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
            perplexity = float(jnp.exp(avg_loss))
            print(f"Epoch {epoch_idx + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")

    return params, iter_results, epoch_results


def eval_step_backprop(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    loss_type: str = "cross_entropy",
) -> Tuple[float, int, int]:
    """
    Single evaluation step for backprop-trained model. No latent inference phase. Use structure configured for feedforward graph_state_initializer.

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
    state = initialize_graph_state(
        structure, batch_size, rng_key,
        clamps=clamps,
        state_init_config=structure.config["graph_state_initializer"],
        params=params
    )

    # Get predictions
    output_node = structure.task_map["y"]
    predictions = state.nodes[output_node].z_mu
    targets = batch["y"]

    loss = compute_loss(state, targets, output_node, loss_type)

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
        raise TypeError("test_loader must support len()")

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


def evaluate_backprop_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Evaluate backprop-trained autoregressive model on test data.

    Computes:
    - Average cross-entropy loss
    - Perplexity (exp of average loss)
    - Accuracy (next-token prediction)

    Args:
        params: Trained model parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation config with use_causal_mask (default True)
        rng_key: JAX random key
        debug: If True, print diagnostic info for first batch

    Returns:
        Dictionary with loss, perplexity, and accuracy
    """
    use_causal_mask = config.get("use_causal_mask", True)

    try:
        num_batches_total = len(test_loader)
    except TypeError:
        raise TypeError("test_loader must support len() for evaluation")

    batch_keys = jax.random.split(rng_key, num_batches_total)

    # JIT compile: returns (loss, predictions) tuple
    jit_forward = jax.jit(
        lambda p, b, k: compute_loss_autoregressive(p, structure, b, k, use_causal_mask)
    )

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch_idx, batch_data in enumerate(test_loader):
        # Convert batch to JAX format
        if isinstance(batch_data, dict):
            batch = {k: jnp.array(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")

        batch_size = batch["x"].shape[0]
        seq_len = batch["x"].shape[1]

        # Single forward pass returns both loss and predictions
        loss, predictions = jit_forward(params, batch, batch_keys[batch_idx])

        # Debug diagnostics for first batch
        if debug and batch_idx == 0:
            tgt = batch["y"]  # (batch, seq_len, vocab_size) one-hot

            # Check individual loss components
            log_preds = jnp.log(predictions + 1e-10)
            per_token_loss = -jnp.sum(tgt * log_preds, axis=-1)  # (batch, seq_len)
            print(f"  [DEBUG] per-token CE loss: min={float(jnp.min(per_token_loss)):.4f}, max={float(jnp.max(per_token_loss)):.4f}, mean={float(jnp.mean(per_token_loss)):.4f}")

            token_intrinsic_perplexity = jnp.exp(-jnp.sum(predictions * log_preds, axis=-1))  # (batch, seq_len)
            print(f"  [DEBUG] per-token intrinsic perplexity: min={float(jnp.min(token_intrinsic_perplexity)):.4f}, max={float(jnp.max(token_intrinsic_perplexity)):.4f}, mean={float(jnp.mean(token_intrinsic_perplexity)):.4f}")

            # Check if there are extreme values
            correct_probs = jnp.sum(tgt * predictions, axis=-1)  # prob assigned to correct class
            print(f"  [DEBUG] prob of correct token: min={float(jnp.min(correct_probs)):.6f}, max={float(jnp.max(correct_probs)):.6f}, mean={float(jnp.mean(correct_probs)):.6f}")

            print(f"  [DEBUG] batch loss: {float(loss):.4f}")

        total_loss += float(loss)

        # Get target tokens
        targets = batch["y"]
        if targets.ndim == 3:
            target_tokens = jnp.argmax(targets, axis=-1)
        else:
            target_tokens = targets

        pred_tokens = jnp.argmax(predictions, axis=-1)
        correct = jnp.sum(pred_tokens == target_tokens)
        total_correct += int(correct)
        total_tokens += batch_size * seq_len
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    perplexity = float(jnp.exp(avg_loss))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_batches": num_batches,
    }