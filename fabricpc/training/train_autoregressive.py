"""
Autoregressive training loop for transformer predictive coding networks.

This module implements training for autoregressive (next-token prediction) tasks
using the predictive coding framework. Key features:

1. Teacher forcing: During training, use ground truth tokens as input
2. Causal masking: Ensure predictions only depend on past tokens
3. Sequence-level energy: Aggregate prediction errors across positions
4. Local Hebbian learning: Weight updates based on local gradients

The training loop supports both:
- Full sequence prediction (all positions predict next token)
- Last-token prediction (only predict final token, for efficiency)
"""

from typing import Dict, Tuple, Any, List, Optional, Callable, cast
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeParams
from fabricpc.core.inference import run_inference, gather_inputs
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.nodes.base import _get_node_class_from_info


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """
    Create a causal attention mask for autoregressive modeling.

    Returns:
        Mask of shape (seq_len, seq_len) where mask[i,j] = 1 if j <= i else 0
        This ensures position i can only attend to positions 0...i
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))


def compute_loss(
    final_state: GraphState,
    targets: jnp.ndarray,
    output_node: str,
    loss_type: str = "cross_entropy",
) -> jnp.ndarray:
    """
    Compute differentiable loss

    Args:
        final_state: Final graph state after forward pass
        targets: Target values (labels) for output node, expects one-hot for cross-entropy
        output_node: Name of the output node
        loss_type: Loss function type: "cross_entropy" or "mse"

    Returns:
        Scalar loss value (mean over batch)
    """

    # Get prediction from output node
    predictions = final_state.nodes[output_node].z_mu

    # Compute loss
    if loss_type == "cross_entropy":
        log_preds = jnp.log(predictions + 1e-10)
        loss = -jnp.mean(jnp.sum(targets * log_preds, axis=-1))

    elif loss_type == "mse":
        loss = jnp.mean((predictions - targets) ** 2)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss


def compute_local_weight_gradients_ar(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """
    Compute local weight gradients for autoregressive training.

    This is similar to the standard local gradient computation but
    can be extended for sequence-specific optimizations.

    Args:
        params: Current model parameters
        final_state: Converged state after inference
        structure: Graph structure

    Returns:
        GraphParams containing gradients
    """
    gradients = {}

    for node_name, node in structure.nodes.items():
        node_info = node.node_info
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        in_edges_data = gather_inputs(node_info, structure, final_state)
        node_class = _get_node_class_from_info(node_info)

        # Compute local gradients
        node_state, grad_params = node_class.forward_learning(
            params.nodes[node_name],
            in_edges_data,
            final_state.nodes[node_name],
            node_info,
        )

        gradients[node_name] = grad_params

    return GraphParams(nodes=gradients)


def train_step_autoregressive(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    infer_steps: int,
    eta_infer: float = 0.1,
    use_causal_mask: bool = True,
) -> Tuple[GraphParams, optax.OptState, float, float, GraphState]:
    """
    Single autoregressive training step.

    This implements the predictive coding training loop for sequence prediction:
    1. Clamp input sequence and target sequence using task_map
    2. Generate and apply causal masking via task_map["causal_mask"]
    3. Run inference to convergence
    4. Compute local gradients
    5. Update weights

    Args:
        params: Current model parameters
        opt_state: Optimizer state
        batch: Batch with keys matching task_map (e.g., 'x' for input, 'y' for target)
            x: (batch, seq_len, vocab_size) or (batch, seq_len)
            y: (batch, seq_len, vocab_size) or (batch, seq_len)
        structure: Graph structure with task_map defining input/output nodes.
            For causal masking, task_map should include "causal_mask" -> node_name
        optimizer: Optax optimizer
        rng_key: JAX random key
        infer_steps: Number of inference steps
        eta_infer: Inference learning rate
        use_causal_mask: Whether to apply causal masking

    Returns:
        Tuple of (updated_params, updated_opt_state, avg_energy, output_cross_entropy, final_state)
    """
    batch_size = batch["x"].shape[0]
    seq_len = batch["x"].shape[1]

    # Map tasks to nodes using task_map
    clamps = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map:
            node_name = structure.task_map[task_name]
            clamps[node_name] = task_value

    # Generate and clamp causal mask if enabled and configured in task_map
    if use_causal_mask:
        if "causal_mask" not in structure.task_map:
            raise ValueError("Causal masking enabled but 'causal_mask' not in task_map")
        # Create causal mask: (seq_len, seq_len) where mask[i,j] = 1 if j <= i
        causal_mask = create_causal_mask(seq_len)
        # Broadcast to (batch, 1, seq_len, seq_len) for attention scores
        causal_mask = causal_mask[None, None, :, :]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))

        # Clamp the causal mask to the node specified in task_map
        mask_node = structure.task_map["causal_mask"]
        clamps[mask_node] = causal_mask

    # Initialize state
    init_state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
    )

    # Run inference
    final_state = run_inference(
        params, init_state, clamps, structure, infer_steps, eta_infer
    )

    # Compute total energy (sum over non-source nodes)
    energy = jnp.array(0.0)
    for node_name, node in structure.nodes.items():
        if node.node_info.in_degree > 0:
            energy = energy + jnp.sum(final_state.nodes[node_name].energy)

    avg_energy = energy / batch_size

    # Compute local gradients
    grads = compute_local_weight_gradients_ar(params, final_state, structure)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    # Compute output cross-entropy loss for perplexity metric - not used for gradients
    output_cross_entropy = compute_loss(
        final_state, batch["y"], structure.task_map["y"], loss_type="cross_entropy"
    )

    return (
        params,
        opt_state,
        avg_energy.astype(float),
        output_cross_entropy.astype(float),
        final_state,
    )


def train_autoregressive(
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
    Main training loop for autoregressive predictive coding.

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader yielding batches with 'x' and 'y' keys
        config: Training configuration:
            - optimizer: Optimizer config dict
            - num_epochs: Number of training epochs
            - infer_steps: Inference steps per training step
            - eta_infer: Inference learning rate
            - use_causal_mask: Whether to use causal masking (default True)
            - gradient_accumulation_steps: Steps to accumulate gradients (default 1)
        rng_key: JAX random key
        verbose: Whether to print progress
        epoch_callback: Optional callback (epoch, params, structure, config, rng) -> any
        iter_callback: Optional callback (epoch, batch_idx, loss) -> any

    Returns:
        Tuple of (trained_params, energy_history, epoch_results)
    """
    from fabricpc.training.optimizers import create_optimizer

    # Create optimizer
    optimizer = create_optimizer(config.get("optimizer", {"type": "adam", "lr": 1e-3}))
    opt_state = optimizer.init(params)

    # Training hyperparameters
    infer_steps = config.get("infer_steps", 20)
    eta_infer = config.get("eta_infer", 0.1)
    num_epochs = config.get("num_epochs", 10)
    use_causal_mask = config.get("use_causal_mask", True)
    grad_accum_steps = config.get("gradient_accumulation_steps", 1)

    # JIT compile training step
    jit_train_step = jax.jit(
        lambda p, o, b, k: train_step_autoregressive(
            p, o, b, structure, optimizer, k, infer_steps, eta_infer, use_causal_mask
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

        batch_energies = []
        epoch_energy = 0.0
        epoch_ce_loss = 0.0

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
            params, opt_state, energy, ce_loss, _ = jit_train_step(
                params, opt_state, batch, batch_keys[batch_idx]
            )

            epoch_energy += energy
            epoch_ce_loss += ce_loss

            if iter_callback is not None:
                batch_energies.append(iter_callback(epoch_idx, batch_idx, energy))
            else:
                batch_energies.append(energy)

        iter_results.append(batch_energies)
        avg_energy = epoch_energy / num_batches
        avg_ce_loss = epoch_ce_loss / num_batches

        # Epoch callback
        if epoch_callback is not None:
            epoch_results.append(
                epoch_callback(epoch_idx, params, structure, config, rng_key)
            )
        else:
            epoch_results.append(None)

        if verbose:
            perplexity = float(jnp.exp(avg_ce_loss))
            print(
                f"Train Epoch {epoch_idx + 1}/{num_epochs}, Energy: {avg_energy:.4f}, Loss: {avg_ce_loss:.4f}, Perplexity: {perplexity:.2f}"
            )

    return params, iter_results, epoch_results


def _generation_step(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jax.Array],
    step_idx: int,
    params: GraphParams,
    structure: GraphStructure,
    input_node: str,
    output_node: str,
    seq_len: int,
    vocab_size: int,
    batch_size: int,
    infer_steps: int,
    eta_infer: float,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jax.Array], jnp.ndarray]:
    """
    Single generation step for use with jax.lax.scan.

    Uses a sliding window approach with fixed-size buffers to maintain
    constant shapes required by JAX scan.

    Args:
        carry: Tuple of (context_window, output_buffer, rng_key)
            - context_window: Fixed-size (batch, seq_len) window of recent tokens
            - output_buffer: Fixed-size (batch, max_new_tokens) buffer for generated tokens
            - rng_key: JAX random key
        step_idx: Current step index, used to write to output buffer
        params, structure, etc.: Static parameters passed via closure

    Returns:
        Tuple of (new_carry, next_token)
    """
    context_window, output_buffer, rng_key = carry
    rng_key, sample_key, init_key = jax.random.split(rng_key, 3)

    # Convert context window to one-hot
    input_onehot = jax.nn.one_hot(context_window, vocab_size)

    # Create clamps (only input, not output)
    clamps = {input_node: input_onehot}

    # Initialize and run inference
    state = initialize_graph_state(
        structure,
        batch_size,
        init_key,
        clamps=clamps,
        params=params,
    )
    if infer_steps > 0:
        final_state = run_inference(
            params, state, clamps, structure, infer_steps, eta_infer
        )
    else:
        final_state = state

    # Get output for the last position
    # z_mu contains the predicted output after activation (softmax for output node)
    # z_latent is the raw latent state before activation
    output_probs = final_state.nodes[output_node].z_mu
    output_last = output_probs[:, -1, :]  # (batch, vocab_size)

    # Convert to log-probabilities for sampling
    # z_mu after softmax should be probabilities, convert to log-probs
    # Adding epsilon avoids log(0)
    logits = jnp.log(output_last + 1e-10)

    # Apply temperature (divide log-probs, equivalent to taking prob^(1/T))
    logits = logits / temperature

    # Apply top-k filtering (always run but with large k if not specified)
    effective_top_k = top_k if top_k is not None else vocab_size
    top_k_logits, top_k_indices = jax.lax.top_k(logits, effective_top_k)
    # Set non-top-k logits to -inf
    neg_inf_mask = jnp.full_like(logits, float("-inf"))
    logits = neg_inf_mask.at[jnp.arange(batch_size)[:, None], top_k_indices].set(
        top_k_logits
    )

    # Apply top-p (nucleus) filtering
    if top_p is not None:
        sorted_indices = jnp.argsort(-logits, axis=-1)
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
        # Find cutoff
        cutoff_mask = cumsum_probs > top_p
        # Shift mask to keep at least one token
        cutoff_mask = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=bool), cutoff_mask[:, :-1]], axis=-1
        )
        sorted_logits = jnp.where(cutoff_mask, float("-inf"), sorted_logits)
        # Unsort
        unsort_indices = jnp.argsort(sorted_indices, axis=-1)
        logits = jnp.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    # Sample from distribution
    next_token = jax.random.categorical(sample_key, logits, axis=-1)  # (batch,)

    # Update context window: shift left and append new token
    new_context = jnp.concatenate([context_window[:, 1:], next_token[:, None]], axis=1)

    # Write to output buffer at current step index
    new_output_buffer = output_buffer.at[:, step_idx].set(next_token)

    return (new_context, new_output_buffer, rng_key), next_token


def generate_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    prompt: jnp.ndarray,
    max_new_tokens: int,
    rng_key: jax.Array,
    temperature: float = 1.0,
    infer_steps: int = 20,
    eta_infer: float = 0.1,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> jnp.ndarray:
    """
    Generate tokens autoregressively using the trained model.

    This function is JIT-compiled for efficient generation. The inner loop
    uses jax.lax.scan with fixed-size buffers for optimal performance.

    Args:
        params: Trained model parameters
        structure: Graph structure
        prompt: Initial token indices, shape (seq_len,) or (batch, seq_len)
        max_new_tokens: Number of new tokens to generate
        rng_key: JAX random key
        temperature: Sampling temperature (higher = more random, 1=neutral, <1=less random)
        infer_steps: Inference steps per generation step
        eta_infer: Inference learning rate
        top_k: If set, only sample from top-k tokens
        top_p: If set, use nucleus sampling with this probability threshold

    Returns:
        Generated token indices, shape (prompt_len + max_new_tokens,) or (batch, prompt_len + max_new_tokens)
    """
    # Handle batched vs unbatched input
    if prompt.ndim == 1:
        prompt = prompt[None, :]  # Add batch dimension
        unbatch = True
    else:
        unbatch = False

    batch_size, prompt_len = prompt.shape
    input_node = structure.task_map.get("x")
    output_node = structure.task_map.get("y")

    if input_node is None or output_node is None:
        raise ValueError("Structure must have 'x' and 'y' in task_map")

    vocab_size = structure.nodes[output_node].node_info.shape[-1]
    seq_len = structure.nodes[input_node].node_info.shape[0]

    # Prepare initial context window (pad or truncate prompt to seq_len)
    if prompt_len >= seq_len:
        # Use last seq_len tokens as context
        context_window = prompt[:, -seq_len:]
    else:
        # Pad at the beginning with zeros
        pad_len = seq_len - prompt_len
        context_window = jnp.pad(prompt, ((0, 0), (pad_len, 0)), constant_values=0)

    # Create JIT-compiled generation step with static arguments closed over
    @jax.jit
    def jit_generate_loop(context: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        """JIT-compiled generation loop using lax.scan."""

        def scan_fn(carry, step_idx):
            return _generation_step(
                carry,
                step_idx,
                params=params,
                structure=structure,
                input_node=input_node,
                output_node=output_node,
                seq_len=seq_len,
                vocab_size=vocab_size,
                batch_size=batch_size,
                infer_steps=infer_steps,
                eta_infer=eta_infer,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Initialize output buffer for generated tokens
        output_buffer = jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)

        # Run the generation loop
        init_carry = (context, output_buffer, rng)
        (_, final_output_buffer, _), _ = jax.lax.scan(
            scan_fn, init_carry, jnp.arange(max_new_tokens)
        )

        return final_output_buffer

    # Run the JIT-compiled generation
    generated_tokens = jit_generate_loop(context_window, rng_key)

    # Concatenate prompt with generated tokens
    result = jnp.concatenate([prompt, generated_tokens], axis=1)

    if unbatch:
        result = result[0]  # Remove batch dimension

    return result


def _eval_step_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    batch: Dict[str, jnp.ndarray],
    rng_key: jax.Array,
    infer_steps: int,
    eta_infer: float,
    use_causal_mask: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Single evaluation step for autoregressive model (JIT-compilable).

    Args:
        params: Model parameters
        structure: Graph structure
        batch: Batch with 'x' and 'y'
        rng_key: Random key
        infer_steps: Number of inference steps
        eta_infer: Inference learning rate
        use_causal_mask: Whether to use causal masking

    Returns:
        Tuple of (output_cross_entropy, predictions)
    """
    batch_size = batch["x"].shape[0]
    seq_len = batch["x"].shape[1]

    # Create clamps (input only for evaluation)
    clamps = {structure.task_map["x"]: batch["x"]}

    # Add causal mask if enabled
    if use_causal_mask:
        causal_mask = create_causal_mask(seq_len)
        causal_mask = causal_mask[None, None, :, :]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))
        clamps[structure.task_map["causal_mask"]] = causal_mask

    # Initialize and run inference
    state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
    )
    if infer_steps > 0:
        final_state = run_inference(
            params, state, clamps, structure, infer_steps, eta_infer
        )
    else:
        final_state = state

    # Compute loss and get predictions
    output_node = structure.task_map["y"]
    output_cross_entropy = compute_loss(
        final_state, batch["y"], output_node, loss_type="cross_entropy"
    )  # For perplexity metric
    predictions = final_state.nodes[output_node].z_mu

    return output_cross_entropy, predictions


def evaluate_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Evaluate autoregressive model on test data.

    Computes:
    - Average loss (cross-entropy)
    - Perplexity (if cross-entropy loss)
    - Accuracy (next-token prediction)

    Args:
        params: Trained parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation config (infer_steps, eta_infer, use_causal_mask)
        rng_key: Random key
        debug: If True, print detailed diagnostics for first batch

    Returns:
        Dictionary of metrics
    """
    infer_steps = config["infer_steps"]
    eta_infer = config["eta_infer"]
    use_causal_mask = config["use_causal_mask"]

    output_node = structure.task_map.get("y")
    if output_node is None:
        raise ValueError("Structure must have 'y' in task_map")

    if use_causal_mask and "causal_mask" not in structure.task_map:
        raise ValueError("Causal masking enabled but 'causal_mask' not in task_map")

    try:
        num_batches_total = len(test_loader)
    except TypeError:
        raise TypeError("test_loader must support len()")

    batch_keys = jax.random.split(rng_key, num_batches_total)

    # JIT compile the evaluation step
    jit_eval_step = jax.jit(
        lambda p, b, k: _eval_step_autoregressive(
            p, structure, b, k, infer_steps, eta_infer, use_causal_mask
        )
    )

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch_idx, batch_data in enumerate(test_loader):
        # Convert batch
        if isinstance(batch_data, dict):
            batch = {k: jnp.array(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        else:
            raise ValueError(f"Unsupported batch format: {type(batch_data)}")

        batch_size = batch["x"].shape[0]

        # Run JIT-compiled evaluation step
        loss, predictions = jit_eval_step(params, batch, batch_keys[batch_idx])
        total_loss += float(loss)

        # Debug diagnostics for first batch
        if debug and batch_idx == 0:
            tgt = batch["y"]  # (batch, seq_len, vocab_size) one-hot

            # Check individual loss components
            log_preds = jnp.log(predictions + 1e-10)
            per_token_loss = -jnp.sum(tgt * log_preds, axis=-1)  # (batch, seq_len)
            print(
                f"  [DEBUG] per-token CE loss: min={float(jnp.min(per_token_loss)):.4f}, max={float(jnp.max(per_token_loss)):.4f}, mean={float(jnp.mean(per_token_loss)):.4f}"
            )

            token_intrinsic_perplexity = jnp.exp(
                -jnp.sum(predictions * log_preds, axis=-1)
            )  # (batch, seq_len)
            print(
                f"  [DEBUG] per-token intrinsic perplexity: min={float(jnp.min(token_intrinsic_perplexity)):.4f}, max={float(jnp.max(token_intrinsic_perplexity)):.4f}, mean={float(jnp.mean(token_intrinsic_perplexity)):.4f}"
            )

            # Check if there are extreme values
            correct_probs = jnp.sum(
                tgt * predictions, axis=-1
            )  # prob assigned to correct class
            print(
                f"  [DEBUG] prob of correct token: min={float(jnp.min(correct_probs)):.6f}, max={float(jnp.max(correct_probs)):.6f}, mean={float(jnp.mean(correct_probs)):.6f}"
            )

            print(f"  [DEBUG] batch loss: {float(loss):.4f}")

        # Compute accuracy
        targets = batch["y"]
        pred_tokens = jnp.argmax(predictions, axis=-1)
        if targets.ndim == 3:
            target_tokens = jnp.argmax(targets, axis=-1)
        else:
            target_tokens = targets

        correct = jnp.sum(pred_tokens == target_tokens)
        total_correct += int(correct)
        total_tokens += int(batch_size * predictions.shape[1])
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    perplexity = float(jnp.exp(avg_loss))
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_batches": num_batches,
    }
