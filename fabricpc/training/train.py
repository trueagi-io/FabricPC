"""
Training and evaluation for JAX predictive coding networks.

Supports automatic device detection: uses JIT on a single device,
pmap across multiple devices. All functions work transparently on
1 or N devices.
"""

from typing import Dict, Tuple, Any, cast, List, Optional, Callable
import math
import warnings
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm as _tqdm_cls

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import run_inference
from fabricpc.graph.graph_net import compute_local_weight_gradients

# ── pmap utilities ──────────────────────────────────────────────────


def replicate_params(params: GraphParams, n_devices: int) -> GraphParams:
    """
    Replicate parameters across devices for pmap.

    Args:
        params: Single-device parameters
        n_devices: Number of devices

    Returns:
        Replicated parameters with leading device axis
    """
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), params)


def replicate_opt_state(opt_state: optax.OptState, n_devices: int) -> optax.OptState:
    """
    Replicate optimizer state across devices.

    Args:
        opt_state: Single-device optimizer state
        n_devices: Number of devices

    Returns:
        Replicated optimizer state with leading device axis
    """
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), opt_state)


def shard_batch(
    batch: Dict[str, jnp.ndarray], n_devices: int
) -> Dict[str, jnp.ndarray]:
    """
    Shard a batch across devices for pmap.

    Reshapes batch from (total_batch_size, ...) to (n_devices, batch_per_device, ...).

    Args:
        batch: Batch with shape (total_batch_size, ...)
        n_devices: Number of devices

    Returns:
        Sharded batch with shape (n_devices, batch_per_device, ...)

    Example:
        >>> batch = {'x': jnp.zeros((128, 784)), 'y': jnp.zeros((128, 10))}
        >>> sharded = shard_batch(batch, n_devices=4)
        >>> sharded['x'].shape
        (4, 32, 784)
    """

    def shard_array(x):
        batch_size = x.shape[0]
        if batch_size % n_devices != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by number of devices {n_devices}"
            )
        batch_per_device = batch_size // n_devices
        return x.reshape(n_devices, batch_per_device, *x.shape[1:])

    return jax.tree_util.tree_map(shard_array, batch)


def unshard_energies(energies: jnp.ndarray) -> float:
    """
    Average energy from all devices.

    Args:
        energies: Energy values from each device, shape (n_devices,)

    Returns:
        Average energy across devices
    """
    return float(jnp.mean(energies))


def _convert_batch(batch_data) -> Dict[str, jnp.ndarray]:
    """Convert loader output to dict of JAX arrays."""
    if isinstance(batch_data, (list, tuple)):
        return {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
    elif isinstance(batch_data, dict):
        return {k: jnp.array(v) for k, v in batch_data.items()}
    else:
        raise ValueError(f"Unsupported batch format: {type(batch_data)}")


# ── gradient computation ────────────────────────────────────────────


def get_graph_param_gradient(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
) -> Tuple[GraphParams, float, GraphState]:
    """
    Compute parameter gradients for a batch without updating parameters.

    Shared by both single-device (JIT) and multi-device (pmap) training.

    Args:
        params: Current model parameters
        batch: Batch of data with task-specific keys
        structure: Graph structure
        rng_key: JAX random key for state initialization

    Returns:
        Tuple of (grads, energy_per_sample, final_state)
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

    # Compute energy (ignore terminal input nodes), normalized per sample
    energy = sum(
        [
            sum(final_state.nodes[node_name].energy)
            for node_name in structure.nodes
            if structure.nodes[node_name].node_info.in_degree > 0
        ]
    )
    energy = energy / batch_size

    # Compute LOCAL gradients for each node
    grads = compute_local_weight_gradients(params, final_state, structure)

    return grads, energy, final_state


# ── training steps ──────────────────────────────────────────────────


def train_step(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
) -> Tuple[GraphParams, optax.OptState, float, GraphState]:
    """
    Single training step with local weight updates (JIT path).

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
        Tuple of (updated_params, updated_opt_state, energy_per_sample, final_state)
    """
    grads, energy, final_state = get_graph_param_gradient(
        params, batch, structure, rng_key
    )

    # Update parameters using optimizer
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, energy, final_state


def train_step_pmap(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
) -> Tuple[GraphParams, optax.OptState, jnp.ndarray, GraphState]:
    """
    Training step for pmap: compute grads, average across devices, update.

    Uses local Hebbian learning (same as single-device) via shared
    get_graph_param_gradient, then averages gradients across devices.

    Args:
        params: Replicated parameters (has device axis)
        opt_state: Replicated optimizer state (has device axis)
        batch: Sharded batch (has device axis)
        structure: Graph structure
        rng_key: JAX random key for this device
        optimizer: Optax optimizer

    Returns:
        Tuple of (updated_params, updated_opt_state, energy_per_device, final_state)
    """
    grads, energy, final_state = get_graph_param_gradient(
        params, batch, structure, rng_key
    )

    # Average gradients across all devices (data parallelism)
    grads = jax.lax.pmean(grads, axis_name="devices")

    # Update parameters (each device now has same gradients)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, energy, final_state


def create_pmap_train_step(
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
):
    """
    Create a pmap'd training step with static arguments captured in closure.

    We can't use static_broadcasted_argnums because GraphStructure contains dicts.
    Instead, we capture structure and optimizer in a closure.

    Args:
        structure: Graph structure (static)
        optimizer: Optimizer (static)

    Returns:
        Pmap'd training step function
    """

    def step_fn(params, opt_state, batch, rng_key):
        return train_step_pmap(
            params,
            opt_state,
            batch,
            structure,
            rng_key,
            optimizer,
        )

    return jax.pmap(step_fn, axis_name="devices")


# ── main training loop ──────────────────────────────────────────────


def train_pcn(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
    use_tqdm: bool = True,
    epoch_callback: Optional[Callable] = None,
    iter_callback: Optional[Callable] = None,
    pmap_single_device: bool = False,
) -> Tuple[GraphParams, List[Any], List[Any]]:
    """
    Train a predictive coding network with local learning.

    Automatically detects available devices: uses JIT compilation on a single
    device, pmap data parallelism across multiple devices.

    Args:
        params: Initial parameters
        structure: Graph structure
        train_loader: Data loader (iterable yielding batches)
        optimizer: Optax optimizer (e.g., optax.adam(1e-3))
        config: Training configuration with keys:
            - num_epochs: Number of training epochs (supports fractional, e.g. 1.5)
        rng_key: JAX random key (will be split for each batch)
        verbose: Whether to print progress info (device count, epoch summaries)
        use_tqdm: Whether to show a TQDM progress bar counting total minibatches
        epoch_callback: Optional function called at end of each epoch:
            (epoch_idx, params, structure, config, rng_key) -> any
        iter_callback: Optional function called at end of each batch:
            (epoch_idx, batch_idx, energy) -> any
        pmap_single_device: If True, forces the pmap code path even on a
            single device. Useful for testing pmap logic.

    Returns:
        Tuple of (trained_params, iter_results, epoch_results):
        - trained_params: Updated model parameters
        - iter_results: 2D list of energy values [epochs][batches]
        - epoch_results: List of epoch_callback return values

    Example:
        >>> rng_key = jax.random.PRNGKey(0)
        >>> params_key, train_key = jax.random.split(rng_key, 2)
        >>> params = initialize_params(structure, params_key)
        >>> optimizer = optax.adam(1e-3)
        >>> config = {"num_epochs": 10}
        >>> trained_params, energies, metrics = train_pcn(
        ...     params, structure, train_loader, optimizer, config, train_key
        ... )
    """
    # Device detection
    n_devices = jax.device_count()
    use_pmap = (n_devices > 1) or pmap_single_device

    if verbose:
        print(f"Training on {n_devices} device(s): {jax.devices()}")

    # Initialize optimizer
    opt_state = optimizer.init(params)

    # Set up step function based on device path
    if use_pmap:
        params = replicate_params(params, n_devices)
        opt_state = replicate_opt_state(opt_state, n_devices)
        step_fn = create_pmap_train_step(structure, optimizer)
    else:
        step_fn = jax.jit(
            lambda p, o, b, k: train_step(p, o, b, structure, optimizer, k)
        )

    # Epoch parameters
    num_epochs = config.get("num_epochs", 10)
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)

    # Compute total batches for TQDM progress bar
    num_batches = len(train_loader)
    total_batches = 0
    for e in range(total_epochs):
        if e == total_epochs - 1 and frac > 0:
            total_batches += round(frac * num_batches)
        else:
            total_batches += num_batches

    # Progress bar across all epochs
    progress = _tqdm_cls(total=total_batches, disable=not use_tqdm, leave=True)
    _shard_warned = False

    # Training loop
    iter_results = []
    epoch_results = []
    for epoch_idx in range(total_epochs):
        # On the final epoch, truncate if fractional
        is_last_epoch = epoch_idx == total_epochs - 1
        if is_last_epoch and frac > 0:
            max_batches = round(frac * num_batches)
        else:
            max_batches = num_batches

        progress.set_description(f"Epoch {epoch_idx + 1}/{total_epochs}")

        # Split RNG keys for this epoch
        epoch_rng_key, rng_key = jax.random.split(rng_key)

        if use_pmap:
            # Per-device keys
            if n_devices > 1:
                device_keys = jax.random.split(epoch_rng_key, n_devices)
            else:
                device_keys = jnp.expand_dims(epoch_rng_key, axis=0)

            batch_keys_per_device = jax.vmap(
                lambda k: jax.random.split(k, max_batches)
            )(device_keys)
        else:
            batch_keys = jax.random.split(epoch_rng_key, max_batches)

        batch_energies = []
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            batch = _convert_batch(batch_data)

            if use_pmap:
                batch_key = batch_keys_per_device[:, batch_idx]
                try:
                    batch_sharded = shard_batch(batch, n_devices)
                except ValueError as e:
                    if not _shard_warned:
                        warnings.warn(
                            f"Skipping batch (size not divisible by {n_devices}): {e}"
                        )
                        _shard_warned = True
                    progress.update(1)
                    continue

                params, opt_state, energies, _ = step_fn(
                    params, opt_state, batch_sharded, batch_key
                )
                energy = unshard_energies(energies)
            else:
                batch_key = batch_keys[batch_idx]
                params, opt_state, energy, _ = step_fn(
                    params, opt_state, batch, batch_key
                )
                energy = float(energy)

            if iter_callback is not None:
                batch_energies.append(iter_callback(epoch_idx, batch_idx, energy))
            else:
                batch_energies.append(energy)

            progress.update(1)

        iter_results.append(batch_energies)

        # Compute average energy for epoch
        avg_energy = (
            sum(batch_energies) / len(batch_energies) if batch_energies else 0.0
        )

        # Epoch callback (pass single-device params)
        if epoch_callback:
            if use_pmap:
                single_params = jax.tree_util.tree_map(lambda x: x[0], params)
                epoch_results.append(
                    epoch_callback(epoch_idx, single_params, structure, config, rng_key)
                )
            else:
                epoch_results.append(
                    epoch_callback(epoch_idx, params, structure, config, rng_key)
                )
        else:
            epoch_results.append(None)

        if verbose and not use_tqdm:
            print(f"Epoch {epoch_idx + 1}/{total_epochs}, energy: {avg_energy:.4f}")

    progress.close()

    # Extract params from first device (all devices have same params due to pmean)
    if use_pmap:
        params = jax.tree_util.tree_map(lambda x: x[0], params)

    return params, iter_results, epoch_results


# ── evaluation ──────────────────────────────────────────────────────


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
    final_state = run_inference(params, state, clamps, structure)

    # Compute total network energy
    total_energy = jnp.array(0.0)
    for node_name in structure.nodes:
        total_energy = total_energy + jnp.sum(final_state.nodes[node_name].energy)

    avg_energy = total_energy / batch_size
    # Note: In evaluation mode, the output node won't be clamped to a label and doesn't contribute to energy. Evaluation energy is only from internal nodes.
    # Use a proper metric to assess task performance, not energy.

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
    pmap_single_device: bool = False,
) -> Dict[str, float]:
    """
    Evaluate predictive coding network on test data.

    Automatically detects available devices and uses pmap for multi-device
    evaluation, or JIT for single-device.

    Args:
        params: Trained parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation configuration
        rng_key: JAX random key (will be split for each batch)
        pmap_single_device: If True, forces pmap path on single device.

    Returns:
        Dictionary of evaluation metrics {"energy": avg_energy, "accuracy": accuracy}
    """
    from fabricpc.graph.state_initializer import initialize_graph_state

    n_devices = jax.device_count()
    use_pmap = (n_devices > 1) or pmap_single_device

    # Estimate number of batches
    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000

    if use_pmap:
        # ── pmap eval path ──
        epoch_key, rng_key = jax.random.split(rng_key)
        if n_devices > 1:
            device_keys = jax.random.split(epoch_key, n_devices)
        else:
            device_keys = jnp.expand_dims(epoch_key, axis=0)

        replicated_params = replicate_params(params, n_devices)

        batch_keys_per_device = jax.vmap(lambda k: jax.random.split(k, num_batches))(
            device_keys
        )

        def inference_fn(params_obj, sharded_batch, randgen_key):
            batch_size_ = next(iter(sharded_batch.values())).shape[0]
            clamps = {}
            for task_name, task_value in sharded_batch.items():
                if task_name in structure.task_map and task_name == "x":
                    node_name = structure.task_map[task_name]
                    clamps[node_name] = task_value

            state = initialize_graph_state(
                structure, batch_size_, randgen_key, clamps=clamps, params=params_obj
            )
            final_state = run_inference(params_obj, state, clamps, structure)
            return final_state

        pmap_inference = jax.pmap(inference_fn, axis_name="devices")

        total_correct = 0
        total_samples = 0

        for batch_idx, batch_data in enumerate(test_loader):
            batch_key_for_step = batch_keys_per_device[:, batch_idx]
            batch = _convert_batch(batch_data)

            batch_size = next(iter(batch.values())).shape[0]
            if batch_size % n_devices != 0:
                continue

            batch_sharded = shard_batch(batch, n_devices)
            final_states = pmap_inference(
                replicated_params, batch_sharded, batch_key_for_step
            )

            if "y" in structure.task_map:
                y_node = structure.task_map["y"]
                predictions = final_states.nodes[y_node].z_mu
                predictions = predictions.reshape(batch_size, -1)
                targets = batch["y"]

                pred_labels = jnp.argmax(predictions, axis=1)
                true_labels = jnp.argmax(targets, axis=1)
                correct = jnp.sum(pred_labels == true_labels)

                total_correct += int(correct)
                total_samples += batch_size

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {"accuracy": accuracy}

    else:
        # ── JIT eval path ──
        jit_eval_step = jax.jit(lambda p, b, k: eval_step(p, b, structure, k))
        batch_keys = jax.random.split(rng_key, num_batches)

        total_energy = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch_data in enumerate(test_loader):
            batch = _convert_batch(batch_data)

            batch_energy, correct, batch_size = jit_eval_step(
                params, batch, batch_keys[batch_idx]
            )

            total_energy += float(batch_energy) * int(batch_size)
            total_correct += int(correct)
            total_samples += int(batch_size)

        avg_energy = total_energy / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {"energy": avg_energy, "accuracy": accuracy}


def evaluate_transformer(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, float]:
    """
    Evaluate PC Transformer using all available devices.

    Computes accuracy, cross-entropy loss, perplexity, and average energy.
    Uses pmap for device-parallel inference (works on 1 or N devices).

    Args:
        params: Trained parameters
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation configuration
        rng_key: JAX random key

    Returns:
        Dictionary with keys: "accuracy", "cross_entropy", "perplexity", "energy"
    """
    from fabricpc.graph.state_initializer import initialize_graph_state

    n_devices = jax.device_count()

    # Split keys for devices
    epoch_key, rng_key = jax.random.split(rng_key)
    if n_devices > 1:
        device_keys = jax.random.split(epoch_key, n_devices)
    else:
        device_keys = jnp.expand_dims(epoch_key, axis=0)

    # Replicate params across devices
    replicated_params = replicate_params(params, n_devices)

    # Handle loader length safely
    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000

    batch_keys_per_device = jax.vmap(lambda k: jax.random.split(k, num_batches))(
        device_keys
    )

    # pmap'd inference function
    def inference_fn(params_obj, sharded_batch, randgen_key):
        batch_size_ = next(iter(sharded_batch.values())).shape[0]
        clamps = {}
        for task_name, task_value in sharded_batch.items():
            if task_name in structure.task_map and task_name == "x":
                node_name = structure.task_map[task_name]
                clamps[node_name] = task_value

        state = initialize_graph_state(
            structure, batch_size_, randgen_key, clamps=clamps, params=params_obj
        )
        final_state = run_inference(params_obj, state, clamps, structure)
        return final_state

    pmap_inference = jax.pmap(inference_fn, axis_name="devices")

    total_correct = 0
    total_samples = 0
    total_ce = 0.0
    total_tokens = 0
    total_energy = 0.0

    for batch_idx, batch_data in enumerate(test_loader):
        batch_key_for_step = batch_keys_per_device[:, batch_idx]

        batch = _convert_batch(batch_data)

        batch_size = next(iter(batch.values())).shape[0]
        if batch_size % n_devices != 0:
            continue

        batch_sharded = shard_batch(batch, n_devices)
        final_states = pmap_inference(
            replicated_params, batch_sharded, batch_key_for_step
        )

        # Calculate energy per device (internal + external/output error)
        def get_device_energy(fs, batch_y):
            e = 0.0
            # Internal energy
            for node_name in structure.nodes:
                if structure.nodes[node_name].node_info.in_degree > 0:
                    e += jnp.sum(fs.nodes[node_name].energy)

            # External energy (Output prediction error)
            if "y" in structure.task_map:
                y_node = structure.task_map["y"]
                pred = fs.nodes[y_node].z_latent

                # Handle shapes: batch_y might be indices or one-hot
                if batch_y.ndim == pred.ndim:
                    error = pred - batch_y
                    e += jnp.sum(error**2)
                elif batch_y.ndim == pred.ndim - 1:
                    tgt_oh = jax.nn.one_hot(batch_y, pred.shape[-1])
                    error = pred - tgt_oh
                    e += jnp.sum(error**2)

            return e

        device_energies = jax.vmap(get_device_energy)(final_states, batch_sharded["y"])
        total_energy += float(jnp.sum(device_energies))
        total_samples += batch_size

        if "y" in structure.task_map:
            y_node = structure.task_map["y"]
            preds = final_states.nodes[y_node].z_latent

            # Reshape to (total_batch, ...)
            preds_flat = preds.reshape(batch_size, *preds.shape[2:])
            targets = batch["y"]

            # --- Accuracy ---
            pred_labels = jnp.argmax(preds_flat, axis=-1)

            if targets.ndim == preds_flat.ndim:
                true_labels = jnp.argmax(targets, axis=-1)
            else:
                true_labels = targets

            total_correct += int(jnp.sum(pred_labels == true_labels))

            # Stable CE computation
            softmax_preds = jax.nn.softmax(preds_flat, axis=-1)

            if targets.ndim == preds_flat.ndim:
                batch_ce = -jnp.sum(
                    targets * jnp.log(jnp.clip(softmax_preds, 1e-10, 1.0))
                )
                target_tokens = targets.shape[0] * (
                    targets.shape[1] if targets.ndim > 1 else 1
                )
            else:
                targets_one_hot = jax.nn.one_hot(targets, preds_flat.shape[-1])
                batch_ce = -jnp.sum(
                    targets_one_hot * jnp.log(jnp.clip(softmax_preds, 1e-10, 1.0))
                )
                target_tokens = targets.size

            total_ce += float(batch_ce)
            total_tokens += target_tokens

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    mean_ce = total_ce / total_tokens if total_tokens > 0 else 0.0
    perplexity = float(jnp.exp(mean_ce)) if mean_ce > 0 else float("inf")
    avg_energy = total_energy / total_samples if total_samples > 0 else 0.0

    return {
        "accuracy": accuracy,
        "cross_entropy": mean_ce,
        "perplexity": perplexity,
        "energy": avg_energy,
    }
