"""
Multi-GPU training utilities using JAX pmap.

This module provides data-parallel training across multiple GPUs using pmap,
which replicates the computation across devices and averages gradients.

  | Map-reduce                                               | Inside pmap (pmean) | Outside pmap (jnp.mean) |
  |----------------------------------------------------------|---------------------|-------------------------|
  | Do devices need synchronized value for next computation? | ✓                   |                         |
  | Is it just for logging/metrics?                          |                     | ✓                       |
  | Does it affect model state (params, opt_state)?          | ✓                   |                         |
  | Is it a final output we just want to aggregate?          |                     | ✓                       |

  In train_step_pmap, only grads needs pmean because it feeds into the optimizer. The energy can be gathered and averaged on the host since it's only used for logging.
  Duplicating the optimizer across devices costs memory but is faster for training since each device can update its own copy in parallel.
"""

from typing import Dict, Tuple, Any, cast, Iterable
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import run_inference
from fabricpc.graph.state_initializer import initialize_graph_state
from fabricpc.training.train import get_graph_param_gradient


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


def train_step_pmap(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
    infer_steps: int,
    eta_infer: float = 0.1,
) -> Tuple[GraphParams, optax.OptState, jnp.ndarray, GraphState]:
    """
    Training step parallelized across devices using pmap.

    Uses local Hebbian learning (same as single-GPU training) via shared
    get_graph_param_gradient function, then averages gradients across devices.

    Args:
        params: Replicated parameters (has device axis)
        opt_state: Replicated optimizer state (has device axis)
        batch: Sharded batch (has device axis)
        rng_key: JAX random key for this device
        structure: Graph structure
        optimizer: Optax optimizer
        infer_steps: Number of inference steps
        eta_infer: Inference learning rate

    Returns:
        Tuple of (updated_params, updated_opt_state, energy_per_device, final_state)
    """
    # Compute gradients using local Hebbian learning (shared code with single-GPU)
    grads, energy, final_state = get_graph_param_gradient(
        params, batch, structure, rng_key, infer_steps, eta_infer
    )

    # Normalize energy by batch size for this shard
    batch_size = next(iter(batch.values())).shape[0]
    energy = energy / batch_size

    # Average gradients across all devices (data parallelism)
    grads = jax.lax.pmean(grads, axis_name="devices")

    # Update parameters (each device now has same gradients)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, energy, final_state


# Create pmap version of train_step
# Note: We can't use static_broadcasted_argnums because GraphStructure contains dicts
# Instead, we'll create a closure that captures the static arguments
def create_pmap_train_step(
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    infer_steps: int,
    eta_infer: float,
):
    """
    Create a pmap'ed training step with static arguments captured in closure.

    Args:
        structure: Graph structure (static)
        optimizer: Optimizer (static)
        infer_steps: Number of inference steps (static)
        eta_infer: Inference learning rate (static)

    Returns:
        Pmap'ed training step function
    """

    def step_fn(params, opt_state, batch, rng_key):
        return train_step_pmap(
            params,
            opt_state,
            batch,
            structure,
            rng_key,
            optimizer,
            infer_steps,
            eta_infer,
        )

    return jax.pmap(step_fn, axis_name="devices")


def train_pcn_multi_gpu(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    config: dict,
    rng_key: jax.Array,
    verbose: bool = True,
) -> GraphParams:
    """
    Train PCN using all available GPUs with data parallelism.

    Args:
        params: Initial parameters (single device)
        structure: Graph structure
        train_loader: Data loader
        config: Training configuration
        rng_key: JAX random key
        verbose: Whether to print progress

    Returns:
        Trained parameters (single device)

    Example:
        >>> params = initialize_params(structure, jax.random.PRNGKey(0))
        >>> trained = train_pcn_multi_gpu(params, structure, train_loader, config)
    """
    from fabricpc.training.optimizers import create_optimizer

    # Get available devices
    n_devices = jax.device_count()
    if verbose:
        print(f"Training on {n_devices} device(s): {jax.devices()}")

    # Create shard even for single gpu to ensure consistency. Don't fallback to single-gpu method.
    if n_devices == 1:
        if verbose:
            print(
                "Only 1 device available, using multi-gpu training function with single device."
            )

    # Create optimizer
    optimizer = create_optimizer(config.get("optimizer", {"type": "adam", "lr": 1e-3}))
    opt_state = optimizer.init(params)

    # Replicate params and optimizer state across devices
    params = replicate_params(params, n_devices)
    opt_state = replicate_opt_state(opt_state, n_devices)

    # Get training hyperparameters
    infer_steps = config.get("infer_steps", 20)
    eta_infer = config.get("eta_infer", 0.1)
    num_epochs = config.get("num_epochs", 10)

    # Create pmap'ed training step (uses local Hebbian learning like single-GPU)
    pmap_train_step = create_pmap_train_step(
        structure, optimizer, infer_steps, eta_infer
    )

    # Training loop
    for epoch in range(num_epochs):
        epoch_energies = []

        # Split keys for devices
        epoch_key, rng_key = jax.random.split(rng_key)
        if n_devices > 1:
            device_keys = jax.random.split(epoch_key, n_devices)
        else:
            # Wrap key in array without splitting (preserves exact key for single-GPU equivalence)
            device_keys = jnp.expand_dims(epoch_key, axis=0)

        # Estimate number of batches for key splitting
        num_batches = len(train_loader)

        # Create keys for all batches (split on each device)
        batch_keys_per_device = jax.vmap(lambda k: jax.random.split(k, num_batches))(
            device_keys
        )

        for batch_idx, batch_data in enumerate(train_loader):
            batch_key_for_step = batch_keys_per_device[:, batch_idx]

            # Convert batch to JAX format
            if isinstance(batch_data, (list, tuple)):
                batch = {
                    "x": jnp.array(batch_data[0]),
                    "y": jnp.array(batch_data[1]),
                }
            elif isinstance(batch_data, dict):
                batch = {k: jnp.array(v) for k, v in batch_data.items()}
            else:
                raise ValueError(f"unsupported batch format: {type(batch_data)}")

            # Shard batch across devices
            try:
                batch_sharded = shard_batch(batch, n_devices)
            except ValueError as e:
                if verbose and batch_idx == 0:
                    print(
                        f"Warning: Skipping batch (size not divisible by {n_devices}): {e}"
                    )
                continue

            # Training step (parallelized across devices)
            params, opt_state, energies, final_states = pmap_train_step(
                params, opt_state, batch_sharded, batch_key_for_step
            )

            # Average energy from all devices
            avg_energy = unshard_energies(energies)
            epoch_energies.append(avg_energy)

        # Compute average energy for epoch
        avg_energy = (
            sum(epoch_energies) / len(epoch_energies) if epoch_energies else 0.0
        )

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Energy: {avg_energy:.4f}")

    # Extract params from first device (all devices have same params due to pmean)
    params = jax.tree_util.tree_map(lambda x: x[0], params)

    return params


def evaluate_transformer_multi_gpu(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, float]:
    """
    Evaluate PC Transformer using all available GPUs and compute accuracy, cross-entropy loss,
    perplexity, and average energy.
    """
    n_devices = jax.device_count()

    # Split keys for devices
    epoch_key, rng_key = jax.random.split(rng_key)
    if n_devices > 1:
        device_keys = jax.random.split(epoch_key, n_devices)
    else:
        device_keys = jnp.expand_dims(epoch_key, axis=0)

    # Replicate params across devices
    params = replicate_params(params, n_devices)

    infer_steps = config.get("infer_steps", 20)
    eta_infer = config.get("eta_infer", 0.1)

    # Handle loader length safely
    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000  # Fallback

    batch_keys_per_device = jax.vmap(lambda k: jax.random.split(k, num_batches))(
        device_keys
    )

    # pmap'ed inference function
    def inference_fn(
        params_obj: GraphParams,
        sharded_batch: Dict[str, jnp.ndarray],
        randgen_key: jax.Array,
    ):
        batch_size_ = next(iter(sharded_batch.values())).shape[0]
        clamps = {}
        for task_name, task_value in sharded_batch.items():
            if task_name in structure.task_map and task_name == "x":
                node_name = structure.task_map[task_name]
                clamps[node_name] = task_value

        state = initialize_graph_state(
            structure, batch_size_, randgen_key, clamps=clamps, params=params_obj
        )

        final_state = run_inference(
            params_obj, state, clamps, structure, infer_steps, eta_infer
        )
        return final_state

    pmap_inference = jax.pmap(inference_fn, axis_name="devices")

    total_correct = 0
    total_samples = 0
    total_ce = 0.0
    total_tokens = 0
    total_energy = 0.0

    for batch_idx, batch_data in enumerate(test_loader):
        batch_key_for_step = batch_keys_per_device[:, batch_idx]

        # Convert batch to dict of JAX arrays
        if isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        else:
            batch = {k: jnp.array(v) for k, v in batch_data.items()}

        batch_size = next(iter(batch.values())).shape[0]
        # Skip last incomplete batch if not divisible by n_devices to avoid shape mismatch in pmap
        if batch_size % n_devices != 0:
            continue

        # Shard batch
        batch_sharded = shard_batch(batch, n_devices)

        # Run inference on all GPUs
        final_states = pmap_inference(params, batch_sharded, batch_key_for_step)

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
            # z_latent: (n_devices, device_batch, seq_len, vocab_size)
            preds = final_states.nodes[y_node].z_latent

            # Reshape to (total_batch, ...)
            preds_flat = preds.reshape(batch_size, *preds.shape[2:])
            targets = batch["y"]

            # --- Accuracy ---
            pred_labels = jnp.argmax(preds_flat, axis=-1)

            if targets.ndim == preds_flat.ndim:
                # One-hot targets
                true_labels = jnp.argmax(targets, axis=-1)
            else:
                # Integer targets
                true_labels = targets

            total_correct += int(jnp.sum(pred_labels == true_labels))

            # Stable CE computation
            softmax_preds = jax.nn.softmax(preds_flat, axis=-1)

            if targets.ndim == preds_flat.ndim:
                # One-hot
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


def evaluate_pcn_multi_gpu(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,  # Added: RNG key for reproducibility
) -> Dict[str, float]:
    """
    Evaluate PCN using all available GPUs.

    Args:
        params: Trained parameters (single device)
        structure: Graph structure
        test_loader: Test data loader
        config: Evaluation configuration
        rng_key: JAX random key for reproducibility

    Returns:
        Dictionary of metrics
    """
    n_devices = jax.device_count()

    # Split keys for devices
    epoch_key, rng_key = jax.random.split(rng_key)
    if n_devices > 1:
        device_keys = jax.random.split(epoch_key, n_devices)
    else:
        # Create shard even for single gpu to ensure consistency. Don't fallback to single-gpu method.
        # Wrap key in array without splitting (preserves exact key for single-GPU equivalence)
        device_keys = jnp.expand_dims(epoch_key, axis=0)

    # Replicate params across devices
    params = replicate_params(params, n_devices)

    infer_steps = config.get("infer_steps", 20)
    eta_infer = config.get("eta_infer", 0.1)
    # Estimate number of batches for key splitting
    num_batches = len(test_loader)

    # Create keys for all batches (split on each device)
    batch_keys_per_device = jax.vmap(lambda k: jax.random.split(k, num_batches))(
        device_keys
    )

    # Create pmap'ed inference function
    def inference_fn(
        params_obj: GraphParams,
        sharded_batch: Dict[str, jnp.ndarray],
        randgen_key: jax.Array,
    ) -> GraphState:
        batch_size_ = next(iter(sharded_batch.values())).shape[0]
        clamps = {}
        for task_name, task_value in sharded_batch.items():
            if task_name in structure.task_map and task_name == "x":
                node_name = structure.task_map[task_name]
                clamps[node_name] = task_value

        state = initialize_graph_state(
            structure,
            batch_size_,
            randgen_key,
            clamps=clamps,
            params=params_obj,
        )
        final_state = run_inference(
            params_obj, state, clamps, structure, infer_steps, eta_infer
        )
        return final_state

    pmap_inference = jax.pmap(inference_fn, axis_name="devices")

    batch_energies = []  # energy of the network
    batch_output_loss = []  # loss of the output node
    total_correct = 0
    total_samples = 0

    for batch_idx, batch_data in enumerate(test_loader):
        batch_key_for_step = batch_keys_per_device[:, batch_idx]
        # Convert batch
        if isinstance(batch_data, (list, tuple)):
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        else:
            batch = {k: jnp.array(v) for k, v in batch_data.items()}

        batch_size = next(iter(batch.values())).shape[0]

        # Skip if batch size not divisible by n_devices
        if batch_size % n_devices != 0:
            continue

        # Shard batch
        batch_sharded = shard_batch(batch, n_devices)

        # Run inference
        final_states = pmap_inference(params, batch_sharded, batch_key_for_step)

        # Gather results from all devices
        # final_states is a GraphState with .nodes dict, arrays have shape (n_devices, batch_per_device, ...)
        if "y" in structure.task_map:
            y_node = structure.task_map["y"]
            # Access via .nodes attribute (GraphState is a NamedTuple, not a dict)
            predictions = final_states.nodes[
                y_node
            ].z_mu  # (n_devices, batch_per_device, *shape)
            predictions = predictions.reshape(batch_size, -1)
            targets = batch["y"]

            pred_labels = jnp.argmax(predictions, axis=1)
            true_labels = jnp.argmax(targets, axis=1)
            correct = jnp.sum(pred_labels == true_labels)

            total_correct += int(correct)
            total_samples += batch_size

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return {"accuracy": accuracy}
