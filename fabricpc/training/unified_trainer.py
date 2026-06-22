"""
Unified trainer for FabricPC — predictive coding (PC) and backpropagation.

This single, self-contained module trains FabricPC graphs with either learning
rule, selected by the ``algorithm`` parameter (``"pc"`` or ``"backprop"``):

    from fabricpc.training.unified_trainer import train, evaluate
    train(params, structure, loader, optimizer, config, key, algorithm="pc")
    evaluate(params, structure, loader, config, key, algorithm="backprop")

``algorithm`` is an explicit signature parameter (not a ``config`` key) so the
call site states it plainly.

It is a *genuine* unification (not three files copied together): the per-batch
step is decomposed so that **all shared scaffolding lives in one method each**
and the two algorithms differ in exactly three places, all funnelled through
``_forward`` / ``compute_gradients`` / ``_eval_step``:

    sub-step          PC                                  backprop
    ----------------  ----------------------------------  -------------------------------
    1 batch->dict     convert_batch                       convert_batch          (shared)
    2 build clamps    build_clamps(clamp_target=True)     build_clamps(clamp_target=False)
    3 produce state   initialize_graph_state+run_inference initialize_graph_state (feedforward)
    4 objective       _evaluation_metric (energy)         _evaluation_metric (loss)  (shared)
    5 gradients       compute_local_weight_gradients      jax.value_and_grad(objective)
    6 apply update    apply_update                        apply_update           (shared)

The metric/objective (energy or loss, or a user metric) is the single
``_evaluation_metric`` helper, selected by one ``metric_name`` (a built-in like
``"energy"``/``"cross_entropy"``/``"mse"``, or any name for a user ``metric_fn``).
The gradient math is reused **verbatim**; only the surrounding scaffolding
(clamping, state init, the metric, optimizer update, the epoch/batch loop,
sharding, evaluation) is unified into common methods.

Autoregressive (next-token) training is an *orthogonal* mode enabled by
``autoregressive=True`` — it works with *either* algorithm and only adds (a) a
causal-mask clamp in :func:`build_clamps` and (b) loss/perplexity reporting in
:func:`evaluate`. It is NOT a third algorithm. Unlike the legacy AR step, the
unified AR-PC step does not compute an unused cross-entropy inside the jitted
step (the training metric stays the PC energy), so it is cleaner and run-to-run
deterministic.

Scope: PC + backprop, standard or autoregressive."""

from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import math
import warnings

import jax
import jax.numpy as jnp
import optax
from tqdm.auto import tqdm as _tqdm_cls

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.core.inference import run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.graph_initialization.state_initializer import (
    initialize_graph_state,
    FeedforwardStateInit,
)

ALGORITHMS = (
    "pc",
    "backprop",
)  # autoregressive is an orthogonal flag, not an algorithm


# ── shared, algorithm-agnostic helpers ──────────────────────────────


def convert_batch(batch_data) -> Dict[str, jnp.ndarray]:
    """Normalize a loader batch (dict, or ``(x, y)`` tuple/list) to a dict of JAX arrays."""
    if isinstance(batch_data, (list, tuple)):
        return {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
    elif isinstance(batch_data, dict):
        return {k: jnp.array(v) for k, v in batch_data.items()}
    raise ValueError(f"Unsupported batch format: {type(batch_data)}")


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Lower-triangular causal mask of shape ``(seq_len, seq_len)``: ``1`` where ``j <= i``.

    Ensures position ``i`` may only attend to positions ``0..i`` in autoregressive
    (next-token) training. Matches ``train_autoregressive.create_causal_mask``.
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))


def _require_causal_mask_node(structure: GraphStructure) -> None:
    """Single source for the causal-mask prerequisite: a ``'causal_mask'`` task node.

    Used both by :func:`_validate_autoregressive` (fail-fast, before the JIT'd loop)
    and by :func:`build_clamps` (self-guard, since it is also called directly).
    """
    if "causal_mask" not in structure.task_map:
        raise ValueError(
            "Causal masking requires a 'causal_mask' entry in task_map; build the "
            "graph with a causal-mask node or disable it (use_causal_mask=False)."
        )


def build_clamps(
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    *,
    clamp_target: bool,
    causal_mask: bool = False,
) -> Dict[str, jnp.ndarray]:
    """Map task keys to clamped node states via ``structure.task_map``.

    ``clamp_target=True`` clamps every task key present in the batch (PC training
    clamps both input ``x`` and target ``y`` — the target must sit in-state to
    drive the local prediction errors). ``clamp_target=False`` clamps only ``x``
    (backprop and all evaluation: the output runs free).

    ``causal_mask=True`` additionally injects a generated causal mask into the
    ``"causal_mask"`` task node — the *only* graph change autoregressive mode
    makes. The mask is derived from the input sequence length and broadcast to
    ``(batch, 1, seq, seq)`` for attention scores (matching the legacy AR step).
    """
    clamps: Dict[str, jnp.ndarray] = {}
    for task_name, task_value in batch.items():
        if task_name in structure.task_map and (clamp_target or task_name == "x"):
            clamps[structure.task_map[task_name]] = task_value
    if causal_mask:
        _require_causal_mask_node(structure)
        batch_size, seq_len = batch["x"].shape[0], batch["x"].shape[1]
        mask = create_causal_mask(seq_len)[None, None, :, :]
        mask = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))
        clamps[structure.task_map["causal_mask"]] = mask
    return clamps


def validate_feedforward_init(structure: GraphStructure) -> None:
    """Backprop requires a feedforward state initializer (single forward pass, no inference)."""
    init = structure.config["graph_state_initializer"]
    if not isinstance(init, FeedforwardStateInit):
        raise ValueError(
            f"GraphState initializer must be FeedforwardStateInit for backprop training, "
            f"got {type(init).__name__}"
        )


def _default_metric_name(algo: str, autoregressive: bool = False) -> str:
    """Built-in default metric. Autoregressive reports the output cross-entropy
    (so perplexity is meaningful) for *either* algorithm; otherwise PC -> graph
    energy, backprop -> cross-entropy. (Training never passes ``autoregressive``,
    so AR-PC still trains on energy — only evaluation flips the AR default.)"""
    if autoregressive:
        return "cross_entropy"
    return "energy" if algo == "pc" else "cross_entropy"


def _metric_energy(state, batch, structure, internal_only):
    """Per-sample total graph energy (the nodes' own energy functionals, e.g. GaussianEnergy).

    ``internal_only=True`` sums in_degree>0 nodes (training objective); ``False`` sums all
    nodes (evaluation).
    """
    batch_size = next(iter(batch.values())).shape[0]
    energy = jnp.array(0.0)
    for name in structure.nodes:
        if internal_only and structure.nodes[name].node_info.in_degree == 0:
            continue
        energy = energy + jnp.sum(state.nodes[name].energy)
    return energy / batch_size


def _metric_cross_entropy(state, batch, structure, _internal_only):
    # _internal_only is unused (only _metric_energy honors it) — kept for the
    # uniform KNOWN_METRICS dispatch signature.
    pred = state.nodes[structure.task_map["y"]].z_mu
    return -jnp.mean(jnp.sum(batch["y"] * jnp.log(pred + 1e-10), axis=-1))


def _metric_mse(state, batch, structure, _internal_only):
    # _internal_only unused (see _metric_cross_entropy).
    pred = state.nodes[structure.task_map["y"]].z_mu
    return jnp.mean((pred - batch["y"]) ** 2)


# Built-in metrics, selected by name. Signature: (state, batch, structure, internal_only) -> scalar.
KNOWN_METRICS = {
    "energy": _metric_energy,
    "cross_entropy": _metric_cross_entropy,
    "mse": _metric_mse,
}


def _evaluation_metric(
    state: GraphState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    *,
    algo: str,
    metric_fn: Optional[Callable] = None,
    metric_name: Optional[str] = None,
    internal_only: bool = True,
    autoregressive: bool = False,
) -> jnp.ndarray:
    """The single scalar each algorithm optimizes/reports — one metric, one name.

    A SINGLE ``metric_name`` selects the metric:
      - if ``metric_fn`` is given, that user callable ``(state, batch, structure) -> scalar``
        is used and ``metric_name`` is merely its label (any string, default ``"Untitled"``);
      - else ``metric_name`` names a built-in in :data:`KNOWN_METRICS`
        (``"energy"``, ``"cross_entropy"``, ``"mse"``);
      - else (``metric_name is None``) the algorithm default from
        :func:`_default_metric_name` (PC -> ``"energy"``, backprop -> ``"cross_entropy"``,
        or ``"cross_entropy"`` for autoregressive evaluation).

    ``autoregressive`` only flips the *default* (it is passed by evaluation, never
    by training, so AR-PC still trains on energy). Gradient descent is performed
    w.r.t. exactly this quantity (backprop differentiates it; for PC it is the
    energy the local rule descends).
    """
    if metric_fn is not None:
        return metric_fn(state, batch, structure)
    name = metric_name or _default_metric_name(algo, autoregressive)
    if name not in KNOWN_METRICS:
        raise ValueError(
            f"Unknown metric_name {name!r}; known built-ins: {sorted(KNOWN_METRICS)}, "
            f"or pass a metric_fn for a custom metric."
        )
    return KNOWN_METRICS[name](state, batch, structure, internal_only)


def _metric_label(algo: str, metric_fn, metric_name: Optional[str]) -> str:
    """The metric's *identity* (training display label): explicit ``metric_name`` wins;
    else ``"Untitled"`` for a custom ``metric_fn``; else the algorithm's built-in default
    (PC -> ``"energy"``, backprop -> ``"cross_entropy"``)."""
    if metric_name:
        return metric_name
    if metric_fn is not None:
        return "Untitled"
    return _default_metric_name(algo)


def _result_key(
    algo: str, metric_fn, metric_name: Optional[str], autoregressive: bool
) -> str:
    """The eval result-dict key, following the legacy evaluators' *contract* — the
    generic objective-family name, which is NOT the metric's identity (hence this is
    distinct from :func:`_metric_label`): ``evaluate_pcn`` -> ``"energy"``, while
    ``evaluate_backprop`` and ``evaluate_autoregressive`` -> ``"loss"``. So backprop's
    key is ``"loss"`` even though its metric identity is ``"cross_entropy"``, and AR is
    ``"loss"`` for either algorithm. An explicit ``metric_name`` (or custom
    ``metric_fn``) overrides it."""
    if metric_name:
        return metric_name
    if metric_fn is not None:
        return "Untitled"
    if autoregressive:
        return "loss"
    return "energy" if algo == "pc" else "loss"


def _accuracy(
    predictions: jnp.ndarray, targets: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Classification accuracy as ``(num_correct, num_predictions)``; handles one-hot or index targets."""
    pred_labels = jnp.argmax(predictions, axis=-1)
    if targets.ndim > 1 and targets.shape[-1] > 1:
        true_labels = jnp.argmax(targets, axis=-1)
    else:
        true_labels = targets
    correct = jnp.sum(pred_labels == true_labels)
    return correct, jnp.prod(jnp.array(pred_labels.shape))


def _validate_algo(algo: str, structure: GraphStructure) -> None:
    """Raise if ``algo`` is unknown or its required arguments are missing.

    Each algorithm has prerequisites encoded in the graph: PC needs an inference
    algorithm; backprop needs a feedforward state initializer. Autoregressive is
    an orthogonal mode (the ``autoregressive`` flag), not an algorithm; ``"auto"``
    is rejected so callers reach for that flag instead.
    """
    if algo == "pc":
        if "inference" not in structure.config:
            raise ValueError(
                "algorithm='pc' requires an inference algorithm: build the graph "
                "with graph(..., inference=...)."
            )
    elif algo == "backprop":
        validate_feedforward_init(structure)
    elif algo == "auto":
        raise NotImplementedError(
            "algorithm='auto' is not an algorithm; autoregressive is an orthogonal "
            "mode — pass algorithm='pc'|'backprop' with autoregressive=True."
        )
    else:
        raise ValueError(f"Unknown algorithm {algo!r}; choose from {ALGORITHMS}")


def _validate_autoregressive(structure: GraphStructure, use_causal_mask: bool) -> None:
    """Fail fast (before the JIT'd loop) if AR causal masking lacks its task node."""
    if use_causal_mask:
        _require_causal_mask_node(structure)


def apply_update(
    params: GraphParams,
    opt_state: optax.OptState,
    grads: GraphParams,
    optimizer: optax.GradientTransformation,
) -> Tuple[GraphParams, optax.OptState]:
    """Shared optax update step (identical for PC and backprop)."""
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))
    return params, opt_state


# ── pmap / device utilities ─────────────────────────────────────────


def replicate_params(params: GraphParams, n_devices: int) -> GraphParams:
    """Replicate a pytree across devices (adds a leading device axis)."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), params)


def replicate_opt_state(opt_state: optax.OptState, n_devices: int) -> optax.OptState:
    """Replicate optimizer state across devices."""
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_devices), opt_state)


def shard_batch(
    batch: Dict[str, jnp.ndarray], n_devices: int
) -> Dict[str, jnp.ndarray]:
    """Reshape a batch from ``(B, ...)`` to ``(n_devices, B//n_devices, ...)`` for pmap."""

    def shard_array(x):
        if x.shape[0] % n_devices != 0:
            raise ValueError(
                f"Batch size {x.shape[0]} must be divisible by number of devices {n_devices}"
            )
        return x.reshape(n_devices, x.shape[0] // n_devices, *x.shape[1:])

    return jax.tree_util.tree_map(shard_array, batch)


def unshard_energies(energies: jnp.ndarray) -> float:
    """Average a per-device metric to a scalar."""
    return float(jnp.mean(energies))


# ── the one place the algorithms differ ─────────────────────────────


def _forward(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    *,
    algo: str,
    clamp_target: bool,
    causal_mask: bool = False,
) -> GraphState:
    """Produce the graph state for one batch — the sole inference-vs-feedforward branch.

    PC clamps (per ``clamp_target``), initializes, then relaxes to equilibrium via
    ``run_inference``. Backprop clamps the input only and takes a single forward
    pass (no inference). ``causal_mask`` adds the autoregressive mask clamp. The
    feedforward-init check is hoisted to the callers (``run_training_loop`` /
    ``evaluate``) so it runs once, not per batch.
    """
    batch_size = next(iter(batch.values())).shape[0]
    clamps = build_clamps(
        batch, structure, clamp_target=clamp_target, causal_mask=causal_mask
    )
    state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )
    if algo == "pc":
        state = run_inference(params, state, clamps, structure)
    return state


def compute_gradients(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    *,
    algo: str,
    metric_fn: Optional[Callable] = None,
    metric_name: Optional[str] = None,
    autoregressive: bool = False,
    use_causal_mask: bool,
) -> Tuple[GraphParams, jnp.ndarray]:
    """Return ``(grads, scalar_metric)`` for one batch.

    PC: clamp x+y, relax, local-Hebbian grads, metric = per-sample internal energy.
    Backprop: clamp x, single forward, autodiff grads of the output loss, metric = loss.
    The metric (and the quantity backprop differentiates) is ``_evaluation_metric``;
    the shared forward+metric step is factored into ``forward_metric`` so the two
    branches differ only where they genuinely must: clamping, the ``internal_only``
    flag, and HOW the gradient is taken (local-Hebbian from the state vs. autodiff
    of the metric). Autoregressive mode only adds the causal-mask clamp; the
    gradient rule and the reported training metric are unchanged (AR-PC still trains
    on energy — and, unlike the legacy AR step, computes no unused cross-entropy).
    """
    causal = autoregressive and use_causal_mask

    def forward_metric(p, *, clamp_target, internal_only):
        state = _forward(
            p,
            batch,
            structure,
            rng_key,
            algo=algo,
            clamp_target=clamp_target,
            causal_mask=causal,
        )
        metric = _evaluation_metric(
            state,
            batch,
            structure,
            algo=algo,
            metric_fn=metric_fn,
            metric_name=metric_name,
            internal_only=internal_only,
        )
        return metric, state

    if algo == "pc":
        metric, state = forward_metric(params, clamp_target=True, internal_only=True)
        grads = compute_local_weight_gradients(params, state, structure)
        return grads, metric
    if algo == "backprop":
        # gradient IS d(metric)/d(params); [0] selects the scalar to differentiate.
        loss, grads = jax.value_and_grad(
            lambda p: forward_metric(p, clamp_target=False, internal_only=False)[0]
        )(params)
        return grads, loss
    raise ValueError(f"Unknown algorithm {algo!r}; choose from {ALGORITHMS}")


def _eval_step(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
    *,
    algo: str,
    metric_fn: Optional[Callable],
    metric_name: Optional[str],
    autoregressive: bool,
    use_causal_mask: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One evaluation step: ``(objective, obj_weight, num_correct, num_predictions)``.

    Output runs free for both algorithms (clamp x only). The scalar objective is
    PC: per-sample total energy; backprop: output loss. Autoregressive evaluation
    reports the output cross-entropy for *either* algorithm (so perplexity is
    meaningful) and adds the causal-mask clamp.

    ``obj_weight`` is the objective's own denominator, kept separate from the
    accuracy ``count`` and chosen to reproduce each legacy evaluator's reduction
    EXACTLY (so the dataset average agrees even on ragged batches):
      - non-AR PC energy -> ``batch_size`` (per-sample; ``evaluate_pcn``);
      - non-AR backprop loss -> ``count`` (per-token; ``evaluate_backprop``);
      - autoregressive loss -> ``1`` (simple per-batch mean ``total_loss /
        num_batches``; ``evaluate_autoregressive`` / ``..._backprop_autoregressive``).
    For classification (``count == batch``) the first two coincide.
    """
    causal = autoregressive and use_causal_mask
    batch_size = next(iter(batch.values())).shape[0]
    state = _forward(
        params,
        batch,
        structure,
        rng_key,
        algo=algo,
        clamp_target=False,
        causal_mask=causal,
    )
    # AR resolves to cross-entropy via _default_metric_name; non-AR uses the algo default.
    objective = _evaluation_metric(
        state,
        batch,
        structure,
        algo=algo,
        metric_fn=metric_fn,
        metric_name=metric_name,
        internal_only=False,
        autoregressive=autoregressive,
    )
    resolved = metric_name or _default_metric_name(algo, autoregressive)
    objective_is_energy = metric_fn is None and resolved == "energy"
    if "y" in structure.task_map:
        correct, count = _accuracy(
            state.nodes[structure.task_map["y"]].z_mu, batch["y"]
        )
    else:  # no target task: degrade gracefully like the legacy eval_step
        correct, count = jnp.array(0), batch_size
    # obj_weight matches each legacy reducer (see docstring): AR per-batch mean,
    # energy per-sample, else per-token.
    if autoregressive:
        obj_weight = 1
    elif objective_is_energy:
        obj_weight = batch_size
    else:
        obj_weight = count
    return objective, obj_weight, correct, count


# ── per-batch train steps (single device + pmap) ────────────────────


def _make_step(
    structure,
    optimizer,
    algo,
    metric_fn,
    metric_name,
    *,
    pmean,
    autoregressive,
    use_causal_mask,
):
    """Build the per-batch train step, closing over the static config.

    ``pmean=True`` averages gradients across devices and returns a ``pmap``'d step
    (data-parallel); ``pmean=False`` returns a single-device ``jit`` step. The
    per-batch logic is otherwise identical for both, for PC and backprop, standard
    or autoregressive — so there is exactly one step body, not two.
    """

    def step(params, opt_state, batch, rng_key):
        grads, metric = compute_gradients(
            params,
            batch,
            structure,
            rng_key,
            algo=algo,
            metric_fn=metric_fn,
            metric_name=metric_name,
            autoregressive=autoregressive,
            use_causal_mask=use_causal_mask,
        )
        if pmean:
            grads = jax.lax.pmean(grads, axis_name="devices")
        params, opt_state = apply_update(params, opt_state, grads, optimizer)
        return params, opt_state, metric

    return jax.pmap(step, axis_name="devices") if pmean else jax.jit(step)


# ── the one shared epoch/batch driver (used by both algorithms) ──────


def run_training_loop(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    *,
    algo: str,
    metric_fn: Optional[Callable],
    metric_name: Optional[str],
    autoregressive: bool,
    use_causal_mask: bool,
    verbose: bool,
    use_tqdm: bool,
    epoch_callback: Optional[Callable],
    iter_callback: Optional[Callable],
    pmap_single_device: bool,
) -> Tuple[GraphParams, List[Any], List[Any]]:
    """Single device-aware loop shared by PC and backprop (and data-parallel for both)."""
    n_devices = jax.device_count()
    use_pmap = (n_devices > 1) or pmap_single_device
    if verbose:
        print(f"Training on {n_devices} device(s): {jax.devices()}")

    opt_state = optimizer.init(params)
    num_epochs = config.get("num_epochs", 10)
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)

    if use_pmap:
        params = replicate_params(params, n_devices)
        opt_state = replicate_opt_state(opt_state, n_devices)
    step_fn = _make_step(
        structure,
        optimizer,
        algo,
        metric_fn,
        metric_name,
        pmean=use_pmap,
        autoregressive=autoregressive,
        use_causal_mask=use_causal_mask,
    )

    num_batches = len(train_loader)
    total_batches = sum(
        (
            round(frac * num_batches)
            if (e == total_epochs - 1 and frac > 0)
            else num_batches
        )
        for e in range(total_epochs)
    )
    progress = _tqdm_cls(total=total_batches, disable=not use_tqdm, leave=True)
    label = _metric_label(algo, metric_fn, metric_name)
    _shard_warned = False

    iter_results: List[Any] = []
    epoch_results: List[Any] = []
    for epoch_idx in range(total_epochs):
        is_last_epoch = epoch_idx == total_epochs - 1
        max_batches = (
            round(frac * num_batches) if (is_last_epoch and frac > 0) else num_batches
        )
        progress.set_description(f"Epoch {epoch_idx + 1}/{total_epochs}")

        epoch_rng_key, rng_key = jax.random.split(rng_key)
        # Single name in both branches so static checkers can see it is always
        # bound; shape differs: pmap → (n_devices, max_batches, 2),
        # single-device → (max_batches, 2). The per-batch indexing below
        # branches on use_pmap to read the right axis.
        if use_pmap:
            if n_devices > 1:
                device_keys = jax.random.split(epoch_rng_key, n_devices)
            else:
                device_keys = jnp.expand_dims(epoch_rng_key, axis=0)
            batch_keys = jax.vmap(lambda k: jax.random.split(k, max_batches))(
                device_keys
            )
        else:
            batch_keys = jax.random.split(epoch_rng_key, max_batches)

        batch_metrics: List[Any] = []
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            batch = convert_batch(batch_data)

            if use_pmap:
                batch_key = batch_keys[:, batch_idx]
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
                params, opt_state, metric = step_fn(
                    params, opt_state, batch_sharded, batch_key
                )
                metric = unshard_energies(metric)
            else:
                params, opt_state, metric = step_fn(
                    params, opt_state, batch, batch_keys[batch_idx]
                )
                metric = float(metric)

            batch_metrics.append(
                iter_callback(epoch_idx, batch_idx, metric)
                if iter_callback is not None
                else metric
            )
            progress.set_postfix(
                {label: f"{metric:.4f}", "epoch": f"{epoch_idx + 1}/{total_epochs}"}
            )
            progress.update(1)

        iter_results.append(batch_metrics)
        avg_metric = sum(batch_metrics) / len(batch_metrics) if batch_metrics else 0.0

        if epoch_callback is not None:
            cb_params = (
                jax.tree_util.tree_map(lambda x: x[0], params) if use_pmap else params
            )
            epoch_results.append(
                epoch_callback(epoch_idx, cb_params, structure, config, rng_key)
            )
        else:
            epoch_results.append(None)

        if verbose and not use_tqdm:
            print(f"Epoch {epoch_idx + 1}/{total_epochs}, {label}: {avg_metric:.4f}")

    progress.close()
    if use_pmap:
        params = jax.tree_util.tree_map(lambda x: x[0], params)
    return params, iter_results, epoch_results


# ── public API ──────────────────────────────────────────────────────


def train(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    algorithm: str = "pc",
    autoregressive: bool = False,
    use_causal_mask: bool = True,
    metric_fn: Optional[Callable] = None,
    metric_name: Optional[str] = None,
    verbose: bool = True,
    use_tqdm: bool = True,
    epoch_callback: Optional[Callable] = None,
    iter_callback: Optional[Callable] = None,
    pmap_single_device: bool = False,
) -> Tuple[GraphParams, List[Any], List[Any]]:
    """Train a FabricPC graph with PC or backprop, standard or autoregressive.

    ``algorithm`` is an explicit signature parameter (not a ``config`` key) so
    the algorithm choice is visible at the call site rather than buried inside
    a dict. :func:`_validate_algo` raises if the chosen algorithm's required
    graph arguments are missing (e.g. backprop without a feedforward
    initializer). The objective optimized is the metric from
    :func:`_evaluation_metric`, chosen by a SINGLE ``metric_name``: a built-in
    (``"energy"``, ``"cross_entropy"``, ``"mse"``) or, with a
    user ``metric_fn`` ``(state, batch, structure) -> scalar``, any custom metric
    (``metric_name`` then just labels it, default ``"Untitled"``). ``metric_name=None``
    uses the algorithm default (PC -> energy, backprop -> cross-entropy). (The legacy
    ``config["loss_type"]`` is superseded by ``metric_name``/``metric_fn`` — e.g. for
    MSE pass ``metric_name="mse"``; it is not read from ``config``.)

    ``autoregressive=True`` enables next-token training for *either* algorithm: it
    adds a causal-mask clamp (gated by the ``use_causal_mask`` parameter, default
    True) and nothing else — the gradient rule and reported training metric are
    unchanged (AR-PC trains on energy, as the legacy AR trainer does). Auto-detects
    devices (JIT on one, pmap data-parallel on many) for **both** algorithms.
    Returns ``(trained_params, iter_results, epoch_results)``.
    """
    _validate_algo(algorithm, structure)
    if autoregressive:
        _validate_autoregressive(structure, use_causal_mask)
    return run_training_loop(
        params,
        structure,
        train_loader,
        optimizer,
        config,
        rng_key,
        algo=algorithm,
        metric_fn=metric_fn,
        metric_name=metric_name,
        autoregressive=autoregressive,
        use_causal_mask=use_causal_mask,
        verbose=verbose,
        use_tqdm=use_tqdm,
        epoch_callback=epoch_callback,
        iter_callback=iter_callback,
        pmap_single_device=pmap_single_device,
    )


def evaluate(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
    algorithm: str = "pc",
    autoregressive: bool = False,
    use_causal_mask: bool = True,
    metric_fn: Optional[Callable] = None,
    metric_name: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate a FabricPC graph; mirror of :func:`train`'s explicit-algorithm dispatch.

    ``algorithm`` is a signature parameter (not a ``config`` key), matching
    :func:`train`. One shared loop; the per-batch objective is
    :func:`_evaluation_metric`, keyed in the result by its name — the algorithm
    default (PC -> ``"energy"``; backprop -> ``"loss"``), a named built-in, or a
    user ``metric_fn`` keyed by ``metric_name`` (default ``"Untitled"``).
    ``autoregressive=True`` instead reports the output cross-entropy as ``"loss"``
    for *either* algorithm, plus ``perplexity`` and ``num_batches``, and adds the
    causal-mask clamp — matching legacy ``evaluate_autoregressive`` /
    ``evaluate_backprop_autoregressive``. Always returns ``accuracy``; adds
    ``perplexity`` whenever the reported objective is a cross-entropy loss
    (AR for either algorithm, or backprop).

    Note: this runs on a single device (numerically identical to the legacy
    evaluators' JIT path, the parity reference); eval is a cheap single pass, so
    multi-device/pmap evaluation is intentionally out of scope. Use
    :func:`evaluate_transformer` for device-parallel PC evaluation.
    """
    _validate_algo(algorithm, structure)
    if autoregressive:
        _validate_autoregressive(structure, use_causal_mask)
    metric_key = _result_key(algorithm, metric_fn, metric_name, autoregressive)

    jit_eval = jax.jit(
        lambda p, b, k: _eval_step(
            p,
            b,
            structure,
            k,
            algo=algorithm,
            metric_fn=metric_fn,
            metric_name=metric_name,
            autoregressive=autoregressive,
            use_causal_mask=use_causal_mask,
        )
    )
    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000
    batch_keys = jax.random.split(rng_key, num_batches)

    total_obj = 0.0
    total_weight = 0
    total_correct = 0
    total_n = 0
    n_batches = 0
    for batch_idx, batch_data in enumerate(test_loader):
        batch = convert_batch(batch_data)
        objective, weight, correct, count = jit_eval(
            params, batch, batch_keys[batch_idx]
        )
        total_obj += float(objective) * int(weight)
        total_weight += int(weight)
        total_correct += int(correct)
        total_n += int(count)
        n_batches += 1

    avg_obj = total_obj / total_weight if total_weight > 0 else 0.0
    accuracy = total_correct / total_n if total_n > 0 else 0.0
    result = {metric_key: avg_obj, "accuracy": accuracy}
    # Perplexity = exp(mean cross-entropy): report it only when the objective IS a
    # cross-entropy — the AR default (either algorithm) or the backprop default —
    # and no custom metric_fn overrides it (else exp(objective) is meaningless).
    reports_ce = (
        metric_fn is None
        and (metric_name or _default_metric_name(algorithm, autoregressive))
        == "cross_entropy"
    )
    if reports_ce:
        result["perplexity"] = float(jnp.exp(avg_obj))
    if autoregressive:
        result["num_batches"] = n_batches
    return result


def evaluate_transformer(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
) -> Dict[str, float]:
    """PC transformer evaluation (device-parallel): accuracy, cross-entropy, perplexity, energy.

    Carried over from the legacy trainer so this file covers the full PC surface.
    """
    n_devices = jax.device_count()

    epoch_key, rng_key = jax.random.split(rng_key)
    if n_devices > 1:
        device_keys = jax.random.split(epoch_key, n_devices)
    else:
        device_keys = jnp.expand_dims(epoch_key, axis=0)
    replicated_params = replicate_params(params, n_devices)

    try:
        num_batches = len(test_loader)
    except TypeError:
        num_batches = 1000
    batch_keys_per_device = jax.vmap(lambda k: jax.random.split(k, num_batches))(
        device_keys
    )

    def inference_fn(params_obj, sharded_batch, randgen_key):
        batch_size_ = next(iter(sharded_batch.values())).shape[0]
        clamps = build_clamps(sharded_batch, structure, clamp_target=False)
        state = initialize_graph_state(
            structure, batch_size_, randgen_key, clamps=clamps, params=params_obj
        )
        return run_inference(params_obj, state, clamps, structure)

    pmap_inference = jax.pmap(inference_fn, axis_name="devices")

    total_correct = total_samples = total_tokens = 0
    total_ce = total_energy = 0.0
    for batch_idx, batch_data in enumerate(test_loader):
        batch_key_for_step = batch_keys_per_device[:, batch_idx]
        batch = convert_batch(batch_data)
        batch_size = next(iter(batch.values())).shape[0]
        if batch_size % n_devices != 0:
            continue
        batch_sharded = shard_batch(batch, n_devices)
        final_states = pmap_inference(
            replicated_params, batch_sharded, batch_key_for_step
        )

        def get_device_energy(fs, batch_y):
            e = 0.0
            for node_name in structure.nodes:
                if structure.nodes[node_name].node_info.in_degree > 0:
                    e += jnp.sum(fs.nodes[node_name].energy)
            if "y" in structure.task_map:
                pred = fs.nodes[structure.task_map["y"]].z_latent
                if batch_y.ndim == pred.ndim:
                    e += jnp.sum((pred - batch_y) ** 2)
                elif batch_y.ndim == pred.ndim - 1:
                    e += jnp.sum((pred - jax.nn.one_hot(batch_y, pred.shape[-1])) ** 2)
            return e

        device_energies = jax.vmap(get_device_energy)(final_states, batch_sharded["y"])
        total_energy += float(jnp.sum(device_energies))
        total_samples += batch_size

        if "y" in structure.task_map:
            preds = final_states.nodes[structure.task_map["y"]].z_latent
            preds_flat = preds.reshape(batch_size, *preds.shape[2:])
            targets = batch["y"]
            pred_labels = jnp.argmax(preds_flat, axis=-1)
            true_labels = (
                jnp.argmax(targets, axis=-1)
                if targets.ndim == preds_flat.ndim
                else targets
            )
            total_correct += int(jnp.sum(pred_labels == true_labels))

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


# ── autoregressive generation ───────────────────────────────────────


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
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jax.Array], jnp.ndarray]:
    """Single ``lax.scan`` generation step over a fixed-size sliding context window.

    ``carry`` is ``(context_window, output_buffer, rng_key)``; static args are
    closed over by :func:`generate_autoregressive`. Returns ``(new_carry, next_token)``.
    Ported from ``train_autoregressive._generation_step``.
    """
    context_window, output_buffer, rng_key = carry
    rng_key, sample_key, init_key = jax.random.split(rng_key, 3)

    # 1D input node (seq_len,) takes int token indices (EmbeddingNode); a 2D input
    # node (seq_len, vocab) takes one-hot vectors (Linear).
    input_shape = structure.nodes[input_node].node_info.shape
    if len(input_shape) == 1:
        input_data = context_window
    else:
        input_data = jax.nn.one_hot(context_window, vocab_size)

    clamps = {input_node: input_data}  # input only — output runs free
    state = initialize_graph_state(
        structure, batch_size, init_key, clamps=clamps, params=params
    )
    final_state = run_inference(params, state, clamps, structure)

    # z_mu is post-activation (softmax) probabilities; take the last position.
    output_probs = final_state.nodes[output_node].z_mu
    output_last = output_probs[:, -1, :]
    logits = jnp.log(output_last + 1e-10) / temperature

    # top-k filtering (run unconditionally with effective k = vocab when unset)
    effective_top_k = top_k if top_k is not None else vocab_size
    top_k_logits, top_k_indices = jax.lax.top_k(logits, effective_top_k)
    neg_inf_mask = jnp.full_like(logits, float("-inf"))
    logits = neg_inf_mask.at[jnp.arange(batch_size)[:, None], top_k_indices].set(
        top_k_logits
    )

    # top-p (nucleus) filtering
    if top_p is not None:
        sorted_indices = jnp.argsort(-logits, axis=-1)
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
        cutoff_mask = cumsum_probs > top_p
        cutoff_mask = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=bool), cutoff_mask[:, :-1]], axis=-1
        )
        sorted_logits = jnp.where(cutoff_mask, float("-inf"), sorted_logits)
        unsort_indices = jnp.argsort(sorted_indices, axis=-1)
        logits = jnp.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    next_token = jax.random.categorical(sample_key, logits, axis=-1)
    new_context = jnp.concatenate([context_window[:, 1:], next_token[:, None]], axis=1)
    new_output_buffer = output_buffer.at[:, step_idx].set(next_token)
    return (new_context, new_output_buffer, rng_key), next_token


def generate_autoregressive(
    params: GraphParams,
    structure: GraphStructure,
    prompt: jnp.ndarray,
    max_new_tokens: int,
    rng_key: jax.Array,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> jnp.ndarray:
    """Autoregressively sample ``max_new_tokens`` from a trained model.

    JIT-compiled; the inner loop is a ``lax.scan`` over fixed-size buffers.
    ``prompt`` may be ``(seq_len,)`` or ``(batch, seq_len)``; returns the prompt
    concatenated with the generated tokens (batch dim dropped if the input was 1D).
    Ported from ``train_autoregressive.generate_autoregressive``.
    """
    if prompt.ndim == 1:
        prompt = prompt[None, :]
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

    # Build the initial context window (pad-left or truncate the prompt to seq_len).
    if prompt_len >= seq_len:
        context_window = prompt[:, -seq_len:]
    else:
        context_window = jnp.pad(
            prompt, ((0, 0), (seq_len - prompt_len, 0)), constant_values=0
        )

    @jax.jit
    def jit_generate_loop(context: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
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
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        output_buffer = jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
        init_carry = (context, output_buffer, rng)
        (_, final_output_buffer, _), _ = jax.lax.scan(
            scan_fn, init_carry, jnp.arange(max_new_tokens)
        )
        return final_output_buffer

    generated_tokens = jit_generate_loop(context_window, rng_key)
    result = jnp.concatenate([prompt, generated_tokens], axis=1)
    if unbatch:
        result = result[0]
    return result
