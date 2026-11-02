"""Hopfield rows: get a stored pattern back from a noisy copy.

With today's settings these are trained one-layer denoisers, not yet a test
of attractor recall; see the notes below.

The scope asks for associative-memory retrieval: recall a whole pattern from
a corrupted cue, with Hopfield nodes. This follows the binary experiment in
``examples/storkey_hopfield_recall.py``: seven random +1/-1 patterns of 64
bits, a three-node graph probe -> StorkeyHopfield -> output, trained on
(noisy copy, clean pattern) pairs with 15% of the bits flipped.

A few things are worth saying plainly, because the node's name suggests more
than it does:

- ``StorkeyHopfield`` does not use the Storkey learning rule. Its W starts
  small and is learned by gradient descent, so what we train here is a
  denoiser that has seen 100 noisy copies of each pattern for 30 epochs, not
  a memory written once from clean patterns.
- So every row also reports the real Storkey rule (Storkey 1997) on the same
  patterns and the same noisy probes, as ``storkey_rule_*``. It does not
  depend on training, so it is the same number in all three rows; it is a
  fixed ruler, not a fourth method.
- Backprop does have a recall mode on this graph: one forward pass through
  the node, probe -> tanh(blend of probe and probe @ W). sPC and ePC read the
  recalled pattern after inference settles, but with these settings settling
  changes the one-pass readout by 0.002 or less (the ``forward_*`` scores
  show it). So all three rows are one-layer denoisers that differ only in
  how W was trained. They do not test attractor recall yet.
- Every row also scores the do-nothing baseline, the noisy probe handed
  back unchanged, as ``probe_*``, and the lift over it as
  ``bit_accuracy_lift_*``. A row that learned nothing has a lift of 0.

Recall is read from the Hopfield node's latent after settling (as in the
example) and turned into bits with sign(), where 0 counts as +1. The example
recalls with 100 inference steps; the rows use the training graph's 20,
because on our check 100 steps moved bit accuracy by 0.002 at most.

Scores, at each flip rate p in NOISE_LEVELS (named ``_p20`` for 20%):

- ``bit_accuracy``: fraction of bits that match the stored pattern. The
  overlap m used in the physics papers is just 2 * bit_accuracy - 1.
- ``exact_recall``: fraction of probes brought back with every bit right.
- ``nearest_correct``: fraction whose output is closer to the right pattern
  than to any other stored one. The example calls this "exact"; there a
  tie goes to whichever pattern comes first, here a tie does not count.

At p = 0 any network that keeps the probe's signs is perfect, so it only
checks that nothing is destroyed, not that the patterns are stored. p = 0.5
is chance (bit accuracy near 0.5): any recall there is luck.
"""

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.inference import InferenceSGD
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode, StorkeyHopfield
from fabricpc.training.metrics import EvalMetric

DIM = 64  # bits per pattern
N_PATTERNS = 7  # 7 / 64 = 0.11 patterns per bit, below the classic 0.14 limit
TRAIN_NOISE = 0.15  # fraction of bits flipped in the training copies
TRAIN_COPIES = 100  # noisy copies of each pattern per epoch
NUM_EPOCHS = 30
BATCH_SIZE = 64
HOPFIELD_STRENGTH = 1.0

NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.5)
NOISE_PCTS = tuple(int(round(p * 100)) for p in NOISE_LEVELS)
PROBES_PER_PATTERN = 20  # at every noise level
PROBE_BATCH = 100
MAIN_METRIC = "bit_accuracy_p20"

# Sweeps of the Storkey network's one-bit-at-a-time updates. At this size it
# settles in a handful; the rest are spare.
STORKEY_SWEEPS = 10


def _rng(seed: int, stream: int) -> np.random.Generator:
    """A numpy generator for one use of a trial seed, kept apart from the
    others so the patterns, training copies, and probes never share draws."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream]))


def make_patterns(seed: int, n_patterns: int = N_PATTERNS, dim: int = DIM):
    """Random +1/-1 patterns, the same for every row that gets this seed."""
    bits = _rng(seed, 0).integers(0, 2, size=(n_patterns, dim))
    return (2.0 * bits - 1.0).astype(np.float32)


def flip_bits(clean, p: float, rng: np.random.Generator):
    """Flip each bit with probability p."""
    clean = np.asarray(clean, dtype=np.float32)
    flip = rng.random(clean.shape) < p
    return np.where(flip, -clean, clean).astype(np.float32)


class RecallTrainLoader:
    """(noisy copy, clean pattern) pairs, like the example's loader.

    Every epoch makes ``copies`` fresh noisy copies of each pattern and
    shuffles them. A new loader built with the same seed replays the same
    epochs, so all three rows train on the same data.
    """

    def __init__(self, patterns, noise, copies, batch_size, seed):
        self.patterns = np.asarray(patterns, dtype=np.float32)
        self.noise = noise
        self.copies = copies
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = _rng(self.seed, 1000 + self._epoch)
        self._epoch += 1
        clean = np.repeat(self.patterns, self.copies, axis=0)
        noisy = flip_bits(clean, self.noise, rng)
        order = rng.permutation(len(clean))
        clean, noisy = clean[order], noisy[order]
        for start in range(0, len(clean) - self.batch_size + 1, self.batch_size):
            end = start + self.batch_size
            yield noisy[start:end], clean[start:end]

    def __len__(self):
        return (len(self.patterns) * self.copies) // self.batch_size


class ProbeLoader:
    """The test set: noisy probes of every pattern at every noise level.

    Batches are dicts. Besides ``x`` (the probe) and ``y`` (the clean
    pattern) they carry what the recall scores need: ``noise_pct`` (the
    flip rate in percent), ``pattern`` (which stored pattern it came from),
    and ``memories`` (all stored patterns, repeated for every sample so the
    batch stays one row per sample). The probes are shuffled once, so every
    batch mixes the noise levels and a short learning curve still sees all
    of them. Same seed, same probes, in the same order.
    """

    def __init__(self, patterns, probes_per_pattern, batch_size, seed):
        patterns = np.asarray(patterns, dtype=np.float32)
        rng = _rng(seed, 2)
        xs, ys, pcts, idx = [], [], [], []
        pattern_ids = np.repeat(np.arange(len(patterns)), probes_per_pattern)
        for p, pct in zip(NOISE_LEVELS, NOISE_PCTS):
            clean = patterns[pattern_ids]
            xs.append(flip_bits(clean, p, rng))
            ys.append(clean)
            pcts.append(np.full(len(clean), pct, dtype=np.int32))
            idx.append(pattern_ids.astype(np.int32))
        order = rng.permutation(sum(len(x) for x in xs))
        self._x = np.concatenate(xs)[order]
        self._y = np.concatenate(ys)[order]
        self._pct = np.concatenate(pcts)[order]
        self._idx = np.concatenate(idx)[order]
        self._patterns = patterns
        self.batch_size = batch_size

    def __iter__(self):
        for start in range(0, len(self._x), self.batch_size):
            end = start + self.batch_size
            n = len(self._x[start:end])
            yield {
                "x": self._x[start:end],
                "y": self._y[start:end],
                "noise_pct": self._pct[start:end],
                "pattern": self._idx[start:end],
                "memories": np.broadcast_to(
                    self._patterns, (n,) + self._patterns.shape
                ),
            }

    def __len__(self):
        return -(-len(self._x) // self.batch_size)


# --- the Storkey learning rule, used as a fixed reference ------------------


def storkey_weights(patterns: jax.Array) -> jax.Array:
    """Write the patterns into a weight matrix with Storkey's (1997) rule.

    One pattern at a time: w_ij += (xi_i xi_j - xi_i h_ji - h_ij xi_j) / n,
    where h_ij is the field at i from every unit except i and j. The
    diagonal is kept at zero.
    """
    n = patterns.shape[-1]

    def add(w, xi):
        field = w @ xi
        # h_ij = field_i - w_ij xi_j (w_ii is zero, so it drops out)
        h = field[:, None] - w * xi[None, :]
        w = w + (jnp.outer(xi, xi) - xi[:, None] * h.T - h * xi[None, :]) / n
        return w * (1.0 - jnp.eye(n)), None

    w, _ = jax.lax.scan(add, jnp.zeros((n, n), patterns.dtype), patterns)
    return w


def storkey_recall(w: jax.Array, probe: jax.Array, sweeps: int = STORKEY_SWEEPS):
    """Classic recall: update one bit at a time, in order, for ``sweeps``
    passes. A bit becomes the sign of its field, with 0 counting as +1."""
    n = probe.shape[-1]

    def step(t, s):
        i = t % n
        return s.at[i].set(jnp.where(w[i] @ s >= 0, 1.0, -1.0))

    return jax.lax.fori_loop(0, sweeps * n, step, probe)


# --- recall scores ---------------------------------------------------------


def _bits(z):
    return jnp.where(z >= 0, 1.0, -1.0)


def _level_mask(batch, pct):
    return (jnp.asarray(batch["noise_pct"]) == pct).astype(jnp.float32)


def _scores(recalled, batch):
    """bit accuracy, exact recall and nearest-correct for each sample."""
    y = jnp.asarray(batch["y"], dtype=recalled.dtype)
    match = recalled == y
    bit = jnp.mean(match, axis=-1)
    exact = jnp.all(match, axis=-1).astype(jnp.float32)
    memories = jnp.asarray(batch["memories"], dtype=recalled.dtype)
    dist = jnp.sum((memories - recalled[:, None, :]) ** 2, axis=-1)
    # Strictly closer to the right pattern than to every other one; a tie
    # does not count, so the order of the patterns cannot decide it.
    right = jax.nn.one_hot(batch["pattern"], dist.shape[-1], dtype=bool)
    to_right = jnp.sum(jnp.where(right, dist, 0.0), axis=-1)
    to_others = jnp.min(jnp.where(right, jnp.inf, dist), axis=-1)
    nearest = (to_right < to_others).astype(jnp.float32)
    return {"bit_accuracy": bit, "exact_recall": exact, "nearest_correct": nearest}


# evaluate() calls every metric on the same batch and state inside one
# traced step. All 45 scores come from three recalls (the model's, the
# Storkey rule's and the probe itself), so each set is worked out once per
# step and reused. That
# keeps the traced step, and its compile, small.
_cache: Dict[str, tuple] = {}


def _once(slot, key, make):
    """make(), unless it was already made for this very ``key`` object."""
    hit = _cache.get(slot)
    if hit is not None and hit[0] is key:
        return hit[1]
    value = make()
    _cache[slot] = (key, value)
    return value


def _storkey_recalled(batch):
    # Every sample carries the same stored patterns (see ProbeLoader), so
    # the weights are written once from the first sample's copy.
    w = storkey_weights(jnp.asarray(batch["memories"][0], dtype=jnp.float32))
    probes = jnp.asarray(batch["x"], dtype=jnp.float32)
    return jax.vmap(lambda probe: storkey_recall(w, probe))(probes)


def _model_metric(kind: str, pct: int) -> EvalMetric:
    def fn(state, batch, structure):
        z = state.nodes["hopfield"].z_latent
        scores = _once("model", z, lambda: _scores(_bits(z), batch))
        mask = _level_mask(batch, pct)
        return scores[kind] * mask, mask

    return EvalMetric(fn=fn)


def _storkey_metric(kind: str, pct: int) -> EvalMetric:
    def fn(state, batch, structure):
        scores = _once(
            "storkey", batch, lambda: _scores(_storkey_recalled(batch), batch)
        )
        mask = _level_mask(batch, pct)
        return scores[kind] * mask, mask

    return EvalMetric(fn=fn)


def _probe_scores(batch):
    return _once(
        "probe",
        batch,
        lambda: _scores(_bits(jnp.asarray(batch["x"], dtype=jnp.float32)), batch),
    )


def _probe_metric(kind: str, pct: int) -> EvalMetric:
    """The do-nothing baseline: the noisy probe handed back unchanged."""

    def fn(state, batch, structure):
        mask = _level_mask(batch, pct)
        return _probe_scores(batch)[kind] * mask, mask

    return EvalMetric(fn=fn)


def _lift_metric(pct: int) -> EvalMetric:
    """Bit accuracy minus the do-nothing baseline's, per sample."""

    def fn(state, batch, structure):
        z = state.nodes["hopfield"].z_latent
        model = _once("model", z, lambda: _scores(_bits(z), batch))
        mask = _level_mask(batch, pct)
        lift = model["bit_accuracy"] - _probe_scores(batch)["bit_accuracy"]
        return lift * mask, mask

    return EvalMetric(fn=fn)


def recall_metrics(pcts: Sequence[int] = NOISE_PCTS) -> Dict[str, EvalMetric]:
    """Every recall score at every noise level, named like bit_accuracy_p20."""
    out = {}
    for pct in pcts:
        for kind in ("bit_accuracy", "exact_recall", "nearest_correct"):
            out[f"{kind}_p{pct:02d}"] = _model_metric(kind, pct)
        for kind in ("bit_accuracy", "exact_recall"):
            out[f"storkey_rule_{kind}_p{pct:02d}"] = _storkey_metric(kind, pct)
            out[f"probe_{kind}_p{pct:02d}"] = _probe_metric(kind, pct)
        out[f"bit_accuracy_lift_p{pct:02d}"] = _lift_metric(pct)
    return out


# --- the rows --------------------------------------------------------------


def _solver_for(algorithm: str):
    if algorithm == "epc":
        # Not ePC's defaults. With eta 0.001 and 5 steps the latent hardly
        # leaves the probe, and W never learned to denoise: after 30 epochs
        # the row scored exactly the noisy probe (bit accuracy 0.796 at 20%,
        # exact recall 0). At eta 0.1 and 20 steps it learns like sPC
        # (0.950 vs 0.949 on seed 0), and eta times the stiffness stays near
        # 0.2, far from the safety net's 1.0.
        return EPCInference(eta_infer=0.1, infer_steps=20)
    return InferenceSGD(eta_infer=0.05, infer_steps=20)


def _recall_graph_factory(algorithm: str):
    """probe -> StorkeyHopfield -> output, as built in the recall example."""

    def build(rng_key: jax.Array):
        probe = IdentityNode(shape=(DIM,), name="probe")
        hopfield = StorkeyHopfield(
            shape=(DIM,),
            name="hopfield",
            hopfield_strength=HOPFIELD_STRENGTH,
            use_bias=False,
            enforce_symmetry=True,
            zero_diagonal=False,
            weight_init=NormalInitializer(mean=0.0, std=0.01),
        )
        output = IdentityNode(shape=(DIM,), name="output")
        structure = graph(
            nodes=[probe, hopfield, output],
            edges=[
                Edge(source=probe, target=hopfield.slot("in")),
                Edge(source=hopfield, target=output.slot("in")),
            ],
            task_map=TaskMap(x=probe, y=output),
            inference=_solver_for(algorithm),
        )
        return initialize_params(structure, rng_key), structure

    return build


def _recall_loaders(seed: int):
    patterns = make_patterns(seed)
    train = RecallTrainLoader(
        patterns, TRAIN_NOISE, TRAIN_COPIES, BATCH_SIZE, seed=seed
    )
    test = ProbeLoader(patterns, PROBES_PER_PATTERN, PROBE_BATCH, seed=seed)
    return train, test


def hopfield_family() -> dict:
    """The three rows of ``patterns64-hopfield``, keyed by row id."""
    # Imported here: the registry imports this module while it is being built.
    from fabricpc.bench.definitions.registry import ALGORITHMS, BenchmarkRow, _safety_net

    scores = recall_metrics()
    rows = {}
    for algo in ALGORITHMS:
        row_id = f"patterns64-hopfield-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="patterns64",
            model="hopfield",
            algorithm=algo,
            model_factory=_recall_graph_factory(algo),
            loader_factory=_recall_loaders,
            optimizer_factory=lambda total_steps: optax.adamw(1e-3, weight_decay=0.01),
            train_config={"num_epochs": NUM_EPOCHS},
            batch_size=BATCH_SIZE,
            rate_control=_safety_net(algo),
            tier=2,
            metric=MAIN_METRIC,
            eval_metrics=scores,
        )
    return rows
