"""The deep FC-ResNet rows: mnist-fcresnet{depth}-{spc,epc,backprop}.

Sponsor issue #59 asks us to test muPC past the depths pcx tried, to 100
layers and more. These rows are the network from ``examples/mupc_demo.py``
at depths 8, 16, 32, 64 and 128, so one command per depth gives accuracy
against depth for all three methods on the same graph:

    input(784) -> stem(64) -> depth x LinearResidual(64, tanh) -> output(10)

Each block is a single PC node, ``z = tanh(W x + b) + x``: the weight path
goes into the node's "in" slot and the identity skip into its "skip" slot.
muPC scales the weight path by gain / sqrt(64 * depth) and leaves the skip
alone. That skip is the whole point: muPC only keeps deep nets trainable
because the identity path carries the signal, so a plain deep chain would
not do. The stem gets 1 / sqrt(784) and the softmax readout (Xavier, cross
entropy) is left out of muPC, all as in the demo.

Everything else is the demo's defaults too, so the sponsor's own table can
be checked against ours: width 64, batch 256, 3 epochs, AdamW at 0.002 with
weight decay 0.01, and sPC settles with rate 0.1 for max(20, 3 * (depth +
2)) steps. The demo's table (sPC, one run each) reads 92.0, 89.7, 85.6,
84.1 and 82.2% at depths 8, 16, 32, 64 and 128. We do not use those as
expected scores, because they are single runs with the demo's seed.

Two things differ from the demo, both for all three methods alike: Adam's
epsilon is 1e-12 like our other deep rows, and the PC rows get the usual
rate safety net. ePC uses its defaults, like the MLP rows.

This is the demo's recipe, not the muPC paper's (Innocenti et al., NeurIPS
2025, arXiv 2505.13124). The paper applies the activation before the matmul
(W tanh(z) + z), has no biases and no Kaiming gain, puts muPC on the readout
with a squared-error loss, and trains with Adam at 0.1, batch 64, one epoch,
and as many settling steps as there are hidden layers. The paper's accuracy
stays flat with depth; the demo's falls. A paper-faithful variant needs a
pre-activation option on LinearResidual first, so it is left for later.

Depths 32 and up are tier 2: on a laptop CPU they are slow (see the probe).
"""

import re
from typing import Dict, Optional

import jax
import optax

from fabricpc.core.activations import (
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge, GraphNamespace
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import IdentityNode, Linear, LinearResidual

DEPTHS = (8, 16, 32, 64, 128)
TIER_ONE_MAX_DEPTH = 16

# The demo's defaults (examples/mupc_demo.py).
WIDTH = 64
BATCH_SIZE = 256
NUM_EPOCHS = 3
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01
ETA_INFER = 0.1

_ID_PATTERN = re.compile(r"^(?P<model>.+-fcresnet)(?P<depth>\d+)-")


def infer_steps(depth: int) -> int:
    """The demo's settling budget: three steps per layer, at least 20, so
    the output error has time to reach the stem of a deep net."""
    return max(20, 3 * (depth + 2))


def _solver_for(algorithm: str, depth: int):
    """Backprop never settles, but the graph needs a solver to build, so it
    gets the sPC one."""
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGD(eta_infer=ETA_INFER, infer_steps=infer_steps(depth))


def build_fc_resnet(depth: int, inference):
    """The demo's FC-ResNet with LinearResidual blocks (one PC node each)."""
    mupc_init = MuPCInitializer()
    pixels = IdentityNode(shape=(784,), name="input")
    stem = Linear(
        shape=(WIDTH,), weight_init=mupc_init, flatten_input=True, name="stem"
    )
    nodes = [pixels, stem]
    edges = [Edge(source=pixels, target=stem.slot("in"))]
    prev = stem
    for i in range(depth):
        with GraphNamespace(f"block{i}"):
            res = LinearResidual(
                shape=(WIDTH,),
                activation=TanhActivation(),
                weight_init=mupc_init,
                name="res",
            )
        nodes.append(res)
        edges.append(Edge(source=prev, target=res.slot("in")))  # weight path
        edges.append(Edge(source=prev, target=res.slot("skip")))  # identity
        prev = res
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=XavierInitializer(),
        name="output",
    )
    nodes.append(output)
    edges.append(Edge(source=prev, target=output.slot("in")))
    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=pixels, y=output),
        inference=inference,
        scaling=MuPCConfig(include_output=False),
    )


def _model_factory(algorithm: str, depth: int):
    def build(rng_key: jax.Array):
        structure = build_fc_resnet(depth, _solver_for(algorithm, depth))
        return initialize_params(structure, rng_key), structure

    return build


def _optimizer(total_steps: int) -> optax.GradientTransformation:
    from fabricpc.bench.definitions.registry import _DEEP_ADAM_EPS

    return optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=_DEEP_ADAM_EPS)


def fcresnet_depth_family(depths=DEPTHS) -> Dict[str, object]:
    """Three rows per depth, keyed by row id."""
    # Imported here because the registry imports this module while it is
    # still being built.
    from fabricpc.bench.definitions.registry import (
        ALGORITHMS,
        BenchmarkRow,
        _flat_image_loaders,
        _safety_net,
    )

    rows = {}
    for depth in depths:
        for algo in ALGORITHMS:
            row_id = f"mnist-fcresnet{depth}-{algo}"
            rows[row_id] = BenchmarkRow(
                id=row_id,
                dataset="mnist",
                model=f"fcresnet{depth}",
                algorithm=algo,
                model_factory=_model_factory(algo, depth),
                loader_factory=_flat_image_loaders("mnist", BATCH_SIZE),
                optimizer_factory=_optimizer,
                train_config={"num_epochs": NUM_EPOCHS},
                batch_size=BATCH_SIZE,
                tier=1 if depth <= TIER_ONE_MAX_DEPTH else 2,
                rate_control=_safety_net(algo),
                depth=depth,
            )
    return rows


# ---------------------------------------------------------------- reporting


def depth_of(row_id: str) -> Optional[int]:
    """A row's depth: the registry's if it has one, else read off the id
    (so old results from a depth we no longer register still show up)."""
    try:
        from fabricpc.bench.definitions.registry import ROWS

        row = ROWS.get(row_id)
        if row is not None and row.depth is not None:
            return row.depth
    except Exception:  # the report should work even if the registry cannot load
        pass
    match = _ID_PATTERN.match(row_id)
    return int(match.group("depth")) if match else None


def _cell(stats, n=None) -> str:
    if not stats or stats.get("mean") is None:
        return "-"
    text = f"{stats['mean']:.4f}"
    if stats.get("se") is not None:
        text += f" ± {stats['se']:.4f}"
    if n is not None:
        text += f" (n={n})"
    return text


def depth_table(loaded_rows):
    """(header, body) of a depth-by-method table, or None if no row has a
    depth. ``loaded_rows`` are the report's row dicts (``report.load_row``),
    each with an optional ``source``: the results folder it came from.

    Runs of the same row from different results folders (say a full GPU run
    and a 1-seed check on a laptop) get a line each, with a folder column,
    so one never hides the other. Each cell shows its seed count when known.
    """
    methods = ("backprop", "spc", "epc")
    cells = {}
    for row in loaded_rows:
        depth = depth_of(row["id"])
        match = _ID_PATTERN.match(row["id"])
        if depth is None or not match:
            continue
        key = (row.get("source") or "", match.group("model"), depth)
        metric = row.get("metric")
        stats = (row.get("stats") or {}).get(metric) if metric else None
        cells.setdefault(key, {})[row.get("algorithm")] = _cell(stats, row.get("n"))
    if not cells:
        return None
    by_folder = len({source for source, _, _ in cells}) > 1
    order = sorted(cells, key=lambda k: (k[1], k[2], k[0]))
    body = []
    for source, model, depth in order:
        line = [model, str(depth)]
        line += [cells[(source, model, depth)].get(m, "-") for m in methods]
        body.append([source or "."] + line if by_folder else line)
    head = ["model", "depth", *methods]
    return (["results folder", *head] if by_folder else head), body
