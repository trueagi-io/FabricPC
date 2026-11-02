"""The list of benchmark rows.

A row is one dataset, one model, one training algorithm. Rows in the same
family (same dataset and model) share a graph so the algorithms can be
compared on equal footing. Row ids look like ``mnist-mlp-spc``.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Tuple

import jax
import optax

from fabricpc.bench.definitions.bpe import tinyshakespeare_bpe_transformer_family
from fabricpc.bench.definitions.cifar100 import cifar100_vgg5_family
from fabricpc.bench.definitions.deep import fcresnet_depth_family
from fabricpc.bench.definitions.tinyimagenet import tinyimagenet_vgg5_family
from fabricpc.core.activations import (
    GeluActivation,
    ReLUActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD, InferenceSGDNormClip
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import build_resnet18, create_deep_transformer, create_vgg
from fabricpc.nodes import IdentityNode, Linear
from fabricpc.training import metrics as training_metrics

# The three ways we can train the same graph.
ALGORITHMS: Tuple[str, ...] = ("spc", "epc", "backprop")

# What each row's model factory returns: (params, structure).
ModelFactory = Callable[[jax.Array], Tuple[object, object]]
LoaderFactory = Callable[[int], Tuple[object, object]]


@dataclass(frozen=True)
class Reference:
    """The score a full run of a row is expected to land near.

    ``floor`` is the smallest allowed miss, in the metric's own units (0.005
    is half a percentage point of accuracy). ``fabricpc.bench.stats.band`` widens
    it to two standard errors when the run is noisier than that.
    """

    metric: str
    value: float
    source: str  # where the number came from, so anyone can check it
    floor: float = 0.005


@dataclass(frozen=True)
class RateControl:
    """Train a PC row with a rate that follows the settle's stiffness.

    See ``fabricpc.training.InferenceRateController``: every ``every``
    updates the rate is set so eta_infer * stiffness stays at ``target``,
    never above the rate the row's solver was built with.
    """

    target: float
    every: int = 20
    window: int = 3


# The safety net every PC row trains with: the inference rate shrinks only
# when a run nears the settle's stability limit, eta * stiffness = 2
# (fabricpc.training.InferenceRateController). Where it acts comes from the
# end-of-training values of our own runs: ePC rows were healthy at 0.07 or
# below (MNIST, Fashion-MNIST, VGG-5) and broke at 3 to 15 (the autoencoder,
# 3 of 5 seeds); sPC rows were healthy up to 1.7 (Fashion-MNIST 1.6, VGG-5
# 1.0 to 1.7) and degraded above 2 (autoencoder 1.9 to 2.4, the transformer
# about 650). So ePC acts at 1.0 and sPC at 1.8, and runs that were healthy
# train as before. Probing every 50 updates adds roughly a tenth to the
# training time.
_SAFETY_TARGET = {"epc": 1.0, "spc": 1.8}
_SAFETY_EVERY = 50


def _safety_net(algorithm: str) -> Optional["RateControl"]:
    if algorithm not in _SAFETY_TARGET:
        return None
    return RateControl(target=_SAFETY_TARGET[algorithm], every=_SAFETY_EVERY)


@dataclass(frozen=True)
class BenchmarkRow:
    """Everything needed to run one benchmark row."""

    id: str
    dataset: str
    model: str
    algorithm: str
    model_factory: ModelFactory
    loader_factory: LoaderFactory
    # Built once per trial from the run's total training steps, which a
    # learning-rate schedule needs; rows with a constant rate ignore it.
    optimizer_factory: Callable[[int], optax.GradientTransformation]
    train_config: Mapping[str, object]
    batch_size: int
    n_trials: int = 5
    tier: int = 1
    reference: Optional[Reference] = None  # None until a full run sets one
    metric: str = "accuracy"  # the score this row is judged and compared on
    rate_control: Optional[RateControl] = None  # PC rows: follow the stiffness
    # Scores to evaluate instead of evaluate()'s defaults, for rows that are
    # not classifiers. PC rows also get the settled energy.
    eval_metrics: Optional[Mapping[str, training_metrics.EvalMetric]] = None
    depth: Optional[int] = None  # residual blocks, for rows swept over depth

    def metrics_for(self, trainer_algorithm: str):
        """What to pass as ``evaluate(..., metrics=)``: None means its
        defaults. ``trainer_algorithm`` is the trainer's ("pc" or
        "backprop"); PC evaluations also report the settled energy."""
        if self.eval_metrics is None:
            return None
        chosen = dict(self.eval_metrics)
        if trainer_algorithm == "pc":
            chosen["energy"] = training_metrics.internal_energy
        return chosen


def _solver_for(algorithm: str):
    """Pick the inference solver for an algorithm.

    Backprop does not settle latents, but the graph still needs an inference
    object to build, so it gets the same one as spc.
    """
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGD(eta_infer=0.05, infer_steps=20)


def _mnist_mlp_factory(algorithm: str) -> ModelFactory:
    """Build the MNIST MLP from the mnist demo: 784 -> 256 -> 64 -> 10."""

    def build(rng_key: jax.Array):
        pixels = IdentityNode(shape=(784,), name="pixels")
        hidden1 = Linear(
            shape=(256,),
            activation=SigmoidActivation(),
            name="hidden1",
            weight_init=XavierInitializer(),
        )
        hidden2 = Linear(
            shape=(64,),
            activation=SigmoidActivation(),
            name="hidden2",
            weight_init=XavierInitializer(),
        )
        output = Linear(
            shape=(10,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
            weight_init=XavierInitializer(),
        )
        structure = graph(
            nodes=[pixels, hidden1, hidden2, output],
            edges=[
                Edge(source=pixels, target=hidden1.slot("in")),
                Edge(source=hidden1, target=hidden2.slot("in")),
                Edge(source=hidden2, target=output.slot("in")),
            ],
            task_map=TaskMap(x=pixels, y=output),
            inference=_solver_for(algorithm),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _flat_image_loaders(dataset: str, batch_size: int) -> LoaderFactory:
    """Loaders for one trial of a flattened-image dataset (MNIST-like).

    ``dataset`` is "mnist" or "fashionmnist". The seed fixes the shuffle order.
    """

    def build(seed: int):
        # Imported here so the registry can be listed without tensorflow.
        from fabricpc.utils.data import dataloader

        loader_cls = {
            "mnist": dataloader.MnistLoader,
            "fashionmnist": dataloader.FashionMnistLoader,
        }[dataset]
        train = loader_cls(
            "train",
            batch_size=batch_size,
            tensor_format="flat",
            shuffle=True,
            seed=seed,
        )
        test = loader_cls(
            "test", batch_size=batch_size, tensor_format="flat", shuffle=False
        )
        return train, test

    return build


# Mean test accuracy from our first full runs: 5 seeds x 20 epochs on a
# Colab T4, 2026-09-23. These are our own numbers, not pcx's.
_MLP_STAGE1_SOURCE = "Team 16 first full run, Colab T4, 5 seeds x 20 epochs, 2026-09-23"
_MLP_STAGE1_ACCURACY = {
    "mnist": {"spc": 0.9818, "epc": 0.9815, "backprop": 0.9816},
    "fashionmnist": {"spc": 0.8840, "epc": 0.8881, "backprop": 0.8888},
}


def _mlp_family(dataset: str) -> Dict[str, BenchmarkRow]:
    """The 784-256-64-10 MLP on a flattened-image dataset, three algorithms."""
    rows = {}
    batch_size = 200
    for algo in ALGORITHMS:
        row_id = f"{dataset}-mlp-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset=dataset,
            model="mlp",
            algorithm=algo,
            model_factory=_mnist_mlp_factory(algo),
            loader_factory=_flat_image_loaders(dataset, batch_size),
            optimizer_factory=lambda total_steps: optax.adamw(0.001, weight_decay=0.1),
            train_config={"num_epochs": 20},
            batch_size=batch_size,
            rate_control=_safety_net(algo),
            reference=Reference(
                metric="accuracy",
                value=_MLP_STAGE1_ACCURACY[dataset][algo],
                source=_MLP_STAGE1_SOURCE,
            ),
        )
    return rows


def _vgg_solver_for(algorithm: str):
    """Solver settings for the VGG rows.

    pcx trained VGG-5 with 8 settling steps on CIFAR-10. The state learning
    rate here is a starting point for the GPU runs, not a tuned value.
    """
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGD(eta_infer=0.015, infer_steps=8)


def _cifar10_vgg5_factory(algorithm: str) -> ModelFactory:
    """VGG-5 on 32x32x3 images with 10 classes, pcx's channel layout.

    Each conv block is one PC node with its max pool inside (fuse_pool), as
    in pcx. A separate MaxPool node per block doubles the PC chain to eight
    latent layers, and sPC then fails to train (37% after 50 epochs, against
    81% after 15 with the fused layout); backprop is the same either way.
    """

    def build(rng_key: jax.Array):
        structure = create_vgg(
            5,
            input_shape=(32, 32, 3),
            num_classes=10,
            inference=_vgg_solver_for(algorithm),
            fuse_pool=True,
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _cifar10_loaders(batch_size: int) -> LoaderFactory:
    """CIFAR-10 loaders for one trial: training images get random flips and
    4-pixel random crops, as in pcx; test images are left alone."""

    def build(seed: int):
        from fabricpc.utils.data import dataloader

        train = dataloader.AugmentedImageLoader(
            dataloader.Cifar10Loader(
                "train", batch_size=batch_size, shuffle=True, seed=seed
            ),
            flip_prob=0.5,
            crop_pad=4,
            seed=seed,
        )
        test = dataloader.Cifar10Loader("test", batch_size=batch_size, shuffle=False)
        return train, test

    return build


# VGG-5 weights: AdamW with pcx's rounded rate and decay, warmed up linearly
# over the first tenth of training and then cosine-decayed to zero. These
# are starting values, not tuned ones.
_VGG_PEAK_LR = 2.6e-4
_VGG_WEIGHT_DECAY = 1.2e-5
_VGG_WARMUP_FRACTION = 0.1

# Adam's epsilon for the deep rows. sPC's error reaches the first layers of
# VGG-5 and the transformer scaled down by about eta_infer per hop, so their
# weight gradients are 1e-8 to 1e-10: at the default epsilon of 1e-8 Adam
# barely moves them (the transformer's embedding stayed frozen under sPC).
# The gradients point the right way, so a smaller epsilon lets Adam give
# them normal-sized steps. Backprop's gradients are far above both values,
# so its runs do not change. All three methods use the same optimizer.
_DEEP_ADAM_EPS = 1e-12


def _vgg_lr_schedule(total_steps: int) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=_VGG_PEAK_LR,
        warmup_steps=max(1, int(total_steps * _VGG_WARMUP_FRACTION)),
        decay_steps=max(2, total_steps),
        end_value=0.0,
    )


def _cifar10_vgg5_family() -> Dict[str, BenchmarkRow]:
    # Batch size and epoch count follow pcx's VGG-5 CIFAR-10 runs. The weight
    # learning rate and weight decay are pcx's centered-nudging values, rounded.
    rows = {}
    batch_size = 128
    for algo in ALGORITHMS:
        row_id = f"cifar10-vgg5-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="cifar10",
            model="vgg5",
            algorithm=algo,
            model_factory=_cifar10_vgg5_factory(algo),
            loader_factory=_cifar10_loaders(batch_size),
            optimizer_factory=lambda total_steps: optax.adamw(
                _vgg_lr_schedule(total_steps),
                weight_decay=_VGG_WEIGHT_DECAY,
                eps=_DEEP_ADAM_EPS,
            ),
            train_config={"num_epochs": 50},
            batch_size=batch_size,
            rate_control=_safety_net(algo),
        )
    return rows


def _resnet_solver_for(algorithm: str):
    """The ResNet-18 demo's solver settings, kept identical here.

    ePC: eta 3e-4, 5 steps (the demo's stable point over 100 epochs).
    sPC: norm-clipped SGD, eta 0.2, 120 steps. Backprop reuses the sPC
    object only because the graph needs one to build.
    """
    if algorithm == "epc":
        return EPCInference(eta_infer=3e-4, infer_steps=5)
    return InferenceSGDNormClip(eta_infer=0.2, infer_steps=120, max_norm=1.0)


def _cifar10_resnet18_factory(algorithm: str) -> ModelFactory:
    """ResNet-18 with muPC parameterization and GELU, as in the demo."""

    def build(rng_key: jax.Array):
        structure = build_resnet18(
            weight_init=MuPCInitializer(),
            inference=_resnet_solver_for(algorithm),
            scaling=MuPCConfig(include_output=False),
            output_weight_init=XavierInitializer(),
            activation=GeluActivation(),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _cifar10_resnet18_family() -> Dict[str, BenchmarkRow]:
    # Batch size, learning rate, weight decay, and epoch count follow the
    # demo's 100-epoch runs. No augmentation yet; see the design doc.
    rows = {}
    batch_size = 256
    for algo in ALGORITHMS:
        row_id = f"cifar10-resnet18-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="cifar10",
            model="resnet18",
            algorithm=algo,
            model_factory=_cifar10_resnet18_factory(algo),
            loader_factory=_cifar10_loaders(batch_size),
            optimizer_factory=lambda total_steps: optax.adamw(1e-3, weight_decay=0.01),
            train_config={"num_epochs": 100},
            batch_size=batch_size,
            rate_control=_safety_net(algo),
        )
    return rows


# The transformer demo's tuned character-level settings (val perplexity 12.2).
_CHAR_TRANSFORMER = {
    "embed_dim": 64,
    "num_heads": 8,
    "mlp_dim": 256,
    "depth": 2,
    "seq_len": 128,
    "vocab_size": 65,  # Tiny Shakespeare's character set
    "batch_size": 16,
    "num_epochs": 5,
    "infer_steps": 12,
    "lr": 0.00012108621644524519,
    "eta_infer": 0.0174852165627398,
    "weight_init_std": 0.015166293102182283,
    "max_norm": 5.0,
}


def _transformer_solver_for(algorithm: str):
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGDNormClip(
        eta_infer=_CHAR_TRANSFORMER["eta_infer"],
        infer_steps=_CHAR_TRANSFORMER["infer_steps"],
        max_norm=_CHAR_TRANSFORMER["max_norm"],
        latent_decay=0.0,
    )


def _char_transformer_lr(total_steps: int) -> optax.Schedule:
    """Cosine decay from the tuned rate to a tenth of it, as the demo does."""
    return optax.cosine_decay_schedule(
        init_value=_CHAR_TRANSFORMER["lr"],
        decay_steps=max(1, total_steps),
        alpha=0.1,
    )


def _char_transformer_factory(algorithm: str) -> ModelFactory:
    def build(rng_key: jax.Array):
        c = _CHAR_TRANSFORMER
        structure = create_deep_transformer(
            depth=c["depth"],
            embed_dim=c["embed_dim"],
            num_heads=c["num_heads"],
            mlp_dim=c["mlp_dim"],
            seq_len=c["seq_len"],
            vocab_size=c["vocab_size"],
            inference=_transformer_solver_for(algorithm),
            weight_init={"type": "normal", "std": c["weight_init_std"]},
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _tinyshakespeare_loaders(batch_size: int, seq_len: int) -> LoaderFactory:
    def build(seed: int):
        from fabricpc.utils.data.dataloader import CharDataLoader

        train = CharDataLoader(
            "train", seq_len=seq_len, batch_size=batch_size, shuffle=True, seed=seed
        )
        test = CharDataLoader(
            "test", seq_len=seq_len, batch_size=batch_size, shuffle=False
        )
        return train, test

    return build


def _tinyshakespeare_transformer_family() -> Dict[str, BenchmarkRow]:
    # The tuned char-level settings of examples/transformer_v2_demo.py
    # (CHAR_DEFAULTS, val perplexity 12.22 there), including its cosine
    # learning-rate decay and weight init. Scored on perplexity (lower is
    # better); accuracy is also recorded.
    rows = {}
    c = _CHAR_TRANSFORMER
    for algo in ALGORITHMS:
        row_id = f"tinyshakespeare-transformer-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="tinyshakespeare",
            model="transformer",
            algorithm=algo,
            model_factory=_char_transformer_factory(algo),
            loader_factory=_tinyshakespeare_loaders(c["batch_size"], c["seq_len"]),
            optimizer_factory=lambda total_steps: optax.adam(
                _char_transformer_lr(total_steps), eps=_DEEP_ADAM_EPS
            ),
            train_config={"num_epochs": c["num_epochs"]},
            batch_size=c["batch_size"],
            rate_control=_safety_net(algo),
            metric="perplexity",
        )
    return rows


def _mnist_autoencoder_factory(algorithm: str) -> ModelFactory:
    """784 -> 128 -> 32 -> 128 -> 784, ReLU inside, sigmoid out.

    The pixels are in [0, 1] and the output node's Gaussian energy is the
    squared reconstruction error, so backprop minimizes MSE and PC settles
    on the same quantity.
    """

    def build(rng_key: jax.Array):
        pixels = IdentityNode(shape=(784,), name="pixels")
        sizes = (("enc", 128), ("code", 32), ("dec", 128))
        hidden = [
            Linear(
                shape=(size,),
                activation=ReLUActivation(),
                name=name,
                weight_init=XavierInitializer(),
            )
            for name, size in sizes
        ]
        recon = Linear(
            shape=(784,),
            activation=SigmoidActivation(),
            name="recon",
            weight_init=XavierInitializer(),
        )
        chain = [pixels, *hidden, recon]
        structure = graph(
            nodes=chain,
            edges=[
                Edge(source=a, target=b.slot("in")) for a, b in zip(chain, chain[1:])
            ],
            task_map=TaskMap(x=pixels, y=recon),
            inference=_solver_for(algorithm),
        )
        return initialize_params(structure, rng_key), structure

    return build


class _AsReconstruction:
    """Wraps an (image, label) loader so each batch is (image, image)."""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            yield batch[0], batch[0]

    def __len__(self):
        return len(self._loader)


def _mnist_reconstruction_loaders(batch_size: int) -> LoaderFactory:
    """MNIST with pixels left in [0, 1] (no mean/std normalization), so the
    sigmoid output can match them."""

    def build(seed: int):
        from fabricpc.utils.data import dataloader

        def make(split, shuffle):
            return _AsReconstruction(
                dataloader.MnistLoader(
                    split,
                    batch_size=batch_size,
                    tensor_format="flat",
                    shuffle=shuffle,
                    seed=seed,
                    normalize_mean=0.0,
                    normalize_std=1.0,
                )
            )

        return make("train", True), make("test", False)

    return build


def _mnist_autoencoder_family() -> Dict[str, BenchmarkRow]:
    """The faculty advisor's story: an autoencoder under backprop and PC,
    compared on reconstruction and hidden sparsity rather than accuracy."""
    from fabricpc.bench.definitions.metrics import reconstruction_mse, sparsity

    scores = {
        "reconstruction_mse": reconstruction_mse,
        "code_sparsity": sparsity(("code",)),
        "hidden_sparsity": sparsity(("enc", "code", "dec")),
        "target_energy": training_metrics.target_energy,
    }
    rows = {}
    batch_size = 200
    for algo in ALGORITHMS:
        row_id = f"mnist-autoencoder-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="mnist",
            model="autoencoder",
            algorithm=algo,
            model_factory=_mnist_autoencoder_factory(algo),
            loader_factory=_mnist_reconstruction_loaders(batch_size),
            optimizer_factory=lambda total_steps: optax.adam(1e-3),
            train_config={"num_epochs": 20},
            batch_size=batch_size,
            rate_control=_safety_net(algo),
            tier=2,
            metric="reconstruction_mse",
            eval_metrics=scores,
        )
    return rows


def _hopfield_family() -> Dict[str, BenchmarkRow]:
    """Associative memory: recall a stored pattern from a noisy copy
    (fabricpc.bench.definitions.rows_hopfield)."""
    from fabricpc.bench.definitions.rows_hopfield import hopfield_family

    return hopfield_family()


def _deep_convnet_families() -> Dict[str, BenchmarkRow]:
    """cifar10-vgg7, cifar10-vgg9 and cifar10-resnet18lean
    (fabricpc.bench.definitions.rows_deep_convnets)."""
    from fabricpc.bench.definitions.rows_deep_convnets import deep_convnet_families

    return deep_convnet_families()


ROWS: Dict[str, BenchmarkRow] = {}
ROWS.update(_mlp_family("mnist"))
ROWS.update(_mlp_family("fashionmnist"))
ROWS.update(_cifar10_vgg5_family())
ROWS.update(_cifar10_resnet18_family())
ROWS.update(cifar100_vgg5_family())
ROWS.update(tinyimagenet_vgg5_family())
ROWS.update(_tinyshakespeare_transformer_family())
ROWS.update(tinyshakespeare_bpe_transformer_family())
ROWS.update(_mnist_autoencoder_family())
ROWS.update(_hopfield_family())
ROWS.update(_deep_convnet_families())
ROWS.update(fcresnet_depth_family())


@dataclass(frozen=True)
class Comparison:
    """The three algorithms on one model family, and the pairs we test.

    Rows of a family share a dataset, a model, and the seeds, so their
    trials pair up one to one. Each contrast is ``(a, b)``, read as a - b.
    """

    id: str
    rows: Tuple[str, ...]
    contrasts: Tuple[Tuple[str, str], ...]
    metric: str = "accuracy"  # compared on this unless told otherwise


def _family_comparison(family: str) -> Comparison:
    spc, epc, bp = (f"{family}-{algo}" for algo in ALGORITHMS)
    return Comparison(
        id=family,
        rows=(spc, epc, bp),
        # Does PC keep up with backprop, and is ePC better than sPC?
        contrasts=((spc, bp), (epc, bp), (epc, spc)),
        metric=ROWS[bp].metric if bp in ROWS else "accuracy",
    )


COMPARISONS: Dict[str, Comparison] = {
    family: _family_comparison(family)
    for family in dict.fromkeys(row_id.rsplit("-", 1)[0] for row_id in ROWS)
}
