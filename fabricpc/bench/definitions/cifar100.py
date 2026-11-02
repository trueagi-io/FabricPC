"""The CIFAR-100 VGG-5 rows: cifar100-vgg5-{spc,epc,backprop}.

The same recipe as the CIFAR-10 VGG-5 rows in ``registry`` (fused conv and
pool nodes, flips and 4-pixel crops, warmup then cosine decay, Adam epsilon
1e-12, the safety net on the PC rows), with 100 classes and the settings
pcx shipped for CIFAR-100.

Where the numbers come from: pcx's VGG-5 CIFAR-100 configs, on the
``build-docs`` branch of github.com/liukidar/pcx under
``examples/s4_1_discriminative_mode/VGG5/cifar100/``. Our rows clamp the
output to the label and use cross-entropy, which is pcx's PC-CE method, so
we take ``VGG5_PCN_CE_cifar100.yaml``:

    T = 12, act_fn = hard_tanh, batch 128, 50 epochs
    weights: AdamW, lr 0.00021550147504766553, wd 0.007599378216624812
    states:  SGD, lr 0.011510981082033004, momentum 0.25

What pcx reports (arXiv 2407.01163v2, Table 1, VGG-5, CIFAR-100 top-1):
67.19 +- 0.24 for PC with centered nudging, the best PC number, but that
method is not one of ours and pcx ships no config for it. The methods closest
to our rows score 60.00 +- 0.19 (PC-CE) and 60.82 +- 0.10 (backprop with
cross-entropy). None of these is used as an expected score: like the
CIFAR-10 rows, these rows get one from their own first full GPU run.

This lives in its own module so the registry only needs one line for it.
Its imports from the registry happen inside the functions, because the
registry imports this module while it is still being built.
"""

from typing import Dict

import jax
import optax

from fabricpc.core.activations import HardTanhActivation
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.inference_epc import EPCInference
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import create_vgg

# pcx tuned each method on its own for CIFAR-100, so each row uses pcx's
# own weight settings, rounded like the CIFAR-10 rows' ones. PC comes from
# VGG5_PCN_CE_cifar100.yaml; its weight decay is about 600 times CIFAR-10's
# (1.2e-5). Backprop comes from VGG5_BP_CE_cifar100.yaml, which uses a much
# smaller decay (2.2e-5) and a slightly lower rate.
_PC_PEAK_LR = 2.2e-4
_PC_WEIGHT_DECAY = 7.6e-3
_BP_PEAK_LR = 2.1e-4
_BP_WEIGHT_DECAY = 2.2e-5
_WARMUP_FRACTION = 0.1

# pcx settles for 12 steps here (8 on CIFAR-10). Its state optimizer is SGD
# at 0.0115 with momentum 0.25; InferenceSGD has no momentum, and with it
# the step works out to about 0.0115 / (1 - 0.25) = 0.015, the same rate the
# CIFAR-10 rows use, so we keep 0.015.
_INFER_STEPS = 12
_ETA_INFER = 0.015


def _solver_for(algorithm: str):
    """Inference settings. Backprop never settles, but the graph still needs
    a solver to build, so it gets the sPC one."""
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGD(eta_infer=_ETA_INFER, infer_steps=_INFER_STEPS)


def _model_factory(algorithm: str):
    """VGG-5 on 32x32x3 images with 100 classes, pool fused into each conv
    node like the CIFAR-10 rows, and pcx's hard tanh (clip to [-1, 1])."""

    def build(rng_key: jax.Array):
        structure = create_vgg(
            5,
            input_shape=(32, 32, 3),
            num_classes=100,
            inference=_solver_for(algorithm),
            activation=HardTanhActivation(),
            fuse_pool=True,
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _loaders(batch_size: int):
    """CIFAR-100 loaders for one trial: training images get random flips and
    4-pixel crops, as in pcx; test images are left alone."""

    def build(seed: int):
        from fabricpc.utils.data import dataloader

        train = dataloader.AugmentedImageLoader(
            dataloader.Cifar100Loader(
                "train", batch_size=batch_size, shuffle=True, seed=seed
            ),
            flip_prob=0.5,
            crop_pad=4,
            seed=seed,
        )
        test = dataloader.Cifar100Loader("test", batch_size=batch_size, shuffle=False)
        return train, test

    return build


def _weight_settings(algorithm: str):
    """(peak learning rate, weight decay) for a row's weights."""
    if algorithm == "backprop":
        return _BP_PEAK_LR, _BP_WEIGHT_DECAY
    return _PC_PEAK_LR, _PC_WEIGHT_DECAY


def _lr_schedule(total_steps: int, peak_lr: float) -> optax.Schedule:
    """Same shape as the CIFAR-10 rows: up from zero over the first tenth,
    then a cosine down to zero. (pcx starts at the rate, peaks at 1.1 times
    it and ends at a tenth; we keep one shape across our VGG rows.)"""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=max(1, int(total_steps * _WARMUP_FRACTION)),
        decay_steps=max(2, total_steps),
        end_value=0.0,
    )


def _optimizer_for(algorithm: str):
    """The row's optimizer factory, with that method's pcx weight settings."""
    peak_lr, weight_decay = _weight_settings(algorithm)

    def make(total_steps: int) -> optax.GradientTransformation:
        from fabricpc.bench.registry import _DEEP_ADAM_EPS

        return optax.adamw(
            _lr_schedule(total_steps, peak_lr),
            weight_decay=weight_decay,
            eps=_DEEP_ADAM_EPS,
        )

    return make


def cifar100_vgg5_family() -> Dict[str, object]:
    """The three cifar100-vgg5 rows, keyed by row id."""
    from fabricpc.bench.definitions.registry import ALGORITHMS, BenchmarkRow, _safety_net

    rows = {}
    batch_size = 128
    for algo in ALGORITHMS:
        row_id = f"cifar100-vgg5-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="cifar100",
            model="vgg5",
            algorithm=algo,
            model_factory=_model_factory(algo),
            loader_factory=_loaders(batch_size),
            optimizer_factory=_optimizer_for(algo),
            train_config={"num_epochs": 50},
            batch_size=batch_size,
            rate_control=_safety_net(algo),
        )
    return rows
