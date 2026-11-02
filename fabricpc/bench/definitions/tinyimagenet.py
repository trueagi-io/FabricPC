"""The Tiny-ImageNet VGG-5 rows: tinyimagenet-vgg5-{spc,epc,backprop}.

The same recipe as the CIFAR VGG-5 rows (fused conv and pool nodes, warmup
then cosine decay, Adam epsilon 1e-12, the safety net on the PC rows), with
200 classes, 56x56 crops of the 64x64 images, and the settings pcx shipped
for Tiny-ImageNet.

Where the numbers come from: pcx's VGG-5 Tiny-ImageNet configs, on the
``build-docs`` branch of github.com/liukidar/pcx under
``examples/s4_1_discriminative_mode/VGG5/tinyimagenet/``. Our PC rows clamp
the output to the label and use cross-entropy, which is pcx's PC-CE method,
so they take ``VGG5_PCN_CE_tinyimagenet.yaml``; the backprop row takes
``VGG5_BP_CE_tinyimagenet.yaml``:

    PC-CE: T = 7, act_fn = hard_tanh, batch 128, 50 epochs
           weights: AdamW, lr 4.463151374486534e-05, wd 0.0014288405442168734
           states:  SGD, lr 0.010044339247287183, momentum 0.05
    BP-CE: act_fn = hard_tanh, batch 128, 50 epochs
           weights: AdamW, lr 8.709909364802776e-05, wd 2.0892453778883236e-05

Data, as in pcx's scripts: training images get a random left-right flip and
a random 56x56 window with no padding; the model is scored on the centre
56x56 of the 10,000 ``val`` images, because the official test split has no
labels (pcx's "test" numbers are val numbers too). Both use ImageNet's
channel mean and std. So VGG-5 sees 56 -> 28 -> 14 -> 7 -> 3, and the last
layer reads 3x3x512 features, the same shapes as pcx.

What pcx reports (arXiv 2407.01163v2, Table 1, VGG-5, Tiny-ImageNet, top-1
and top-5): the best number, 46.40 +- 0.1 top-1, is negative nudging (NN),
which is not one of our methods. The fair numbers for our rows are PC-CE,
41.29 +- 0.2 top-1 (66.68 top-5), and backprop with cross-entropy, 43.72 +-
0.1 top-1 (69.23 top-5). pcx keeps the best epoch of each run and averages
5 of 7 seeds; we score the last epoch. None of these is used as an expected
score: the rows get one from their own first full GPU run.

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

NUM_CLASSES = 200
IMAGE_SIZE = 64  # what the dataset stores
CROP_SIZE = 56  # what the model sees, as in pcx

# pcx tuned each method on its own, so each row uses pcx's own weight
# settings, rounded. PC (VGG5_PCN_CE_tinyimagenet.yaml) has half backprop's
# rate and about 70 times its weight decay.
_PC_PEAK_LR = 4.5e-5
_PC_WEIGHT_DECAY = 1.4e-3
_BP_PEAK_LR = 8.7e-5
_BP_WEIGHT_DECAY = 2.1e-5
_WARMUP_FRACTION = 0.1

# pcx settles for 7 steps here. Its state optimizer is SGD at 0.01004 with
# momentum 0.05; InferenceSGD has no momentum, and with it the step works out
# to about 0.01004 / (1 - 0.05) = 0.0106. That is lower than the CIFAR rows'
# 0.015, so we keep pcx's value rather than CIFAR's.
_INFER_STEPS = 7
_ETA_INFER = 0.0106


def _solver_for(algorithm: str):
    """Inference settings. Backprop never settles, but the graph still needs
    a solver to build, so it gets the sPC one."""
    if algorithm == "epc":
        return EPCInference()
    return InferenceSGD(eta_infer=_ETA_INFER, infer_steps=_INFER_STEPS)


def _model_factory(algorithm: str):
    """VGG-5 on 56x56x3 crops with 200 classes, pool fused into each conv
    node like the CIFAR rows, and pcx's hard tanh (clip to [-1, 1])."""

    def build(rng_key: jax.Array):
        structure = create_vgg(
            5,
            input_shape=(CROP_SIZE, CROP_SIZE, 3),
            num_classes=NUM_CLASSES,
            inference=_solver_for(algorithm),
            activation=HardTanhActivation(),
            fuse_pool=True,
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _loaders(batch_size: int):
    """Tiny-ImageNet loaders for one trial: training images get random flips
    and random 56x56 windows; val images are cut to their centre 56x56."""

    def build(seed: int):
        from fabricpc.utils.data import dataloader

        train = dataloader.AugmentedImageLoader(
            dataloader.TinyImageNetLoader(
                "train", batch_size=batch_size, shuffle=True, seed=seed
            ),
            flip_prob=0.5,
            crop_pad=0,
            crop_size=CROP_SIZE,
            seed=seed,
        )
        test = dataloader.TinyImageNetLoader(
            "val", batch_size=batch_size, shuffle=False, center_crop=CROP_SIZE
        )
        return train, test

    return build


def _weight_settings(algorithm: str):
    """(peak learning rate, weight decay) for a row's weights."""
    if algorithm == "backprop":
        return _BP_PEAK_LR, _BP_WEIGHT_DECAY
    return _PC_PEAK_LR, _PC_WEIGHT_DECAY


def _lr_schedule(total_steps: int, peak_lr: float) -> optax.Schedule:
    """Same shape as the CIFAR VGG rows: up from zero over the first tenth,
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


def tinyimagenet_vgg5_family() -> Dict[str, object]:
    """The three tinyimagenet-vgg5 rows, keyed by row id."""
    from fabricpc.bench.definitions.registry import ALGORITHMS, BenchmarkRow, _safety_net

    rows = {}
    batch_size = 128
    for algo in ALGORITHMS:
        row_id = f"tinyimagenet-vgg5-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="tinyimagenet",
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
