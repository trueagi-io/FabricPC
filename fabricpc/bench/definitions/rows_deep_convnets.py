"""The deeper CIFAR-10 conv rows: cifar10-vgg7, cifar10-vgg9 and
cifar10-resnet18lean, each as {spc, epc, backprop}.

They are here so they are ready to run on a GPU later; none of them has an
expected score yet (each gets one from its first full run).

VGG-7 and VGG-9 use the cifar10-vgg5 recipe unchanged: fused conv and pool
nodes (``create_vgg(..., fuse_pool=True)``), flips and 4-pixel crops, batch
128, 50 epochs, AdamW with the VGG-5 rate and decay, warmup then cosine
decay, Adam epsilon 1e-12, the same settling settings and the rate safety
net on the PC rows. Only the depth changes, so the three families make a
depth series.

The layer layouts need the sponsor's confirmation. ``create_vgg`` uses
3x3 "SAME" convs with a 2x2 max pool after every second conv:
VGG-7 is 128,128 | 256,256 | 512,512 and VGG-9 adds a 512,512 block. pcx
(arXiv 2407.01163, Table 5) lists VGG-7 with the same channels
[128, 128, 256, 256, 512, 512] but paddings [1, 1, 1, 0, 1, 0], so some of
its convs shrink the map and its pooling is not laid out exactly like
ours. pcx ships VGG-7 configs only for CIFAR-100 and Tiny ImageNet (none
for CIFAR-10), and no VGG-9 code at all, so VGG-9 here is our own layout.

Settling steps: we keep T = 8, VGG-5's value. pcx searched T in 8-12 for
VGG-7 and 9-18 for VGG-9 (Table 7), which hints that deeper nets want more
steps, but those are search ranges, not the values pcx picked. On our
VGG-5, T = 15 and 20 did not beat 8 over a full run, and a long settle
moves sPC's update further from backprop's during training (the cosine
fell from 0.85 at T = 8 to 0.46 at T = 150). But there is a warning sign. On one CIFAR-10 batch at
initialization (Mac check, 2026-09-30), sPC's weight update lined up with
backprop's (cosine near 1) only in the top three or four conv layers:

    VGG-5, T=8:  all 4 convs aligned except conv1 (cosine 0.63)
    VGG-7, T=8:  conv1-conv3 at cosine -0.03 to 0.15; T=12 fixes conv3 only
    VGG-9, T=8:  conv1-conv5 at cosine -0.05 to 0.09; T=12 fixes conv5 only

The layers below get updates 1e-5 to 1e-8 the size of backprop's, which
is float noise: the error shrinks by about 30 times per layer on its way
down, so more steps help little and a larger settling rate might help
more. We did not change anything on one batch at init; a GPU probe of the
settling rate or step count should come before a full sPC run, and any
change should apply to the spc rows only.

cifar10-resnet18lean is cifar10-resnet18 with ``build_resnet18(lean=True)``:
the eight residual adds and the final average pool are folded into
weighted nodes, so no PC latent is parameter-free (21 latents instead of
30, same parameters, same forward pass). Everything else is the plain
rows' recipe, so the two families can be compared directly. The plain
rows are unchanged.

A first look on the Mac (2026-10-02, one batch of 32 at initialization,
same weights, the sPC rows' solver): the lean layout's sPC update lined up
with backprop's (cosine above 0.5) in more of the 21 weight layers at
every step count we tried, 10 vs 7 at T = 8, 18 vs 12 at T = 30 and 13 vs
12 at T = 120 (mean cosine 0.48 vs 0.35, 0.71 vs 0.58, 0.60 vs 0.50). That
is one batch and one seed, not a training result.

This module is imported lazily by the registry, which it imports from.
"""

from typing import Dict

import jax
import optax

from fabricpc.graph_initialization import initialize_params

# Depth for each VGG family; both follow cifar10-vgg5 exactly otherwise.
VGG_DEPTHS = {"vgg7": 7, "vgg9": 9}


def _vgg_factory(depth: int, algorithm: str):
    """VGG-7 or VGG-9 on 32x32x3 images with 10 classes, pool fused into
    the last conv node of each block, like the VGG-5 rows."""
    from fabricpc.bench.definitions.registry import _vgg_solver_for
    from fabricpc.models import create_vgg

    def build(rng_key: jax.Array):
        structure = create_vgg(
            depth,
            input_shape=(32, 32, 3),
            num_classes=10,
            inference=_vgg_solver_for(algorithm),
            fuse_pool=True,
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _vgg_optimizer(total_steps: int) -> optax.GradientTransformation:
    from fabricpc.bench.definitions.registry import (
        _DEEP_ADAM_EPS,
        _VGG_WEIGHT_DECAY,
        _vgg_lr_schedule,
    )

    return optax.adamw(
        _vgg_lr_schedule(total_steps),
        weight_decay=_VGG_WEIGHT_DECAY,
        eps=_DEEP_ADAM_EPS,
    )


def deep_vgg_families() -> Dict[str, object]:
    """cifar10-vgg7-* and cifar10-vgg9-*, keyed by row id."""
    from fabricpc.bench.definitions.registry import (
        ALGORITHMS,
        BenchmarkRow,
        _cifar10_loaders,
        _safety_net,
    )

    rows = {}
    batch_size = 128
    for model, depth in VGG_DEPTHS.items():
        for algo in ALGORITHMS:
            row_id = f"cifar10-{model}-{algo}"
            rows[row_id] = BenchmarkRow(
                id=row_id,
                dataset="cifar10",
                model=model,
                algorithm=algo,
                model_factory=_vgg_factory(depth, algo),
                loader_factory=_cifar10_loaders(batch_size),
                optimizer_factory=_vgg_optimizer,
                train_config={"num_epochs": 50},
                batch_size=batch_size,
                tier=2,
                rate_control=_safety_net(algo),
            )
    return rows


def _resnet18_lean_factory(algorithm: str):
    """The cifar10-resnet18 graph (muPC, GELU, the demo's solvers) in the
    lean layout."""
    from fabricpc.bench.definitions.registry import _resnet_solver_for
    from fabricpc.core.activations import GeluActivation
    from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
    from fabricpc.core.mupc import MuPCConfig
    from fabricpc.models import build_resnet18

    def build(rng_key: jax.Array):
        structure = build_resnet18(
            weight_init=MuPCInitializer(),
            inference=_resnet_solver_for(algorithm),
            scaling=MuPCConfig(include_output=False),
            output_weight_init=XavierInitializer(),
            activation=GeluActivation(),
            lean=True,
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def resnet18_lean_family() -> Dict[str, object]:
    """cifar10-resnet18lean-*, keyed by row id. Same batch size, optimizer
    and epochs as cifar10-resnet18."""
    from fabricpc.bench.definitions.registry import (
        ALGORITHMS,
        BenchmarkRow,
        _cifar10_loaders,
        _safety_net,
    )

    rows = {}
    batch_size = 256
    for algo in ALGORITHMS:
        row_id = f"cifar10-resnet18lean-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="cifar10",
            model="resnet18lean",
            algorithm=algo,
            model_factory=_resnet18_lean_factory(algo),
            loader_factory=_cifar10_loaders(batch_size),
            optimizer_factory=lambda total_steps: optax.adamw(1e-3, weight_decay=0.01),
            train_config={"num_epochs": 100},
            batch_size=batch_size,
            tier=2,
            rate_control=_safety_net(algo),
        )
    return rows


def deep_convnet_families() -> Dict[str, object]:
    """All the rows in this module."""
    rows = deep_vgg_families()
    rows.update(resnet18_lean_family())
    return rows
