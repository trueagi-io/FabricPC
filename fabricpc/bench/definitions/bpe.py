"""The BPE Tiny Shakespeare transformer rows:
tinyshakespeare-bpe-transformer-{spc,epc,backprop}.

The same recipe as the char transformer rows in ``registry`` (the v2
builder, normal weight init, norm-clipped settling, cosine decay to a tenth
of the rate, Adam epsilon 1e-12, the safety net on the PC rows, scored on
perplexity), but on BPE tokens with the settings the sponsor's demo ships
for them: ``BPE_DEFAULTS`` in ``examples/transformer_v2_demo.py``.

    embed 128, 4 heads, MLP 512, 4 blocks, 64 tokens per sequence
    batch 32, 5 epochs, 23 settling steps, eta_infer 0.0656,
    lr 1.68e-5, weight init std 0.0399, norm clip 5.0

The one change from the demo is Adam's epsilon (1e-12 instead of 1e-8), for
the reason given at ``_DEEP_ADAM_EPS`` in the registry. The demo has no
dropout or weight decay and neither do these rows.

How to read the score: the demo's comment gives val perplexity about 1133
and test about 721, but those came from tuning on a 50k-sequence subset of
the training text, and it calls the setting a placeholder. These rows train
on all of it (about 240,900 sequences, roughly 4.8 times as much), so the
demo's numbers are a rough guide, not this row's setting. The yardstick is
guessing each token by how common it is in the training text (a unigram
model with add-one smoothing): about 790 on test and about 674 on val. The
rows are scored on test, so compare them with the test pair only: unigram
790 and the demo's 721. (The demo's val 1133 is about 1.7 times worse than
the val unigram, so it is not a fair number to hold a row to.) Per-token
perplexity also cannot be compared with the char rows, since a BPE token
here is about 4 characters.

The tokenizer is ``BpeDataLoader``'s own (HuggingFace ``tokenizers``, 11711
ids, trained on all three splits). It is trained once, in a few seconds, and
cached with the encoded splits in ``bpe_cache_dir()``, which is outside the
repo so the files never end up in git.

This lives in its own module so the registry only needs one line for it.
Its imports from the registry happen inside the functions, because the
registry imports this module while it is still being built.
"""

import os
from pathlib import Path
from typing import Dict

import jax
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.inference_epc import EPCInference
from fabricpc.graph_initialization import initialize_params
from fabricpc.models import create_deep_transformer

# examples/transformer_v2_demo.py BPE_DEFAULTS, plus the norm clip it builds
# its solver with.
_BPE_TRANSFORMER = {
    "embed_dim": 128,
    "num_heads": 4,
    "mlp_dim": 512,
    "depth": 4,
    "seq_len": 64,
    "vocab_size": 11711,  # BpeDataLoader's default vocabulary
    "batch_size": 32,
    "num_epochs": 5,
    "infer_steps": 23,
    "lr": 1.676456563307537e-05,
    "eta_infer": 0.06558512264378524,
    "weight_init_std": 0.039890499730518045,
    "max_norm": 5.0,
}

# Set this to keep the tokenizer cache somewhere else (a cloud disk, say).
BPE_DIR_ENV = "FABRICPC_BPE_DIR"


def bpe_cache_dir() -> Path:
    """Where the tokenizer and the encoded splits are kept.

    ``$FABRICPC_BPE_DIR`` if set, else ``~/.cache/fabricpc/bpe_tokenized``.
    The loader's own default (``data/bpe_tokenized``) is relative to where
    the command runs, so each folder you ran from would get its own copy.
    """
    chosen = os.environ.get(BPE_DIR_ENV)
    if chosen:
        return Path(chosen).expanduser()
    return Path.home() / ".cache" / "fabricpc" / "bpe_tokenized"


def _solver_for(algorithm: str):
    """ePC uses its defaults like the char rows. Backprop never settles, but
    the graph still needs a solver to build, so it gets the sPC one."""
    if algorithm == "epc":
        return EPCInference()
    c = _BPE_TRANSFORMER
    return InferenceSGDNormClip(
        eta_infer=c["eta_infer"],
        infer_steps=c["infer_steps"],
        max_norm=c["max_norm"],
        latent_decay=0.0,
    )


def _lr_schedule(total_steps: int) -> optax.Schedule:
    """Cosine decay from the demo's rate to a tenth of it, as the demo does."""
    return optax.cosine_decay_schedule(
        init_value=_BPE_TRANSFORMER["lr"],
        decay_steps=max(1, total_steps),
        alpha=0.1,
    )


def _optimizer(total_steps: int) -> optax.GradientTransformation:
    from fabricpc.bench.definitions.registry import _DEEP_ADAM_EPS

    return optax.adam(_lr_schedule(total_steps), eps=_DEEP_ADAM_EPS)


def _model_factory(algorithm: str):
    """The same v2 builder as the char rows, with the BPE sizes."""

    def build(rng_key: jax.Array):
        c = _BPE_TRANSFORMER
        structure = create_deep_transformer(
            depth=c["depth"],
            embed_dim=c["embed_dim"],
            num_heads=c["num_heads"],
            mlp_dim=c["mlp_dim"],
            seq_len=c["seq_len"],
            vocab_size=c["vocab_size"],
            inference=_solver_for(algorithm),
            weight_init={"type": "normal", "std": c["weight_init_std"]},
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return build


def _loaders(batch_size: int, seq_len: int):
    """BPE loaders for one trial. The first call trains the tokenizer and
    encodes the splits into ``bpe_cache_dir()``; later calls just load."""

    def build(seed: int):
        from fabricpc.utils.data import dataloader

        expected = _BPE_TRANSFORMER["vocab_size"]
        shared = {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "bpe_data_dir": str(bpe_cache_dir()),
            "vocab_size": expected,
            "verbose": False,
        }
        train = dataloader.BpeDataLoader("train", shuffle=True, seed=seed, **shared)
        test = dataloader.BpeDataLoader("test", shuffle=False, **shared)
        # The graph is built before the loaders, so a cache made with another
        # vocabulary would feed ids the output layer does not have.
        for loader in (train, test):
            if loader.vocab_size != expected:
                raise ValueError(
                    f"the BPE cache in {bpe_cache_dir()} has "
                    f"{loader.vocab_size} tokens, the rows expect {expected}; "
                    "delete that folder so it is rebuilt"
                )
        return train, test

    return build


def tinyshakespeare_bpe_transformer_family() -> Dict[str, object]:
    """The three tinyshakespeare-bpe-transformer rows, keyed by row id."""
    from fabricpc.bench.definitions.registry import ALGORITHMS, BenchmarkRow, _safety_net

    c = _BPE_TRANSFORMER
    rows = {}
    for algo in ALGORITHMS:
        row_id = f"tinyshakespeare-bpe-transformer-{algo}"
        rows[row_id] = BenchmarkRow(
            id=row_id,
            dataset="tinyshakespeare-bpe",
            model="transformer",
            algorithm=algo,
            model_factory=_model_factory(algo),
            loader_factory=_loaders(c["batch_size"], c["seq_len"]),
            optimizer_factory=_optimizer,
            train_config={"num_epochs": c["num_epochs"]},
            batch_size=c["batch_size"],
            rate_control=_safety_net(algo),
            # 7.5k steps an epoch with 23 settles each on an 11711-way
            # output: much heavier than the char rows.
            tier=2,
            metric="perplexity",
        )
    return rows
