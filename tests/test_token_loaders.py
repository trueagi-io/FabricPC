"""Tests for the shared _TokenSequenceLoader base: sliding-window batching and
the integer-target contract. BPE test is tokenizers-gated."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from fabricpc.utils.data.dataloader import _TokenSequenceLoader, BpeDataLoader


class _ArrayTokenLoader(_TokenSequenceLoader):
    """Minimal concrete loader over an in-memory token array."""

    def __init__(
        self,
        data,
        vocab_size,
        seq_len,
        batch_size,
        shuffle=False,
        seed=0,
        max_samples=None,
    ):
        self.data = np.asarray(data, dtype=np.int32)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._init_sequence_indexing(max_samples)


def test_sequence_indexing_counts():
    loader = _ArrayTokenLoader(np.arange(100), vocab_size=100, seq_len=8, batch_size=4)
    assert loader.num_sequences == 100 - 8
    assert len(loader) == (100 - 8) // 4


def test_max_samples_caps_sequences():
    loader = _ArrayTokenLoader(
        np.arange(1000), vocab_size=50, seq_len=8, batch_size=4, max_samples=20
    )
    assert loader.num_sequences == 20
    assert len(loader) == 20 // 4


def test_yields_integer_targets_not_onehot():
    # Targets are integer ids (batch, seq_len); one-hot happens in
    # the training step, not the loader.
    loader = _ArrayTokenLoader(np.arange(50), vocab_size=50, seq_len=5, batch_size=2)
    x, y = next(iter(loader))
    assert x.dtype == np.int32 and x.shape == (2, 5)
    assert y.dtype == np.int32 and y.shape == (2, 5)


def test_next_token_alignment():
    data = np.arange(50, dtype=np.int32)
    loader = _ArrayTokenLoader(
        data, vocab_size=50, seq_len=5, batch_size=2, shuffle=False
    )
    x, y = next(iter(loader))
    assert np.array_equal(x[0], data[0:5])
    assert np.array_equal(y[0], data[1:6])
    assert np.array_equal(x[1], data[1:6])
    assert np.array_equal(y[1], data[2:7])


def test_drops_incomplete_last_batch():
    loader = _ArrayTokenLoader(
        np.arange(20), vocab_size=20, seq_len=4, batch_size=3, shuffle=False
    )
    batches = list(loader)
    assert len(batches) == (20 - 4) // 3
    assert all(x.shape[0] == 3 for x, _ in batches)


def test_bpe_dataloader_from_cache(tmp_path):
    """BpeDataLoader loads from a pre-built cache and yields
    integer targets. Skipped if the optional `tokenizers` dep is absent."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    bpe_dir = tmp_path / "bpe"
    bpe_dir.mkdir()

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("to be or not to be that is the question " * 50, encoding="utf-8")
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.train([str(corpus)], BpeTrainer(vocab_size=60, special_tokens=["[UNK]"]))
    tok.save(str(bpe_dir / "tokenizer.json"))

    ids = np.tile(
        np.array(
            tok.encode("to be or not to be that is the question").ids, dtype=np.int32
        ),
        20,
    )
    for split in ("train", "validation", "test"):
        np.save(bpe_dir / f"{split}.npy", ids)

    loader = BpeDataLoader(
        "train", seq_len=4, batch_size=2, shuffle=False, bpe_data_dir=str(bpe_dir)
    )
    assert loader.vocab_size == tok.get_vocab_size()
    assert loader.num_sequences == ids.size - 4  # len(data) - seq_len
    x, y = next(iter(loader))
    assert x.dtype == np.int32 and x.shape == (2, 4)
    assert y.dtype == np.int32 and y.shape == (2, 4)
    assert isinstance(loader.decode(x[0]), str)
