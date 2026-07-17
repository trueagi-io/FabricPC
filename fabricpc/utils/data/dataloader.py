import numpy as np
from fabricpc.utils.data.data_utils import one_hot, split_np_seed

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    _TOKENIZERS_AVAILABLE = True
except ImportError:  # optional dependency (BPE only)
    _TOKENIZERS_AVAILABLE = False

_TOKENIZERS_INSTALL_HINT = (
    "BpeDataLoader requires the 'tokenizers' package. "
    "Install the text-data extras with: pip install -e '.[tfds]'"
)


class _TfdsImageLoader:
    """Shared base for TFDS image-classification loaders.

    Loads a tfds split, builds a shuffle/batch/prefetch tf.data pipeline, and
    yields (normalized float32 images, one-hot labels) numpy batches. Uses
    tfds data parallelism based on C++ that bypasses GIL and does not inherit
    GPU state; avoids os.fork warnings with JAX.

    Subclasses set two class attributes:
        _DATASET_NAME: tfds dataset name (version-pinned where needed).
        _NUM_CLASSES: number of classes for one-hot labels.

    Iteration statefulness: with ``shuffle=True`` the tf.data pipeline
    reshuffles on every pass, so the batch order advances with each epoch of
    one training run. Construct a fresh instance per logical training run;
    paired experiment runners obtain per-arm instances by calling their
    ``data_loader_factory`` once per arm.

    Args:
        split: Dataset split to load. Use 'train' for training data or
               'test' for test data. Also supports slicing syntax like
               'train[:80%]' or 'train[80%:]' for custom splits.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility. When set, ensures deterministic
              shuffling across runs and machines. If None, shuffling is random.
        tensor_format: 'NHWC' for image tensors or 'flat' for flattened rows.
        normalize_mean: Mean for normalization (scalar or per-channel tuple).
        normalize_std: Std for normalization (scalar or per-channel tuple).
    """

    _DATASET_NAME: str
    _NUM_CLASSES: int

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean=0.0,
        normalize_std=1.0,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        # Disable GPU for TensorFlow (we only use it for data loading)
        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = np.asarray(normalize_mean, dtype=np.float32)
        self.normalize_std = np.asarray(normalize_std, dtype=np.float32)

        # Split seed into two independent seeds for file and buffer shuffling
        file_seed, buffer_seed = split_np_seed(seed, n=2)

        # Configure read options for reproducibility
        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,  # Sequential reading for determinism
        )

        ds, info = tfds.load(
            self._DATASET_NAME,
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        if shuffle:
            # These datasets fit in memory, so the shuffle buffer is the
            # full split.
            ds = ds.shuffle(buffer_size=self.num_examples, seed=buffer_seed)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        self.ds = ds

    def __iter__(self):
        for images, labels in self.ds:
            images = images.numpy().astype(np.float32) / 255.0
            images = (images - self.normalize_mean) / self.normalize_std

            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)

            labels = one_hot(labels.numpy(), num_classes=self._NUM_CLASSES)

            yield images, labels

    def __len__(self):
        return self._num_batches


class MnistLoader(_TfdsImageLoader):
    """MNIST loader (28x28 grayscale, 10 digit classes).

    Yields (batch, 28, 28, 1) images ('NHWC') or (batch, 784) rows ('flat').
    Defaults normalize with the MNIST per-pixel mean/std. Dataset version is
    pinned for cross-machine reproducibility. See :class:`_TfdsImageLoader`
    for the shared constructor arguments and iteration contract.
    """

    _DATASET_NAME = "mnist:3.0.1"
    _NUM_CLASSES = 10

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
    ):
        super().__init__(
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            tensor_format=tensor_format,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )


class Cifar100Loader(_TfdsImageLoader):
    """CIFAR-100 loader (32x32 RGB, 100 fine-grained classes).

    Yields (batch, 32, 32, 3) images ('NHWC') or (batch, 3072) rows ('flat').
    Defaults normalize per channel with the CIFAR-100 mean/std. See
    :class:`_TfdsImageLoader` for the shared constructor arguments and
    iteration contract.
    """

    _DATASET_NAME = "cifar100"
    _NUM_CLASSES = 100

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean: tuple = (0.5071, 0.4867, 0.4408),
        normalize_std: tuple = (0.2675, 0.2565, 0.2761),
    ):
        super().__init__(
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            tensor_format=tensor_format,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )


class Cifar10Loader(_TfdsImageLoader):
    """CIFAR-10 loader (32x32 RGB, 10 classes).

    Yields (batch, 32, 32, 3) images ('NHWC') or (batch, 3072) rows ('flat').
    Defaults normalize per channel with the CIFAR-10 mean/std. See
    :class:`_TfdsImageLoader` for the shared constructor arguments and
    iteration contract.
    """

    _DATASET_NAME = "cifar10"
    _NUM_CLASSES = 10

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean: tuple = (0.4914, 0.4822, 0.4465),
        normalize_std: tuple = (0.2470, 0.2435, 0.2616),
    ):
        super().__init__(
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            tensor_format=tensor_format,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )


class _TokenSequenceLoader:
    """Shared sliding-window next-token batching for token-level loaders.

    Subclasses load their corpus into `self.data` (1D int32 token ids) and set
    `self.vocab_size`, `self.seq_len`, `self.batch_size`, `self.shuffle`, and
    `self.seed`, then call `self._init_sequence_indexing(max_samples)`. This base
    provides the identical batching for both char and BPE:
        x = data[i : i+seq_len],  y = data[i+1 : i+seq_len+1]

    Targets are yielded as integer token ids (batch, seq_len) int32. One-hot
    encoding to (batch, seq_len, vocab_size) for the CrossEntropy output node
    happens inside the training/eval step (compute_loss and the PC y-clamp),
    not here. This keeps the host->device transfer int32 and avoids a large
    one-hot per batch for big vocabularies (~96 MB at 16x128x11711 for BPE; the
    int ids are ~8 KB).
    """

    def _init_sequence_indexing(self, max_samples):
        # Each sequence needs seq_len input tokens + 1 shifted target token.
        self.num_sequences = len(self.data) - self.seq_len
        if max_samples is not None:
            self.num_sequences = min(self.num_sequences, max_samples)
        self._num_batches = self.num_sequences // self.batch_size
        self._epoch = 0

    def __iter__(self):
        indices = np.arange(self.num_sequences)
        if self.shuffle:
            epoch_seed = self.seed + self._epoch if self.seed is not None else None
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if len(batch_idx) < self.batch_size:
                continue  # drop incomplete last batch

            x = np.stack([self.data[i : i + self.seq_len] for i in batch_idx])
            y = np.stack([self.data[i + 1 : i + self.seq_len + 1] for i in batch_idx])
            # Yield integer target ids; one-hot happens in the training/eval step
            # to keep the host->device transfer int32 (see class docstring).
            yield x, y

    def __len__(self):
        return self._num_batches


class CharDataLoader(_TokenSequenceLoader):
    """JAX-compatible character-level dataloader using TFDS.

    Loads the tiny_shakespeare dataset from TensorFlow Datasets and
    yields batches of (x_indices, y_indices) for next-character prediction.
    Targets are integer ids; one-hot encoding happens in the training/eval
    step (see _TokenSequenceLoader).

    The vocabulary is always built from the train split to ensure consistent
    char-to-index mappings across all splits.

    Args:
        split: Dataset split ('train', 'validation', or 'test').
        seq_len: Number of characters per input sequence.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sequence start positions each epoch.
        seed: Random seed for reproducible shuffling.
        max_samples: If set, cap the number of sequences to this value.
            Useful for fast hyperparameter tuning on a subset of data.
    """

    # Class-level cache for vocabulary (built once from train split)
    _vocab = None

    def __init__(
        self,
        split: str,
        seq_len: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        max_samples: int = None,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Build vocabulary from the train split (cached across instances)
        if CharDataLoader._vocab is None:
            train_ds = tfds.load("tiny_shakespeare", split="train")
            train_text = next(iter(train_ds))["text"].numpy().decode("utf-8")
            chars = sorted(set(train_text))
            CharDataLoader._vocab = {
                "chars": chars,
                "vocab_size": len(chars),
                "char_to_idx": {ch: i for i, ch in enumerate(chars)},
                "idx_to_char": {i: ch for i, ch in enumerate(chars)},
            }

        self.chars = CharDataLoader._vocab["chars"]
        self.vocab_size = CharDataLoader._vocab["vocab_size"]
        self.char_to_idx = CharDataLoader._vocab["char_to_idx"]
        self.idx_to_char = CharDataLoader._vocab["idx_to_char"]

        # Load the requested split and encode to indices
        ds = tfds.load("tiny_shakespeare", split=split)
        text = next(iter(ds))["text"].numpy().decode("utf-8")
        self.data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)

        self._init_sequence_indexing(max_samples)

    def decode(self, indices) -> str:
        """Convert an array of character indices back to a string."""
        return "".join(self.idx_to_char[int(i)] for i in indices)


class BpeDataLoader(_TokenSequenceLoader):
    """JAX-compatible BPE tokenized data loader using TFDS.

    Loads pre-encoded Tiny Shakespeare BPE token sequences from .npy files
    and yields batches of (x_indices, y_indices) for next-token prediction.
    Targets are integer ids; one-hot encoding happens in the training/eval
    step (see _TokenSequenceLoader).

    The vocabulary is built from all splits of the Tiny Shakespeare dataset
    using a BPE tokenizer with vocab_size=11711. On first use, the tokenizer
    is trained and all splits are encoded and cached to bpe_data_dir.
    Subsequent runs load directly from cache.

    Args:
        split: Dataset split ('train', 'validation', or 'test').
        seq_len: Number of tokens per input sequence.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle sequence start positions each epoch.
        seed: Random seed for reproducible shuffling.
        max_samples: If set, cap the number of sequences to this value.
            Useful for fast hyperparameter tuning on a subset of data.
        bpe_data_dir: Directory to cache tokenizer and encoded splits.
        vocab_size: BPE vocabulary size (default: 11711).
    """

    splits = ["train", "validation", "test"]

    def __init__(
        self,
        split: str,
        seq_len: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        max_samples: int = None,
        bpe_data_dir: str = "data/bpe_tokenized",
        vocab_size: int = 11711,
        verbose: bool = True,
    ):
        from pathlib import Path

        if not _TOKENIZERS_AVAILABLE:
            raise ImportError(_TOKENIZERS_INSTALL_HINT)

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        bpe_dir = Path(bpe_data_dir)
        tokenizer_path = bpe_dir / "tokenizer.json"
        token_path = bpe_dir / f"{split}.npy"

        # Prepare tokenizer and encoded splits if not already cached
        if not tokenizer_path.exists() or not token_path.exists():
            self._prepare(bpe_dir, vocab_size=vocab_size, verbose=verbose)

        tok = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = tok.get_vocab_size()
        vocab = tok.get_vocab()
        self.token_to_idx = vocab
        self.idx_to_token = {v: k for k, v in vocab.items()}
        self.data = np.load(token_path)
        self._tok = tok

        self._init_sequence_indexing(max_samples)

    @staticmethod
    def _prepare(bpe_dir, vocab_size: int = 11711, verbose: bool = True):
        """Train BPE tokenizer on all splits and encode each split to .npy."""
        import tensorflow_datasets as tfds
        import tensorflow as tf
        from pathlib import Path

        tf.config.set_visible_devices([], "GPU")
        bpe_dir = Path(bpe_dir)
        bpe_dir.mkdir(parents=True, exist_ok=True)

        tokenizer_path = bpe_dir / "tokenizer.json"

        def load_split(split):
            ds = tfds.load("tiny_shakespeare", split=split)
            return next(iter(ds))["text"].numpy().decode("utf-8")

        # Train tokenizer on all splits for full vocabulary coverage
        if not tokenizer_path.exists():
            if verbose:
                print("BpeDataLoader: training BPE tokenizer on Tiny Shakespeare...")
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
            )
            # Write splits to temp files for training
            tmp_files = []
            for split in BpeDataLoader.splits:
                text = load_split(split)
                tmp = bpe_dir / f"_tmp_{split}.txt"
                tmp.write_text(text, encoding="utf-8")
                tmp_files.append(str(tmp))

            tokenizer.train(files=tmp_files, trainer=trainer)

            for f in tmp_files:
                Path(f).unlink()

            tokenizer.save(str(tokenizer_path))
            if verbose:
                print(f"BpeDataLoader: tokenizer saved to {tokenizer_path}")
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Encode and cache each split
        for split in BpeDataLoader.splits:
            token_path = bpe_dir / f"{split}.npy"
            if not token_path.exists():
                if verbose:
                    print(f"BpeDataLoader: encoding {split} split...")
                text = load_split(split)
                ids = tokenizer.encode(text).ids
                np.save(token_path, np.array(ids, dtype=np.int32))
                if verbose:
                    print(f"BpeDataLoader: {split} saved to {token_path}")

    def decode(self, indices) -> str:
        return self._tok.decode([int(i) for i in indices], skip_special_tokens=True)


class FashionMnistLoader(_TfdsImageLoader):
    """Fashion-MNIST loader (28x28 grayscale, 10 clothing categories).

    Drop-in replacement for MnistLoader. Yields (batch, 28, 28, 1) images
    ('NHWC') or (batch, 784) rows ('flat'). Defaults normalize with the
    Fashion-MNIST per-pixel mean/std. See :class:`_TfdsImageLoader` for the
    shared constructor arguments and iteration contract.
    """

    _DATASET_NAME = "fashion_mnist"
    _NUM_CLASSES = 10

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",
        normalize_mean: float = 0.2860,
        normalize_std: float = 0.3530,
    ):
        super().__init__(
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            tensor_format=tensor_format,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )


class FewShotLoader:
    """Class-balanced K-shot data loader using TensorFlow Datasets.

    Subsamples exactly K examples per class (deterministically via seed) and
    yields shuffled minibatches, including a final partial batch when the
    sample count is not a multiple of batch_size. Both arms in a paired
    experiment receive identical training data when given the same seed.

    The full split is loaded once per process and memoized in a class-level
    cache keyed on (dataset_name, split); constructions after the first
    reuse the cached raw arrays, so building one loader per arm per trial in
    paired experiments costs only the K-shot subsample.

    Iteration statefulness: with ``shuffle=True`` the shuffle order advances
    with each pass (epoch). Construct a fresh instance per logical training
    run; two same-seed instances yield identical epoch-shuffle streams.

    Args:
        dataset_name: TFDS dataset name (e.g., 'fashion_mnist', 'mnist:3.0.1').
        split: Dataset split ('train' or 'test').
        k_per_class: Number of examples to keep per class.
        batch_size: Number of samples per batch.
        num_classes: Number of classes in the dataset (default: 10).
        shuffle: Whether to shuffle the subsample each epoch.
        seed: Random seed for both subsampling and shuffling.
        tensor_format: 'NHWC' or 'flat'.
        normalize_mean: Mean for normalization.
        normalize_std: Std for normalization.
    """

    # (dataset_name, split) -> (raw images, int32 labels); raw dtype as
    # loaded (uint8 for the MNIST/CIFAR families), pre-normalization.
    _raw_split_cache = {}

    @classmethod
    def _load_raw_split(cls, dataset_name: str, split: str):
        key = (dataset_name, split)
        if key not in cls._raw_split_cache:
            import tensorflow_datasets as tfds
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")

            ds = tfds.load(dataset_name, split=split, as_supervised=True)
            all_images = []
            all_labels = []
            for img, label in ds:
                all_images.append(img.numpy())
                all_labels.append(int(label.numpy()))

            cls._raw_split_cache[key] = (
                np.asarray(all_images),
                np.asarray(all_labels, dtype=np.int32),
            )
        return cls._raw_split_cache[key]

    def __init__(
        self,
        dataset_name: str,
        split: str,
        k_per_class: int,
        batch_size: int,
        num_classes: int = 10,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "flat",
        normalize_mean: float = 0.2860,
        normalize_std: float = 0.3530,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.num_classes = num_classes
        self._epoch = 0

        raw_images, raw_labels = self._load_raw_split(dataset_name, split)

        # Class-balanced subsampling
        rng = np.random.default_rng(seed)
        selected_indices = []
        for c in range(num_classes):
            class_indices = np.where(raw_labels == c)[0]
            if len(class_indices) < k_per_class:
                chosen = class_indices  # use all if fewer than K
            else:
                chosen = rng.choice(class_indices, size=k_per_class, replace=False)
            selected_indices.append(chosen)
        selected_indices = np.concatenate(selected_indices)

        # Normalize only the selected subsample; the cached raw arrays stay
        # untouched for other constructions.
        images = raw_images[selected_indices].astype(np.float32) / 255.0
        self.images = (images - self.normalize_mean) / self.normalize_std
        self.labels = raw_labels[selected_indices]
        self.num_samples = len(selected_indices)
        self._num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            epoch_seed = (
                self.seed + 10000 + self._epoch if self.seed is not None else None
            )
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            images = self.images[batch_idx]

            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)

            labels = one_hot(self.labels[batch_idx], num_classes=self.num_classes)
            yield images, labels

    def __len__(self):
        return self._num_batches


class NoisyTestLoader:
    """Wrapper that adds Gaussian noise to a base loader's images at test time.

    Useful for evaluating noise robustness of trained models. Noise is
    applied after normalization, so noise_std is in normalized units.

    Args:
        base_loader: Any iterable loader yielding (images, labels) batches.
        noise_std: Standard deviation of Gaussian noise (0.0 = no noise).
        seed: Random seed for reproducible noise.
    """

    def __init__(self, base_loader, noise_std: float = 0.0, seed: int = None):
        self.base_loader = base_loader
        self.noise_std = noise_std
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for images, labels in self.base_loader:
            if self.noise_std > 0:
                noise = rng.normal(0, self.noise_std, images.shape).astype(np.float32)
                images = images + noise
            yield images, labels

    def __len__(self):
        return len(self.base_loader)
