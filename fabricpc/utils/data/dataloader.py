import numpy as np
from fabricpc.utils.data.data_utils import one_hot, split_np_seed


class MnistLoader:
    """JAX-compatible data loader using TensorFlow Datasets.

    Provides the same interface as PyTorch DataLoader but uses tfds
    data parallelism based on C++ that bypasses GIL and does not inherit GPU state.
    Avoids os.fork warnings with JAX.

    Args:
        split: Dataset split to load. Use 'train' for training data or
               'test' for test data. Also supports slicing syntax like
               'train[:80%]' or 'train[80%:]' for custom splits.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility. When set, ensures deterministic
              shuffling across runs and machines. If None, shuffling is random.
        normalize_mean: Mean for normalization (default: MNIST mean).
        normalize_std: Std for normalization (default: MNIST std).
    """

    def __init__(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = None,
        tensor_format: str = "NHWC",  # image tensor 'flat' or 'NHWC' batch-height-width-channels
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
    ):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        # Disable GPU for TensorFlow (we only use it for data loading)
        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        # Split seed into two independent seeds for file and buffer shuffling
        file_seed, buffer_seed = split_np_seed(seed, n=2)

        # Configure read options for reproducibility
        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,  # Sequential reading for determinism
        )

        # Load dataset with pinned version for cross-machine reproducibility
        ds, info = tfds.load(
            "mnist:3.0.1",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        # Build pipeline
        if shuffle:
            ds = ds.shuffle(
                buffer_size=self.num_examples, seed=buffer_seed
            )  # mnist fits in memory (~60MB) so the buffer is the full dataset
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.ds = ds

    def __iter__(self):
        for images, labels in self.ds:
            # Convert to numpy, normalize, and flatten
            images = images.numpy().astype(np.float32) / 255.0
            images = (images - self.normalize_mean) / self.normalize_std

            # images shape is (Batch, 28, 28, 1)
            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)  # Flatten to (Batch, 784)

            # One-hot encode labels
            labels = one_hot(labels.numpy(), num_classes=10)

            yield images, labels

    def __len__(self):
        return self._num_batches


class Cifar100Loader:
    """JAX-compatible CIFAR-100 data loader using TensorFlow Datasets.

    Loads the CIFAR-100 dataset (32x32 RGB images, 100 fine-grained classes)
    and yields batches of (images, one_hot_labels).

    Args:
        split: Dataset split to load ('train' or 'test').
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility.
        tensor_format: 'NHWC' for (batch, 32, 32, 3) or 'flat' for (batch, 3072).
        normalize_mean: Per-channel mean for normalization (default: CIFAR-100 mean).
        normalize_std: Per-channel std for normalization (default: CIFAR-100 std).
    """

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
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = np.array(normalize_mean, dtype=np.float32)
        self.normalize_std = np.array(normalize_std, dtype=np.float32)

        file_seed, buffer_seed = split_np_seed(seed, n=2)

        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,
        )

        ds, info = tfds.load(
            "cifar100",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        if shuffle:
            ds = ds.shuffle(buffer_size=self.num_examples, seed=buffer_seed)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.ds = ds

    def __iter__(self):
        for images, labels in self.ds:
            images = images.numpy().astype(np.float32) / 255.0
            # Per-channel normalization: (H, W, C) broadcast
            images = (images - self.normalize_mean) / self.normalize_std

            if self.tensor_format == "flat":
                images = images.reshape(images.shape[0], -1)

            labels = one_hot(labels.numpy(), num_classes=100)

            yield images, labels

    def __len__(self):
        return self._num_batches


class CharDataLoader:
    """JAX-compatible character-level dataloader using TFDS.

    Loads the tiny_shakespeare dataset from TensorFlow Datasets and
    yields batches of (x_indices, y_onehot) for next-character prediction.

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
        self._epoch = 0

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

        # Each sequence needs seq_len input chars + 1 target char
        self.num_sequences = len(self.data) - seq_len
        if max_samples is not None:
            self.num_sequences = min(self.num_sequences, max_samples)
        self._num_batches = self.num_sequences // batch_size

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

            x = np.stack(
                [self.data[i : i + self.seq_len] for i in batch_idx]
            )  # (batch, seq_len) int32
            y_idx = np.stack(
                [self.data[i + 1 : i + self.seq_len + 1] for i in batch_idx]
            )  # (batch, seq_len)
            y_onehot = np.eye(self.vocab_size, dtype=np.float32)[y_idx]

            yield x, y_onehot

    def __len__(self):
        return self._num_batches

    def decode(self, indices) -> str:
        """Convert an array of character indices back to a string."""
        return "".join(self.idx_to_char[int(i)] for i in indices)


class FashionMnistLoader:
    """JAX-compatible Fashion-MNIST data loader using TensorFlow Datasets.

    Drop-in replacement for MnistLoader with the Fashion-MNIST dataset
    (28x28 grayscale, 10 clothing categories).

    Args:
        split: Dataset split ('train' or 'test').
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data each epoch.
        seed: Random seed for reproducibility.
        tensor_format: 'NHWC' for (batch, 28, 28, 1) or 'flat' for (batch, 784).
        normalize_mean: Mean for normalization (default: Fashion-MNIST mean).
        normalize_std: Std for normalization (default: Fashion-MNIST std).
    """

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
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        file_seed, buffer_seed = split_np_seed(seed, n=2)

        read_config = tfds.ReadConfig(
            shuffle_seed=file_seed,
            interleave_cycle_length=1,
        )

        ds, info = tfds.load(
            "fashion_mnist",
            split=split,
            with_info=True,
            as_supervised=True,
            read_config=read_config,
            shuffle_files=shuffle and seed is not None,
        )
        self.num_examples = info.splits[split].num_examples
        self._num_batches = (self.num_examples + batch_size - 1) // batch_size

        if shuffle:
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

            labels = one_hot(labels.numpy(), num_classes=10)

            yield images, labels

    def __len__(self):
        return self._num_batches


class FewShotLoader:
    """Class-balanced K-shot data loader using TensorFlow Datasets.

    Loads a full dataset, subsamples exactly K examples per class
    (deterministically via seed), and yields shuffled minibatches.
    Both arms in an A/B experiment receive identical training data
    when given the same seed.

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
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.num_classes = num_classes
        self._epoch = 0

        # Load entire dataset into memory
        ds = tfds.load(dataset_name, split=split, as_supervised=True)
        all_images = []
        all_labels = []
        for img, label in ds:
            all_images.append(img.numpy())
            all_labels.append(int(label.numpy()))

        all_images = np.array(all_images, dtype=np.float32) / 255.0
        all_images = (all_images - self.normalize_mean) / self.normalize_std
        all_labels = np.array(all_labels, dtype=np.int32)

        # Class-balanced subsampling
        rng = np.random.default_rng(seed)
        selected_indices = []
        for c in range(num_classes):
            class_indices = np.where(all_labels == c)[0]
            if len(class_indices) < k_per_class:
                chosen = class_indices  # use all if fewer than K
            else:
                chosen = rng.choice(class_indices, size=k_per_class, replace=False)
            selected_indices.append(chosen)
        selected_indices = np.concatenate(selected_indices)

        self.images = all_images[selected_indices]
        self.labels = all_labels[selected_indices]
        self.num_samples = len(selected_indices)
        self._num_batches = self.num_samples // batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            epoch_seed = (
                self.seed + 10000 + self._epoch if self.seed is not None else None
            )
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
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
