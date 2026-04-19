"""
Split-MNIST Data Loading for Continual Learning.

Provides JAX-compatible data loaders for sequential task learning
on MNIST split by digit pairs.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterator, Sequence
import numpy as np

from fabricpc.utils.data.data_utils import one_hot


@dataclass
class TaskData:
    """Container for a single task's data loaders."""

    task_id: int
    classes: Tuple[int, int]  # The two digit classes for this task
    train_loader: "SplitMnistTaskLoader"
    test_loader: "SplitMnistTaskLoader"

    def __repr__(self) -> str:
        return f"TaskData(task_id={self.task_id}, classes={self.classes})"


class SplitMnistTaskLoader:
    """
    Data loader for a single Split-MNIST task (two digit classes).

    Provides JAX-compatible iteration over batches of (images, labels).

    Args:
        images: Array of shape (N, 28, 28, 1) or (N, 784)
        labels: Array of shape (N,) with integer labels
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle each epoch
        seed: Random seed for reproducibility
        tensor_format: "flat" (N, 784) or "NHWC" (N, 28, 28, 1)
        remap_labels: If True, remap class labels to 0/1 for binary classification
        num_classes: Number of output classes (10 for full MNIST, 2 for remapped)
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        tensor_format: str = "flat",
        remap_labels: bool = False,
        num_classes: int = 10,
        normalize_mean: float = 0.1307,
        normalize_std: float = 0.3081,
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.remap_labels = remap_labels
        self.num_classes = num_classes
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self._epoch = 0
        self._num_samples = len(images)
        self._num_batches = (self._num_samples + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(self._num_samples)

        if self.shuffle:
            epoch_seed = self.seed + self._epoch if self.seed is not None else None
            rng = np.random.default_rng(epoch_seed)
            rng.shuffle(indices)

        self._epoch += 1

        for start in range(0, self._num_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]

            # Get batch data
            batch_images = self.images[batch_idx].astype(np.float32)
            batch_labels = self.labels[batch_idx]

            # Normalize
            batch_images = (batch_images - self.normalize_mean) / self.normalize_std

            # Format images
            if self.tensor_format == "flat":
                batch_images = batch_images.reshape(len(batch_idx), -1)
            elif self.tensor_format == "NHWC":
                if batch_images.ndim == 3:
                    batch_images = batch_images[..., np.newaxis]

            # One-hot encode labels
            batch_labels_onehot = one_hot(batch_labels, num_classes=self.num_classes)

            yield batch_images, batch_labels_onehot


def _load_mnist_keras():
    """Load MNIST using Keras (more reliable than tfds for some versions)."""
    try:
        from tensorflow.keras.datasets import mnist

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return train_images, train_labels, test_images, test_labels
    except ImportError:
        return None
    except Exception as e:
        # Network / SSL failures (common on macOS python.org builds that lack
        # system CA certs) should fall back to the manual downloader rather
        # than crash the whole pipeline.
        print(
            f"Keras MNIST load failed ({type(e).__name__}: {e}); falling back to manual download."
        )
        return None


def _build_ssl_opener():
    """Return a urllib opener that verifies TLS using certifi when present.

    On macOS Python installers from python.org, the default SSL context often
    cannot find system CA certificates, so downloads from HTTPS hosts fail with
    CERTIFICATE_VERIFY_FAILED. If `certifi` is installed, we use its CA bundle;
    otherwise we fall back to the default context (unchanged behaviour).
    """
    import ssl
    import urllib.request

    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    https_handler = urllib.request.HTTPSHandler(context=context)
    return urllib.request.build_opener(https_handler)


def _load_mnist_manual(data_root: str):
    """Load MNIST by downloading raw files if needed."""
    import os
    import gzip
    import shutil

    # Using PyTorch's S3 mirror since yann.lecun.com is no longer available
    mnist_urls = {
        "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs(data_root, exist_ok=True)
    opener = _build_ssl_opener()

    def download_and_parse(url, filename, is_images=True):
        filepath = os.path.join(data_root, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            with opener.open(url) as response, open(filepath, "wb") as out:
                shutil.copyfileobj(response, out)

        with gzip.open(filepath, "rb") as f:
            if is_images:
                # Skip magic number and dimensions
                f.read(16)
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data.reshape(-1, 28, 28)
            else:
                # Skip magic number and count
                f.read(8)
                return np.frombuffer(f.read(), dtype=np.uint8)

    train_images = download_and_parse(
        mnist_urls["train_images"], "train-images.gz", True
    )
    train_labels = download_and_parse(
        mnist_urls["train_labels"], "train-labels.gz", False
    )
    test_images = download_and_parse(mnist_urls["test_images"], "test-images.gz", True)
    test_labels = download_and_parse(mnist_urls["test_labels"], "test-labels.gz", False)

    return train_images, train_labels, test_images, test_labels


class SplitMnistLoader:
    """
    Load MNIST split into sequential tasks by digit pairs.

    Creates a sequence of tasks, each containing two digit classes.
    Default task_pairs: [(0,1), (2,3), (4,5), (6,7), (8,9)]

    Args:
        task_pairs: Sequence of (digit1, digit2) tuples defining tasks
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle training data
        seed: Random seed for reproducibility
        tensor_format: "flat" (N, 784) or "NHWC" (N, 28, 28, 1)
        remap_labels: If True, remap labels to 0/1 for binary classification
        data_root: Path to store/load MNIST data
        num_classes: Number of output classes (10 or 2 if remapped)

    Example:
        >>> loader = SplitMnistLoader(
        ...     task_pairs=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
        ...     batch_size=256,
        ... )
        >>> for task_data in loader.tasks:
        ...     print(f"Task {task_data.task_id}: classes {task_data.classes}")
        ...     for images, labels in task_data.train_loader:
        ...         # Train on this task
        ...         pass
    """

    def __init__(
        self,
        task_pairs: Sequence[Tuple[int, int]] = (
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
        ),
        batch_size: int = 256,
        shuffle: bool = True,
        seed: Optional[int] = None,
        tensor_format: str = "flat",
        remap_labels: bool = False,
        data_root: str = "./data",
        num_classes: int = 10,
    ):
        self.task_pairs = list(task_pairs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.remap_labels = remap_labels
        self.num_classes = 2 if remap_labels else num_classes

        # Load full MNIST dataset
        self._load_mnist(data_root)

        # Build task loaders
        self.tasks: List[TaskData] = []
        for task_id, classes in enumerate(self.task_pairs):
            task_data = self._create_task(task_id, classes)
            self.tasks.append(task_data)

    def _load_mnist(self, data_root: str):
        """Load MNIST dataset using available method."""
        # Try Keras first (most reliable)
        result = _load_mnist_keras()
        if result is not None:
            train_images, train_labels, test_images, test_labels = result
        else:
            # Fall back to manual download
            train_images, train_labels, test_images, test_labels = _load_mnist_manual(
                data_root
            )

        self._train_images = train_images.astype(np.float32) / 255.0
        self._train_labels = train_labels.astype(np.int32)
        self._test_images = test_images.astype(np.float32) / 255.0
        self._test_labels = test_labels.astype(np.int32)

    def _create_task(self, task_id: int, classes: Tuple[int, int]) -> TaskData:
        """Create task data for a pair of digit classes."""
        class_a, class_b = classes

        # Filter training data
        train_mask = (self._train_labels == class_a) | (self._train_labels == class_b)
        train_images = self._train_images[train_mask]
        train_labels = self._train_labels[train_mask]

        # Filter test data
        test_mask = (self._test_labels == class_a) | (self._test_labels == class_b)
        test_images = self._test_images[test_mask]
        test_labels = self._test_labels[test_mask]

        # Optionally remap labels to 0/1
        if self.remap_labels:
            train_labels = np.where(train_labels == class_a, 0, 1).astype(np.int32)
            test_labels = np.where(test_labels == class_a, 0, 1).astype(np.int32)

        # Create task-specific seed
        task_seed = self.seed + task_id if self.seed is not None else None

        # Create loaders
        train_loader = SplitMnistTaskLoader(
            images=train_images,
            labels=train_labels,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=task_seed,
            tensor_format=self.tensor_format,
            remap_labels=self.remap_labels,
            num_classes=self.num_classes,
        )

        test_loader = SplitMnistTaskLoader(
            images=test_images,
            labels=test_labels,
            batch_size=self.batch_size,
            shuffle=False,
            seed=task_seed,
            tensor_format=self.tensor_format,
            remap_labels=self.remap_labels,
            num_classes=self.num_classes,
        )

        return TaskData(
            task_id=task_id,
            classes=(class_a, class_b),
            train_loader=train_loader,
            test_loader=test_loader,
        )

    def __len__(self) -> int:
        """Return number of tasks."""
        return len(self.tasks)

    def __iter__(self) -> Iterator[TaskData]:
        """Iterate over tasks."""
        return iter(self.tasks)

    def __getitem__(self, idx: int) -> TaskData:
        """Get task by index."""
        return self.tasks[idx]


def build_split_mnist_loaders(
    config: "ExperimentConfig",
    data_root: str = "./data",
) -> List[TaskData]:
    """
    Build Split-MNIST task loaders from experiment configuration.

    Args:
        config: ExperimentConfig with task_pairs and training settings
        data_root: Path to MNIST data

    Returns:
        List of TaskData objects, one per task
    """
    loader = SplitMnistLoader(
        task_pairs=config.task_pairs,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
        tensor_format="flat",  # FabricPC uses flat format by default
        remap_labels=False,  # Keep original 0-9 labels
        data_root=data_root,
        num_classes=10,
    )
    return loader.tasks
