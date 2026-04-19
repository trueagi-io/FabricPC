"""
Split-CIFAR-100 Data Loading for Continual Learning.

Provides JAX-compatible data loaders for sequential task learning on
CIFAR-100 split into groups of classes. Structured to mirror the Split-MNIST
loader in ``fabricpc.continual.data`` so the same SequentialTrainer can
consume it unchanged.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Iterator, Sequence
import numpy as np

from fabricpc.utils.data.data_utils import one_hot
from fabricpc.continual.data import TaskData

CIFAR100_MEAN = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32)
CIFAR100_STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)


class SplitCifar100TaskLoader:
    """
    Data loader for a single Split-CIFAR-100 task (a group of classes).

    Yields (images, one_hot_labels). Images are per-channel normalized and,
    in ``"flat"`` tensor_format, flattened to (N, 3072).
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        tensor_format: str = "flat",
        num_classes: int = 100,
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.num_classes = num_classes

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

            batch_images = self.images[batch_idx].astype(np.float32)
            batch_labels = self.labels[batch_idx]

            batch_images = (batch_images - CIFAR100_MEAN) / CIFAR100_STD

            if self.tensor_format == "flat":
                batch_images = batch_images.reshape(len(batch_idx), -1)
            elif self.tensor_format == "NHWC":
                pass
            else:
                raise ValueError(f"Unknown tensor_format: {self.tensor_format}")

            batch_labels_onehot = one_hot(batch_labels, num_classes=self.num_classes)
            yield batch_images, batch_labels_onehot


def _load_cifar100_keras():
    """Load CIFAR-100 via keras.datasets (uses fine labels)."""
    try:
        from tensorflow.keras.datasets import cifar100

        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data(
            label_mode="fine"
        )
        return train_images, train_labels, test_images, test_labels
    except ImportError:
        return None
    except Exception as e:
        print(
            f"Keras CIFAR-100 load failed ({type(e).__name__}: {e}); "
            "falling back to manual download."
        )
        return None


def _load_cifar100_manual(data_root: str):
    """Download CIFAR-100 python tarball and parse it without any TF dependency."""
    import os
    import tarfile
    import pickle
    import shutil
    import urllib.request
    import ssl

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    tar_path = os.path.join(data_root, "cifar-100-python.tar.gz")
    extract_dir = os.path.join(data_root, "cifar-100-python")
    os.makedirs(data_root, exist_ok=True)

    if not os.path.exists(extract_dir):
        if not os.path.exists(tar_path):
            print("Downloading CIFAR-100 (~170MB)...")
            try:
                import certifi

                context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                context = ssl.create_default_context()
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=context)
            )
            with opener.open(url) as response, open(tar_path, "wb") as out:
                shutil.copyfileobj(response, out)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_root)

    def _unpickle(path):
        with open(path, "rb") as f:
            return pickle.load(f, encoding="bytes")

    train = _unpickle(os.path.join(extract_dir, "train"))
    test = _unpickle(os.path.join(extract_dir, "test"))

    def _reshape(batch):
        data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[b"fine_labels"], dtype=np.int32)
        return data, labels

    train_images, train_labels = _reshape(train)
    test_images, test_labels = _reshape(test)
    return train_images, train_labels, test_images, test_labels


class SplitCifar100Loader:
    """
    Load CIFAR-100 split into sequential tasks by class groups.

    ``class_groups`` is a sequence of class-index tuples. The default is 20
    sequential groups of 5 classes each: ((0..4), (5..9), ..., (95..99)).
    """

    def __init__(
        self,
        class_groups: Optional[Sequence[Sequence[int]]] = None,
        classes_per_task: int = 5,
        num_tasks: int = 20,
        batch_size: int = 256,
        shuffle: bool = True,
        seed: Optional[int] = None,
        tensor_format: str = "flat",
        data_root: str = "./data",
        num_classes: int = 100,
    ):
        if class_groups is None:
            class_groups = tuple(
                tuple(range(i * classes_per_task, (i + 1) * classes_per_task))
                for i in range(num_tasks)
            )
        self.class_groups = [tuple(g) for g in class_groups]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.tensor_format = tensor_format
        self.num_classes = num_classes

        self._load_cifar100(data_root)

        self.tasks: List[TaskData] = []
        for task_id, classes in enumerate(self.class_groups):
            self.tasks.append(self._create_task(task_id, classes))

    def _load_cifar100(self, data_root: str):
        result = _load_cifar100_keras()
        if result is None:
            result = _load_cifar100_manual(data_root)
        train_images, train_labels, test_images, test_labels = result

        self._train_images = train_images.astype(np.float32) / 255.0
        self._train_labels = np.asarray(train_labels, dtype=np.int32).reshape(-1)
        self._test_images = test_images.astype(np.float32) / 255.0
        self._test_labels = np.asarray(test_labels, dtype=np.int32).reshape(-1)

    def _create_task(self, task_id: int, classes: Sequence[int]) -> TaskData:
        class_set = set(int(c) for c in classes)

        train_mask = np.isin(self._train_labels, list(class_set))
        train_images = self._train_images[train_mask]
        train_labels = self._train_labels[train_mask]

        test_mask = np.isin(self._test_labels, list(class_set))
        test_images = self._test_images[test_mask]
        test_labels = self._test_labels[test_mask]

        task_seed = self.seed + task_id if self.seed is not None else None

        train_loader = SplitCifar100TaskLoader(
            images=train_images,
            labels=train_labels,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=task_seed,
            tensor_format=self.tensor_format,
            num_classes=self.num_classes,
        )
        test_loader = SplitCifar100TaskLoader(
            images=test_images,
            labels=test_labels,
            batch_size=self.batch_size,
            shuffle=False,
            seed=task_seed,
            tensor_format=self.tensor_format,
            num_classes=self.num_classes,
        )

        return TaskData(
            task_id=task_id,
            classes=tuple(classes),
            train_loader=train_loader,
            test_loader=test_loader,
        )

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[TaskData]:
        return iter(self.tasks)

    def __getitem__(self, idx: int) -> TaskData:
        return self.tasks[idx]


def build_split_cifar100_loaders(
    config,
    data_root: str = "./data",
) -> List[TaskData]:
    """
    Build Split-CIFAR-100 task loaders from an ExperimentConfig.

    Expects:
        config.task_pairs           -> tuple of class-group tuples
        config.num_output_classes   -> 100
        config.training.batch_size
        config.seed
    """
    loader = SplitCifar100Loader(
        class_groups=config.task_pairs,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
        tensor_format="flat",
        data_root=data_root,
        num_classes=getattr(config, "num_output_classes", 100),
    )
    return loader.tasks
