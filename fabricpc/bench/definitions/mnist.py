import tensorflow_datasets as tfds
import numpy as np

class TFDSMnistIterator:
    """Wraps a TFDS dataset into a batch iterator with seed reproducibility and standard MNIST normalization."""
    def __init__(self, dataset, batch_size, shuffle=True, seed=None):
        if shuffle:
            # Pass seed into shuffle for reproducible runs across trials
            dataset = dataset.shuffle(buffer_size=1024, seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        self._iter = iter(tfds.as_numpy(dataset))
        self._dataset = dataset

    def __iter__(self):
        self._iter = iter(tfds.as_numpy(self._dataset))
        return self

    def __next__(self):
        images, labels = next(self._iter)
        # Convert to float32 and scale to [0, 1]
        images = images.astype(np.float32) / 255.0
        # Apply standard MNIST normalization (mean=0.1307, std=0.3081)
        images = (images - 0.1307) / 0.3081
        # Flatten images to match (batch_size, 784)
        images = images.reshape(images.shape[0], -1)
        return images, labels

    def __len__(self):
        return len(self._dataset)


def load_mnist(batch_size=64, train=True, seed=None):
    """Load MNIST dataset using TensorFlow Datasets v3.0.1 with standard splits."""
    split = "train" if train else "test"
    # Pin TFDS version to mnist:3.0.1 as requested
    ds = tfds.load("mnist:3.0.1", split=split, as_supervised=True)
    return TFDSMnistIterator(ds, batch_size=batch_size, shuffle=train, seed=seed)