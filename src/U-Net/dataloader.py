import json
import os

import numpy as np
import tensorflow as tf

# =============================================================================
# DATALOADER (dataloader.py)
# =============================================================================
# Loads .npy snippets and creates TensorFlow Datasets.
# Images are stored as raw uint16 and normalized on-the-fly using train-only stats.
# =============================================================================


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_input = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_target = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_input(inputs)
        labels = self.augment_target(labels)
        return inputs, labels


def one_hot_encode(mask, num_classes):
    """
    Converts class labels [1..8] to one-hot encoded [0..7].
    """
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)
    mask = mask - 1
    return tf.one_hot(mask, num_classes)


def _load_stats(stats_path):
    if stats_path is None:
        return None, None

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std = np.where(std == 0.0, 1.0, std).astype(np.float32)
    return mean, std


def load_npy_dataset(img_path, mask_path, stats_path=None, normalize_images=True, expected_channels=7):
    """
    Loads .npy files and creates a TensorFlow Dataset.

    Args:
        img_path: Path to image .npy directory.
        mask_path: Path to mask .npy directory.
        stats_path: Path to JSON file with keys 'mean' and 'std'.
        normalize_images: If True, apply (img - mean)/std on load.
        expected_channels: Number of image channels after channel dropping.

    Returns:
        tf.data.Dataset with (image, mask) pairs.
    """
    img_files = sorted(
        [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith(".npy")]
    )
    mask_files = sorted(
        [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith(".npy")]
    )

    if len(img_files) != len(mask_files):
        raise ValueError(
            f"Image/mask snippet count mismatch: {len(img_files)} vs {len(mask_files)}"
        )

    mean, std = _load_stats(stats_path)
    if normalize_images and (mean is None or std is None):
        raise ValueError("normalize_images=True requires a valid stats_path with mean/std.")

    def load_sample(img_file_path, mask_file_path):
        img = np.load(img_file_path.numpy().decode("utf-8"), allow_pickle=False)
        mask = np.load(mask_file_path.numpy().decode("utf-8"), allow_pickle=False)

        img = img.astype(np.float32)
        if normalize_images:
            img = (img - mean) / std

        mask = mask.astype(np.float32)
        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices((img_files, mask_files))
    dataset = dataset.map(
        lambda img_file_path, mask_file_path: tf.py_function(
            load_sample,
            [img_file_path, mask_file_path],
            [tf.float32, tf.float32],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.map(
        lambda img, mask: (
            tf.ensure_shape(img, [512, 512, expected_channels]),
            tf.ensure_shape(mask, [512, 512, 1]),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset


def split_dataset(dataset, train_ratio=0.8):
    """
    Splits dataset into training and validation sets.
    """
    dataset_size = dataset.cardinality().numpy()
    train_size = int(dataset_size * train_ratio)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    return train_dataset, val_dataset


def prepare_dataset(dataset, batch_size=8, num_classes=8, is_training=True):
    dataset = dataset.map(
        lambda img, mask: (img, one_hot_encode(mask, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if is_training:
        # dataset = dataset.cache() # Disabled to prevent OOM (Out Of Memory) on HPC
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(Augment(), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat()
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
