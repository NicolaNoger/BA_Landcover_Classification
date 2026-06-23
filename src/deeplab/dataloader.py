"""
Dataloader for .npy image/mask snippets.

Images are stored as raw values and normalized on load using train-set
statistics (channel-wise mean/std, computed once and stored as JSON).

Augmentation (training only):
    - horizontal flip
    - vertical flip
    - 90-degree rotation (0/90/180/270)
All three are applied jointly to image and mask with shared random state,
so spatial correspondence between pixels and labels is preserved.
"""

import json
import os

import numpy as np
import tensorflow as tf


def load_normalization_stats(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std = np.where(std == 0.0, 1.0, std).astype(np.float32)
    return mean, std


def one_hot_encode(mask, num_classes):
    """Converts integer class labels [1..num_classes] to one-hot [0..num_classes-1]."""
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32) - 1
    return tf.one_hot(mask, num_classes)


def load_npy_dataset(img_dir, mask_dir, stats_path, num_channels=7):
    """Builds a tf.data.Dataset of (image, mask) pairs from matching .npy files."""
    img_files = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".npy")
    )
    mask_files = sorted(
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".npy")
    )
    assert len(img_files) == len(mask_files), "Image/mask count mismatch."

    mean, std = load_normalization_stats(stats_path)

    def load_sample(img_path, mask_path):
        img = np.load(img_path.numpy().decode("utf-8")).astype(np.float32)
        img = (img - mean) / std
        mask = np.load(mask_path.numpy().decode("utf-8")).astype(np.float32)
        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices((img_files, mask_files))
    dataset = dataset.map(
        lambda i, m: tf.py_function(load_sample, [i, m], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda img, mask: (
            tf.ensure_shape(img, [512, 512, num_channels]),
            tf.ensure_shape(mask, [512, 512, 1]),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset


def augment(image, mask):
    """Applies a shared random horizontal flip, vertical flip, and 90-degree
    rotation to an image/mask pair."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    mask = tf.image.rot90(mask, k=k)

    return image, mask


def prepare_dataset(dataset, batch_size, num_classes, is_training):
    dataset = dataset.map(
        lambda img, mask: (img, one_hot_encode(mask, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if is_training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
    else:
        dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)