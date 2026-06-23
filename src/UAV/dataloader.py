"""
Dataloader for the UAV cascade stage.

Same methodology as src/deeplab/dataloader.py (raw values on disk, z-score
normalization deferred to load time with train-set stats, joint image/mask D4
augmentation). The only differences are dimensional:

  - tiles are 1024x1024 (not 512x512),
  - snippets on disk have 5 channels [R, G, B, nDSM_cm, Coarse_Class] (uint16),
    and are assembled into the 12-channel model input on load:
        * channels 0..3 (R, G, B, nDSM)   -> z-score normalized with stats[:4]
        * channel 4 (Coarse_Class, 1..8, 0 = nodata) -> one-hot to 8 channels
          (primary-stage taxonomy is fixed at 8; class 0 -> all-zero vector)
    giving (1024, 1024, 4 + 8) = (1024, 1024, 12).

Augmentation (training only): horizontal flip, vertical flip, 90-degree rotation,
applied jointly to image and mask — identical to the aerial pipeline.
"""

import json
import os

import numpy as np
import tensorflow as tf

TILE = 1024
N_CONT = 4                 # R, G, B, nDSM  -> z-score normalized
COARSE_NUM_CLASSES = 8     # primary-stage taxonomy -> one-hot context channels
MODEL_CHANNELS = N_CONT + COARSE_NUM_CLASSES  # 12


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


def _coarse_to_onehot(coarse):
    """Vectorized one-hot of a (H, W) class-id array (1..8, 0 = nodata) -> (H, W, 8)."""
    onehot = np.zeros((*coarse.shape, COARSE_NUM_CLASSES), dtype=np.float32)
    valid = coarse > 0
    rows, cols = np.nonzero(valid)
    onehot[rows, cols, coarse[valid] - 1] = 1.0
    return onehot


def load_npy_dataset(img_dir, mask_dir, stats_path, use_coarse=True):
    """Builds a tf.data.Dataset of (image, mask) pairs from matching .npy files.

    The on-disk snippets always have 5 channels [R, G, B, nDSM_cm, Coarse_Class].
    `use_coarse` selects the model input assembled on load (cascade ablation):
        * True  -> 4 z-score normalized + 8 one-hot coarse context = 12 channels
        * False -> 4 z-score normalized only (coarse channel dropped) = 4 channels
    Both variants read the exact same tiles, so the comparison is controlled.
    """
    img_files = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".npy")
    )
    mask_files = sorted(
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".npy")
    )
    assert len(img_files) == len(mask_files), "Image/mask count mismatch."

    mean, std = load_normalization_stats(stats_path)
    mean, std = mean[:N_CONT], std[:N_CONT]  # coarse channel is one-hot encoded, not normalized
    out_channels = MODEL_CHANNELS if use_coarse else N_CONT

    def load_sample(img_path, mask_path):
        stack = np.load(img_path.numpy().decode("utf-8")).astype(np.float32)  # (H, W, 5)
        cont = (stack[:, :, :N_CONT] - mean) / std
        if use_coarse:
            coarse = stack[:, :, N_CONT].astype(np.int32)                      # 0..8
            img = np.concatenate([cont, _coarse_to_onehot(coarse)], axis=-1).astype(np.float32)
        else:
            img = cont.astype(np.float32)
        mask = np.load(mask_path.numpy().decode("utf-8")).astype(np.float32)
        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices((img_files, mask_files))
    dataset = dataset.map(
        lambda i, m: tf.py_function(load_sample, [i, m], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda img, mask: (
            tf.ensure_shape(img, [TILE, TILE, out_channels]),
            tf.ensure_shape(mask, [TILE, TILE, 1]),
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
        dataset = dataset.shuffle(500)  # smaller buffer: 1024x1024x12 tiles are large
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
    else:
        dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)
