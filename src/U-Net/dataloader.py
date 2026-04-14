import numpy as np
import tensorflow as tf
import os

# =============================================================================
# DATALOADER (dataloader.py)
# =============================================================================
# This script loads the prepared .npy files and creates TensorFlow Datasets
# Input:  .npy files from img_tiles/ and mask_tiles/
# Output: tf.data.Dataset objects for training and validation
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
    One-Hot Encoding of masks
    
    Converts class labels [1..8] to zero-based indices [0..7]
    then applies one-hot encoding for multi-class segmentation.
    
    Args:
        mask: Input mask with shape (H, W, 1) containing class labels [1..8]
        num_classes: Number of classes (8)
        
    Returns:
        One-hot encoded mask with shape (H, W, num_classes)
    """
    mask = tf.squeeze(mask, axis=-1)  # Remove last dimension: (H, W, 1) -> (H, W)
    mask = tf.cast(mask, tf.int32)
    
    # Convert from [1..8] to [0..7]
    mask = mask - 1
    
    # Apply one-hot encoding
    one_hot_mask = tf.one_hot(mask, num_classes)
    return one_hot_mask



def load_npy_dataset(img_path, mask_path):
    """
    Loads .npy files and creates a TensorFlow Dataset
    Input:  img_path = path to image .npy file
            mask_path = path to mask .npy file
    Output: tf.data.Dataset with (image, mask) pairs
    """
    img_files = sorted([os.path.join(img_path, f) 
                        for f in os.listdir(img_path) 
                        if f.endswith('.npy')])
    mask_files = sorted([os.path.join(mask_path, f) 
                         for f in os.listdir(mask_path) 
                         if f.endswith('.npy')])
    
    def load_sample(img_path, mask_path):
        img = np.load(img_path.numpy().decode('utf-8'))
        mask = np.load(mask_path.numpy().decode('utf-8'))
        return img, mask
    
    dataset = tf.data.Dataset.from_tensor_slices((img_files, mask_files))
    
    dataset = dataset.map(
        lambda img_path, mask_path: tf.py_function(
            load_sample, 
            [img_path, mask_path], 
            [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.map(lambda img, mask: (
        tf.ensure_shape(img, [512, 512, 5]),
        tf.ensure_shape(mask, [512, 512, 1])
    ))
    
    return dataset




def split_dataset(dataset, train_ratio=0.8):
    """
    Splits dataset into training and validation sets
    Input:  dataset = complete dataset
            train_ratio = ratio for training (0.8 = 80% training, 20% validation)
    Output: train_dataset, val_dataset
    """
    dataset_size = dataset.cardinality().numpy()
    train_size = int(dataset_size * train_ratio)
    
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    return train_dataset, val_dataset




def prepare_dataset(dataset, batch_size=8, num_classes=8, is_training=True):

    dataset = dataset.map(
        lambda img, mask: (img, one_hot_encode(mask, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Only cache training data to avoid OOM during validation
    if is_training:
        dataset = dataset.cache()
        BUFFER_SIZE = 1000
        dataset = dataset.shuffle(BUFFER_SIZE) 
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(Augment(), num_parallel_calls=tf.data.AUTOTUNE)  
        dataset = dataset.repeat()  
    else:
        # No caching for validation to save memory
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset
