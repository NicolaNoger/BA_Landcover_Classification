"""
Training script for the UAV cascade stage (DeepLabV3+).

This is the UAV counterpart of src/deeplab/train_deeplab.py. The training
methodology is IDENTICAL — focal loss (class-weighted alpha), mixed precision,
legacy Adam with gradient clipping, ReduceLROnPlateau, EarlyStopping, on-the-fly
z-score normalization and D4 augmentation, 85/15 train/val split, and the same
per-class IoU / confusion-matrix evaluation.

Only dimension-related settings differ from the aerial run:
    - INPUT_SHAPE (1024, 1024, C), C set by USE_COARSE:
        * 12  [R, G, B, nDSM + 8 one-hot coarse-context]  (full cascade)
        * 4   [R, G, B, nDSM]                              (no-coarse ablation)
    - BATCH_SIZE 4                   (1024^2 x 12ch is ~16x the aerial sample size)
    - image_to_rgb reads channels 0..2 (UAV R, G, B) for previews
    - paths point at the UAV training data

The USE_COARSE flag drives the cascade vs. RGB+nDSM ablation: both variants read
the same on-disk tiles and the same train/test split, so the only difference is
whether the coarse semantic context is fed to the model.

The class taxonomy below (NUM_CLASSES, CLASS_NAMES, FOCAL_ALPHA, CLASS_COLORS) is
the refined 12-class UAV taxonomy (the target labels). This is independent of the
8-class coarse-context channel, which carries the aerial stage's taxonomy: the
cascade refines a coarse 8-class context into the finer 12-class UAV prediction.
"""

import gc
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from deeplab_v3plus import build_deeplabv3plus
from dataloader import load_npy_dataset, prepare_dataset


class Config:
    PROJECT_ROOT = "/cfs/earth/scratch/nogernic/BA_2026"
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "uav_training_data")

    TRAIN_IMG_PATH = os.path.join(DATA_PATH, "train", "img_snippets")
    TRAIN_MASK_PATH = os.path.join(DATA_PATH, "train", "mask_snippets")
    TEST_IMG_PATH = os.path.join(DATA_PATH, "test", "img_snippets")
    TEST_MASK_PATH = os.path.join(DATA_PATH, "test", "mask_snippets")
    NORM_STATS_PATH = os.path.join(DATA_PATH, "normalization_stats.json")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")

    NUM_CLASSES = 12
    CLASS_NAMES = [
        "Building", "Greenhouse", "Street", "Car",
        "Pavement", "Trees", "Hedge", "Bush",
        "Garden", "Gravel", "Acre", "Mowed_Grass",
    ]

    # Cascade ablation switch. True: feed the coarse-context channels (4 continuous
    # + 8 one-hot = 12 ch). False: RGB + nDSM only (4 ch), no coarse support.
    # Both variants read the same on-disk tiles, so the comparison is controlled.
    USE_COARSE = False

    # UAV stage: 1024x1024 tiles. Channel count follows USE_COARSE.
    INPUT_SHAPE = (1024, 1024, 12 if USE_COARSE else 4)
    TRAIN_RATIO = 0.85

    BATCH_SIZE = 4          # 1024^2 x 12 channels is ~16x the aerial sample size
    EPOCHS = 50
    LEARNING_RATE = 1e-4

    FOCAL_GAMMA = 2.0
    # Per-class focal loss weights from softened inverse class frequency:
    # alpha_c = sqrt(1 / freq_c), normalized to mean 1, with freq_c measured on the
    # training region only. The sqrt softening raises the floor on the dominant
    # classes (Trees, Acre) so focal gamma does not suppress them twice, while still
    # up-weighting rare classes (Car, Hedge). Same rationale as the aerial stage,
    # which hand-tuned weights toward the rarer classes; adjust here if a class
    # under-performs after the first run (see pixel_distribution.csv).
    FOCAL_ALPHA = [
        0.53, 0.95, 0.72, 2.51,   # Building, Greenhouse, Street, Car
        0.87, 0.39, 1.84, 1.40,   # Pavement, Trees, Hedge, Bush
        0.82, 1.02, 0.44, 0.50,   # Garden, Gravel, Acre, Mowed_Grass
    ]

    # Set to a weights path (e.g. ".../best_model_ckpt") to continue
    # training from a previous run. Leave as None to train from scratch.
    LOAD_CHECKPOINT = None
    FINETUNE_LEARNING_RATE = 1e-5


def enable_mixed_precision():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "No GPU visible to TensorFlow. Check that you requested a GPU "
            "in your SLURM job (e.g. --gres=gpu:1) and that the node has "
            "CUDA drivers matching this TensorFlow build."
        )

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Training on GPU: {gpus[0].name} ({len(gpus)} visible)")

    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def focal_loss(gamma, alpha):
    alpha_t = tf.reshape(tf.constant(alpha, dtype=tf.float32), [1, 1, 1, -1])

    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - pt, gamma)
        return tf.reduce_sum(alpha_t * focal_weight * cross_entropy, axis=-1)

    return loss_fn


# Metrics operate on one-hot targets and softmax predictions, so they argmax
# both before delegating to the standard Keras logic.
class ArgmaxAccuracy(tf.keras.metrics.Metric):
    """Pixel accuracy for one-hot encoded targets and predictions."""

    def __init__(self, name="accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        match = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
        self.correct.assign_add(tf.reduce_sum(tf.cast(match, tf.float32)))
        self.total.assign_add(tf.cast(tf.size(match), tf.float32))

    def result(self):
        return self.correct / (self.total + tf.keras.backend.epsilon())

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class ArgmaxMeanIoU(tf.keras.metrics.MeanIoU):
    """Mean IoU for one-hot encoded targets and predictions."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            tf.argmax(y_true, -1), tf.argmax(y_pred, -1), sample_weight
        )


def build_model():
    model = build_deeplabv3plus(Config.INPUT_SHAPE, Config.NUM_CLASSES)

    if Config.LOAD_CHECKPOINT:
        model.load_weights(Config.LOAD_CHECKPOINT)
        learning_rate = Config.FINETUNE_LEARNING_RATE
        print(f"Loaded weights from {Config.LOAD_CHECKPOINT}, continuing at lr={learning_rate}")
    else:
        learning_rate = Config.LEARNING_RATE

    model.compile(
        # legacy optimizer: the TF 2.11 default optimizer can't trace AutoCastVariables under mixed_float16.
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipvalue=1.0),
        loss=focal_loss(Config.FOCAL_GAMMA, Config.FOCAL_ALPHA),
        metrics=[
            ArgmaxAccuracy(),
            ArgmaxMeanIoU(num_classes=Config.NUM_CLASSES, name="iou"),
        ],
    )
    return model


class PerClassIoUCallback(tf.keras.callbacks.Callback):
    """Computes per-class IoU on a fixed validation subset after each epoch,
    via a manual confusion matrix, and appends results to a CSV.

    A fixed subset (rather than the full validation set) keeps this fast
    enough to run every epoch while still tracking class-level trends.
    """

    def __init__(self, val_dataset, output_dir, num_classes, class_names, num_batches=10):
        super().__init__()
        images, true_labels = [], []
        for imgs, masks in val_dataset.take(num_batches):
            images.append(imgs.numpy())
            true_labels.append(np.argmax(masks.numpy(), axis=-1))

        self.images = np.concatenate(images, axis=0)
        self.true_labels = np.concatenate(true_labels, axis=0)
        self.num_classes = num_classes
        self.csv_path = os.path.join(output_dir, "per_class_iou_history.csv")

        header = "epoch," + ",".join(class_names) + ",mean_iou\n"
        with open(self.csv_path, "w") as f:
            f.write(header)

    def on_epoch_end(self, epoch, logs=None):
        n = self.num_classes
        cm = np.zeros((n, n), dtype=np.int64)

        # Run the model batch by batch via __call__ rather than predict():
        # calling predict() inside a callback every epoch leaks host memory in
        # TF 2.11 and eventually triggers an OOM kill on long runs.
        batch_size = Config.BATCH_SIZE
        for start in range(0, len(self.images), batch_size):
            preds = self.model(self.images[start:start + batch_size], training=False)
            pred_labels = np.argmax(preds.numpy(), axis=-1)
            true_labels = self.true_labels[start:start + batch_size]
            np.add.at(cm, (true_labels.flatten(), pred_labels.flatten()), 1)

        iou_per_class = []
        for c in range(n):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = tp + fp + fn
            iou_per_class.append(tp / denom if denom > 0 else 0.0)

        mean_iou = float(np.mean(iou_per_class))
        row = f"{epoch + 1}," + ",".join(f"{v:.4f}" for v in iou_per_class) + f",{mean_iou:.4f}\n"
        with open(self.csv_path, "a") as f:
            f.write(row)

        gc.collect()
        print(f"  val mean IoU: {mean_iou:.4f}")


def load_data():
    train_pool = load_npy_dataset(
        Config.TRAIN_IMG_PATH, Config.TRAIN_MASK_PATH, Config.NORM_STATS_PATH,
        use_coarse=Config.USE_COARSE,
    )
    test_set = load_npy_dataset(
        Config.TEST_IMG_PATH, Config.TEST_MASK_PATH, Config.NORM_STATS_PATH,
        use_coarse=Config.USE_COARSE,
    )

    n_total = train_pool.cardinality().numpy()
    # Buffer kept small: each element is an assembled 12-channel 1024x1024 tile
    # (~52 MB), so a full-size shuffle buffer would need tens of GB of host RAM
    # and OOM-kill the job. 256 still gives a well-mixed, seed-fixed split.
    train_pool = train_pool.shuffle(256, seed=42, reshuffle_each_iteration=False)

    n_train = int(n_total * Config.TRAIN_RATIO)
    train_set = train_pool.take(n_train)
    val_set = train_pool.skip(n_train)

    steps_per_epoch = n_train // Config.BATCH_SIZE
    validation_steps = (n_total - n_train) // Config.BATCH_SIZE

    train_batches = prepare_dataset(train_set, Config.BATCH_SIZE, Config.NUM_CLASSES, is_training=True)
    val_batches = prepare_dataset(val_set, Config.BATCH_SIZE, Config.NUM_CLASSES, is_training=False)
    test_batches = prepare_dataset(test_set, Config.BATCH_SIZE, Config.NUM_CLASSES, is_training=False)

    return train_batches, val_batches, test_batches, steps_per_epoch, validation_steps


def build_callbacks(output_dir, val_batches):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            # TF checkpoint format, not .h5: h5py can't represent mixed_float16 weights on this HPC.
            os.path.join(output_dir, "best_model_ckpt"),
            save_best_only=True, save_weights_only=True, monitor="val_loss",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_history.csv"),
        ),
        PerClassIoUCallback(
            val_batches, output_dir, Config.NUM_CLASSES, Config.CLASS_NAMES,
        ),
        EpochVisualizationCallback(
            val_batches, output_dir, Config.CLASS_NAMES, every_n_epochs=5,
        ),
    ]


def evaluate(model, test_batches, output_dir):
    """Runs inference over the full test set and saves every evaluation
    metric needed to assess the model: per-class IoU / precision / recall,
    a per-class classification report, a confusion matrix and macro /
    weighted F1. All metrics are both printed and written to disk.
    """
    y_true, y_pred = [], []
    for images, masks in test_batches:
        preds = model.predict(images, verbose=0)
        y_true.append(np.argmax(masks.numpy(), axis=-1).flatten())
        y_pred.append(np.argmax(preds, axis=-1).flatten())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    labels = list(range(Config.NUM_CLASSES))
    names = Config.CLASS_NAMES

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=names, digits=4, zero_division=0,
    )
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # Per-class IoU / precision / recall straight from the confusion matrix.
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    mean_iou = float(iou.mean())

    # --- Save everything to disk ---
    with open(os.path.join(output_dir, "per_class_metrics_test.csv"), "w") as f:
        f.write("class,iou,precision,recall\n")
        for name, i, p, r in zip(names, iou, precision, recall):
            f.write(f"{name},{i:.4f},{p:.4f},{r:.4f}\n")
        f.write(f"mean,{mean_iou:.4f},,\n")

    np.savetxt(os.path.join(output_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nMean IoU:    {mean_iou:.4f}\n")
        f.write(f"Macro F1:    {f1_macro:.4f}\n")
        f.write(f"Weighted F1: {f1_weighted:.4f}\n")

    # --- Print a compact summary to the console ---
    print("\n" + "=" * 64)
    print("TEST SET EVALUATION")
    print("=" * 64)
    print(report)
    print(f"{'class':<22}{'IoU':>9}{'Precision':>11}{'Recall':>9}")
    print("-" * 51)
    for name, i, p, r in zip(names, iou, precision, recall):
        print(f"{name:<22}{i:>9.4f}{p:>11.4f}{r:>9.4f}")
    print("-" * 51)
    print(f"{'mean IoU':<22}{mean_iou:>9.4f}")
    print(f"Macro F1: {f1_macro:.4f}   Weighted F1: {f1_weighted:.4f}")
    print("=" * 64)


def save_model(model, output_dir):
    """Saves final weights. Falls back to TF checkpoint format if .h5 saving
    fails — this happens on some HPC/h5py combinations with this model size.
    """
    try:
        model.save_weights(os.path.join(output_dir, "final_model.weights.h5"))
    except Exception:
        model.save_weights(os.path.join(output_dir, "final_model_tf_ckpt"))

    model.save(os.path.join(output_dir, "final_model.keras"))


def compute_pixel_distribution(mask_dir, class_names, output_dir, sample_size=500):
    """Computes per-class pixel counts and proportions over a sample of
    training masks, and saves the result as CSV. Useful for sanity-checking
    the FOCAL_ALPHA weights against the actual class imbalance.
    """
    mask_files = sorted(
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".npy")
    )[:sample_size]

    counts = np.zeros(len(class_names), dtype=np.int64)
    for path in mask_files:
        mask = np.load(path).astype(np.int32) - 1  # labels stored as [1..N]
        for c in range(len(class_names)):
            counts[c] += np.sum(mask == c)

    total = counts.sum()
    proportions = counts / max(total, 1)

    csv_path = os.path.join(output_dir, "pixel_distribution.csv")
    with open(csv_path, "w") as f:
        f.write("class,pixels,proportion\n")
        for name, count, prop in zip(class_names, counts, proportions):
            f.write(f"{name},{count},{prop:.4f}\n")

    print(f"Pixel distribution (sampled {len(mask_files)} masks) saved to {csv_path}")
    return counts, proportions


def save_prediction_samples(model, test_batches, output_dir, num_samples=3):
    """Saves a few (image, ground truth, prediction) triplets from the test
    set as .npy arrays, for later figure generation (e.g. in QGIS or
    matplotlib) without needing to rerun inference.
    """
    samples_dir = os.path.join(output_dir, "prediction_samples")
    os.makedirs(samples_dir, exist_ok=True)

    saved = 0
    for images, masks in test_batches:
        preds = model.predict(images, verbose=0)
        true_labels = np.argmax(masks.numpy(), axis=-1)
        pred_labels = np.argmax(preds, axis=-1)

        for i in range(images.shape[0]):
            if saved >= num_samples:
                print(f"Saved {saved} prediction sample(s) to {samples_dir}")
                return
            np.save(os.path.join(samples_dir, f"sample_{saved + 1}_image.npy"), images[i].numpy())
            np.save(os.path.join(samples_dir, f"sample_{saved + 1}_true.npy"), true_labels[i])
            np.save(os.path.join(samples_dir, f"sample_{saved + 1}_pred.npy"), pred_labels[i])
            saved += 1

    print(f"Saved {saved} prediction sample(s) to {samples_dir}")


# 12-class UAV palette (0-based). The first 8 entries are the tab10 colors used by
# evaluate.py / train_deeplab.py (so Building stays blue, Trees brown, etc.), and
# four further distinct colors cover the refined UAV-only classes 9-12. The same
# array is mirrored in compare_models.py so every UAV figure shares these colors.
CLASS_COLORS = np.array([
    [ 31, 119, 180],   # 1  Building
    [255, 127,  14],   # 2  Greenhouse
    [ 44, 160,  44],   # 3  Street
    [214,  39,  40],   # 4  Car
    [148, 103, 189],   # 5  Pavement
    [140,  86,  75],   # 6  Trees
    [227, 119, 194],   # 7  Hedge
    [127, 127, 127],   # 8  Bush
    [188, 189,  34],   # 9  Garden
    [ 23, 190, 207],   # 10 Gravel
    [255, 215,   0],   # 11 Acre
    [  0, 128, 128],   # 12 Mowed_Grass
], dtype=np.uint8)


def label_to_rgb(label_map):
    """Colorizes a 0-based label map to a uint8 RGB image."""
    return CLASS_COLORS[np.clip(label_map, 0, len(CLASS_COLORS) - 1)]


def image_to_rgb(image):
    """Builds a contrast-stretched uint8 RGB image from the UAV R, G, B channels
    (indices 0, 1, 2), using a per-channel 2-98 percentile stretch. NaN-safe."""
    rgb = np.nan_to_num(np.asarray(image[:, :, 0:3], dtype=np.float32))
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for c in range(3):
        p2, p98 = np.percentile(rgb[:, :, c], (2, 98))
        if p98 > p2:
            stretched = np.clip((rgb[:, :, c] - p2) / (p98 - p2), 0, 1)
            out[:, :, c] = (stretched * 255).astype(np.uint8)
    return out


class EpochVisualizationCallback(tf.keras.callbacks.Callback):
    """Every `every_n_epochs`, saves a PNG strip [RGB | ground truth |
    prediction] for one fixed validation sample, so training progress can be
    inspected visually.

    Uses PIL only (no matplotlib) so it stays fast and robust on the HPC
    backend, and is wrapped in a guard so a visualization failure can never
    abort training.
    """

    def __init__(self, val_dataset, output_dir, class_names, every_n_epochs=5):
        super().__init__()
        for images, masks in val_dataset.take(1):
            self.image = images[0].numpy()
            self.true_label = np.argmax(masks[0].numpy(), axis=-1)
            break

        self.output_dir = os.path.join(output_dir, "epoch_samples")
        os.makedirs(self.output_dir, exist_ok=True)
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        try:
            from PIL import Image

            pred = self.model(self.image[np.newaxis, ...], training=False).numpy()[0]
            pred_label = np.argmax(pred, axis=-1)

            rgb = image_to_rgb(self.image)
            gt = label_to_rgb(self.true_label)
            pred_rgb = label_to_rgb(pred_label)

            separator = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)
            strip = np.concatenate([rgb, separator, gt, separator, pred_rgb], axis=1)
            Image.fromarray(strip).save(
                os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}.png")
            )
        except Exception as err:  # never let a preview kill training
            print(f"  [EpochVisualization] epoch {epoch + 1} skipped: {err}")


def main():
    enable_mixed_precision()

    variant = "coarse" if Config.USE_COARSE else "rgbndsm"
    output_dir = os.path.join(
        Config.OUTPUT_DIR, f"uav_deeplab_{variant}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    os.makedirs(output_dir, exist_ok=True)

    compute_pixel_distribution(Config.TRAIN_MASK_PATH, Config.CLASS_NAMES, output_dir)

    train_batches, val_batches, test_batches, steps_per_epoch, validation_steps = load_data()
    model = build_model()

    model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=build_callbacks(output_dir, val_batches),
    )

    evaluate(model, test_batches, output_dir)
    save_prediction_samples(model, test_batches, output_dir)
    save_model(model, output_dir)


if __name__ == "__main__":
    main()
