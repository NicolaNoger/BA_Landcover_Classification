import tensorflow as tf
import numpy as np
import os
import json
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

# PIL is used for PNG saving in callbacks (avoids matplotlib backend issues on HPC)
try:
    from PIL import Image as PilImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    SKLEARN_AVAILABLE = True
except (ImportError, TypeError) as e:
    print(f"Warning: sklearn/scipy not available: {e}")
    print("  Confusion matrix and F1 scores will be skipped")
    SKLEARN_AVAILABLE = False


from deeplab_v3plus import build_deeplabv3plus
from dataloader import load_npy_dataset, prepare_dataset



tf.config.optimizer.set_jit(False)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")


class Config:
    # Paths – auto-detected: HPC takes priority if the directory exists
    _HPC_ROOT   = "/cfs/earth/scratch/nogernic/BA_2026"
    _LOCAL_ROOT = "A:/STUDIUM/06_Fruelingssemester26/BA"
    PROJECT_ROOT = _HPC_ROOT if os.path.exists(_HPC_ROOT) else _LOCAL_ROOT
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training_data")

    TRAIN_IMG_PATH = os.path.join(DATA_PATH, "train", "img_snippets")
    TRAIN_MASK_PATH = os.path.join(DATA_PATH, "train", "mask_snippets")
    TEST_IMG_PATH = os.path.join(DATA_PATH, "test", "img_snippets")
    TEST_MASK_PATH = os.path.join(DATA_PATH, "test", "mask_snippets")
    NORM_STATS_PATH = os.path.join(DATA_PATH, "normalization_stats.json")

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")

    # Dataset Parameter
    NUM_CLASSES = 8
    CLASS_NAMES = [
        "Building",
        "Impervious_Surface",
        "Cropland",
        "Intensive_Culture",
        "Grassland_Garden",
        "Tree_Canopy",
        "Water",
        "Railway",
    ]
    TRAIN_RATIO = 0.85  # Split only the pre-defined train snippets into train/val
    VAL_RATIO = 0.15

    # Training Hyperparameter
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 5e-4

    # Model Parameter
    INPUT_SHAPE = (512, 512, 7)  # 8 channels minus Number_of_Returns
    STEPS_PER_EPOCH = None
    VALIDATION_STEPS = None
    
    # Loss Function Selection
    USE_FOCAL_LOSS = True 
    FOCAL_GAMMA = 2.0      
    # Empirically adjusted class weights based on Inverse-Frequency (downweight Water, upweight Intensive Culture)
    FOCAL_ALPHA = [1.41, 0.78, 0.78, 4.0, 0.69, 0.90, 0.30, 9.17]  
    
    # Checkpoint Loading (set a path string if you want to continue from a compatible checkpoint)
    LOAD_CHECKPOINT = None
    RESUME_TRAINING = False  # ← False = Fine-tuning mode

    # ---------------------------------------------------------------------------
    # Monitoring & Visualization (added)
    # ---------------------------------------------------------------------------

    # Channels used to build the RGB preview image (0-based index in the stack)
    # Assumed order: NIR=0, R=1, G=2, B=3, nDSM=4, ...
    VIZ_RGB_CHANNELS = (1, 2, 3)

    # How often (in epochs) to save side-by-side prediction PNG samples
    EPOCH_VIZ_INTERVAL = 5

    # Number of fixed validation samples to visualize each interval
    EPOCH_VIZ_SAMPLES = 3

    # Number of validation batches used inside PerClassIoUCallback
    # (keep low to avoid slowing down training)
    PER_CLASS_IOU_VAL_BATCHES = 30

    # Max number of mask files sampled for pixel-distribution analysis
    # (set to None to scan all train masks – slower but more accurate)
    PIXEL_DIST_SAMPLE_LIMIT = 500


def class_label(class_id):
    """Returns label in 1-based numbering format from notes."""
    if 0 <= class_id < len(Config.CLASS_NAMES):
        return f"{class_id + 1}_{Config.CLASS_NAMES[class_id]}"
    return f"{class_id + 1}"


def create_output_directory():
    """creates output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(Config.OUTPUT_DIR, f"deeplab_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data():
    """Loads and prepares pre-split train/test data with train-only normalization stats."""
    print("\n" + "="*70)
    print("STEP 1: LOAD DATA")
    print("="*70)
    print("Loading data from:")
    print(f"  - Train Images: {Config.TRAIN_IMG_PATH}")
    print(f"  - Train Masks:  {Config.TRAIN_MASK_PATH}")
    print(f"  - Test Images:  {Config.TEST_IMG_PATH}")
    print(f"  - Test Masks:   {Config.TEST_MASK_PATH}")
    print(f"  - Norm Stats:   {Config.NORM_STATS_PATH}")

    train_all_dataset = load_npy_dataset(
        Config.TRAIN_IMG_PATH,
        Config.TRAIN_MASK_PATH,
        stats_path=Config.NORM_STATS_PATH,
        normalize_images=True,
        expected_channels=Config.INPUT_SHAPE[-1],
    )
    test_dataset = load_npy_dataset(
        Config.TEST_IMG_PATH,
        Config.TEST_MASK_PATH,
        stats_path=Config.NORM_STATS_PATH,
        normalize_images=True,
        expected_channels=Config.INPUT_SHAPE[-1],
    )

    train_all_samples = train_all_dataset.cardinality().numpy()
    test_size = test_dataset.cardinality().numpy()
    print(f"Train pool samples: {train_all_samples}")
    print(f"Fixed test samples: {test_size}")

    print("\nShuffling train pool with seed=42 for reproducible train/val split...")
    train_all_dataset = train_all_dataset.shuffle(1000, seed=42, reshuffle_each_iteration=False)

    train_size = int(train_all_samples * Config.TRAIN_RATIO)
    val_size = train_all_samples - train_size

    train_dataset = train_all_dataset.take(train_size)
    val_dataset = train_all_dataset.skip(train_size)

    total_for_report = train_all_samples + test_size
    print(f"Training Samples:   {train_size} ({train_size/max(total_for_report,1)*100:.1f}% of all)")
    print(f"Validation Samples: {val_size} ({val_size/max(total_for_report,1)*100:.1f}% of all)")
    print(f"Test Samples:       {test_size} ({test_size/max(total_for_report,1)*100:.1f}% of all)")

    Config.STEPS_PER_EPOCH = max(1, train_size // Config.BATCH_SIZE)
    Config.VALIDATION_STEPS = min(100, max(1, val_size // Config.BATCH_SIZE))
    print(f"Steps per Epoch:    {Config.STEPS_PER_EPOCH}")
    print(f"Validation Steps:   {Config.VALIDATION_STEPS} (limited to save memory)")

    print("\nPreparing Training Dataset...")
    train_batches = prepare_dataset(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_classes=Config.NUM_CLASSES,
        is_training=True,
    )
    print("Training Dataset ready")

    print("Preparing Validation Dataset...")
    val_batches = prepare_dataset(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_classes=Config.NUM_CLASSES,
        is_training=False,
    )
    print("Validation Dataset ready")

    print("Preparing Test Dataset...")
    test_batches = prepare_dataset(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        num_classes=Config.NUM_CLASSES,
        is_training=False,
    )
    print("Test Dataset ready")

    print("\nChecking Data Format...")
    for images, masks in train_batches.take(1):
        print(f"Batch Images Shape: {images.shape}")
        print(f"Batch Masks Shape:  {masks.shape}")
        print(f"  Image dtype:        {images.dtype}")
        print(f"  Mask dtype:         {masks.dtype}")
        print(f"  Image value range:  [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        print(f"  Mask unique values: {np.unique(masks.numpy())}")

    return train_batches, val_batches, test_batches


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-class classification.
    
    Focal Loss focuses training on hard examples by down-weighting easy examples.
    useful for class imbalance problems.
    
    Formula: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
               - gamma=0: equivalent to categorical crossentropy
               - gamma=2: default
               - gamma=5: very strong focus on hard examples
        alpha: Class balancing parameter (0-1). Lower values give less weight to well-classified examples.
    
    Returns:
        Loss function compatible with Keras
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Ensure alpha is a tensor if it's a list (for class balancing)
        alpha_t = tf.constant(alpha, dtype=tf.float32) if isinstance(alpha, list) else alpha
        
        # Calculate focal weight: (1 - pt)^gamma
        # pt is the probability of the true class
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        # Apply focal weight and alpha
        focal_loss_value = alpha_t * focal_weight * cross_entropy
        
        # Sum over classes and return mean over batch
        return tf.reduce_sum(focal_loss_value, axis=-1)
    
    return focal_loss_fixed


def build_model():
    """Builds and compiles the U-Net model"""
    print("\n" + "="*70)
    print("STEP 2: BUILD MODEL")
    print("="*70)
    
    # 1. Create Model
    model = build_deeplabv3plus(
        input_shape=Config.INPUT_SHAPE,
        num_classes=Config.NUM_CLASSES
    )
    print("Model created")
    
    # 2. compile Model
    print("\nCompiling Model...")
    class_weights = {i: 1.0 for i in range(Config.NUM_CLASSES)}
    
    print(f"Using class weights: {class_weights}")
    
    # Select loss function
    if Config.USE_FOCAL_LOSS:
        loss_fn = focal_loss(gamma=Config.FOCAL_GAMMA, alpha=Config.FOCAL_ALPHA)
        loss_name = f"Focal Loss (gamma={Config.FOCAL_GAMMA}, alpha={Config.FOCAL_ALPHA})"
        print(f"Loss Function: {loss_name}")
        print("  → Focuses on hard-to-classify examples")
    else:
        loss_fn = 'categorical_crossentropy'
        loss_name = "Categorical Crossentropy"
        print(f"Loss Function: {loss_name}")
    
    # ---------------------------------------------------------------------------
    # Custom metrics: Keras MeanIoU and accuracy metrics do NOT handle
    # one-hot labels + softmax predictions automatically.
    # These wrappers apply argmax first so the values are correct.
    # Without this, iou stays constant (0.4375) and accuracy is ~0.03.
    # ---------------------------------------------------------------------------

    class ArgmaxMeanIoU(tf.keras.metrics.MeanIoU):
        """MeanIoU that works with one-hot labels and softmax predictions."""
        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
            return super().update_state(y_true, y_pred, sample_weight)

    class ArgmaxAccuracy(tf.keras.metrics.Metric):
        """Pixel accuracy that works with one-hot labels and softmax predictions."""
        def __init__(self, name='accuracy', **kwargs):
            super().__init__(name=name, **kwargs)
            self._correct = self.add_weight(name='correct', initializer='zeros')
            self._total   = self.add_weight(name='total',   initializer='zeros')

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred_cls = tf.argmax(y_pred, axis=-1)
            y_true_cls = tf.argmax(y_true, axis=-1)
            match = tf.cast(tf.equal(y_pred_cls, y_true_cls), tf.float32)
            self._correct.assign_add(tf.reduce_sum(match))
            self._total.assign_add(tf.cast(tf.size(y_true_cls), tf.float32))

        def result(self):
            return self._correct / (self._total + tf.keras.backend.epsilon())

        def reset_state(self):
            self._correct.assign(0.0)
            self._total.assign(0.0)

    # Use legacy Adam optimizer to avoid XLA issues with libdevice
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=Config.LEARNING_RATE),
        loss=loss_fn,
        metrics=[
            ArgmaxAccuracy(name='accuracy'),
            tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy'),
            ArgmaxMeanIoU(num_classes=Config.NUM_CLASSES, name='iou'),
        ]
    )
    print("Model compiled")

    # 3. Load checkpoint if specified
    if Config.LOAD_CHECKPOINT:
        print("\n" + "-" * 70)
        print("LOADING CHECKPOINT")
        print("-" * 70)
        try:
            model.load_weights(Config.LOAD_CHECKPOINT)
            print(f"Loaded weights from: {Config.LOAD_CHECKPOINT}")
            
            if Config.RESUME_TRAINING:
                print("  Mode: RESUME TRAINING (continuing from checkpoint)")
                print("  Note: Optimizer state is NOT restored (starts fresh)")
            else:
                print("  Mode: FINE-TUNING (using pretrained weights)")
                print("  Tip: You can change hyperparameters for fine-tuning")
        except Exception as e:
            print(f"ERROR: Could not load checkpoint: {e}")
            print("  Starting training from scratch instead")
    else:
        print("\nNo checkpoint specified - training from scratch")
    
    # 4. show Model Summary
    print("\nModel Architecture:")
    print("-" * 70)
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    
    return model, class_weights


# =============================================================================
# CLASS COLOR PALETTE  (0-based class index → RGB uint8)
# Used by all PIL-based visualizations – no matplotlib required.
# =============================================================================
CLASS_COLORS_VIZ = np.array([
    [ 20,  20,  20],   # 0: background / unlabeled
    [220,  60,  60],   # 1: Building
    [180, 180, 180],   # 2: Impervious_Surface
    [230, 210,  50],   # 3: Cropland
    [255, 155,  30],   # 4: Intensive_Culture
    [100, 200,  90],   # 5: Grassland_Garden
    [ 30, 120,  30],   # 6: Tree_Canopy
    [ 60, 140, 220],   # 7: Water
    [130,  60, 180],   # 8: Railway
], dtype=np.uint8)


def _mask_hw_to_rgb(mask_hw: np.ndarray) -> np.ndarray:
    """Convert HxW class mask (0-based) to HxWx3 uint8 color image."""
    idx = np.clip(mask_hw.astype(np.int32), 0, len(CLASS_COLORS_VIZ) - 1)
    return CLASS_COLORS_VIZ[idx]


def _img_to_rgb_uint8(img_hwc: np.ndarray) -> np.ndarray:
    """
    Convert a normalized float32 HxWxC image to uint8 RGB using
    percentile stretching (2–98 %) on the configured RGB channels.
    Works without matplotlib.
    """
    r, g, b = Config.VIZ_RGB_CHANNELS
    result = np.zeros((*img_hwc.shape[:2], 3), dtype=np.uint8)
    for out_c, in_c in enumerate([r, g, b]):
        ch = img_hwc[:, :, in_c].astype(np.float32)
        p2, p98 = np.percentile(ch, (2, 98))
        if p98 > p2:
            ch = np.clip((ch - p2) / (p98 - p2) * 255, 0, 255)
        else:
            ch = np.zeros_like(ch)
        result[:, :, out_c] = ch.astype(np.uint8)
    return result


# =============================================================================
# NEW FUNCTION 1: Pixel-class distribution analysis
# Call once before training to understand class imbalance.
# =============================================================================

def compute_pixel_distribution(mask_dir: str, output_dir: str) -> dict:
    """
    Scans mask .npy files and counts pixels per class.

    Saves:
        pixel_distribution.csv   – counts + percentages per class
        pixel_distribution.txt   – human-readable table + suggested class weights

    Returns:
        dict mapping class_id (0-based) → pixel_count
    """
    print("\n" + "="*70)
    print("STEP 0: PIXEL CLASS DISTRIBUTION")
    print("="*70)

    mask_files = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir) if f.endswith(".npy")
    ])

    if not mask_files:
        print(f"  Warning: No mask files found in {mask_dir}")
        return {}

    limit = Config.PIXEL_DIST_SAMPLE_LIMIT
    if limit and len(mask_files) > limit:
        # Evenly-spaced sample so we get global coverage
        indices   = np.linspace(0, len(mask_files) - 1, limit, dtype=int)
        mask_files = [mask_files[i] for i in indices]
        print(f"  Sampling {limit} of {len(mask_files)} mask files for speed.")
    else:
        print(f"  Scanning {len(mask_files)} mask files...")

    counts = np.zeros(Config.NUM_CLASSES + 1, dtype=np.int64)  # index 0 unused

    for path in mask_files:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        unique, cnts = np.unique(arr.astype(np.int32), return_counts=True)
        for u, c in zip(unique, cnts):
            if 0 <= u <= Config.NUM_CLASSES:
                counts[u] += c

    total_pixels = counts[1:].sum()  # class 0 = background, skip

    # Inverse-frequency class weights (for training, if needed)
    class_weights = {}
    rows = []
    print(f"\n  {'Class':<30} {'Pixels':>12} {'%':>7}  {'Inv-Freq Weight':>16}")
    print("  " + "-"*70)
    for i in range(Config.NUM_CLASSES):
        cls_id   = i + 1          # 1-based class label in mask
        cnt      = int(counts[cls_id])
        pct      = cnt / max(total_pixels, 1) * 100
        inv_w    = (total_pixels / (Config.NUM_CLASSES * max(cnt, 1)))
        class_weights[i] = round(inv_w, 4)
        name     = Config.CLASS_NAMES[i]
        print(f"  {cls_id}_{name:<28} {cnt:>12,} {pct:>7.2f}%  {inv_w:>16.4f}")
        rows.append({
            "class_id":        cls_id,
            "class_name":      name,
            "pixel_count":     cnt,
            "percentage":      round(pct, 4),
            "inv_freq_weight": round(inv_w, 4),
        })
    print("  " + "-"*70)
    print(f"  {'Total':<30} {total_pixels:>12,} {'100.00%':>7}")

    # Save CSV
    try:
        import csv
        csv_path = os.path.join(output_dir, "pixel_distribution.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["class_id", "class_name", "pixel_count", "percentage", "inv_freq_weight"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved: {csv_path}")
    except Exception as e:
        print(f"  Warning: Could not save pixel_distribution.csv: {e}")

    # Save TXT summary
    txt_path = os.path.join(output_dir, "pixel_distribution.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("PIXEL CLASS DISTRIBUTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Files scanned: {len(mask_files)}\n")
        f.write(f"Total pixels:  {total_pixels:,}\n\n")
        f.write(f"{'Class':<30} {'Pixels':>12} {'%':>7}  {'Inv-Freq Weight':>16}\n")
        f.write("-" * 70 + "\n")
        for r in rows:
            f.write(
                f"{r['class_id']}_{r['class_name']:<28} "
                f"{r['pixel_count']:>12,} {r['percentage']:>7.2f}%  "
                f"{r['inv_freq_weight']:>16.4f}\n"
            )
        f.write("-" * 70 + "\n")
        f.write(f"\nSuggested class_weights dict (0-based keys):\n")
        f.write(json.dumps(class_weights, indent=2) + "\n")
    print(f"  Saved: {txt_path}")

    return {i: int(counts[i + 1]) for i in range(Config.NUM_CLASSES)}


# =============================================================================
# NEW FUNCTION 2: Per-class IoU callback
# Computes IoU for every class after each epoch and logs to CSV.
# Uses a confusion-matrix approach so it runs in a single pass.
# =============================================================================

class PerClassIoUCallback(tf.keras.callbacks.Callback):
    """
    Computes per-class IoU on a fixed subset of the validation set after
    each epoch and appends results to per_class_iou_history.csv.

    Monitoring per-class IoU lets you spot:
      - Classes the model gives up on early (IoU → 0)
      - When individual classes start over-fitting
      - Imbalance effects that mIoU hides
    """

    def __init__(self, val_dataset, output_dir: str, num_batches: int = 30):
        super().__init__()
        # Pre-collect fixed validation samples so they are identical each epoch
        self._images, self._true = [], []
        for imgs, masks in val_dataset.take(num_batches):
            self._images.append(imgs.numpy())
            # masks are one-hot → convert to class indices
            self._true.append(np.argmax(masks.numpy(), axis=-1))
        self._images = np.concatenate(self._images, axis=0)  # N x H x W x C
        self._true   = np.concatenate(self._true,   axis=0)  # N x H x W
        self._output_dir = output_dir
        self._csv_path   = os.path.join(output_dir, "per_class_iou_history.csv")
        self._num_classes = Config.NUM_CLASSES

        # Write CSV header
        header = "epoch," + ",".join(
            f"iou_{class_label(i)}" for i in range(self._num_classes)
        ) + ",mean_iou\n"
        with open(self._csv_path, "w", encoding="utf-8") as f:
            f.write(header)

    def on_epoch_end(self, epoch, logs=None):
        try:
            preds = self.model.predict(self._images, batch_size=Config.BATCH_SIZE, verbose=0)
            pred_classes = np.argmax(preds, axis=-1)  # N x H x W

            # Build confusion matrix over all pixels
            n = self._num_classes
            cm = np.zeros((n, n), dtype=np.int64)
            true_flat = self._true.flatten()
            pred_flat = pred_classes.flatten()
            np.add.at(cm, (true_flat, pred_flat), 1)

            # IoU per class: TP / (TP + FP + FN)
            iou_per_class = []
            for c in range(n):
                tp  = cm[c, c]
                fp  = cm[:, c].sum() - tp
                fn  = cm[c, :].sum() - tp
                denom = tp + fp + fn
                iou_per_class.append(tp / denom if denom > 0 else 0.0)

            mean_iou = float(np.mean(iou_per_class))

            # Log to CSV
            row = f"{epoch + 1}," + ",".join(f"{v:.6f}" for v in iou_per_class)
            row += f",{mean_iou:.6f}\n"
            with open(self._csv_path, "a", encoding="utf-8") as f:
                f.write(row)

            # Print compact table to console
            print(f"\n  [PerClassIoU] Epoch {epoch + 1}  –  mean IoU: {mean_iou:.4f}")
            for i, iou in enumerate(iou_per_class):
                bar = "█" * int(iou * 20) + "░" * (20 - int(iou * 20))
                flag = " ⚠" if iou < 0.20 else ""
                print(f"    {class_label(i):<35} {bar}  {iou:.3f}{flag}")

            if logs is not None:
                logs["val_mean_iou_per_class"] = mean_iou

        except Exception as e:
            print(f"\n  [PerClassIoUCallback] Warning: {e}")


# =============================================================================
# NEW FUNCTION 3: Per-epoch visual samples callback
# Saves RGB / GT-mask / Pred-mask side-by-side PNGs using PIL only.
# No matplotlib needed → safe on HPC.
# =============================================================================

class EpochVisualizationCallback(tf.keras.callbacks.Callback):
    """
    Every EPOCH_VIZ_INTERVAL epochs, saves a side-by-side PNG comparison:
        [RGB image] | [Ground Truth mask] | [Predicted mask]

    Images are saved to:
        <output_dir>/epoch_samples/epoch_<N>/sample_<i>.png

    Watching these across epochs lets you visually detect:
      - Underfitting (blurry, wrong classes everywhere)
      - Overfitting (train samples look perfect, val looks odd)
      - Which classes appear / disappear over time
    """

    def __init__(self, val_dataset, output_dir: str):
        super().__init__()
        self._output_dir = output_dir
        self._interval   = Config.EPOCH_VIZ_INTERVAL
        self._n_samples  = Config.EPOCH_VIZ_SAMPLES
        self._available  = PIL_AVAILABLE

        if not self._available:
            print("  [EpochVizCallback] PIL not available – visualization skipped.")
            return

        # Pre-collect fixed samples (same snippets every epoch for comparability)
        self._images, self._true = [], []
        for imgs, masks in val_dataset.take(1):
            imgs_np  = imgs.numpy()
            masks_np = np.argmax(masks.numpy(), axis=-1)
            for i in range(min(self._n_samples, imgs_np.shape[0])):
                self._images.append(imgs_np[i])
                self._true.append(masks_np[i])

    def on_epoch_end(self, epoch, logs=None):
        if not self._available:
            return
        if (epoch + 1) % self._interval != 0:
            return

        try:
            epoch_dir = os.path.join(
                self._output_dir, "epoch_samples", f"epoch_{epoch + 1:03d}"
            )
            os.makedirs(epoch_dir, exist_ok=True)

            imgs_arr = np.stack(self._images, axis=0)
            preds    = self.model.predict(imgs_arr, batch_size=len(imgs_arr), verbose=0)
            pred_cls = np.argmax(preds, axis=-1)

            for i in range(len(self._images)):
                rgb   = _img_to_rgb_uint8(self._images[i])           # HxWx3
                gt    = _mask_hw_to_rgb(self._true[i])                # HxWx3
                pred  = _mask_hw_to_rgb(pred_cls[i])                  # HxWx3

                # Horizontal concat with 4 px white separator
                sep   = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)
                strip = np.concatenate([rgb, sep, gt, sep, pred], axis=1)

                out_path = os.path.join(epoch_dir, f"sample_{i + 1}.png")
                PilImage.fromarray(strip).save(out_path)

            print(
                f"\n  [EpochViz] Epoch {epoch + 1}: "
                f"{len(self._images)} sample(s) saved → {epoch_dir}/"
            )
        except Exception as e:
            print(f"\n  [EpochVizCallback] Warning: {e}")


# =============================================================================
# NEW FUNCTION 4: Overfitting gap analysis (appended to plot_training_history)
# =============================================================================

def _print_overfitting_analysis(history_dict: dict, output_dir: str) -> None:
    """
    Analyses the train/val loss gap to flag potential over- or underfitting.
    Saves a human-readable report as overfitting_analysis.txt.
    """
    loss     = [float(v) for v in history_dict.get("loss",     [])]
    val_loss = [float(v) for v in history_dict.get("val_loss", [])]

    if not loss or not val_loss:
        return

    n_epochs   = len(loss)
    gaps       = [vl - tl for tl, vl in zip(loss, val_loss)]
    best_epoch = int(np.argmin(val_loss)) + 1
    final_gap  = gaps[-1]
    max_gap    = max(gaps)

    lines = []
    lines.append("OVERFITTING / UNDERFITTING ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Epochs trained:          {n_epochs}")
    lines.append(f"Best val_loss epoch:     {best_epoch}  ({min(val_loss):.6f})")
    lines.append(f"Final train_loss:        {loss[-1]:.6f}")
    lines.append(f"Final val_loss:          {val_loss[-1]:.6f}")
    lines.append(f"Final gap (val-train):   {final_gap:+.6f}")
    lines.append(f"Max gap seen:            {max_gap:+.6f}")
    lines.append("")

    # Heuristic interpretation
    if final_gap > 0.15:
        verdict = "⚠  OVERFITTING  – val_loss significantly above train_loss."
        advice  = (
            "   Suggestions: increase dropout, add data augmentation,\n"
            "   reduce model capacity, or stop earlier (best epoch was "
            f"epoch {best_epoch})."
        )
    elif final_gap < -0.05:
        verdict = "⚠  UNDERFITTING – train_loss above val_loss (unusual)."
        advice  = (
            "   Possible causes: too-strong regularization, very noisy\n"
            "   training labels, or too few training steps."
        )
    elif loss[-1] > 0.5 and val_loss[-1] > 0.5:
        verdict = "⚠  UNDERFITTING – both losses still high."
        advice  = (
            "   Suggestions: train for more epochs, increase learning rate,\n"
            "   or check data pipeline for issues."
        )
    else:
        verdict = "✅ No clear over/underfitting detected."
        advice  = "   Train and val losses track closely."

    lines.append(verdict)
    lines.append(advice)
    lines.append("")
    lines.append("Per-epoch loss gap (val_loss - train_loss):")
    lines.append(f"  {'Epoch':>6}  {'train_loss':>12}  {'val_loss':>12}  {'gap':>10}")
    lines.append("  " + "-" * 48)
    for ep, (tl, vl, g) in enumerate(zip(loss, val_loss, gaps), start=1):
        marker = " ←best" if ep == best_epoch else ""
        lines.append(f"  {ep:>6}  {tl:>12.6f}  {vl:>12.6f}  {g:>+10.6f}{marker}")

    report = "\n".join(lines)
    print("\n" + report)

    txt_path = os.path.join(output_dir, "overfitting_analysis.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n  Saved: {txt_path}")


# =============================================================================
# NEW FUNCTION 5: Per-class IoU in final evaluation
# Called inside evaluate_model to extend the existing output.
# =============================================================================

def _compute_per_class_iou(true_flat: np.ndarray, pred_flat: np.ndarray,
                            output_dir: str) -> None:
    """
    Computes and saves per-class IoU, Precision, and Recall from flattened
    pixel arrays.  Called at the end of evaluate_model().
    """
    n   = Config.NUM_CLASSES
    cm  = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (true_flat, pred_flat), 1)

    iou_list, prec_list, rec_list = [], [], []
    for c in range(n):
        tp    = cm[c, c]
        fp    = cm[:, c].sum() - tp
        fn    = cm[c, :].sum() - tp
        denom_iou  = tp + fp + fn
        denom_prec = tp + fp
        denom_rec  = tp + fn
        iou_list.append(tp / denom_iou   if denom_iou  > 0 else 0.0)
        prec_list.append(tp / denom_prec if denom_prec > 0 else 0.0)
        rec_list.append(tp / denom_rec   if denom_rec  > 0 else 0.0)

    mean_iou = float(np.mean(iou_list))

    # Print table
    print("\n" + "="*70)
    print("PER-CLASS IoU  |  PRECISION  |  RECALL  (Test Set)")
    print("="*70)
    print(f"  {'Class':<35} {'IoU':>8}  {'Precision':>10}  {'Recall':>8}")
    print("  " + "-"*65)
    for i in range(n):
        flag = " ⚠" if iou_list[i] < 0.20 else ""
        print(
            f"  {class_label(i):<35} {iou_list[i]:>8.4f}  "
            f"{prec_list[i]:>10.4f}  {rec_list[i]:>8.4f}{flag}"
        )
    print("  " + "-"*65)
    print(f"  {'Mean IoU':<35} {mean_iou:>8.4f}")
    print("="*70)

    # Save CSV
    try:
        import csv
        rows = [
            {
                "class_id":   i + 1,
                "class_name": Config.CLASS_NAMES[i],
                "iou":        round(iou_list[i], 6),
                "precision":  round(prec_list[i], 6),
                "recall":     round(rec_list[i], 6),
            }
            for i in range(n)
        ]
        csv_path = os.path.join(output_dir, "per_class_iou_test.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "iou", "precision", "recall"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")
    except Exception as e:
        print(f"  Warning: Could not save per_class_iou_test.csv: {e}")


# =============================================================================
# NEW FUNCTION 6: Save consolidated training summary as JSON
# =============================================================================

def save_training_summary_json(history, output_dir: str) -> None:
    """
    Saves a single JSON file with all key training metrics for easy
    programmatic access (e.g. comparison across runs).
    """
    h = history.history
    loss     = [float(v) for v in h.get("loss",     [])]
    val_loss = [float(v) for v in h.get("val_loss", [])]
    iou      = [float(v) for v in h.get("iou",      [])]
    val_iou  = [float(v) for v in h.get("val_iou",  [])]

    best_ep = int(np.argmin(val_loss)) if val_loss else -1

    summary = {
        "run_timestamp":     datetime.now().isoformat(),
        "epochs_trained":    len(loss),
        "best_epoch":        best_ep + 1,
        "best_val_loss":     round(val_loss[best_ep], 6) if val_loss else None,
        "final_train_loss":  round(loss[-1], 6)     if loss     else None,
        "final_val_loss":    round(val_loss[-1], 6) if val_loss else None,
        "final_train_iou":   round(iou[-1], 6)      if iou      else None,
        "final_val_iou":     round(val_iou[-1], 6)  if val_iou  else None,
        "overfitting_gap":   round(val_loss[-1] - loss[-1], 6) if (loss and val_loss) else None,
        "config": {
            "input_shape":    list(Config.INPUT_SHAPE),
            "num_classes":    Config.NUM_CLASSES,
            "batch_size":     Config.BATCH_SIZE,
            "epochs_planned": Config.EPOCHS,
            "learning_rate":  Config.LEARNING_RATE,
            "use_focal_loss": Config.USE_FOCAL_LOSS,
            "focal_gamma":    Config.FOCAL_GAMMA,
            "focal_alpha":    Config.FOCAL_ALPHA,
            "train_ratio":    Config.TRAIN_RATIO,
        },
        "class_names": Config.CLASS_NAMES,
        # Full epoch-by-epoch history for plotting
        "history": {k: [float(v) for v in vals] for k, vals in h.items()},
    }

    json_path = os.path.join(output_dir, "training_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Training summary saved: {json_path}")


def setup_callbacks(output_dir, val_batches=None):
    """Creates callbacks for training"""
    print("\n" + "="*70)
    print("STEP 3: SETUP CALLBACKS")
    print("="*70)
    
    callbacks = []
    
    # 0. Memory Cleanup Callback - Clear cache after each epoch to avoid OOM
    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            import gc
            gc.collect()
    
    callbacks.append(MemoryCleanupCallback())
    print("MemoryCleanupCallback: Forces garbage collection after each epoch")

    # 1. ModelCheckpoint - saves best model weights
    # Use TensorFlow checkpoint format (not .h5) to avoid h5py issues
    checkpoint_path = os.path.join(output_dir, "best_model_weights", "checkpoint")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,  # Only save weights, not full model
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    print(f"ModelCheckpoint: {checkpoint_path} (TF checkpoint format)")
    
    
    # 2. EarlyStopping - Stops training if no progress
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Waits 5 epochs without improvement
        mode='min',
        restore_best_weights=False,  # Don't restore, use ModelCheckpoint instead
        verbose=1
    )
    callbacks.append(early_stop)
    print("EarlyStopping: patience=5 (no weight restore)")
    
    # 3. ReduceLROnPlateau - Reduces learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Halves learning rate
        patience=3,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    callbacks.append(reduce_lr)
    print("ReduceLROnPlateau: factor=0.5, patience=3")
    
    # 4. CSVLogger - Saves training history
    csv_path = os.path.join(output_dir, "training_history.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(csv_path)
    callbacks.append(csv_logger)
    print(f"CSVLogger: {csv_path}")
    
    # 5. TensorBoard - For visualization 
    tensorboard_dir = os.path.join(output_dir, "tensorboard_logs")
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)
    print(f"TensorBoard: {tensorboard_dir}")
    print("  (Start with: tensorboard --logdir=models)")

    # 6. PerClassIoUCallback – per-class IoU logged every epoch
    if val_batches is not None:
        per_class_iou_cb = PerClassIoUCallback(
            val_dataset  = val_batches,
            output_dir   = output_dir,
            num_batches  = Config.PER_CLASS_IOU_VAL_BATCHES,
        )
        callbacks.append(per_class_iou_cb)
        print(
            f"PerClassIoUCallback: logs per-class IoU every epoch "
            f"({Config.PER_CLASS_IOU_VAL_BATCHES} val batches) → per_class_iou_history.csv"
        )

    # 7. EpochVisualizationCallback – RGB/GT/Pred PNGs every N epochs
    if val_batches is not None and PIL_AVAILABLE:
        epoch_viz_cb = EpochVisualizationCallback(
            val_dataset = val_batches,
            output_dir  = output_dir,
        )
        callbacks.append(epoch_viz_cb)
        print(
            f"EpochVisualizationCallback: saves prediction PNGs every "
            f"{Config.EPOCH_VIZ_INTERVAL} epochs → epoch_samples/"
        )
    elif not PIL_AVAILABLE:
        print("EpochVisualizationCallback: skipped (PIL not installed)")

    return callbacks


def train_model(model, train_batches, val_batches, callbacks, class_weights=None):
    """Trains the model"""
    print("\n" + "="*70)
    print("STEP 4: START TRAINING")
    print("="*70)

    print(f"\nTraining Configuration:")
    print(f"  Epochs:             {Config.EPOCHS}")
    print(f"  Batch Size:         {Config.BATCH_SIZE}")
    print(f"  Steps per Epoch:    {Config.STEPS_PER_EPOCH}")
    print(f"  Validation Steps:   {Config.VALIDATION_STEPS}")
    print(f"  Number of Classes:  {Config.NUM_CLASSES}")
    if class_weights:
        print(f"  Class Weights:      Enabled (balancing underrepresented classes)")
    print("\nTraining starts...\n")

    # GPU Info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Training with GPU: {gpus[0].name}")
    else:
        print("Warning: No GPU found, training on CPU!")
    print()
    
    # Perform training
    history = model.fit(
        train_batches,
        epochs=Config.EPOCHS,
        steps_per_epoch=Config.STEPS_PER_EPOCH,
        validation_data=val_batches,
        validation_steps=Config.VALIDATION_STEPS,
        callbacks=callbacks,
        verbose=1  # shows Progress Bar
    )
    
    print("\nTraining completed!")
    return history


def plot_training_history(history, output_dir):
    """Plots training history"""
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    # Try to plot
    try:
        # Disable matplotlib completely if environment variable set
        import os as os_module
        if os_module.environ.get('DISABLE_PLOTS', '0') == '1':
            print("Plotting disabled via DISABLE_PLOTS=1")
            return
        
        # Simple text-based summary instead of plots
        print("\nTraining Summary:")
        print("-" * 70)
        
        loss = [float(x) if hasattr(x, '__float__') else float(np.array(x)) for x in history.history['loss']]
        val_loss = [float(x) if hasattr(x, '__float__') else float(np.array(x)) for x in history.history['val_loss']]
        
        print(f"Final Training Loss:   {loss[-1]:.4f}")
        print(f"Final Validation Loss: {val_loss[-1]:.4f}")
        print(f"Best Validation Loss:  {min(val_loss):.4f} (Epoch {val_loss.index(min(val_loss))+1})")
        
        if 'accuracy' in history.history:
            acc = [float(x) if hasattr(x, '__float__') else float(np.array(x)) for x in history.history['accuracy']]
            val_acc = [float(x) if hasattr(x, '__float__') else float(np.array(x)) for x in history.history['val_accuracy']]
            print(f"Final Training Accuracy:   {acc[-1]:.4f}")
            print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")
            print(f"Best Validation Accuracy:  {max(val_acc):.4f} (Epoch {val_acc.index(max(val_acc))+1})")
        
        print("\n Training history available in CSV file for plotting")
        print("  You can plot it later with: pandas.read_csv('training_history_detailed.csv')")

        # Overfitting gap analysis (added)
        _print_overfitting_analysis(history.history, output_dir)
        
    except Exception as e:
        print(f"Note: Visualization skipped due to matplotlib compatibility issues")
        print(f"  Training data saved in CSV format for later analysis")


def visualize_predictions(model, val_batches, output_dir, num_samples=3):
    """Visualizes predictions - saves raw data instead of images"""
    print("\nSaving prediction samples...")
    
    try:
        # Get one batch and save raw predictions as numpy files
        for images, masks in val_batches.take(1):
            predictions = model.predict(images, verbose=0)
            
            # Convert everything to numpy arrays
            images_np = images.numpy()
            masks_np = masks.numpy()
            predictions_np = predictions
            
            # Convert one-hot back to classes
            true_masks = np.argmax(masks_np, axis=-1)
            pred_masks = np.argmax(predictions_np, axis=-1)
            
            # Save first num_samples as numpy files 
            for i in range(min(num_samples, images_np.shape[0])):
                sample_dir = os.path.join(output_dir, f"prediction_sample_{i+1}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save all data
                np.save(os.path.join(sample_dir, "image.npy"), images_np[i])
                np.save(os.path.join(sample_dir, "true_mask.npy"), true_masks[i])
                np.save(os.path.join(sample_dir, "pred_mask.npy"), pred_masks[i])
                
                # Save a simple text summary
                summary_path = os.path.join(sample_dir, "summary.txt")
                with open(summary_path, 'w') as f:
                    f.write(f"Prediction Sample {i+1}\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Image shape: {images_np[i].shape}\n")
                    f.write(f"True mask shape: {true_masks[i].shape}\n")
                    f.write(f"Predicted mask shape: {pred_masks[i].shape}\n\n")
                    f.write("Class distribution in true mask:\n")
                    for class_id in range(Config.NUM_CLASSES):
                        count = np.sum(true_masks[i] == class_id)
                        percentage = count / true_masks[i].size * 100
                        f.write(f"  {class_label(class_id)}: {count:7d} pixels ({percentage:5.2f}%)\n")
                    f.write("\nClass distribution in predicted mask:\n")
                    for class_id in range(Config.NUM_CLASSES):
                        count = np.sum(pred_masks[i] == class_id)
                        percentage = count / pred_masks[i].size * 100
                        f.write(f"  {class_label(class_id)}: {count:7d} pixels ({percentage:5.2f}%)\n")
                
                print(f" Sample {i+1} saved: {sample_dir}/")
            
            print("\nNote: Raw prediction data saved as .npy files")
            print("      You can visualize them later with matplotlib/QGIS")
            break
                    
    except Exception as e:
        print(f"Warning: Could not save prediction samples: {e}")
        import traceback
        traceback.print_exc()


def evaluate_model(model, val_batches, output_dir):
    """Evaluates model with confusion matrix and F1 score"""
    print("\n" + "="*70)
    print("STEP 5b: MODEL EVALUATION")
    print("="*70)
    
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    
    print("Collecting predictions for evaluation...")
    
    all_true_labels = []
    all_pred_labels = []
    
    # Collect predictions from validation set
    num_batches = min(100, Config.VALIDATION_STEPS)  # Limit for speed
    for i, (images, masks) in enumerate(val_batches.take(num_batches)):
        if i % 20 == 0:
            print(f"  Processing batch {i+1}/{num_batches}...")
        
        predictions = model.predict(images, verbose=0)
        
        # Convert one-hot to class labels
        true_labels = np.argmax(masks.numpy(), axis=-1).flatten()
        pred_labels = np.argmax(predictions, axis=-1).flatten()
        
        all_true_labels.append(true_labels)
        all_pred_labels.append(pred_labels)
    
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    
    print(f"\nEvaluated on {len(all_true_labels):,} pixels")

    labels = list(range(Config.NUM_CLASSES))
    class_labels = [class_label(i) for i in labels]
    
    # 1. Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
    
    # Save confusion matrix as CSV using basic file I/O to avoid pandas issues
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    try:
        with open(cm_csv_path, "w", encoding="utf-8") as f:
            header = ",".join([f'Pred_{name}' for name in class_labels])
            f.write("," + header + "\n")
            for i, row in enumerate(cm):
                f.write(f"True_{class_labels[i]}," + ",".join(map(str, row)) + "\n")
        print(f" Confusion matrix saved as CSV: {cm_csv_path}")
        
        # Print confusion matrix to console
        print("\nConfusion Matrix:")
        print("-" * 70)
        # Simple table formatting
        header_str = "      " + " ".join([f"{i:4d}" for i in range(len(class_labels))])
        print(header_str)
        for i, row in enumerate(cm):
            print(f"T_{i:<2d} [" + " ".join([f"{val:4d}" for val in row]) + "]")
        
    except Exception as e:
        print(f"Warning: Could not save confusion matrix as CSV: {e}")
        # Fallback: print to console
        print("\nConfusion Matrix (numpy array):")
        print(cm)
    
    # Skip plotting - matplotlib has compatibility issues on this system
    print("(Confusion matrix plot skipped due to matplotlib compatibility issues)")
    
    # 2. Classification Report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    report = classification_report(
        all_true_labels, 
        all_pred_labels,
        labels=labels,
        target_names=class_labels,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Save report to file
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(report)
    print(f" Classification report saved: {report_path}")
    
    # 3. F1 Scores
    print("\n" + "="*70)
    print("F1 SCORES")
    print("="*70)
    
    # Per-class F1
    f1_per_class = f1_score(all_true_labels, all_pred_labels, labels=labels, average=None, zero_division=0)
    print("\nPer-Class F1 Scores:")
    for i, f1 in enumerate(f1_per_class):
        print(f"  {class_label(i)}: {f1:.4f}")
    
    # Overall F1 scores
    f1_macro = f1_score(all_true_labels, all_pred_labels, labels=labels, average='macro', zero_division=0)
    f1_weighted = f1_score(all_true_labels, all_pred_labels, labels=labels, average='weighted', zero_division=0)
    
    print(f"\nMacro-averaged F1:    {f1_macro:.4f}")
    print(f"Weighted-averaged F1: {f1_weighted:.4f}")
    
    # Save F1 scores
    f1_path = os.path.join(output_dir, "f1_scores.txt")
    with open(f1_path, 'w') as f:
        f.write("F1 SCORES\n")
        f.write("="*70 + "\n\n")
        f.write("Per-Class F1 Scores:\n")
        for i, f1 in enumerate(f1_per_class):
            f.write(f"  {class_label(i)}: {f1:.4f}\n")
        f.write(f"\nMacro-averaged F1:    {f1_macro:.4f}\n")
        f.write(f"Weighted-averaged F1: {f1_weighted:.4f}\n")
    print(f" F1 scores saved: {f1_path}")
    
    # Per-class IoU, Precision, Recall (added)
    _compute_per_class_iou(all_true_labels, all_pred_labels, output_dir)

    print("="*70)

def save_model(model, output_dir):
    """Saves final model"""
    print("\n" + "="*70)
    print("STEP 6: SAVE MODEL")
    print("="*70)
    
    # Save weights first (most reliable, avoids h5py issues)
    weights_path = os.path.join(output_dir, "final_model_weights.h5")
    try:
        model.save_weights(weights_path)
        print(f" Model weights saved: {weights_path}")
    except Exception as e:
        print(f"Warning: Could not save weights as .h5: {e}")
        weights_path_tf = os.path.join(output_dir, "final_model_weights")
        model.save_weights(weights_path_tf)
        print(f" Model weights saved: {weights_path_tf} (TF format)")
    
    # Try to save complete model (SavedModel format)
    model_path = os.path.join(output_dir, "final_model")
    try:
        model.save(model_path, save_format='tf')
        print(f" Complete model saved: {model_path} (SavedModel format)")
    except Exception as e:
        print(f"Warning: Could not save complete model: {e}")
        print(f"   To load: model = build_deeplabv3plus(...); model.load_weights('{weights_path}')")


def main():
    """Main function - Executes complete training"""
    print("\n" + "="*70)
    print("DEEPLABV3+ TRAINING PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = create_output_directory()
    
    try:
        # 1. Load data (now returns train, val, AND test)
        train_batches, val_batches, test_batches = load_data()

        # 1b. Pixel distribution analysis – run once before training (added)
        compute_pixel_distribution(Config.TRAIN_MASK_PATH, output_dir)
        
        # 2. Build model
        model, class_weights = build_model()
        
        # 3. Setup callbacks – pass val_batches for new monitoring callbacks (added)
        callbacks = setup_callbacks(output_dir, val_batches=val_batches)
        
        # 4. Train model with class weights
        history = train_model(model, train_batches, val_batches, callbacks, class_weights)
        
        # 5. Visualize training results (on validation set)
        plot_training_history(history, output_dir)

        # 5b. Save consolidated training summary JSON (added)
        save_training_summary_json(history, output_dir)
        
        # 6. Final evaluation on TEST SET (unseen data!)
        print("\n" + "="*70)
        print("FINAL EVALUATION ON TEST SET")
        print("="*70)
        evaluate_model(model, test_batches, output_dir)
        visualize_predictions(model, test_batches, output_dir, num_samples=3)
        
        # 7. Save model
        save_model(model, output_dir)
        
        print("\n" + "="*70)
        print(" TRAINING SUCCESSFUL!")
        print("="*70)
        print(f"All files saved in: {output_dir}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR DURING TRAINING!")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be in: {output_dir}")


if __name__ == "__main__":
    main()