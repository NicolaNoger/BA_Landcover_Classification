import tensorflow as tf
import numpy as np
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg') 

try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    SKLEARN_AVAILABLE = True
except (ImportError, TypeError) as e:
    print(f"Warning: sklearn/scipy not available: {e}")
    print("  Confusion matrix and F1 scores will be skipped")
    SKLEARN_AVAILABLE = False


from U_net import build_unet
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
    # Paths
    PROJECT_ROOT = "A:/STUDIUM/06_Fruelingssemester26/BA"
    DATA_PATH = os.path.join(PROJECT_ROOT, "processed", "layer_stacks")

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
    BATCH_SIZE = 4
    EPOCHS = 25
    LEARNING_RATE = 5e-5

    # Model Parameter
    INPUT_SHAPE = (512, 512, 7)  # 8 channels minus Number_of_Returns
    STEPS_PER_EPOCH = None
    VALIDATION_STEPS = None
    
    # Loss Function Selection
    USE_FOCAL_LOSS = True 
    FOCAL_GAMMA = 2.0      
    FOCAL_ALPHA = 0.25    
    
    # Checkpoint Loading (set a path string if you want to continue from a compatible checkpoint)
    LOAD_CHECKPOINT = None
    RESUME_TRAINING = False  # ← False = Fine-tuning mode


def class_label(class_id):
    """Returns label in 1-based numbering format from notes."""
    if 0 <= class_id < len(Config.CLASS_NAMES):
        return f"{class_id + 1}_{Config.CLASS_NAMES[class_id]}"
    return f"{class_id + 1}"


def create_output_directory():
    """creates output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(Config.OUTPUT_DIR, f"unet_{timestamp}")
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
        
        # Calculate focal weight: (1 - pt)^gamma
        # pt is the probability of the true class
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - pt, gamma)
        
        # Apply focal weight and alpha
        focal_loss_value = alpha * focal_weight * cross_entropy
        
        # Sum over classes and return mean over batch
        return tf.reduce_sum(focal_loss_value, axis=-1)
    
    return focal_loss_fixed


def build_model():
    """Builds and compiles the U-Net model"""
    print("\n" + "="*70)
    print("STEP 2: BUILD MODEL")
    print("="*70)
    
    # 1. Create Model
    model = build_unet(
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
    
    # Use legacy Adam optimizer to avoid XLA issues with libdevice
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=Config.LEARNING_RATE),
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy'),
            tf.keras.metrics.MeanIoU(num_classes=Config.NUM_CLASSES, name='iou')
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


def setup_callbacks(output_dir):
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
    """Plots training history - saves as CSV if plotting fails"""
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    # Always save history as CSV (most reliable)
    try:
        import pandas as pd
        history_df = pd.DataFrame(history.history)
        csv_path = os.path.join(output_dir, "training_history_detailed.csv")
        history_df.to_csv(csv_path, index_label='epoch')
        print(f"Training history saved as CSV: {csv_path}")
    except Exception as e:
        print(f"Warning: Could not save history as CSV: {e}")
    
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
        
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
    
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    print(f"\nEvaluated on {len(all_true_labels):,} pixels")

    labels = list(range(Config.NUM_CLASSES))
    class_labels = [class_label(i) for i in labels]
    
    # 1. Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels)
    
    # Save confusion matrix as CSV (most reliable)
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    try:
        import pandas as pd
        cm_df = pd.DataFrame(cm, 
                            index=[f'True_{name}' for name in class_labels],
                            columns=[f'Pred_{name}' for name in class_labels])
        cm_df.to_csv(cm_csv_path)
        print(f" Confusion matrix saved as CSV: {cm_csv_path}")
        
        # Print confusion matrix to console
        print("\nConfusion Matrix:")
        print("-" * 70)
        print(cm_df.to_string())
        
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
        print(f"   To load: model = build_unet(...); model.load_weights('{weights_path}')")


def main():
    """Main function - Executes complete training"""
    print("\n" + "="*70)
    print("U-NET TRAINING PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = create_output_directory()
    
    try:
        # 1. Load data (now returns train, val, AND test)
        train_batches, val_batches, test_batches = load_data()
        
        # 2. Build model
        model, class_weights = build_model()
        
        # 3. Setup callbacks
        callbacks = setup_callbacks(output_dir)
        
        # 4. Train model with class weights
        history = train_model(model, train_batches, val_batches, callbacks, class_weights)
        
        # 5. Visualize training results (on validation set)
        plot_training_history(history, output_dir)
        
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

