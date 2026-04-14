"""
Script to visualize training results after training completes.
Use this to create plots from the saved CSV and numpy files.

Usage:
    python visualize_results.py models/unet_.../
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
CLASS_LABELS = [f"{i+1}_{name}" for i, name in enumerate(CLASS_NAMES)]

def plot_training_history(csv_path, output_path):
    """Plots training history from CSV file"""
    print(f"Loading training history from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(df['epoch'], df['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'accuracy' in df.columns:
        axes[1].plot(df['epoch'], df['accuracy'], label='Training Accuracy', linewidth=2)
        axes[1].plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # IoU
    if 'iou' in df.columns:
        axes[2].plot(df['epoch'], df['iou'], label='Training IoU', linewidth=2)
        axes[2].plot(df['epoch'], df['val_iou'], label='Validation IoU', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('IoU', fontsize=12)
        axes[2].set_title('Intersection over Union', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Plot saved: {output_path}")
    plt.close()


def plot_confusion_matrix(csv_path, output_path, use_hardcoded=False):
    """Plots confusion matrix from CSV file or hardcoded array"""
    
    if use_hardcoded:
        print("No hardcoded confusion matrix available for 8 classes, skipping plot")
        return
    else:
        print(f"Loading confusion matrix from: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot with better formatting
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', ax=ax, 
                cbar_kws={'label': 'Number of Pixels'})
    
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - Test Set Evaluation', fontsize=16, fontweight='bold')
    
    if df.shape[0] == len(CLASS_LABELS) and df.shape[1] == len(CLASS_LABELS):
        ax.set_xticklabels(CLASS_LABELS, rotation=45, ha='right')
        ax.set_yticklabels(CLASS_LABELS, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Plot saved: {output_path}")
    plt.close()


def visualize_predictions(sample_dir, output_path):
    """Visualizes prediction from saved numpy files"""
    print(f"Loading prediction from: {sample_dir}")
    
    # Load data
    image = np.load(os.path.join(sample_dir, 'image.npy'))
    true_mask = np.load(os.path.join(sample_dir, 'true_mask.npy'))
    pred_mask = np.load(os.path.join(sample_dir, 'pred_mask.npy'))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB image (channels 1,2,3)
    rgb_img = image[:, :, 1:4]
    rgb_min = rgb_img.min()
    rgb_max = rgb_img.max()
    rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min + 1e-7)
    
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original Image (RGB)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(true_mask, cmap='tab10', vmin=0, vmax=len(CLASS_NAMES)-1)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    im = axes[2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=len(CLASS_NAMES)-1)
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Class', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Plot saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <model_directory>")
        print("Example: python visualize_results.py models/unet_final/")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    
    print("="*70)
    print("VISUALIZING TRAINING RESULTS")
    print("="*70)
    print(f"Model directory: {model_dir}\n")
    
    # 1. Training history
    history_csv = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(history_csv):
        plot_training_history(history_csv, os.path.join(model_dir, "training_history.png"))
    else:
        print(f"Warning: {history_csv} not found, skipping training history plot")
    
    # 2. Confusion matrix
    cm_csv = os.path.join(model_dir, "confusion_matrix.csv")
    cm_output = os.path.join(model_dir, "confusion_matrix.png")
    
    if os.path.exists(cm_csv):
        plot_confusion_matrix(cm_csv, cm_output, use_hardcoded=False)
    else:
        print(f"Warning: {cm_csv} not found")
        print("Skipping confusion matrix plot")
    
    # 3. Predictions
    for i in range(1, 4):
        sample_dir = os.path.join(model_dir, f"prediction_sample_{i}")
        if os.path.exists(sample_dir):
            visualize_predictions(sample_dir, os.path.join(model_dir, f"prediction_sample_{i}.png"))
        else:
            print(f"Warning: {sample_dir} not found, skipping sample {i}")
    
    print("\n" + "="*70)
    print(" VISUALIZATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
