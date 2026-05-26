"""
generate_ic_queue.py  –  Class audit: Intensive_Culture → Cropland

Scans all GT masks for Intensive_Culture (class 4) blobs and queues them
for manual review. No confidence threshold or hysteresis needed – every
IC blob above MIN_BLOB_SIZE_PX is flagged regardless of model prediction.

Model is optional: if MODEL_PATH exists, inference is run and the
prediction overlay is included as visual context. If MODEL_PATH is not
set or not found, only RGB + GT mask are shown (still fully usable).

Queue format is identical to generate_queue.py → labeling_review.py
works unchanged, with one extra button: "🌾 Relabel to Cropland".

Usage:
    # Without model (fast, GT mask only):
    python generate_ic_queue.py --no-model

    # With model (shows prediction as context):
    python generate_ic_queue.py --model path/to/final_model
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from PIL import Image
from scipy.ndimage import label as scipy_label, binary_dilation
import tensorflow as tf


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    PROJECT_ROOT   = "/cfs/earth/scratch/nogernic/BA_2026"
    DATA_PATH      = os.path.join(PROJECT_ROOT, "data", "training_data")

    # Run on train data (where the IC labeling errors are suspected)
    IMG_PATH       = os.path.join(DATA_PATH, "train", "img_snippets")
    MASK_PATH      = os.path.join(DATA_PATH, "train", "mask_snippets")
    NORM_STATS_PATH = os.path.join(DATA_PATH, "normalization_stats.json")

    # Optional: set to None or use --no-model to skip inference
    MODEL_PATH     = os.path.join(
        PROJECT_ROOT, "models", "deeplab_20260512_163440", "final_model"
    )

    # Output (separate from the main label review)
    OUTPUT_DIR     = os.path.join(DATA_PATH, "ic_review")
    QUEUE_FILE     = os.path.join(OUTPUT_DIR, "ic_review_queue.json")
    OVERLAYS_DIR   = os.path.join(OUTPUT_DIR, "overlays")

    # Inference
    BATCH_SIZE     = 8
    NUM_CLASSES    = 8
    INPUT_CHANNELS = 7

    # Target class to audit
    AUDIT_CLASS_ID   = 4       # Intensive_Culture
    AUDIT_CLASS_NAME = "Intensive_Culture"

    # Blob filtering
    # Lower than generate_queue because we want to catch all IC patches,
    # including small misLabeled fields. ~45x45px = 2025px minimum.
    MIN_BLOB_SIZE_PX = 2000

    # Max blobs per snippet to avoid queueing hundreds of tiny fragments
    # from one image. Sorted by size descending so largest patches come first.
    MAX_BLOBS_PER_SNIPPET = 20

    # RGB channels for overlay preview (0-based index in the stack)
    RGB_CHANNELS = (1, 2, 3)


# =============================================================================
# CLASSES & COLORS
# =============================================================================

CLASS_NAMES = [
    "",                   # 0
    "Building",           # 1
    "Impervious_Surface", # 2
    "Cropland",           # 3
    "Intensive_Culture",  # 4
    "Grassland_Garden",   # 5
    "Tree_Canopy",        # 6
    "Water",              # 7
    "Railway",            # 8
]

CLASS_COLORS = np.array([
    [ 20,  20,  20],  # 0
    [220,  60,  60],  # 1 Building
    [180, 180, 180],  # 2 Impervious_Surface
    [230, 210,  50],  # 3 Cropland
    [255, 155,  30],  # 4 Intensive_Culture
    [100, 200,  90],  # 5 Grassland_Garden
    [ 30, 120,  30],  # 6 Tree_Canopy
    [ 60, 140, 220],  # 7 Water
    [130,  60, 180],  # 8 Railway
], dtype=np.uint8)

BLOB_HIGHLIGHT_COLOR = (255, 220, 0)
BLOB_FILL_ALPHA_MASK = 0.20
BLOB_FILL_ALPHA_RGB  = 0.0
BLOB_BORDER_ITERS    = 1


# =============================================================================
# IMAGE HELPERS  (same as generate_queue.py)
# =============================================================================

def img_to_rgb_uint8(img_hwc: np.ndarray) -> np.ndarray:
    def stretch(arr):
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255)
        else:
            arr = np.zeros_like(arr)
        return arr.astype(np.uint8)
    r, g, b = Config.RGB_CHANNELS
    return np.stack([stretch(img_hwc[:, :, r]),
                     stretch(img_hwc[:, :, g]),
                     stretch(img_hwc[:, :, b])], axis=-1)


def mask_to_color_rgb(mask_hw: np.ndarray) -> np.ndarray:
    idx = np.clip(mask_hw, 0, 8).astype(np.int32)
    return CLASS_COLORS[idx]


def apply_blob_highlight(rgb_img: np.ndarray, blob_mask: np.ndarray,
                         fill_alpha: float = BLOB_FILL_ALPHA_MASK) -> np.ndarray:
    result = rgb_img.astype(np.float32)
    color  = BLOB_HIGHLIGHT_COLOR
    if fill_alpha > 0.0:
        for c, cv in enumerate(color):
            result[:, :, c] = np.where(
                blob_mask,
                result[:, :, c] * (1.0 - fill_alpha) + cv * fill_alpha,
                result[:, :, c],
            )
    border = binary_dilation(blob_mask, iterations=BLOB_BORDER_ITERS) & ~blob_mask
    for c, cv in enumerate(color):
        result[:, :, c] = np.where(border, float(cv), result[:, :, c])
    return result.clip(0, 255).astype(np.uint8)


def save_overlays(snippet_name: str, blob_idx: int, img_raw: np.ndarray,
                  gt_mask_hw: np.ndarray, blob_mask: np.ndarray,
                  out_dir: str, pred_mask_hw: np.ndarray | None = None) -> dict:
    """
    Save overlay PNGs. pred_mask_hw is optional (None when model is skipped).
    Always shows full 512×512 snippet for spatial context.
    """
    prefix   = f"{snippet_name}_ic{blob_idx:03d}"
    rgb_full = img_to_rgb_uint8(img_raw)
    gt_full  = mask_to_color_rgb(gt_mask_hw)

    paths = {
        "overlay_rgb":   os.path.join(out_dir, f"{prefix}_rgb.png"),
        "overlay_gt":    os.path.join(out_dir, f"{prefix}_gt.png"),
        "overlay_pred":  None,
        "blob_mask_npy": os.path.join(out_dir, f"{prefix}_blobmask.npy"),
    }

    Image.fromarray(
        apply_blob_highlight(rgb_full, blob_mask, fill_alpha=BLOB_FILL_ALPHA_RGB)
    ).save(paths["overlay_rgb"])

    Image.fromarray(
        apply_blob_highlight(gt_full, blob_mask, fill_alpha=BLOB_FILL_ALPHA_MASK)
    ).save(paths["overlay_gt"])

    if pred_mask_hw is not None:
        pred_full = mask_to_color_rgb(pred_mask_hw)
        paths["overlay_pred"] = os.path.join(out_dir, f"{prefix}_pred.png")
        Image.fromarray(
            apply_blob_highlight(pred_full, blob_mask, fill_alpha=BLOB_FILL_ALPHA_MASK)
        ).save(paths["overlay_pred"])

    np.save(paths["blob_mask_npy"], blob_mask)
    return paths


# =============================================================================
# BLOB EXTRACTION
# =============================================================================

def extract_ic_blobs(
    snippet_name: str,
    img_raw: np.ndarray,
    gt_mask_hw: np.ndarray,
    out_dir: str,
    img_npy_path: str,
    mask_npy_path: str,
    pred_probs: np.ndarray | None = None,
) -> list:
    """
    Find all Intensive_Culture (class 4) connected components in the GT mask
    and create one queue entry per blob.

    If pred_probs is given, the model's prediction and dominant predicted class
    per blob are included in the entry for context.
    """
    # Binary mask: pixels labelled as AUDIT_CLASS
    ic_mask = (gt_mask_hw == Config.AUDIT_CLASS_ID)
    if not ic_mask.any():
        return []

    labeled, n_features = scipy_label(ic_mask)
    if n_features == 0:
        return []

    # Compute blob sizes upfront; sort largest first
    blob_sizes = [
        (feat_id, int((labeled == feat_id).sum()))
        for feat_id in range(1, n_features + 1)
    ]
    blob_sizes.sort(key=lambda x: x[1], reverse=True)

    # Optional: prepare prediction arrays
    pred_class_hw  = None
    if pred_probs is not None:
        pred_class_hw = (np.argmax(pred_probs, axis=-1) + 1).astype(np.uint8)

    entries      = []
    blob_counter = 0

    for feat_id, size in blob_sizes:
        if blob_counter >= Config.MAX_BLOBS_PER_SNIPPET:
            break
        if size < Config.MIN_BLOB_SIZE_PX:
            continue   # remaining blobs are smaller (sorted), so we can break
            break

        blob = labeled == feat_id
        rows, cols = np.where(blob)
        bbox = [int(rows.min()), int(cols.min()),
                int(rows.max()), int(cols.max())]

        # Dominant model prediction inside this blob (for context)
        pred_class_id   = None
        pred_class_name = None
        pred_confidence = None
        if pred_class_hw is not None:
            # Most common predicted class in the blob
            unique, counts = np.unique(pred_class_hw[blob], return_counts=True)
            dominant_idx    = int(unique[np.argmax(counts)])
            pred_class_id   = dominant_idx
            pred_class_name = CLASS_NAMES[dominant_idx] if 0 <= dominant_idx <= 8 else str(dominant_idx)
            # Mean confidence of the dominant class prediction
            dominant_conf   = np.max(pred_probs, axis=-1)[blob]
            pred_confidence = round(float(dominant_conf.mean()), 4)

        paths = save_overlays(
            snippet_name  = snippet_name,
            blob_idx      = blob_counter,
            img_raw       = img_raw,
            gt_mask_hw    = gt_mask_hw,
            blob_mask     = blob,
            out_dir       = out_dir,
            pred_mask_hw  = pred_class_hw,
        )

        entry = {
            # review_mode tells labeling_review.py to show the IC-specific buttons
            "review_mode":       "class_audit",
            "audit_class_id":    Config.AUDIT_CLASS_ID,
            "audit_class_name":  Config.AUDIT_CLASS_NAME,
            "target_class_id":   3,      # Cropland – the likely correct label
            "target_class_name": "Cropland",

            "region_id":         f"{snippet_name}_ic{blob_counter:03d}",
            "snippet":           snippet_name,
            "label_class_id":    Config.AUDIT_CLASS_ID,
            "label_class_name":  Config.AUDIT_CLASS_NAME,
            "pred_class_id":     pred_class_id,
            "pred_class_name":   pred_class_name,
            "confidence_mean":   pred_confidence,
            "confidence_min":    None,
            "blob_size_px":      size,
            "bbox":              bbox,
            "overlay_rgb_path":  paths["overlay_rgb"],
            "overlay_gt_path":   paths["overlay_gt"],
            "overlay_pred_path": paths["overlay_pred"],
            "blob_mask_npy":     paths["blob_mask_npy"],
            "img_npy_path":      img_npy_path,
            "mask_npy_path":     mask_npy_path,
        }
        entries.append(entry)
        blob_counter += 1

    return entries


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Queue all Intensive_Culture GT blobs for manual review."
    )
    p.add_argument("--model",    default=Config.MODEL_PATH,
                   help="Path to trained model (optional, for prediction context)")
    p.add_argument("--no-model", action="store_true",
                   help="Skip model inference entirely (faster, no pred overlay)")
    p.add_argument("--min-blob", type=int, default=Config.MIN_BLOB_SIZE_PX,
                   help=f"Min blob size in px (default: {Config.MIN_BLOB_SIZE_PX})")
    return p.parse_args()


def main():
    args = parse_args()
    Config.MODEL_PATH        = args.model
    Config.MIN_BLOB_SIZE_PX  = args.min_blob
    use_model = not args.no_model and os.path.exists(Config.MODEL_PATH)

    print("=" * 70)
    print("IC CLASS AUDIT QUEUE GENERATOR")
    print("=" * 70)
    print(f"Audit class:    {Config.AUDIT_CLASS_ID} – {Config.AUDIT_CLASS_NAME}")
    print(f"Target label:   3 – Cropland  (suggested relabel)")
    print(f"Min blob size:  {Config.MIN_BLOB_SIZE_PX} px")
    print(f"Model context:  {'yes – ' + Config.MODEL_PATH if use_model else 'no (--no-model)'}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.OVERLAYS_DIR, exist_ok=True)

    # Load model (optional)
    model = None
    mean  = None
    std   = None

    if use_model:
        print("\nLoading model …")
        model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
        print(f"  Input:  {model.input_shape}")
        with open(Config.NORM_STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mean = np.array(stats["mean"], dtype=np.float32)
        std  = np.where(
            np.array(stats["std"], dtype=np.float32) == 0, 1.0,
            np.array(stats["std"], dtype=np.float32)
        ).astype(np.float32)

    # Collect snippet files
    img_files = sorted([
        os.path.join(Config.IMG_PATH, f)
        for f in os.listdir(Config.IMG_PATH) if f.endswith(".npy")
    ])
    mask_files = sorted([
        os.path.join(Config.MASK_PATH, f)
        for f in os.listdir(Config.MASK_PATH) if f.endswith(".npy")
    ])
    if len(img_files) != len(mask_files):
        raise ValueError(f"Image/mask mismatch: {len(img_files)} vs {len(mask_files)}")

    n_snippets = len(img_files)
    print(f"\n{n_snippets} snippets found.")
    print("Scanning GT masks for Intensive_Culture blobs …\n")

    queue       = []
    total_blobs = 0
    ic_snippets = 0  # snippets that contain at least one IC blob

    # Batch accumulators (only used when model is active)
    b_imgs, b_raws, b_masks, b_names = [], [], [], []
    b_img_paths, b_mask_paths = [], []

    def flush_with_model():
        nonlocal total_blobs, ic_snippets
        if not b_imgs:
            return
        arr       = np.stack(b_imgs, axis=0)
        preds_all = model.predict(arr, batch_size=Config.BATCH_SIZE, verbose=0)
        for i in range(len(b_imgs)):
            entries = extract_ic_blobs(
                snippet_name  = b_names[i],
                img_raw       = b_raws[i],
                gt_mask_hw    = b_masks[i],
                out_dir       = Config.OVERLAYS_DIR,
                img_npy_path  = b_img_paths[i],
                mask_npy_path = b_mask_paths[i],
                pred_probs    = preds_all[i],
            )
            if entries:
                ic_snippets += 1
            queue.extend(entries)
            total_blobs += len(entries)
        b_imgs.clear(); b_raws.clear(); b_masks.clear(); b_names.clear()
        b_img_paths.clear(); b_mask_paths.clear()

    for idx, (img_path, mask_path) in enumerate(zip(img_files, mask_files)):
        mask_raw = np.load(mask_path, allow_pickle=False)
        mask_hw  = (mask_raw[:, :, 0] if mask_raw.ndim == 3 else mask_raw).astype(np.uint8)

        # Quick check: skip snippets with no IC pixels at all
        if not (mask_hw == Config.AUDIT_CLASS_ID).any():
            # Still need to flush if batch is full (model path)
            if use_model and len(b_imgs) == Config.BATCH_SIZE:
                flush_with_model()
            done = idx + 1
            pct  = done / n_snippets * 100
            bar  = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r  [{bar}] {done}/{n_snippets} ({pct:.0f}%)  IC blobs: {total_blobs}", end="")
            continue

        img_raw = np.load(img_path, allow_pickle=False).astype(np.float32)

        if use_model:
            # Batch inference path
            img_norm = (img_raw - mean) / std
            b_imgs.append(img_norm)
            b_raws.append(img_raw)
            b_masks.append(mask_hw)
            b_names.append(Path(img_path).stem)
            b_img_paths.append(img_path)
            b_mask_paths.append(mask_path)

            if len(b_imgs) == Config.BATCH_SIZE or idx == n_snippets - 1:
                flush_with_model()
        else:
            # No-model path: direct blob extraction, no inference
            entries = extract_ic_blobs(
                snippet_name  = Path(img_path).stem,
                img_raw       = img_raw,
                gt_mask_hw    = mask_hw,
                out_dir       = Config.OVERLAYS_DIR,
                img_npy_path  = img_path,
                mask_npy_path = mask_path,
                pred_probs    = None,
            )
            if entries:
                ic_snippets += 1
            queue.extend(entries)
            total_blobs += len(entries)

        done = idx + 1
        pct  = done / n_snippets * 100
        bar  = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {done}/{n_snippets} ({pct:.0f}%)  IC blobs: {total_blobs}", end="")

    print()

    # Sort by blob size descending: largest fields first (most impactful to review)
    queue.sort(key=lambda e: e["blob_size_px"], reverse=True)

    with open(Config.QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Snippets scanned:      {n_snippets}")
    print(f"Snippets with IC:      {ic_snippets}")
    print(f"IC blobs in queue:     {total_blobs}")
    print(f"Queue saved:           {Config.QUEUE_FILE}")

    if total_blobs > 0:
        sizes = [e["blob_size_px"] for e in queue]
        print(f"Blob size range:       {min(sizes):,} – {max(sizes):,} px")
        print(f"Median blob size:      {int(np.median(sizes)):,} px")

        if use_model:
            # Show what the model predicted for IC regions
            from collections import Counter
            pred_counts = Counter(
                e["pred_class_name"] for e in queue if e["pred_class_name"]
            )
            print("\nModel predictions inside IC blobs (dominant class per blob):")
            for cls, cnt in pred_counts.most_common():
                bar = "█" * int(cnt / max(pred_counts.values()) * 30)
                print(f"  {cls:<25} {bar}  {cnt}")

    print(f"\nNext step:")
    print(f"  Set QUEUE_FILE = '{Config.QUEUE_FILE}' in labeling_review.py")
    print(f"  streamlit run labeling_review.py")


if __name__ == "__main__":
    main()