"""
generate_queue.py  –  Queue generator for label correction

Pipeline:
    1. Load trained model
    2. Run inference over all test snippets (batch inference)
    3. For each snippet: flag pixels with confidence > THRESHOLD and pred != GT
    4. Extract connected components (blobs) → one queue entry per blob
    5. Save overlay PNGs (RGB / GT mask / Pred mask, blob highlighted)
    6. Write review_queue.json → input for labeling_review.py

Usage:
        python generate_queue.py
        python generate_queue.py --model path/to/final_model --threshold 0.92
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
from skimage.morphology import reconstruction, binary_closing, disk
import tensorflow as tf


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    PROJECT_ROOT       = "/cfs/earth/scratch/nogernic/BA_2026"
    DATA_PATH          = os.path.join(PROJECT_ROOT, "data", "training_data")

    TEST_IMG_PATH      = os.path.join(DATA_PATH, "test", "img_snippets")
    TEST_MASK_PATH     = os.path.join(DATA_PATH, "test", "mask_snippets")
    NORM_STATS_PATH    = os.path.join(DATA_PATH, "normalization_stats.json")

    # Path to saved model (SavedModel folder or .keras)
    # default: latest deeplab run in models/
    MODEL_PATH         = os.path.join(PROJECT_ROOT, "models", "deeplab_20260511_115648", "final_model")

    # Ausgabe
    OUTPUT_DIR         = os.path.join(DATA_PATH, "label_review")
    QUEUE_FILE         = os.path.join(OUTPUT_DIR, "review_queue.json")
    OVERLAYS_DIR       = os.path.join(OUTPUT_DIR, "overlays")

    # Inference
    BATCH_SIZE         = 4
    NUM_CLASSES        = 8
    INPUT_CHANNELS     = 7

    # Blob filtering
    # Hysteresis (Seed & Grow):
    #   seed pixels: high-confidence wrong predictions (definite core)
    #   grow pixels: lower-confidence wrong predictions adjacent to seed
    #   → border pixels at 60-80% confidence are included IF they touch a seed
    #   → isolated low-confidence regions are ignored
    CONFIDENCE_THRESHOLD = 0.80   # Seed threshold – high-confidence core pixels
    GROW_THRESHOLD       = 0.60   # Grow threshold – border pixels connected to seed
    # Morphological closing after grow:
    #   fills small internal gaps (e.g. roof shadows inside a building)
    #   disk(6) = ~12px diameter, fast, does not merge separate objects
    CLOSING_RADIUS       = 15
    MIN_BLOB_SIZE_PX     = 15000   # Minimum blob size after grow + closing
    MAX_BLOBS_PER_PAIR   = 30     # Safety cap per (GT-class, Pred-class) combination

    # Which channels to show as RGB (0-based)
    # Channel order in the stack: NIR=0, R=1, G=2, B=3, ...
    RGB_CHANNELS       = (1, 2, 3)


# =============================================================================
# CLASSES & COLORS
# =============================================================================

CLASS_NAMES = [
    "",                   # 0: nicht belegt
    "Building",           # 1
    "Impervious_Surface", # 2
    "Cropland",           # 3
    "Intensive_Culture",  # 4
    "Grassland_Garden",   # 5
    "Tree_Canopy",        # 6
    "Water",              # 7
    "Railway",            # 8
]

# Colors per class as uint8 RGB — visually distinguishable
CLASS_COLORS = np.array([
    [20,  20,  20 ],  # 0: unknown/background
    [220, 60,  60 ],  # 1: Building
    [180, 180, 180],  # 2: Impervious_Surface
    [230, 210, 50 ],  # 3: Cropland
    [255, 155, 30 ],  # 4: Intensive_Culture
    [100, 200, 90 ],  # 5: Grassland_Garden
    [30,  120, 30 ],  # 6: Tree_Canopy
    [60,  140, 220],  # 7: Water
    [130, 60,  180],  # 8: Railway
], dtype=np.uint8)

# Border and fill use DIFFERENT colors, neither of which is a class color, so
# at least one is always clearly visible — even when the blob lies over the
# yellow Cropland class (where a yellow highlight used to disappear).
BLOB_BORDER_COLOR    = (255, 0, 255)   # magenta, thick outline (all three panels)
BLOB_FILL_COLOR      = (0, 255, 255)   # cyan fill (mask panels only)
BLOB_FILL_ALPHA_MASK = 0.30            # fill strength on GT / pred mask panels
BLOB_FILL_ALPHA_RGB  = 0.0             # no fill on RGB – keep image readable
BLOB_BORDER_ITERS    = 4               # thick ~4-px outline (was 1)


def cname(class_id_1based: int) -> str:
    """Class name for 1-based ID."""
    if 1 <= class_id_1based <= 8:
        return CLASS_NAMES[class_id_1based]
    return f"Klasse_{class_id_1based}"


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def img_to_rgb_uint8(img_chw_float: np.ndarray) -> np.ndarray:
    """
    Multi-channel float32 image (HxWxC) -> uint8 RGB (HxWx3).
    Uses channels R, G, B according to Config.RGB_CHANNELS.
    Applies percentile stretch (2%–98%) per channel for contrast.
    """
    def stretch(arr: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255)
        else:
            arr = np.zeros_like(arr)
        return arr.astype(np.uint8)

    r, g, b = Config.RGB_CHANNELS
    return np.stack([
        stretch(img_chw_float[:, :, r]),
        stretch(img_chw_float[:, :, g]),
        stretch(img_chw_float[:, :, b]),
    ], axis=-1)


def mask_to_color_rgb(mask_hw: np.ndarray) -> np.ndarray:
    """Mask (HxW, values 1-8) -> RGB color image uint8."""
    idx = np.clip(mask_hw, 0, 8).astype(np.int32)
    return CLASS_COLORS[idx]


def apply_blob_highlight(
    rgb_img: np.ndarray,
    blob_mask: np.ndarray,
    fill_alpha: float = BLOB_FILL_ALPHA_MASK,
) -> np.ndarray:
    """
    Apply a cyan semi-transparent fill + thick magenta border over the blob.

    rgb_img:    uint8 HxWx3
    blob_mask:  bool  HxW
    fill_alpha: 0.0 = no fill (border only), >0 = tinted fill

    Border and fill use distinct colors (magenta / cyan), so at least one stays
    visible over any class color. For RGB images pass fill_alpha=0.0 so the
    original image stays fully readable; for mask panels pass
    fill_alpha=BLOB_FILL_ALPHA_MASK for a light cyan tint.
    """
    result = rgb_img.astype(np.float32)

    # Semi-transparent cyan fill (skipped when fill_alpha == 0)
    if fill_alpha > 0.0:
        for c, cv in enumerate(BLOB_FILL_COLOR):
            result[:, :, c] = np.where(
                blob_mask,
                result[:, :, c] * (1.0 - fill_alpha) + cv * fill_alpha,
                result[:, :, c],
            )

    # Thick magenta border: dilate by BLOB_BORDER_ITERS pixels, subtract blob
    border = binary_dilation(blob_mask, iterations=BLOB_BORDER_ITERS) & ~blob_mask
    for c, cv in enumerate(BLOB_BORDER_COLOR):
        result[:, :, c] = np.where(border, float(cv), result[:, :, c])

    return result.clip(0, 255).astype(np.uint8)


def save_overlays(
    snippet_name: str,
    blob_idx: int,
    img_raw: np.ndarray,
    gt_mask_hw: np.ndarray,
    pred_mask_hw: np.ndarray,
    blob_mask: np.ndarray,
    out_dir: str,
) -> dict:
    """
    Save three PNG overlays for the blob, always showing the FULL 512x512 snippet.

    Showing the full snippet (instead of a bounding-box crop) ensures that:
      - Multiple blobs from the same snippet at nearby locations are clearly
        distinguishable (you see where in the full image the blob sits).
      - The user has enough spatial context to judge the label.

    Three panels saved:
      _rgb.png   – true-color image, border only (no fill) so image stays readable
      _gt.png    – ground-truth mask with light fill + border
      _pred.png  – prediction mask with light fill + border

    Returns a dict with absolute paths.
    """
    # Always full snippet – no crop
    rgb_full  = img_to_rgb_uint8(img_raw)
    gt_full   = mask_to_color_rgb(gt_mask_hw)
    pred_full = mask_to_color_rgb(pred_mask_hw)

    prefix = f"{snippet_name}_blob{blob_idx:03d}"
    paths = {
        "overlay_rgb":   os.path.join(out_dir, f"{prefix}_rgb.png"),
        "overlay_gt":    os.path.join(out_dir, f"{prefix}_gt.png"),
        "overlay_pred":  os.path.join(out_dir, f"{prefix}_pred.png"),
        "blob_mask_npy": os.path.join(out_dir, f"{prefix}_blobmask.npy"),
    }

    # RGB: no fill, only a thick magenta border so the image stays fully readable
    Image.fromarray(
        apply_blob_highlight(rgb_full, blob_mask, fill_alpha=BLOB_FILL_ALPHA_RGB)
    ).save(paths["overlay_rgb"])

    # GT and pred masks: light fill + border so the blob region is clearly visible
    Image.fromarray(
        apply_blob_highlight(gt_full, blob_mask, fill_alpha=BLOB_FILL_ALPHA_MASK)
    ).save(paths["overlay_gt"])

    Image.fromarray(
        apply_blob_highlight(pred_full, blob_mask, fill_alpha=BLOB_FILL_ALPHA_MASK)
    ).save(paths["overlay_pred"])

    np.save(paths["blob_mask_npy"], blob_mask)

    return paths


# =============================================================================
# BLOB EXTRACTION
# =============================================================================

def extract_blobs(
    snippet_name: str,
    img_raw: np.ndarray,
    gt_mask_hw: np.ndarray,
    pred_probs: np.ndarray,
    out_dir: str,
    img_npy_path: str,
    mask_npy_path: str,
) -> list:
    """
    Extract flagged blobs from a snippet and create queue entries.

    Returns: list of dicts (one dict per queue entry)
    """
    pred_class_hw = (np.argmax(pred_probs, axis=-1) + 1).astype(np.uint8)
    confidence_hw = np.max(pred_probs, axis=-1).astype(np.float32)

    wrong_class = pred_class_hw != gt_mask_hw

    # --- Hysteresis thresholding (Seed & Grow) ---
    # seed: high-confidence wrong pixels → definite disagreement cores
    # grow: lower-confidence wrong pixels → candidate border pixels
    # reconstruction expands seed into grow, but ONLY where pixels are adjacent.
    # Isolated low-confidence patches (no seed neighbour) are discarded.
    seed_mask = wrong_class & (confidence_hw >= Config.CONFIDENCE_THRESHOLD)
    grow_mask = wrong_class & (confidence_hw >= Config.GROW_THRESHOLD)

    if not seed_mask.any():
        return []

    # seed ⊆ grow is guaranteed since CONFIDENCE_THRESHOLD > GROW_THRESHOLD
    disagree_global = reconstruction(
        seed_mask.astype(np.uint8),
        grow_mask.astype(np.uint8),
        method="dilation",
    ).astype(bool)

    if not disagree_global.any():
        return []

    queue_entries = []
    blob_counter  = 0

    unique_gt_classes = np.unique(gt_mask_hw[disagree_global])

    for gt_cls in unique_gt_classes:
        mask_gt_cls  = disagree_global & (gt_mask_hw == gt_cls)
        unique_preds = np.unique(pred_class_hw[mask_gt_cls])

        for pred_cls in unique_preds:
            if pred_cls == gt_cls:
                continue

            specific = mask_gt_cls & (pred_class_hw == pred_cls)

            # Morphological closing: fills small internal gaps (e.g. roof shadows)
            # Applied only to the bounding box of the current blob for speed.
            # disk(CLOSING_RADIUS) should be small (≤8) to avoid merging
            # separate objects or causing edge artefacts on small crops.
            if Config.CLOSING_RADIUS > 0 and specific.any():
                rows, cols = np.where(specific)
                pad = Config.CLOSING_RADIUS
                y1  = max(0,               int(rows.min()) - pad)
                y2  = min(specific.shape[0], int(rows.max()) + pad + 1)
                x1  = max(0,               int(cols.min()) - pad)
                x2  = min(specific.shape[1], int(cols.max()) + pad + 1)
                crop = specific[y1:y2, x1:x2]
                closed = binary_closing(crop, footprint=disk(Config.CLOSING_RADIUS))
                out = np.zeros_like(specific)
                out[y1:y2, x1:x2] = closed
                specific = out

            labeled, n_features = scipy_label(specific)

            for feat_id in range(1, n_features + 1):
                if blob_counter >= Config.MAX_BLOBS_PER_PAIR * len(unique_gt_classes):
                    break

                blob = labeled == feat_id
                size = int(blob.sum())

                if size < Config.MIN_BLOB_SIZE_PX:
                    continue

                rows, cols = np.where(blob)
                bbox = [int(rows.min()), int(cols.min()),
                        int(rows.max()), int(cols.max())]

                conf_mean = float(confidence_hw[blob].mean())
                conf_min  = float(confidence_hw[blob].min())

                paths = save_overlays(
                    snippet_name  = snippet_name,
                    blob_idx      = blob_counter,
                    img_raw       = img_raw,
                    gt_mask_hw    = gt_mask_hw,
                    pred_mask_hw  = pred_class_hw,
                    blob_mask     = blob,
                    out_dir       = out_dir,
                )

                entry = {
                    "region_id":         f"{snippet_name}_blob{blob_counter:03d}",
                    "snippet":           snippet_name,
                    "label_class_id":    int(gt_cls),
                    "label_class_name":  cname(int(gt_cls)),
                    "pred_class_id":     int(pred_cls),
                    "pred_class_name":   cname(int(pred_cls)),
                    "confidence_mean":   round(conf_mean, 4),
                    "confidence_min":    round(conf_min, 4),
                    "blob_size_px":      size,
                    "bbox":              bbox,
                    "overlay_rgb_path":  paths["overlay_rgb"],
                    "overlay_gt_path":   paths["overlay_gt"],
                    "overlay_pred_path": paths["overlay_pred"],
                    "blob_mask_npy":     paths["blob_mask_npy"],
                    "img_npy_path":      img_npy_path,
                    "mask_npy_path":     mask_npy_path,
                }

                queue_entries.append(entry)
                blob_counter += 1

    return queue_entries


# =============================================================================
# HAUPTPROGRAMM
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Create review_queue.json for labeling review.")
    p.add_argument("--model",      default=Config.MODEL_PATH,
                   help="Path to trained model (SavedModel folder or .keras)")
    p.add_argument("--threshold",  type=float, default=Config.CONFIDENCE_THRESHOLD,
                   help=f"Seed confidence threshold (default: {Config.CONFIDENCE_THRESHOLD})")
    p.add_argument("--grow",       type=float, default=Config.GROW_THRESHOLD,
                   help=f"Grow confidence threshold (default: {Config.GROW_THRESHOLD})")
    p.add_argument("--min-blob",   type=int,   default=Config.MIN_BLOB_SIZE_PX,
                   help=f"Minimum blob size in pixels (default: {Config.MIN_BLOB_SIZE_PX})")
    p.add_argument("--mask-dir",   default=Config.TEST_MASK_PATH,
                   help="Mask directory (default: current test masks). Point to the "
                        "original pre-review masks for a before-review demo queue.")
    p.add_argument("--out-dir",    default=None,
                   help="Output directory for the queue + overlays (default: data/.../label_review).")
    p.add_argument("--limit",      type=int,   default=None,
                   help="Process only N evenly-spaced snippets instead of all (quick demo).")
    return p.parse_args()


def main():
    args = parse_args()
    Config.MODEL_PATH            = args.model
    Config.CONFIDENCE_THRESHOLD  = args.threshold
    Config.GROW_THRESHOLD        = args.grow
    Config.MIN_BLOB_SIZE_PX      = args.min_blob
    Config.TEST_MASK_PATH        = args.mask_dir
    if args.out_dir:
        Config.OUTPUT_DIR   = args.out_dir
        Config.QUEUE_FILE   = os.path.join(args.out_dir, "review_queue.json")
        Config.OVERLAYS_DIR = os.path.join(args.out_dir, "overlays")

    print("=" * 70)
    print("GENERATE REVIEW QUEUE")
    print("=" * 70)
    print(f"Started:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model:          {Config.MODEL_PATH}")
    print(f"Seed threshold: {Config.CONFIDENCE_THRESHOLD:.0%}")
    print(f"Grow threshold: {Config.GROW_THRESHOLD:.0%}")
    print(f"Min blob size:  {Config.MIN_BLOB_SIZE_PX} px")
    print(f"Closing radius: {Config.CLOSING_RADIUS} px")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.OVERLAYS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    if not os.path.exists(Config.MODEL_PATH):
        print(f"\nERROR: Model not found: {Config.MODEL_PATH}")
        print("→ Adjust Config.MODEL_PATH or use --model argument.")
        sys.exit(1)

    print(f"\nLoading model …")
    model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
    print(f"  Input:  {model.input_shape}")
    print(f"  Output: {model.output_shape}")

    # ------------------------------------------------------------------
    # 2. Normalisierungs-Stats laden
    # ------------------------------------------------------------------
    with open(Config.NORM_STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    std  = np.where(std == 0, 1.0, std).astype(np.float32)
    print(f"  Loaded normalization stats from: {Config.NORM_STATS_PATH}")

    # ------------------------------------------------------------------
    # 3. Snippet-Dateien sammeln
    # ------------------------------------------------------------------
    img_files = sorted([
        os.path.abspath(os.path.join(Config.TEST_IMG_PATH, f))
        for f in os.listdir(Config.TEST_IMG_PATH) if f.endswith(".npy")
    ])
    mask_files = sorted([
        os.path.abspath(os.path.join(Config.TEST_MASK_PATH, f))
        for f in os.listdir(Config.TEST_MASK_PATH) if f.endswith(".npy")
    ])

    if len(img_files) != len(mask_files):
        raise ValueError(f"Image/mask mismatch: {len(img_files)} vs {len(mask_files)}")

    if args.limit and args.limit < len(img_files):
        sel = np.linspace(0, len(img_files) - 1, args.limit, dtype=int)
        img_files  = [img_files[i] for i in sel]
        mask_files = [mask_files[i] for i in sel]
        print(f"\nDemo mode: limited to {len(img_files)} evenly-spaced snippets.")

    n_snippets = len(img_files)
    print(f"\n{n_snippets} Test-Snippets gefunden.")
    print(f"Masks from:     {Config.TEST_MASK_PATH}")
    print(f"Output to:      {Config.OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # 4. Batch-Inferenz + Blob-Extraktion
    # ------------------------------------------------------------------
    queue        = []
    total_blobs  = 0
    n_batches    = (n_snippets + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE

    # Akkumulatoren
    b_imgs, b_raws, b_masks, b_names = [], [], [], []
    b_img_paths, b_mask_paths = [], []

    def flush():
        nonlocal total_blobs
        if not b_imgs:
            return

        arr        = np.stack(b_imgs, axis=0)
        preds_all  = model.predict(arr, verbose=0)   # B x H x W x N

        for i in range(len(b_imgs)):
            entries = extract_blobs(
                snippet_name  = b_names[i],
                img_raw       = b_raws[i],
                gt_mask_hw    = b_masks[i],
                pred_probs    = preds_all[i],
                out_dir       = Config.OVERLAYS_DIR,
                img_npy_path  = b_img_paths[i],
                mask_npy_path = b_mask_paths[i],
            )
            queue.extend(entries)
            total_blobs += len(entries)

        b_imgs.clear();  b_raws.clear(); b_masks.clear(); b_names.clear()
        b_img_paths.clear(); b_mask_paths.clear()

    print(f"\nStarte Inferenz über {n_snippets} Snippets in {n_batches} Batches …\n")
    print(f"\nStart inference over {n_snippets} snippets in {n_batches} batches …\n")

    for idx, (img_path, mask_path) in enumerate(zip(img_files, mask_files)):
        # -- Laden --
        img_raw  = np.load(img_path,  allow_pickle=False).astype(np.float32)
        mask_raw = np.load(mask_path, allow_pickle=False)

        # Maske: (H, W, 1) → (H, W)
        if mask_raw.ndim == 3:
            mask_hw = mask_raw[:, :, 0].astype(np.uint8)
        else:
            mask_hw = mask_raw.astype(np.uint8)

        img_norm = (img_raw - mean) / std

        b_imgs.append(img_norm)
        b_raws.append(img_raw)
        b_masks.append(mask_hw)
        b_names.append(Path(img_path).stem)
        b_img_paths.append(img_path)
        b_mask_paths.append(mask_path)

        # -- Batch voll oder letztes Element --
        if len(b_imgs) == Config.BATCH_SIZE or idx == n_snippets - 1:
            flush()
            done = idx + 1
            pct  = done / n_snippets * 100
            bar  = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r  [{bar}] {done}/{n_snippets} ({pct:.0f}%)  Blobs: {total_blobs}", end="")

    print()  # Newline nach Fortschrittsanzeige

    # ------------------------------------------------------------------
    # 5. Queue speichern
    # ------------------------------------------------------------------
    # Nach Confidence (absteigend) sortieren – wichtigste Korrekturen zuerst
    queue.sort(key=lambda e: e["confidence_mean"], reverse=True)

    with open(Config.QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6. Zusammenfassung
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Snippets processed:    {n_snippets}")
    print(f"Blobs in queue:        {total_blobs}")
    print(f"Queue saved:           {Config.QUEUE_FILE}")
    print(f"Overlays:              {Config.OVERLAYS_DIR}/")
    print()

    if total_blobs == 0:
        print("No blobs found.")
        print("   -> Lower confidence threshold (--threshold 0.85)?")
        print("   -> Lower min-blob size (--min-blob 100)?")
    else:
        # Statistik: Häufigste (GT → Pred) Paare
        pairs: dict = {}
        for e in queue:
            key = f"{e['label_class_name']} → {e['pred_class_name']}"
            pairs[key] = pairs.get(key, 0) + 1
        top = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top-10 class conflicts (GT -> Pred):")
        for pair, count in top:
            print(f"  {count:4d}×  {pair}")

    print()
    print("Next step:")
    print(f"  streamlit run labeling_review.py")


if __name__ == "__main__":
    main()