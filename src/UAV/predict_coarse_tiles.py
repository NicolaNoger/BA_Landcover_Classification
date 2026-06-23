"""
Coarse-stage inference: classify whole orthophoto tiles with the trained aerial
DeepLabV3+ model and write georeferenced class maps + colour previews.

This is the first step of the UAV cascade. The class maps produced here are the
semantic-context source for the UAV stage: uav_image_snipper.py resamples them
(nearest-neighbour) onto the UAV grid and stacks the coarse class as an extra
channel next to UAV RGB + nDSM.

Runs on the HPC (TensorFlow 2.11, GPU) like train_deeplab.py. Paths are
auto-detected (HPC root takes priority if present, else local).

For each input tile it:
  1. reads the 8-band enriched stack and drops band index 6 (number_of_returns)
     to match the 7-channel training input,
  2. z-score normalises with the train-only normalization_stats.json,
  3. runs sliding-window inference (512x512, stride 256) and averages the
     softmax probabilities over the overlap (kept in RAM; ~5 GB/tile),
  4. writes a single-band uint8 class GeoTIFF (classes 1-8, 0 = nodata),
  5. writes a colourised, downsampled PNG preview of the whole classified tile.

Output:
  data/processed/uav_coarse_context/coarse_class_tile{N}.tif
  data/processed/uav_coarse_context/coarse_class_tile{N}_preview.png
"""

import os
import json

import numpy as np
import rasterio
from PIL import Image

import tensorflow as tf

from deeplab_v3plus import build_deeplabv3plus


# -----------------------------------------------------------------------------
# CONFIG (auto-detected HPC / local)
# -----------------------------------------------------------------------------

class Config:
    _HPC_ROOT = "/cfs/earth/scratch/nogernic/BA_2026"
    _LOCAL_ROOT = "A:/STUDIUM/06_Fruelingssemester26/BA"
    PROJECT_ROOT = _HPC_ROOT if os.path.exists(_HPC_ROOT) else _LOCAL_ROOT

    # Trained aerial (coarse) model whose predictions become UAV context.
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "deeplab_20260618_133209")
    CHECKPOINT = os.path.join(MODEL_DIR, "best_model_ckpt")  # TF checkpoint prefix

    # Input enriched stacks (8-band) and the train-only normalization stats.
    # Several layouts are tried so the same script works on HPC and locally.
    STACK_CANDIDATES = [
        os.path.join(PROJECT_ROOT, "data", "processed", "layer_stacks_enriched"),
        os.path.join(PROJECT_ROOT, "data", "layer_stacks_enriched"),
        os.path.join(PROJECT_ROOT, "data", "images"),
    ]
    STATS_CANDIDATES = [
        os.path.join(PROJECT_ROOT, "data", "processed", "training_data", "normalization_stats.json"),
        os.path.join(PROJECT_ROOT, "data", "training_data", "normalization_stats.json"),
    ]

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "uav_coarse_context")

    TILE_NUMBERS = [24, 25]           # tiles overlapping the UAV ROI
    STACK_NAME = "clipped_stacked_buffered_{n}_plus_lidar_ndvi.tif"

    DROP_CHANNEL_IDX = 6              # number_of_returns -> 7-channel input
    NUM_CLASSES = 8
    INPUT_SHAPE = (512, 512, 7)

    WINDOW = 512
    STRIDE = 256
    BATCH = 8

    PREVIEW_MAX_SIDE = 4096          # downsample colour PNG to this max side


# Index 0 = nodata; 1..8 = the project tab10 class palette, identical to the one in
# evaluate.py / train_deeplab.py so coarse maps share the figure colors.
CLASS_COLORS_VIZ = np.array([
    [  0,   0,   0],   # 0: nodata
    [ 31, 119, 180],   # 1: Building
    [255, 127,  14],   # 2: Impervious_Surface
    [ 44, 160,  44],   # 3: Cropland
    [214,  39,  40],   # 4: Intensive_Culture
    [148, 103, 189],   # 5: Grassland_Garden
    [140,  86,  75],   # 6: Tree_Canopy
    [227, 119, 194],   # 7: Water
    [127, 127, 127],   # 8: Railway
], dtype=np.uint8)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _first_existing(candidates, what):
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"None of the {what} candidates exist: {candidates}")


def load_stats(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    std = np.where(std == 0.0, 1.0, std).astype(np.float32)
    return mean, std


def window_starts(size, window, stride):
    """Top-left offsets covering [0, size), with the last window flush to the edge."""
    if size <= window:
        return [0]
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def predict_tile(model, stack_norm, valid_mask, cfg):
    """
    Sliding-window inference with softmax averaging over overlaps.

    stack_norm: (H, W, 7) float32, already normalised.
    valid_mask: (H, W) bool, True where the tile has data.
    Returns (H, W) uint8 class map (1..NUM_CLASSES, 0 = nodata).
    """
    h, w, _ = stack_norm.shape
    win, stride = cfg.WINDOW, cfg.STRIDE

    prob_sum = np.zeros((h, w, cfg.NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    positions = [(y, x)
                 for y in window_starts(h, win, stride)
                 for x in window_starts(w, win, stride)]

    batch_imgs, batch_pos = [], []

    def flush():
        if not batch_imgs:
            return
        preds = model.predict(np.stack(batch_imgs, axis=0), verbose=0)  # (B,win,win,C)
        for (yy, xx), p in zip(batch_pos, preds):
            prob_sum[yy:yy + win, xx:xx + win] += p
            counts[yy:yy + win, xx:xx + win] += 1.0
        batch_imgs.clear()
        batch_pos.clear()

    total = len(positions)
    for i, (y, x) in enumerate(positions):
        if not valid_mask[y:y + win, x:x + win].any():
            continue  # fully nodata window -> skip
        batch_imgs.append(stack_norm[y:y + win, x:x + win])
        batch_pos.append((y, x))
        if len(batch_imgs) >= cfg.BATCH:
            flush()
        if (i + 1) % 200 == 0:
            print(f"    window {i + 1}/{total}")
    flush()

    counts_safe = np.maximum(counts, 1.0)
    avg = prob_sum / counts_safe[..., None]
    class_map = (np.argmax(avg, axis=-1) + 1).astype(np.uint8)
    class_map[(counts == 0) | (~valid_mask)] = 0
    return class_map


def save_class_geotiff(class_map, src_profile, out_path):
    profile = src_profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0, compress="LZW")
    profile.pop("photometric", None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(class_map, 1)


def save_preview_png(class_map, out_path, max_side):
    h, w = class_map.shape
    scale = max(h, w) / float(max_side)
    if scale > 1.0:
        small = Image.fromarray(class_map).resize(
            (max(1, int(w / scale)), max(1, int(h / scale))), Image.NEAREST
        )
        class_small = np.asarray(small)
    else:
        class_small = class_map
    rgb = CLASS_COLORS_VIZ[np.clip(class_small, 0, len(CLASS_COLORS_VIZ) - 1)]
    Image.fromarray(rgb).save(out_path)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # GPU memory growth. Inference runs in float32 (default policy).
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")
    print(f"Devices: {[g.name for g in gpus] or 'CPU'}")

    stack_dir = _first_existing(cfg.STACK_CANDIDATES, "stack dir")
    stats_path = _first_existing(cfg.STATS_CANDIDATES, "stats")
    mean, std = load_stats(stats_path)
    print(f"Stack dir: {stack_dir}")
    print(f"Stats:     {stats_path}  (channels={len(mean)})")
    assert len(mean) == cfg.INPUT_SHAPE[-1], "Stats channel count != model input channels."

    print("Building model and loading checkpoint...")
    model = build_deeplabv3plus(cfg.INPUT_SHAPE, cfg.NUM_CLASSES)
    model.load_weights(cfg.CHECKPOINT)
    print(f"Loaded weights: {cfg.CHECKPOINT}")

    for n in cfg.TILE_NUMBERS:
        stack_path = os.path.join(stack_dir, cfg.STACK_NAME.format(n=n))
        print(f"\n=== Tile {n}: {stack_path} ===")

        with rasterio.open(stack_path) as src:
            stack = src.read().transpose(1, 2, 0)  # (H, W, 8)
            profile = src.profile

        # Drop number_of_returns -> 7 channels, in training order.
        stack = np.delete(stack, cfg.DROP_CHANNEL_IDX, axis=2).astype(np.float32)

        # Valid = any channel non-zero (clipped tiles are zero-filled outside).
        valid_mask = np.any(stack != 0, axis=2)
        print(f"  size {stack.shape[1]}x{stack.shape[0]} | valid px {valid_mask.mean():.1%}")

        stack_norm = (stack - mean) / std
        del stack

        class_map = predict_tile(model, stack_norm, valid_mask, cfg)
        del stack_norm

        tif_out = os.path.join(cfg.OUTPUT_DIR, f"coarse_class_tile{n}.tif")
        png_out = os.path.join(cfg.OUTPUT_DIR, f"coarse_class_tile{n}_preview.png")
        save_class_geotiff(class_map, profile, tif_out)
        save_preview_png(class_map, png_out, cfg.PREVIEW_MAX_SIDE)
        print(f"  -> {tif_out}")
        print(f"  -> {png_out}")

        uniq, cnts = np.unique(class_map, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(uniq, cnts)}
        print(f"  class pixel counts: {dist}")

    print("\nDone. Class maps are the coarse-context source for uav_image_snipper.py.")


if __name__ == "__main__":
    main()
