import json
import os
import re
from pathlib import Path

import numpy as np
import rasterio

"""
Image Snipper: extract and save 512x512 snippets from image/mask tiles.

Pipeline design:
- Fixed tile-based split: one tile is reserved as test (tile #21), all others are train.
- Mean/Std are computed ONLY from train tiles.
- Number_of_Returns channel is dropped (index 6, 0-based).
- Snippets are stored as raw uint16/uint8 on disk (no float32 normalization on disk).
- Normalization is deferred to the DataLoader at training time.
"""


def extract_number(text):
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0


def tile_id(file_path):
    stem = Path(file_path).stem.lower()
    stem = re.sub(r"^(clipped_stacked_buffered_|mask_)", "", stem)
    return stem


def resolve_image_dir(project_root):
    images_base = os.path.join(project_root, "data/aerial/Buffered_images/images")
    candidates = [
        images_base,
        os.path.join(images_base, "layer_stacks_enriched"),
        os.path.join(images_base, "NIR_RGB_nDSM"),
    ]
    for cand in candidates:
        if os.path.isdir(cand) and any(f.lower().endswith(".tif") for f in os.listdir(cand)):
            return cand
    raise FileNotFoundError("No image directory with .tif files found.")


def resolve_mask_dir(project_root):
    candidates = [
        os.path.join(project_root, "data/aerial/Buffered_images/Masks"),
        os.path.join(project_root, "data/aerial/Buffered_images/masks"),
    ]
    for cand in candidates:
        if os.path.isdir(cand) and any(f.lower().endswith(".tif") for f in os.listdir(cand)):
            return cand
    raise FileNotFoundError("No mask directory with .tif files found.")


def collect_pairs(image_dir, mask_dir):
    img_files = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(".tif")],
        key=lambda p: extract_number(os.path.basename(p)),
    )
    mask_files = sorted(
        [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(".tif")],
        key=lambda p: extract_number(os.path.basename(p)),
    )

    img_by_tile = {tile_id(p): p for p in img_files}
    mask_by_tile = {tile_id(p): p for p in mask_files}
    common_tiles = sorted(set(img_by_tile).intersection(mask_by_tile), key=extract_number)

    pairs = [(tid, img_by_tile[tid], mask_by_tile[tid]) for tid in common_tiles]
    return pairs


def to_image_uint16(arr_hwc):
    if np.issubdtype(arr_hwc.dtype, np.integer):
        return arr_hwc.astype(np.uint16)
    arr = np.clip(arr_hwc, 0, np.iinfo(np.uint16).max)
    return arr.astype(np.uint16)


def to_mask_uint8(arr_hw_or_hwc):
    if arr_hw_or_hwc.ndim == 2:
        arr = arr_hw_or_hwc[:, :, np.newaxis]
    else:
        arr = arr_hw_or_hwc
    arr = np.clip(arr, 0, np.iinfo(np.uint8).max)
    return arr.astype(np.uint8)


if __name__ == "__main__":
    project_root = "A:/STUDIUM/06_Fruelingssemester26/BA/"
    output_root = os.path.join(project_root, "data/processed/training_data")

    train_img_out = os.path.join(output_root, "train/img_snippets")
    train_mask_out = os.path.join(output_root, "train/mask_snippets")
    test_img_out = os.path.join(output_root, "test/img_snippets")
    test_mask_out = os.path.join(output_root, "test/mask_snippets")
    stats_path = os.path.join(output_root, "normalization_stats.json")

    for d in [train_img_out, train_mask_out, test_img_out, test_mask_out]:
        os.makedirs(d, exist_ok=True)

    DROP_CHANNEL_IDX = 6  # Number_of_Returns
    SNIP_SIZE = 512
    STRIDE = 256
    TEST_TILE_NUMBER = 21

    image_dir = resolve_image_dir(project_root)
    mask_dir = resolve_mask_dir(project_root)
    pairs = collect_pairs(image_dir, mask_dir)

    if not pairs:
        raise RuntimeError("No paired image/mask tiles found.")

    test_pairs = []
    train_pairs = []
    for tid, img, msk in pairs:
        if extract_number(tid) == TEST_TILE_NUMBER:
            test_pairs.append((tid, img, msk))
        else:
            train_pairs.append((tid, img, msk))

    if len(test_pairs) == 0:
        raise RuntimeError(f"No tile matched TEST_TILE_NUMBER={TEST_TILE_NUMBER}.")

    print(f"Total paired tiles: {len(pairs)}")
    print(f"Train tiles: {len(train_pairs)} | Test tiles: {len(test_pairs)} (tile #{TEST_TILE_NUMBER})")

    print("\nPass 1/2: computing train-only mean/std...")
    total_sum = None
    total_sq_sum = None
    total_count = 0

    for idx, (tid, img_file, _) in enumerate(train_pairs, start=1):
        with rasterio.open(img_file) as src:
            image = src.read().transpose((1, 2, 0))

        image = np.delete(image, DROP_CHANNEL_IDX, axis=2)
        image = image.astype(np.float64)

        if total_sum is None:
            channels = image.shape[-1]
            total_sum = np.zeros(channels, dtype=np.float64)
            total_sq_sum = np.zeros(channels, dtype=np.float64)
            print(f"  Using {channels} channels (dropped index {DROP_CHANNEL_IDX}).")

        h, w, _ = image.shape
        for y in range(0, h - SNIP_SIZE + 1, STRIDE):
            for x in range(0, w - SNIP_SIZE + 1, STRIDE):
                snip = image[y:y + SNIP_SIZE, x:x + SNIP_SIZE]
                total_sum += snip.sum(axis=(0, 1))
                total_sq_sum += np.square(snip).sum(axis=(0, 1))
                total_count += snip.shape[0] * snip.shape[1]

        print(f"  [{idx}/{len(train_pairs)}] stats from tile {tid}")

    mean = total_sum / max(total_count, 1)
    variance = total_sq_sum / max(total_count, 1) - np.square(mean)
    std = np.sqrt(np.clip(variance, 0.0, None))
    std = np.where(std == 0.0, 1.0, std)

    stats = {
        "description": "Train-only normalization stats for on-the-fly DataLoader normalization",
        "test_tile_number": TEST_TILE_NUMBER,
        "drop_channel_index": DROP_CHANNEL_IDX,
        "snip_size": SNIP_SIZE,
        "stride": STRIDE,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved normalization stats:")
    print(f"  {stats_path}")
    print(f"  mean={np.array2string(mean, precision=4)}")
    print(f"  std ={np.array2string(std, precision=4)}")

    print("\nPass 2/2: writing raw snippets (uint16 images, uint8 masks)...")
    train_snips = 0
    test_snips = 0

    def write_snips(split_pairs, out_img_dir, out_mask_dir, split_name):
        nonlocal_train = 0
        for idx, (tid, img_file, mask_file) in enumerate(split_pairs, start=1):
            with rasterio.open(img_file) as img_src, rasterio.open(mask_file) as mask_src:
                image = img_src.read().transpose((1, 2, 0))
                mask = mask_src.read(1)

            image = np.delete(image, DROP_CHANNEL_IDX, axis=2)
            image = to_image_uint16(image)
            mask = to_mask_uint8(mask)

            h, w, _ = image.shape
            tile_snips = 0
            for y in range(0, h - SNIP_SIZE + 1, STRIDE):
                for x in range(0, w - SNIP_SIZE + 1, STRIDE):
                    img_snip = image[y:y + SNIP_SIZE, x:x + SNIP_SIZE]
                    mask_snip = mask[y:y + SNIP_SIZE, x:x + SNIP_SIZE]

                    out_name = f"{split_name}_{tid}_{tile_snips:04d}.npy"
                    np.save(os.path.join(out_img_dir, f"img_{out_name}"), img_snip)
                    np.save(os.path.join(out_mask_dir, f"mask_{out_name}"), mask_snip)

                    tile_snips += 1
                    nonlocal_train += 1

            print(f"  [{idx}/{len(split_pairs)}] {split_name} tile {tid}: {tile_snips} snippets")
        return nonlocal_train

    train_snips = write_snips(train_pairs, train_img_out, train_mask_out, "train")
    test_snips = write_snips(test_pairs, test_img_out, test_mask_out, "test")

    print("\nDone.")
    print(f"  Train snippets: {train_snips} -> {train_img_out}")
    print(f"  Test snippets:  {test_snips} -> {test_img_out}")
    print("  Snippets are raw (not normalized). Normalize on load in DataLoader.")
