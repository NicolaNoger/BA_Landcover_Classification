import json
import os

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge as rio_merge

"""
UAV Image Snipper: build the high-resolution UAV training stack and cut it into tiles.

This is the UAV-stage analogue of src/U-Net/image_snipper.py. It mirrors that
pipeline (raw tiles on disk, normalization deferred to the DataLoader, train-only
mean/std) but adds the UAV-specific preprocessing for the cascade:

Pipeline design:
- Inputs are a single co-registered UAV survey:
    * RGB orthomosaic (4 bands: R, G, B, Alpha)  ~0.015 m GSD, EPSG:2056
    * DSM (absolute surface elevation, float32)   same grid as the RGB
- The DSM is an *absolute* surface model (~520 m), not a height above ground.
  It is converted to an nDSM by subtracting the SwissALTI3D DTM, which is
  resampled (bilinear) onto the UAV grid per tile.
- Coarse semantic context: the trained primary-stage DeepLabV3+ classifies the
  overlapping orthophoto tiles (predict_coarse_tiles.py -> coarse_class_tile{24,25}.tif).
  Those argmax class maps (0.1 m) are merged and resampled (nearest, since class
  IDs must not be interpolated) onto the UAV grid as one extra channel.
- Final channel stack: [R, G, B, nDSM, Coarse_Class]  (5 channels). NIR / LiDAR
  intensity are not available at UAV scale; the coarse class supplies the
  semantic context of the cascade.
- Tiles are 1024x1024 with stride 512. RGB / coarse-class are stored as small
  integers, nDSM as uint16 centimetres, packed together into one uint16 array per
  tile. Storing raw avoids doubling disk usage; normalization is deferred to the
  DataLoader at training time.
- The mosaic is a single contiguous area, so train/test is a *spatial* hold-out:
  the rightmost TEST_FRACTION (by easting) is the test region, with a one-tile gap
  so overlapping tiles never straddle the train/test boundary (no spatial leakage).
- Tiles whose valid-pixel fraction is below MIN_VALID_FRACTION are dropped
  (the mosaic border is nodata).
- If a label raster aligned to the mosaic grid is provided via MASK_PATH, it is
  tiled in lock-step with the images (identical windows, identical train/test
  split) and saved as uint8 mask snippets. Otherwise mask output is skipped.
"""


# Channel layout of the stored stack.
CHANNELS = ("Red", "Green", "Blue", "nDSM_cm", "Coarse_Class")
NDSM_CHANNEL_IDX = 3
COARSE_CHANNEL_IDX = 4
NDSM_MAX_M = 60.0  # clip nDSM to [0, 60] m before scaling to cm


def compute_windows(width, height, snip_size, stride):
    """Top-left (col_off, row_off) of every full snip_size tile on a stride grid."""
    windows = []
    for row in range(0, height - snip_size + 1, stride):
        for col in range(0, width - snip_size + 1, stride):
            windows.append((col, row))
    return windows


def assign_split(col_off, snip_size, split_x):
    """
    Spatial hold-out by easting. Returns 'train', 'test' or None (gap/buffer).

    Tiles fully left of split_x -> train, fully right -> test. Tiles that cross
    split_x are dropped so train and test never share pixels.
    """
    if col_off + snip_size <= split_x:
        return "train"
    if col_off >= split_x:
        return "test"
    return None


def load_dtm(dtm_path):
    """Read the full DTM once into memory (float32, nodata -> NaN) for per-tile reprojection."""
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            dtm[dtm == nodata] = np.nan
        return dtm, src.transform, src.crs


def load_coarse_context(coarse_paths):
    """
    Merge the coarse class GeoTIFFs (tiles 24 + 25) into one in-memory mosaic.

    Returns (class_array (H, W) uint8, transform, crs) or None if no file exists.
    Class IDs are 1..8, 0 = nodata.
    """
    existing = [p for p in coarse_paths if os.path.isfile(p)]
    if not existing:
        return None
    srcs = [rasterio.open(p) for p in existing]
    try:
        mosaic, transform = rio_merge(srcs, nodata=0)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    return mosaic[0].astype(np.uint8), transform, crs


def ndsm_for_window(window, base_transform, dst_crs, dsm_window, dsm_nodata,
                    dtm, dtm_transform, dtm_crs):
    """
    nDSM (height above ground, metres) for a single tile window.

    DTM is resampled bilinearly onto the tile grid and subtracted from the DSM.
    Returns (ndsm_m, valid_mask) where invalid = DSM nodata or DTM nodata.
    """
    win_h, win_w = dsm_window.shape
    dst_transform = window_transform(window, base_transform)

    dtm_on_grid = np.full((win_h, win_w), np.nan, dtype=np.float32)
    reproject(
        source=dtm,
        destination=dtm_on_grid,
        src_transform=dtm_transform,
        src_crs=dtm_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    dsm_valid = dsm_window != dsm_nodata
    dtm_valid = ~np.isnan(dtm_on_grid)
    valid = dsm_valid & dtm_valid

    ndsm = np.zeros((win_h, win_w), dtype=np.float32)
    ndsm[valid] = dsm_window[valid] - dtm_on_grid[valid]
    return ndsm, valid


def coarse_for_window(window, base_transform, dst_crs, coarse):
    """
    Coarse class IDs resampled (nearest) onto a tile window grid. Returns (H, W) uint8.
    `coarse` is (class_array, transform, crs) or None -> all zeros.
    """
    dst_transform = window_transform(window, base_transform)
    out = np.zeros((window.height, window.width), dtype=np.uint8)
    if coarse is None:
        return out
    class_arr, c_transform, c_crs = coarse
    reproject(
        source=class_arr,
        destination=out,
        src_transform=c_transform,
        src_crs=c_crs,
        src_nodata=0,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return out


def build_stack(rgb_window, alpha_window, ndsm_m, height_valid, coarse_window):
    """
    Pack [R, G, B, nDSM_cm, Coarse_Class] into one uint16 array (H, W, 5).

    Invalid pixels (alpha == 0 or no height) are zeroed in every channel.
    Returns (stack_uint16, valid_mask).
    """
    valid = (alpha_window > 0) & height_valid

    ndsm_cm = np.clip(ndsm_m, 0.0, NDSM_MAX_M) * 100.0
    ndsm_cm = np.clip(ndsm_cm, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    h, w = ndsm_cm.shape
    stack = np.zeros((h, w, len(CHANNELS)), dtype=np.uint16)
    stack[:, :, 0] = rgb_window[0]
    stack[:, :, 1] = rgb_window[1]
    stack[:, :, 2] = rgb_window[2]
    stack[:, :, NDSM_CHANNEL_IDX] = ndsm_cm
    stack[:, :, COARSE_CHANNEL_IDX] = coarse_window.astype(np.uint16)

    stack[~valid] = 0
    return stack, valid


if __name__ == "__main__":
    project_root = "A:/STUDIUM/06_Fruelingssemester26/BA/"

    rgb_path = os.path.join(
        project_root,
        "data/aerial/GT_RGB_mosaic/20260528_RGB_GT_transparent_mosaic_group1.tif",
    )
    dsm_path = os.path.join(
        project_root, "data/aerial/GT_RGB_mosaic/20260528_RGB_GT_dsm.tif"
    )
    dtm_path = os.path.join(project_root, "data/Swissalti/swissalti_merged.tif")

    # Coarse semantic-context class maps (predict_coarse_tiles.py output).
    coarse_dir = os.path.join(project_root, "data/processed/uav_coarse_context")
    coarse_paths = [
        os.path.join(coarse_dir, "coarse_class_tile24.tif"),
        os.path.join(coarse_dir, "coarse_class_tile25.tif"),
    ]

    # Optional label raster aligned to the mosaic grid; mask snippets are written
    # only if this file exists.
    mask_path = os.path.join(
        project_root, "data/aerial/GT_RGB_mosaic/GT_mask_raster.tif"
    )

    output_root = os.path.join(project_root, "data/processed/uav_training_data")
    train_img_out = os.path.join(output_root, "train/img_snippets")
    train_mask_out = os.path.join(output_root, "train/mask_snippets")
    test_img_out = os.path.join(output_root, "test/img_snippets")
    test_mask_out = os.path.join(output_root, "test/mask_snippets")
    stats_path = os.path.join(output_root, "normalization_stats.json")

    for d in [train_img_out, train_mask_out, test_img_out, test_mask_out]:
        os.makedirs(d, exist_ok=True)

    SNIP_SIZE = 1024
    STRIDE = 512
    TEST_FRACTION = 0.20          # rightmost share (by easting) held out as test
    MIN_VALID_FRACTION = 0.50     # drop tiles with too much nodata border

    write_masks = os.path.isfile(mask_path)

    print("Loading DTM into memory...")
    dtm, dtm_transform, dtm_crs = load_dtm(dtm_path)

    print("Loading coarse context...")
    coarse = load_coarse_context(coarse_paths)
    if coarse is None:
        print("  WARNING: no coarse class map found in")
        print(f"    {coarse_dir}")
        print("    -> Coarse_Class channel will be 0. Run predict_coarse_tiles.py first")
        print("       and copy coarse_class_tile24/25.tif here for the full 5-channel stack.")
    else:
        print(f"  Coarse mosaic: {coarse[0].shape[1]}x{coarse[0].shape[0]} px, "
              f"classes {sorted(np.unique(coarse[0]).tolist())}")

    rgb_src = rasterio.open(rgb_path)
    dsm_src = rasterio.open(dsm_path)
    mask_src = rasterio.open(mask_path) if write_masks else None

    assert rgb_src.width == dsm_src.width and rgb_src.height == dsm_src.height, \
        "RGB and DSM grids differ; co-registration assumption violated."
    if mask_src is not None:
        assert mask_src.width == rgb_src.width and mask_src.height == rgb_src.height, \
            "Mask grid differs from the RGB grid; reproject the label raster first."

    base_transform = rgb_src.transform
    dsm_nodata = dsm_src.nodata
    width, height = rgb_src.width, rgb_src.height
    split_x = int(width * (1.0 - TEST_FRACTION))

    windows = compute_windows(width, height, SNIP_SIZE, STRIDE)
    print(f"\nMosaic: {width} x {height} px | candidate tiles: {len(windows)}")
    print(f"Tile: {SNIP_SIZE}px stride {STRIDE} | test = easting beyond x={split_x}px")
    print(f"Masks: {'ON (' + mask_path + ')' if write_masks else 'OFF (no label raster yet)'}")

    # Train-only running stats (valid pixels only).
    total_sum = np.zeros(len(CHANNELS), dtype=np.float64)
    total_sq_sum = np.zeros(len(CHANNELS), dtype=np.float64)
    total_count = 0

    counts = {"train": 0, "test": 0}
    skipped_nodata = 0
    skipped_gap = 0

    print("\nProcessing tiles...")
    for i, (col_off, row_off) in enumerate(windows):
        split = assign_split(col_off, SNIP_SIZE, split_x)
        if split is None:
            skipped_gap += 1
            continue

        window = Window(col_off, row_off, SNIP_SIZE, SNIP_SIZE)

        rgb_window = rgb_src.read([1, 2, 3], window=window)
        alpha_window = rgb_src.read(4, window=window)
        dsm_window = dsm_src.read(1, window=window)

        ndsm_m, height_valid = ndsm_for_window(
            window, base_transform, rgb_src.crs,
            dsm_window, dsm_nodata,
            dtm, dtm_transform, dtm_crs,
        )
        coarse_window = coarse_for_window(window, base_transform, rgb_src.crs, coarse)

        stack, valid = build_stack(rgb_window, alpha_window, ndsm_m, height_valid, coarse_window)

        if valid.mean() < MIN_VALID_FRACTION:
            skipped_nodata += 1
            continue

        if split == "train":
            valid_px = stack[valid].astype(np.float64)  # (N, C)
            total_sum += valid_px.sum(axis=0)
            total_sq_sum += np.square(valid_px).sum(axis=0)
            total_count += valid_px.shape[0]

        idx = counts[split]
        out_name = f"{split}_{row_off:05d}_{col_off:05d}_{idx:04d}.npy"
        img_dir = train_img_out if split == "train" else test_img_out
        np.save(os.path.join(img_dir, f"img_{out_name}"), stack)

        if write_masks:
            mask_window = mask_src.read(1, window=window)
            mask_window = np.clip(mask_window, 0, np.iinfo(np.uint8).max).astype(np.uint8)
            mask_dir = train_mask_out if split == "train" else test_mask_out
            np.save(os.path.join(mask_dir, f"mask_{out_name}"), mask_window[:, :, np.newaxis])

        counts[split] += 1
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(windows)}] train={counts['train']} test={counts['test']}")

    rgb_src.close()
    dsm_src.close()
    if mask_src is not None:
        mask_src.close()

    mean = total_sum / max(total_count, 1)
    variance = total_sq_sum / max(total_count, 1) - np.square(mean)
    std = np.sqrt(np.clip(variance, 0.0, None))
    std = np.where(std == 0.0, 1.0, std)

    stats = {
        "description": "Train-only UAV normalization stats (valid pixels only) for on-the-fly DataLoader normalization",
        "channels": list(CHANNELS),
        "snip_size": SNIP_SIZE,
        "stride": STRIDE,
        "test_fraction": TEST_FRACTION,
        "min_valid_fraction": MIN_VALID_FRACTION,
        "ndsm_max_m": NDSM_MAX_M,
        "ndsm_unit": "centimetres",
        "coarse_context": coarse is not None,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nDone.")
    print(f"  Train tiles: {counts['train']} -> {train_img_out}")
    print(f"  Test tiles:  {counts['test']} -> {test_img_out}")
    print(f"  Skipped (nodata border): {skipped_nodata} | (train/test gap): {skipped_gap}")
    print(f"  Channels: {CHANNELS}")
    print(f"  Stats: {stats_path}")
    print(f"  mean={np.array2string(mean, precision=2)}")
    print(f"  std ={np.array2string(std, precision=2)}")
    if not write_masks:
        print(f"\n  NOTE: no label raster at {mask_path} -> mask snippets not written.")
        print("        Provide a single-band uint8 label raster on the mosaic grid")
        print("        and re-run to also produce aligned mask snippets.")
