"""
Ablation comparison for the UAV cascade stage: coarse-context vs. RGB+nDSM.

Loads the two trained UAV DeepLabV3+ runs produced by train_uav.py
    - "with coarse"  (USE_COARSE=True,  12 input channels)
    - "RGB + nDSM"   (USE_COARSE=False,  4 input channels)
and evaluates BOTH on the exact same UAV test split, then writes the metrics and
figures needed to compare them. The evaluation logic, the tab10 palette and the
PIL dark-theme figure style are identical to src/deeplab/evaluate.py, so the
numbers and the look stay consistent across the whole thesis.

Both models read the same on-disk 5-channel tiles; only the model input assembled
on load differs (the coarse channel is one-hot expanded and appended, or dropped),
so any metric difference is attributable to the coarse semantic context alone.

    Metrics (printed and saved as CSV / TXT)
        - per-class IoU / F1 for each model and their delta (coarse - rgbndsm)
        - mean IoU, pixel accuracy, macro and weighted F1 for each model
        - one confusion matrix per model

    Figures (PNG, rendered directly with PIL)
        - grouped per-class IoU bar chart (with coarse vs. RGB+nDSM)
        - one confusion matrix heatmap per model
        - qualitative RGB | ground truth | pred(coarse) | pred(RGB+nDSM) panels

Usage:
    python compare_models.py                       # newest coarse + rgbndsm runs
    python compare_models.py --coarse-dir ../../models/uav_deeplab_coarse_2026...
                             --rgbndsm-dir ../../models/uav_deeplab_rgbndsm_2026...
    python compare_models.py --split test --num-qualitative 8
"""

import argparse
import csv
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deeplab_v3plus import build_deeplabv3plus
from dataloader import load_normalization_stats, _coarse_to_onehot, N_CONT, COARSE_NUM_CLASSES
from train_uav import Config


# 12-class UAV palette (0-based), identical to the array in train_uav.py so every
# UAV figure shares the same colors. First 8 are the tab10 colors of evaluate.py;
# entries 9-12 cover the refined UAV-only classes.
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

# Dark theme colors (RGB) for the PIL-rendered figures, identical to evaluate.py.
_BG = (26, 26, 46)      # figure background
_PANEL = (22, 33, 62)   # plot-area background
_FG = (235, 235, 245)   # text / ticks
_GRID = (70, 70, 100)   # gridlines

# The two ablation variants, in a fixed order so colors / columns stay stable.
VARIANTS = ("coarse", "rgbndsm")
VARIANT_LABELS = {"coarse": "with coarse", "rgbndsm": "RGB + nDSM"}
VARIANT_CHANNELS = {"coarse": N_CONT + COARSE_NUM_CLASSES, "rgbndsm": N_CONT}  # 12 / 4


# -----------------------------------------------------------------------------
# Model and data loading
# -----------------------------------------------------------------------------

def newest_run(variant):
    """Newest models/uav_deeplab_<variant>_* run directory."""
    runs = sorted(glob.glob(os.path.join(Config.OUTPUT_DIR, f"uav_deeplab_{variant}_*")))
    if not runs:
        raise FileNotFoundError(
            f"No uav_deeplab_{variant}_* run found in {Config.OUTPUT_DIR}. "
            f"Train it first (set Config.USE_COARSE={'True' if variant == 'coarse' else 'False'})."
        )
    return runs[-1]


def load_uav_model(model_dir, num_channels):
    """Rebuilds the UAV DeepLabV3+ for the given channel count and loads weights.

    Mirrors evaluate.py's loader: rebuild the architecture (so the custom loss /
    metrics need not be registered) and try the saved weight formats in order,
    falling back to the full .keras model.
    """
    input_shape = (Config.INPUT_SHAPE[0], Config.INPUT_SHAPE[1], num_channels)
    model = build_deeplabv3plus(input_shape, Config.NUM_CLASSES)

    candidates = [
        os.path.join(model_dir, "final_model.weights.h5"),
        os.path.join(model_dir, "best_model_ckpt"),
        os.path.join(model_dir, "final_model_tf_ckpt"),
    ]
    last_err = None
    for path in candidates:
        if os.path.exists(path) or os.path.exists(path + ".index"):
            try:
                model.load_weights(path)
                print(f"  loaded weights: {path}")
                return model
            except Exception as err:  # try the next candidate format
                last_err = err

    keras_path = os.path.join(model_dir, "final_model.keras")
    if os.path.exists(keras_path):
        import tensorflow as tf
        print(f"  loaded full model: {keras_path}")
        return tf.keras.models.load_model(keras_path, compile=False)

    raise FileNotFoundError(f"No loadable weights in {model_dir} (last error: {last_err})")


def list_split_files(split):
    if split == "test":
        d_img, d_mask = Config.TEST_IMG_PATH, Config.TEST_MASK_PATH
    elif split == "train":
        d_img, d_mask = Config.TRAIN_IMG_PATH, Config.TRAIN_MASK_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    img_files = sorted(
        os.path.join(d_img, f) for f in os.listdir(d_img) if f.endswith(".npy")
    )
    mask_files = sorted(
        os.path.join(d_mask, f) for f in os.listdir(d_mask) if f.endswith(".npy")
    )
    assert len(img_files) == len(mask_files), "Image/mask count mismatch."
    return img_files, mask_files


def load_uav_image(path, mean, std, use_coarse):
    """Assembles one model input from a 5-channel tile, exactly as the dataloader
    does: z-score normalize R, G, B, nDSM; append one-hot coarse context if used."""
    stack = np.load(path).astype(np.float32)                  # (H, W, 5)
    cont = (stack[:, :, :N_CONT] - mean[:N_CONT]) / std[:N_CONT]
    if use_coarse:
        coarse = stack[:, :, N_CONT].astype(np.int32)         # 0..8
        return np.concatenate([cont, _coarse_to_onehot(coarse)], axis=-1).astype(np.float32)
    return cont.astype(np.float32)


def load_mask(path):
    """Loads one mask snippet as 0-based class indices (stored as [1..N])."""
    mask = np.load(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32) - 1


# -----------------------------------------------------------------------------
# Metrics (all derived from one confusion matrix, identical to evaluate.py)
# -----------------------------------------------------------------------------

def accumulate_confusion_matrix(model, img_files, mask_files, mean, std, use_coarse, batch_size):
    n = Config.NUM_CLASSES
    cm = np.zeros((n, n), dtype=np.int64)

    processed = 0
    for start in range(0, len(img_files), batch_size):
        batch = img_files[start:start + batch_size]
        imgs = np.stack([load_uav_image(p, mean, std, use_coarse) for p in batch])
        preds = np.argmax(model.predict(imgs, verbose=0), axis=-1).astype(np.int32)
        for pred in preds:
            gt = load_mask(mask_files[processed])
            valid = (gt >= 0) & (gt < n)
            np.add.at(cm, (gt[valid].ravel(), pred[valid].ravel()), 1)
            processed += 1
        print(f"  evaluated {processed}/{len(img_files)} snippets", end="\r")
    print()
    return cm


def metrics_from_confusion_matrix(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    iou = tp / np.maximum(tp + fp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    total = cm.sum()
    accuracy = tp.sum() / max(total, 1)
    weights = support / max(support.sum(), 1)

    return {
        "iou": iou, "precision": precision, "recall": recall, "f1": f1,
        "support": support,
        "mean_iou": float(iou.mean()),
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * weights).sum()),
    }


def save_comparison_metrics(metrics, out_dir):
    """Writes the side-by-side per-class table (CSV) and a formatted report (TXT)."""
    names = Config.CLASS_NAMES
    mc, mr = metrics["coarse"], metrics["rgbndsm"]

    # Per-class CSV: IoU / F1 for both models and their delta (coarse - rgbndsm).
    with open(os.path.join(out_dir, "comparison_per_class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "iou_coarse", "iou_rgbndsm", "iou_delta",
                    "f1_coarse", "f1_rgbndsm", "f1_delta", "support"])
        for i, name in enumerate(names):
            w.writerow([
                name,
                f"{mc['iou'][i]:.4f}", f"{mr['iou'][i]:.4f}", f"{mc['iou'][i] - mr['iou'][i]:+.4f}",
                f"{mc['f1'][i]:.4f}", f"{mr['f1'][i]:.4f}", f"{mc['f1'][i] - mr['f1'][i]:+.4f}",
                int(mc["support"][i]),
            ])

    # Formatted text report.
    lines = ["UAV cascade ablation: coarse context vs. RGB + nDSM", "=" * 70, ""]
    header = f"{'class':<22}{'IoU coarse':>12}{'IoU rgbnd':>11}{'dIoU':>9}{'F1 coarse':>11}{'F1 rgbnd':>10}{'dF1':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, name in enumerate(names):
        lines.append(
            f"{name:<22}{mc['iou'][i]:>12.4f}{mr['iou'][i]:>11.4f}"
            f"{mc['iou'][i] - mr['iou'][i]:>+9.4f}"
            f"{mc['f1'][i]:>11.4f}{mr['f1'][i]:>10.4f}{mc['f1'][i] - mr['f1'][i]:>+9.4f}"
        )
    lines.append("-" * len(header))

    def summary_row(label, key, fmt="{:.4f}"):
        cv, rv = mc[key], mr[key]
        return f"{label:<22}{fmt.format(cv):>12}{fmt.format(rv):>11}{cv - rv:>+9.4f}"

    lines.append("")
    lines.append(f"{'metric':<22}{'coarse':>12}{'RGB+nDSM':>11}{'delta':>9}")
    lines.append("-" * 54)
    lines.append(summary_row("mean IoU", "mean_iou"))
    lines.append(summary_row("pixel accuracy", "accuracy"))
    lines.append(summary_row("macro F1", "macro_f1"))
    lines.append(summary_row("weighted F1", "weighted_f1"))
    report = "\n".join(lines)

    with open(os.path.join(out_dir, "comparison_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)


# -----------------------------------------------------------------------------
# PIL plotting helpers (matplotlib is unusable in this env; see evaluate.py)
# -----------------------------------------------------------------------------

def _font(size=14):
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10 returns a TTF
    except TypeError:
        return ImageFont.load_default()


def _text(draw, xy, s, font, fill=_FG, anchor="la"):
    draw.text(xy, s, font=font, fill=fill, anchor=anchor)


def _draw_legend(draw, x, y, width, cols=4, row_h=24):
    """Class color legend in a grid starting at (x, y)."""
    cw = width / cols
    font = _font(12)
    for i, name in enumerate(Config.CLASS_NAMES):
        cx = x + (i % cols) * cw
        cy = y + (i // cols) * row_h
        draw.rectangle([cx, cy, cx + 16, cy + 16],
                       fill=tuple(int(v) for v in CLASS_COLORS[i]))
        _text(draw, (cx + 22, cy + 8), f"{i + 1} {name}", font, anchor="lm")


def colorize(label_map):
    return CLASS_COLORS[np.clip(label_map, 0, len(CLASS_COLORS) - 1)]


def rgb_from_uav_image(raw_path):
    """Display uint8 RGB from a UAV tile's R, G, B (channels 0, 1, 2) with a
    per-channel 2-98 percentile contrast stretch. NaN-safe."""
    img = np.nan_to_num(np.load(raw_path).astype(np.float32)[:, :, 0:3])
    out = np.zeros(img.shape, dtype=np.uint8)
    for c in range(3):
        p2, p98 = np.percentile(img[:, :, c], (2, 98))
        if p98 > p2:
            stretched = np.clip((img[:, :, c] - p2) / (p98 - p2), 0, 1)
            out[:, :, c] = (stretched * 255).astype(np.uint8)
    return out


def plot_iou_comparison(metrics, out_dir):
    """Grouped horizontal per-class IoU bars: 'with coarse' (solid class color)
    vs. 'RGB + nDSM' (lighter), mirroring the IoU/F1 style of evaluate.py."""
    names = Config.CLASS_NAMES
    n = len(names)
    mc, mr = metrics["coarse"], metrics["rgbndsm"]

    left, top, right, bottom = 210, 56, 70, 44
    W = 940
    H = top + n * 56 + bottom
    plot_w = W - left - right
    img = Image.new("RGB", (W, H), _BG)
    d = ImageDraw.Draw(img)
    d.rectangle([left, top, left + plot_w, H - bottom], fill=_PANEL)

    # legend: solid bar = with coarse, lighter bar = RGB + nDSM
    sw_solid, sw_light = (120, 150, 190), (170, 190, 215)
    lx = left
    d.rectangle([lx, 14, lx + 16, 28], fill=sw_solid)
    _text(d, (lx + 22, 21), "IoU with coarse (solid)", _font(12), anchor="lm")
    lx += 200
    d.rectangle([lx, 14, lx + 16, 28], fill=sw_light)
    _text(d, (lx + 22, 21), "IoU RGB+nDSM (lighter)", _font(12), anchor="lm")

    for t in np.arange(0, 1.01, 0.2):
        x = left + t * plot_w
        d.line([(x, top), (x, H - bottom)], fill=_GRID)
        _text(d, (x, H - bottom + 4), f"{t:.1f}", _font(11), anchor="ma")
    _text(d, (left + plot_w / 2, H - 14), "IoU", _font(13), anchor="ma")

    row_h = (H - top - bottom) / n
    for i, name in enumerate(names):
        cy = top + i * row_h + row_h / 2
        color = tuple(int(v) for v in CLASS_COLORS[i])
        light = tuple(int(v * 0.45 + 120) for v in CLASS_COLORS[i])
        bh = row_h * 0.3
        delta = mc["iou"][i] - mr["iou"][i]
        _text(d, (left - 10, cy), f"{i + 1} {name}", _font(12), anchor="rm")
        # upper bar: with coarse
        d.rectangle([left, cy - bh - 1, left + mc["iou"][i] * plot_w, cy - 1], fill=color)
        _text(d, (left + mc["iou"][i] * plot_w + 5, cy - bh / 2 - 1),
              f'{mc["iou"][i]:.2f}', _font(10), anchor="lm")
        # lower bar: RGB + nDSM, annotated with the delta
        d.rectangle([left, cy + 1, left + mr["iou"][i] * plot_w, cy + bh + 1], fill=light)
        _text(d, (left + mr["iou"][i] * plot_w + 5, cy + bh / 2 + 1),
              f'{mr["iou"][i]:.2f}  ({delta:+.2f})', _font(10), anchor="lm")

    img.save(os.path.join(out_dir, "iou_comparison.png"))


def plot_confusion_matrix(cm, out_path, title):
    """Row-normalized confusion matrix heatmap, identical style to evaluate.py."""
    names = Config.CLASS_NAMES
    n = len(names)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    cell = 62
    fn = _font(12)
    names_x = 34
    name_w = max(fn.getlength(f"{i + 1} {names[i]}") for i in range(n))
    left = int(names_x + name_w + 14)
    top, right, bottom = 56, 24, 30
    W = left + n * cell + right
    H = top + n * cell + bottom
    img = Image.new("RGB", (W, H), _BG)
    d = ImageDraw.Draw(img)

    _text(d, (W / 2, 8), title, _font(15), anchor="ma")
    _text(d, (left + n * cell / 2, 28), "Predicted", _font(13), anchor="ma")
    for j in range(n):
        _text(d, (left + j * cell + cell / 2, top - 6), str(j + 1), _font(12), anchor="mb")

    ylbl = Image.new("RGBA", (n * cell, 22), (0, 0, 0, 0))
    ImageDraw.Draw(ylbl).text((n * cell / 2, 11), "True", font=_font(13), fill=_FG, anchor="mm")
    ylbl = ylbl.rotate(90, expand=True)
    img.paste(ylbl, (6, top), ylbl)

    for i in range(n):
        _text(d, (names_x, top + i * cell + cell / 2),
              f"{i + 1} {names[i]}", fn, anchor="lm")
        for j in range(n):
            v = float(norm[i, j])
            color = (int(255 - v * 224), int(255 - v * 136), int(255 - v * 75))  # white->blue
            x = left + j * cell
            y = top + i * cell
            d.rectangle([x, y, x + cell, y + cell], fill=color, outline=_BG)
            _text(d, (x + cell / 2, y + cell / 2), f"{v:.2f}", _font(11),
                  fill=(255, 255, 255) if v > 0.5 else (15, 15, 25), anchor="mm")

    img.save(out_path)


def _save_panels(panels, titles, out_path, suptitle=None):
    """Saves N panels side by side with titles and a class legend (variable N)."""
    h, w = panels[0].shape[:2]
    sep, header = 6, 26
    top_extra = 26 if suptitle else 0
    footer = 60
    n = len(panels)
    W = n * w + (n - 1) * sep
    H = top_extra + header + h + footer
    img = Image.new("RGB", (W, H), _BG)
    d = ImageDraw.Draw(img)
    if suptitle:
        _text(d, (W / 2, 5), suptitle, _font(16), anchor="ma")
    x = 0
    for panel, title in zip(panels, titles):
        img.paste(Image.fromarray(panel), (x, top_extra + header))
        _text(d, (x + w / 2, top_extra + header - 5), title, _font(14), anchor="mb")
        x += w + sep
    _draw_legend(d, 10, top_extra + header + h + 8, W - 20)
    img.save(out_path)


def plot_qualitative_samples(models, img_files, mask_files, mean, std, out_dir, n_samples):
    """RGB | ground truth | pred(with coarse) | pred(RGB+nDSM) panels for evenly
    spaced samples, so the two models can be compared visually on the same tiles."""
    indices = np.linspace(0, len(img_files) - 1, n_samples, dtype=int)
    samples_dir = os.path.join(out_dir, "qualitative")
    os.makedirs(samples_dir, exist_ok=True)

    titles = ["RGB input", "Ground truth", "Pred (with coarse)", "Pred (RGB+nDSM)"]
    for n, idx in enumerate(indices, start=1):
        rgb = rgb_from_uav_image(img_files[idx])
        gt = colorize(load_mask(mask_files[idx]))

        x_coarse = load_uav_image(img_files[idx], mean, std, use_coarse=True)[np.newaxis, ...]
        x_rgb = load_uav_image(img_files[idx], mean, std, use_coarse=False)[np.newaxis, ...]
        pred_c = colorize(np.argmax(models["coarse"].predict(x_coarse, verbose=0)[0], axis=-1))
        pred_r = colorize(np.argmax(models["rgbndsm"].predict(x_rgb, verbose=0)[0], axis=-1))

        _save_panels([rgb, gt, pred_c, pred_r], titles,
                     os.path.join(samples_dir, f"sample_{n:02d}.png"))

    print(f"Saved {len(indices)} qualitative sample(s) to {samples_dir}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare the coarse-context and RGB+nDSM UAV DeepLabV3+ runs."
    )
    p.add_argument("--coarse-dir", default=None,
                   help="Run dir of the with-coarse model (default: newest uav_deeplab_coarse_*).")
    p.add_argument("--rgbndsm-dir", default=None,
                   help="Run dir of the RGB+nDSM model (default: newest uav_deeplab_rgbndsm_*).")
    p.add_argument("--split", default="test", choices=["test", "train"],
                   help="Which split to evaluate on (default: test).")
    p.add_argument("--out", default=None,
                   help="Output directory (default: models/uav_ablation_comparison).")
    p.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--num-qualitative", type=int, default=6,
                   help="Number of RGB/GT/pred/pred sample figures (default: 6).")
    return p.parse_args()


def main():
    args = parse_args()
    run_dirs = {
        "coarse": args.coarse_dir or newest_run("coarse"),
        "rgbndsm": args.rgbndsm_dir or newest_run("rgbndsm"),
    }
    out_dir = args.out or os.path.join(Config.OUTPUT_DIR, "uav_ablation_comparison")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Split:   {args.split}")
    for v in VARIANTS:
        print(f"{VARIANT_LABELS[v]:<14} model: {run_dirs[v]}")
    print(f"Output:  {out_dir}\n")

    mean, std = load_normalization_stats(Config.NORM_STATS_PATH)
    img_files, mask_files = list_split_files(args.split)
    print(f"Evaluating {len(img_files)} snippets per model...\n")

    models, metrics = {}, {}
    for v in VARIANTS:
        print(f"[{VARIANT_LABELS[v]}]")
        models[v] = load_uav_model(run_dirs[v], VARIANT_CHANNELS[v])
        cm = accumulate_confusion_matrix(
            models[v], img_files, mask_files, mean, std,
            use_coarse=(v == "coarse"), batch_size=args.batch_size,
        )
        metrics[v] = metrics_from_confusion_matrix(cm)
        np.savetxt(os.path.join(out_dir, f"confusion_matrix_{v}.csv"), cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(
            cm, os.path.join(out_dir, f"confusion_matrix_{v}.png"),
            f"Confusion matrix - {VARIANT_LABELS[v]}",
        )

    save_comparison_metrics(metrics, out_dir)
    plot_iou_comparison(metrics, out_dir)
    plot_qualitative_samples(models, img_files, mask_files, mean, std, out_dir, args.num_qualitative)

    print(f"\nDone. Comparison metrics and figures written to {out_dir}")


if __name__ == "__main__":
    main()
