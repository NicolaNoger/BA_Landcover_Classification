"""
Standalone evaluation and visualization for a trained DeepLabV3+ run.

Loads a saved model from a training output directory and, without retraining,
produces the metrics and figures needed for the thesis / documentation:

    Metrics (printed and saved as CSV/TXT)
        - per-class IoU / precision / recall / F1
        - overall accuracy, mean IoU, macro and weighted F1
        - confusion matrix

    Figures (PNG, rendered directly with PIL)
        - confusion matrix heatmap (row-normalized)
        - per-class IoU and F1 bar chart
        - training history curves (loss / accuracy / IoU)        [if CSV present]
        - per-class IoU over epochs                              [if CSV present]
        - qualitative RGB / ground-truth / prediction triplets
        - optional stitched mosaic of contiguous patches         [--stitch-cols]

All metrics are derived from a single accumulated confusion matrix, so memory
stays flat regardless of test-set size.

Works for both the DeepLabV3+ and the U-Net runs (same data, classes and input
shape); select with --arch. The newest matching run is auto-selected.

Usage:
    python evaluate.py                       # newest deeplab run
    python evaluate.py --arch unet           # newest U-Net run
    python evaluate.py --model-dir ../../models/deeplab_20260617_120000
    python evaluate.py --arch unet --model-dir ../../models/unet_20260510_183058
    python evaluate.py --split test --num-qualitative 8
    python evaluate.py --stitch-cols 49      # mosaic, only if you know the grid width

Note on stitching: snippets carry no stored grid or geo-reference. The mosaic
assumes the snippets of one tile were exported in row-major order and were not
shuffled. Pass the original number of columns via --stitch-cols; without it the
mosaic is skipped rather than guessed.
"""

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deeplab_v3plus import build_deeplabv3plus
from dataloader import load_normalization_stats
from train_deeplab import Config


# Project-wide tab10 palette (0-based), shared by every figure and by the mask
# colorization, so map colors and chart colors stay identical. These are the
# exact RGB values of matplotlib/seaborn "tab10" for the first 8 classes.
CLASS_COLORS = np.array([
    [ 31, 119, 180],   # Building
    [255, 127,  14],   # Impervious_Surface
    [ 44, 160,  44],   # Cropland
    [214,  39,  40],   # Intensive_Culture
    [148, 103, 189],   # Grassland_Garden
    [140,  86,  75],   # Tree_Canopy
    [227, 119, 194],   # Water
    [127, 127, 127],   # Railway
], dtype=np.uint8)

# Dark theme colors (RGB) for the PIL-rendered figures. All figures are drawn
# directly with PIL because matplotlib is unusable in this conda env: a broken
# numpy install (mixed 1.26 / 2.0) makes every matplotlib render — Agg and SVG
# alike — raise "object __array__ method not producing an array".
_BG = (26, 26, 46)      # figure background
_PANEL = (22, 33, 62)   # plot-area background
_FG = (235, 235, 245)   # text / ticks
_GRID = (70, 70, 100)   # gridlines


# -----------------------------------------------------------------------------
# Model and data loading
# -----------------------------------------------------------------------------

def _build_architecture(arch):
    """Rebuilds an untrained model for the requested architecture. DeepLab and
    U-Net share the same input shape, classes and data, so only the builder and
    checkpoint layout differ."""
    if arch == "unet":
        # U_net.py lives in the sibling Unet/ directory; it only depends on
        # numpy/tensorflow, so adding it to sys.path causes no import clashes.
        unet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Unet")
        if unet_dir not in sys.path:
            sys.path.append(unet_dir)
        from U_net import build_unet
        return build_unet(Config.INPUT_SHAPE, Config.NUM_CLASSES)
    return build_deeplabv3plus(Config.INPUT_SHAPE, Config.NUM_CLASSES)


def load_trained_model(model_dir, arch):
    """Rebuilds the architecture and loads weights from a run directory.

    Rebuilding (rather than tf.keras.models.load_model) avoids having to
    register the custom loss/metrics, since evaluation does not need them.
    Falls back to the full saved model if no weights file is usable.
    """
    if arch == "unet":
        candidates = [
            os.path.join(model_dir, "best_model_weights", "checkpoint"),
            os.path.join(model_dir, "final_model_weights.h5"),
            os.path.join(model_dir, "final_model_weights"),
        ]
        keras_path = os.path.join(model_dir, "final_model")
    else:
        candidates = [
            os.path.join(model_dir, "final_model.weights.h5"),
            os.path.join(model_dir, "best_model_ckpt"),
            os.path.join(model_dir, "final_model_tf_ckpt"),
        ]
        keras_path = os.path.join(model_dir, "final_model.keras")

    model = _build_architecture(arch)

    last_err = None
    for path in candidates:
        if os.path.exists(path) or os.path.exists(path + ".index"):
            try:
                model.load_weights(path)
                print(f"Loaded weights from: {path}")
                return model
            except Exception as err:  # try the next candidate format
                last_err = err

    if os.path.exists(keras_path):
        import tensorflow as tf

        print(f"Loaded full model from: {keras_path}")
        return tf.keras.models.load_model(keras_path, compile=False)

    raise FileNotFoundError(
        f"No loadable weights found in {model_dir} (last error: {last_err})"
    )


def list_split_files(split, img_dir=None, mask_dir=None):
    """Returns sorted (image_files, mask_files) for the requested split.

    img_dir / mask_dir override the split defaults, e.g. to evaluate the same
    model against a different mask version (pre- vs post-label-review).
    """
    if split == "test":
        d_img, d_mask = Config.TEST_IMG_PATH, Config.TEST_MASK_PATH
    elif split == "train":
        d_img, d_mask = Config.TRAIN_IMG_PATH, Config.TRAIN_MASK_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    d_img = img_dir or d_img
    d_mask = mask_dir or d_mask

    img_files = sorted(
        os.path.join(d_img, f) for f in os.listdir(d_img) if f.endswith(".npy")
    )
    mask_files = sorted(
        os.path.join(d_mask, f) for f in os.listdir(d_mask) if f.endswith(".npy")
    )
    assert len(img_files) == len(mask_files), "Image/mask count mismatch."
    return img_files, mask_files


def load_image(path, mean, std):
    """Loads and normalizes one image snippet exactly as the dataloader does."""
    return (np.load(path).astype(np.float32) - mean) / std


def load_mask(path):
    """Loads one mask snippet as 0-based class indices (stored as [1..N])."""
    mask = np.load(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32) - 1


def predict_labels(model, img_files, mean, std, batch_size):
    """Yields predicted label maps (argmax) batch by batch."""
    for start in range(0, len(img_files), batch_size):
        batch = img_files[start:start + batch_size]
        imgs = np.stack([load_image(p, mean, std) for p in batch])
        preds = model.predict(imgs, verbose=0)
        yield np.argmax(preds, axis=-1).astype(np.int32)


# -----------------------------------------------------------------------------
# Metrics (all derived from one confusion matrix)
# -----------------------------------------------------------------------------

def accumulate_confusion_matrix(model, img_files, mask_files, mean, std, batch_size):
    n = Config.NUM_CLASSES
    cm = np.zeros((n, n), dtype=np.int64)

    processed = 0
    for preds in predict_labels(model, img_files, mean, std, batch_size):
        for pred in preds:
            gt = load_mask(mask_files[processed])
            valid = (gt >= 0) & (gt < n)
            np.add.at(cm, (gt[valid].ravel(), pred[valid].ravel()), 1)
            processed += 1
        print(f"  evaluated {processed}/{len(img_files)} snippets", end="\r")
    print()
    return cm


def metrics_from_confusion_matrix(cm):
    """Returns a dict of per-class and aggregate metrics from a confusion matrix."""
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


def save_metrics(cm, m, out_dir):
    names = Config.CLASS_NAMES

    with open(os.path.join(out_dir, "per_class_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "iou", "precision", "recall", "f1", "support"])
        for i, name in enumerate(names):
            writer.writerow([
                name, f"{m['iou'][i]:.4f}", f"{m['precision'][i]:.4f}",
                f"{m['recall'][i]:.4f}", f"{m['f1'][i]:.4f}", int(m["support"][i]),
            ])

    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    lines = ["DeepLabV3+ evaluation", "=" * 60, ""]
    header = f"{'class':<22}{'IoU':>9}{'Precision':>11}{'Recall':>9}{'F1':>9}{'support':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, name in enumerate(names):
        lines.append(
            f"{name:<22}{m['iou'][i]:>9.4f}{m['precision'][i]:>11.4f}"
            f"{m['recall'][i]:>9.4f}{m['f1'][i]:>9.4f}{int(m['support'][i]):>12,}"
        )
    lines.append("-" * len(header))
    lines.append(f"{'mean IoU':<22}{m['mean_iou']:>9.4f}")
    lines.append(f"{'pixel accuracy':<22}{m['accuracy']:>9.4f}")
    lines.append(f"{'macro F1':<22}{m['macro_f1']:>9.4f}")
    lines.append(f"{'weighted F1':<22}{m['weighted_f1']:>9.4f}")
    report = "\n".join(lines)

    with open(os.path.join(out_dir, "evaluation_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def colorize(label_map):
    """Maps a 0-based label map to an RGB image using the class palette."""
    clipped = np.clip(label_map, 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[clipped]


def rgb_from_image(raw_path):
    """Builds a display uint8 RGB image from raw channels 1,2,3 (R,G,B) with a
    per-channel 2-98 percentile contrast stretch. NaN-safe, and uint8 so it is
    passed to imshow without triggering float-array rendering issues."""
    img = np.nan_to_num(np.load(raw_path).astype(np.float32)[:, :, 1:4])
    out = np.zeros(img.shape, dtype=np.uint8)
    for c in range(3):
        p2, p98 = np.percentile(img[:, :, c], (2, 98))
        if p98 > p2:
            stretched = np.clip((img[:, :, c] - p2) / (p98 - p2), 0, 1)
            out[:, :, c] = (stretched * 255).astype(np.uint8)
    return out


# -----------------------------------------------------------------------------
# PIL plotting helpers (matplotlib is unusable in this env, see note above)
# -----------------------------------------------------------------------------

def _font(size=14):
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10 returns a TTF
    except TypeError:
        return ImageFont.load_default()


def _text(draw, xy, s, font, fill=_FG, anchor="la"):
    draw.text(xy, s, font=font, fill=fill, anchor=anchor)


def _dashed(draw, a, b, color, dash=7, width=2):
    (ax, ay), (bx, by) = a, b
    dist = math.hypot(bx - ax, by - ay)
    if dist == 0:
        return
    steps = max(int(dist / dash), 1)
    for s in range(0, steps, 2):
        t0, t1 = s / steps, min((s + 1) / steps, 1.0)
        draw.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0),
                   (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=color, width=width)


def _draw_legend(draw, x, y, width, cols=4, row_h=24):
    """Draws a class color legend in a grid starting at (x, y)."""
    cw = width / cols
    font = _font(12)
    for i, name in enumerate(Config.CLASS_NAMES):
        cx = x + (i % cols) * cw
        cy = y + (i // cols) * row_h
        draw.rectangle([cx, cy, cx + 16, cy + 16],
                       fill=tuple(int(v) for v in CLASS_COLORS[i]))
        _text(draw, (cx + 22, cy + 8), f"{i + 1} {name}", font, anchor="lm")


def _line_chart(series, x_vals, title, xlabel, ylabel, out_path,
                y_range=(0.0, 1.0), size=(1000, 600)):
    """Renders a multi-line chart with PIL.

    series: list of (label, y_values, rgb_tuple, dashed_bool).
    """
    W, H = size
    top = 56 if title else 28
    left, right, bottom = 90, 30, 56
    x0, y0, x1, y1 = left, top, W - right, H - bottom
    img = Image.new("RGB", (W, H), _BG)
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1, y1], fill=_PANEL)

    if title:
        _text(d, (W / 2, 14), title, _font(18), anchor="ma")
    _text(d, ((x0 + x1) / 2, H - 16), xlabel, _font(13), anchor="ma")
    span = int(y1 - y0)
    lbl = Image.new("RGBA", (span, 22), (0, 0, 0, 0))
    ImageDraw.Draw(lbl).text((span / 2, 11), ylabel, font=_font(13), fill=_FG, anchor="mm")
    lbl = lbl.rotate(90, expand=True)
    img.paste(lbl, (8, y0), lbl)

    xmin, xmax = min(x_vals), max(x_vals)
    ymin, ymax = y_range

    def px(x, y):
        return (x0 + (x - xmin) / (xmax - xmin + 1e-12) * (x1 - x0),
                y1 - (y - ymin) / (ymax - ymin + 1e-12) * (y1 - y0))

    for t in np.linspace(ymin, ymax, 6):
        yy = px(xmin, t)[1]
        d.line([(x0, yy), (x1, yy)], fill=_GRID)
        _text(d, (x0 - 6, yy), f"{t:.2f}", _font(11), anchor="rm")

    for t in np.linspace(xmin, xmax, min(len(x_vals), 8)):
        xx = px(t, ymin)[0]
        _text(d, (xx, y1 + 5), f"{int(round(t))}", _font(11), anchor="ma")

    for label, vals, color, dashed in series:
        pts = [px(x, v) for x, v in zip(x_vals, vals) if np.isfinite(v)]
        if len(pts) < 2:
            continue
        if dashed:
            for a, b in zip(pts[:-1], pts[1:]):
                _dashed(d, a, b, color)
        else:
            d.line(pts, fill=color, width=2)

    ly = top + 8
    for label, vals, color, dashed in series:
        d.rectangle([x1 - 150, ly, x1 - 136, ly + 12], fill=color)
        _text(d, (x1 - 132, ly + 6), label, _font(11), anchor="lm")
        ly += 16

    img.save(out_path)


def _save_triptych(panels, titles, out_path, suptitle=None):
    """Saves [panel | panel | panel] side by side with titles and a legend."""
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


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def plot_confusion_matrix(cm, out_dir):
    names = Config.CLASS_NAMES
    n = len(names)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    cell = 62
    fn = _font(12)
    names_x = 34                                   # class names start just right of "True"
    name_w = max(fn.getlength(f"{i + 1} {names[i]}") for i in range(n))
    left = int(names_x + name_w + 14)
    top, right, bottom = 40, 24, 30
    W = left + n * cell + right
    H = top + n * cell + bottom
    img = Image.new("RGB", (W, H), _BG)
    d = ImageDraw.Draw(img)

    # x-axis label "Predicted" above the column indices
    _text(d, (left + n * cell / 2, 6), "Predicted", _font(15), anchor="ma")
    for j in range(n):
        _text(d, (left + j * cell + cell / 2, top - 6), str(j + 1), _font(12), anchor="mb")

    # y-axis label "True", rotated, just left of the class names
    ylbl = Image.new("RGBA", (n * cell, 22), (0, 0, 0, 0))
    ImageDraw.Draw(ylbl).text((n * cell / 2, 11), "True", font=_font(15), fill=_FG, anchor="mm")
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

    img.save(os.path.join(out_dir, "confusion_matrix.png"))


def plot_per_class_bars(m, out_dir):
    names = Config.CLASS_NAMES
    n = len(names)
    left, top, right, bottom = 210, 56, 60, 44
    W = 940
    H = top + n * 56 + bottom
    plot_w = W - left - right
    img = Image.new("RGB", (W, H), _BG)
    d = ImageDraw.Draw(img)
    d.rectangle([left, top, left + plot_w, H - bottom], fill=_PANEL)

    # legend (top margin): solid bar = IoU, lighter bar = F1
    sw_solid, sw_light = (120, 150, 190), (170, 190, 215)
    lx = left
    d.rectangle([lx, 14, lx + 16, 28], fill=sw_solid)
    _text(d, (lx + 22, 21), "IoU (solid)", _font(12), anchor="lm")
    lx += 130
    d.rectangle([lx, 14, lx + 16, 28], fill=sw_light)
    _text(d, (lx + 22, 21), "F1 (lighter)", _font(12), anchor="lm")

    for t in np.arange(0, 1.01, 0.2):
        x = left + t * plot_w
        d.line([(x, top), (x, H - bottom)], fill=_GRID)
        _text(d, (x, H - bottom + 4), f"{t:.1f}", _font(11), anchor="ma")
    _text(d, (left + plot_w / 2, H - 14), "Score", _font(13), anchor="ma")

    row_h = (H - top - bottom) / n
    for i, name in enumerate(names):
        cy = top + i * row_h + row_h / 2
        color = tuple(int(v) for v in CLASS_COLORS[i])
        light = tuple(int(v * 0.45 + 120) for v in CLASS_COLORS[i])
        bh = row_h * 0.3
        _text(d, (left - 10, cy), f"{i + 1} {name}", _font(12), anchor="rm")
        d.rectangle([left, cy - bh - 1, left + m["iou"][i] * plot_w, cy - 1], fill=color)
        _text(d, (left + m["iou"][i] * plot_w + 5, cy - bh / 2 - 1),
              f'{m["iou"][i]:.2f}', _font(10), anchor="lm")
        d.rectangle([left, cy + 1, left + m["f1"][i] * plot_w, cy + bh + 1], fill=light)
        _text(d, (left + m["f1"][i] * plot_w + 5, cy + bh / 2 + 1),
              f'{m["f1"][i]:.2f}', _font(10), anchor="lm")

    img.save(os.path.join(out_dir, "per_class_iou_f1.png"))


def _read_history_csv(path):
    cols = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    cols.setdefault(k, []).append(np.nan)
    return cols


def plot_training_history(model_dir, out_dir):
    csv_path = os.path.join(model_dir, "training_history.csv")
    if not os.path.exists(csv_path):
        return
    cols = _read_history_csv(csv_path)
    if not cols:
        return
    epochs = cols.get("epoch") or list(range(1, len(next(iter(cols.values()))) + 1))

    def panel(key, val_key, title, fname, y_range):
        if key not in cols:
            return
        series = [("train", cols[key], (31, 119, 180), False)]
        if val_key in cols:
            series.append(("val", cols[val_key], (255, 127, 14), True))
        _line_chart(series, epochs, "", "Epoch", title,
                    os.path.join(out_dir, fname), y_range=y_range)

    if "loss" in cols:
        finite = [v for v in cols["loss"] if np.isfinite(v)]
        finite += [v for v in cols.get("val_loss", []) if np.isfinite(v)]
        ymax = (max(finite) if finite else 1.0) * 1.1 + 1e-6
        panel("loss", "val_loss", "Loss", "history_loss.png", (0, ymax))
    panel("accuracy", "val_accuracy", "Accuracy", "history_accuracy.png", (0, 1))
    panel("iou", "val_iou", "Mean IoU", "history_iou.png", (0, 1))


def plot_per_class_iou_history(model_dir, out_dir):
    csv_path = os.path.join(model_dir, "per_class_iou_history.csv")
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    header = list(rows[0].keys())
    series = []
    for i, name in enumerate(Config.CLASS_NAMES):
        # Match either a plain "Building" column (deeplab) or an
        # "iou_1_Building"-style column (unet); take the first that contains it.
        col = next((c for c in header if c == name or c.endswith(name) or name in c), None)
        if col:
            series.append((name, [float(r[col]) for r in rows],
                           tuple(int(v) for v in CLASS_COLORS[i]), False))
    if "mean_iou" in rows[0]:
        series.append(("mean", [float(r["mean_iou"]) for r in rows], (255, 255, 255), True))

    _line_chart(series, epochs, "", "Epoch", "IoU",
                os.path.join(out_dir, "per_class_iou_history.png"),
                y_range=(0, 1), size=(1100, 650))


def plot_qualitative_samples(model, img_files, mask_files, mean, std, out_dir, n_samples):
    """Saves RGB | ground truth | prediction triptychs for evenly spaced samples."""
    indices = np.linspace(0, len(img_files) - 1, n_samples, dtype=int)
    samples_dir = os.path.join(out_dir, "qualitative")
    os.makedirs(samples_dir, exist_ok=True)

    titles = ["RGB input", "Ground truth", "Prediction"]
    for n, idx in enumerate(indices, start=1):
        rgb = rgb_from_image(img_files[idx])
        gt = colorize(load_mask(mask_files[idx]))
        pred = model.predict(load_image(img_files[idx], mean, std)[np.newaxis, ...], verbose=0)[0]
        pred = colorize(np.argmax(pred, axis=-1))
        _save_triptych([rgb, gt, pred], titles,
                       os.path.join(samples_dir, f"sample_{n:02d}.png"))

    print(f"Saved {len(indices)} qualitative sample(s) to {samples_dir}")


def stitch_mosaic(model, img_files, mask_files, mean, std, out_dir, cols, tile, max_rows):
    """Assembles contiguous snippets of one tile into RGB / GT / prediction
    mosaics, saved as three PNGs.

    Assumes row-major export order within a tile. Only call this when the grid
    width (cols) is known; otherwise the spatial layout cannot be trusted.
    """
    def tile_id(path):
        # filename pattern: <prefix>_<split>_<tile>_<index>.npy
        return os.path.basename(path).split("_")[-2]

    tiles = sorted({tile_id(p) for p in img_files})
    chosen = tile or tiles[0]
    if chosen not in tiles:
        print(f"Stitch skipped: tile '{chosen}' not found (available: {tiles}).")
        return

    pairs = sorted(
        (i, mk) for i, mk in zip(img_files, mask_files) if tile_id(i) == chosen
    )
    n_full_rows = len(pairs) // cols
    if max_rows:
        n_full_rows = min(n_full_rows, max_rows)
    if n_full_rows == 0:
        print(f"Stitch skipped: fewer than {cols} snippets for tile '{chosen}'.")
        return

    pairs = pairs[: n_full_rows * cols]
    patch = 512
    rgb_canvas = np.zeros((n_full_rows * patch, cols * patch, 3), dtype=np.uint8)
    gt_canvas = np.zeros_like(rgb_canvas)
    pred_canvas = np.zeros_like(rgb_canvas)

    for k, (img_path, mask_path) in enumerate(pairs):
        r, c = divmod(k, cols)
        ys, xs = r * patch, c * patch
        rgb_canvas[ys:ys + patch, xs:xs + patch] = rgb_from_image(img_path)
        gt_canvas[ys:ys + patch, xs:xs + patch] = colorize(load_mask(mask_path))
        pred = model.predict(load_image(img_path, mean, std)[np.newaxis, ...], verbose=0)[0]
        pred_canvas[ys:ys + patch, xs:xs + patch] = colorize(np.argmax(pred, axis=-1))

    for canvas, name in [(rgb_canvas, "rgb"), (gt_canvas, "gt"), (pred_canvas, "pred")]:
        Image.fromarray(canvas).save(
            os.path.join(out_dir, f"mosaic_tile_{chosen}_{name}.png")
        )
    print(f"Saved stitched mosaic ({n_full_rows} x {cols} patches) for tile {chosen}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def newest_model_dir(arch):
    runs = sorted(glob.glob(os.path.join(Config.OUTPUT_DIR, f"{arch}_*")))
    if not runs:
        raise FileNotFoundError(f"No {arch}_* run found in {Config.OUTPUT_DIR}")
    return runs[-1]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and visualize a DeepLabV3+ or U-Net run.")
    p.add_argument("--arch", default="deeplab", choices=["deeplab", "unet"],
                   help="Model architecture to load (default: deeplab).")
    p.add_argument("--model-dir", default=None,
                   help="Run directory with saved weights (default: newest <arch>_* run in models/).")
    p.add_argument("--split", default="test", choices=["test", "train"],
                   help="Which split to evaluate on (default: test).")
    p.add_argument("--out", default=None,
                   help="Output directory (default: <model-dir>/evaluation_<split>).")
    p.add_argument("--mask-dir", default=None,
                   help="Override the mask directory (e.g. pre-review masks for a before/after comparison).")
    p.add_argument("--img-dir", default=None,
                   help="Override the image directory.")
    p.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--num-qualitative", type=int, default=6,
                   help="Number of RGB/GT/prediction sample figures (default: 6).")
    p.add_argument("--stitch-cols", type=int, default=None,
                   help="Grid width to assemble a spatial mosaic. Omit to skip stitching.")
    p.add_argument("--stitch-tile", default=None,
                   help="Tile id to stitch (default: first tile in the split).")
    p.add_argument("--stitch-max-rows", type=int, default=None,
                   help="Cap the number of mosaic rows (default: all full rows).")
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir or newest_model_dir(args.arch)
    out_dir = args.out or os.path.join(model_dir, f"evaluation_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Arch:   {args.arch}")
    print(f"Model:  {model_dir}")
    print(f"Split:  {args.split}")
    print(f"Output: {out_dir}\n")

    model = load_trained_model(model_dir, args.arch)
    mean, std = load_normalization_stats(Config.NORM_STATS_PATH)
    img_files, mask_files = list_split_files(args.split, args.img_dir, args.mask_dir)
    print(f"Evaluating {len(img_files)} snippets...\n")

    cm = accumulate_confusion_matrix(model, img_files, mask_files, mean, std, args.batch_size)
    m = metrics_from_confusion_matrix(cm)
    save_metrics(cm, m, out_dir)

    # Figures
    plot_confusion_matrix(cm, out_dir)
    plot_per_class_bars(m, out_dir)
    plot_training_history(model_dir, out_dir)
    plot_per_class_iou_history(model_dir, out_dir)
    plot_qualitative_samples(model, img_files, mask_files, mean, std, out_dir, args.num_qualitative)

    if args.stitch_cols:
        stitch_mosaic(model, img_files, mask_files, mean, std, out_dir,
                      args.stitch_cols, args.stitch_tile, args.stitch_max_rows)

    print(f"\nDone. All metrics and figures written to {out_dir}")


if __name__ == "__main__":
    main()
