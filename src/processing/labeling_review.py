"""
labeling_review.py  –  Interactive label correction tool

Start with:
    streamlit run labeling_review.py

Expects:
    review_queue.json    (produced by generate_queue.py)
    overlays/            (produced by generate_queue.py)

Writes corrected masks to:
    test/mask_snippets_clean/   (originals are left untouched)

Decisions per blob:
    Accept prediction  -> patch blob pixels to predicted class id
    Keep label         -> no change, mark blob as reviewed
    Both wrong         -> mark blob as unclear (no patch)
    Skip               -> review later
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime
# pathlib.Path not used in this script

# =============================================================================
# CONFIGURATION — adjust paths to your environment
# =============================================================================

PROJECT_ROOT    = "A:/STUDIUM/06_Fruelingssemester26/BA/data"
DATA_PATH       = os.path.join(PROJECT_ROOT, "processed", "training_data")

QUEUE_FILE      = os.path.join(DATA_PATH, "label_review", "review_queue.json")
REVIEWS_FILE    = os.path.join(DATA_PATH, "label_review", "reviews_log.json")


def find_fallback_file(filename: str) -> str | None:
    """Search workspace for a filename fallback and return first match."""
    # First check absolute path as-is
    if os.path.exists(filename):
        return filename
    # Walk project root for fallback
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if os.path.basename(filename) in files:
            return os.path.join(root, os.path.basename(filename))
    return None


# Resolve possible alternate locations (if generate_queue.py was run from another cwd)
resolved_queue = find_fallback_file(QUEUE_FILE)
if resolved_queue:
    QUEUE_FILE = resolved_queue

resolved_reviews = find_fallback_file(REVIEWS_FILE)
if resolved_reviews:
    REVIEWS_FILE = resolved_reviews

# Original-Masken (nur lesen!)
ORIGINAL_MASK_DIR = os.path.join(DATA_PATH, "test", "mask_snippets")

# Korrigierte Masken (schreiben – Originale NICHT überschreiben)
CLEAN_MASK_DIR    = os.path.join(DATA_PATH, "test", "mask_snippets_clean")

# =============================================================================
# KLASSEN-METADATEN
# =============================================================================

CLASS_NAMES = {
    1: "Building",
    2: "Impervious_Surface",
    3: "Cropland",
    4: "Intensive_Culture",
    5: "Grassland_Garden",
    6: "Tree_Canopy",
    7: "Water",
    8: "Railway",
}

CLASS_COLORS_HEX = {
    1: "#DC3C3C",
    2: "#B4B4B4",
    3: "#E6D232",
    4: "#FF9B1E",
    5: "#64C85A",
    6: "#1E781E",
    7: "#3C8CDC",
    8: "#823CB4",
}

DECISION_LABELS = {
    "accept_prediction": "Accept prediction",
    "keep_label":        "Keep label",
    "both_wrong":        "Both wrong",
    "skipped":           "Skipped",
}

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

@st.cache_data
def load_queue(queue_file: str) -> list:
    """Load review_queue.json. Cached to avoid re-loading on every interaction."""
    if not os.path.exists(queue_file):
        return []
    with open(queue_file, "r", encoding="utf-8") as f:
        queue = json.load(f)

    # If queue entries reference remote/HPC absolute paths, try to remap to
    # locally copied overlays/npy files under DATA_PATH/label_review.
    local_dir = os.path.join(DATA_PATH, "label_review")
    for item in queue:
        for key in ("overlay_rgb_path", "overlay_gt_path", "overlay_pred_path",
                    "blob_mask_npy", "img_npy_path", "mask_npy_path"):
            p = item.get(key)
            if not p:
                continue
            # if path exists locally already, keep it
            if os.path.exists(p):
                continue
            # try remapping to local_dir by basename
            candidate = os.path.join(local_dir, os.path.basename(p))
            if os.path.exists(candidate):
                item[key] = candidate

    return queue


def load_reviews(reviews_file: str) -> dict:
    """Load existing reviews file if present."""
    if os.path.exists(reviews_file):
        with open(reviews_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_reviews(reviews: dict, reviews_file: str) -> None:
    os.makedirs(os.path.dirname(reviews_file), exist_ok=True)
    with open(reviews_file, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)


def load_image_safe(path: str) -> Image.Image | None:
    """Load PNG image; return None if file is missing."""
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None


def placeholder_image(text: str, size: int = 300) -> Image.Image:
    """Create a gray placeholder image."""
    img = Image.new("RGB", (size, size), color=(60, 60, 60))
    return img


def load_mask_for_snippet(snippet_name: str, mask_npy_path: str) -> np.ndarray | None:
    """Load mask for a snippet.
    Priority: clean directory -> original.
    Returns (H, W) uint8 array (values 1-8) or None if missing.
    """
    clean_path = os.path.join(CLEAN_MASK_DIR, os.path.basename(mask_npy_path))
    if os.path.exists(clean_path):
        arr = np.load(clean_path, allow_pickle=False)
    elif os.path.exists(mask_npy_path):
        arr = np.load(mask_npy_path, allow_pickle=False)
    else:
        return None

    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def patch_and_save_mask(
    mask_npy_path: str,
    blob_mask_npy: str,
    pred_class_id: int,
) -> bool:
    """Patch mask pixels inside blob to predicted class id and save
    resulting mask into CLEAN_MASK_DIR. Returns True on success."""
    try:
        os.makedirs(CLEAN_MASK_DIR, exist_ok=True)

        # Maske laden (clean > original)
        mask = load_mask_for_snippet(
            snippet_name  = "",       # nicht benötigt hier
            mask_npy_path = mask_npy_path,
        )
        if mask is None:
            st.error(f"Mask file not found: {mask_npy_path}")
            return False

        # Blob-Maske laden
        blob = np.load(blob_mask_npy, allow_pickle=False).astype(bool)

        # Patch anwenden
        mask[blob] = np.uint8(pred_class_id)

        # In clean-Verzeichnis speichern (mit korrekter Form HxWx1)
        clean_path = os.path.join(CLEAN_MASK_DIR, os.path.basename(mask_npy_path))
        np.save(clean_path, mask[:, :, np.newaxis])
        return True

    except Exception as e:
        st.error(f"Error patching mask: {e}")
        return False


def record_decision(
    region_id: str,
    decision: str,
    item: dict,
    reviews: dict,
    reviews_file: str,
) -> None:
    """Record a decision in the reviews log file."""
    reviews[region_id] = {
        "decision":         decision,
        "snippet":          item.get("snippet"),
        "label_class_id":   item.get("label_class_id"),
        "label_class_name": item.get("label_class_name"),
        "pred_class_id":    item.get("pred_class_id"),
        "pred_class_name":  item.get("pred_class_name"),
        "confidence_mean":  item.get("confidence_mean"),
        "blob_size_px":     item.get("blob_size_px"),
        "timestamp":        datetime.now().isoformat(),
    }
    save_reviews(reviews, reviews_file)


# =============================================================================
# STATISTIK-SIDEBAR
# =============================================================================

def render_sidebar(queue: list, reviews: dict, current_index: int) -> None:
    st.sidebar.title("Progress")

    total     = len(queue)
    reviewed  = len(reviews)
    remaining = total - reviewed

    st.sidebar.metric("Total",      total)
    st.sidebar.metric("Reviewed",   reviewed)
    st.sidebar.metric("Remaining",  remaining)

    if total > 0:
        pct = reviewed / total * 100
        st.sidebar.progress(reviewed / total, text=f"{pct:.1f}%")

    if reviews:
        st.sidebar.divider()
        st.sidebar.subheader("Decisions")
        counts = {}
        for v in reviews.values():
            d = v.get("decision", v) if isinstance(v, dict) else v
            counts[d] = counts.get(d, 0) + 1

        for key, label in DECISION_LABELS.items():
            st.sidebar.metric(label, counts.get(key, 0))

    st.sidebar.divider()
    st.sidebar.subheader("Settings")
    st.sidebar.caption(f"Queue: `{QUEUE_FILE}`")
    st.sidebar.caption(f"Reviews: `{REVIEWS_FILE}`")
    st.sidebar.caption(f"Corrected masks: `{CLEAN_MASK_DIR}`")

    # Navigation: jump to a specific index
    target = st.sidebar.number_input(
        "Jump to blob index (0-based)",
        min_value=0, max_value=max(0, total - 1),
        value=current_index, step=1,
    )
    if st.sidebar.button("Jump"):
        st.session_state.current_index = int(target)
        st.rerun()


# =============================================================================
# KLASSEN-BADGE
# =============================================================================

def class_badge(class_id: int, class_name: str) -> str:
    """Return an HTML badge for a class."""
    color = CLASS_COLORS_HEX.get(class_id, "#888888")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-weight:bold;font-size:0.9em;">'
        f'{class_id} – {class_name}</span>'
    )


# =============================================================================
# HAUPTANWENDUNG
# =============================================================================

def main():
    st.set_page_config(
        page_title="Label Review",
        page_icon="🏷️",
        layout="wide",
    )

    # ------------------------------------------------------------------
    # Session-State initialisieren
    # ------------------------------------------------------------------
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Queue + Reviews laden
    queue   = load_queue(QUEUE_FILE)
    reviews = load_reviews(REVIEWS_FILE)

    # Sidebar
    render_sidebar(queue, reviews, st.session_state.current_index)

    # ------------------------------------------------------------------
    # No queue present
    # ------------------------------------------------------------------
    if not queue:
        st.title("Label Review Tool")
        st.error(f"No queue found: `{QUEUE_FILE}`")
        st.info("Please run `python generate_queue.py` first.")
        st.stop()

    # ------------------------------------------------------------------
    # Skip already reviewed items
    # ------------------------------------------------------------------
    # Find next not-reviewed item starting from current_index
    start_idx = st.session_state.current_index
    idx       = start_idx

    while idx < len(queue) and queue[idx]["region_id"] in reviews:
        idx += 1

    # All done?
    if idx >= len(queue):
        st.title("Label Review Tool")
        st.success(f"All {len(queue)} blobs have been reviewed!")

        total_patched = sum(
            1 for v in reviews.values()
            if (v.get("decision") if isinstance(v, dict) else v) == "accept_prediction"
        )
        st.metric("Masks patched", total_patched)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Restart", use_container_width=True):
                st.session_state.current_index = 0
                st.rerun()
        with col2:
            # Export: show reviews as JSON
            if st.button("Show reviews (JSON)", use_container_width=True):
                st.json(reviews)
        st.stop()

    st.session_state.current_index = idx
    item = queue[idx]

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    st.title("Label Review Tool")

    col_h1, col_h2, col_h3, col_h4 = st.columns([2, 2, 2, 2])
    with col_h1:
        st.metric("Blob", f"{idx + 1} / {len(queue)}")
    with col_h2:
        st.metric("Size", f"{item['blob_size_px']:,} px")
    with col_h3:
        st.metric("Mean Confidence", f"{item['confidence_mean']:.1%}")
    with col_h4:
        st.metric("Min Confidence", f"{item['confidence_min']:.1%}")

    st.divider()

    # ------------------------------------------------------------------
    # SNIPPET-INFO
    # ------------------------------------------------------------------
    st.subheader(f"{item['snippet']}")
    st.caption(f"Region ID: `{item['region_id']}`")

    col_gt, col_arrow, col_pred = st.columns([5, 1, 5])
    with col_gt:
        st.markdown("**Ground Truth (current label):**")
        st.markdown(
            class_badge(item["label_class_id"], item["label_class_name"]),
            unsafe_allow_html=True,
        )
    with col_arrow:
        st.markdown("<br><br><h2 style='text-align:center'>→</h2>", unsafe_allow_html=True)
    with col_pred:
        st.markdown("**Modell-Vorhersage:**")
        st.markdown(
            class_badge(item["pred_class_id"], item["pred_class_name"]),
            unsafe_allow_html=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # OVERLAY-BILDER
    # ------------------------------------------------------------------
    st.subheader("Visual analysis")
    st.caption(
        "Highlighted area = flagged blob. "
        "Left: RGB image · Middle: current label · Right: model prediction"
    )

    col_rgb, col_gt_img, col_pred_img = st.columns(3)

    rgb_img  = load_image_safe(item.get("overlay_rgb_path"))
    gt_img   = load_image_safe(item.get("overlay_gt_path"))
    pred_img = load_image_safe(item.get("overlay_pred_path"))

    placeholder = placeholder_image("Missing image")

    with col_rgb:
        st.image(rgb_img or placeholder, caption="RGB (true color)", use_container_width=True)
    with col_gt_img:
        caption_gt = f"Label: {item['label_class_name']}"
        st.image(gt_img or placeholder, caption=caption_gt, use_container_width=True)
    with col_pred_img:
        caption_pred = f"Prediction: {item['pred_class_name']} ({item['confidence_mean']:.0%})"
        st.image(pred_img or placeholder, caption=caption_pred, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # ENTSCHEIDUNGS-BUTTONS
    # ------------------------------------------------------------------
    st.subheader("Your decision")

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    def decide(decision: str) -> None:
        """Process a decision, optionally patch the mask, and continue."""
        if decision == "accept_prediction":
            success = patch_and_save_mask(
                mask_npy_path = item["mask_npy_path"],
                blob_mask_npy = item["blob_mask_npy"],
                pred_class_id = item["pred_class_id"],
            )
            if not success:
                return   # Error already shown by patch_and_save_mask

        record_decision(
            region_id    = item["region_id"],
            decision     = decision,
            item         = item,
            reviews      = reviews,
            reviews_file = REVIEWS_FILE,
        )
        st.session_state.current_index = idx + 1
        st.rerun()

    with col_b1:
        if st.button(
            f"Accept prediction\n{item['pred_class_name']}",
            use_container_width=True, key="btn_accept",
            type="primary",
            help="Set blob pixels in the mask to the model prediction and save.",
        ):
            decide("accept_prediction")

    with col_b2:
        if st.button(
            f"Keep label\n{item['label_class_name']}",
            use_container_width=True, key="btn_keep",
            help="The original label is correct. No change.",
        ):
            decide("keep_label")

    with col_b3:
        if st.button(
            "Both wrong\nManual correction",
            use_container_width=True, key="btn_both",
            help="Neither label nor prediction is correct. Mark blob as unclear.",
        ):
            decide("both_wrong")

    with col_b4:
        if st.button(
            "Skip\nDecide later",
            use_container_width=True, key="btn_skip",
            help="Skip this blob for now (it remains in the queue).",
        ):
            decide("skipped")

    st.divider()

    # ------------------------------------------------------------------
    # KONTEXT: Alle Blobs dieses Snippets
    # ------------------------------------------------------------------
    same_snippet = [
        (i, q) for i, q in enumerate(queue)
        if q["snippet"] == item["snippet"]
    ]

    if len(same_snippet) > 1:
        with st.expander(
            f"Show all {len(same_snippet)} blobs in this snippet", expanded=False
        ):
            for q_idx, q_item in same_snippet:
                reviewed_marker = "[x]" if q_item["region_id"] in reviews else "[ ]"
                active_marker   = " ← current" if q_idx == idx else ""
                label = (
                    f"{reviewed_marker} Blob {q_idx + 1}: "
                    f"{q_item['label_class_name']} → {q_item['pred_class_name']} "
                    f"({q_item['confidence_mean']:.0%}, {q_item['blob_size_px']:,} px)"
                    f"{active_marker}"
                )
                if st.button(label, key=f"nav_blob_{q_idx}"):
                    st.session_state.current_index = q_idx
                    st.rerun()

    # ------------------------------------------------------------------
    # LEGENDE
    # ------------------------------------------------------------------
    with st.expander("Class legend", expanded=False):
        cols = st.columns(4)
        for i, (cid, cname) in enumerate(CLASS_NAMES.items()):
            with cols[i % 4]:
                st.markdown(
                    class_badge(cid, cname),
                    unsafe_allow_html=True,
                )
                st.write("")


if __name__ == "__main__":
    main()