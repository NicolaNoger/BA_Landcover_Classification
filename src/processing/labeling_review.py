import streamlit as st
import json
import os
import numpy as np
from PIL import Image
from datetime import datetime

# =============================================================================
# KONFIGURATION
# =============================================================================
QUEUE_FILE = "review_queue.json"
REVIEWS_FILE = "reviews_log.json"
DEMO_MODE = True  # Setze auf False wenn echte Queue-Datei vorhanden

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def create_dummy_image(shape, label_text):
    """Erstelle ein Dummy-Bild mit Text."""
    arr = np.random.randint(0, 255, shape, dtype=np.uint8)
    img = Image.fromarray(arr)
    return img

def load_queue():
    """Lade oder erstelle Demo-Queue."""
    if os.path.exists(QUEUE_FILE) and not DEMO_MODE:
        with open(QUEUE_FILE, 'r') as f:
            return json.load(f)
    else:
        # Demo-Queue mit 5 Beispielen
        return [
            {
                'snippet': f'snippet_{i:04d}',
                'region_id': f'region_{i}',
                'label_class': np.random.choice(['Building', 'Grassland', 'Tree', 'Water']),
                'pred_class': np.random.choice(['Building', 'Grassland', 'Tree', 'Water']),
                'confidence': np.random.uniform(0.7, 0.99),
                'rgb_path': f'dummy_rgb_{i}.png',
                'gt_mask_path': f'dummy_mask_gt_{i}.png',
                'pred_mask_path': f'dummy_mask_pred_{i}.png',
            }
            for i in range(5)
        ]

def load_reviews():
    """Lade bisherige Reviews."""
    if os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_review(decision, snippet_id):
    """Speichere eine Review-Entscheidung."""
    reviews = load_reviews()
    reviews[snippet_id] = {
        'decision': decision,
        'timestamp': datetime.now().isoformat()
    }
    with open(REVIEWS_FILE, 'w') as f:
        json.dump(reviews, f, indent=2)

def get_dummy_or_real_image(path):
    """Lade Bild oder zeige Dummy an."""
    if os.path.exists(path):
        return Image.open(path)
    else:
        # Generiere Dummy (RGB, Maske, etc.)
        if 'rgb' in path.lower():
            return create_dummy_image((128, 128, 3), "RGB")
        else:
            return create_dummy_image((128, 128, 3), "Mask")

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(page_title="Label Review Tool", layout="wide")

# 1. State initialisieren
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
    st.session_state.queue = load_queue()
    st.session_state.reviews = load_reviews()

queue = st.session_state.queue
reviews = st.session_state.reviews
current_index = st.session_state.current_index

if current_index >= len(queue):
    st.success("✅ Alle Items reviewed!")
    st.write(reviews)
    if st.button("🔄 Neu starten"):
        st.session_state.current_index = 0
        st.rerun()
else:
    current_item = queue[current_index]
    
    # 2. Header
    st.title("🏷️ Label Review Tool")
    col_progress, col_info = st.columns([1, 4])
    with col_progress:
        st.metric("Progress", f"{current_index + 1}/{len(queue)}")
    
    # 3. Item-Info
    st.divider()
    st.subheader(f"Snippet: {current_item['snippet']}")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.write(f"**Region ID:** {current_item['region_id']}")
    with col_info2:
        st.write(f"**Ground Truth:** {current_item['label_class']}")
    with col_info3:
        st.write(f"**Prediction:** {current_item['pred_class']} (Conf: {current_item['confidence']:.2%})")
    
    st.divider()
    
    # 4. Bilder anzeigen
    st.subheader("Visuelle Vergleich")
    col_rgb, col_gt, col_pred = st.columns(3)
    
    with col_rgb:
        img_rgb = get_dummy_or_real_image(current_item['rgb_path'])
        st.image(img_rgb, caption="RGB Bild", use_container_width=True)
    
    with col_gt:
        img_gt = get_dummy_or_real_image(current_item['gt_mask_path'])
        st.image(img_gt, caption="Ground Truth Maske", use_container_width=True)
    
    with col_pred:
        img_pred = get_dummy_or_real_image(current_item['pred_mask_path'])
        st.image(img_pred, caption="Prediction Maske", use_container_width=True)
    
    st.divider()
    
    # 5. Review Buttons
    st.subheader("Entscheidung treffen")
    
    col_btn_accept, col_btn_keep, col_btn_both, col_btn_skip = st.columns(4)
    
    def handle_decision(decision):
        save_review(decision, current_item['snippet'])
        st.session_state.reviews[current_item['snippet']] = decision
        st.session_state.current_index += 1
        st.rerun()
    
    with col_btn_accept:
        if st.button("✅ Prediction übernehmen", use_container_width=True, key="accept"):
            handle_decision("accept_prediction")
    
    with col_btn_keep:
        if st.button("❌ Label behalten", use_container_width=True, key="keep"):
            handle_decision("keep_label")
    
    with col_btn_both:
        if st.button("⚠️ Beide falsch", use_container_width=True, key="both"):
            handle_decision("both_wrong")
    
    with col_btn_skip:
        if st.button("⏭️ Überspringen", use_container_width=True, key="skip"):
            handle_decision("skipped")
    
    st.divider()
    
    # 6. Statistik
    if reviews:
        st.subheader("📊 Review Statistik")
        decision_counts = {}
        for v in reviews.values():
            decision_counts[v] = decision_counts.get(v, 0) + 1
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("✅ Akzeptiert", decision_counts.get("accept_prediction", 0))
        with col_stat2:
            st.metric("❌ Behalten", decision_counts.get("keep_label", 0))
        with col_stat3:
            st.metric("⚠️ Beide falsch", decision_counts.get("both_wrong", 0))
        with col_stat4:
            st.metric("⏭️ Übersprungen", decision_counts.get("skipped", 0))