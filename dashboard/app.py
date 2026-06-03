"""
app.py
======
Interpretable Brain Tumor Classification System
Streamlit dashboard — Sections 1-10 per spec.
"""

import io
import os
import base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ─── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title='Brain Tumor AI | Interpretable Classification System',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─── Local imports ────────────────────────────────────────────────────────────
from model_utils import (
    CLASS_NAMES, CLASS_COLORS, CLASS_DESCRIPTIONS,
    DEVICE, load_model, count_parameters,
)
from gradcam_utils import generate_gradcam_panels
from dashboard_utils import (
    DEMO_METRICS, DEMO_CONF_MATRIX,
    generate_demo_training_history, generate_demo_roc_data,
    make_training_chart, make_confusion_matrix_chart,
    make_roc_chart, make_probability_chart,
    make_results_dataframe, make_prediction_report,
    ndarray_to_b64_png,
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
  }
  [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
  [data-testid="stSidebar"] .sidebar-badge {
    display: inline-block;
    background: rgba(99,102,241,0.25);
    color: #A5B4FC !important;
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px 2px;
  }

  /* ── Main area ── */
  .main .block-container { padding-top: 1.5rem; max-width: 1380px; }
  .stApp { background: #F8FAFC; }

  /* ── Hero header ── */
  .hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2040 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 75% 50%, rgba(99,102,241,0.25) 0%, transparent 60%);
  }
  .hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.3);
    color: #A5B4FC;
    border: 1px solid rgba(99,102,241,0.5);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
  }
  .hero h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #F1F5F9 !important;
    line-height: 1.2 !important;
    margin: 0 0 0.5rem !important;
  }
  .hero p {
    color: #94A3B8;
    font-size: 0.95rem;
    margin: 0;
    max-width: 600px;
  }
  .hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #CBD5E1;
    margin: 1rem 6px 0 0;
  }
  .hero-pill span { font-size: 0.9rem; }

  /* ── Metric cards ── */
  .metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    display: flex; flex-direction: column; gap: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow .2s;
  }
  .metric-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
  .metric-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
                  text-transform: uppercase; color: #64748B; }
  .metric-value { font-size: 2rem; font-weight: 700; color: #0F172A;
                  font-family: 'DM Mono', monospace; line-height: 1.1; }
  .metric-sub   { font-size: 0.75rem; color: #94A3B8; }
  .metric-icon  { font-size: 1.4rem; margin-bottom: 4px; }

  /* ── Section headers ── */
  .section-header {
    font-size: 1.15rem; font-weight: 700; color: #0F172A;
    border-bottom: 2px solid #6366F1;
    padding-bottom: 8px; margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: 10px;
  }

  /* ── Prediction panel ── */
  .pred-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  }
  .pred-label {
    font-size: 1.6rem; font-weight: 700; color: #0F172A;
    font-family: 'Playfair Display', serif;
  }
  .pred-confidence {
    font-size: 1rem; color: #64748B; margin-top: 2px;
  }
  .confidence-badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 1.1rem;
    color: #fff;
    margin-top: 10px;
  }

  /* ── Grad-CAM panels ── */
  .panel-label {
    text-align: center;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748B;
    margin-top: 6px;
  }

  /* ── Class explorer cards ── */
  .class-card {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
  }
  .class-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    vertical-align: middle;
  }

  /* ── Table ── */
  .results-table { border-radius: 10px; overflow: hidden; }

  /* ── Info box ── */
  .info-box {
    background: #EFF6FF;
    border-left: 4px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.875rem;
    color: #1E40AF;
    margin: 1rem 0;
  }
  .warn-box {
    background: #FFFBEB;
    border-left: 4px solid #F59E0B;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.875rem;
    color: #92400E;
    margin: 1rem 0;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 12px !important;
    background: #F8FAFC !important;
  }

  /* Plotly chart border */
  .stPlotlyChart { border-radius: 12px; overflow: hidden; }

  /* Hide Streamlit branding */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    checkpoint = 'brain_tumor_classifier.pth'
    model = load_model(checkpoint if os.path.exists(checkpoint) else None)
    return model

@st.cache_data(show_spinner=False)
def get_demo_data():
    return {
        'history': generate_demo_training_history(25),
        'roc':     generate_demo_roc_data(),
        'metrics': DEMO_METRICS,
        'conf_matrix': DEMO_CONF_MATRIX,
    }


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Brain Tumor AI")
    st.markdown("---")

    st.markdown("### Project Overview")
    st.markdown(
        "Deep learning system for **automated MRI-based brain tumor "
        "classification** with explainable AI via Grad-CAM."
    )
    st.markdown("---")

    st.markdown("### Dataset")
    for line in [
        "📁 Brain Tumor MRI Dataset",
        "🏷 4 Classes",
        "📷 ~7,000 images",
        "✂️ Train / Test split",
    ]:
        st.markdown(line)
    st.markdown("---")

    st.markdown("### Model")
    for badge in ['DenseNet169', 'Transfer Learning', 'Grad-CAM XAI', 'PyTorch']:
        st.markdown(f'<span class="sidebar-badge">{badge}</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Classes")
    for cls in CLASS_NAMES:
        color = CLASS_COLORS[cls]
        st.markdown(
            f'<span class="class-dot" style="background:{color}"></span> '
            f'**{cls.capitalize()}**',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.72rem;color:#475569;">Interpretable Brain Tumor AI · v1.0</p>',
        unsafe_allow_html=True,
    )


# ─── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-tag">Medical AI · Deep Learning · Explainable AI</div>
  <h1>Interpretable Brain Tumor<br>Classification System</h1>
  <p>Deep Learning-Based MRI Analysis with Explainable AI using Grad-CAM</p>
  <div>
    <span class="hero-pill"><span>🏥</span> Clinical-Grade Pipeline</span>
    <span class="hero-pill"><span>🔬</span> DenseNet169</span>
    <span class="hero-pill"><span>👁</span> Grad-CAM Explainability</span>
    <span class="hero-pill"><span>⚡</span> Real-Time Inference</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Load resources ───────────────────────────────────────────────────────────

with st.spinner("Initialising model…"):
    model   = get_model()
    demo    = get_demo_data()
    metrics = demo['metrics']
    metrics['total_params'] = count_parameters(model)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · Project Overview — Metric Cards
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📊 Project Overview</div>', unsafe_allow_html=True)

card_data = [
    ('🎯', 'Test Accuracy',  f'{metrics["accuracy"]:.2%}',  'Weighted avg'),
    ('🎯', 'Precision',      f'{metrics["precision"]:.2%}', 'Weighted avg'),
    ('📡', 'Recall',         f'{metrics["recall"]:.2%}',    'Weighted avg'),
    ('⚖️', 'F1 Score',       f'{metrics["f1"]:.2%}',        'Weighted avg'),
    ('🏷', 'Classes',        str(metrics['num_classes']),   'Tumor types'),
    ('🧮', 'Parameters',     f'{metrics["total_params"]/1e6:.1f}M', 'DenseNet169'),
]

cols = st.columns(6)
for col, (icon, label, value, sub) in zip(cols, card_data):
    col.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-icon">{icon}</div>
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · Training Analytics
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📈 Training Analytics</div>', unsafe_allow_html=True)

history = demo['history']
fig_train = make_training_chart(history)
st.plotly_chart(fig_train, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🔢 Confusion Matrix</div>', unsafe_allow_html=True)

col_cm, col_cm_info = st.columns([2, 1])
with col_cm:
    fig_cm = make_confusion_matrix_chart(demo['conf_matrix'])
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cm_info:
    st.markdown("#### How to Read")
    st.markdown(
        "Each row represents the **true class**; each column the **predicted class**. "
        "Diagonal cells (top-left → bottom-right) are correct predictions. "
        "Off-diagonal cells are misclassifications."
    )
    st.markdown("#### Per-Class Accuracy")
    for i, cls in enumerate(CLASS_NAMES):
        row    = demo['conf_matrix'][i]
        acc    = row[i] / row.sum()
        color  = CLASS_COLORS[cls]
        st.markdown(
            f'<div style="margin:6px 0">'
            f'<span class="class-dot" style="background:{color}"></span>'
            f'<b>{cls.capitalize()}</b>: {acc:.2%}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · ROC Curves
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📉 ROC Curve Analysis</div>', unsafe_allow_html=True)

fig_roc = make_roc_chart(demo['roc'])
st.plotly_chart(fig_roc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5–7 · MRI Prediction Portal + Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🏥 MRI Image Prediction Portal</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="info-box">Upload a brain MRI scan (JPG / JPEG / PNG). '
    'The model will predict the tumor class and generate a Grad-CAM '
    'heatmap highlighting the regions that influenced the decision.</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    'Drop an MRI image here or click to browse',
    type=['jpg', 'jpeg', 'png'],
    label_visibility='collapsed',
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert('RGB')

    with st.spinner('Running inference + Grad-CAM…'):
        result = generate_gradcam_panels(model, pil_img)

    pred_label  = result['pred_label']
    confidence  = result['confidence']
    probs       = result['probabilities']
    badge_color = CLASS_COLORS[pred_label]

    # ── Prediction summary ──────────────────────────────────────────────────
    pcol1, pcol2 = st.columns([1, 2])

    with pcol1:
        st.markdown('<div class="pred-card">', unsafe_allow_html=True)
        st.image(pil_img, caption='Uploaded MRI', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with pcol2:
        st.markdown('<div class="pred-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pred-label">Prediction: {pred_label.capitalize()}</div>'
            f'<div class="pred-confidence">Confidence Score</div>'
            f'<div class="confidence-badge" style="background:{badge_color}">'
            f'{confidence:.2%}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # ── SECTION 6: Probability bar chart ──
        fig_prob = make_probability_chart(probs)
        st.plotly_chart(fig_prob, use_container_width=True)

        # Download report
        report_txt = make_prediction_report(pred_label, confidence, probs, metrics)
        st.download_button(
            '📥 Download Prediction Report',
            data=report_txt,
            file_name='prediction_report.txt',
            mime='text/plain',
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 7: Grad-CAM 3-panel ────────────────────────────────────────
    st.markdown(
        '<div class="section-header">👁 Explainable AI — Grad-CAM Visualisation</div>',
        unsafe_allow_html=True,
    )

    g1, g2, g3 = st.columns(3)
    panels = [
        (g1, result['original_rgb'],  'Panel 1 — Original MRI'),
        (g2, result['heatmap_rgb'],   'Panel 2 — Grad-CAM Heatmap'),
        (g3, result['overlay_rgb'],   'Panel 3 — Overlay (Tumour Location)'),
    ]
    for col, arr, label in panels:
        with col:
            st.image(arr, use_container_width=True)
            st.markdown(f'<div class="panel-label">{label}</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box" style="margin-top:1rem;">'
        '🔴 <b>Red / warm regions</b> indicate areas that contributed <b>most strongly</b> '
        'to the model\'s decision. '
        '🔵 <b>Blue / cool regions</b> had little influence. '
        'The overlay (Panel 3) superimposes the heatmap onto the original MRI to show '
        'exact tumour localisation.</div>',
        unsafe_allow_html=True,
    )

    # Download Grad-CAM overlay
    overlay_png = ndarray_to_b64_png(result['overlay_rgb'])
    st.download_button(
        '📥 Download Grad-CAM Overlay',
        data=base64.b64decode(overlay_png),
        file_name='gradcam_overlay.png',
        mime='image/png',
    )

else:
    st.markdown(
        '<div class="warn-box">No image uploaded yet. '
        'Upload a brain MRI scan above to run the classification pipeline.</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 · Class Explorer
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🔬 Class Explorer</div>', unsafe_allow_html=True)

tabs = st.tabs([c.capitalize() for c in CLASS_NAMES])
for tab, cls in zip(tabs, CLASS_NAMES):
    with tab:
        color = CLASS_COLORS[cls]
        desc  = CLASS_DESCRIPTIONS[cls]

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f'<div class="class-card">'
                f'<h4 style="color:{color};margin-top:0">'
                f'<span class="class-dot" style="background:{color}"></span>'
                f'{cls.capitalize()}</h4>'
                f'<p style="color:#374151;font-size:0.9rem">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            # class stats from demo confusion matrix
            idx     = CLASS_NAMES.index(cls)
            row     = demo['conf_matrix'][idx]
            total   = int(row.sum())
            correct = int(row[idx])
            acc     = correct / (total + 1e-9)
            auc_val = demo['roc'][cls]['auc']

            st.markdown(
                f'<div class="class-card">'
                f'<b>Class Statistics</b>'
                f'<hr style="border-color:#E2E8F0;margin:8px 0">'
                f'<table style="width:100%;font-size:0.87rem;color:#374151">'
                f'<tr><td>Test samples</td><td><b>{total}</b></td></tr>'
                f'<tr><td>Correct</td><td><b>{correct}</b></td></tr>'
                f'<tr><td>Accuracy</td><td><b>{acc:.2%}</b></td></tr>'
                f'<tr><td>AUC-ROC</td><td><b>{auc_val:.3f}</b></td></tr>'
                f'</table></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 · Model Interpretability
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🤖 Model Interpretability</div>', unsafe_allow_html=True)

with st.expander("Grad-CAM Methodology", expanded=True):
    st.markdown("""
**Gradient-weighted Class Activation Mapping (Grad-CAM)** produces a coarse localisation map
that highlights the important regions in the image for predicting the concept.

**Algorithm steps:**
1. Pass the input image through the DenseNet169 forward pass.
2. Obtain the gradient of the score for the predicted class with respect to the
   feature maps of the last convolutional layer.
3. Global-average-pool the gradients over the spatial dimensions to obtain *importance weights*.
4. Compute a weighted combination of activation maps and apply ReLU.
5. Up-sample the resulting CAM to the input resolution.
6. Overlay onto the original image using a JET colourmap.
""")

with st.expander("DenseNet169 Architecture"):
    st.markdown("""
| Component | Detail |
|-----------|--------|
| Backbone  | DenseNet-169 (ImageNet pre-trained) |
| Dense Blocks | 4 blocks with 32 growth rate |
| Custom Head | Dropout → Linear(1664→512) → ReLU → Dropout → Linear(512→4) |
| Parameters | ~14.3 M |
| Input size | 224 × 224 × 3 |
| Output | Softmax over 4 classes |

**Why DenseNet?** Dense connections ensure maximum information flow between layers,
mitigate vanishing-gradient problems, and encourage feature reuse — critical for
subtle medical imaging tasks.
""")

with st.expander("Feature Extraction & Transfer Learning"):
    st.markdown("""
The model uses **ImageNet pre-trained weights** as the starting point.
All layers are fine-tuned end-to-end during training.

**Pre-processing pipeline (matching training):**
- Resize to 224 × 224
- Normalise with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

**Training augmentations (training only):**
- Random horizontal flip (p = 0.5)
- Random rotation ±15°
""")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 · Results Summary
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📋 Results Summary</div>', unsafe_allow_html=True)

df_results = make_results_dataframe(metrics)
st.dataframe(df_results, use_container_width=True, hide_index=True)

csv_bytes = df_results.to_csv(index=False).encode()
st.download_button(
    '📥 Export Results as CSV',
    data=csv_bytes,
    file_name='brain_tumor_results.csv',
    mime='text/csv',
)

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#94A3B8;font-size:0.8rem;">'
    'Interpretable Brain Tumor Classification System · DenseNet169 + Grad-CAM · '
    'Built with Streamlit & Plotly</p>',
    unsafe_allow_html=True,
)
