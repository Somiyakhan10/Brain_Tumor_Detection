"""
dashboard_utils.py
==================
Chart builders (Plotly) + demo/fallback data generators.
All chart functions return go.Figure objects ready for st.plotly_chart().
"""

import io
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from model_utils import CLASS_NAMES, CLASS_COLORS

# ─── Palette ──────────────────────────────────────────────────────────────────

PALETTE = {
    'bg':           '#FFFFFF',
    'surface':      '#F8FAFC',
    'border':       '#E2E8F0',
    'primary':      '#0F172A',
    'accent':       '#6366F1',
    'accent2':      '#06B6D4',
    'muted':        '#64748B',
    'train_color':  '#6366F1',
    'val_color':    '#06B6D4',
}

_PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color=PALETTE['primary']),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(
        gridcolor=PALETTE['border'],
        showline=True,
        linecolor=PALETTE['border'],
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor=PALETTE['border'],
        showline=True,
        linecolor=PALETTE['border'],
        zeroline=False,
    ),
)


def _apply_layout(fig: go.Figure, title: str = '') -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=PALETTE['primary'])),
                      **_PLOTLY_LAYOUT)
    return fig


# ─── Demo Data ────────────────────────────────────────────────────────────────

def generate_demo_training_history(epochs: int = 25) -> dict:
    """Realistic-looking training curves for demo / no-checkpoint mode."""
    rng = np.random.default_rng(42)
    ep  = np.arange(1, epochs + 1)

    train_loss = 1.4 * np.exp(-0.18 * ep) + 0.07 + rng.normal(0, 0.012, epochs)
    val_loss   = 1.5 * np.exp(-0.15 * ep) + 0.11 + rng.normal(0, 0.022, epochs)
    train_acc  = 1 - 0.62 * np.exp(-0.22 * ep) + rng.normal(0, 0.006, epochs)
    val_acc    = 1 - 0.65 * np.exp(-0.19 * ep) + rng.normal(0, 0.010, epochs)

    return {
        'epochs':      ep.tolist(),
        'train_loss':  np.clip(train_loss, 0, 2).tolist(),
        'val_loss':    np.clip(val_loss,   0, 2).tolist(),
        'train_acc':   np.clip(train_acc,  0, 1).tolist(),
        'val_acc':     np.clip(val_acc,    0, 1).tolist(),
    }


DEMO_METRICS = {
    'accuracy':  0.9742,
    'precision': 0.9748,
    'recall':    0.9742,
    'f1':        0.9744,
    'num_classes': 4,
    'total_params': 14_307_972,
}

DEMO_CONF_MATRIX = np.array([
    [293,   4,   2,   1],
    [  3, 301,   5,   1],
    [  1,   2, 298,   1],
    [  0,   1,   0, 305],
])

def generate_demo_roc_data() -> dict:
    """Synthetic per-class ROC curves."""
    rng = np.random.default_rng(0)
    result = {}
    aucs = {'glioma': 0.998, 'meningioma': 0.991, 'pituitary': 0.997, 'notumor': 0.999}
    for cls in CLASS_NAMES:
        n  = 300
        scores = rng.beta(8, 2, n)   # good classifier → scores cluster near 1
        labels = (scores > 0.45).astype(int)
        # rough FPR/TPR
        thresholds = np.linspace(0, 1, 200)
        fpr, tpr = [], []
        for t in thresholds:
            pred = (scores >= t).astype(int)
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            tn = ((pred == 0) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            fpr.append(fp / (fp + tn + 1e-9))
            tpr.append(tp / (tp + fn + 1e-9))
        result[cls] = {'fpr': fpr[::-1], 'tpr': tpr[::-1], 'auc': aucs[cls]}
    return result


# ─── Chart Builders ───────────────────────────────────────────────────────────

def make_training_chart(history: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Loss', 'Accuracy'],
        horizontal_spacing=0.12,
    )
    ep = history['epochs']

    for row, (train_key, val_key, yname) in enumerate([
        ('train_loss', 'val_loss',  'Loss'),
        ('train_acc',  'val_acc',   'Accuracy'),
    ], start=1):
        fig.add_trace(
            go.Scatter(
                x=ep, y=history[train_key],
                name='Training',
                line=dict(color=PALETTE['train_color'], width=2.5),
                hovertemplate=f'Epoch %{{x}}<br>Train {yname}: %{{y:.4f}}<extra></extra>',
                showlegend=(row == 1),
            ), row=1, col=row,
        )
        fig.add_trace(
            go.Scatter(
                x=ep, y=history[val_key],
                name='Validation',
                line=dict(color=PALETTE['val_color'], width=2.5, dash='dot'),
                hovertemplate=f'Epoch %{{x}}<br>Val {yname}: %{{y:.4f}}<extra></extra>',
                showlegend=(row == 1),
            ), row=1, col=row,
        )

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text='Training Analytics', font=dict(size=16, color=PALETTE['primary'])),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=360,
    )
    for ax in ['xaxis', 'xaxis2', 'yaxis', 'yaxis2']:
        fig.update_layout(**{ax: dict(
            gridcolor=PALETTE['border'],
            showline=True, linecolor=PALETTE['border'], zeroline=False,
            title_font=dict(size=12),
        )})
    return fig


def make_confusion_matrix_chart(conf_matrix: np.ndarray) -> go.Figure:
    labels      = [c.capitalize() for c in CLASS_NAMES]
    row_sums    = conf_matrix.sum(axis=1, keepdims=True)
    conf_norm   = conf_matrix / (row_sums + 1e-9)

    text_vals = [[
        f'<b>{conf_matrix[i,j]}</b><br>({conf_norm[i,j]:.1%})'
        for j in range(len(CLASS_NAMES))
    ] for i in range(len(CLASS_NAMES))]

    fig = go.Figure(go.Heatmap(
        z=conf_norm,
        x=labels, y=labels,
        text=text_vals,
        texttemplate='%{text}',
        textfont=dict(size=12, family='DM Sans'),
        colorscale=[
            [0.0,  '#EFF6FF'],
            [0.5,  '#93C5FD'],
            [1.0,  '#1D4ED8'],
        ],
        showscale=True,
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Norm: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text='Confusion Matrix', font=dict(size=16)),
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=420,
    )
    return fig


def make_roc_chart(roc_data: dict) -> go.Figure:
    fig = go.Figure()

    for cls in CLASS_NAMES:
        d   = roc_data[cls]
        col = CLASS_COLORS[cls]
        fig.add_trace(go.Scatter(
            x=d['fpr'], y=d['tpr'],
            name=f'{cls.capitalize()} (AUC={d["auc"]:.3f})',
            line=dict(color=col, width=2.5),
            mode='lines',
            hovertemplate=f'{cls.capitalize()}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>',
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        line=dict(color=PALETTE['muted'], width=1.5, dash='dash'),
        mode='lines',
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text='ROC Curve Analysis', font=dict(size=16)),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.55, y=0.05, bgcolor='rgba(255,255,255,0.8)'),
        height=440,
    )
    return fig


def make_probability_chart(probs: list[float]) -> go.Figure:
    labels = [c.capitalize() for c in CLASS_NAMES]
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    pct    = [p * 100 for p in probs]

    fig = go.Figure(go.Bar(
        y=labels, x=pct,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.05)', width=1),
        ),
        text=[f'{p:.1f}%' for p in pct],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
    ))

    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text='Class Probability Distribution', font=dict(size=15)),
        xaxis=dict(title='Probability (%)', range=[0, 115], **_PLOTLY_LAYOUT.get('xaxis', {})),
        height=280,
        showlegend=False,
    )
    return fig


# ─── Image / Download Helpers ─────────────────────────────────────────────────

def ndarray_to_b64_png(arr: np.ndarray) -> str:
    """Encode a (H,W,3) uint8 ndarray as base-64 PNG string."""
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def make_results_dataframe(metrics: dict) -> pd.DataFrame:
    rows = [
        ('Accuracy',  f'{metrics["accuracy"]:.4f}'),
        ('Precision', f'{metrics["precision"]:.4f}'),
        ('Recall',    f'{metrics["recall"]:.4f}'),
        ('F1 Score',  f'{metrics["f1"]:.4f}'),
    ]
    return pd.DataFrame(rows, columns=['Metric', 'Value'])


def make_prediction_report(
    pred_label: str,
    confidence: float,
    probs: list[float],
    metrics: dict,
) -> str:
    """Return a plain-text prediction report."""
    lines = [
        '=' * 60,
        'BRAIN TUMOR CLASSIFICATION — PREDICTION REPORT',
        '=' * 60,
        '',
        f'Predicted Class : {pred_label.upper()}',
        f'Confidence      : {confidence:.2%}',
        '',
        'Class Probabilities:',
    ]
    for cls, p in zip(CLASS_NAMES, probs):
        lines.append(f'  {cls:<15} {p:.4f}  ({p*100:.2f}%)')
    lines += [
        '',
        'Model Evaluation Metrics:',
        f'  Accuracy  : {metrics.get("accuracy",  "N/A")}',
        f'  Precision : {metrics.get("precision", "N/A")}',
        f'  Recall    : {metrics.get("recall",    "N/A")}',
        f'  F1 Score  : {metrics.get("f1",        "N/A")}',
        '',
        'Model: DenseNet169 + Transfer Learning',
        'XAI  : Gradient-weighted Class Activation Mapping (Grad-CAM)',
        '=' * 60,
    ]
    return '\n'.join(lines)
