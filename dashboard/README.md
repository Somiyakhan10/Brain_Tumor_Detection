# 🧠 Interpretable Brain Tumor Classification System

> Deep Learning-Based MRI Analysis with Explainable AI using Grad-CAM

---

## Overview

A professional, fully-interactive Streamlit dashboard that wraps a **DenseNet169** brain
tumour classifier trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

The dashboard supports:
- Live MRI upload & real-time inference
- 3-panel Grad-CAM explainability (Original · Heatmap · Overlay)
- Interactive training analytics, confusion matrix, and ROC curves
- Prediction report & Grad-CAM image download
- Full results export (CSV)

---

## Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Project Overview | KPI metric cards (Accuracy, Precision, Recall, F1, Classes, Parameters) |
| 2 | Training Analytics | Interactive loss & accuracy curves (Plotly) |
| 3 | Confusion Matrix | Interactive heatmap with per-cell counts & normalised values |
| 4 | ROC Curve Analysis | Multi-class ROC curves with per-class AUC |
| 5 | MRI Prediction Portal | Upload MRI → get prediction + confidence |
| 6 | Probability Distribution | Horizontal bar chart of class probabilities |
| 7 | Grad-CAM (XAI) | 3-panel visualisation: Original · Heatmap · Overlay |
| 8 | Class Explorer | Per-class description, statistics, and AUC |
| 9 | Model Interpretability | Grad-CAM methodology, DenseNet architecture, preprocessing |
| 10 | Results Summary | Exportable metrics table |

---

## Project Structure

```
brain_tumor_dashboard/
├── app.py               # Main Streamlit application
├── model_utils.py       # DenseNet169 architecture, loader, constants
├── gradcam_utils.py     # GradCAM class + 3-panel generation
├── dashboard_utils.py   # Plotly charts + demo/fallback data
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 · Clone / copy the project

```bash
cd brain_tumor_dashboard
```

### 2 · Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install PyTorch first (adjust for your OS/CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest
pip install -r requirements.txt
```

### 3 · (Optional) Place your trained model checkpoint

Copy `brain_tumor_classifier.pth` (saved by `main()` in the training script)
into the `brain_tumor_dashboard/` directory.

If no checkpoint is found the dashboard runs in **demo mode** with random weights
and pre-generated example metrics — fully functional for UI demonstration.

### 4 · Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model Architecture

```
BrainTumorClassifier
└── backbone: DenseNet-169 (ImageNet pre-trained)
    └── classifier:
        ├── Dropout(0.5)
        ├── Linear(1664 → 512)
        ├── ReLU()
        ├── Dropout(0.5)
        └── Linear(512 → 4)
```

**Classes:** `glioma` · `meningioma` · `pituitary` · `notumor`

---

## Inference Pipeline

1. PIL image → resize 224×224 → ImageNet normalisation
2. Forward pass through DenseNet169
3. Softmax probabilities → `argmax` = predicted class
4. Grad-CAM: backprop target class score → global-avg-pool gradients at last
   conv block → weighted activation sum → ReLU → bilinear upsample → JET LUT
5. 3-panel render: Original · Heatmap · Overlay (60% orig + 40% heat)

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.3
- Streamlit ≥ 1.35
- See `requirements.txt` for full list

---

## License

For research and educational use. The training dataset is subject to its own
[Kaggle licence](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
