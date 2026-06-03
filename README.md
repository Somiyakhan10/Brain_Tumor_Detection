<img width="8811" height="310" alt="deepseek_mermaid_20260603_4e775e" src="https://github.com/user-attachments/assets/e2063ef1-85a3-4981-b002-1b565b0a09c4" /><div align="center">
    <h1> Interpretable Brain Tumor Classification System</h1>
    <h3>Deep Learning-based MRI Analysis with Explainable AI using Grad-CAM</h3>
    
<a href="https://huggingface.co/spaces/somiya-khan01/Brain_Tumor_Classification" target="_blank">
        <button style="background-color: #3b82f6; color: white; font-size: 16px; font-weight: bold; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-family: sans-serif;">
            🚀 Launch Live Demo
        </button>
    </a>
</div>

---

##  About

A **Streamlit-based Web Application** to detect and classify brain tumors from MRI scans. The Deep Learning model is trained on a publicly available dataset of ~7,000 MRI images labeled across four classes: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

The system uses **DenseNet169** architecture with transfer learning from ImageNet. Beyond classification, the application provides **explainable AI** via **Grad-CAM** (Gradient-weighted Class Activation Mapping), which highlights the exact regions in the MRI that influenced the model's decision — making it valuable for clinical interpretation and radiologist validation.

---

##  Live Demo

| Feature | Status |
|---------|--------|
| MRI Upload |  Available |
| Real-time Prediction |  < 2 seconds |
| Grad-CAM Visualization |  3-Panel Display |
| Confidence Score |  Displayed |
| Probability Distribution |  Bar Chart |
| Report Download |  Available |

**Live Demo Link:** [https://huggingface.co/spaces/somiya-khan01/Brain_Tumor_Classification](https://huggingface.co/spaces/somiya-khan01/Brain_Tumor_Classification)

---

##  Model Architecture: DenseNet169

<img src="images/densenet_architecture.png" alt="DenseNet169 Architecture" width="100%"/>

*Figure: Complete DenseNet169 architecture with 4 dense blocks and transition layers*

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 × 3 |
| Base Architecture | DenseNet169 (ImageNet pre-trained) |
| Growth Rate | 32 |
| Dropout Rate | 0.5 |
| Total Parameters | ~14.3 Million |
| Output Classes | 4 (Glioma, Meningioma, Pituitary, No Tumor) |

### Residual / Identity Block

<img width="7086" height="310" alt="deepseek_mermaid_20260603_84f76b" src="https://github.com/user-attachments/assets/570988ba-c25e-42e0-b106-e1a53f7b31ea" />


*Figure: Identity block showing skip connection (Add + ReLU)*

### Convolution Block

<img width="8811" height="310" alt="deepseek_mermaid_20260603_4e775e" src="https://github.com/user-attachments/assets/c63615c6-b834-4916-bd68-ca7e52ad80af" />




*Figure: Convolution block with projection skip connection*

---

##  Grad-CAM Explainability Pipeline

<img width="6565" height="310" alt="deepseek_mermaid_20260603_28b7a4 (1)" src="https://github.com/user-attachments/assets/8a5d9e09-278b-4b1f-83d2-55cdb36a05fa" />



*Figure: Grad-CAM pipeline showing how heatmaps are generated from MRI inputs*

### How Grad-CAM Works

| Step | Description |
|------|-------------|
| 1 | Input MRI is passed through DenseNet169 |
| 2 | Feature maps from last convolutional layer are extracted |
| 3 | Gradients of predicted class are computed |
| 4 | Importance weights are calculated via global average pooling |
| 5 | Weighted combination of activation maps creates heatmap |
| 6 | Heatmap is upsampled and overlaid on original MRI |

---

##  Model Performance

### Sample Test Predictions

<img src="images/sample_predictions.png" alt="Sample Predictions" width="100%"/>

*Figure: Sample predictions from the model on unseen test data*

| True Label | Predicted Label | Confidence | Result |
|------------|----------------|------------|--------|
| Glioma | Meningioma | 77.7% |  Incorrect |
| No Tumor | No Tumor | 87.1% |  Correct |
| Meningioma | Meningioma | 100.0% |  Correct |
| Pituitary | Pituitary | 99.8% |  Correct |
| No Tumor | No Tumor | 100.0% |  Correct |



### Performance Metrics Summary

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 93.96% |
| **Weighted Precision** | 93.98% |
| **Weighted Recall** | 94.12% |
| **Weighted F1 Score** | 94.05% |

### Per-Class Performance

| Class | Test Samples | Correct | Accuracy | AUC-ROC |
|-------|-------------|---------|----------|---------|
| Glioma | 140 | 125 | 89.29% | 0.945 |
| Meningioma | 136 | 118 | 86.76% | 0.928 |
| No Tumor | 150 | 142 | 94.67% | 0.981 |
| Pituitary | 140 | 128 | 91.43% | 0.963 |

---



| Class | AUC-ROC |
|-------|---------|
| Glioma | 0.945 |
| Meningioma | 0.928 |
| No Tumor | 0.981 |
| Pituitary | 0.963 |

---

### Grad-CAM Tumor Localization (3-Panel Display)

#### Example 1: Pituitary Tumor (98.1% Confidence)

<img width="1087" height="390" alt="image" src="https://github.com/user-attachments/assets/2a75cfb9-8127-48fd-a423-aee6e13f9a17" />


| Panel | Description |
|-------|-------------|
| **Panel 1** | Original MRI scan |
| **Panel 2** | Grad-CAM heatmap (red = tumor region) |
| **Panel 3** | Overlay showing tumor localization |

#### Example 2: No Tumor (100.0% Confidence)
<img width="1063" height="382" alt="image" src="https://github.com/user-attachments/assets/22fadcb7-c659-4ce5-bd9b-fc88f3b7fca1" />


| Panel | Description |
|-------|-------------|
| **Panel 1** | Original MRI scan (healthy) |
| **Panel 2** | Minimal red regions (no tumor detected) |
| **Panel 3** | No specific tumor localization |

---

### Interpretation Guide

| Color | Meaning |
|-------|---------|
| 🔴 **Red / Warm regions** | Areas that contributed **most strongly** to the model's decision (tumor location) |
| 🔵 **Blue / Cool regions** | Areas that had **little influence** on the prediction (healthy tissue) |
| **Overlay** | Heatmap superimposed on original MRI showing exact tumor localization |

---

##  Dataset

The model is trained on the **Brain Tumor MRI Dataset** available on Kaggle.

### Dataset Information

| Property | Details |
|----------|---------|
| **Source** | Kaggle - Brain Tumor MRI Dataset |
| **Total Images** | ~7,000 |
| **Classes** | 4 (Glioma, Meningioma, Pituitary, No Tumor) |
| **Image Format** | JPG, JPEG, PNG |
| **Split** | Training / Testing |

### Dataset Distribution

| Class | Training Samples | Testing Samples |
|-------|-----------------|-----------------|
| Glioma | ~1,300 | ~140 |
| Meningioma | ~1,200 | ~136 |
| No Tumor | ~1,400 | ~150 |
| Pituitary | ~1,300 | ~140 |

### Dataset Link

- **Kaggle Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Citation

> Nickparvar, M. (2021). Brain Tumor MRI Dataset. Kaggle.  
> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

##  Tech Stack

| Category | Technologies |
|----------|--------------|
| **Deep Learning Framework** | PyTorch, TorchVision |
| **Model Architecture** | DenseNet169 (ImageNet pre-trained) |
| **Explainability** | Grad-CAM, OpenCV |
| **Web Application** | Streamlit |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Data Processing** | NumPy, Pandas, PIL |
| **Deployment** | Hugging Face Spaces |

---

## 📁 Project Structure
