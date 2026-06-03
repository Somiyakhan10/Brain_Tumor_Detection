<div align="center">
    <h1>🧠 Interpretable Brain Tumor Classification System</h1>
    <h3>Deep Learning-based MRI Analysis with Explainable AI using Grad-CAM</h3>
    
<a href="https://huggingface.co/spaces/somiya-khan01/Brain_Tumor_Classification" target="_blank">
        <button style="background-color: #3b82f6; color: white; font-size: 16px; font-weight: bold; padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-family: sans-serif;">
            🚀 Launch Live Demo
        </button>
    </a>
</div>

---

## 📌 About

A **Streamlit-based Web Application** to detect and classify brain tumors from MRI scans. The Deep Learning model is trained on a publicly available dataset of ~7,000 MRI images labeled across four classes: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

The system uses **DenseNet169** architecture with transfer learning from ImageNet. Beyond classification, the application provides **explainable AI** via **Grad-CAM** (Gradient-weighted Class Activation Mapping), which highlights the exact regions in the MRI that influenced the model's decision — making it valuable for clinical interpretation and radiologist validation.

### MRI Tumor Classification
<img width="944" height="439" alt="image" src="https://github.com/user-attachments/assets/707329cb-33ec-466f-b6ec-fe9ad922204c" />

*Real-time MRI classification with confidence scores and Grad-CAM visualization*

### Model Performance Dashboard
<img width="946" height="429" alt="image" src="https://github.com/user-attachments/assets/1520bbb5-39b2-4c2d-a5dc-062612aa1474" />

*Interactive dashboard with training curves, confusion matrix, and ROC analysis*

---

## 🎥 Demo

| Feature | Demo Status |
|---------|-------------|
| MRI Upload | ✅ Available |
| Real-time Prediction | ✅ < 2 seconds |
| Grad-CAM Visualization | ✅ 3-Panel Display |
| Confidence Score | ✅ Displayed |
| Probability Distribution | ✅ Bar Chart |
| Report Download | ✅ Available |
| Grad-CAM Overlay Download | ✅ Available |

**Live Demo Link:** [https://huggingface.co/spaces/somiya-khan01/Brain_Tumor_Classification](https://huggingface.co/spaces/somiya-khan01/Brain_Tumor_Classification)

---

## 🏗️ Model Architecture

### DenseNet169

DenseNet169 was selected for this task due to its:
- **Dense connections** ensuring maximum information flow between layers
- **Vanishing gradient mitigation** through feature reuse
- **Parameter efficiency** compared to traditional CNNs
- **Strong performance** on medical imaging tasks
