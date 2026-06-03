# 🧠 Interpretable Brain Tumor Classification System

**Deep Learning-based MRI Analysis with Explainable AI using Grad-CAM**

[![Streamlit App](https://img.shields.io/badge/🚀-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)
[![GitHub](https://img.shields.io/badge/📁-GitHub_Repository-181717?style=for-the-badge&logo=github)](https://github.com/YOUR_USERNAME/Brain-Tumor-Classification)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

---

## 📌 About

A **Streamlit-based Web Application** to detect and classify brain tumors from MRI scans. The Deep Learning model is trained on a publicly available dataset of ~7,000 MRI images labeled across four classes: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

The system uses **DenseNet169** architecture with transfer learning from ImageNet. Beyond classification, the application provides **explainable AI** via **Grad-CAM** (Gradient-weighted Class Activation Mapping), which highlights the exact regions in the MRI that influenced the model's decision — making it valuable for clinical interpretation and radiologist validation.

---

## 🎥 Demo

![Demo Animation](images/demo.gif)

*Click the Live Demo badge above to try the application*

| Feature | Demo Status |
|---------|-------------|
| MRI Upload | ✅ Available |
| Real-time Prediction | ✅ < 2 seconds |
| Grad-CAM Visualization | ✅ 3-Panel Display |
| Report Download | ✅ Available |

**Live Demo Link:** [https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)

---

## 🏗️ Model Architecture

### DenseNet169

DenseNet169 was selected for this task due to its:
- **Dense connections** ensuring maximum information flow between layers
- **Vanishing gradient mitigation** through feature reuse
- **Parameter efficiency** compared to traditional CNNs
- **Strong performance** on medical imaging tasks
