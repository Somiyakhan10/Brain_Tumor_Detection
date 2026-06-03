# 🧠 Interpretable Brain Tumor Classification System

**Deep Learning-based MRI Analysis with Explainable AI using Grad-CAM**

[![Streamlit App](https://img.shields.io/badge/🚀-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)
[![GitHub](https://img.shields.io/badge/📁-GitHub_Repository-181717?style=for-the-badge&logo=github)](https://github.com/YOUR_USERNAME/YOUR_REPO)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

---

## 📌 Live Demo

> **Try the application live on Hugging Face Spaces**

[![Watch Demo Video]](https://youtu.com/YOUR_VIDEO_LINK_HERE)

*Click the image above to watch a full walkthrough of the application*

**Live Demo Link:** [https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)

| Feature | Status |
|---------|--------|
| MRI Upload | ✅ Supported |
| Real-time Inference | ✅ < 2 seconds |
| Grad-CAM Visualization | ✅ 3-Panel Display |
| Report Download | ✅ Available |

---

## 📖 Overview

This project implements an **end-to-end interpretable deep learning system** for automated brain tumor classification from MRI scans. The system not only predicts tumor type with high accuracy but also **explains its decision** using Grad-CAM (Gradient-weighted Class Activation Mapping), highlighting the exact regions in the MRI that influenced the prediction.

The model is built on **DenseNet169** architecture with transfer learning from ImageNet, fine-tuned on a publicly available brain tumor MRI dataset of ~7,000 images across four classes.

### Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multi-Class Classification** | Glioma, Meningioma, Pituitary Tumor, No Tumor |
| 🤖 **DenseNet169 Backbone** | Pre-trained on ImageNet, fine-tuned for medical imaging |
| 👁 **Grad-CAM Explainability** | 3-panel visualization showing tumor localization |
| 📊 **Comprehensive Analytics** | Training curves, confusion matrix, ROC curves |
| 📥 **Report Export** | Download prediction reports and Grad-CAM overlays |
| 🎨 **Professional Dark Mode UI** | Clean, clinical-grade interface |

---

## 🏗️ System Architecture
