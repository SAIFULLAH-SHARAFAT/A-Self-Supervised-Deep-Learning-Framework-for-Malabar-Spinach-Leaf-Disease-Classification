# 🌿 Malabar Spinach Leaf Disease Classification
*A Self‑Supervised Deep Learning Framework using Attention and Vision Transformers*

[![Paper](https://img.shields.io/badge/Paper-Draft-blue.svg)](#)
[![HuggingFace](https://img.shields.io/badge/Weights-HuggingFace-black.svg)](https://huggingface.co/saifullah03/SpinachCBAMResNet50)
[![Code](https://img.shields.io/badge/Notebook-GitHub-green.svg)](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification/blob/main/spinach-Vresnet%2CSwin%2CCBAM.ipynb)

This repository hosts code and results for the paper:

> **“A Self‑Supervised Deep Learning Framework for Malabar Spinach Leaf Disease Classification Using Attention and Vision Transformers.”**  
> Nilavro Das Kabya, MD Shaifullah Sharafat, Rahimul Islam Emu, Mehrab Karim Opee, Riasat Khan (North South University)

The project targets **three classes** on Malabar spinach leaves: **Alternaria**, **Straw mite**, and **Healthy**. We combine **self‑supervised SimSiam pretraining**, **CBAM attention**, and **hybrid losses** to achieve high accuracy with **edge‑friendly** models.

---

## 🔗 Resources

- **Trained Weights (Hugging Face):**  
  https://huggingface.co/saifullah03/SpinachCBAMResNet50  
  - `simsiam_cbam_pretrained_final.pth` → **self‑supervised backbone only** (for further fine‑tuning)  
  - `best_finetuned_cbam.pth` → **final CBAM classifier** (✅ **use this for deployment**)  

- **Notebook (Swin, ResNet, CBAM + Grad-CAM):** [GitHub Link](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification/blob/main/spinach-Vresnet%2CSwin%2CCBAM.ipynb)

---

## 🧭 Project Overview

- **Architectures:** SpinachCNN (custom), Spinach‑ResSENet (SE), Vision Transformers (ViT, SwinV2), and **SimSiam‑CBAM‑ResNet‑50**.
- **Training strategy:** Self‑supervised pretraining (SimSiam) → supervised fine‑tuning (CE and CE+SupCon).
- **Explainability:** Grad‑CAM, Grad‑CAM++, LayerCAM for lesion‑focused heatmaps.
- **Edge focus:** Competitive accuracy with fewer parameters for realistic deployment.

---

## 🧪 Benchmarks (Single‑Crop Malabar Spinach)

| Model                              | Backbone           | Pretraining       | Attention | Test Acc. (%) | Macro ROC‑AUC | Params (M) |
|-----------------------------------|--------------------|-------------------|-----------|---------------|---------------|-----------:|
| SpinachCNN                        | Custom CNN         | None              | None      | 91.00         | 0.992         | 5.49       |
| Spinach‑ResSENet                  | ResNet + SE        | None              | SE        | 96.01         | 0.996         | 5.53       |
| SpinachViT                        | ViT‑Small          | None              | —         | 90.70         | 0.985         | 85.5       |
| SimSiam‑ResNet‑50                 | ResNet‑50          | SimSiam           | None      | 94.95         | 0.9984        | 23.5       |
| **SimSiam‑CBAM‑ResNet‑50**        | ResNet‑50 + CBAM   | SimSiam           | **CBAM**  | **96.97**     | **0.9982**    | 23.6       |
| **SwinV2‑Small (Hybrid)**         | SwinV2‑Small       | ImageNet‑21k      | Windowed  | **97.98**     | **1.0000**    | 28.0       |

> **Deployment note:** SwinV2‑Small is most accurate but heavier. **SimSiam‑CBAM‑ResNet‑50** offers the best trade‑off for web/edge.

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
