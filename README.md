# 🌿 Malabar Spinach Leaf Disease Classification

*A Self-Supervised Attention Framework with Vision Transformers*

[![Paper](https://img.shields.io/badge/Paper-Draft-blue.svg)](#)
[![HuggingFace Models](https://img.shields.io/badge/Models-HuggingFace-black.svg)](https://huggingface.co/saifullah03/Spinach_leaf_disease/tree/main)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange.svg)](https://huggingface.co/datasets/saifullah03/Spinach/tree/main)
[![Code](https://img.shields.io/badge/Notebook-GitHub-green.svg)](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification/blob/main/spinach-Vresnet%2CSwin%2CCBAM.ipynb)

---

## Web Demo Video
<img src="WEB+Deployment.gif" width="1200" alt="Web Demo">
---
## HD Preview:
<p align="center">
  <a href="https://www.linkedin.com/posts/shaifullah-sharafat_a-self-supervised-deep-learning-framework-activity-7367346264226631680-4d5A" target="_blank">
    <img src="assets/linkedin-button.png" alt="Watch on LinkedIn" width="200"/>
  </a>
</p>

## 🔗 Resources

* **Pretrained Models (Hugging Face):**
  [Spinach Leaf Disease Models](https://huggingface.co/saifullah03/Spinach_leaf_disease/tree/main)

  * `simsiam_cbam_pretrained_final.pth` → Self-supervised backbone (SimSiam only).
  * `best_finetuned_cbam.pth` → ✅ Fine-tuned CBAM classifier (**use this for deployment**).
  * Additional checkpoints (`*.ckpt`) available for reproducibility.

* **Dataset (3 Classes: Alternaria, Straw mite, Healthy):**
  [Spinach Dataset (Hugging Face)](https://huggingface.co/datasets/saifullah03/Spinach/tree/main)

* **Web Application (Frontend + Backend):**
  [Spinach CBAM App](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification/tree/main/spinach_cbam_app)

---

## 🧭 Project Overview

This project proposes a **lightweight and interpretable pipeline** for **spinach leaf disease detection**:

* **Architectures:**

  * Custom CNN (*SpinachCNN*)
  * ResNet + SE (*Spinach-ResSENet*)
  * Vision Transformers (ViT, SwinV2)
  * **SimSiam-CBAM-ResNet-50 (main deployment model)**

* **Training Strategy:**

  * Self-supervised pretraining (SimSiam) → supervised fine-tuning (Cross-Entropy + Supervised Contrastive).

* **Explainability (XAI):**

  * Grad-CAM, Grad-CAM++, LayerCAM heatmaps show lesion-focused regions.

* **Deployment:**

  * FastAPI backend + simple HTML/JS frontend for real-time leaf upload, prediction, confidence score, and Grad-CAM visualization.

---

## 🧪 Benchmarks

| Model                      | Backbone         | Pretraining  | Attention | Test Acc. (%) | TTA Acc. (%) | Macro ROC-AUC | Params (M) |
| -------------------------- | ---------------- | ------------ | --------- | ------------- | ------------ | ------------- | ---------: |
| SpinachCNN                 | Custom CNN       | None         | None      | 91.00         | 96.68        | 0.992         |       5.49 |
| Spinach-ResSENet           | ResNet + SE      | None         | SE        | 96.01         | 95.35        | 0.996         |       5.53 |
| SpinachViT                 | ViT-Small        | None         | —         | 90.70         | 91.30        | 0.985         |       85.5 |
| SimSiam-ResNet-50          | ResNet-50        | SimSiam      | None      | 94.95         | 95.00        | 0.9984        |       23.5 |
| **SimSiam-CBAM-ResNet-50** | ResNet-50 + CBAM | SimSiam      | **CBAM**  | **96.97**     | **97.00**    | **0.9982**    |       23.6 |
| **SwinV2-Small (Hybrid)**  | SwinV2-Small     | ImageNet-21k | Windowed  | **97.98**     | **98.99**    | **1.0000**    |       28.0 |

👉 **Note:** SwinV2 achieves the highest accuracy, but SimSiam-CBAM-ResNet-50 provides the best trade-off for web/edge deployment.

---

## 📦 Installation

```bash
git clone https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification.git
cd A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification

# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🚀 Quickstart

Run inference on a test image:

```bash
python inference.py \
  --model weights/best_finetuned_cbam.pth \
  --image samples/test_leaf.jpg
```

Expected output: predicted class, confidence score, Grad-CAM visualization.

---

## 🌐 Web Deployment

The FastAPI app enables farmers/researchers to upload images and get:

* Predicted class (Healthy / Alternaria / Straw mite)
* Confidence score
* Grad-CAM heatmaps
* Disease-specific management advice

```bash
cd spinach_cbam_app
uvicorn app.main:app --reload
```

Then open: [http://127.0.0.1:8000](http://127.0.0.1:8000)


## 📖 Citation

If you use this work, please cite:

```bibtex
@article{kabya2025spinach,
  title={Towards Practical AI for Agriculture: A Self-Supervised Attention Framework for Spinach Leaf Disease Detection},
  author={Kabya, Nilavro Das and Sharafat, MD Shaifullah and Emu, Rahimul Islam and Opee, Mehrab Karim and Khan, Riasat},
  journal={},
  year={2025}
}
```

Do you want me to also make a **short “Quick Research Summary” section** (like a one-paragraph abstract) for people who won’t read the paper but still land on your repo?
