# 🌿 Malabar Spinach Leaf Disease Classification

*A Self-Supervised Attention Framework with Vision Transformers*

[![Paper](https://img.shields.io/badge/Paper-Draft-blue.svg)](#)
[![Models](https://img.shields.io/badge/Models-HuggingFace-black.svg)](https://huggingface.co/saifullah03/Spinach_leaf_disease)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange.svg)](https://huggingface.co/datasets/saifullah03/malabar_spinach_leaf_disease_dataset)
[![Code](https://img.shields.io/badge/Notebooks-GitHub-green.svg)](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification)

---

## 🎥 Web Demo Video

<img src="WEB+Deployment.gif" width="1200" alt="Web Demo">

---

## 🔗 HD Preview (LinkedIn)

<p align="center">
  <a href="https://www.linkedin.com/posts/shaifullah-sharafat_a-self-supervised-deep-learning-framework-activity-7367346264226631680-4d5A" target="_blank">
    <img src="https://img.shields.io/badge/▶%20Watch%20on-LinkedIn-blue?style=for-the-badge&logo=linkedin"/>
  </a>
</p>

---

# 🔗 Resources

### 🧠 Pretrained Models (HuggingFace)

[https://huggingface.co/saifullah03/Spinach_leaf_disease](https://huggingface.co/saifullah03/Spinach_leaf_disease)

* `simsiam_cbam_pretrained_final.pth` → Self-supervised SimSiam backbone
* `best_finetuned_cbam.pth` → **Final deployment model (SimSiam-CBAM-ResNet-50, CE)**
* Additional experimental checkpoints (`*.ckpt`) available

### 🗂 Dataset (3 Classes)

[https://huggingface.co/datasets/saifullah03/malabar_spinach_leaf_disease_dataset](https://huggingface.co/datasets/saifullah03/malabar_spinach_leaf_disease_dataset)

* **Raw images:** ~700+
* **Augmented total:** ~2100+
* **Classes:** Alternaria • Straw Mite • Healthy
* Split: 70% train / 15% val / 15% test
* Collected and verified by agricultural experts (Habiganj Agricultural University)

### 📚 Code Notebooks

| Model / Experiment                                     | Notebook                                                                                                                                                                                                               |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SimSiam + CBAM (final model)**                       | [simsiam-cbam-plos-rev.ipynb](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification/blob/main/RevP1%28Latest%29/simsiam-cbam-plos-rev.ipynb) |
| EfficientNetB0, SpinachCNN, Spinach-ResSENet, ViT-B/16 | [spinachleaf.ipynb](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification/blob/main/RevP1%28Latest%29/spinachleaf.ipynb)                     |
| Vanilla SimSiam (ResNet-50)                            | [vanilla-resnet.ipynb](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification/blob/main/RevP1%28Latest%29/vanilla-resnet.ipynb)               |
| SwinV2-Base (scratch + pretrained)                     | [swin-spinach.ipynb](https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification/blob/main/RevP1%28Latest%29/swin-spinach.ipynb)                   |

---

# 🧭 Project Overview

This repository presents a **domain-optimized, interpretable, and deployment-friendly framework** for Malabar spinach leaf disease diagnosis, integrating:

### ✔ Architectures Studied

* **SpinachCNN** (lightweight baseline)
* **Spinach-ResSENet (ResNet + SE)**
* **SpinachViT (ViT-B/16)**
* **SwinV2-Base** (ImageNet-22k → 1k pretrained)
* **SimSiam-CBAM-ResNet-50** → **Final recommended model for real-time deployment**

### ✔ Two-Stage Training Pipeline

1. **Self-Supervised Pretraining (SimSiam)** on unlabeled leaf images
2. **Supervised Fine-Tuning** (Cross-Entropy or CE + SupCon)

### ✔ Explainability (XAI)

* Grad-CAM
* Grad-CAM++
* LayerCAM
  → Highlights biologically meaningful lesion regions

### ✔ Deployment

* **FastAPI backend**
* **HTML/CSS/JavaScript frontend**
* Real-time inference (<800 ms on Ryzen 5600G CPU)
* Automatic Grad-CAM visualization & disease recommendations

---

# 🧪 Final Benchmark Results

This table matches the **final manuscript**, **model-size table**, and **training notebooks**.

| Model                           | Backbone         | Pretraining         | Attention | Test Acc. (%) | TTA (%)   | ROC-AUC    | Params (M) |
| ------------------------------- | ---------------- | ------------------- | --------- | ------------- | --------- | ---------- | ---------: |
| SpinachCNN                      | Custom CNN       | None                | None      | 91.00         | 96.68     | 0.9921     |       5.49 |
| Spinach-ResSENet                | ResNet + SE      | None                | SE        | 96.01         | 95.35     | 0.9963     |       5.53 |
| SpinachViT (ViT-B/16)           | ViT-Base / 16    | None                | —         | 90.70         | 91.30     | 0.9850     |      85.50 |
| SimSiam-ResNet-50 (CE)          | ResNet-50        | SimSiam (in-domain) | None      | 91.92         | 91.92     | 0.9921     |      24.56 |
| SimSiam-ResNet-50 (Hybrid)      | ResNet-50        | SimSiam (in-domain) | None      | 94.61         | 94.61     | 0.9850     |      24.82 |
| **SimSiam-CBAM-ResNet-50 (CE)** | ResNet-50 + CBAM | SimSiam (in-domain) | **CBAM**  | **97.31**     | **97.31** | **0.9983** |  **27.08** |
| SwinV2-Base (scratch)           | SwinV2-Base      | None                | Window    | 94.95         | 95.96     | 0.9970     |      86.90 |
| **SwinV2-Base (in22k→1k)**      | SwinV2-Base      | ImageNet-22k → 1k   | Window    | **97.98**     | **98.99** | **1.0000** |      86.90 |

👉 **Key Insight:**
SwinV2-Base achieves the highest accuracy,
but **SimSiam-CBAM-ResNet-50** offers the **best model-size / latency / accuracy trade-off** for **web and edge devices**.

---

# 📦 Installation

```bash
git clone https://github.com/SAIFULLAH-SHARAFAT/A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification.git
cd A-Self-Supervised-Deep-Learning-Framework-for-Malabar-Spinach-Leaf-Disease-Classification

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

# 🚀 Quick Inference Example

```bash
python inference.py \
  --model weights/best_finetuned_cbam.pth \
  --image samples/test_leaf.jpg
```

Returns:

* Disease class
* Confidence score
* Grad-CAM heatmap

---

# 🌐 Run the Web App (FastAPI)

```bash
cd spinach_cbam_app
uvicorn app.main:app --reload
```

Open in browser:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

# 📖 Citation

```bibtex
@article{10.1371/journal.pone.0340989,
    doi = {10.1371/journal.pone.0340989},
    author = {Kabya, Nilavro Das AND Sharafat, MD Shaifullah AND Emu, Rahimul Islam AND Opee, Mehrab Karim AND Khan, Riasat},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Towards practical AI for agriculture: A self-supervised attention framework for Spinach leaf disease detection},
    year = {2026},
    month = {01},
    volume = {21},
    url = {https://doi.org/10.1371/journal.pone.0340989},
    pages = {1-30},
    abstract = {Malabar spinach is a nutrient-dense leafy vegetable widely cultivated and consumed in Bangladesh. Its productivity is often compromised by Alternaria leaf spot and straw mite infestations. This work proposes an efficient and interpretable deep learning framework for automatic Malabar spinach leaf disease classification. A curated dataset of Malabar spinach images collected from Habiganj Agricultural University and supplemented with public samples was categorized into three classes: Alternaria, straw mite, and healthy leaves. A lightweight SpinachCNN established a strong baseline, while Spinach-ResSENet, enhanced with squeeze-and-excitation modules, improved channel-wise attention and feature discrimination. A customized Vision Transformer (SpinachViT) and SwinV2-Base were further investigated to assess the benefits of transformer-based architectures under limited data. To mitigate annotation scarcity, we employed SimSiam-based self-supervised pretraining on unlabeled images, followed by supervised fine-tuning with cross-entropy or a hybrid objective combining cross-entropy and supervised contrastive loss. The best-performing domain-optimized model, SimSiam-CBAM-ResNet-50, incorporated Convolutional Block Attention Modules and achieved 97.31% test accuracy, 0.9983 macro ROC-AUC, and low calibration error, while maintaining robustness to Gaussian and salt-and-pepper noise. Although a SwinV2-Base benchmark pretrained on ImageNet-22k reached slightly higher accuracy (97.98%, 98.99% with test-time augmentation), its 86.9M parameters and reliance on large-scale pretraining reduce feasibility for edge deployment. In contrast, the SimSiam-CBAM model offers a more parameter-efficient and deployment-friendly solution for real-world agricultural applications. Model decisions are interpretable via Grad-CAM, Grad-CAM++, and LayerCAM, which consistently highlight biologically relevant lesion regions. The spinach dataset used in this study is publicly available on: https://huggingface.co/datasets/saifullah03/malabar_spinach_leaf_disease_dataset.},
    number = {1},
}
```

---
