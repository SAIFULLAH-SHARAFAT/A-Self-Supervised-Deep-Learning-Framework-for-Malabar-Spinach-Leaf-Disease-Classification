# 🌿 Malabar Spinach Leaf Disease Classification
A Self-Supervised Deep Learning Framework for Malabar Spinach Leaf Disease Classification using Attention and Vision Transformers.  

This repository contains the code, datasets, and experimental results supporting the paper:  
**"A Self-Supervised Deep Learning Framework for Malabar Spinach Leaf Disease Classification Using Attention and Vision Transformers"**  
*(Das Kabya et al., North South University, 2025)*  

---

## 📖 Project Overview
Malabar spinach (Basella alba) is a nutrient-rich vegetable widely consumed in Bangladesh, but its yield is often compromised by **Alternaria leaf spot** and **straw mite infestations**.  

This project introduces the **first self-supervised and attention-based pipeline** for classifying Malabar spinach diseases, leveraging:
- **Custom CNNs (SpinachCNN, Spinach-ResSENet)**  
- **Vision Transformers (SpinachViT, SwinV2)**  
- **Self-Supervised Pretraining with SimSiam**  
- **Attention-enhanced ResNet (CBAM-ResNet-50)**  
- **Hybrid Loss Functions (Cross-Entropy + Supervised Contrastive Loss)**  

Our models are designed to be **lightweight, interpretable, and edge-deployable**, with strong performance on a curated spinach dataset.

---

## 📊 Key Results
| Model | Backbone | Attention | Pretraining | Test Accuracy (%) | Macro ROC-AUC | Params (M) |
|-------|----------|-----------|-------------|------------------|---------------|------------|
| SpinachCNN | Custom CNN | None | None | 91.00 | 0.992 | 5.49 |
| Spinach-ResSENet | ResNet + SE | SE | None | 96.01 | 0.996 | 5.53 |
| SpinachViT | ViT-Small | — | None | 90.7 | 0.985 | 85.5 |
| SimSiam-ResNet-50 | ResNet-50 | None | SimSiam | 94.95 | 0.998 | 23.5 |
| **SimSiam-CBAM-ResNet-50** | ResNet-50 + CBAM | Yes | SimSiam | **96.97** | **0.9982** | 23.6 |
| SwinV2-Small | Swin Transformer | Windowed | ImageNet-21k | **97.98** | **1.0000** | 28.0 |

> 🔑 Takeaway: SwinV2 reached the highest accuracy but is too heavy for real-world farming. The **SimSiam-CBAM-ResNet-50** is the most practical domain-optimized solution.  

---

## 🗂️ Dataset
- **Total samples:** 2100 images  
- **Classes:**  
  - Healthy (39.9%)  
  - Straw mite (31.9%)  
  - Alternaria leaf spot (28.1%)  
- Collected from **Habiganj Agricultural University** and supplemented with public datasets.  

👉 Dataset will be released **after paper acceptance**.  

---

## ⚙️ Methods
- **Data Augmentation:** flips, rotations, color jitter, Gaussian noise, salt-and-pepper noise  
- **Models:** SpinachCNN, Spinach-ResSENet, SpinachViT, Swin Transformer, SimSiam-CBAM-ResNet-50  
- **Training:**  
  - Image size: `224x224`  
  - Optimizers: AdamW / SGD + cosine annealing  
  - Loss: CE + Supervised Contrastive Loss  
- **Evaluation Metrics:** Accuracy, F1-score, ROC-AUC, Calibration Error  

---

## 🚀 Getting Started
### 1. Clone repo
```bash
git clone https://github.com/<your-username>/Malabar-Spinach-Leaf-Disease.git
cd Malabar-Spinach-Leaf-Disease
