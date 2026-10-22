# MobileViT-XAI: A Lightweight Hybrid Architecture with Dual-Level Fused Explainable AI for Brain Tumor Classification in MRI Scans

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end neuro-oncological diagnostic system integrating a lightweight **MobileViT** hybrid architecture with a novel **Dual-Level Fused Explainable AI (XAI)** framework[cite: 6]. By combining early convolutional layers for fine local edge detection with deep Vision Transformer blocks for global brain cavity context, the model achieves high classification accuracy with minimal computational overhead[cite: 6, 7]. Grad-CAM heatmaps from multi-depth layers are mathematically fused to produce precise, stable, and clinically verifiable tumor boundary visualizations deployed via an interactive Streamlit application[cite: 6, 7, 8].

---

## Author & Affiliation

- **Harsh Soni** ([harsh.soni20@yahoo.com](mailto:harsh.soni20@yahoo.com)) – *Department of Computer Science, Netaji Subhas University of Technology, Dwarka, New Delhi, India*[cite: 6]

---

## Key Features

- **Hybrid MobileViT Architecture:** Combines Inverted Residual (MobileNetV2) blocks for local texture extraction with Multi-Head Self-Attention (MHSA) transformer blocks for global anatomical context[cite: 6, 8].
- **Ultra-Lightweight Footprint:** Requires only **5.6M parameters** (~22 MB model size), achieving real-time inference on consumer-grade hardware (Intel i5 CPU / entry GPUs)[cite: 7, 9, 10].
- **Dual-Level Fused Explainable AI (XAI):** Mathematically combines gradient activations from both shallow CNN layers and deep Transformer blocks ($\beta = 0.5$), eliminating coarse, unstable attributions common in single-layer Grad-CAM[cite: 7, 8, 10].
- **Clinical-Grade Decision Support:** Peak **96.71% validation accuracy**, **0.96 macro F1-score**, and **>0.98 macro ROC-AUC** across four MRI classes[cite: 6, 7, 9, 10].
- **Zero-Code Streamlit Web Interface:** Interactive clinical dashboard allowing physicians to drag and drop MRI scans, perform inference, and inspect multi-stage XAI heatmaps instantly[cite: 7, 10].

---

## Benchmark Results

### 1. Classification Performance (4-Class MRI)

Evaluated across **Glioma**, **Meningioma**, **Pituitary Adenoma**, and **No Tumor** on a 3,500-image dataset[cite: 6, 8]:

| Category | Precision | Recall | F1-Score | ROC-AUC | Diagnostic Tier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pituitary Adenoma** | 0.94[cite: 10] | 1.00[cite: 10] | 0.97[cite: 10] | 0.999[cite: 10] | Near-Perfect[cite: 10] |
| **No Tumor (Healthy)**[cite: 8] | 0.89[cite: 10] | 1.00[cite: 10] | 0.94[cite: 10] | 0.995[cite: 10] | Exceptional[cite: 10] |
| **Meningioma**[cite: 8] | 0.93[cite: 10] | 0.93[cite: 10] | 0.93[cite: 10] | 0.991[cite: 10] | High-Confidence[cite: 10] |
| **Glioma**[cite: 8] | 0.99[cite: 10] | 0.80[cite: 10] | 0.89[cite: 10] | 0.973[cite: 10] | Excellent[cite: 10] |
| **Overall / Weighted** | **0.96**[cite: 9] | **0.93**[cite: 10] | **0.96**[cite: 9] | **>0.98**[cite: 7, 10] | **Clinical Grade**[cite: 10] |

### 2. Efficiency Comparison with Baseline Models

| Model Architecture | Parameter Count | Checkpoint Size | Accuracy | Explainability (XAI) |
| :--- | :---: | :---: | :---: | :---: |
| VGG-16[cite: 10] | 138.0M[cite: 10] | ~528 MB[cite: 10] | 87.20%[cite: 10] | No[cite: 10] |
| ResNet-50[cite: 10] | 25.6M[cite: 10] | ~98 MB[cite: 10] | 92.50%[cite: 10] | No[cite: 10] |
| ViT (Base)[cite: 10] | 86.0M[cite: 10] | ~330 MB[cite: 10] | 94.10%[cite: 10] | No[cite: 10] |
| **MobileViT-XAI (Ours)**[cite: 10] | **5.6M**[cite: 10] | **~22 MB**[cite: 10] | **96.71%**[cite: 10] | **Yes (Dual Fused)**[cite: 7, 10] |

---

## Dual-Level Fused XAI Pipeline

Traditional explainable AI models generate heatmaps exclusively from the final convolutional layer, producing blurry and unstable attributions[cite: 7]. Our framework fuses multi-depth activations[cite: 7]:

$$\mathcal{L}_{\text{Fused}} = \text{ReLU}\left(\beta \sum_{k} w_{k}^{\text{cnn}} A_{k}^{\text{cnn}} + (1 - \beta) \sum_{k} w_{k}^{\text{tr}} A_{k}^{\text{tr}}\right)$$
[cite: 8]

1. **Stage 1 (Raw MRI):** Input preprocessed $224 \times 224$ axial MRI scan[cite: 8, 10].
2. **Stage 2 (Early CNN Block):** Captures high-resolution local textural edges and fine tumor margins[cite: 10].
3. **Stage 3 (Deep Transformer Block):** Captures receptive global attention maps across the entire cranial cavity[cite: 7, 10].
4. **Stage 4 (Fused Output):** Weighted mathematical blend producing a high-resolution, stable diagnostic hotspot centered on the pathology[cite: 8, 10].

---

## Repository Structure

```text
├── dataset/
│   ├── Testing/            # Test split (glioma, meningioma, notumor, pituitary)
│   └── Training/           # Train split (glioma, meningioma, notumor, pituitary)
├── H&S_files/
│   ├── Code_work/          # Model architecture, training scripts, & Streamlit app
│   └── h&s_report.pdf      # Complete academic research paper
├── .gitignore              # Ignores large raw MRI scans and model weights
├── requirements.txt        # Python package dependencies
└── README.md
