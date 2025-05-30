# DAUNet: A Lightweight UNet Variant with Deformable Convolutions and Parameter-Free Attention for Medical Image Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official PyTorch implementation of **DAUNet**, a lightweight and effective medical image segmentation model. DAUNet extends the traditional UNet by incorporating **Deformable V2 Convolutions** for spatial adaptability and **SimAM** (a parameter-free attention module) to enhance feature refinement — all while keeping the model computationally efficient.

---

## 🔬 Overview

Medical image segmentation is vital for diagnosis, treatment planning, and monitoring. DAUNet is designed to handle:

- Anatomical variability
- Low-contrast imaging (e.g., ultrasound, CT angiography)
- Incomplete context due to occlusion or artifacts

Our model shows superior performance in segmenting complex anatomical structures using fewer parameters compared to other state-of-the-art models.

---

## 📌 Key Features

- ✅ Deformable V2 Convolution-based bottleneck
- ✅ SimAM (Parameter-Free Attention) in decoder and skip connections
- ✅ Robust performance under missing context
- ✅ Fewer parameters than traditional UNet variants
- ✅ Supports 2D medical imaging tasks

---

## 📁 Datasets

We evaluate DAUNet on:

1. **FH-PS-AoP**: Fetal head and pubic symphysis segmentation from transperineal ultrasound  
   → Dataset URL: *[Will be added by user]*

2. **FUMPE**: Pulmonary embolism segmentation from CT angiography  
   → Dataset URL: [https://www.severinalab.com/FUMPE](https://www.severinalab.com/FUMPE)

---

## 🛠 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/DAUNet.git
   cd DAUNet
