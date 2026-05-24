# CS³-Diff: Collaborative Spatio-Spectral-Scale Guided Diffusion for Low-Light Image Enhancement

# 📌 Overview

Low-light image enhancement (LLIE) aims to improve visual quality and support robust downstream vision tasks under challenging illumination conditions. In this paper, we propose CS³-Diff:, a collaborative spatio-spectral-scale guided diffusion framework for LLIE. Specifically, a Dual-stage Wavelet-based Structural Prior Guidance (DWSPG) module is introduced to enhance edge-aware structural priors in the wavelet domain. To improve spatial consistency during patch-wise inference, we further design a Global-Scale Positional Embedding (GSPE) that jointly encodes timestep, spatial, and scale information. In addition, a Residual Frequency-domain Phase Mixer (RFPM) is proposed to facilitate faithful texture reconstruction through frequency-domain phase modulation.

<img width="1045" height="823" alt="Figure2" src="https://github.com/user-attachments/assets/070f80a7-61fc-40b0-9fcb-60d6a220664d" />

---
# ⚙️ Environment Setup

```bash
- Platform: NVIDIA GPU with CUDA support (recommended)  
- Python: 3.8  
```
Install PyTorch and project dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

---

# 📂 Repository Structure
```text
├── models/        # Model implementations
├── datasets/      # Dataset loaders
├── utils/         # Utilities and metrics
├── configs/       # Configuration files
├── checkpoints/   # Pretrained models
```


# 📊 Dataset 

We evaluate CS³-Diff on the following benchmark datasets:

- LOL-v1
- LOL-v2-Real
- LOL-v2-Synthetic
- LSRW

Download the datasets and organize them under:
```bash
./datasets/scratch/LLIE
```
Directory structure:
```text
LLIE
├── LOLv1
│   ├── train
│   │   ├── input
│   │   └── gt
│   └── test
│       ├── input
│       └── gt
├── LOLv2-Real_captured
├── LOLv2-Synthetic
├── LSRW
```
Please ensure the directory structure is strictly followed.

# 📦 Pre-trained Models

Pretrained checkpoints:

Baidu Cloud: https://pan.baidu.com/s/1UuqeO4yuhZ2JhcgtHIfRMg
Extraction code: 9912

Place the downloaded files under:
```bash
./checkpoints
```

## 🧪 Inference 

Run evaluation with:
```bash
python inference.py --config configs/lowlight.yml --checkpoint <checkpoint_path>
```
Example:
```bash
python inference.py --config configs/lowlight.yml --checkpoint checkpoints/lolv2-real.pth
```
