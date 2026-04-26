# CS³-Diff: Collaborative Spatio-Spectral-Scale Guided Diffusion for Low-Light Image Enhancement

## Overview

This repository provides the official implementation of **CS³-Diff**, a diffusion-based framework for low-light image enhancement (LLIE). The project includes training and evaluation scripts, mode[...]

All required dependencies are listed in `requirements.txt`.

---

## Environment Setup

- Platform: NVIDIA GPU with CUDA support (recommended)  
- Python: 3.12  

Install PyTorch and project dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## Dataset Preparation

Download the datasets and organize them under:
```bash
./datasets/scratch/LLIE
```
Directory structure:
```text
LLIE
├── LOLv1
│   ├── Train
│   │   ├── input
│   │   └── gt
│   └── Test
│       ├── input
│       └── gt
├── LOLv2
│   ├── Real_captured
│   │   ├── Train
│   │   └── Test
│   └── Synthetic
│       ├── Train
│       └── Test

```
Please ensure the directory structure is strictly followed.

## Pre-trained Models

Pretrained checkpoints:

Baidu Cloud: https://pan.baidu.com/s/1oOooNYFCznpJ1SC-eJ0s_g
Extraction code: 9912

Place the downloaded files under:
```bash
./checkpoints
```
## Training

Run training with:

```bash
python train_diffusion.py --config configs/lowlight.yml
```
You can modify the following settings in configs/lowlight.yml:

Dataset paths
Batch size
Number of epochs
Other hyperparameters


## Inference / Evaluation

Run evaluation with:
```bash
python eval_only_v3.py --config configs/lowlight.yml --checkpoint <checkpoint_path>
```
Example:
```bash
python eval_only_v3.py --config configs/lowlight.yml --checkpoint checkpoints/lolv2-real.pth
```
## Repository Structure
```text
├── models/        # Model implementations
├── datasets/      # Dataset loaders
├── utils/         # Utilities and metrics
├── configs/       # Configuration files
├── checkpoints/   # Pretrained models
```
## Notes
Ensure CUDA and GPU drivers are correctly installed.
Upgrade pip if needed:
python -m pip install --upgrade pip
Ensure sufficient storage for large datasets.
Code Availability

The complete implementation is currently under preparation and will be fully released upon paper acceptance.
