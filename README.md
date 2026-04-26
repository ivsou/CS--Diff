# CS³-Diff: Collaborative Spatio-Spectral-Scale Guided Diffusion for Low-Light Image Enhancement

## Overview

This repository provides the official implementation of **CS³-Diff**, a diffusion-based framework for low-light image enhancement (LLIE). The project includes training and evaluation scripts, model implementations, dataset loaders, and utility functions.

All required dependencies are listed in `requirements.txt`.

---

## Environment Setup

- Platform: NVIDIA GPU with CUDA support (recommended)  
- Python: 3.12  

Install PyTorch and project dependencies:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt

## Dataset Preparation

Download the datasets and organize them under the following directory:

./datasets/scratch/LLIE

The directory structure should follow:

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

Please ensure that file names and folder hierarchy strictly follow the above format.

Pre-trained Models

Pretrained checkpoints can be downloaded from:

Baidu Cloud: https://pan.baidu.com/s/1oOooNYFCznpJ1SC-eJ0s_g
Extraction code: 9912

After downloading, place the files under:

./checkpoints
Training

Run training with:

python train_diffusion.py --config configs/lowlight.yml

You may modify the following settings in configs/lowlight.yml:

Dataset paths
Batch size
Number of epochs
Other hyperparameters
Inference / Evaluation

Run evaluation with:

python eval_only_v3.py --config configs/lowlight.yml --checkpoint <checkpoint_path>

Example:

python eval_only_v3.py --config configs/lowlight.yml --checkpoint checkpoints/lolv2-real.pth
Repository Structure
├── models/        # Model implementations
├── datasets/      # Dataset loaders and preprocessing
├── utils/         # Utilities and evaluation metrics
├── configs/       # Configuration files
├── checkpoints/   # Pretrained models
Notes
Ensure CUDA and GPU drivers are correctly installed.

If dependency installation fails, upgrade pip:

python -m pip install --upgrade pip
Large datasets may require sufficient storage and proper file permissions.
Code Availability

The complete implementation is currently under preparation and will be fully released upon paper acceptance.
