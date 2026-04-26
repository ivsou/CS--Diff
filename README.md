
**Overview**

This repository implements a diffusion-based approach for Low-light Image Enhancement (LLIE). It contains training and evaluation scripts, model implementations, dataset loaders, and utility functions. Project dependencies are listed in `requirements.txt`.

**Environment Setup**

- Recommended platform: NVIDIA GPU with CUDA support.\
- Python: 3.12

Install the primary packages and the project requirements with:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Note: Install the `torch` build that matches your CUDA version; consult the official PyTorch installation guide for the correct wheel if needed.

**Dataset: Placement and Structure**

Download the datasets and place them under the following folder in this repository:

`.\datasets\scratch\LLIE`

Organize the dataset directory exactly as shown below:

```text
LLIE
|--LOLv1
|    |--Train
|    |    |--input
|    |    |    |--100.png
|    |    |    |--101.png
|    |    |    |    ...
|    |    |--gt
|    |    |    |--100.png
|    |    |    |--101.png
|    |    |    |    ...
|    |--Test
|    |    |--input
|    |    |    |--111.png
|    |    |    |--146.png
|    |    |    |    ...
|    |    |--gt
|    |    |    |--111.png
|    |    |    |--146.png
|    |    |    |    ...
|--LOLv2
|    |--Real_captured
|    |    |--Train
|    |    |    |--input
|    |    |    |    |--00001.png
|    |    |    |    |--00002.png
|    |    |    |    |    ...
|    |    |    |--gt
|    |    |    |    |--00001.png
|    |    |    |    |--00002.png
|    |    |    |    |    ...
|    |    |--Test
|    |    |    |--input
|    |    |    |    |--00690.png
|    |    |    |    |--00691.png
|    |    |    |    |    ...
|    |    |    |--gt
|    |    |    |    |--00690.png
|    |    |    |    |--00691.png
|    |    |    |    |    ...
|    |--Synthetic
|    |    |--Train
|    |    |    |--input
|    |    |    |    |--r000da54ft.png
|    |    |    |    |--r02e1abe2t.png
|    |    |    |    |    ...
|    |    |    |--gt
|    |    |    |    |--r000da54ft.png
|    |    |    |    |--r02e1abe2t.png
|    |    |    |    |    ...
|    |    |--Test
|    |    |    |--input
|    |    |    |    |--r00816405t.png
|    |    |    |    |--r02189767t.png
|    |    |    |    |    ...
|    |    |    |--gt
+|    |    |    |    |--r00816405t.png
|    |    |    |    |--r02189767t.png
|    |    |    |    |    ...
```

**Pre-trained Checkpoints**

Download the pretrained checkpoints from the provided Baidu cloud link and extract them into the repository `checkpoints` folder:

Link: https://pan.baidu.com/s/1oOooNYFCznpJ1SC-eJ0s_g  Extraction code: `9912`

Place the downloaded files under:

`.\checkpoints`

**Training**

Run training for each dataset configuration as follows (the config file `configs/lowlight.yml` is used for all runs):

- LOLv2-Real:

```bash
python train_diffusion.py --config configs/lowlight.yml
```

- LOLv2-Syn:

```bash
python train_diffusion.py --config configs/lowlight.yml
```

- LOLv1:

```bash
python train_diffusion.py --config configs/lowlight.yml
```

- LSRW:

```bash
python train_diffusion.py --config configs/lowlight.yml
```

Note: Use additional CLI options or edit `configs/lowlight.yml` to change dataset paths, batch sizes, number of epochs, and other hyperparameters.

**Inference / Evaluation**

Run evaluation for each checkpoint using `eval_only_v3.py` and specify the corresponding checkpoint file:

- LOLv2-Real:

```bash
python eval_only_v3.py --config configs/lowlight.yml --checkpoint checkpoints/lolv2-real.pth
```

- LOLv2-Syn:

```bash
python eval_only_v3.py --config configs/lowlight.yml --checkpoint checkpoints/lolv2-syn.pth
```

- LOLv1:

```bash
python eval_only_v3.py --config configs/lowlight.yml --checkpoint checkpoints/lolv1.pth
```

- LSRW:

```bash
python eval_only_v3.py --config configs/lowlight.yml --checkpoint checkpoints/lsrw.pth
```

**Repository Layout**

- `models/` — model implementations (`unet.py`, `ddm.py`, `restoration.py`, etc.)\
- `datasets/` — dataset loaders and preprocessing (`datasets/scratch/lowlight.py`)\
- `utils/` — utilities and metrics\
- `configs/` — configuration files (e.g., `lowlight.yml`)\
- `checkpoints/` — place pretrained and export weights here

**Notes & Troubleshooting**

- If you encounter CUDA-related errors, verify your GPU drivers, CUDA toolkit installation, and that your installed `torch` matches the CUDA version.\
- Upgrade `pip` if dependency installation fails: `python -m pip install --upgrade pip`.\
- For large datasets, ensure sufficient disk space and correct file permissions.
