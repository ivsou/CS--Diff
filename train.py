import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models.ddm import DenoisingDiffusionV3


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training CS³-Diff denoising diffusion model v3 with LPIPS')
    parser.add_argument("--config", type=str, required=False, default="lowlight.yml",
                        help="Path to the config file")
    parser.add_argument('--resume', default=r'', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/images_v3/', type=str,
                        help="Directory for saved validation image patches")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    # Resolve the config path without duplicating the configs/ prefix.
    config_path = args.config if args.config.startswith('configs/') else os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    
    # Resolve the device defensively in case config.gpu_ids is missing.
    if not hasattr(config, 'gpu_ids') or not getattr(config, 'gpu_ids'):
        config.gpu_ids = [0] if torch.cuda.is_available() else []
    device_str = f"cuda:{config.gpu_ids[0]}" if torch.cuda.is_available() and config.gpu_ids else "cpu"
    config.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    # Set the random seed.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to {args.seed}")

    # Print run configuration.
    print("=" * 80)
    print("CS³-Diff Training v3 (with LPIPS)")
    print("=" * 80)
    print(f"Device: {config.device}")
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume if args.resume else 'False'}")
    print("=" * 80)

    # Build the dataset using the configured loader.
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    
    # Prepare the diffusion model.
    try:
        if args.resume and args.resume != '' and hasattr(config, 'training') and hasattr(config.training, 'finetune_lr'):
            if not hasattr(config, 'optim') or config.optim is None:
                config.optim = argparse.Namespace()
            config.optim.lr = float(config.training.finetune_lr)
            print(f"Finetune mode: overriding optimizer lr to {config.optim.lr}")
    except Exception:
        pass

    diffusion = DenoisingDiffusionV3(args, config)
    
    # Start training.
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
