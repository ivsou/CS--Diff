import torch
import shutil
import os
import torchvision.utils as tvu

"""Utility helpers for saving images and checkpoints."""


def save_image(img, file_directory):
    """Save an image tensor to disk, creating parent directories if needed."""
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename):
    """Save a checkpoint tensor dictionary using the project's file convention."""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    """Load a checkpoint on the requested device, or on the default device."""
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
