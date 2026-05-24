import os
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import sys

# Attempt to import LPIPS.
try:
    import lpips
    LPIPS_AVAILABLE = True
    print("LPIPS backend loaded successfully")
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS backend unavailable; LPIPS computation will be skipped")

def calculate_psnr(img1, img2, test_y_channel=False):
    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=False):
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type {img_type} is not supported.')
    return img


def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type {dst_type} is not supported.')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _autocorrect_scale_if_needed(t1: torch.Tensor, t2: torch.Tensor):
    try:
        t1_min = float(torch.min(t1))
        t1_max = float(torch.max(t1))
        t2_min = float(torch.min(t2))
        t2_max = float(torch.max(t2))
    except Exception:
        return t1, t2

    if (t1_min >= -1.05 and t1_max <= 1.05) and (t2_min >= -1.05 and t2_max <= 1.05):
        t1 = (t1 + 1.0) / 2.0
        t2 = (t2 + 1.0) / 2.0
        print("Detected tensors in [-1, 1]; converting to [0, 1] for metric computation")
    return t1, t2


def calculate_psnr_torch(img1, img2):
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2)

    def _to_np_hwc(t):
        t = t.detach().cpu().numpy().astype(np.float64)
        if t.ndim == 3:
            t = np.transpose(t, (1, 2, 0))
        t = np.clip(t * 255.0, 0.0, 255.0).astype(np.float64)
        return t

    if img1.dim() == 4 and img2.dim() == 4:
        bs = img1.shape[0]
        results = []
        for i in range(bs):
            a = img1[i]
            b = img2[i]
            a, b = _autocorrect_scale_if_needed(a, b)
            results.append(calculate_psnr(_to_np_hwc(a), _to_np_hwc(b)))
        return results

    img1, img2 = _autocorrect_scale_if_needed(img1, img2)
    img1_np = _to_np_hwc(img1)
    img2_np = _to_np_hwc(img2)
    return calculate_psnr(img1_np, img2_np)


def calculate_ssim_torch(img1, img2):
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2)

    def _to_np_hwc(t):
        t = t.detach().cpu().numpy().astype(np.float64)
        if t.ndim == 3:
            t = np.transpose(t, (1, 2, 0))
        t = np.clip(t * 255.0, 0.0, 255.0).astype(np.float64)
        return t

    if img1.dim() == 4 and img2.dim() == 4:
        bs = img1.shape[0]
        results = []
        for i in range(bs):
            a = img1[i]
            b = img2[i]
            a, b = _autocorrect_scale_if_needed(a, b)
            results.append(calculate_ssim(_to_np_hwc(a), _to_np_hwc(b)))
        return results

    img1, img2 = _autocorrect_scale_if_needed(img1, img2)
    img1_np = _to_np_hwc(img1)
    img2_np = _to_np_hwc(img2)
    return calculate_ssim(img1_np, img2_np)


class LPIPSCalculator:
    def __init__(self, net='alex', device='cuda'):
        self.device = device
        self.lpips_available = LPIPS_AVAILABLE
        if self.lpips_available:
            try:
                self.lpips_fn = lpips.LPIPS(net=net).to(device)
                print(f"LPIPS calculator initialized with network: {net}")
            except Exception as e:
                print(f"LPIPS initialization failed: {e}")
                self.lpips_available = False
        else:
            self.lpips_fn = None
            print("LPIPS is unavailable; returning the default value")

    def calculate_lpips(self, img1, img2):
        if not self.lpips_available:
            return 0.0
        try:
            if not isinstance(img1, torch.Tensor):
                img1 = torch.tensor(img1, dtype=torch.float32)
            if not isinstance(img2, torch.Tensor):
                img2 = torch.tensor(img2, dtype=torch.float32)
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            if img1.dim() == 3:
                img1 = img1.unsqueeze(0)
            if img2.dim() == 3:
                img2 = img2.unsqueeze(0)
            try:
                img1_min = float(torch.min(img1))
                img1_max = float(torch.max(img1))
                img2_min = float(torch.min(img2))
                img2_max = float(torch.max(img2))
            except Exception:
                img1_min = img1_max = img2_min = img2_max = None
            if img1.dtype == torch.uint8 or (img1_min is not None and img1_max is not None and img1_max > 2.0):
                img1 = img1.float() / 255.0
            if img2.dtype == torch.uint8 or (img2_min is not None and img2_max is not None and img2_max > 2.0):
                img2 = img2.float() / 255.0
            img1_norm = img1 * 2.0 - 1.0
            img2_norm = img2 * 2.0 - 1.0
            with torch.no_grad():
                lpips_value = self.lpips_fn(img1_norm, img2_norm)
            try:
                v = lpips_value.detach().cpu().numpy()
                if hasattr(v, 'shape') and v.shape != ():
                    v_scalar = float(v.mean())
                else:
                    v_scalar = float(v)
            except Exception:
                try:
                    v_scalar = float(lpips_value.item())
                except Exception:
                    v_scalar = None
            if v_scalar is not None and v_scalar > 1.0:
                print(f"LPIPS value is unusually high: {v_scalar:.4f}; img1 range={img1_min},{img1_max}; img2 range={img2_min},{img2_max}; shapes={tuple(img1.shape)},{tuple(img2.shape)}")
            return v_scalar if v_scalar is not None else lpips_value.item()
        except Exception as e:
            print(f"LPIPS calculation failed: {e}")
            return 0.0


def calculate_lpips_torch(img1, img2, lpips_calculator=None):
    if lpips_calculator is None:
        lpips_calculator = LPIPSCalculator()
    return lpips_calculator.calculate_lpips(img1, img2)


def calculate_lpips_tensor(img1, img2, lpips_calculator=None):
    if lpips_calculator is None:
        lpips_calculator = LPIPSCalculator()
    if not lpips_calculator.lpips_available:
        return torch.tensor(0.0)
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1, dtype=torch.float32)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2, dtype=torch.float32)
    device = lpips_calculator.device
    img1 = img1.to(device)
    img2 = img2.to(device)
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    img1_norm = img1 * 2.0 - 1.0
    img2_norm = img2 * 2.0 - 1.0
    lpips_val = lpips_calculator.lpips_fn(img1_norm, img2_norm)
    return lpips_val.mean()
