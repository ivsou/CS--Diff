import os
import json
from typing import Dict, List, Tuple
import torch

try:
    import distutils.version
except AttributeError:
    import distutils
    import distutils.util
    try:
        from packaging import version
        distutils.version = version
    except ImportError:
        pass

try:
    from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
    TENSORBOARD_AVAILABLE = True
    print("TensorBoard backend loaded: torch.utils.tensorboard")
except ImportError as e:
    print(f"TensorBoard backend load failed: torch.utils.tensorboard ({e})")
    try:
        from tensorboardX import SummaryWriter as TorchSummaryWriter
        TENSORBOARD_AVAILABLE = True
        print("TensorBoard backend loaded: tensorboardX")
    except ImportError as e2:
        print(f"TensorBoard backend load failed: tensorboardX ({e2})")
        TENSORBOARD_AVAILABLE = False

class DummySummaryWriter:
    def __init__(self, *args, **kwargs):
        self.log_dir = args[0] if args else kwargs.get('log_dir', 'tensorboard_logs')
        print(f"TensorBoard is unavailable; logs will be written to {self.log_dir}")
    def add_scalar(self, tag, scalar_value, global_step=None):
        pass
    def add_image(self, tag, img_tensor, global_step=None):
        pass
    def close(self):
        pass

class MetricsTrackerV3:
    def __init__(self, metrics_dir: str, tensorboard_dir: str, resume_training: bool = False):
        self.metrics_dir = metrics_dir
        self.tensorboard_dir = tensorboard_dir
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.metrics_file = os.path.join(metrics_dir, "training_metrics.txt")
        self.checkpoint_log = os.path.join(metrics_dir, "checkpoint_log.txt")
        self.best_psnr = -float('inf')
        self.best_ssim = -float('inf')
        self.best_lpips = float('inf')
        self.best_combined = -float('inf')
        if TENSORBOARD_AVAILABLE:
            self.writer = TorchSummaryWriter(tensorboard_dir)
            print(f"TensorBoard logging enabled: {tensorboard_dir}")
        else:
            self.writer = DummySummaryWriter(tensorboard_dir)
            print("TensorBoard is unavailable; using a no-op writer")
        if resume_training and os.path.exists(self.metrics_file):
            self._load_existing_metrics()
            print("MetricsTrackerV3 initialized in resume mode")
        else:
            with open(self.metrics_file, 'w') as f:
                header = f"{'Step':<8}{'Epoch':<8}{'PSNR':<10}{'SSIM':<10}{'LPIPS':<10}{'Combined':<12}{'Best_PSNR':<12}{'Best_SSIM':<12}{'Best_LPIPS':<12}{'Best_Combined':<15}{'Notes':<30}\n"
                f.write(header)
            print("MetricsTrackerV3 initialized with a fresh state")
        if not os.path.exists(self.checkpoint_log):
            with open(self.checkpoint_log, 'w') as f:
                f.write("Checkpoint Log v3 - Training Checkpoints\n")
                f.write("=" * 50 + "\n")

    def _load_existing_metrics(self):
        try:
            with open(self.metrics_file, 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:
                if line.strip() and not line.startswith('Step'):
                    parts = line.strip().split()
                    if len(parts) >= 10:
                        try:
                            best_psnr = float(parts[6])
                            best_ssim = float(parts[7])
                            best_lpips = float(parts[8])
                            best_combined = float(parts[9])
                            self.best_psnr = max(self.best_psnr, best_psnr)
                            self.best_ssim = max(self.best_ssim, best_ssim)
                            self.best_lpips = min(self.best_lpips, best_lpips)
                            self.best_combined = max(self.best_combined, best_combined)
                        except (ValueError, IndexError):
                            continue
            print("Loaded existing metrics history")
        except Exception as e:
            print(f"Failed to load existing metrics: {e}")
            print("Initializing with fresh metrics")

    def update_metrics(self, step: int, epoch: int, psnr: float, ssim: float, lpips: float):
        combined_score = psnr * 2 + ssim * 100 - lpips * 10
        is_best_psnr = psnr > self.best_psnr
        is_best_ssim = ssim > self.best_ssim
        is_best_lpips = lpips < self.best_lpips
        is_best_combined = combined_score > self.best_combined
        if is_best_psnr:
            self.best_psnr = psnr
        if is_best_ssim:
            self.best_ssim = ssim
        if is_best_lpips:
            self.best_lpips = lpips
        if is_best_combined:
            self.best_combined = combined_score
        self._last_combined_improved = is_best_combined
        notes = []
        if is_best_psnr:
            notes.append("✓PSNR")
        if is_best_ssim:
            notes.append("✓SSIM")
        if is_best_lpips:
            notes.append("✓LPIPS")
        if is_best_combined:
            notes.append("✓Combined")
        notes_str = " ".join(notes) if notes else ""
        with open(self.metrics_file, 'a') as f:
            line = f"{step:<8}{epoch:<8}{psnr:<10.4f}{ssim:<10.4f}{lpips:<10.4f}{combined_score:<12.4f}{self.best_psnr:<12.4f}{self.best_ssim:<12.4f}{self.best_lpips:<12.4f}{self.best_combined:<15.4f}{notes_str:<30}\n"
            f.write(line)
        try:
            self.writer.add_scalar('Metrics/PSNR', psnr, step)
            self.writer.add_scalar('Metrics/SSIM', ssim, step)
            self.writer.add_scalar('Metrics/LPIPS', lpips, step)
            self.writer.add_scalar('Metrics/Combined', combined_score, step)
            self.writer.add_scalar('Best/PSNR', self.best_psnr, step)
            self.writer.add_scalar('Best/SSIM', self.best_ssim, step)
            self.writer.add_scalar('Best/LPIPS', self.best_lpips, step)
            self.writer.add_scalar('Best/Combined', self.best_combined, step)
        except Exception as e:
            print(f"TensorBoard logging failed: {e}")
        print(f"Step {step} | PSNR: {psnr:.4f} | SSIM: {ssim:.4f} | LPIPS: {lpips:.4f} | Combined: {combined_score:.4f}")
        if notes:
            print(f"New best metrics: {notes_str}")
        return is_best_psnr or is_best_ssim or is_best_lpips or is_best_combined

    def is_best_combined_latest(self):
        return hasattr(self, '_last_combined_improved') and self._last_combined_improved

    def add_loss(self, step: int, loss: float):
        try:
            self.writer.add_scalar('Training/Loss', loss, step)
        except Exception as e:
            print(f"TensorBoard loss logging failed: {e}")

    def log_checkpoint(self, step: int, checkpoint_path: str, psnr: float, ssim: float, lpips: float):
        with open(self.checkpoint_log, 'a') as f:
            f.write(f"Step {step}: {checkpoint_path}\n")
            f.write(f"  - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}\n")
            f.write(f"  - Time: {os.path.basename(checkpoint_path)}\n\n")

    def save_validation_images(self, step: int, gt_images: torch.Tensor, 
                              input_images: torch.Tensor, output_images: torch.Tensor):
        try:
            if gt_images.shape[0] > 0:
                gt0 = gt_images[0].detach().cpu()
                inp0 = input_images[0].detach().cpu()
                out0 = output_images[0].detach().cpu()
                gt0 = torch.clamp(gt0, 0.0, 1.0)
                inp0 = torch.clamp(inp0, 0.0, 1.0)
                out0 = torch.clamp(out0, 0.0, 1.0)
                try:
                    self.writer.add_image('Validation/GT', gt0, step, dataformats='CHW')
                    self.writer.add_image('Validation/Input', inp0, step, dataformats='CHW')
                    self.writer.add_image('Validation/Output', out0, step, dataformats='CHW')
                    comparison = torch.cat([inp0, out0, gt0], dim=2)
                    self.writer.add_image('Validation/Comparison', comparison, step, dataformats='CHW')
                except TypeError:
                    self.writer.add_image('Validation/GT', gt0, step)
                    self.writer.add_image('Validation/Input', inp0, step)
                    self.writer.add_image('Validation/Output', out0, step)
                    comparison = torch.cat([inp0, out0, gt0], dim=2)
                    self.writer.add_image('Validation/Comparison', comparison, step)
        except Exception as e:
            print(f"TensorBoard image logging failed: {e}")

    def close(self):
        try:
            self.writer.close()
        except:
            pass


class ValidationImageSaverV3:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    def save_validation_images(self, step: int, gt_images: torch.Tensor, 
                              input_images: torch.Tensor, output_images: torch.Tensor):
        import torchvision.utils as vutils
        step_dir = os.path.join(self.save_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)
        batch_size = gt_images.shape[0]
        for i in range(batch_size):
            gt_img = gt_images[i]
            input_img = input_images[i]
            output_img = output_images[i]
            gt_img = torch.clamp(gt_img, 0, 1)
            input_img = torch.clamp(input_img, 0, 1)
            output_img = torch.clamp(output_img, 0, 1)
            sample_prefix = f"sample_{i:02d}"
            vutils.save_image(gt_img, os.path.join(step_dir, f"{sample_prefix}_gt.png"))
            vutils.save_image(input_img, os.path.join(step_dir, f"{sample_prefix}_input.png"))
            vutils.save_image(output_img, os.path.join(step_dir, f"{sample_prefix}_output.png"))
        print(f"Saved {batch_size} validation image sets to {step_dir}")
        print("Each set contains GT, input, and output images")
