import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNetV2
from utils.metrics import calculate_psnr_torch, calculate_ssim_torch, LPIPSCalculator
from utils.metrics_tracker import MetricsTrackerV3, ValidationImageSaverV3
from utils.sampling import compute_alpha
import torch.nn.functional as F

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def noise_estimation_loss(ddpm, x0, t, e, b, i, j, osize):
    try:
        t_device = t.to(b.device)
    except Exception:
        t_device = t

    a = compute_alpha(b, t_device.long())  # [B,1,1,1]

    # Construct noisy observation x_t for GT part only (GT channels are channels 3:)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()

    # Model may be wrapped with DataParallel
    model = ddpm.model
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float(), i, j, osize)

    # Base noise MSE (mean over all elements)
    noise_mse = (e - output).square().mean()

    # Optional reconstruction / perceptual losses
    recon_weight = 0.0
    lpips_weight = 0.0
    ssim_weight = 0.0
    try:
        cfg = getattr(ddpm, 'config', None)
        if cfg is not None and hasattr(cfg, 'training'):
            recon_weight = float(getattr(cfg.training, 'recon_loss_weight', 0.0))
            lpips_weight = float(getattr(cfg.training, 'lpips_loss_weight', 0.0))
            ssim_weight = float(getattr(cfg.training, 'ssim_loss_weight', 0.0))
    except Exception:
        pass

    if (recon_weight and recon_weight > 0.0) or (lpips_weight and lpips_weight > 0.0):
        a_sqrt = a.sqrt()
        one_minus_a_sqrt = (1.0 - a).sqrt()
        x0_pred = (x - output * one_minus_a_sqrt) / (a_sqrt + 1e-12)
        x0_gt = x0[:, 3:, :, :]

        total = noise_mse

        if recon_weight and recon_weight > 0.0:
            recon_l1 = torch.abs(x0_pred - x0_gt).mean()
            total = total + recon_weight * recon_l1

        if ssim_weight and ssim_weight > 0.0:
            try:
                ssim_val = calculate_ssim_torch(x0_pred, x0_gt)
                # calculate_ssim_torch may return a list for batch outputs
                if isinstance(ssim_val, (list, tuple, np.ndarray)):
                    ssim_val = float(np.mean(ssim_val))
                ssim_loss = 1.0 - float(ssim_val)
                total = total + ssim_weight * ssim_loss
            except Exception as e:
                print(f"SSIM loss calculation failed: {e}")

        if lpips_weight and lpips_weight > 0.0:
            try:
                from utils.metrics import calculate_lpips_tensor
                lpips_val = calculate_lpips_tensor(x0_pred, x0_gt, ddpm.lpips_calculator)
                total = total + lpips_weight * lpips_val
            except Exception as e:
                print(f"LPIPS loss calculation failed: {e}")

        return total

    return noise_mse

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_shape):
    flops = 0
    model.eval()
    with torch.no_grad():
        input = torch.randn(1, *input_shape)
        model(input)
        flops = torch.cuda.memory_stats(0)["allocated_bytes.all.current"]
    model.train()
    return flops

class DenoisingDiffusionV3(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        # Backward-compatible fallback: ensure config.device exists.
        # Some helper scripts create a SimpleNamespace from yaml and may not set
        # `device`. Prefer the first gpu in config.gpu_ids when available.
        if not hasattr(self.config, 'device') or self.config.device is None:
            try:
                if hasattr(self.config, 'gpu_ids') and isinstance(self.config.gpu_ids, (list, tuple)) and len(self.config.gpu_ids) > 0 and torch.cuda.is_available():
                    self.config.device = torch.device(f"cuda:{self.config.gpu_ids[0]}")
                else:
                    self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                # Fallback to CPU if anything unexpected happens
                self.config.device = torch.device("cpu")

        self.device = self.config.device

        # instantiate model and move to device
        self.model = DiffusionUNetV2(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        num_params = count_parameters(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        # Gradient clipping threshold (prefer config.optim, then config.training). Default 0.5 per paper.
        try:
            self.grad_clip = float(getattr(self.config.optim, 'grad_clip', getattr(self.config.training, 'grad_clip', 0.5)))
        except Exception:
            self.grad_clip = 0.5

        # LR scheduler support: optional cosine schedule with warmup (step-based)
        self.lr_scheduler = None
        try:
            sched_name = getattr(self.config.optim, 'lr_schedule', None) if hasattr(self.config, 'optim') else None
            if sched_name is None:
                sched_name = getattr(self.config, 'lr_schedule', None)

            warmup_steps = 0
            try:
                warmup_steps = int(getattr(self.config.optim, 'warmup_steps', getattr(self.config, 'warmup_steps', getattr(self.config.training, 'warmup_steps', 0))))
            except Exception:
                warmup_steps = 0

            if sched_name == 'cosine':
                from torch.optim.lr_scheduler import LambdaLR
                base_lr = self.optimizer.param_groups[0]['lr']
                total_steps = int(getattr(self.config.training, 'n_iters', getattr(self.config.training, 'n_steps', 200000)))

                def _lr_lambda(step):
                    if step < max(1, warmup_steps):
                        return float(step) / float(max(1, warmup_steps))
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    progress = max(0.0, min(1.0, progress))
                    return 0.5 * (1.0 + np.cos(np.pi * progress))

                self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
                print(f"LR scheduler: cosine with warmup_steps={warmup_steps}, total_steps={total_steps}")
        except Exception as e:
            print(f"LR scheduler init failed: {e}")

        # Initialize LPIPS calculator (can be skipped to avoid downloading weights during complexity tests)
        if bool(getattr(args, 'skip_lpips_init', False)):
            self.lpips_calculator = None
            print("Skip LPIPS initialization")
        else:
            self.lpips_calculator = LPIPSCalculator(net='alex', device=self.device)

        # Initialize metrics tracker and image saver (v3)
        tensorboard_config = getattr(self.config, 'tensorboard', None)
        if tensorboard_config and hasattr(tensorboard_config, 'log_dir'):
            tensorboard_dir = tensorboard_config.log_dir
        else:
            tensorboard_dir = './tensorboard_logs'

        evaluation_config = getattr(self.config, 'evaluation', None)
        if evaluation_config and hasattr(evaluation_config, 'metrics_dir'):
            metrics_dir = evaluation_config.metrics_dir
        else:
            metrics_dir = './metrics'

        if evaluation_config and hasattr(evaluation_config, 'validation_images_dir'):
            validation_images_dir = evaluation_config.validation_images_dir
        else:
            validation_images_dir = './results/images'
        
        # Determine resume mode if a valid checkpoint path is provided
        is_resume = bool(args.resume and os.path.isfile(args.resume))

        self.metrics_tracker = MetricsTrackerV3(metrics_dir, tensorboard_dir, resume_training=is_resume)
        self.image_saver = ValidationImageSaverV3(validation_images_dir)

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        if args.resume:
            states = torch.load(args.resume, map_location=self.device)
            # Support both dict and legacy list formats
            try:
                if isinstance(states, dict):
                    model_state = states.get('model', None)
                    if model_state is None:
                        model_state = states

                    # flexible prefix handling
                    try:
                        self.model.load_state_dict(model_state)
                    except Exception:
                        from collections import OrderedDict
                        new_state = OrderedDict()
                        own_state = self.model.state_dict()
                        for k, v in model_state.items():
                            new_key = k
                            if not k.startswith('module.') and list(own_state.keys())[0].startswith('module.'):
                                new_key = 'module.' + k
                            if k.startswith('module.') and not list(own_state.keys())[0].startswith('module.'):
                                new_key = k.replace('module.', '', 1)
                            new_state[new_key] = v
                        self.model.load_state_dict(new_state)

                    if 'optimizer' in states and self.optimizer is not None:
                        try:
                            self.optimizer.load_state_dict(states.get('optimizer'))
                        except Exception:
                            pass

                    self.start_epoch = int(states.get('epoch', 0)) if 'epoch' in states else 0
                    self.step = int(states.get('step', 0)) if 'step' in states else 0

                    if 'ema' in states and states['ema']:
                        try:
                            self.ema_helper.load_state_dict(states['ema'])
                        except Exception:
                            pass

                elif isinstance(states, (list, tuple)):
                    self.model.load_state_dict(states[0])
                    try:
                        self.optimizer.load_state_dict(states[1])
                    except Exception:
                        pass
                    self.start_epoch = int(states[2])
                    self.step = int(states[3])
                    if len(states) > 4:
                        try:
                            self.ema_helper.load_state_dict(states[4])
                        except Exception:
                            pass

                print(f"Loaded checkpoint {args.resume} (epoch {self.start_epoch}, step {self.step})")
            except Exception as e:
                print(f"Failed to load resume checkpoint {args.resume}: {e}")

    def train(self, DATASET):
        cudnn.benchmark = True
        # Dataset may split a validation set from training based on config.training.val_split
        train_loader, val_loader = DATASET.get_loaders(parse_patches=True)

        # AMP and OOM fallback configuration
        try:
            use_amp = bool(getattr(self.config.training, 'use_amp', True)) and torch.cuda.is_available()
        except Exception:
            use_amp = torch.cuda.is_available()

        oom_splits = int(getattr(self.config.training, 'oom_splits', 2))
        # Create GradScaler for AMP
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print(f"Epoch {epoch}")
            data_start = time.time()
            data_time = 0

            for inter, (x, y, i_val, j_val, osize_val) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if hasattr(x, 'ndim') and x.ndim == 5 else x
                n = x.size(0)

                # Expected: x contains 9 channels [input_3ch, enhanced_3ch, gt_3ch]
                input_img = x[:, :3, :, :]
                gt_img = x[:, 6:9, :, :]
                x = torch.cat([input_img, gt_img], dim=1)  # shape [B,6,H,W]

                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)

                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling for timesteps
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                i_val = i_val.squeeze().to(self.device)
                j_val = j_val.squeeze().to(self.device)
                osize_val = osize_val.squeeze().to(self.device)
                i_val = i_val.view(n)
                j_val = j_val.view(n)
                osize_val = osize_val.view(n)

                try:
                    with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                        loss = noise_estimation_loss(self, x, t, e, b, i_val, j_val, osize_val)

                    loss_value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
                    self.metrics_tracker.add_loss(self.step, loss_value)

                    if self.step % 10 == 0:
                        print(f"step: {self.step}, loss: {loss_value:.4f}, data time: {data_time / (inter + 1):.2f}")

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()

                    if self.grad_clip and self.grad_clip > 0.0:
                        try:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                        except Exception:
                            pass

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.ema_helper.update(self.model)
                    if self.lr_scheduler is not None:
                        try:
                            self.lr_scheduler.step()
                        except Exception:
                            pass

                except RuntimeError as e:
                    msg = str(e).lower()
                    if 'out of memory' in msg or 'cudnn' in msg:
                        print(f"CUDA OOM detected at step {self.step}: {e}. Trying fallback with {oom_splits} splits")
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                        try:
                            self.optimizer.zero_grad()
                            n_total = x.size(0)
                            splits = min(oom_splits, n_total)
                            slice_size = (n_total + splits - 1) // splits
                            for s in range(splits):
                                start = s * slice_size
                                end = min((s + 1) * slice_size, n_total)
                                if start >= end:
                                    continue
                                xs = x[start:end].contiguous()
                                ts = t[start:end].contiguous()
                                es = e[start:end].contiguous()
                                i_s = i_val[start:end].contiguous()
                                j_s = j_val[start:end].contiguous()
                                os_s = osize_val[start:end].contiguous()

                                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                                    loss_s = noise_estimation_loss(self, xs, ts, es, b, i_s, j_s, os_s)

                                frac = float(xs.size(0)) / float(n_total)
                                loss_s = loss_s * frac
                                self.scaler.scale(loss_s).backward()

                            try:
                                self.scaler.unscale_(self.optimizer)
                                if self.grad_clip and self.grad_clip > 0.0:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                            except Exception:
                                pass

                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.ema_helper.update(self.model)
                        except RuntimeError as e2:
                            print(f"OOM fallback failed: {e2}. Skipping this batch.")
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            continue
                    else:
                        raise
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.validation(val_loader, epoch)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    save_init_checkpoint = getattr(self.config.training, 'save_init_checkpoint', True)
                    if self.step == 1 and not save_init_checkpoint:
                        data_start = time.time()
                        continue

                    states = [
                        self.model.state_dict(),
                        self.optimizer.state_dict(),
                        epoch,
                        self.step,
                        self.ema_helper.state_dict(),
                    ]
                    checkpoint_path = os.path.join(self.config.training.log_path, f"checkpoint_{self.step}.pth")
                    try:
                        os.makedirs(self.config.training.log_path, exist_ok=True)
                    except Exception:
                        pass
                    torch.save(states, checkpoint_path)
                    print(f"Saved checkpoint at step {self.step}: {checkpoint_path}")
                    self.manage_checkpoints()

                data_start = time.time()

    def manage_checkpoints(self):
        try:
            keep_checkpoints = getattr(self.config.training, 'keep_checkpoints', 3)
            checkpoint_dir = self.config.training.log_path

            checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pth")
            checkpoint_files = glob.glob(checkpoint_pattern)

            if len(checkpoint_files) > keep_checkpoints:
                def extract_step(filepath):
                    try:
                        filename = os.path.basename(filepath)
                        step_str = filename.replace('checkpoint_', '').replace('.pth', '')
                        return int(step_str)
                    except Exception:
                        return 0

                checkpoint_files.sort(key=extract_step)

                files_to_delete = checkpoint_files[:-keep_checkpoints]
                total_deleted_size = 0

                for file_path in files_to_delete:
                    try:
                        file_size = os.path.getsize(file_path)
                        total_deleted_size += file_size
                        os.remove(file_path)
                        print(f"Deleted old checkpoint: {os.path.basename(file_path)} ({file_size/1024/1024:.1f}MB)")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")

                if total_deleted_size > 0:
                    print(f"Storage saved: {total_deleted_size/1024/1024:.1f}MB")
                    print(f"Keeping latest {keep_checkpoints} checkpoints")

        except Exception as e:
            print(f"Checkpoint management failed: {e}")

    def validation(self, val_loader, epoch):
        with torch.no_grad():
            print(f"Running validation at epoch: {epoch}")

            psnr_values = []
            ssim_values = []

            all_gt_images = []
            all_input_images = []
            all_output_images = []

            max_batches = None
            max_samples_per_batch = None
            save_num_images = 3
            try:
                max_batches = getattr(self.config.training, 'validation_batches', None)
            except Exception:
                max_batches = None
            try:
                max_samples_per_batch = getattr(self.config.training, 'validation_max_samples_per_batch', None)
            except Exception:
                max_samples_per_batch = None
            try:
                save_num_images = int(getattr(self.config.training, 'validation_save_images', 3))
            except Exception:
                save_num_images = 3

            for batch_idx, (x, _, ii, jj, osize) in enumerate(val_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)

                input_img = x[:, :3, :, :]
                gt_img = x[:, 6:9, :, :]
                x_combined = torch.cat([input_img, gt_img], dim=1)

                x_cond = x_combined[:, :3, :, :].to(self.device)
                x_gt = x_combined[:, 3:, :, :].to(self.device)
                x_input = x_cond.clone()

                x_cond = data_transform(x_cond)
                x_gt = data_transform(x_gt)

                osize = osize.squeeze().to(self.device).view(n)

                x_noise = torch.randn(n, 3, x_cond.shape[-2], x_cond.shape[-1], device=self.device)
                ii = ii.squeeze().view(n)
                jj = jj.squeeze().view(n)

                x_gen = self.sample_image(x_cond, x_noise, ii, jj, osize)
                x_gen = inverse_data_transform(x_gen)
                x_gt = inverse_data_transform(x_gt)
                x_input = torch.clamp(x_input, 0.0, 1.0)

                for j in range(n):
                    if (max_samples_per_batch is None) or (j < int(max_samples_per_batch)):
                        img_gen = x_gen[j] if x_gen[j].dim() == 3 else x_gen[j].squeeze(0)
                        img_gt = x_gt[j] if x_gt[j].dim() == 3 else x_gt[j].squeeze(0)

                        psnr = calculate_psnr_torch(img_gen, img_gt)
                        ssim = calculate_ssim_torch(img_gen, img_gt)
                        psnr_values.append(psnr)
                        ssim_values.append(ssim)

                        if len(all_gt_images) < save_num_images:
                            all_gt_images.append(img_gt.cpu())
                            all_input_images.append(x_input[j].cpu())
                            all_output_images.append(img_gen.cpu())

            avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
            avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
            avg_lpips = 0.0

            print(f"Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

            improved_metrics = self.metrics_tracker.update_metrics(self.step, epoch, avg_psnr, avg_ssim, avg_lpips)

            if improved_metrics and self.metrics_tracker.is_best_combined_latest():
                print("Best combined score achieved, saving best checkpoint")
                states = {
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                    'step': self.step,
                    'ema': self.ema_helper.state_dict(),
                    'best_psnr': self.metrics_tracker.best_psnr,
                    'best_ssim': self.metrics_tracker.best_ssim,
                    'best_lpips': self.metrics_tracker.best_lpips,
                    'best_combined': self.metrics_tracker.best_combined,
                    'current_psnr': avg_psnr,
                    'current_ssim': avg_ssim,
                    'current_lpips': avg_lpips,
                    'config': self.config
                }
                best_checkpoint_path = os.path.join(self.config.training.log_path, "best_model.pth")
                self.save_checkpoint_optimized(states, best_checkpoint_path)
                print(f"Best checkpoint saved to: {best_checkpoint_path}")

            if all_gt_images and all_input_images and all_output_images:
                gt_batch = torch.stack(all_gt_images)
                input_batch = torch.stack(all_input_images) 
                output_batch = torch.stack(all_output_images)

                self.image_saver.save_validation_images(
                    self.step, gt_batch, input_batch, output_batch
                )

    def sample_image(self, x, y, i, j, osize, last=True, **kwargs):
        skip = max(1, self.config.diffusion.num_diffusion_timesteps // max(1, int(getattr(self.args, 'sampling_timesteps', 50))))
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        # x: conditional input (3 channels), y: initial noise tensor (3 channels)
        # Begin sampling loop
        x_noisy = y.clone()

        with tqdm.tqdm(list(reversed(list(seq))), desc="Sampling") as pbar:
            for timestep in pbar:
                # ensure all runtime tensors are on the same device as the inputs
                device = x_noisy.device
                t = torch.tensor([timestep] * x_noisy.shape[0], device=device)

                batch_bs = x_noisy.shape[0]
                def _ensure_batch(tensor, default_val):
                    try:
                        if tensor is None:
                            return torch.full((batch_bs,), default_val, device=x.device, dtype=torch.long)
                        tt = tensor.to(x.device)
                        tt = tt.view(-1)
                        if tt.numel() == batch_bs:
                            return tt
                        else:
                            # if tensor has at least one element, repeat its first element
                            return tt[:1].repeat(batch_bs)
                    except Exception:
                        return torch.full((batch_bs,), default_val, device=x.device, dtype=torch.long)

                i_use = _ensure_batch(i, 0)
                j_use = _ensure_batch(j, 0)
                osize_use = _ensure_batch(osize, x.shape[-2])

                predicted_noise = self.model(torch.cat([x, x_noisy], dim=1), t.float(), i_use, j_use, osize_use)

                if not isinstance(predicted_noise, torch.Tensor):
                    try:
                        predicted_noise = torch.as_tensor(predicted_noise, device=x_noisy.device).float()
                    except Exception as e:
                        raise RuntimeError(f"predicted_noise has unexpected type {type(predicted_noise)} and cannot be converted") from e

                # Use compute_alpha for consistent alpha computation between training and sampling
                # compute_alpha returns shape [B,1,1,1]
                t_long = t.long()
                # ensure betas/alpha tensors live on the same device as x_noisy
                betas_device = self.betas.to(device)
                alpha_t = compute_alpha(betas_device, t_long)  # [B,1,1,1]

                # Protect lower bound for t-1 to avoid negative indices
                t_minus_1 = (t_long - 1).clamp(min=0)
                alpha_t_minus_1 = compute_alpha(betas_device, t_minus_1)

                # Compute denoised image (numerical stability: avoid division by zero)
                a_t_sqrt = torch.sqrt(torch.clamp(alpha_t, min=1e-12)).to(device)
                one_minus_a_t_sqrt = torch.sqrt(torch.clamp(1.0 - alpha_t, min=0.0)).to(device)
                # ensure predicted_noise and alpha tensors are on same device as x_noisy
                device = x_noisy.device
                try:
                    predicted_noise = predicted_noise.to(device)
                except Exception:
                    pass

                a_t_sqrt = a_t_sqrt.to(device)
                one_minus_a_t_sqrt = one_minus_a_t_sqrt.to(device)
                x_noisy = x_noisy.to(device)

                x_recon = (x_noisy - one_minus_a_t_sqrt * predicted_noise) / (a_t_sqrt + 1e-12)

                # For non-zero timesteps, add stochastic noise term
                if (t_long > 0).any():
                    noise = torch.randn_like(x_noisy)
                    # Ensure per-batch selection of self.betas broadcasts to [B,1,1,1]
                    betas_t = self.betas[t_long].view(-1, 1, 1, 1)
                    # Compute sigma_t safely (avoid division by zero and negative sqrt)
                    ratio = (torch.clamp(1.0 - alpha_t_minus_1, min=0.0) / torch.clamp(1.0 - alpha_t, min=1e-12))
                    sigma_sq = torch.clamp(ratio * betas_t, min=0.0)
                    sigma_t = torch.sqrt(sigma_sq + 1e-12)
                    x_noisy = torch.sqrt(torch.clamp(alpha_t_minus_1, min=0.0)) * x_recon + sigma_t * noise
                else:
                    x_noisy = x_recon

        return x_noisy

    def save_checkpoint_optimized(self, states, checkpoint_path):
        try:
            # Check whether to compress the checkpoint
            compress = getattr(self.config.training, 'compress_checkpoint', False)
            
            if compress:
                # Compressed save
                import gzip
                import pickle
                # Ensure parent directory exists
                try:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                except Exception:
                    pass
                with gzip.open(checkpoint_path + '.gz', 'wb') as f:
                    pickle.dump(states, f)
                print(f"✅ Compressed checkpoint saved to: {checkpoint_path}.gz")
            else:
                # Standard save
                # Ensure parent directory exists
                try:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                except Exception:
                    pass
                torch.save(states, checkpoint_path)
                print(f"✅ Checkpoint saved to: {checkpoint_path}")
                
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            # Fallback save mechanism
            try:
                torch.save(states, checkpoint_path)
                print(f"✅ Fallback save successful: {checkpoint_path}")
            except Exception as e2:
                print(f"❌ Fallback save also failed: {e2}")
