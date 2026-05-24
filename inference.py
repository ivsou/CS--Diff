import argparse
import os
import yaml
import torch
import tqdm

import datasets
from models.ddm import DenoisingDiffusionV3, data_transform, inverse_data_transform
from utils.metrics import calculate_psnr_torch, calculate_ssim_torch, LPIPSCalculator


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Evaluation entrypoint for DarkIR-Enhanced SEM Diffusion (v3)")
    parser.add_argument("--config", type=str, default="configs/lowlight.yml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to best checkpoint")
    parser.add_argument("--sampling_timesteps", type=int, default=25, help="Sampling steps for DDPM sampler")
    parser.add_argument("--save_dir", type=str, default="results/images_v3/final_eval", help="Dir to save final eval images")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation")
    args = parser.parse_args()

    # Load config
    cfg_path = args.config if args.config.startswith("configs/") else os.path.join("configs", args.config)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # Attach device
    if not hasattr(config, 'gpu_ids') or not getattr(config, 'gpu_ids'):
        config.gpu_ids = [0] if torch.cuda.is_available() else []
    device_str = f"cuda:{config.gpu_ids[0]}" if torch.cuda.is_available() and config.gpu_ids else "cpu"
    config.device = torch.device(device_str if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    # Mimic training args minimal interface for DenoisingDiffusionV3
    ddpm_args = argparse.Namespace(
        sampling_timesteps=args.sampling_timesteps,
        resume=args.checkpoint,
        image_folder=config.evaluation.validation_images_dir if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'validation_images_dir') else args.save_dir,
        seed=61
    )

    return args, config, ddpm_args


@torch.no_grad()
def run_final_evaluation(diffusion: DenoisingDiffusionV3, config, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    DATASET = datasets.__dict__[config.data.dataset](config)
    val_split = getattr(config.training, 'val_split', 0) if hasattr(config, 'training') else 0
    if val_split and float(val_split) > 0:
        import os as _os
        import importlib as _importlib
        test_dir = _os.path.join(config.data.data_dir, 'data', config.data.dataset, 'test')
        ds_mod = _importlib.import_module(f"datasets.{config.data.dataset}")
        test_dataset = ds_mod.lowlightDataset(dir=test_dir,
                                              n=config.training.patch_n,
                                              patch_size=config.data.image_size,
                                              transforms=DATASET.transforms,
                                              filelist='lowlighttesta.txt',
                                              parse_patches=True,
                                              train=False)
        import torch.utils.data as _data
        test_loader = _data.DataLoader(test_dataset, batch_size=config.sampling.batch_size,
                                       shuffle=False, num_workers=config.data.num_workers,
                                       pin_memory=True)
        eval_loader = test_loader
    else:
        train_loader, val_loader = DATASET.get_loaders(parse_patches=True)
        eval_loader = val_loader

    lpips_calc = LPIPSCalculator(net='alex', device=config.device)

    psnr_values, ssim_values, lpips_values = [], [], []

    for batch_idx, (x, _, ii, jj, osize) in enumerate(tqdm.tqdm(eval_loader, desc="Final Eval")):
        x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
        n = x.size(0)

        input_img = x[:, :3, :, :].to(config.device)
        gt_img = x[:, 6:9, :, :].to(config.device)
        x_cond = data_transform(input_img)
        x_gt = data_transform(gt_img)

        image_size = input_img.shape[-1]
        x_noise = torch.randn(n, 3, image_size, image_size, device=config.device)
        ii = ii.squeeze().view(n).to(config.device)
        jj = jj.squeeze().view(n).to(config.device)
        osize = osize.squeeze().view(n).to(config.device)

        x_gen = diffusion.sample_image(x_cond, x_noise, ii, jj, osize)

        x_gen = inverse_data_transform(x_gen)
        x_gt = inverse_data_transform(x_gt)

        try:
            gen_min = float(x_gen.min())
            gen_max = float(x_gen.max())
            gen_mean = float(x_gen.mean())
            gt_min = float(x_gt.min())
            gt_max = float(x_gt.max())
            gt_mean = float(x_gt.mean())
            print(f"Batch {batch_idx}: generated range {gen_min:.4f}..{gen_max:.4f}, mean={gen_mean:.4f}; ground-truth range {gt_min:.4f}..{gt_max:.4f}, mean={gt_mean:.4f}")

            if gen_min < -0.05 or gen_max > 1.05:
                print(f"Generated batch is outside the expected [0, 1] range (min={gen_min:.4f}, max={gen_max:.4f}); clipping to [0, 1].")
                x_gen = torch.clamp(x_gen, 0.0, 1.0)
            if gt_min < -0.05 or gt_max > 1.05:
                print(f"Ground-truth batch is outside the expected [0, 1] range (min={gt_min:.4f}, max={gt_max:.4f}); clipping to [0, 1].")
                x_gt = torch.clamp(x_gt, 0.0, 1.0)
        except Exception as e:
            print(f"Batch diagnostics failed: {e}")

        for j in range(n):
            img_gen = x_gen[j]
            img_gt = x_gt[j]

            psnr = calculate_psnr_torch(img_gen, img_gt)
            ssim = calculate_ssim_torch(img_gen, img_gt)
            lpips = lpips_calc.calculate_lpips(img_gen, img_gt)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips)

    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0.0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
    avg_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else 0.0

    # Save final metrics to CS³-Diff/results/final_metrics.txt
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "final_metrics.txt")
    with open(report_path, "w") as f:
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"LPIPS: {avg_lpips:.4f}\n")
    print(f"Final metrics saved to {report_path}")


def main():
    args, config, ddpm_args = parse_args_and_config()

    diffusion = DenoisingDiffusionV3(ddpm_args, config)

    ckpt = None
    try:
        ckpt = torch.load(ddpm_args.resume, map_location=config.device)
    except Exception as e:
        print(f"Failed to load checkpoint {ddpm_args.resume}: {e}")

    if ckpt is not None:
        try:
            if 'config' in ckpt and ckpt['config'] is not None:
                ck_cfg = ckpt['config']
                ck_timesteps = None
                cur_timesteps = None
                try:
                    ck_timesteps = getattr(ck_cfg.diffusion, 'num_diffusion_timesteps', None)
                except Exception:
                    try:
                        ck_timesteps = ck_cfg.get('diffusion', {}).get('num_diffusion_timesteps', None)
                    except Exception:
                        ck_timesteps = None
                try:
                    cur_timesteps = getattr(config.diffusion, 'num_diffusion_timesteps', None)
                except Exception:
                    cur_timesteps = None
                if ck_timesteps is not None and cur_timesteps is not None and ck_timesteps != cur_timesteps:
                    print(f"Checkpoint diffusion.num_diffusion_timesteps={ck_timesteps} does not match the current config value {cur_timesteps}")
        except Exception:
            pass

        try:
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model_state = ckpt['model']
            else:
                model_state = ckpt if not isinstance(ckpt, (list, tuple)) else ckpt[0]

            try:
                diffusion.model.load_state_dict(model_state)
            except Exception:
                from collections import OrderedDict
                new_state = OrderedDict()
                own_state = diffusion.model.state_dict()
                own_keys = list(own_state.keys())
                for k, v in model_state.items():
                    new_key = k
                    if not k.startswith('module.') and own_keys and own_keys[0].startswith('module.'):
                        new_key = 'module.' + k
                    if k.startswith('module.') and own_keys and not own_keys[0].startswith('module.'):
                        new_key = k.replace('module.', '', 1)
                    new_state[new_key] = v
                diffusion.model.load_state_dict(new_state)

            if isinstance(ckpt, dict) and 'ema' in ckpt and ckpt['ema']:
                try:
                    diffusion.ema_helper.load_state_dict(ckpt['ema'])
                    diffusion.ema_helper.ema(diffusion.model)
                    print("Applied EMA weights for evaluation")
                except Exception as e:
                    print(f"Failed to apply EMA weights: {e}")
        except Exception as e:
            print(f"Failed to load model from checkpoint: {e}")

    diffusion.model.eval()

    run_final_evaluation(diffusion, config, args.save_dir)


if __name__ == "__main__":
    main()
