import os
import torch
import torchvision
from torchvision.transforms import Resize
import utils


def data_transform(x):
    return 2 * x - 1.0


def inverse_data_transform(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion
        self.ema_model = None

        if getattr(args, 'resume', None) and os.path.isfile(args.resume):
            self._load_resume_checkpoint(args.resume)
        else:
            print('Pre-trained diffusion model path is missing or not provided; proceeding without checkpoint')

    def _load_resume_checkpoint(self, resume_path):
        try:
            self.diffusion.load_ddm_ckpt(resume_path, ema=True)
        except Exception:
            pass

        try:
            self.diffusion.model.eval()
        except Exception:
            pass

        try:
            ema_helper = getattr(self.diffusion, 'ema_helper', None)
            if ema_helper is not None and getattr(ema_helper, 'shadow', None):
                self.ema_model = ema_helper.ema_copy(self.diffusion.model)
                self.ema_model.eval()
        except Exception:
            self.ema_model = None

    def _get_device(self):
        return getattr(self.diffusion, 'device', None) or torch.device('cpu')

    def _get_sampler(self):
        candidates = [self.ema_model, self.diffusion]
        for candidate in candidates:
            if candidate is None:
                continue
            if hasattr(candidate, 'sample_image'):
                return candidate
            if hasattr(candidate, 'module') and hasattr(candidate.module, 'sample_image'):
                return candidate.module
        return self.diffusion

    def _ensure_condition_range(self, cond_input, prefix='[restoration]'):
        try:
            from utils.sampling import data_transform as _data_transform

            try:
                mn = float(cond_input.min())
                mx = float(cond_input.max())
            except Exception:
                mn, mx = None, None

            if mn is not None and mn >= -0.05 and mx <= 1.05:
                return _data_transform(cond_input)

            if mn is not None and (mn < -1.2 or mx > 1.2):
                print(f'{prefix} WARNING: cond_input range unexpected: min={mn}, max={mx}. Clipping to [-1,1].')
                return torch.clamp(cond_input, -1.0, 1.0)
            return cond_input
        except Exception:
            return 2.0 * cond_input - 1.0

    def restore(self, val_loader, validation='lowlight', r=None, use_align=False):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)

        with torch.no_grad():
            for _, (x, y, wd, ht) in enumerate(val_loader):
                try:
                    src_name = os.path.splitext(os.path.basename(y[0]))[0]
                except Exception:
                    src_name = str(y)

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :6, :, :].to(self._get_device())

                x_output = self.diffusive_restoration(x_cond, r=r, fullresolusion=False)
                x_output = inverse_data_transform(x_output)

                if x_output.shape[0] > 1:
                    x_output = x_output[:1]

                ht_val = int(ht.item()) if hasattr(ht, 'item') else int(ht)
                wd_val = int(wd.item()) if hasattr(wd, 'item') else int(wd)
                x_output = Resize([ht_val, wd_val])(x_output)

                if use_align:
                    target = x[:, 6:, :, :]
                    gt_mean = torch.mean(target)
                    sr_mean = torch.mean(x_output)
                    if sr_mean != 0:
                        x_output = x_output * (gt_mean / (sr_mean + 1e-12))

                out_path = os.path.join(image_folder, f'{src_name}.png')
                try:
                    print(
                        f"[restoration] saving {out_path}: tensor.shape={tuple(x_output.shape)}, "
                        f"min={float(x_output.min()):.6f}, max={float(x_output.max()):.6f}, "
                        f"mean={float(x_output.mean()):.6f}"
                    )
                except Exception:
                    pass

                utils.logging.save_image(x_output, out_path)

    def diffusive_restoration(self, x_cond, r=None, fullresolusion=False):
        device = self._get_device()
        sampler = self._get_sampler()

        if not fullresolusion:
            patch_size = 64
            corners = self._build_patch_corners(x_cond, patch_size, r=8)
            corners1 = self._build_patch_corners(x_cond, 96, r=8)
            corners2 = self._build_patch_corners(x_cond, 128, r=8)

            x = torch.randn(x_cond.size(0), 3, x_cond.size(2), x_cond.size(3), device=device)
            ii = torch.tensor([item[0] for item in corners], dtype=torch.long, device=device)
            jj = torch.tensor([item[1] for item in corners], dtype=torch.long, device=device)
            osize = torch.full((len(corners),), patch_size, dtype=torch.long, device=device)

            cond_input = self._ensure_condition_range(x_cond[:, :3, :, :].to(device))

            try:
                return sampler.sample_image(
                    cond_input,
                    x,
                    ii,
                    jj,
                    osize,
                    patch_locs=corners,
                    patch_size=patch_size,
                    patch_locs1=corners1,
                    patch_locs2=corners2,
                )
            except TypeError:
                return sampler.sample_image(cond_input, x, ii, jj, osize)

        x = torch.randn(x_cond.size(0), 3, x_cond.size(2), x_cond.size(3), device=device)
        ii = torch.tensor(-1, device=device).unsqueeze(0)
        jj = torch.tensor(-1, device=device).unsqueeze(0)
        osize = torch.tensor(x_cond.size(2), device=device).unsqueeze(0)

        cond_input = self._ensure_condition_range(x_cond[:, :3, :, :].to(device), prefix='[restoration:fullres]')
        return sampler.sample_image(cond_input, x, ii, jj, osize)

    def _build_patch_corners(self, x_cond, output_size, r=None):
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=output_size, r=r)
        return [(i, j) for i in h_list for j in w_list]

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, _, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
