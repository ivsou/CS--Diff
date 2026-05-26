import os
import numpy as np
import tqdm
import torch
import torch.nn as nn

from models.unet import DiffusionUNetV2
from utils.sampling import compute_alpha


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def get_beta_schedule(
    beta_schedule,
    *,
    beta_start,
    beta_end,
    num_diffusion_timesteps
):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )

    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start,
            beta_end,
            num_diffusion_timesteps,
            dtype=np.float64,
        )

    elif beta_schedule == "const":
        betas = beta_end * np.ones(
            num_diffusion_timesteps,
            dtype=np.float64,
        )

    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps,
            1,
            num_diffusion_timesteps,
            dtype=np.float64,
        )

    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start

    else:
        raise NotImplementedError(beta_schedule)

    assert betas.shape == (num_diffusion_timesteps,)

    return betas


class DiffusionSampler(object):

    def __init__(self, args, config):

        super().__init__()

        self.args = args
        self.config = config

        if not hasattr(self.config, "device") or self.config.device is None:

            if (
                hasattr(self.config, "gpu_ids")
                and isinstance(self.config.gpu_ids, (list, tuple))
                and len(self.config.gpu_ids) > 0
                and torch.cuda.is_available()
            ):
                self.config.device = torch.device(
                    f"cuda:{self.config.gpu_ids[0]}"
                )

            else:
                self.config.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

        self.device = self.config.device

        # ---------------------------------------------------------
        # Diffusion Backbone
        # ---------------------------------------------------------

        self.model = DiffusionUNetV2(config)
        self.model.to(self.device)
        self.model = nn.DataParallel(self.model)

        # ---------------------------------------------------------
        # Diffusion Schedule
        # ---------------------------------------------------------

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float().to(self.device)

        self.num_timesteps = betas.shape[0]

        # ---------------------------------------------------------
        # Load Checkpoint
        # ---------------------------------------------------------

        if hasattr(args, "resume") and args.resume:

            self.load_checkpoint(args.resume)

    def load_checkpoint(self, checkpoint_path):

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        states = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        try:

            if isinstance(states, dict):

                model_state = states.get("model", states)

            elif isinstance(states, (list, tuple)):

                model_state = states[0]

            else:
                model_state = states

            try:

                self.model.load_state_dict(model_state)

            except Exception:

                from collections import OrderedDict

                new_state = OrderedDict()

                own_state = self.model.state_dict()

                own_key = list(own_state.keys())[0]

                for k, v in model_state.items():

                    new_key = k

                    if (
                        not k.startswith("module.")
                        and own_key.startswith("module.")
                    ):
                        new_key = "module." + k

                    elif (
                        k.startswith("module.")
                        and not own_key.startswith("module.")
                    ):
                        new_key = k.replace("module.", "", 1)

                    new_state[new_key] = v

                self.model.load_state_dict(new_state)

            print(f"Loaded checkpoint: {checkpoint_path}")

        except Exception as e:

            raise RuntimeError(
                f"Failed to load checkpoint: {e}"
            )

    @torch.no_grad()
    def sample_image(
        self,
        x,
        y,
        i=None,
        j=None,
        osize=None,
        last=True,
        **kwargs
    ):

        self.model.eval()

        sampling_timesteps = int(
            getattr(self.args, "sampling_timesteps", 50)
        )

        skip = max(
            1,
            self.config.diffusion.num_diffusion_timesteps
            // max(1, sampling_timesteps),
        )

        seq = range(
            0,
            self.config.diffusion.num_diffusion_timesteps,
            skip,
        )

        x_noisy = y.clone()

        with tqdm.tqdm(
            reversed(list(seq)),
            desc="Sampling",
        ) as pbar:

            for timestep in pbar:

                device = x_noisy.device

                t = torch.tensor(
                    [timestep] * x_noisy.shape[0],
                    device=device,
                )

                batch_bs = x_noisy.shape[0]

                def _ensure_batch(tensor, default_val):

                    if tensor is None:

                        return torch.full(
                            (batch_bs,),
                            default_val,
                            device=x.device,
                            dtype=torch.long,
                        )

                    tt = tensor.to(x.device).view(-1)

                    if tt.numel() == batch_bs:
                        return tt

                    return tt[:1].repeat(batch_bs)

                i_use = _ensure_batch(i, 0)

                j_use = _ensure_batch(j, 0)

                osize_use = _ensure_batch(
                    osize,
                    x.shape[-2],
                )

                predicted_noise = self.model(
                    torch.cat([x, x_noisy], dim=1),
                    t.float(),
                    i_use,
                    j_use,
                    osize_use,
                )

                if not isinstance(predicted_noise, torch.Tensor):

                    predicted_noise = torch.as_tensor(
                        predicted_noise,
                        device=x_noisy.device,
                    ).float()

                t_long = t.long()

                betas_device = self.betas.to(device)

                alpha_t = compute_alpha(
                    betas_device,
                    t_long,
                )

                t_minus_1 = (t_long - 1).clamp(min=0)

                alpha_t_minus_1 = compute_alpha(
                    betas_device,
                    t_minus_1,
                )

                a_t_sqrt = torch.sqrt(
                    torch.clamp(alpha_t, min=1e-12)
                )

                one_minus_a_t_sqrt = torch.sqrt(
                    torch.clamp(1.0 - alpha_t, min=0.0)
                )

                predicted_noise = predicted_noise.to(device)

                x_recon = (
                    x_noisy
                    - one_minus_a_t_sqrt * predicted_noise
                ) / (a_t_sqrt + 1e-12)

                if (t_long > 0).any():

                    noise = torch.randn_like(x_noisy)

                    betas_t = self.betas[t_long].view(
                        -1,
                        1,
                        1,
                        1,
                    )

                    ratio = (
                        torch.clamp(
                            1.0 - alpha_t_minus_1,
                            min=0.0,
                        )
                        / torch.clamp(
                            1.0 - alpha_t,
                            min=1e-12,
                        )
                    )

                    sigma_sq = torch.clamp(
                        ratio * betas_t,
                        min=0.0,
                    )

                    sigma_t = torch.sqrt(
                        sigma_sq + 1e-12
                    )

                    x_noisy = (
                        torch.sqrt(
                            torch.clamp(
                                alpha_t_minus_1,
                                min=0.0,
                            )
                        )
                        * x_recon
                        + sigma_t * noise
                    )

                else:

                    x_noisy = x_recon

        return x_noisy

    @torch.no_grad()
    def enhance(
        self,
        input_tensor,
        i=None,
        j=None,
        osize=None,
    ):

        self.model.eval()

        input_tensor = input_tensor.to(self.device)

        input_tensor = data_transform(input_tensor)

        noise = torch.randn(
            input_tensor.size(0),
            3,
            input_tensor.size(2),
            input_tensor.size(3),
            device=self.device,
        )

        output = self.sample_image(
            input_tensor,
            noise,
            i,
            j,
            osize,
        )

        output = inverse_data_transform(output)

        return output
