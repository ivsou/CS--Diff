import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DWSPG import DWSPG


def get_timestep_embedding(timesteps, embedding_dim):
    """Create sinusoidal timestep embeddings from a 1D tensor."""
    assert len(timesteps.shape) == 1, "timesteps must be a 1D tensor"
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 != 0:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    # Choose a valid GroupNorm group count.
    ng = min(8, in_channels)
    # Fall back to 1 if no valid grouping exists.
    while ng > 1 and (in_channels % ng != 0):
        ng -= 1
    if ng < 1:
        ng = 1
    return torch.nn.GroupNorm(num_groups=ng, num_channels=in_channels, eps=1e-6, affine=True)


def LayerNorm2d(in_channels):
    """2D Layer Normalization for spatial features"""
    return torch.nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-6, affine=True)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.attention(x) * x


class RFPM(nn.Module): 
    def __init__(self, nc, expand=2):
        super().__init__()
        self.process_mag = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )
        self.process_phase = nn.Sequential(
            nn.Conv2d(nc, nc // 2, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc // 2, nc, 1, 1, 0)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        if hasattr(torch, 'fft') and hasattr(torch.fft, 'rfft2'):
            x_freq = torch.fft.rfft2(x, norm='backward')
            mag = torch.abs(x_freq)
            pha = torch.angle(x_freq)

            mag_processed = self.process_mag(mag)
            pha_processed = pha + 0.1 * self.process_phase(pha)

            real = mag_processed * torch.cos(pha_processed)
            imag = mag_processed * torch.sin(pha_processed)
            x_out = torch.complex(real, imag)
            x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
            return x_out

        x_freq_ri = torch.rfft(x, signal_ndim=2, normalized=False, onesided=True)
        real = x_freq_ri[..., 0]
        imag = x_freq_ri[..., 1]
        mag = torch.sqrt(real * real + imag * imag + 1e-12)
        pha = torch.atan2(imag, real)
        
        mag_processed = self.process_mag(mag)
        pha_processed = pha + 0.1 * self.process_phase(pha)
        
        real = mag_processed * torch.cos(pha_processed)
        imag = mag_processed * torch.sin(pha_processed)
        x_out = torch.irfft(
            torch.stack([real, imag], dim=-1),
            signal_ndim=2,
            normalized=False,
            onesided=True,
            signal_sizes=(H, W))
        return x_out


class SpAM(nn.Module): 
    def __init__(self, channels, dw_expand=2):
        super().__init__()
        self.dw_channel = dw_expand * channels
        
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, self.dw_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, 1, 1, 0, bias=True)
        self.conv3 = nn.Conv2d(self.dw_channel, self.dw_channel, 3, 1, 1, bias=True)
        self.simple_gate = SimpleGate()
        self.channel_attention = SimplifiedChannelAttention(self.dw_channel // 2)
        self.conv_out = nn.Conv2d(self.dw_channel // 2, channels, 1, 1, 0, bias=True)
        
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.simple_gate(x)
        x = self.channel_attention(x)
        x = self.conv_out(x)
        return residual + self.beta * x


class DiSpAM(nn.Module):
    def __init__(self, channels, dw_expand=2, dilations=[1, 4, 9]):
        super().__init__()
        self.dw_channel = dw_expand * channels
        self.dilations = dilations
        
        self.norm = LayerNorm2d(channels)
        self.conv_expand = nn.Conv2d(channels, self.dw_channel, 1, 1, 0, bias=True)
        self.conv_pre = nn.Conv2d(self.dw_channel, self.dw_channel, 3, 1, 1, bias=True)
        
        self.dilated_branches = nn.ModuleList()
        for dilation in dilations:
            self.dilated_branches.append(
                nn.Conv2d(self.dw_channel, self.dw_channel, 3, 1, dilation, 
                         groups=self.dw_channel, bias=True, dilation=dilation)
            )
        
        self.simple_gate = SimpleGate()
        self.channel_attention = SimplifiedChannelAttention(self.dw_channel // 2)
        self.conv_out = nn.Conv2d(self.dw_channel // 2, channels, 1, 1, 0, bias=True)
        
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv_expand(x)
        x = self.conv_pre(x)
        
        branch_outputs = []
        for branch in self.dilated_branches:
            branch_outputs.append(branch(x))
        
        x = sum(branch_outputs)
        x = self.simple_gate(x)
        x = self.channel_attention(x)
        x = self.conv_out(x)
        return residual + self.beta * x


class GatedFFN(nn.Module):
    def __init__(self, channels, expand=2):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, expand * channels, 1, 1, 0, bias=True)
        self.simple_gate = SimpleGate()
        self.conv2 = nn.Conv2d(expand * channels // 2, channels, 1, 1, 0, bias=True)
        
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.simple_gate(x)
        x = self.conv2(x)
        return residual + self.gamma * x


class SFEblock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_attention = SpAM(channels)
        self.freq_mlp = RFPM(channels, expand=2)

        self.fusion_weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x_spatial = self.spatial_attention(x)
        x_freq = self.freq_mlp(x_spatial)
        return x_spatial + self.fusion_weight * x_freq


class MGDBlock(nn.Module):
    def __init__(self, channels, dilations=[1, 4, 9]):
        super().__init__()
        self.multi_scale_attention = DiSpAM(channels, dilations=dilations)
        self.gated_ffn = GatedFFN(channels)

    def forward(self, x):
        x = self.multi_scale_attention(x)
        x = self.gated_ffn(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlockV2(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, 
                 dropout, temb_channels=512, use_enhanced_blocks=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.use_conv_shortcut = conv_shortcut
        self.use_enhanced_blocks = use_enhanced_blocks

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = Normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if use_enhanced_blocks:
            self.enhanced_block = SFEblock(self.out_channels)
            self.conv2 = None
        else:
            self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.enhanced_block = None
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        temb_proj = self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = h + temb_proj
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        
        if self.use_enhanced_blocks and self.enhanced_block is not None:
            h = self.enhanced_block(h)
        else:
            h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        
        return x + h


class REABlock(nn.Module):
    """Residual Attention (REA) block: ResnetBlock -> DiSpAM -> ResnetBlock.

    Implements the REA block used in the bottleneck (residual-attention-residual).
    Forward signature: (x, temb)
    """
    def __init__(self, channels, temb_channels=512, dropout=0.0):
        super().__init__()
        self.res1 = ResnetBlockV2(
            in_channels=channels,
            out_channels=channels,
            temb_channels=temb_channels,
            dropout=dropout,
            use_enhanced_blocks=False,
        )
        self.attn = DiSpAM(channels)
        self.res2 = ResnetBlockV2(
            in_channels=channels,
            out_channels=channels,
            temb_channels=temb_channels,
            dropout=dropout,
            use_enhanced_blocks=False,
        )

    def forward(self, x, temb):
        h = self.res1(x, temb)
        h = self.attn(h)
        h = self.res2(h, temb)
        return h


class DiffusionUNetV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.ch_mult = ch_mult
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.dwspg = DWSPG(in_channels=config.model.in_channels, hidden_dim=ch // 2)

        adjusted_in_channels = in_channels

        self.temb = nn.Sequential(
            nn.Linear(self.ch, self.temb_ch),
            nn.SiLU(),
            nn.Linear(self.temb_ch, self.temb_ch),
        )

        self.conv_in = nn.Conv2d(adjusted_in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        for i_level in range(self.num_resolutions):
            in_ch = ch * in_ch_mult[i_level]
            out_ch = ch * ch_mult[i_level]
            block = nn.ModuleList([
                ResnetBlockV2(
                    in_channels=in_ch if i_block == 0 else out_ch,
                    out_channels=out_ch,
                    temb_channels=self.temb_ch,
                    dropout=dropout,
                    use_enhanced_blocks=True
                )
                for i_block in range(num_res_blocks)
            ])
            down = nn.Module()
            down.block = block
            if i_level < self.num_resolutions - 1:
                down.downsample = Downsample(out_ch, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = REABlock(ch * ch_mult[-1], temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            current_ch = ch * ch_mult[i_level]
            
            for i_block in range(num_res_blocks + 1):
                if i_block == 0:
                    if i_level == self.num_resolutions - 1:
                        input_ch = ch * ch_mult[-1]
                    else:
                        prev_ch = ch * ch_mult[i_level + 1]
                        skip_ch = ch * ch_mult[i_level]
                        input_ch = prev_ch + skip_ch
                    output_ch = current_ch
                else:
                    input_ch = current_ch
                    output_ch = current_ch
                
                use_enhanced = i_block == num_res_blocks and i_level < self.num_resolutions - 1
                
                if use_enhanced:
                    resnet_block = ResnetBlockV2(
                        in_channels=input_ch,
                        out_channels=output_ch,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        use_enhanced_blocks=False
                    )
                    enhanced_block = MGDBlock(output_ch)
                    block.append(nn.Sequential(resnet_block, enhanced_block))
                else:
                    block.append(
                        ResnetBlockV2(
                            in_channels=input_ch,
                            out_channels=output_ch,
                            temb_channels=self.temb_ch,
                            dropout=dropout,
                            use_enhanced_blocks=False
                        )
                    )
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsample(current_ch, resamp_with_conv)
            self.up.insert(0, up)

        final_ch = ch * ch_mult[0] if len(ch_mult) > 0 else ch
        self.norm_out = Normalize(final_ch)
        self.conv_out = nn.Conv2d(final_ch, config.model.out_ch, kernel_size=3, stride=1, padding=1)
        self.output_bias = nn.Parameter(torch.zeros(1, config.model.out_ch, 1, 1), requires_grad=True)

    def forward(self, x, t, i, j, osize):
        assert x.shape[2] == x.shape[3], "Input must be square"
        
        if isinstance(osize, (int, float)):
            osize = torch.tensor([osize] * x.shape[0], device=x.device, dtype=torch.long)
        elif osize.dim() == 0:
            osize = osize.unsqueeze(0).repeat(x.shape[0])
        elif osize.dim() > 1:
            osize = osize.flatten()
        
        if len(osize) == 1 and x.shape[0] > 1:
            osize = osize.repeat(x.shape[0])
        elif len(osize) != x.shape[0]:
            osize = torch.full((x.shape[0],), osize[0], device=x.device, dtype=torch.long)

        try:
            max_t = int(getattr(self.config.diffusion, 'num_diffusion_timesteps', 1000) - 1)
        except Exception:
            max_t = 999

        osize_f = osize.float()
        osize_f = osize_f.clone()
        osize_f[osize_f <= 0] = float(self.resolution)

        i = i.to(osize_f.device).view(-1).float()
        j = j.to(osize_f.device).view(-1).float()
        osize_f = osize_f.to(osize_f.device)

        i_scaled = (i / osize_f) * float(max_t)
        j_scaled = (j / osize_f) * float(max_t)
        osize_scaled = (osize_f / float(self.resolution)) * float(max_t)

        t = t.to(i_scaled.device).view(-1)
        temb1 = get_timestep_embedding(t, self.ch // 4)
        temb2 = get_timestep_embedding(i_scaled, self.ch // 4)
        temb3 = get_timestep_embedding(j_scaled, self.ch // 4)
        temb4 = get_timestep_embedding(osize_scaled, self.ch // 4)

        if self.config.data.conditional:
            lowlight_img = x[:, :self.config.model.in_channels]
            condition_input = x[:, self.config.model.in_channels:]
            enhanced_lowlight, structure_info = self.dwspg(lowlight_img)
            x_enhanced = torch.cat([enhanced_lowlight, condition_input], dim=1)
        else:
            x_enhanced, structure_info = self.dwspg(x)
        temb = torch.cat([temb1, temb2, temb3, temb4], dim=1)
        temb = self.temb(temb)
        
        hs = [self.conv_in(x_enhanced)]
        skip_connections = []
        
        for i_level in range(self.num_resolutions):
            for i_block, block in enumerate(self.down[i_level].block):
                hs.append(block(hs[-1], temb))
            
            if i_level < self.num_resolutions - 1:
                skip_connections.append(hs[-1])
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        h = hs[-1]
        h = self.mid(h, temb)
        
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(len(self.up[i_level].block)):
                if i_block == 0:
                    if i_level == self.num_resolutions - 1:
                        pass
                    else:
                        skip_idx = i_level
                        skip = skip_connections[skip_idx]
                        
                        if skip.shape[-2:] != h.shape[-2:]:
                            skip = F.interpolate(skip, size=h.shape[-2:], mode='bilinear', align_corners=False)
                        
                        h = torch.cat([h, skip], dim=1)
                
                block = self.up[i_level].block[i_block]
                if isinstance(block, nn.Sequential):
                                            h = block[0](h, temb)
                                            h = block[1](h)
                else:
                    h = block(h, temb)
            
            if i_level > 0:
                h = self.up[i_level].upsample(h)
        
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = h + self.output_bias

        return h
