import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_sobel_kernel(x, kernel):
    return torch.tensor(kernel, dtype=torch.float32, device=x.device).view(1, 1, 3, 3)


def _to_grayscale(x):
    if x.shape[1] == 3:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x[:, 0:1]


def _match_spatial(x, ref):
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)


def _haar_idwt(ll, lh, hl, hh):
    batch_size, channels, height, width = ll.shape
    out = ll.new_zeros(batch_size, channels, height * 2, width * 2)
    out[:, :, 0::2, 0::2] = ll + lh + hl + hh
    out[:, :, 0::2, 1::2] = ll - lh + hl - hh
    out[:, :, 1::2, 0::2] = ll + lh - hl - hh
    out[:, :, 1::2, 1::2] = ll - lh - hl + hh
    return out


class StructureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

    def dwt_transform(self, x):
        x1 = x[:, :, 0::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 0::2]
        x4 = x[:, :, 1::2, 1::2]

        ll = (x1 + x2 + x3 + x4) / 4.0
        lh = (x1 - x2 + x3 - x4) / 4.0
        hl = (x1 + x2 - x3 - x4) / 4.0
        hh = (x1 - x2 - x3 + x4) / 4.0
        return ll, lh, hl, hh

    def extract_structure(self, x):
        gray = _to_grayscale(x)
        sobel_x = _build_sobel_kernel(x, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = _build_sobel_kernel(x, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)

        edge_max = edge.flatten(2).amax(dim=2).view(edge.shape[0], 1, 1, 1)
        edge_norm = edge / (edge_max + 1e-6)
        return edge_norm.repeat(1, self.in_channels, 1, 1)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt_transform(x)
        structure = self.extract_structure(x)
        return {
            'I_LL': ll,
            'I_HL': hl,
            'I_LH': lh,
            'I_HH': hh,
            'structure': structure,
        }


class DWSPG(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.structure_extractor = StructureExtractor(in_channels)
        self.enhancement_stage = EnhancementStage(in_channels, hidden_dim)
        self.fusion_stage = FusionStage(in_channels, hidden_dim)

    def forward(self, x):
        x_pos = (x + 1.0) * 0.5

        structure_info = self.structure_extractor(x_pos)
        enhanced_structure = self.enhancement_stage(structure_info)
        enhanced_pos = self.fusion_stage(x_pos, enhanced_structure)
        enhanced = enhanced_pos * 2.0 - 1.0
        return enhanced, structure_info


class EnhancementStage(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.max_attn_tokens = 1024

        self.k_conv = nn.Conv2d(in_channels, hidden_dim, 1)
        self.q_conv = nn.Conv2d(in_channels, hidden_dim, 1)
        self.v_conv = nn.Conv2d(in_channels, hidden_dim, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.linear1 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.linear2 = nn.Conv2d(hidden_dim, in_channels, 1)
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        )

        num_groups = min(8, in_channels) if in_channels >= 8 else 1
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.merge_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1)

    def _attention(self, k, q, v):
        q = _match_spatial(q, k)
        v = _match_spatial(v, k)

        batch_size, channels, height, width = k.shape
        token_count = height * width

        pooled = token_count > self.max_attn_tokens
        orig_size = (height, width)

        if pooled:
            pooled_side = int(self.max_attn_tokens ** 0.5)
            k = F.adaptive_avg_pool2d(k, (pooled_side, pooled_side))
            q = F.adaptive_avg_pool2d(q, (pooled_side, pooled_side))
            v = F.adaptive_avg_pool2d(v, (pooled_side, pooled_side))
            height = width = pooled_side

        tokens = height * width
        k_t = k.view(batch_size, channels, tokens).permute(0, 2, 1)
        q_t = q.view(batch_size, channels, tokens).permute(0, 2, 1)
        v_t = v.view(batch_size, channels, tokens).permute(0, 2, 1)

        attn = torch.bmm(q_t, k_t.transpose(1, 2)) / (channels ** 0.5)
        attn = self.softmax(attn)

        out = torch.bmm(attn, v_t).permute(0, 2, 1).view(batch_size, channels, height, width)
        if pooled:
            out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
        return out

    def forward(self, structure_info):
        ll = structure_info['I_LL']
        structure = structure_info['structure']

        k = self.k_conv(ll)
        q = self.q_conv(structure)
        v = self.v_conv(ll)

        attn_out = self._attention(k, q, v)
        enhanced = self.linear1(attn_out)
        enhanced = F.relu(enhanced)
        enhanced = self.linear2(enhanced)
        enhanced = enhanced + self.residual_block(enhanced)

        ll_idwt = _haar_idwt(ll, structure_info['I_LH'], structure_info['I_HL'], structure_info['I_HH'])
        ll_idwt = _match_spatial(ll_idwt, enhanced)

        fusion = torch.cat([enhanced, ll_idwt], dim=1)
        enhanced = self.fusion_conv(fusion)
        enhanced = self.norm(enhanced)
        enhanced = self.merge_conv(enhanced)
        return enhanced


class FusionStage(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.alpha = 0.1

        self.fusion_branches = nn.ModuleList([
            FusionBranch(in_channels, hidden_dim) for _ in range(3)
        ])

        self.final_merge = nn.Sequential(
            nn.Conv2d(in_channels * 4, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 3, 1, 1),
        )

        for module in self.final_merge.modules():
            if isinstance(module, nn.Conv2d):
                try:
                    module.weight.data.mul_(0.1)
                    if module.bias is not None:
                        module.bias.data.zero_()
                except Exception:
                    pass

    def forward(self, x, enhanced_structure):
        enhanced_structure = _match_spatial(enhanced_structure, x)

        branch_outputs = [branch(x, enhanced_structure) for branch in self.fusion_branches]
        all_features = torch.cat([x] + branch_outputs, dim=1)

        fused_logits = self.final_merge(all_features)
        fused_pos = torch.sigmoid(fused_logits)
        return x * (1.0 - self.alpha) + fused_pos * self.alpha


class FusionBranch(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, in_channels),
            nn.Sigmoid(),
        )
        self.output_conv = nn.Conv2d(hidden_dim, in_channels, 1)

    def forward(self, x, enhanced_structure):
        feat = self.feature_conv(torch.cat([x, enhanced_structure], dim=1))

        batch_size, channels, _, _ = feat.shape
        attention_weights = self.mlp(self.avg_pool(feat).view(batch_size, channels)).view(batch_size, -1, 1, 1)

        weighted_x = x * attention_weights
        return self.output_conv(feat) + weighted_x
