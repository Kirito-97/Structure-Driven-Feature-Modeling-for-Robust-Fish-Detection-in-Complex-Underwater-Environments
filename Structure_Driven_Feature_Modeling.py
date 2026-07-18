import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# 1. Structure-aware Channel Attention
# ===============================

class StructureChannelAttention(nn.Module):
    """
    Channel attention focusing on structure-sensitive channels
    (avg + max pooling, no median)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = channels // reduction

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        # learnable modulation strength
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        maxv = F.adaptive_max_pool2d(x, 1)

        att = self.mlp(avg + maxv)
        att = torch.sigmoid(att)

        # residual-style channel reweighting (alpha is learnable)
        return x * (1.0 + self.alpha * att)


# ===============================
# 2 Multi-scale Structure Difference
# ===============================

class MultiScaleStructureDifference(nn.Module):
    """
    Multi-scale structure difference extraction (3x3, 5x5, 7x7 blur).
    """
    def __init__(self, channels):
        super().__init__()
        self.blur3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.blur5 = nn.AvgPool2d(5, stride=1, padding=2)
        self.blur7 = nn.AvgPool2d(7, stride=1, padding=3)

    def forward(self, x):
        diff3 = torch.abs(x - self.blur3(x))
        diff5 = torch.abs(x - self.blur5(x))
        diff7 = torch.abs(x - self.blur7(x))
        return diff3, diff5, diff7


def norm_struct(s):
    # normalize the structure map by its spatial mean
    return s / (s.mean(dim=(2, 3), keepdim=True) + 1e-6)


normalize_struct = norm_struct  # alias for MultiScaleStructureSpatialAttention


# ===============================
# 3 Multi-scale Structure Spatial Attention
# ===============================

class MultiScaleStructureSpatialAttention(nn.Module):
    """
    Spatial attention driven by multi-scale structure differences
    """
    def __init__(self, channels):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False) # the last layer is Conv2d(channels, 1, ...), output only 1 channel, shape is (B, 1, H, W)
        )

        self.beta = nn.Parameter(torch.tensor(0.7))

    def forward(self, feat, struct_list):
        # normalize and fuse multi-scale structures
        struct = 0
        for s in struct_list:
            struct = struct + normalize_struct(s)
        struct = struct / len(struct_list)

        x = torch.cat([feat, struct], dim=1)
        att = self.fuse(x)
        att = torch.sigmoid(att)

        return feat * (1.0 + self.beta * att)


# ===============================
# 4. Structure-Anything Head
# ===============================

class StructureAnythingHead(nn.Module):
    """
    RGB feature -> Structure-enhanced feature
    """
    def __init__(self, channels, ca_reduction=4):
        super().__init__()

        # Spectral encoder (lightweight, stable)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.channel_attn = StructureChannelAttention(
            channels, reduction=ca_reduction
        )

        self.structure_diff = MultiScaleStructureDifference(channels)
        self.spatial_attn = MultiScaleStructureSpatialAttention(channels)

        # final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. spectral projection
        f = self.encoder(x)

        # 2. channel-wise structure selection
        f = self.channel_attn(f)

        # 3. multi-scale structure extraction (normalization and fusion are done in spatial_attn)
        diff3, diff5, diff7 = self.structure_diff(f)
        struct_list = [diff3, diff5, diff7]

        # 4. multi-scale structure-driven spatial attention
        f = self.spatial_attn(f, struct_list)

        # 5. refinement + identity
        f = self.refine(f) + x

        return f


# ===============================
# 5. RGB–Structure Fusion
# ===============================

class StructureFusion(nn.Module):
    """
    RGB feature + Structure feature fusion
    """
    def __init__(self, channels):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, data):
        rgb_feat, struct_feat = data
        x = torch.cat([rgb_feat, struct_feat], dim=1)
        return self.fusion(x)


__all__ = [
    "StructureAnythingHead",
    "StructureFusion"
]
