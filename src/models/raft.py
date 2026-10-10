"""Pure RAFT core for non-rigid atmospheric optical flow.

Port of the RAFT model cell from the original hybrid-training notebook:
ResNet encoders (1ch satellite input) -> 4D all-pairs correlation pyramid ->
ConvGRU iterative refinement. FP16-safe via scaled dot-product correlation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard ResNet block used in RAFT encoders."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.InstanceNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class RAFTEncoder(nn.Module):
    """RAFT feature/context encoder. Outputs 1/8-scale features.

    Modified for ``in_channels=1`` (satellite IR) instead of 3 (RGB).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 96, stride=2)
        self.layer3 = ResidualBlock(96, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class CorrBlock(nn.Module):
    """RAFT 4D correlation volume + pooling pyramid lookup."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid: list[torch.Tensor] = []

        b, c, h, w = fmap1.shape
        fmap1_flat = fmap1.view(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
        fmap2_flat = fmap2.view(b, c, h * w)  # [B, C, H*W]
        corr = torch.bmm(fmap1_flat, fmap2_flat)
        corr = corr / torch.sqrt(torch.tensor(c, dtype=torch.float32, device=fmap1.device))
        corr = corr.view(b, h, w, h, w).contiguous()

        corr_reshaped = corr.view(b * h * w, 1, h, w)
        self.corr_pyramid.append(corr_reshaped)
        for _ in range(num_levels - 1):
            corr_reshaped = F.avg_pool2d(corr_reshaped, 2, stride=2)
            self.corr_pyramid.append(corr_reshaped)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        r = self.radius
        b, _, h, w = coords.shape
        out_pyramid = []
        coords_hw2 = coords.permute(0, 2, 3, 1)  # [B, H, W, 2] strictly (x, y)
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy_grid, dx_grid = torch.meshgrid(dy, dx, indexing="ij")
            delta = torch.stack([dx_grid, dy_grid], dim=-1)

            centroid = coords_hw2.reshape(b * h * w, 1, 1, 2) / (2**i)
            coords_lvl = centroid + delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl[..., 0] = 2.0 * coords_lvl[..., 0] / max(corr.shape[3] - 1, 1) - 1.0
            coords_lvl[..., 1] = 2.0 * coords_lvl[..., 1] / max(corr.shape[2] - 1, 1) - 1.0

            sampled = F.grid_sample(corr, coords_lvl, align_corners=True)
            sampled = sampled.view(b, h, w, -1).permute(0, 3, 1, 2)
            out_pyramid.append(sampled)
        return torch.cat(out_pyramid, dim=1)


class RAFTUpdateBlock(nn.Module):
    """Recurrent GRU update block (iterative flow refinement)."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(4 * (2 * 4 + 1) ** 2 + 2, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 96, 3, padding=1),
            nn.ReLU(True),
        )
        self.convz = nn.Conv2d(hidden_dim + 128 + 96, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + 128 + 96, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + 128 + 96, hidden_dim, 3, padding=1)
        self.flow_head = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def forward(
        self, net: torch.Tensor, inp: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mot_feat = self.motion_encoder(torch.cat([corr, flow], dim=1))
        x = torch.cat([inp, mot_feat], dim=1)
        z = torch.sigmoid(self.convz(torch.cat([net, x], dim=1)))
        r = torch.sigmoid(self.convr(torch.cat([net, x], dim=1)))
        q = torch.tanh(self.convq(torch.cat([r * net, x], dim=1)))
        net = (1 - z) * net + z * q
        return net, self.flow_head(net)


class PureRAFTCore(nn.Module):
    """Full RAFT pipeline (encoder + context + GRU iterations)."""

    def __init__(self, iters: int = 8) -> None:
        super().__init__()
        self.iters = iters
        self.fnet = RAFTEncoder(in_channels=1, out_channels=256)
        self.cnet = RAFTEncoder(in_channels=1, out_channels=256)
        self.update_block = RAFTUpdateBlock(hidden_dim=128)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        fmap1, fmap2 = self.fnet(image1), self.fnet(image2)
        cnet_feat = self.cnet(image1)
        net, inp = torch.split(cnet_feat, 128, dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=4, radius=4)

        b, _, h, w = fmap1.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        coords0 = torch.stack([x, y], dim=0).float().to(fmap1.device)
        coords0 = coords0.unsqueeze(0).repeat(b, 1, 1, 1)
        coords1 = coords0.clone()

        for _ in range(self.iters):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            net, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            flow_up = 8.0 * F.interpolate(
                coords1 - coords0, scale_factor=8.0, mode="bilinear", align_corners=True
            )
        return flow_up
