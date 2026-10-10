"""End-to-end hybrid interpolation wrapper (RAFT + attention fusion + refine)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.raft import PureRAFTCore


class WarpingLayer(nn.Module):
    """Differentiable bilinear backward warping."""

    def forward(self, img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img.size()
        xx = torch.arange(0, w, device=img.device).view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h, device=img.device).view(-1, 1).repeat(1, w)
        xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
        yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        vgrid = grid + flow
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(w - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(h - 1, 1) - 1.0
        return F.grid_sample(img, vgrid.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)


class AttentionFusion(nn.Module):
    """Lightweight convolutional spatiotemporal attention (anti-ghosting)."""

    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, warp0: torch.Tensor, warp2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.attention(torch.cat([warp0, warp2], dim=1))
        return mask * warp0 + (1 - mask) * warp2, mask


class ResidualRefinement(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + self.block(x), min=0.0, max=1.0)


class HybridOpticalFlowModel(nn.Module):
    """End-to-end hybrid interpolation pipeline powered by the pure RAFT core.

    ``forward(frame_0, frame_2, t)`` supports arbitrary ``t in (0, 1)``
    (temporal decoupling: flow is scaled by ``t`` / ``1 - t``).
    """

    def __init__(self, iters: int = 8) -> None:
        super().__init__()
        self.raft_core = PureRAFTCore(iters=iters)
        self.warper = WarpingLayer()
        self.fusion = AttentionFusion(in_channels=2)
        self.refinement = ResidualRefinement(in_channels=1)

    def forward(
        self, frame_0: torch.Tensor, frame_2: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flow_0_to_2 = self.raft_core(frame_0, frame_2)
        flow_2_to_0 = self.raft_core(frame_2, frame_0)
        t_tensor = t.view(-1, 1, 1, 1).to(frame_0.dtype)
        # Negated for *backward* warping (grid_sample semantics).
        flow_t_to_0 = -flow_0_to_2 * t_tensor
        flow_t_to_2 = -flow_2_to_0 * (1 - t_tensor)
        warp_0 = self.warper(frame_0, flow_t_to_0)
        warp_2 = self.warper(frame_2, flow_t_to_2)
        fused, mask = self.fusion(warp_0, warp_2)
        return self.refinement(fused), mask, warp_0, warp_2
