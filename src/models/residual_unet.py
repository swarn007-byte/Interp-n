"""Attentive Residual U-Net refiner.

Takes ``[t0, t2, raft_baseline]`` (3ch) and predicts a bounded residual
correction added back onto the RAFT baseline:

    pred = clamp(baseline + 0.1 * tanh(head(decoder)), 0, 1)

Port of the Phase-3 refiner from the original development notebook
(GroupNorm + SiLU + SE attention).
Checkpoint ``best_olk_residual_unet.pt`` (``base=32``, ~0.77M params) loads here.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        return x * self.fc(x).view(b, c, 1, 1)


class ConvBlock(nn.Module):
    """Double 3x3 conv + GroupNorm + SiLU + SE attention."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(),
        )
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.net(x))


class SmallResidualUNet(nn.Module):
    """3-level attentive U-Net predicting a residual correction.

    Args:
        base: base channel width (32 in the shipped checkpoint).
    """

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 4, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.residual = nn.Conv2d(base, 1, 1)

    def forward(
        self, x: torch.Tensor, return_residual: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(e3)
        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        baseline = x[:, 2:3]
        residual = 0.1 * torch.tanh(self.residual(d1))
        pred = torch.clamp(baseline + residual, 0.0, 1.0)
        return (pred, residual) if return_residual else pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
