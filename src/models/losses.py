"""Composite edge-preserving losses.

Combines (Phase-3 notebook):
- L1 pixel loss
- SSIM structural loss
- Sobel gradient loss (sharp cyclone-eye / convective boundaries)
- Cold-cloud loss (extra weight below 235K — active convection)
- Residual-consistency loss (predicted vs true RAFT-missed detail)

Plus the lightweight ``hybrid_loss = L1 + 0.5 * (1 - SSIM)`` used for the
full hybrid RAFT training path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.metrics import ssim_torch


class SobelGradientLoss(nn.Module):
    """L1 distance between Sobel edge maps (a.k.a. PyTorchSobelGradientLoss)."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("weight_x", kx.unsqueeze(0).unsqueeze(0))
        self.register_buffer("weight_y", ky.unsqueeze(0).unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight_x = self.weight_x.to(pred.device)
        weight_y = self.weight_y.to(pred.device)
        _, c, _, _ = pred.shape
        pred_p = F.pad(pred, (1, 1, 1, 1), mode="replicate")
        tgt_p = F.pad(target, (1, 1, 1, 1), mode="replicate")
        grad_px = F.conv2d(pred_p, weight_x.repeat(c, 1, 1, 1), groups=c)
        grad_py = F.conv2d(pred_p, weight_y.repeat(c, 1, 1, 1), groups=c)
        grad_tx = F.conv2d(tgt_p, weight_x.repeat(c, 1, 1, 1), groups=c)
        grad_ty = F.conv2d(tgt_p, weight_y.repeat(c, 1, 1, 1), groups=c)
        return F.l1_loss(grad_px, grad_tx) + F.l1_loss(grad_py, grad_ty)


sobel_loss_fn = SobelGradientLoss()


def cold_cloud_loss(pred: torch.Tensor, target: torch.Tensor, threshold_norm: float) -> torch.Tensor:
    """Mean absolute error restricted to pixels colder than ``threshold_norm``."""
    mask = (target < threshold_norm).float()
    if mask.sum() < 1:
        return pred.new_tensor(0.0)
    return (torch.abs(pred - target) * mask).sum() / mask.sum()


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_residual: torch.Tensor,
    baseline: torch.Tensor,
    *,
    l1_weight: float = 1.0,
    ssim_weight: float = 2.5,
    gradient_weight: float = 1.8,
    cold_weight: float = 0.8,
    residual_weight: float = 0.3,
    cold_threshold_k: float = 235.0,
    bt_min: float = 180.0,
    bt_max: float = 330.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full Phase-3 composite loss. Returns (loss, part-dict)."""
    l1 = F.l1_loss(pred, target)
    ssim = 1.0 - ssim_torch(pred, target)
    grad = sobel_loss_fn(pred, target)
    cold_norm = (cold_threshold_k - bt_min) / (bt_max - bt_min)
    cold = cold_cloud_loss(pred, target, threshold_norm=cold_norm)
    true_residual = torch.clamp(target - baseline, -0.1, 0.1)
    residual = F.l1_loss(pred_residual, true_residual)

    loss = (
        l1_weight * l1
        + ssim_weight * ssim
        + gradient_weight * grad
        + cold_weight * cold
        + residual_weight * residual
    )
    parts = {
        "l1": float(l1.detach().cpu()),
        "ssim": float(ssim.detach().cpu()),
        "gradient": float(grad.detach().cpu()),
        "cold_cloud": float(cold.detach().cpu()),
        "residual": float(residual.detach().cpu()),
    }
    return loss, parts


def hybrid_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lightweight ``L1 + 0.5 * (1 - SSIM)`` loss for hybrid RAFT training."""
    return F.l1_loss(pred, target) + 0.5 * (1 - ssim_torch(pred, target))
