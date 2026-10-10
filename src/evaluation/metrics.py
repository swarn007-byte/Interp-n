"""MSE / PSNR / SSIM / FSIM in both torch and numpy flavours.

- torch variants run on-GPU for training/validation loops.
- numpy variants power the Phase-2/Phase-4 notebook-equivalent evaluation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.metrics import structural_similarity as _ssim


def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = calculate_mse(pred, target)
    if float(mse) == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 10 * torch.log10((data_range**2) / mse)


def ssim_torch(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> torch.Tensor:
    """Tensor-native SSIM (mean over the SSIM map)."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    pad = window_size // 2
    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    s1 = F.avg_pool2d(pred**2, window_size, stride=1, padding=pad) - mu1_sq
    s2 = F.avg_pool2d(target**2, window_size, stride=1, padding=pad) - mu2_sq
    s12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return ssim_map.mean()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    return ssim_torch(pred, target, **kwargs)


def calculate_fsim_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sobel-edge SSIM proxy for FSIM (pure torch)."""
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                           device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                           device=pred.device).view(1, 1, 3, 3)
    gx_p = F.conv2d(pred, sobel_x, padding=1)
    gy_p = F.conv2d(pred, sobel_y, padding=1)
    gx_t = F.conv2d(target, sobel_x, padding=1)
    gy_t = F.conv2d(target, sobel_y, padding=1)
    grad_pred = torch.sqrt(gx_p**2 + gy_p**2 + 1e-6)
    grad_target = torch.sqrt(gx_t**2 + gy_t**2 + 1e-6)
    return ssim_torch(grad_pred, grad_target)


def calculate_fsim(im1: np.ndarray, im2: np.ndarray) -> float:
    """Numpy FSIM approximation (phase-congruency + gradient magnitude).

    Port of Phase-4 ``calculate_fsim``: Sobel gradient similarity weighted by
    Laplacian-based phase congruency.
    """
    im1_255, im2_255 = im1 * 255.0, im2 * 255.0
    t1, t2 = 0.85, 160.0
    dx1, dy1 = ndimage.sobel(im1_255, axis=0), ndimage.sobel(im1_255, axis=1)
    dx2, dy2 = ndimage.sobel(im2_255, axis=0), ndimage.sobel(im2_255, axis=1)
    gm1, gm2 = np.sqrt(dx1**2 + dy1**2), np.sqrt(dx2**2 + dy2**2)
    s_g = (2 * gm1 * gm2 + t2) / (gm1**2 + gm2**2 + t2)
    pc1, pc2 = np.abs(ndimage.laplace(im1_255)), np.abs(ndimage.laplace(im2_255))
    s_pc = (2 * pc1 * pc2 + t1) / (pc1**2 + pc2**2 + t1)
    pc_max = np.maximum(pc1, pc2)
    return float(np.sum(s_pc * s_g * pc_max) / (np.sum(pc_max) + 1e-6))


def evaluate_prediction(pred: np.ndarray, target: np.ndarray, bt_min: float = 180.0, bt_max: float = 330.0) -> dict:
    """Numpy evaluation mirroring the Phase-4 report (MSE/MAE-K/PSNR/SSIM/FSIM)."""
    from src.data.preprocessing import denormalize_bt

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mse = float(np.mean((pred - target) ** 2))
    mae_k = float(np.mean(np.abs(denormalize_bt(pred) - denormalize_bt(target))))
    psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else 100.0
    ssim = float(_ssim(pred, target, data_range=1.0))
    fsim = float(calculate_fsim(pred, target))
    return {"MSE": mse, "MAE (Kelvin)": mae_k, "PSNR (dB)": psnr, "SSIM": ssim, "FSIM": fsim}
