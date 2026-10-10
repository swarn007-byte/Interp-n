"""Refiner training loop (Phase-3 notebook, production-ready).

Trains :class:`SmallResidualUNet` on local ``.npz`` baseline bundles with the
composite edge-preserving loss. Splits patch samples into train/val by
resampling the same scene pool (matches the notebook behaviour).
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.goes_dataset import TripletPatchDataset
from src.models.losses import total_loss
from src.models.residual_unet import SmallResidualUNet
from src.utils.device import pick_device


def train_refiner(
    npz_paths: list[str] | None = None,
    epochs: int = 3,
    patch_size: int = 128,
    batch_size: int = 2,
    samples_per_epoch: int = 64,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    out_dir: str = "outputs/unet_refiner",
    checkpoint_name: str = "best_small_residual_unet.pt",
    loss_weights: dict | None = None,
    device: torch.device | None = None,
) -> Path:
    device = device or pick_device()
    print(f"Training refiner on: {device}")
    npz_paths = npz_paths or ["outputs/baseline/triplet_and_predictions.npz"]
    missing = [p for p in npz_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Baseline data missing: {missing}. Run scripts/make_baselines.py first!")

    loss_weights = loss_weights or {
        "l1_weight": 1.0,
        "ssim_weight": 2.5,
        "gradient_weight": 1.8,
        "cold_weight": 0.8,
        "residual_weight": 0.3,
        "cold_threshold_k": 235.0,
    }

    train_ds = TripletPatchDataset(npz_paths, patch_size, samples_per_epoch, augment=True)
    val_ds = TripletPatchDataset(npz_paths, patch_size, max(samples_per_epoch // 4, 32), augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SmallResidualUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val = float("inf")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, pred_res = model(x, return_residual=True)
            loss, _ = total_loss(pred, y, pred_res, x[:, 2:3], **loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred, pred_res = model(x, return_residual=True)
                v_loss, _ = total_loss(pred, y, pred_res, x[:, 2:3], **loss_weights)
                val_losses.append(v_loss.item())

        avg_train, avg_val = float(np.mean(train_losses)), float(np.mean(val_losses))
        scheduler.step()
        print(f"Epoch {epoch:02d}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), out_path / checkpoint_name)
            print("   🔥 Saved new best model checkpoint!")

    print(f"Best val loss: {best_val:.4f}")
    return out_path / checkpoint_name
