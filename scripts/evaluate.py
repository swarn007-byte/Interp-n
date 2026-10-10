#!/usr/bin/env python3
"""Phase-4: evaluate the trained refiner (metrics.json + plots)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_metric

from src.data.preprocessing import denormalize_bt
from src.evaluation.metrics import calculate_fsim
from src.models.residual_unet import SmallResidualUNet
from src.utils.device import pick_device
from src.utils.visualization import plot_error_map, plot_prediction_vs_target, plot_residual_map

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="outputs/baseline/triplet_and_predictions.npz")
    p.add_argument("--weights", default="outputs/unet_refiner/best_small_residual_unet.pt")
    p.add_argument("--out-dir", default="outputs/unet_refiner_eval")
    a = p.parse_args()

    device = pick_device()
    data = np.load(a.npz)
    x = np.stack([data["t0"], data["t2"], data["flow"]], axis=0)[None].astype("float32")
    target = data["target"].astype("float32")

    model = SmallResidualUNet().to(device)
    model.load_state_dict(torch.load(a.weights, map_location=device))
    model.eval()
    with torch.no_grad():
        pred_t, res_t = model(torch.from_numpy(x).to(device), return_residual=True)
        pred, residual = pred_t.detach().cpu().numpy()[0, 0], res_t.detach().cpu().numpy()[0, 0]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "unet_prediction.npz", pred=pred, residual=residual, target=target)

    mse = float(np.mean((pred - target) ** 2))
    report = {
        "MSE": mse,
        "MAE (Kelvin)": float(np.mean(np.abs(denormalize_bt(pred) - denormalize_bt(target)))),
        "PSNR (dB)": float(10 * np.log10(1.0 / mse)) if mse > 0 else 100.0,
        "SSIM": float(ssim_metric(pred, target, data_range=1.0)),
        "FSIM": float(calculate_fsim(pred, target)),
    }
    print(json.dumps(report, indent=2))
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    plot_prediction_vs_target(denormalize_bt(pred), denormalize_bt(target), out / "pred_vs_target.png")
    plot_residual_map(residual, out / "residual_map.png")
    plot_error_map(denormalize_bt(pred), denormalize_bt(target), out / "error_map_kelvin.png")
    print(f"Plots + metrics saved -> {out}")
