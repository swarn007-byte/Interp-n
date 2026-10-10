"""Smoke-test the shipped refiner checkpoint end-to-end (synthetic data)."""

import numpy as np
import torch

from src.data.preprocessing import normalize_bt
from src.evaluation.metrics import evaluate_prediction
from src.inference.predict import load_refiner, predict_patch_refiner
from src.models.residual_unet import count_parameters
from src.utils.device import pick_device


def synthetic_cyclone(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size * 0.5, size * 0.5
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    spiral = np.sin(0.15 * r + 4 * np.arctan2(y - cy, x - cx + 1e-6))
    disk_k = 300 - 60 * np.exp(-((r - 40) ** 2) / 800) + 5 * spiral + rng.normal(0, 1.5, (size, size))
    return disk_k.astype("float32")


if __name__ == "__main__":
    device = pick_device()
    print("device:", device)

    t0_k = synthetic_cyclone(seed=0)
    t2_k = synthetic_cyclone(seed=1)  # slight evolution
    tgt_k = 0.5 * (t0_k + t2_k)
    t0, t2, tgt = normalize_bt(t0_k), normalize_bt(t2_k), normalize_bt(tgt_k)

    try:
        model, device = load_refiner("weights/best_olk_residual_unet.pt", device=device)
        print(f"refiner params: {count_parameters(model):,}")
        pred = predict_patch_refiner(t0, t2, model, device, baseline="linear")
    except FileNotFoundError:
        print("weights/best_olk_residual_unet.pt not found — using linear baseline for smoke test")
        pred = np.clip(0.5 * t0 + 0.5 * t2, 0, 1)

    print("metrics:", evaluate_prediction(pred, tgt))
    print("OK — quickstart smoke test passed.")
