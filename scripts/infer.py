#!/usr/bin/env python3
"""Single-patch inference with the shipped refiner weights."""
import argparse
from pathlib import Path

import numpy as np

from src.data.preprocessing import denormalize_bt, normalize_bt
from src.inference.predict import load_refiner, predict_patch_refiner
from src.utils.device import pick_device


def _load_frame(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype("float32")
    import xarray as xr

    with xr.open_dataset(p, engine="netcdf4") as ds:
        return np.nan_to_num(ds["CMI"].values[2000:2512, 2500:3012], nan=270.0).astype("float32")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--t0", required=True)
    p.add_argument("--t2", required=True)
    p.add_argument("--weights", default="weights/best_olk_residual_unet.pt")
    p.add_argument("--out", default="outputs/infer_patch.npy")
    p.add_argument("--baseline", choices=["linear", "raft"], default="linear")
    p.add_argument("--crop", type=int, default=512)
    a = p.parse_args()

    from src.data.preprocessing import centre_crop

    t0 = normalize_bt(centre_crop(_load_frame(a.t0), a.crop))
    t2 = normalize_bt(centre_crop(_load_frame(a.t2), a.crop))
    device = pick_device()
    model, device = load_refiner(a.weights, device=device)
    pred = predict_patch_refiner(t0, t2, model, device, baseline=a.baseline)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, denormalize_bt(pred))
    print(f"Saved Kelvin prediction -> {a.out}  shape={pred.shape}")
