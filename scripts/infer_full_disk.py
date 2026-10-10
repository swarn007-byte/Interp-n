#!/usr/bin/env python3
"""Full-disk (5424x5424) Hanning-blended inference -> GeoTIFF/NPY/PNG."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.preprocessing import normalize_bt
from src.inference.predict import load_refiner, predict_full_disk_refiner
from src.utils.device import pick_device


def _load_any(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype("float32")
    if p.suffix == ".npz":
        d = np.load(p)
        key = "t0" if "t0" in d else list(d.keys())[0]
        return np.asarray(d[key]).astype("float32")
    import xarray as xr

    with xr.open_dataset(p, engine="netcdf4") as ds:
        var = "CMI" if "CMI" in ds else list(ds.data_vars)[0]
        arr = np.asarray(ds[var].values)
        arr = arr[0] if arr.ndim == 3 else arr
        return np.nan_to_num(arr, nan=270.0).astype("float32")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hanning-blended full-disk interpolation.")
    p.add_argument("--t0", required=True)
    p.add_argument("--t2", required=True)
    p.add_argument("--weights", default="weights/best_olk_residual_unet.pt")
    p.add_argument("--out", default="outputs/full_disk_pred.npy")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--baseline", choices=["linear", "raft"], default="linear")
    p.add_argument("--png", default="outputs/full_disk_pred.png")
    a = p.parse_args()

    t0 = normalize_bt(_load_any(a.t0))
    t2 = normalize_bt(_load_any(a.t2))
    assert t0.shape == t2.shape, f"shape mismatch {t0.shape} vs {t2.shape}"
    print(f"Full-disk input shape: {t0.shape}")

    device = pick_device()
    model, device = load_refiner(a.weights, device=device)
    pred = predict_full_disk_refiner(t0, t2, model, device, a.patch_size, a.stride, baseline=a.baseline)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred)
    print(f"Saved normalised full-disk prediction -> {out}")

    from src.data.preprocessing import denormalize_bt

    plt.figure(figsize=(8, 8))
    plt.imshow(denormalize_bt(pred), cmap="turbo_r", vmin=200, vmax=310)
    plt.axis("off")
    plt.title("Full-disk interpolated frame (Hanning-blended)")
    plt.savefig(a.png, dpi=150, bbox_inches="tight")
    print(f"Preview -> {a.png}")
