#!/usr/bin/env python3
"""Phase-2: load a triplet, run linear/Farneback/RAFT baselines, save .npz bundle."""
import argparse
import os
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

from src.data.preprocessing import denormalize_bt, normalize_bt, read_cmi
from src.models.baselines import farneback_midpoint, linear_midpoint, raft_midpoint


def evaluate(pred, target):
    return {
        "mse": float(np.mean((pred - target) ** 2)),
        "mae_k": float(np.mean(np.abs(denormalize_bt(pred) - denormalize_bt(target)))),
        "ssim": float(ssim_metric(pred, target, data_range=1.0)),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run baseline interpolations for one triplet.")
    p.add_argument("--manifest", default="data/raw/goes16/2025-06-01/12/triplet.txt")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--raft", choices=["small", "large", "skip"], default="small")
    p.add_argument("--out", default="outputs/baseline/triplet_and_predictions.npz")
    a = p.parse_args()

    if not os.path.exists(a.manifest):
        raise SystemExit(f"Manifest not found: {a.manifest}. Run scripts/download_triplet.py first!")
    paths = [ln.strip() for ln in Path(a.manifest).read_text().splitlines() if ln.strip()]

    t0 = normalize_bt(read_cmi(paths[0], a.crop))
    target = normalize_bt(read_cmi(paths[1], a.crop))
    t2 = normalize_bt(read_cmi(paths[2], a.crop))

    linear_pred = linear_midpoint(t0, t2)
    farneback_pred = farneback_midpoint(t0, t2)
    print("Linear:    ", evaluate(linear_pred, target))
    print("Farneback: ", evaluate(farneback_pred, target))

    if a.raft == "skip":
        flow_pred = farneback_pred
    else:
        flow_pred = raft_midpoint(t0, t2, model_size=a.raft)
        print(f"RAFT ({a.raft}):", evaluate(flow_pred, target))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, t0=t0, target=target, t2=t2, linear=linear_pred, flow=flow_pred)
    print(f"Saved -> {out}")
