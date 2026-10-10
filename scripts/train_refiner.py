#!/usr/bin/env python3
"""Phase-3: train the residual U-Net refiner on local .npz bundles."""
import argparse
from src.training.train_refiner import train_refiner

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", nargs="+", default=["outputs/baseline/triplet_and_predictions.npz"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--out-dir", default="outputs/unet_refiner")
    a = p.parse_args()
    train_refiner(
        npz_paths=a.npz, epochs=a.epochs, patch_size=a.patch_size,
        batch_size=a.batch_size, samples_per_epoch=a.samples, out_dir=a.out_dir,
    )
