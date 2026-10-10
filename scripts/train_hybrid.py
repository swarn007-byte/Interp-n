#!/usr/bin/env python3
"""Train the full hybrid RAFT model on S3-streamed GOES data (needs GPU + net)."""
import argparse
from src.training.train_hybrid import train_hybrid, validate_hybrid

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--s3", default="noaa-goes18/ABI-L2-CMIPF/2024/250/*/*C13*.nc")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max-batches", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--resume", default="")
    p.add_argument("--out", default="FINAL_BEST_CHECKPOINT.pth")
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    if a.validate_only:
        validate_hybrid(a.resume or a.out)
    else:
        train_hybrid(
            s3_path=a.s3, epochs=a.epochs, max_batches=a.max_batches,
            batch_size=a.batch_size, resume_path=a.resume, checkpoint_out=a.out,
        )
