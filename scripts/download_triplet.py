#!/usr/bin/env python3
"""Download a GOES triplet and write data/raw/<sat>/<date>/<hour>/triplet.txt."""
import argparse
from src.data.download import download_triplet

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download a GOES triplet (t0, t1, t2).")
    p.add_argument("--satellite", default="goes16")
    p.add_argument("--product", default="ABI-L2-CMIPF")
    p.add_argument("--date", default="2025-06-01")
    p.add_argument("--hour", default="12")
    p.add_argument("--channel", default="13")
    p.add_argument("--out-root", default="data/raw")
    a = p.parse_args()
    download_triplet(a.satellite, a.product, a.date, a.hour, a.channel, a.out_root)
