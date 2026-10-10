#!/usr/bin/env python3
"""24-hour continuous storm streamer -> .npy crops + triplet manifests."""
import argparse
from src.data.download import stream_24h_storm

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2024-05-31")
    p.add_argument("--satellite", default="goes16")
    a = p.parse_args()
    stream_24h_storm(a.date, a.satellite)
