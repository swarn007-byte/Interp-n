"""Public NOAA S3 download utilities (no AWS credentials needed).

Direct port of the Phase-1 data pipeline from the original development notebook:
- HTTPS XML listing of ``noaa-goes{16,18,19}`` buckets
- single-triplet downloader
- 24-hour continuous storm streamer (-> .npy + triplet manifests)
"""

import datetime as dt
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from tqdm import tqdm

S3_HTTPS = "https://{bucket}.s3.amazonaws.com"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def day_of_year(date_text: str) -> tuple[str, str]:
    """Convert ``YYYY-MM-DD`` to (year, 3-digit day-of-year)."""
    date = dt.datetime.strptime(date_text, "%Y-%m-%d").date()
    return str(date.year), f"{date.timetuple().tm_yday:03d}"


def list_keys(bucket: str, prefix: str) -> list[str]:
    """List S3 keys under ``prefix`` via the public XML API (handles pagination)."""
    url = f"{S3_HTTPS.format(bucket=bucket)}/?list-type=2&prefix={prefix}"
    keys: list[str] = []
    while True:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall("s3:Contents", ns):
            key = item.findtext("s3:Key", default="", namespaces=ns)
            if key:
                keys.append(key)
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=ns)
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=ns)
        if truncated.lower() != "true" or not token:
            break
        url = f"{S3_HTTPS.format(bucket=bucket)}/?list-type=2&prefix={prefix}&continuation-token={token}"
    return keys


def download_file(bucket: str, key: str, out_dir: Path) -> Path:
    """Download one S3 key with a progress bar (skips existing files)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(key).name
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already exists: {out_path}")
        return out_path
    url = f"{S3_HTTPS.format(bucket=bucket)}/{key}"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with out_path.open("wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path.name[:32]
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    return out_path


def parse_start_time(key: str) -> str:
    match = re.search(r"_s(\d{14})", key)
    return match.group(1) if match else key


# ---------------------------------------------------------------------------
# triplet download
# ---------------------------------------------------------------------------

def download_triplet(
    satellite: str = "goes16",
    product: str = "ABI-L2-CMIPF",
    date_str: str = "2025-06-01",
    hour_str: str = "12",
    channel_str: str = "13",
    out_root: str = "data/raw",
) -> Path:
    """Download the first 3 frames of a channel/hour and write ``triplet.txt``."""
    bucket = f"noaa-{satellite}"
    year, doy = day_of_year(date_str)
    hour = f"{int(hour_str):02d}"
    channel = f"C{int(channel_str):02d}"
    prefix = f"{product}/{year}/{doy}/{hour}/"

    print(f"Listing s3://{bucket}/{prefix}...")
    keys = list_keys(bucket, prefix)
    channel_keys = sorted([k for k in keys if channel in Path(k).name], key=parse_start_time)
    if len(channel_keys) < 3:
        raise RuntimeError(f"Found only {len(channel_keys)} files for {channel}. Try another date/hour.")

    selected = channel_keys[:3]
    print("Selected frames:")
    for key in selected:
        print(" -", key)

    out_dir = Path(out_root) / satellite / date_str / hour
    downloaded = [download_file(bucket, key, out_dir) for key in selected]
    manifest = out_dir / "triplet.txt"
    manifest.write_text("\n".join(str(p) for p in downloaded) + "\n")
    print(f"\nDone. Triplet manifest saved to: {manifest}")
    return manifest


# ---------------------------------------------------------------------------
# 24-hour continuous streamer
# ---------------------------------------------------------------------------

def stream_24h_storm(date_to_download: str = "2024-05-31", satellite: str = "goes16") -> Path:
    """Download ~2 files/hour for 24h, convert the storm crop to ``.npy``.

    Returns the manifest JSON path. Requires ``s3fs`` (lazy import so the
    rest of the package works without it).
    """
    import s3fs  # lazy: only needed for bulk streaming

    print(f"🛰️ Initializing 24-hour continuous stream downloader for {date_to_download}...")
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"region_name": "us-east-1"},
        config_kwargs={"connect_timeout": 15, "read_timeout": 15},
    )
    bucket = f"noaa-{satellite}"
    product = "ABI-L2-CMIPF"
    date_parsed = dt.datetime.strptime(date_to_download, "%Y-%m-%d")
    year, doy = date_parsed.year, date_parsed.timetuple().tm_yday
    target_channel = "C13"

    dest_dir = f"data/massive_storm_{date_to_download}"
    os.makedirs(dest_dir, exist_ok=True)
    downloaded_npy_paths: list[str] = []

    for hour in range(24):
        prefix = f"{bucket}/{product}/{year}/{doy:03d}/{hour:02d}/"
        try:
            all_files = fs.ls(prefix)
            remote_files = [f for f in all_files if f.endswith(".nc") and f"M6{target_channel}" in f]
            print(f"Hour {hour:02d}: found {len(remote_files)} target frames.")
        except Exception as exc:  # noqa: BLE001 — network listing is best-effort
            print(f"⚠️ Failed to list directory for hour {hour:02d}: {exc}")
            continue

        for remote_file in remote_files[:2]:  # 2/hour keeps demos fast
            fname = remote_file.split("/")[-1]
            local_nc = f"{dest_dir}/{fname}"
            local_npy = f"{dest_dir}/{fname.replace('.nc', '.npy')}"
            if os.path.exists(local_npy):
                downloaded_npy_paths.append(local_npy)
                continue
            try:
                print(f"   📥 Downloading {fname}...")
                fs.download(remote_file, local_nc)
                with xr.open_dataset(local_nc) as ds:
                    cmi_data = ds["CMI"].values[2000:2512, 2500:3012]
                    cmi_data = np.nan_to_num(cmi_data, nan=270.0)
                    np.save(local_npy, cmi_data)
                downloaded_npy_paths.append(local_npy)
                os.remove(local_nc)
            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️ Error processing {fname}: {exc}")
                if os.path.exists(local_nc):
                    os.remove(local_nc)

    downloaded_npy_paths.sort()
    triplets_dir = f"{dest_dir}/triplets"
    os.makedirs(triplets_dir, exist_ok=True)
    entries = []
    for i in range(len(downloaded_npy_paths) - 2):
        triplet_txt = f"{triplets_dir}/triplet_{i:03d}.txt"
        with open(triplet_txt, "w") as fh:
            fh.write(f"{downloaded_npy_paths[i]}\n{downloaded_npy_paths[i+1]}\n{downloaded_npy_paths[i+2]}\n")
        entries.append({"triplet_txt": triplet_txt, "date": date_to_download, "hour": str(i // 2)})

    manifest_payload = {"satellite": satellite, "entries": entries}
    os.makedirs("manifests", exist_ok=True)
    manifest_path = Path(f"manifests/storm_manifest_{date_to_download}.json")
    manifest_path.write_text(json.dumps(manifest_payload, indent=4))
    print(f"\n✅ Preprocessing complete! Compiled {len(entries)} sequences.")
    print(f"📂 Manifest saved to: {manifest_path}")
    return manifest_path
