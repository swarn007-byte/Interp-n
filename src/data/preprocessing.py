"""Radiance / brightness-temperature preprocessing.

Covers everything the notebooks did inline:
- Kelvin normalisation (180K–330K <-> [0, 1])
- 3x3 local-mean NaN / dead-pixel healing matrix
- CMI reading from .npy or raw GOES NetCDF (.nc)
"""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

BT_MIN = 180.0
BT_MAX = 330.0
COLD_CLOUD_THRESHOLD_K = 235.0
DEFAULT_CROP = (slice(2000, 2512), slice(2500, 3012))


def normalize_bt(arr: np.ndarray, bt_min: float = BT_MIN, bt_max: float = BT_MAX) -> np.ndarray:
    """Map brightness temperature (K) to [0, 1]."""
    return np.clip((arr - bt_min) / (bt_max - bt_min), 0.0, 1.0).astype("float32")


def denormalize_bt(arr: np.ndarray, bt_min: float = BT_MIN, bt_max: float = BT_MAX) -> np.ndarray:
    """Map [0, 1] back to brightness temperature (K)."""
    return arr * (bt_max - bt_min) + bt_min


def heal_nans_local_mean(data: np.ndarray, fill_fallback: float = BT_MIN) -> np.ndarray:
    """Replace NaNs with the local 3x3 mean (pre-network telemetry healing matrix).

    This is the deterministic healing step from the deck: dead sensor pixels
    never reach the network. Falls back to ``fill_fallback`` for fully-NaN holes.
    """
    data = np.asarray(data, dtype=np.float64)
    if not np.isnan(data).any():
        return data.astype("float32")
    valid_mask = ~np.isnan(data)
    filled = np.nan_to_num(data, nan=0.0)
    # uniform_filter returns the *mean*; multiply by 9 to recover the 3x3 sum.
    sum_valid = ndimage.uniform_filter(filled, size=3, mode="reflect") * 9.0
    count_valid = ndimage.uniform_filter(valid_mask.astype(float), size=3, mode="reflect") * 9.0
    local_avg = np.divide(sum_valid, count_valid, out=np.zeros_like(sum_valid), where=count_valid != 0)
    data[~valid_mask] = local_avg[~valid_mask]
    return np.nan_to_num(data, nan=fill_fallback).astype("float32")


def read_cmi(path: str | Path, crop: int = 512, nan_fill_k: float = 270.0) -> np.ndarray:
    """Read a CMI field from ``.npy`` or raw GOES NetCDF and centre-crop it.

    Mirrors ``read_cmi`` from the Phase-2 notebook, including the fixed
    ``[2000:2512, 2500:3012]`` storm-region crop for raw NetCDF files.
    """
    path = str(path)
    if path.endswith(".npy"):
        data = np.load(path)
    else:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            data = ds["CMI"].values[DEFAULT_CROP[0], DEFAULT_CROP[1]]
        data = np.nan_to_num(data, nan=nan_fill_k)
    h, w = data.shape
    if h > crop or w > crop:
        dh, dw = (h - crop) // 2, (w - crop) // 2
        data = data[dh : dh + crop, dw : dw + crop]
    return np.asarray(data, dtype="float32")


def centre_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    if h <= crop and w <= crop:
        return arr
    dh, dw = (h - crop) // 2, (w - crop) // 2
    return arr[..., dh : dh + crop, dw : dw + crop]
