"""Overlapping Hanning-window full-disk stitching.

Standard U-Nets crash or leave grid seams on 5424x5424 disks with naive
tiling. This module implements the deck's resolution-agnostic blending:

- extract overlapping patches (default 256px, stride 128)
- taper each predicted patch with a separable 2-D Hanning window
- accumulate ``prediction * window`` / ``window`` (partition-of-unity)

Result: *no visually detectable seam artifacts*.
"""

import numpy as np


def hanning_1d(n: int) -> np.ndarray:
    """1-D Hanning taper, zero at both edges."""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    return np.hanning(n).astype(np.float64)


def hanning_2d(patch_size: int) -> np.ndarray:
    """Separable 2-D Hanning window (outer product of 1-D tapers)."""
    w = hanning_1d(patch_size)
    window = np.outer(w, w)
    # Guard against exact zeros at corners breaking normalisation.
    return np.maximum(window, 1e-8)


def patch_grid(image_hw: tuple[int, int], patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners covering ``image_hw`` (last row/col clamped to edge)."""
    h, w = image_hw
    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)
    return [(y, x) for y in ys for x in xs]


def stitch_predictions(
    patch_preds: list[np.ndarray],
    positions: list[tuple[int, int]],
    full_hw: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """Blend patch predictions into a full-disk image.

    Args:
        patch_preds: list of ``(patch_size, patch_size)`` predictions.
        positions: matching ``(y, x)`` top-left corners.
        full_hw: ``(H, W)`` of the output disk.
        patch_size: patch edge length (must match preds).
    """
    window = hanning_2d(patch_size)
    canvas = np.zeros(full_hw, dtype=np.float64)
    weights = np.zeros(full_hw, dtype=np.float64)
    for pred, (y, x) in zip(patch_preds, positions):
        canvas[y : y + patch_size, x : x + patch_size] += np.asarray(pred) * window
        weights[y : y + patch_size, x : x + patch_size] += window
    return (canvas / np.maximum(weights, 1e-8)).astype("float32")


def full_disk_from_patch_fn(
    full_t0: np.ndarray,
    full_t2: np.ndarray,
    patch_fn,
    patch_size: int = 256,
    stride: int = 128,
) -> np.ndarray:
    """Run ``patch_fn(t0_patch, t2_patch) -> pred_patch`` over a full disk.

    ``patch_fn`` receives/returns normalised ``[0, 1]`` float arrays.
    """
    h, w = full_t0.shape
    assert full_t2.shape == (h, w), "t0/t2 shape mismatch"
    positions = patch_grid((h, w), patch_size, stride)
    preds = [patch_fn(full_t0[y : y + patch_size, x : x + patch_size],
                      full_t2[y : y + patch_size, x : x + patch_size]) for y, x in positions]
    return stitch_predictions(preds, positions, (h, w), patch_size)
