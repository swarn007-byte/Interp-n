"""Matplotlib helpers for satellite IR visualisation.

All helpers work on *normalised* [0, 1] arrays unless stated otherwise.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_triplet_comparison(
    t0_k: np.ndarray,
    target_k: np.ndarray,
    t2_k: np.ndarray,
    pred_k: np.ndarray,
    save_path: str | Path | None = None,
    vmin: float = 200.0,
    vmax: float = 310.0,
) -> plt.Figure:
    """Plot t0 / target / t2 / prediction side-by-side in Kelvin."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title in zip(
        axes,
        [t0_k, target_k, t2_k, pred_k],
        ["Input t0", "Real middle target", "Input t2", "Prediction"],
    ):
        im = ax.imshow(img, cmap="turbo_r", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Brightness Temperature (K)", shrink=0.7)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_prediction_vs_target(
    pred_k: np.ndarray,
    target_k: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(pred_k, cmap="turbo_r", vmin=200, vmax=310)
    axes[0].set_title("Residual U-Net prediction")
    axes[0].axis("off")
    axes[1].imshow(target_k, cmap="turbo_r", vmin=200, vmax=310)
    axes[1].set_title("Real middle target")
    axes[1].axis("off")
    fig.suptitle("Predicted vs real target")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residual_map(
    residual: np.ndarray,
    save_path: str | Path | None = None,
    vmin: float = -0.08,
    vmax: float = 0.08,
) -> plt.Figure:
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(residual, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Normalised residual correction")
    plt.title("Learned residual correction map (UNet correction output)")
    plt.axis("off")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_map(
    pred_k: np.ndarray,
    target_k: np.ndarray,
    save_path: str | Path | None = None,
    vmax: float = 15.0,
) -> plt.Figure:
    error_k = np.abs(pred_k - target_k)
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(error_k, cmap="magma", vmin=0, vmax=vmax)
    plt.colorbar(label="Absolute error (K)")
    plt.title("Absolute error map (Kelvin)")
    plt.axis("off")
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hanning_window(patch_size: int = 256, save_path: str | Path | None = None) -> plt.Figure:
    """Plot the 1-D and 2-D Hanning taper used for full-disk stitching."""
    from src.inference.full_disk import hanning_2d

    w = hanning_2d(patch_size)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(w[patch_size // 2])
    axes[0].set_title(f"1-D Hanning taper (patch={patch_size})")
    axes[0].set_xlabel("pixel")
    axes[0].set_ylabel("weight")
    axes[0].grid(True, alpha=0.3)
    im = axes[1].imshow(w, cmap="viridis")
    axes[1].set_title("2-D Hanning window (outer product)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
