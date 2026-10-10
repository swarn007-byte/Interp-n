"""Patch + full-disk inference wrappers.

Two inference paths (mirrors the two training regimes):
1. **Refiner path** — torchvision RAFT baseline + ``SmallResidualUNet`` residual
   correction (loads ``best_olk_residual_unet.pt``).
2. **Hybrid path** — end-to-end :class:`HybridOpticalFlowModel` at arbitrary ``t``.
"""

from pathlib import Path

import numpy as np
import torch

from src.inference.full_disk import full_disk_from_patch_fn
from src.models.baselines import linear_midpoint
from src.models.residual_unet import SmallResidualUNet
from src.utils.device import pick_device

try:
    from src.models.baselines import raft_midpoint as _torchvision_raft
    _HAS_RAFT = True
except Exception:  # pragma: no cover
    _HAS_RAFT = False


# ---------------------------------------------------------------------------
# refiner path
# ---------------------------------------------------------------------------

def load_refiner(weights: str | Path, base: int = 32, device: torch.device | None = None) -> tuple[SmallResidualUNet, torch.device]:
    device = device or pick_device()
    model = SmallResidualUNet(base=base).to(device)
    state = torch.load(weights, map_location=device)
    # Accept both raw state_dict and {"model_state_dict": ...} checkpoints.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model, device


def _baseline_for_patches(t0: np.ndarray, t2: np.ndarray, method: str = "linear") -> np.ndarray:
    if method == "raft":
        if not _HAS_RAFT:  # pragma: no cover
            raise RuntimeError("torchvision RAFT unavailable")
        try:
            return _torchvision_raft(t0, t2, model_size="small")
        except Exception:
            return linear_midpoint(t0, t2)
    return linear_midpoint(t0, t2)


def predict_patch_refiner(
    t0: np.ndarray,
    t2: np.ndarray,
    model: SmallResidualUNet,
    device: torch.device,
    baseline: str = "linear",
) -> np.ndarray:
    """Predict one normalised patch with the refiner (no Hanning here)."""
    base = _baseline_for_patches(t0, t2, method=baseline)
    x = torch.from_numpy(np.stack([t0, t2, base], axis=0).astype("float32")).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).detach().cpu().numpy()[0, 0]
    return np.clip(pred, 0.0, 1.0).astype("float32")


def predict_full_disk_refiner(
    full_t0: np.ndarray,
    full_t2: np.ndarray,
    model: SmallResidualUNet,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 128,
    baseline: str = "linear",
) -> np.ndarray:
    """Hanning-blended full-disk prediction with the refiner."""
    fn = lambda a, b: predict_patch_refiner(a, b, model, device, baseline=baseline)
    return full_disk_from_patch_fn(full_t0, full_t2, fn, patch_size, stride)


# ---------------------------------------------------------------------------
# hybrid path (arbitrary t)
# ---------------------------------------------------------------------------

def load_hybrid(weights: str | Path | None = None, iters: int = 8, device: torch.device | None = None):
    from src.models.hybrid import HybridOpticalFlowModel

    device = device or pick_device()
    model = HybridOpticalFlowModel(iters=iters).to(device)
    if weights is not None:
        ckpt = torch.load(weights, map_location=device)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(cleaned)
    model.eval()
    return model, device


def predict_patch_hybrid(
    t0: np.ndarray, t2: np.ndarray, model, device: torch.device, t: float = 0.5
) -> np.ndarray:
    f0 = torch.from_numpy(t0.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
    f2 = torch.from_numpy(t2.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
    tt = torch.tensor([t], dtype=torch.float32, device=device)
    kwargs = {"device_type": device.type} if device.type == "cuda" else {}
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred, _, _, _ = model(f0, f2, tt)
        else:
            pred, _, _, _ = model(f0, f2, tt)
    return np.clip(pred.detach().cpu().numpy()[0, 0], 0.0, 1.0).astype("float32")


def predict_full_disk_hybrid(
    full_t0: np.ndarray,
    full_t2: np.ndarray,
    model,
    device: torch.device,
    t: float = 0.5,
    patch_size: int = 256,
    stride: int = 128,
) -> np.ndarray:
    fn = lambda a, b: predict_patch_hybrid(a, b, model, device, t=t)
    return full_disk_from_patch_fn(full_t0, full_t2, fn, patch_size, stride)
