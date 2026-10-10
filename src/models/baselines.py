"""Classical + deep baselines (Phase-2 notebook).

- linear midpoint: ``0.5 * t0 + 0.5 * t2``
- Farneback optical flow (OpenCV) + symmetric 0.5 warping
- RAFT (torchvision small/large) + symmetric 0.5 warping
"""

import cv2
import numpy as np
import torch


def warp_with_flow(image: np.ndarray, flow: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[..., 0] * scale).astype("float32")
    map_y = (y + flow[..., 1] * scale).astype("float32")
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def linear_midpoint(t0: np.ndarray, t2: np.ndarray) -> np.ndarray:
    return np.clip(0.5 * t0 + 0.5 * t2, 0.0, 1.0)


def farneback_midpoint(t0: np.ndarray, t2: np.ndarray) -> np.ndarray:
    a, b = (t0 * 255).astype("uint8"), (t2 * 255).astype("uint8")
    flow_0_to_2 = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    flow_2_to_0 = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    warped_0 = warp_with_flow(t0, flow_0_to_2, 0.5)
    warped_2 = warp_with_flow(t2, flow_2_to_0, 0.5)
    return np.clip(0.5 * warped_0 + 0.5 * warped_2, 0.0, 1.0)


def _to_raft_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.repeat(image[None, ...], 3, axis=0)).float()[None]


def raft_midpoint(t0: np.ndarray, t2: np.ndarray, model_size: str = "small") -> np.ndarray:
    """Torchvision RAFT midpoint prediction (lazy import, downloads weights once)."""
    from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_size == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True).to(device).eval()
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True).to(device).eval()

    transforms = weights.transforms()
    img0, img2 = transforms(_to_raft_tensor(t0), _to_raft_tensor(t2))
    img0, img2 = img0.to(device), img2.to(device)
    with torch.no_grad():
        flow_0_to_2 = model(img0, img2)[-1][0].detach().cpu().permute(1, 2, 0).numpy()
        flow_2_to_0 = model(img2, img0)[-1][0].detach().cpu().permute(1, 2, 0).numpy()
    warped_0 = warp_with_flow(t0, flow_0_to_2, 0.5)
    warped_2 = warp_with_flow(t2, flow_2_to_0, 0.5)
    return np.clip(0.5 * warped_0 + 0.5 * warped_2, 0.0, 1.0)
