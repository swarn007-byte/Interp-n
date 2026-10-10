"""Device helpers — CUDA > MPS > CPU with a single call site."""

import torch


def pick_device() -> torch.device:
    """Pick the best available torch device.

    Order: CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device | None = None) -> str:
    device = device or pick_device()
    return str(device)
