"""PyTorch datasets.

- :class:`AWSSatelliteDataset` — streams GOES NetCDF triplets straight from the
  public NOAA S3 bucket (sequential train/val split, hybrid-training path).
- :class:`TripletPatchDataset` — random 256px patches from local ``.npz``
  baseline bundles (Phase-3 refiner path).
- :class:`LocalTripletDataset` — full-frame local triplets for evaluation.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import BT_MAX, BT_MIN, heal_nans_local_mean


# ---------------------------------------------------------------------------
# S3 streaming dataset (hybrid RAFT training)
# ---------------------------------------------------------------------------

class AWSSatelliteDataset(Dataset):
    """Direct AWS S3 dataloader with sequential train/val splitting.

    Reads GOES NetCDF files from the public bucket (``anon=True``) and returns
    ``{frame_0, frame_2, t, frame_1_gt}`` with ``t=0.5`` by default.
    """

    def __init__(
        self,
        s3_glob_pattern: str,
        split: str = "train",
        train_ratio: float = 0.8,
        seq_len: int = 3,
        time_step: float = 0.5,
    ) -> None:
        import s3fs
        import xarray as xr

        self._xr = xr
        self.fs = s3fs.S3FileSystem(anon=True)
        all_files = sorted(self.fs.glob(s3_glob_pattern))
        if len(all_files) == 0:
            raise ValueError(f"No files found for pattern: {s3_glob_pattern}")
        split_idx = int(len(all_files) * train_ratio)
        if split == "train":
            self.files = all_files[:split_idx]
        elif split == "val":
            self.files = all_files[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")
        print(f"Loaded {len(self.files)} files for the {split} split.")
        self.seq_len = seq_len
        self.time_step = time_step
        self.temp_min = BT_MIN
        self.temp_max = BT_MAX

    def __len__(self) -> int:
        return len(self.files) - self.seq_len + 1

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        norm = (tensor - self.temp_min) / (self.temp_max - self.temp_min)
        return torch.clamp(norm, min=0.0, max=1.0)

    def __getitem__(self, idx: int) -> dict:
        import scipy.ndimage as ndimage

        file_paths = self.files[idx : idx + self.seq_len]
        tensors = []
        for path in file_paths:
            with self.fs.open(path, "rb") as fh:
                ds = self._xr.open_dataset(fh, engine="h5netcdf")
                var_name = list(ds.data_vars)[0]
                data = np.asarray(ds[var_name].values, dtype=np.float64)
                if np.isnan(data).any():
                    valid_mask = ~np.isnan(data)
                    filled = np.nan_to_num(data, nan=0.0)
                    sum_valid = ndimage.uniform_filter(filled, size=3, mode="reflect") * 9.0
                    count_valid = ndimage.uniform_filter(valid_mask.astype(float), size=3, mode="reflect") * 9.0
                    local_avg = np.divide(
                        sum_valid, count_valid, out=np.zeros_like(sum_valid), where=count_valid != 0
                    )
                    data[~valid_mask] = local_avg[~valid_mask]
                    data = np.nan_to_num(data, nan=self.temp_min)
                tensor = torch.from_numpy(np.asarray(data, dtype=np.float32)).float().unsqueeze(0)
                tensors.append(self._normalize(tensor))
        return {
            "frame_0": tensors[0],
            "frame_2": tensors[2],
            "t": torch.tensor([self.time_step], dtype=torch.float32),
            "frame_1_gt": tensors[1],
            "meta": {"paths": file_paths},
        }


# ---------------------------------------------------------------------------
# Local .npz patch dataset (residual-UNet refiner training)
# ---------------------------------------------------------------------------

class TripletPatchDataset(Dataset):
    """Random patches from ``triplet_and_predictions.npz`` bundles.

    Each ``.npz`` holds ``t0 / t2 / flow (RAFT baseline) / target`` in [0, 1].
    The model input is ``[t0, t2, baseline]`` and the label is ``target``.
    """

    def __init__(
        self,
        npz_paths: list[str],
        patch_size: int,
        samples_per_epoch: int,
        augment: bool = True,
    ) -> None:
        self.npz_paths = [str(p) for p in npz_paths]
        if not self.npz_paths:
            raise ValueError("No scenes found.")
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = random.choice(self.npz_paths)
        with np.load(path) as data:
            t0 = data["t0"].astype("float32")
            t2 = data["t2"].astype("float32")
            baseline = data["flow"].astype("float32")
            target = data["target"].astype("float32")

        h, w = target.shape
        ps = self.patch_size
        if h < ps or w < ps:
            raise ValueError(f"Scene {h}x{w} smaller than patch {ps}.")
        y, x = random.randint(0, h - ps), random.randint(0, w - ps)

        t0_t = torch.from_numpy(t0[y : y + ps, x : x + ps]).unsqueeze(0)
        t2_t = torch.from_numpy(t2[y : y + ps, x : x + ps]).unsqueeze(0)
        base_t = torch.from_numpy(baseline[y : y + ps, x : x + ps]).unsqueeze(0)
        tgt_t = torch.from_numpy(target[y : y + ps, x : x + ps]).unsqueeze(0)

        if self.augment:
            if random.random() > 0.5:
                t0_t, t2_t, base_t, tgt_t = (torch.flip(t, dims=[-1]) for t in (t0_t, t2_t, base_t, tgt_t))
            if random.random() > 0.5:
                t0_t, t2_t, base_t, tgt_t = (torch.flip(t, dims=[-2]) for t in (t0_t, t2_t, base_t, tgt_t))
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                t0_t, t2_t, base_t, tgt_t = (
                    torch.rot90(t, k=k, dims=[-2, -1]) for t in (t0_t, t2_t, base_t, tgt_t)
                )
        return torch.cat([t0_t, t2_t, base_t], dim=0), tgt_t


# ---------------------------------------------------------------------------
# Local full-frame dataset (evaluation)
# ---------------------------------------------------------------------------

class LocalTripletDataset(Dataset):
    """Yield full normalised frames from local ``.npy`` triplet manifests."""

    def __init__(self, triplet_txt_paths: list[str | Path]) -> None:
        from src.data.preprocessing import normalize_bt, read_cmi

        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for txt in triplet_txt_paths:
            paths = [ln.strip() for ln in Path(txt).read_text().splitlines() if ln.strip()]
            if len(paths) != 3:
                continue
            frames = [normalize_bt(read_cmi(p, crop=10_000)) for p in paths]
            self.items.append((frames[0], frames[1], frames[2]))
        if not self.items:
            raise ValueError("No triplets found.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        t0, tgt, t2 = self.items[idx]
        to_t = lambda a: torch.from_numpy(a.astype("float32")).unsqueeze(0)
        return {
            "frame_0": to_t(t0),
            "frame_1_gt": to_t(tgt),
            "frame_2": to_t(t2),
            "t": torch.tensor([0.5], dtype=torch.float32),
        }


def heal_wrapper(arr: np.ndarray) -> np.ndarray:
    """Backwards-compatible alias for the healing matrix."""
    return heal_nans_local_mean(arr)
