# Pre-trained weights

The refiner checkpoint from the original experiments is **not** committed to git
(3 MB, but keeps the repo light and avoids LFS).

## Expected file

```
weights/best_olk_residual_unet.pt   # SmallResidualUNet(base=32), ~0.77M params
```

This is the `state_dict` for `src.models.residual_unet.SmallResidualUNet`
(shipped as `best_olk_residual_unet.pt` alongside the original notebook).

## Install it

Copy the file from your local machine / drive:

```bash
cp ~/Downloads/best_olk_residual_unet.pt weights/best_olk_residual_unet.pt
```

All inference scripts default to this path:

```bash
python scripts/infer.py --t0 frame_t0.npy --t2 frame_t2.npy
python scripts/infer_full_disk.py --t0 disk_t0.npy --t2 disk_t2.npy
```

## Train your own

If you don't have the checkpoint, train one:

```bash
python scripts/download_triplet.py --date 2025-06-01 --hour 12
python scripts/make_baselines.py
python scripts/train_refiner.py --epochs 10 --patch-size 256
```

The new checkpoint lands in `outputs/unet_refiner/best_small_residual_unet.pt`.
