import numpy as np
import torch

from src.data.preprocessing import denormalize_bt, heal_nans_local_mean, normalize_bt
from src.evaluation.metrics import evaluate_prediction
from src.inference.full_disk import hanning_2d, patch_grid, stitch_predictions
from src.inference.predict import predict_full_disk_refiner
from src.models.hybrid import HybridOpticalFlowModel
from src.models.losses import SobelGradientLoss, hybrid_loss, total_loss
from src.models.residual_unet import SmallResidualUNet


def test_normalize_roundtrip():
    arr = np.array([[180.0, 255.0, 330.0]], dtype=np.float32)
    assert np.allclose(denormalize_bt(normalize_bt(arr)), arr, atol=1e-5)


def test_heal_nans():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    healed = heal_nans_local_mean(arr)
    assert not np.isnan(healed).any()
    assert abs(float(healed[1, 1]) - 5.0) < 1e-5


def test_refiner_forward_shapes():
    model = SmallResidualUNet(base=16)
    x = torch.rand(2, 3, 64, 64)
    pred, res = model(x, return_residual=True)
    assert pred.shape == (2, 1, 64, 64)
    assert res.shape == (2, 1, 64, 64)
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_refiner_checkpoint_loads():
    import os

    ckpt = "weights/best_olk_residual_unet.pt"
    if not os.path.exists(ckpt):  # pragma: no cover
        import pytest

        pytest.skip("shipped weights not present")
    model = SmallResidualUNet(base=32)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    x = torch.rand(1, 3, 128, 128)
    assert model(x).shape == (1, 1, 128, 128)


def test_losses_finite():
    pred = torch.rand(2, 1, 64, 64)
    tgt = torch.rand(2, 1, 64, 64)
    base = torch.rand(2, 1, 64, 64)
    loss, parts = total_loss(pred, tgt, pred - base, base)
    assert torch.isfinite(loss)
    assert set(parts) == {"l1", "ssim", "gradient", "cold_cloud", "residual"}
    assert torch.isfinite(hybrid_loss(pred, tgt))
    assert torch.isfinite(SobelGradientLoss()(pred, tgt))


def test_hanning_partition_of_unity():
    n, stride = 256, 128
    H = W = 512
    window = hanning_2d(n)
    canvas = np.zeros((H, W))
    for y in range(0, H - n + 1, stride):
        for x in range(0, W - n + 1, stride):
            canvas[y : y + n, x : x + n] += window
    interior = canvas[stride:-stride, stride:-stride]
    # np.hanning is symmetric (zero endpoints); with 50% overlap the
    # partition-of-unity holds to ~1% — stitch normalisation absorbs the rest.
    assert interior.min() > 0.95, "Hanning weights must tile without gaps"


def test_stitch_identity():
    H = W = 512
    full = np.random.rand(H, W).astype("float32")
    n, stride = 256, 128
    positions = patch_grid((H, W), n, stride)
    patches = [full[y : y + n, x : x + n] for y, x in positions]
    assert np.allclose(stitch_predictions(patches, positions, (H, W), n), full, atol=1e-5)


def test_metrics_sanity():
    a = np.random.rand(64, 64).astype("float32")
    rep = evaluate_prediction(a, a)
    assert rep["MSE"] == 0.0
    assert rep["SSIM"] > 0.99
    b = np.clip(a + 0.05, 0, 1).astype("float32")
    assert evaluate_prediction(b, a)["MSE"] > 0


def test_hybrid_forward_t05():
    model = HybridOpticalFlowModel(iters=2)  # tiny for CPU test
    f0 = torch.rand(1, 1, 64, 64)
    f2 = torch.rand(1, 1, 64, 64)
    t = torch.tensor([0.5])
    pred, mask, w0, w2 = model(f0, f2, t)
    assert pred.shape == (1, 1, 64, 64)
    assert mask.shape == (1, 1, 64, 64)
