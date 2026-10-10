"""Hybrid optical-flow training + validation (cleaned-up training run).

Ports the original ``train()`` / ``run_final_validation()`` logic with:
- OneCycleLR + AMP GradScaler + gradient accumulation
- DataParallel when >1 GPU
- checkpoint resume + ``module.``-prefix stripping
"""

import os
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.data.goes_dataset import AWSSatelliteDataset
from src.evaluation.metrics import calculate_fsim_torch, calculate_mse, calculate_psnr, calculate_ssim
from src.models.hybrid import HybridOpticalFlowModel
from src.models.losses import hybrid_loss


def _strip_dataparallel(state: dict) -> OrderedDict:
    clean = OrderedDict()
    for k, v in state.items():
        clean[k[7:] if k.startswith("module.") else k] = v
    return clean


def train_hybrid(
    s3_path: str = "noaa-goes18/ABI-L2-CMIPF/2024/250/*/*C13*.nc",
    epochs: int = 40,
    max_batches: int = 1000,
    batch_size: int = 8,
    accum_steps: int = 4,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    resume_path: str = "",
    checkpoint_out: str = "FINAL_BEST_CHECKPOINT.pth",
) -> float:
    print("\n--- Starting hybrid RAFT training ---")
    dataset = AWSSatelliteDataset(s3_glob_pattern=s3_path, split="train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridOpticalFlowModel().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=max(1, max_batches // accum_steps), epochs=epochs
    )

    print("Pre-loading satellite sequences into RAM for True SGD...")
    all_samples = list(dataset)
    print(f"Loaded {len(all_samples)} massive sequences into RAM!")

    best_loss, start_epoch, epoch_losses = float("inf"), 0, []
    resume_file = resume_path or "FINAL_BEST_CHECKPOINT.pth"
    if resume_file and os.path.exists(resume_file):
        print(f"RESUMING FROM {resume_file}!")
        ckpt = torch.load(resume_file, map_location=device, weights_only=False)
        raw = model.module if isinstance(model, torch.nn.DataParallel) else model
        raw.load_state_dict(_strip_dataparallel(ckpt["model_state_dict"]))
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]
        for _ in range(start_epoch * (max_batches // accum_steps)):
            scheduler.step()

    use_amp = device.type == "cuda"
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for b in range(max_batches):
            f0_l, f1_l, f2_l, t_l = [], [], [], []
            for _ in range(batch_size):
                s = random.choice(all_samples)
                hs, ws = np.random.randint(1000, 4000), np.random.randint(1000, 4000)
                # .clone() avoids pinning giant 5424px views in RAM (notebook gotcha).
                f0_l.append(s["frame_0"].unsqueeze(0)[:, :, hs : hs + 256, ws : ws + 256].clone())
                f1_l.append(s["frame_1_gt"].unsqueeze(0)[:, :, hs : hs + 256, ws : ws + 256].clone())
                f2_l.append(s["frame_2"].unsqueeze(0)[:, :, hs : hs + 256, ws : ws + 256].clone())
                t_l.append(s["t"].unsqueeze(0))
            f0 = torch.cat(f0_l).to(device)
            f1 = torch.cat(f1_l).to(device)
            f2 = torch.cat(f2_l).to(device)
            t = torch.cat(t_l).to(device)

            optimizer.zero_grad(set_to_none=(b % accum_steps == 0))
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred, _, _, _ = model(f0, f2, t)
                    loss = hybrid_loss(pred, f1) / accum_steps
                scaler.scale(loss).backward()
                if (b + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            else:
                pred, _, _, _ = model(f0, f2, t)
                (hybrid_loss(pred, f1) / accum_steps).backward()
                if (b + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                loss = hybrid_loss(pred, f1)
            total_loss += float(loss) * (accum_steps if use_amp else 1)

        avg = total_loss / max_batches
        epoch_losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)
        if avg < best_loss:
            best_loss = avg
            raw = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler else {},
                    "loss": best_loss,
                },
                checkpoint_out,
            )
            print(f"--> Saved new best checkpoint (loss {best_loss:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(range(start_epoch + 1, epochs + 1), epoch_losses, marker="o", color="green", linewidth=2)
    plt.title("Training loss curve (OneCycleLR)")
    plt.xlabel("Epoch")
    plt.ylabel("Hybrid loss (L1 + SSIM)")
    plt.grid(True)
    plt.savefig("loss_graph_final.png")
    plt.close()
    return best_loss


@torch.no_grad()
def validate_hybrid(
    checkpoint: str,
    s3_path: str = "noaa-goes18/ABI-L2-CMIPF/2024/255/18/*C13*.nc",
    max_batches: int = 50,
) -> dict:
    print("\n================ FINAL VALIDATION ================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridOpticalFlowModel().to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(_strip_dataparallel(ckpt["model_state_dict"]))
    model.eval()

    dataset = AWSSatelliteDataset(s3_glob_pattern=s3_path, split="train")
    m_mse = m_psnr = m_ssim = m_fsim = 0.0
    n = 0
    use_amp = device.type == "cuda"
    for sample in dataset:
        if n >= max_batches:
            break
        hs, ws = np.random.randint(1000, 4000), np.random.randint(1000, 4000)
        f0 = sample["frame_0"].unsqueeze(0)[:, :, hs : hs + 256, ws : ws + 256].to(device)
        f1 = sample["frame_1_gt"].unsqueeze(0)[:, :, hs : hs + 256, ws : ws + 256].to(device)
        f2 = sample["frame_2"].unsqueeze(0)[:, :, hs : hs + 256, ws : ws + 256].to(device)
        t = sample["t"].to(device)
        if use_amp:
            with torch.amp.autocast("cuda"):
                pred, _, _, _ = model(f0, f2, t)
        else:
            pred, _, _, _ = model(f0, f2, t)
        m_mse += float(calculate_mse(pred, f1))
        m_psnr += float(calculate_psnr(pred, f1))
        m_ssim += float(calculate_ssim(pred, f1))
        m_fsim += float(calculate_fsim_torch(pred, f1))
        n += 1
    report = {
        "MSE": m_mse / n,
        "PSNR (dB)": m_psnr / n,
        "SSIM": m_ssim / n,
        "FSIM": m_fsim / n,
        "batches": n,
    }
    print(f"\nFinal test on {n} unseen sequences:\n{report}\n==============================================")
    return report
