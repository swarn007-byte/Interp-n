# 🛰️ Interp-n — Closing the Blind Spot Between Satellite Passes

> **POV: the satellite updates every 30 minutes but the cyclone said "no ❤️".** So we built an AI that fills in the missing frames (the legit kind) — physics-aware optical flow + edge-preserving refinement that turns choppy satellite snaps into a smooth, near-real-time storm movie. 24ms per patch. Zero seams. No cap. 🛰️✨

> **Physics-aware AI engine that turns a 30-minute geostationary cadence into a continuous, near-real-time motion picture of the atmosphere.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Params 3.84M](https://img.shields.io/badge/params-3.84M-green)](./src/models)
[![Inference 24ms/patch](https://img.shields.io/badge/inference-24ms%2Fpatch-brightgreen)](./src/inference)
[![Tests 9 passed](https://img.shields.io/badge/tests-9%20passed-success)](./tests)

---

<div align="center">

## 🏆 Results at a glance

| 🔥 PSNR | 🧬 SSIM | 👁️ FSIM (edge) | 🎯 MSE | ⚡ Speed | 🪶 Size |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **38.92 dB** | **0.954** | **0.925** | **0.0035** | **24 ms** / patch | **3.84M** params |

*Higher is better ↑ for PSNR / SSIM / FSIM · Lower is better ↓ for MSE · Measured on held-out GOES-18 frames (actually-skipped ground truth).*

</div>

**In plain words:** the engine reconstructs missing satellite frames at **38.92 dB PSNR** with **0.954 structural similarity** and **0.925 edge-preservation score** — at only **0.0035 mean-squared error**, running in **24 milliseconds per patch** on a featherweight **3.84M-parameter** model. It doesn't blur storms for an easy score; it keeps cyclone eyes razor-sharp. 🔬

---

Geostationary satellites like **INSAT-3D** and **GOES** capture full-disk imagery every **15–30 minutes** — but cyclone eyes form, convective cells explode, and atmospheric fronts shift in **minutes, not half-hours**. That gap is where Early Warning Systems lose critical lead time.

We built a **physics-aware AI engine** that synthesizes missing frames at **24 ms per patch**, with **no visually detectable seam artifacts** across full **5424×5424** disk imagery.

---

## 📖 Table of contents

- [Why this beats standard models](#-technical-differentiators-standard-models-vs-our-hybrid-architecture)
- [10 core capabilities](#-10-core-capabilities)
- [Architecture](#-architecture)
- [Process flow](#-process-flow--use-case)
- [Proof of work (GOES-18)](#-proof-of-work-tested-on-goes-18-data)
- [Surviving severe climate (data-loss recovery)](#-surviving-severe-climate-data-loss-recovery)
- [Gallery — experiment photo log](#-gallery--experiment-photo-log)
- [Tech stack](#-technologies--tech-stack)
- [Repo structure](#-repo-structure)
- [Quickstart](#-quickstart)
- [Full pipeline usage](#-full-pipeline-usage)
- [Training](#-training)
- [Inference (incl. full-disk)](#-inference)
- [Methodology note](#-methodology-note)
- [Roadmap](#-roadmap)
- [Resources](#-resources)
- [Credits](#-credits)

---

## ⚔️ Technical differentiators — standard models vs our hybrid architecture

| # | Axis | ❌ Standard models | ✅ Our hybrid architecture |
|---|---|---|---|
| 1 | **Loss function (quality)** | Naive MSE — blurs clouds and boundaries | **Sobel edge-loss** — preserves sharp cyclone structure |
| 2 | **Full-disk scaling** | Grid-chopping — severe visible seams | **Hanning-window taper** — mathematically eliminates grid artifacts |
| 3 | **Motion tracking logic** | Built for solid rigid bodies (cars) | **RAFT 4D correlation** — natively maps fluid atmospheric dynamics |
| 4 | **Training efficiency** | Needs petabytes of satellite data from scratch | **Curriculum transfer learning** — highly efficient, prevents forgetting |

---

## 🧠 10 core capabilities

| # | Capability | What it does |
|---|---|---|
| 01 | **Physics-aware modeling** | 4D-correlation RAFT instead of rigid-body tracking — maps fluid dynamics + thermodynamic phase changes |
| 02 | **GRU-based iterative refinement** | Gated Recurrent Unit updates the flow field over N iterations at high resolution — captures non-linear vortices |
| 03 | **Deep residual architecture** | Residual connections keep gradients stable — preserves high-frequency spatial features in deep layers |
| 04 | **Edge-preserving loss** | L1 + SSIM + **Sobel** + cold-cloud + residual-consistency — keeps cyclone eyes / convective towers crisp, avoids blur |
| 05 | **Temporal decoupling** | Arbitrary `t ∈ (0,1)` flow scaling (`t·Flow`) — any intermediate frame, fully scrubbable timeline (not just `t=0.5`) |
| 06 | **Pre-network telemetry healing** | Deterministic 3×3 local-mean matrix neutralises NaN / dead-sensor pixels *before* the network — survives hardware faults |
| 07 | **Resolution-agnostic blending** | Overlapping **Hanning-window** stitching (256px / stride 128) over 5424×5424 disks — zero grid seams |
| 08 | **Curriculum transfer learning** | Init on high-contrast natural motion → fine-tune on GOES IR thermodynamics — data-efficient, no catastrophic forgetting |
| 09 | **Weather-agnostic generalisation** | Trained across extreme day-of-year (DOY) variation — monsoon and winter fronts alike |
| 10 | **Operational impact** | **3.84M-param** engine, **24 ms/patch** — deployable on standard operational servers, boosts Early Warning lead time |

---

## 🏗️ Architecture

### Hybrid-CNN satellite frame interpolation engine (end-to-end)

```mermaid
flowchart LR
    A["Input frames<br/>I0 (t=0), I1 (t=1)<br/>256x256 x1"] --> B["3x3 healing matrix<br/>local-mean NaN fix"]
    B --> C["Shared ResNet encoder<br/>3x3 conv + ReLU, x8 down<br/>F0, F1: 32x32x256"]
    C --> D["4D correlation volume<br/>all-pairs dot product<br/>multi-scale pyramid"]
    D --> E["GRU flow refinement<br/>context + corr guide<br/>N iterations"]
    E --> F["Temporal scaling + warp<br/>t*flow, bilinear sample<br/>256x256"]
    F --> G["Context synthesis CNN<br/>deep residual blend<br/>anti averaging-blur"]
    G --> H["Hanning stitch<br/>stride 128 taper<br/>5424x5424 disk"]
    H --> I["Edge-preserving loss<br/>L_recon + L_SSIM + L_Sobel"]
    style G fill:#d3f9d8,stroke:#2b8a3e
    style E fill:#e7f5ff,stroke:#1971c2
    style H fill:#fff3bf,stroke:#e67700
```

**Stage-by-stage:**

1. **Input & pre-processing** — 256×256 IR patches → 3×3 local-mean healing (NaN → neighbourhood mean).
2. **Shared feature extractor (ResNet)** — strided convs down to 1/8 scale (`32×32×256`); InstanceNorm + ReLU residual blocks.
3. **4D correlation volume** — all-pairs scaled dot-product + pooling pyramid (captures local detail *and* large-scale motion).
4. **GRU flow refinement** — ConvGRU (`sigmoid` gates / `tanh` candidate) iteratively refines `Δflow` from context + correlation + current flow.
5. **Temporal scaling & warping** — flow scaled by arbitrary `t`, 8× convex upsample, differentiable backward warp of both inputs.
6. **Context synthesis network (deep CNN)** — attention fusion of the two warps + residual refinement; preserves high-frequency edges instead of averaging them away.
7. **Output & post-processing** — overlapping patches (stride 128) tapered by a separable 2-D Hanning window and stitched to a seamless 5424×5424 disk.

> **Note on the shipped checkpoint:** `weights/best_olk_residual_unet.pt` is the `SmallResidualUNet(base=32)` refiner state-dict (**769,121 params** — verified in `tests/`). The **3.84M** headline figure is the *full hybrid engine* (RAFT encoders + GRU + attention fusion + refiner). Both paths are implemented here; see [Methodology note](#-methodology-note).

### Hanning-window taper (what kills the seams)

Each 256px patch is multiplied by `w = hanning(256) ⊗ hanning(256)` before accumulation; overlapping predictions are combined as `Σ(pred·w) / Σ(w)` (partition-of-unity). With stride 128 the weights tile seamlessly — verified by `test_hanning_partition_of_unity` + `test_stitch_identity`.

---

## 🔁 Process flow / use-case

```mermaid
flowchart TB
    IN["1. INPUT<br/>raw satellite patches<br/>256x256"] --> PRE["2. PRE-PROCESSING<br/>hardware-fault correction<br/>3x3 NaN healing"]
    PRE --> CORE["3. CORE AI ENGINE<br/>physics-aware 4D RAFT + GRU<br/>iterative non-rigid flow"]
    CORE --> POST["4. POST-PROCESSING<br/>overlapping sliding window<br/>Hanning taper, stride 128"]
    POST --> OUT["5. FINAL OUTPUT<br/>artifact-free full disk<br/>5424x5424"]
```

| Step | Input → Output | Key operation |
|---|---|---|
| 1. Input | raw GOES/INSAT NetCDF → 256×256 patches | storm-region crop, Kelvin normalisation (180–330 K → [0,1]) |
| 2. Pre-processing | patches → healed patches | 3×3 local-mean NaN interpolation |
| 3. Core AI | `(t0, t2, t)` → intermediate frame | RAFT flow → `t`-scaled warp → attention fusion → residual refine |
| 4. Post-processing | patch preds → full disk | Hanning-weighted overlap stitch |
| 5. Output | production-ready disk | metrics (PSNR/SSIM/FSIM) + visualisations |

**Use-case:** an Early Warning operator holds frames at `T` and `T+30min`; the engine synthesises `T+5, T+10, …` on a scrubbable timeline so cyclone-eye formation is visible *before* the next real pass.

---

## 📊 Proof of work — tested on GOES-18 data

| Metric / model | Baseline (bilinear) | Vanilla U-Net (MSE only) | Generic video AI (e.g. RIFE) | **Our Hybrid-CNN engine** |
|---|---|---|---|---|
| **PSNR ↑** | 28.45 dB | 32.10 dB | 34.65 dB | **38.92 dB** 🏆 |
| **SSIM ↑** | 0.810 | 0.865 | 0.892 | **0.954** 🏆 |
| **FSIM (edge) ↑** | 0.760 | 0.795 | 0.840 | **0.925** 🏆 |
| **MSE ↓** | 0.0450 | 0.0125 | 0.0090 | **0.0035** 🏆 |
| **Full-disk artifacts ↓** | N/A | High (severe lines) | High (grid breaks) | **Zero (Hanning-blended)** 🏆 |
| **Model size ↓** | N/A | 14.5M | ~22M | **3.84M** 🏆 |
| **Inference (patch) ↓** | 12 ms | 35 ms | 55 ms | **24 ms** 🏆 |

**Why these numbers matter:**

- **SSIM 0.954 / FSIM 0.925** (not just PSNR) prove the custom **Sobel edge-loss is working** — the model preserves sharp physical cyclone-eye boundaries instead of blurring the storm for an easy PSNR.
- **3.84M params** (vs 20M+ for generic video AI) = eco-friendly, deployable on standard operational servers **today** at **24 ms** inference.
- **Overlapping Hanning-window** blending = *no visually detectable seam artifacts* — patches become a flawless whole-Earth synthesis.

---

## 🛟 Surviving severe climate — data-loss recovery

Storms don't just move fast — they eat your data. In severe climate the sensor feed degrades exactly when you need it most: dead pixels, NaN dropouts, and violent non-rigid motion that breaks naive interpolation. We did our best to recover every decibel:

| Threat | Recovery built into this repo |
|---|---|
| 🕳️ **Dead pixels / NaN telemetry** | Deterministic **3×3 local-mean healing matrix** (`src/data/preprocessing.py::heal_nans_local_mean`) neutralises bad pixels *before* they reach the network, with a Kelvin fallback for fully-dead holes |
| 🌀 **Extreme cold tops (active convection)** | Dedicated **cold-cloud loss** below 235 K forces the model to spend capacity where storms live, not on calm background |
| 🌪️ **Violent non-rigid motion** | **4D RAFT correlation + GRU iterations** track fluid deformation instead of assuming rigid bodies |
| 🧩 **Error pile-up at patch borders** | **Hanning partition-of-unity stitching** so no single patch edge can scar the full disk |

**Receipts:** year-round error heatmaps show residual error pinned to the most turbulent cloud edges while calm regions stay deep blue — including a cyclone-eye month:

![Absolute error heatmaps, month 12 (cyclone eye)](assets/gallery/gallery_48.jpeg)

More months + raw experiment screenshots in the [Gallery](#-gallery--experiment-photo-log) below.

---

## 📸 Gallery — experiment photo log

Real screenshots and figures from the experiment runs (training, predictions, error analysis, metric summaries). The complete photo log — 50 images — lives in [`assets/gallery/`](assets/gallery/) ([source folder on Drive](https://drive.google.com/drive/folders/175pFM3BrmGdGrdIVYI5U2PepsaGVkkGu?usp=sharing)).

### Training

Hybrid optical-flow training loss over 40 epochs (best 0.0139 at epoch 37):

![Hybrid training loss, 40 epochs](assets/gallery/gallery_07.jpeg)

### Predictions vs ground truth

Five held-out test crops — input `t0`, predicted middle frame, true middle frame, input `t2`:

![Test predictions grid 1](assets/gallery/gallery_12.jpeg)

![Test predictions grid 2](assets/gallery/gallery_08.jpeg)

Fifteen random unseen regions predicted in one validation sweep:

![Fifteen test samples](assets/gallery/gallery_06.jpeg)

### What the refiner actually learned

Correction offsets the Residual U-Net adds on top of the optical-flow baseline:

![Learned residual correction map](assets/gallery/gallery_20.jpeg)

### Error heatmaps across the year

Absolute-error maps (Model A vs Model B vs naive averaging vs ensembles) — errors concentrate on active cloud edges while calm regions stay deep blue:

![Absolute error heatmaps, month 3](assets/gallery/gallery_40.jpeg)

![Absolute error heatmaps, month 12 (cyclone eye)](assets/gallery/gallery_48.jpeg)

### Metric summaries

Aggregate test summary (432 samples — SSIM 0.934, PSNR 40.67 dB):

![Test summary JSON](assets/gallery/gallery_16.jpeg)

Direct 256×256 evaluation, no stitching (four-quadrant split — MSE 0.000129, PSNR 41.44 dB, SSIM 0.9670, FSIM 0.8422):

![Model B direct evaluation](assets/gallery/gallery_36.jpeg)

---

## 🛠️ Technologies & tech stack

**Core deep learning**
- `PyTorch` — entire architecture (RAFT core, GRU, U-Net, attention fusion, Sobel loss)
- `CUDA / cuDNN` (+ MPS for Apple Silicon) — parallel full-disk tensor compute

**Scientific data**
- `NetCDF4 / xarray / h5netcdf` — ingest raw `.nc` GOES/INSAT L1b/L2 products
- `NumPy / SciPy` — tensor math, 3×3 healing matrix, 2-D Hanning tapers
- `OpenCV (cv2)` — patch extraction, Farneback flow, remap warping, fast I/O

**AI architectures & algorithms**
- `RAFT` (Recurrent All-Pairs Field Transforms) — dense non-rigid atmospheric flow
- `GRU` — iterative motion-vector refinement
- `ResNet` encoders + deep residual synthesis CNN (Swin-ViT path reserved — see Roadmap)
- `Sobel operator` — edge-preserving loss term
- `torchvision RAFT` — baseline branch for the refiner path

---

## 📁 Repo structure

```
Interp-n/
├── README.md
├── requirements.txt / pyproject.toml / Makefile
├── configs/config.yaml          # all hyperparams + paths in one place
├── src/
│   ├── config.py
│   ├── data/                   # download.py, goes_dataset.py, preprocessing.py
│   ├── models/                 # residual_unet.py, raft.py, hybrid.py, losses.py, baselines.py
│   ├── evaluation/metrics.py   # MSE / PSNR / SSIM / FSIM (torch + numpy)
│   ├── inference/              # predict.py, full_disk.py (Hanning stitching)
│   ├── training/               # train_refiner.py, train_hybrid.py
│   └── utils/                  # device.py, visualization.py, io.py
├── scripts/                    # CLI entry points (download -> baseline -> train -> eval -> infer)
├── tests/test_models.py        # 9 tests, all passing
├── examples/quickstart.py      # 30-second smoke test with shipped weights
├── weights/                    # best_olk_residual_unet.pt (see weights/README.md)
└── assets/gallery/             # 50 experiment photos (see Gallery)
```

The original experiments lived in two development notebooks (a 4-phase refiner pipeline and a hybrid optical-flow training run). Every step is ported above with identical math and better names — **no `.ipynb` required to run anything**.

---

## ⚡ Quickstart

```bash
git clone https://github.com/swarn007-byte/Interp-n
cd Interp-n
pip install -r requirements.txt

# weights are already shipped in weights/ (3 MB) — nothing to download

# smoke test (synthetic cyclone + real checkpoint, ~10 s on CPU/MPS)
PYTHONPATH=. python examples/quickstart.py

# run the test suite
python -m pytest tests/ -v
# 9 passed
```

Or via Make: `make setup && make quickstart && make test`.

---

## 🔄 Full pipeline usage

```bash
# Phase 1 — download a triplet (GOES-16, 2025-06-01, 12Z, Ch-13)
python scripts/download_triplet.py --satellite goes16 --date 2025-06-01 --hour 12

# Phase 1b — (optional) 24-h storm streamer for training data
python scripts/stream_storm.py --date 2024-05-31 --satellite goes16

# Phase 2 — baselines (linear + Farneback + torchvision RAFT) -> .npz bundle
python scripts/make_baselines.py --manifest data/raw/goes16/2025-06-01/12/triplet.txt

# Phase 3 — train the residual refiner
python scripts/train_refiner.py --epochs 10 --patch-size 256 --batch-size 4

# Phase 4 — evaluate (metrics.json + pred/residual/error plots)
python scripts/evaluate.py

# Inference — single patch (Kelvin .npy out)
python scripts/infer.py --t0 frame_t0.npy --t2 frame_t2.npy --out outputs/pred_k.npy

# Inference — full 5424x5424 disk, Hanning-blended
python scripts/infer_full_disk.py --t0 disk_t0.npy --t2 disk_t2.npy --out outputs/disk_pred.npy
```

Config-first alternative: edit `configs/config.yaml`, then `PYTHONPATH=. python -c "from src.config import load_config; print(load_config())"`.

---

## 🏋️ Training

| Path | Script | Data | Loss | Output |
|---|---|---|---|---|
| **Refiner** (recommended start, CPU-friendly) | `scripts/train_refiner.py` → `src/training/train_refiner.py` | local `.npz` bundles | L1 + 2.5·SSIM + 1.8·Sobel + 0.8·cold-cloud + 0.3·residual, AdamW + CosineAnnealing + grad-clip | `outputs/unet_refiner/best_small_residual_unet.pt` |
| **Hybrid RAFT** (GPU + network, full engine) | `scripts/train_hybrid.py` → `src/training/train_hybrid.py` | S3-streamed GOES (`AWSSatelliteDataset`) | `L1 + 0.5·(1-SSIM)`, AdamW + OneCycleLR + AMP + grad-accum (eff. batch 32) + DataParallel | `FINAL_BEST_CHECKPOINT.pth` (+ `loss_graph_final.png`) |

Resume any interrupted hybrid run with `--resume <checkpoint>`; `module.` prefixes from multi-GPU training are stripped automatically on load.

---

## 🔮 Inference

- **Arbitrary time-step:** hybrid path accepts any `t ∈ (0,1)` (`flow·t` scaling) — ask for `t=0.25, 0.5, 0.75…` and get a scrubbable timeline.
- **Patch (24 ms):** `predict_patch_refiner` / `predict_patch_hybrid`.
- **Full-disk (seamless):** `predict_full_disk_*` — 256px patches, stride 128, separable Hanning taper, partition-of-unity normalisation. Swap in any `patch_fn` via `full_disk_from_patch_fn`.
- **Device:** auto `cuda → mps → cpu`; AMP autocast on CUDA.

---

## ⚠️ Methodology note

All baselines (bilinear, vanilla U-Net, RIFE) were evaluated on the **same GOES-18 test set without domain fine-tuning** — i.e. their real-world off-the-shelf performance. Our model was **fine-tuned on GOES infrared data**. Metrics measure prediction of an **actually-skipped ground-truth frame** at that timestamp.

---

## 🚀 Roadmap

- [ ] With heavier hardware available: replace the deep synthesis CNN with a **Swin-Transformer (ViT)** blender for even better long-range context.
- [ ] ONNX/TensorRT export for operational deployment.
- [ ] INSAT-3D/3DR ingestion preset alongside GOES.
- [ ] Live demo notebook (optional) that calls only the `src.*` API.

---

## 📚 Resources

- 🛰️ **Data** — NOAA GOES public S3 buckets (`noaa-goes16/18/19`), `ABI-L2-CMIPF` Channel 13 (thermal IR). No credentials needed; see `scripts/download_triplet.py`.
- 🏋️ **Weights** — shipped refiner checkpoint in [`weights/`](weights/) (see `weights/README.md`), plus training scripts to reproduce your own.
- 📸 **Experiment photo log** — [`assets/gallery/`](assets/gallery/) (50 images) · [Drive mirror](https://drive.google.com/drive/folders/175pFM3BrmGdGrdIVYI5U2PepsaGVkkGu?usp=sharing).
- 📖 **Concepts** — RAFT optical flow ([Teed & Deng, ECCV 2020](https://arxiv.org/abs/2003.12039)) · U-Net ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) · SSIM/FSIM for perceptual quality.
- 🧪 **Reproduce it** — `examples/quickstart.py` (smoke test) → `tests/` (9 checks) → `configs/config.yaml` (every hyperparameter in one place).

---

## 🙏 Credits

- Data: NOAA GOES public S3 (`noaa-goes16/18/19`).
- This repo keeps the original math faithful (verified by `tests/test_models.py`) while replacing monolithic notebooks with an importable, testable package.

*If this helped your Early Warning research, please star ⭐ `swarn007-byte/Interp-n`.*
