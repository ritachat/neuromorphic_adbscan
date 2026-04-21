# Neuromorphic ADBScan

**Weather-Robust Object Detection from DVS Event Cameras using Adaptive DBSCAN**

> Extension of **US Patent 10,510,154** (Rita Chattopadhyay, Intel Corporation)  
> from LiDAR point clouds to neuromorphic DVS pixel space — first known application.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Numba](https://img.shields.io/badge/Numba-JIT%20accelerated-green.svg)](https://numba.pydata.org/)
[![License: Research](https://img.shields.io/badge/license-Research-lightgrey.svg)](#license)

---

## Results at a Glance

| Pipeline | Weighted Avg | Heavy Rain | vs Baseline |
|----------|-------------|------------|-------------|
| DBSCAN (no filter) — baseline | 64.6% | 32% | — |
| DBSCAN + GB filter | 84.9% | 70% | +20.3 pp |
| **ADB-Opt + GB filter ★ BEST** | **91.2%** | **91%** | **+26.6 pp** |
| Dual-Opt (ε+k) + GB filter | 90.5% | 91% | +25.9 pp |
| ADB-Opt + DL filter | 89.9% | 88% | +25.3 pp |
| DBSCAN + DL filter | 87.6% | 86% | +23.0 pp |

*28 pipeline combinations (7 detectors × 4 filters) × 7 weather conditions.*  
*Machine-validated on DAVIS346 346×260 px synthetic stream.*

---

## What This Does

This pipeline detects and classifies moving objects — Car, Truck, Ball, Drone, Pedestrian, Cyclist — from a neuromorphic DVS (Dynamic Vision Sensor) event camera stream under adverse weather: rain (light / medium / heavy), snow (light / medium), and combined Rain+Snow.

**ADBScan** (Adaptive DBSCAN) adapts its search radius ε per event as a quadratic function of radial distance from the sensor centre, correcting for the measured 40% event-density drop from centre to frame edge. A Numba JIT-compiled BFS kernel provides a **4.1× speedup** over sklearn DBSCAN.

Two complementary weather filters are provided:
- **GB filter** — Gradient Boosting on 15 hand-crafted track features (width dominates at 71%)
- **DL filter** — EventResNet, a 984K-param 1D ResNet on raw event sequences (x, y, t, polarity)

A novel **polarity physics filter** (no training) detects rain by polarity flip-rate < 0.28 — a signal invisible to RGB cameras.

---

## Project Structure

```
neuromorphic_adbscan/
│
├── core/                           ← Source modules
│   ├── event_reader.py             Synthetic DAVIS346 stream generator
│   ├── noise_filter.py             Multi-scale Background Activity filter
│   ├── adbscan.py                  Numba JIT BFS clustering kernel
│   ├── detector.py                 Base EventDetector + Track/Detection classes
│   ├── detector_adbscan.py         ADBScanDetector  (adaptive ε, fixed k)
│   ├── dual_adaptive_detector.py   DualADBScanDetector (adaptive ε AND k — full patent)
│   ├── detector_3d.py              ADBScan3DDetector (x,y,t space)
│   ├── weather_noise.py            Rain + snow event simulation
│   ├── weather_filter.py           GB weather cluster filter (15 features)
│   ├── dl_weather_filter.py        EventResNet DL weather filter
│   ├── polarity_track_filter.py    Physics-based polarity filter
│   ├── classifier.py               MLP object classifier (6 classes)
│   └── _quiet.py                   Output suppression utility
│
├── verify_install.py               Quick smoke test — 8 checks, < 60 s
├── run_train_dl.py                 Train EventResNet once (~4 min CPU)
├── run_benchmark.py                All 28 combinations, full table + plots
├── run_single_pipeline.py          One pipeline interactively (--detector / --filter)
├── run_polarity_analysis.py        DVS polarity physics analysis + plots
│
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Installation

### 1. Clone

```bash
git clone https://github.com/rchattop/neuromorphic_adbscan.git
cd neuromorphic_adbscan
```

### 2. Create virtual environment + install

```bash
bash setup.sh
source venv/bin/activate
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 3. Verify

```bash
python verify_install.py
```

Expected output (< 60 s):
```
  [OK]  All 13 modules imported
  [OK]  Numba JIT compiled
  [OK]  30,207 events, 6 object classes
  [OK]  29,278 events after noise filter (97% kept)
  [OK]  ADBScan 2D — 12 tracks detected
  [OK]  DualADBScan (ε+k) — 11 tracks
  [OK]  Classifier + GB filter trained
  [OK]  DL filter loaded
  [OK]  ADB-Opt+GB = 88–91%  ✓
  All checks passed.
```

---

## Quick Start

### Run everything (all 28 pipelines)

```bash
# Step 1: train DL filter once (~4 min on CPU, saved for reuse)
python run_train_dl.py

# Step 2: full benchmark — all 28 pipeline × 7 weather conditions
python run_benchmark.py

# Output:
#   Terminal: full ranked table
#   results/benchmark_results.pkl
#   results/plots/ranking.png
#   results/plots/heatmap.png
```

### Run the best pipeline interactively

```bash
# List all options
python run_single_pipeline.py --list

# Best overall (91.2%)
python run_single_pipeline.py --detector ADB-Opt --filter GB

# Full patent implementation (90.5%)
python run_single_pipeline.py --detector Dual-Opt --filter GB

# DL filter only (89.9%)
python run_single_pipeline.py --detector Dual-Opt --filter DL

# Standard DBSCAN + DL (87.6%)
python run_single_pipeline.py --detector DBSCAN --filter DL
```

### Polarity physics analysis

```bash
python run_polarity_analysis.py
# → results/plots/polarity_analysis.png
```

---

## Available Detectors

| Name | Dim | k | Description |
|------|-----|---|-------------|
| `DBSCAN` | 2D | fixed | Standard DBSCAN, flat ε=10 px |
| `ADB-Hard` | 2D | ε-only | ADBScan, patent default coefficients (c₂=0.0001) |
| `ADB-Opt` | 2D | ε-only | ADBScan, **optimal coefficients** (c₂=0.0002) ★ |
| `Dual-Hard` | 2D | ε+k | Full US 10,510,154, patent default |
| `Dual-Opt` | 2D | ε+k | **Full US 10,510,154, optimal** — true patent implementation |
| `3D-DBSCAN` | 3D | fixed | DBSCAN in (x,y,t) spatiotemporal space |
| `3D-ADB-Opt` | 3D | ε-only | ADBScan in (x,y,t) space |

## Available Filters

| Name | Description | Training? |
|------|-------------|-----------|
| `none` | No filter — raw detector output | No |
| `GB` | Gradient Boosting, 15 track shape features | Yes (auto) |
| `DL` | EventResNet 1D CNN, raw event sequences | Yes (run_train_dl.py) |
| `GB+DL` | DL first, then GB (sequential) | Yes (both) |

---

## The Algorithm

### Adaptive ε (US Patent 10,510,154)

Standard DBSCAN uses a global fixed ε. ADBScan adapts ε per event as a function of radial distance d from the sensor centre (cx, cy):

```
d(event)  =  √[(x − cx)² + (y − cy)²]
ε(event)  =  clip(c₂·d² + c₁·d + base,  min = 0.9 px)
```

**Optimal coefficients** (grid-searched over 7 weather conditions):

| Coefficient | Value | Meaning |
|-------------|-------|---------|
| c₂ | 0.0002 | curvature of the ε(d) parabola |
| c₁ | −0.030 | slope — vertex position |
| base | 9.5 px | ε at sensor centre |
| Vertex | d = 75 px | minimum ε location |
| ε_min | 8.37 px | smallest search radius |

**Physical justification:** Event density drops ~40% from sensor centre (k≈42 neighbours) to frame edge (k≈25 neighbours) — measured across 40 windows of the DAVIS346 stream. Adaptive ε compensates exactly for this sparsity.

### Dual Adaptive (full patent — ε AND k)

```
k_local(d) = k_centre·(1 − d/d_max) + k_edge·(d/d_max)
k(event)   = max(2, round(k_base · k_centre / k_local(d)))
```

With k_centre=42, k_edge=25 (measured), k_base=5.

### Numba JIT BFS

The BFS inner loop is compiled to native machine code via `@njit(cache=True)`, using a CSR (Compressed Sparse Row) neighbour graph. This gives a **2,300× speedup** over pure-Python BFS and **4.1×** over sklearn DBSCAN on the full pipeline.

### Polarity Physics Filter (novel — no training)

Rain clusters have **flip_rate < 0.28** (measured mean = 0.095):
- A raindrop moves in one direction → its leading face fires sustained ON events, trailing face fires brief OFF batch → only 1 polarity transition per streak.

Real objects have **flip_rate ≈ 0.47**:
- All 4 bounding box edges fire simultaneously → ON and OFF events intermix every window.

Snow is detected by density rules: `n_events < 250 AND n_windows < 6 AND events/window < 30`.

This signal is **invisible to RGB cameras** — polarity encodes the sign of brightness change, not available in frame-based sensors.

---

## Object Classes

| Class | Size (px) | Speed | Motion |
|-------|-----------|-------|--------|
| Car | 55×22 | 3.5 px/ms | Straight |
| Pedestrian | 16×48 | 0.7 px/ms | Bounce |
| Cyclist | 26×34 | 1.8 px/ms | Straight |
| Ball | 12×12 | 3.2 px/ms | Arc |
| Drone | 13×10 | 2.5 px/ms | Zigzag |
| Truck | 80×30 | 1.5 px/ms | Straight |

---

## Key Findings

1. **ADB-Opt 2D + GB = 91.2%** — best of all 28 combinations. Adaptive ε pre-separates rain clusters spatially; GB's width feature (71% importance) then cleanly classifies them.

2. **GB beats DL for ADBScan; DL beats GB for DBSCAN.** Adaptive ε creates geometrically clean clusters that GB exploits. Flat DBSCAN ε cannot pre-separate clusters, so DL's polarity-sequence reading provides the edge (+2.7% for DBSCAN).

3. **2D always outperforms 3D** — by 8–23 pp. The 5 ms window already provides clean temporal separation; adding the time axis introduces noise sensitivity.

4. **Full patent Dual (ε+k) = 90.5%** — only 0.7 pp below ε-only Opt. Both are publishable results.

5. **GB+DL combined = no gain over DL alone.** DL removes all weather sequences; GB finds nothing left to filter.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: numba` | `pip install numba` |
| `ModuleNotFoundError: torch` | `pip install torch` |
| Slow first run (~2 s pause) | Normal — Numba compiles BFS once, cached after |
| `DL model not found` | Run `python run_train_dl.py` first |
| sklearn joblib warnings | Already suppressed in all scripts |
| Apple Silicon pip torch error | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Windows venv error | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## Citation

If you use this code, please cite the original patent:

```bibtex
@misc{chattopadhyay2019adbscan,
  title  = {Systems and Methods for Adaptive DBSCAN Clustering},
  author = {Chattopadhyay, Rita},
  year   = {2019},
  note   = {US Patent 10,510,154, Intel Corporation}
}
```

---

## License

This implementation is provided for **research and educational purposes**.  
The underlying ADBScan algorithm is protected by **US Patent 10,510,154** (Intel Corporation).  
Commercial use requires a licence from Intel Corporation.
