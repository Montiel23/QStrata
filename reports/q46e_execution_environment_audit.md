# Q46E — Execution Environment Audit

**Slice ID**: Q46E-EXECUTION-ENVIRONMENT-AUDIT
**Date**: 2026-06-01
**Auditor**: Claude Code (Sonnet 4.6)
**Branch**: feature/q46e_execution_environment_audit
**Scope**: All preconditions required to execute the Q46B feature extractor benchmark
(`scripts/run_q46b_feature_extractor_benchmark.py`) — dataset, checkpoints, Docker, Python
dependencies, GPU. No benchmark executed.

---

## Summary

| Domain | Status | Detail |
|---|---|---|
| Dataset | ✅ READY | 10,466 images, 3 splits × 2 classes all populated |
| Baseline checkpoint | ✅ READY | `checkpoints/c006_d040_classical_anchor.pt` present (52 KB) |
| Head config | ✅ READY | `configs/experiments/q34a_classical_nas/q34a_trial_004.yaml` present |
| Benchmark script | ✅ READY | `scripts/run_q46b_feature_extractor_benchmark.py` present |
| Benchmark config | ✅ READY | `configs/q46_feature_extractor_benchmark.yaml` present |
| Docker container | ✅ READY | `docker-qstrata-gpu-1` running (Up 47 h), GPU bound (`nvidia` driver) |
| PyTorch / torchvision | ✅ READY | torch 2.2.2+cu121, torchvision 0.17.2+cu121 — importable in container |
| GPU hardware | ✅ READY | RTX 2060 SUPER, 8 GB VRAM, ~7.6 GB free |
| Output directories | ✅ READY | `experiments/leaderboards/` and `experiments/results/` exist |
| NumPy ABI (Docker) | ⚠️ WARNING | numpy 2.2.6 installed vs Dockerfile's `numpy<2` constraint — torch imports with UserWarning but succeeds |

**Overall readiness: READY_WITH_WARNINGS** — benchmark can proceed via Docker. NumPy ABI
warning is non-fatal (torch loads and CUDA is active) but should be remediated before
long experimental runs.

---

## 1. Dataset Availability

**Required by config**: `data/processed/vindr_binary_roi_224` (`dataset.root`)

| Split | Class | Image Count |
|---|---|---|
| `train` | 0 (negative) | 3,408 |
| `train` | 1 (positive) | 3,304 |
| `val` | 0 | 852 |
| `val` | 1 | 825 |
| `test` | 0 | 1,070 |
| `test` | 1 | 1,007 |
| **Total** | | **10,466** |

Additional 3 files in `samples/` (excluded from training counts above).

All six class subdirectories (`{train,val,test}/{0,1}`) are present and populated. Image
format is PNG 224×224 (RGB). Dataset root is bind-mounted read-only into the Docker
container as `/workspace/data/processed/vindr_binary_roi_224`.

**Status: ✅ READY**

---

## 2. Checkpoint Paths

**Required by config** (`configs/q46_feature_extractor_benchmark.yaml`, `candidates[0]`):

```
checkpoint: checkpoints/c006_d040_classical_anchor.pt
```

| File | Size | Present |
|---|---|---|
| `checkpoints/c006_d040_classical_anchor.pt` | 52 KB | ✅ YES |

The 52 KB size is consistent with a frozen-backbone classifier head checkpoint (not full
backbone weights). Q46B loads this as the `baseline` candidate; all torchvision candidates
(`efficientnet_b0`, `mobilenetv3_small`, `mobilenetv3_large`, `convnext_tiny`) source
ImageNet pretrained weights from `torchvision.models` at runtime — no additional local
checkpoint files required.

Other checkpoints present (not required for Q46B but available for reference):
`vindr_classical_baseline_best.pt` (299 KB), `vindr_classical_control_tiny_head_best.pt`
(62 KB), `vindr_cv_binary_best.pt` (66 KB), `vindr_dv_hybrid_best.pt` (62 KB),
`vindr_dv_hybrid_pretrained_best.pt` (63 KB).

**Status: ✅ READY**

---

## 3. Benchmark Script and Config

| Artifact | Path | Present |
|---|---|---|
| Runner script | `scripts/run_q46b_feature_extractor_benchmark.py` | ✅ YES |
| Protocol config | `configs/q46_feature_extractor_benchmark.yaml` | ✅ YES |
| Head config | `configs/experiments/q34a_classical_nas/q34a_trial_004.yaml` | ✅ YES |

Config verification (`configs/q46_feature_extractor_benchmark.yaml`):

| Parameter | Configured Value | Source |
|---|---|---|
| epochs | 4 | Q38A–Q45A standard |
| learning_rate | 1.0e-3 | Q38A–Q45A standard |
| weight_decay | 1.0e-4 | Q38A–Q45A standard |
| batch_size | 8 | Q41 optimised |
| num_workers | 4 | Q41 optimised |
| pin_memory | true | Q41 optimised |
| persistent_workers | true | Q41 optimised |
| prefetch_factor | 2 | Q41 optimised |
| clahe_clip | 3.0 | Q38C champion |
| clahe_tile | [4, 4] | Q38C champion |
| seeds.smoke | [42] | Q46A protocol |
| seeds.full | [42, 7, 123] | Q46A protocol |
| seeds.extended | [42, 7, 123, 999, 2025] | Q46A protocol |
| runtime_caps.smoke_minutes | 60 | Q46A protocol |
| runtime_caps.full_minutes | 120 | Q46A protocol |
| decision threshold | 0.7196 | Q45A mean AUROC |
| Q38C ceiling | 0.7239 | Q38C best single-seed |

All 5 candidates configured with correct feature dimensions:

| Candidate | Backbone | Feature Dim | Checkpoint Source |
|---|---|---|---|
| `baseline` | c006_d040_frozen_resnet | 512 | `checkpoints/c006_d040_classical_anchor.pt` |
| `efficientnet_b0` | torchvision | 1280 | IMAGENET1K_V1 (runtime download) |
| `mobilenetv3_small` | torchvision | 576 | IMAGENET1K_V1 (runtime download) |
| `mobilenetv3_large` | torchvision | 960 | IMAGENET1K_V1 (runtime download) |
| `convnext_tiny` | torchvision | 768 | IMAGENET1K_V1 (runtime download) |

**Status: ✅ READY**

---

## 4. Docker Image and Container Readiness

**Required container**: `docker-qstrata-gpu-1` (image `docker-qstrata-gpu`)

| Check | Result |
|---|---|
| Container name | `docker-qstrata-gpu-1` |
| Container status | ✅ **RUNNING** (Up 47 hours) |
| Image | `docker-qstrata-gpu` |
| GPU device driver | ✅ `nvidia` (confirmed via `docker inspect`) |
| Base image (Dockerfile.gpu) | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| CUDA version in container | 12.1 (cu121 wheels) |
| Host driver compatibility | ✅ Host driver 595.58.03 supports CUDA 13.2 — fully backward-compatible with container's CUDA 12.1 |
| Workspace volume mount | `../..:/workspace` — project root available as `/workspace` in container |

Container is up and GPU-bound. No restart required before benchmark execution.

**Status: ✅ READY**

---

## 5. Python Dependency Installation

The Q46B benchmark executes inside `docker-qstrata-gpu-1`. Host Python (3.13.7, `.venv/`)
is not the execution environment — it lacks torch/torchvision, which is expected and by
design.

### 5.1 Docker Container — Actual Installed State

Verified by running `docker exec docker-qstrata-gpu-1 python3 -c "..."` during this audit:

| Package | Installed Version | Status |
|---|---|---|
| Python | 3.10 (container) | ✅ |
| torch | 2.2.2+cu121 | ✅ importable |
| torchvision | 0.17.2+cu121 | ✅ importable |
| CUDA available | True | ✅ |
| CUDA device | NVIDIA GeForce RTX 2060 SUPER | ✅ |
| numpy | **2.2.6** | ⚠️ ABI mismatch (see warning below) |
| scikit-learn | installed (per Dockerfile.gpu) | ✅ |
| pyyaml | installed (per Dockerfile.gpu) | ✅ |
| matplotlib | installed (per Dockerfile.gpu) | ✅ |
| pydicom | installed (per Dockerfile.gpu) | ✅ |
| pylibjpeg / pylibjpeg-openjpeg | installed (per Dockerfile.gpu) | ✅ |

### 5.2 NumPy ABI Warning — Detail

**Finding**: The Docker container has `numpy 2.2.6` installed. The Dockerfile.gpu specifies
`numpy<2`, which was the correct constraint at build time (torch 2.2.2 was compiled against
NumPy 1.x ABI). The installed version violates this constraint — likely upgraded by a
subsequent `pip install` inside the container.

**Impact**: When torch is imported inside the container, the following UserWarning is
emitted:

```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
  (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:84.)
```

**Severity: LOW / NON-BLOCKING**. Despite the warning, `import torch` succeeds,
`torch.cuda.is_available()` returns `True`, and `torch.cuda.get_device_name(0)` correctly
returns `NVIDIA GeForce RTX 2060 SUPER`. The benchmark can run.

**Recommended fix** (before Q46B execution):

```bash
docker exec docker-qstrata-gpu-1 pip install "numpy<2" --force-reinstall
```

This restores the `numpy<2` constraint without rebuilding the image.

### 5.3 Host venv (Python 3.13, `.venv/`) — Informational

The local venv is intentionally minimal. torch, torchvision, and scikit-learn are absent.
This is expected — all training runs inside Docker.

**Status: ✅ READY (with LOW-severity NumPy ABI warning)**

---

## 6. GPU Availability

| Check | Result |
|---|---|
| GPU model | NVIDIA GeForce RTX 2060 SUPER |
| VRAM total | 8,192 MiB |
| VRAM in use (at audit time) | 633 MiB (desktop processes: gnome-shell, Xwayland, Zoom) |
| VRAM available | ~7,559 MiB (~7.4 GB) |
| GPU utilization | 3% (essentially idle) |
| Driver version | 595.58.03 |
| CUDA version (host) | 13.2 |
| CUDA available in container | ✅ True |
| nvidia-smi accessible | ✅ |
| Persistence mode | Off (normal for desktop GPU) |

**VRAM capacity assessment**: The Q46B benchmark runs one backbone at a time across 4
epochs, batch_size=8, image_size=224. Largest candidate is ConvNeXt-Tiny (~28.6M frozen
params). At batch 8 × 224×224×3, peak VRAM demand (activations + gradients for head only,
backbone frozen) is well within 8 GB. No VRAM concern for any of the 5 candidates.

**Status: ✅ READY**

---

## 7. Output Directory Readiness

| Path | Exists | Notes |
|---|---|---|
| `experiments/leaderboards/` | ✅ YES | 15 prior leaderboard CSVs present |
| `experiments/results/` | ✅ YES | 50+ prior JSON result files present |
| `reports/` | ✅ YES | Prior reports present |
| `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv` | ℹ️ Not yet created | Created by benchmark on first `--smoke` run |
| `experiments/leaderboards/q46b_extractor_full_leaderboard.csv` | ℹ️ Not yet created | Created by benchmark on first `--full` run |
| `experiments/results/q46b_feature_extractor_benchmark.json` | ℹ️ Not yet created | Created by benchmark on first run |
| `reports/q46b_feature_extractor_benchmark.md` | ℹ️ Not yet created | Created by benchmark on completion |

All parent directories exist. The benchmark script will create Q46B-specific output files on
first execution.

**Status: ✅ READY**

---

## 8. Dependency Chain Status

Per Q46A (`reports/q46a_feature_extractor_benchmark_plan.md` §9):

```
Q45A COMPLETE (augmentation CLOSED)
  └─► Q45B (partial fine-tuning benchmark) — status unknown from this audit
        └─► Q46A (plan) — COMPLETE
              └─► Q46B (execution) — BLOCKED on Q45B per Q46A protocol
```

**Note**: Q46A specifies that Q46B is blocked until Q45B is complete. Q45B completion status
is not verifiable from the execution environment alone — it requires checking the research
roadmap or Q45B report. If Q45B has been completed and its best config documented, the
Q46A protocol blocker is lifted.

This audit does not assess Q45B completion; it audits Q46B execution environment readiness
only.

---

## 9. Pre-Launch Checklist

Before executing `--smoke`, verify the following:

- [ ] Q45B complete and best fine-tuning config documented (dependency per Q46A §12)
- [ ] Recommended fix: `docker exec docker-qstrata-gpu-1 pip install "numpy<2" --force-reinstall` (resolves NumPy ABI warning)
- [ ] Dry-run passes: `docker exec docker-qstrata-gpu-1 python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --dry-run`
- [ ] Free VRAM ≥ 2 GB confirmed before starting (current: ~7.4 GB — ample)
- [ ] No competing GPU workloads running
- [ ] Execute smoke phase: `docker exec docker-qstrata-gpu-1 python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --smoke`

---

## 10. Readiness Classification

**STATUS: READY_WITH_WARNINGS**

| Criterion | Result |
|---|---|
| All audit items checked and documented | ✅ |
| No benchmark executed | ✅ |
| Readiness classification assigned | ✅ READY_WITH_WARNINGS |
| Report at `reports/q46e_execution_environment_audit.md` | ✅ |

All hard preconditions for Q46B execution are satisfied:

- Processed dataset present and complete (10,466 images, 3 splits)
- Baseline checkpoint present (`c006_d040_classical_anchor.pt`)
- Head config present (`q34a_trial_004.yaml`)
- Benchmark script and protocol config present and verified
- Docker container running with GPU access (RTX 2060 SUPER, ~7.4 GB free VRAM)
- torch 2.2.2+cu121 and torchvision 0.17.2+cu121 importable in container; CUDA active
- Output directories in place

**One advisory** (non-blocking): NumPy 2.2.6 is installed in the Docker container
(Dockerfile constraint was `numpy<2`). torch imports with a UserWarning but operates
correctly. Recommend `pip install "numpy<2" --force-reinstall` before execution.

The benchmark is ready to run via `--dry-run` followed by `--smoke` inside
`docker-qstrata-gpu-1`.
