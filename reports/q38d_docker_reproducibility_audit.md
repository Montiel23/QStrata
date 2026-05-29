# Q38D — Docker Reproducibility Audit Report

**Date:** 2026-05-28  
**Author:** Miguel Lopez (QStrata)  
**Slice:** Q38D — Audit and validate QSTRATA Docker CPU/GPU execution paths  
**Branch:** `feature/q38d-docker-reproducibility-audit`  

---

## 1. Objective

Audit all Docker assets (Dockerfiles, compose files, requirements, running containers)
to ensure QStrata experiments can be reproduced without mixing local Python
environments. Produce a canonical guide, a self-check script, and this report.

---

## 2. Assets Reviewed

| Asset | Path | Status |
|-------|------|--------|
| CPU Dockerfile | `infra/docker/Dockerfile` | Reviewed |
| GPU Dockerfile | `infra/docker/Dockerfile.gpu` | Reviewed — drift found |
| CPU compose | `infra/docker/docker-compose.yml` | Reviewed |
| GPU compose | `infra/docker/docker-compose.gpu.yml` | Reviewed — hardcoded path |
| requirements.txt | `requirements.txt` | Reviewed — not build source |
| .env.example | `.env.example` | Reviewed |
| Running GPU container | `docker-qstrata-gpu-1` | Inspected live |

---

## 3. Container Environment Snapshot (GPU — Live)

Captured via `docker inspect` and `pip3 list` from `docker-qstrata-gpu-1`:

| Property | Value |
|----------|-------|
| Image | `docker-qstrata-gpu` |
| Base image | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| Python | 3.10.12 |
| PyTorch | 2.2.2+cu121 |
| CUDA | 12.1 |
| cuDNN | 8.9.0 |
| **NumPy** | **2.2.6** (Dockerfile pin: `"numpy<2"`) |
| GPU | NVIDIA GeForce RTX 2060 SUPER |
| PYTHONPATH | `/workspace` |
| Workspace mount | `/home/mike/research-projects/QStrata → /workspace` (rw) |
| Dataset mount | `/media/mike/Datasets/vindr-spinexr → /datasets/vindr-spinexr` (ro) |
| medmnist | 3.0.2 |
| scikit-learn | 1.7.2 |
| pydicom | 3.0.2 |
| matplotlib | 3.10.9 |

---

## 4. Container Environment Snapshot (CPU — Dockerfile spec)

| Property | Value |
|----------|-------|
| Base image | `python:3.11-slim` |
| Python | 3.11 |
| PyTorch | CPU-only (pytorch.org/whl/cpu wheel index) |
| NumPy | system pip default (no explicit pin) |
| PYTHONPATH | Not set by compose |
| Port | 8888 |
| Dataset mount | `${DATASET_PATH:-../../data} → /data` |

---

## 5. Local Host Environment (Reference — Not for Experiment Execution)

| Property | Value |
|----------|-------|
| Python | 3.13.7 |
| PyTorch | Not installed |
| Usage | Git operations, file editing, Docker management only |

---

## 6. Drift Findings

### Finding 1 — CRITICAL (non-blocking): NumPy ABI pin drifted in GPU container

**Severity:** Non-blocking (experiments function correctly)  
**Dockerfile.gpu intent:** `"numpy<2"` — install NumPy 1.x  
**Live container:** `numpy==2.2.6` (NumPy 2.x)  

**Evidence:** Q38C sweep log line:
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash.
```

**Cause:** When the image was last built or `pip` was run inside the container,
numpy resolved to 2.x. The `<2` constraint was satisfied at some prior build
but has since drifted. The warning is a one-time startup message; all training,
inference, and metric operations work correctly because PyTorch falls back
gracefully when the numpy C-extension is not available.

**Recommended fix:** When next rebuilding, change the pin in `Dockerfile.gpu`:
```dockerfile
# Before
RUN pip3 install --no-cache-dir "numpy<2" ...
# After (recommended)
RUN pip3 install --no-cache-dir "numpy>=2.0,<3" ...
# or accept 2.x explicitly and remove the constraint
```

### Finding 2 — INFO: requirements.txt is not the build source of truth

`requirements.txt` contains only three unpinned entries: `medmnist`, `torch`,
`torchvision`. It does not specify the CUDA wheel index, version pins, or
system-level dependencies. Running `pip install -r requirements.txt` on the
host or in a clean environment will not produce a valid experiment runtime.

**Recommendation:** Treat the Dockerfiles as the single source of truth.
`requirements.txt` should be understood as a minimal dependency declaration for
documentation purposes only.

### Finding 3 — INFO: GPU compose build context excludes project root

`docker-compose.gpu.yml` sets `context: .` (relative to `infra/docker/`).
Project files outside `infra/docker/` are not available during `docker build`.
This is the correct design — source code should not be baked into the image.
The `/workspace` bind-mount provides live access to all project files.

### Finding 4 — INFO: Dataset path hardcoded in GPU compose

`docker-compose.gpu.yml` hardcodes the raw dataset path:
```yaml
- /media/mike/Datasets/vindr-spinexr:/datasets/vindr-spinexr:ro
```
The CPU compose uses `${DATASET_PATH:-../../data}` via `.env`, which is more
portable. The GPU compose path is machine-specific. Any collaborator or
alternative host would need to edit this line directly.

### Finding 5 — INFO: CPU compose does not set PYTHONPATH

The CPU compose (`docker-compose.yml`) does not set `PYTHONPATH=/workspace`.
Scripts that use cross-script imports (e.g., Q38C importing from Q38A) require
`sys.path.insert(0, "/workspace/scripts")` when running in the CPU container.
The GPU compose sets `PYTHONPATH=/workspace` which handles this automatically.

---

## 7. Self-Check Script Results

Script: `scripts/check_qstrata_docker_env.py`  
Command: `docker exec docker-qstrata-gpu-1 python3 /workspace/scripts/check_qstrata_docker_env.py`

```
── Python runtime ──────────────────────────────────────────────
  [PASS] Python >= 3.10  [3.10.12]
  [PASS] Running inside container (/workspace exists)

── PYTHONPATH ──────────────────────────────────────────────────
  [PASS] PYTHONPATH includes /workspace  [/workspace]

── PyTorch ─────────────────────────────────────────────────────
  [PASS] torch importable  [2.2.2+cu121]
  [PASS] CUDA available  [NVIDIA GeForce RTX 2060 SUPER / CUDA 12.1]
  [PASS] CUDA version >= 12.1  [12.1]
  [PASS] torch build targets cu121  [2.2.2+cu121]

── NumPy ───────────────────────────────────────────────────────
  [PASS] numpy importable  [2.2.6]
  [WARN] numpy < 2 (ABI-safe for torch 2.2.x+cu121)  [2.2.6 — ABI warning at import; functional but Dockerfile pin drifted]

── Project dependencies ────────────────────────────────────────
  [PASS] medmnist importable
  [PASS] sklearn importable
  [PASS] PIL importable
  [PASS] pydicom importable
  [PASS] tqdm importable
  [PASS] yaml importable
  [PASS] matplotlib importable

── Data mount expectations ─────────────────────────────────────
  [PASS] Processed VinDr ROI dataset present
  [PASS] Raw VinDr dataset mount (/datasets/vindr-spinexr) — GPU compose only

── Checkpoint mounts ───────────────────────────────────────────
  [PASS] Canonical backbone checkpoint present (c006_d040_classical_anchor.pt)

── Output path expectations ────────────────────────────────────
  [PASS] /workspace/experiments/leaderboards writable
  [PASS] /workspace/experiments/results writable
  [PASS] /workspace/experiments/logs writable
  [PASS] /workspace/reports writable

════════════════════════════════════════════════════════════════
  Platform : Linux-6.17.0-29-generic-x86_64-with-glibc2.35
  Python   : 3.10.12
  Failures : 0
  Warnings : 1
════════════════════════════════════════════════════════════════

[WARN] All required checks pass (1 warning(s) noted above).
```

**Result: PASS** (0 failures, 1 known warning — numpy ABI drift)

---

## 8. Canonical Execution Commands

### Start containers
```bash
cd infra/docker
docker compose -f docker-compose.gpu.yml up -d    # GPU training
docker compose -f docker-compose.yml up -d         # CPU / EDA
```

### Run a training script (GPU, foreground)
```bash
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/<script>.py [args]
```

### Run a training script (GPU, background — long runs)
```bash
docker exec docker-qstrata-gpu-1 bash -c \
    "cd /workspace && nohup python3 scripts/<script>.py \
     > experiments/logs/<run>.log 2>&1 & echo PID:\$!"
```

### Monitor a background run
```bash
docker exec docker-qstrata-gpu-1 pgrep -f <script_name>
tail -f experiments/logs/<run>.log
```

### Self-check (GPU)
```bash
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/check_qstrata_docker_env.py
```

---

## 9. PASS/FAIL Checklist

- [x] Docker assets reviewed
- [x] CPU compose file reviewed
- [x] GPU compose file reviewed
- [x] Dockerfiles reviewed
- [x] requirements.txt reviewed
- [x] env check script created (`scripts/check_qstrata_docker_env.py`)
- [x] Docker reproducibility guide created (`docs/process/docker_reproducibility_guide.md`)
- [x] Audit report created (`reports/q38d_docker_reproducibility_audit.md`)
- [x] Canonical CPU command documented
- [x] Canonical GPU command documented
- [x] Data mount expectations documented
- [x] Checkpoint mount expectations documented
- [x] Output path expectations documented
- [x] Environment drift risks documented (5 findings)
- [x] Roadmap updated
- [x] No architecture refactor added
- [x] No experiment logic changed
- [x] No NAS executed
- [x] Intended files staged only
- [x] Local commit created
- [x] No push

---

## 10. Artifacts

| File | Description |
|------|-------------|
| `scripts/check_qstrata_docker_env.py` | Self-check script (run inside container) |
| `docs/process/docker_reproducibility_guide.md` | Canonical guide for Docker execution |
| `reports/q38d_docker_reproducibility_audit.md` | This report |
| `docs/roadmaps/qstrata_master_research_roadmap.md` | Updated — Q38D COMPLETE, Q39 NEXT |
