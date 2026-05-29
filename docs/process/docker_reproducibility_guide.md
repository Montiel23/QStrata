# QStrata Docker Reproducibility Guide

**Author:** Miguel Lopez (QStrata)  
**Date:** 2026-05-28  
**Slice:** Q38D  

---

## Purpose

This guide defines the canonical Docker-based execution environment for QStrata
experiments. All training scripts, NAS runs, and preprocessing benchmarks must be
executed inside one of the two defined containers — never against the host Python
environment. The host runs Python 3.13 without PyTorch and is not a valid
execution target.

---

## Two Containers, One Mount Point

| Container | Compose file | Dockerfile | Python | PyTorch | Purpose |
|-----------|-------------|------------|--------|---------|---------|
| `qstrata-eda` | `docker-compose.yml` | `Dockerfile` (CPU) | 3.11-slim | CPU-only | EDA, notebooks, data prep |
| `docker-qstrata-gpu-1` | `docker-compose.gpu.yml` | `Dockerfile.gpu` | 3.10 | 2.2.2+cu121 | All training experiments |

Both containers mount the project root to `/workspace` (read-write). All scripts
reference `/workspace/...` paths.

---

## Starting Containers

All compose commands must be run from `infra/docker/`.

```bash
# EDA / notebook container (CPU)
cd infra/docker
docker compose -f docker-compose.yml up -d

# GPU training container
cd infra/docker
docker compose -f docker-compose.gpu.yml up -d
```

Verify containers are running:
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

---

## Canonical GPU Training Command

For any training script under `scripts/`:

```bash
# Foreground (output to terminal)
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/<script_name>.py [args]

# Background (nohup, output to log file) — recommended for long runs
docker exec docker-qstrata-gpu-1 bash -c \
    "cd /workspace && nohup python3 scripts/<script_name>.py \
     > experiments/logs/<run_name>.log 2>&1 & echo PID:\$!"
```

Monitor a background run:
```bash
# Check if process is alive
docker exec docker-qstrata-gpu-1 pgrep -f <script_name>

# Tail live log (nohup buffers; lines appear as Python buffer fills)
tail -f /home/mike/research-projects/QStrata/experiments/logs/<run_name>.log

# Count completed combos in a sweep (lines containing "] done")
grep -c "\] done" experiments/logs/<run_name>.log
```

---

## Canonical CPU (EDA) Command

```bash
docker exec qstrata-eda python3 /workspace/scripts/<script_name>.py [args]
```

For Jupyter Lab, access via `http://localhost:8888` (CPU) or `http://localhost:8889` (GPU).

---

## Mount Expectations

### Project root
Both containers bind-mount the repo:

| Host path | Container path | Access |
|-----------|---------------|--------|
| `/home/mike/research-projects/QStrata` | `/workspace` | `rw` |

All script output (leaderboards, results, logs, reports) is written inside
`/workspace` and is immediately visible on the host filesystem.

### Datasets (GPU container)
| Host path | Container path | Access |
|-----------|---------------|--------|
| `/media/mike/Datasets/vindr-spinexr` | `/datasets/vindr-spinexr` | `ro` |
| `/workspace/data/processed/vindr_binary_roi_224` | same | `rw` (within workspace) |

Scripts reference the processed dataset at `/workspace/data/processed/vindr_binary_roi_224`.
The raw mount at `/datasets/vindr-spinexr` is the upstream source; do not write to it.

### Checkpoints
Checkpoints live inside the workspace:

| Path | Description |
|------|-------------|
| `/workspace/checkpoints/c006_d040_classical_anchor.pt` | Frozen C006-D040 backbone (canonical for Phase 6b) |

### Output paths (must be writable)
| Path | Contents |
|------|---------|
| `/workspace/experiments/leaderboards/` | Per-slice CSV leaderboards |
| `/workspace/experiments/results/` | JSON summaries |
| `/workspace/experiments/logs/` | Training run logs |
| `/workspace/reports/` | Markdown experiment reports |

---

## PYTHONPATH

The GPU compose sets `PYTHONPATH=/workspace`. Scripts that import sibling
scripts (e.g., Q38C imports from Q38A via `sys.path.insert`) rely on this.

The CPU compose does **not** set `PYTHONPATH`. Scripts requiring cross-script
imports must use `sys.path.insert(0, "/workspace/scripts")` explicitly.

---

## Environment Self-Check

Before starting a new experiment, verify the container environment:

```bash
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/check_qstrata_docker_env.py
```

Expected output: all checks PASS, numpy ABI warning is a known non-blocking issue
(see Drift Risks below).

---

## Known Drift Risks

### 1. NumPy ABI mismatch (GPU container)

**Symptom:** `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`
at `import torch`.

**Cause:** `Dockerfile.gpu` pins `"numpy<2"` but the live container has
`numpy==2.2.6`. When the image was rebuilt (or `pip` was run inside the
container), pip resolved numpy to 2.x despite the `<2` intent.

**Impact:** Non-blocking. PyTorch falls back gracefully; all training,
inference, and metric computation work correctly. The warning appears at startup.

**Fix (when rebuilding the image):** Change `"numpy<2"` to `"numpy<2.0"` in
`Dockerfile.gpu` and rebuild:
```bash
cd infra/docker
docker compose -f docker-compose.gpu.yml build --no-cache
```

### 2. CPU vs GPU Python version skew

CPU container uses Python 3.11; GPU container uses Python 3.10. Scripts must
not depend on 3.11-only language features if they are intended to run in both
containers. All current Phase 6b scripts target 3.10+ syntax.

### 3. requirements.txt is not the build source of truth

`requirements.txt` lists only `medmnist`, `torch`, `torchvision` with no
version pins and no wheel index. It cannot reproduce either container
environment on its own. The Dockerfiles are the single source of truth for
the execution environment. Do not use `pip install -r requirements.txt` as a
substitute for running inside the container.

### 4. GPU compose build context is `infra/docker/`, not project root

`docker-compose.gpu.yml` sets `context: .` (relative to `infra/docker/`).
This means files outside `infra/docker/` (including `requirements.txt` and
`scripts/`) are not available during `docker build`. The Dockerfile.gpu
installs all packages directly without copying project files. This is correct
for the current setup — do not add a `COPY` step that would bake source code
into the image.

### 5. Dataset path hardcoded in GPU compose

`docker-compose.gpu.yml` hardcodes `/media/mike/Datasets/vindr-spinexr` as
the raw dataset source. On a different host, this path must be updated before
the compose file will work. The CPU compose uses `${DATASET_PATH:-../../data}`
via `.env`, which is more portable. Consider aligning the GPU compose to use an
env var if multi-host reproducibility is needed.

---

## Rebuilding Images

```bash
# CPU image
cd infra/docker
docker compose -f docker-compose.yml build

# GPU image (may take 10–20 min for PyTorch wheel download)
cd infra/docker
docker compose -f docker-compose.gpu.yml build
```

After rebuilding, re-run the self-check:
```bash
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/check_qstrata_docker_env.py
```

---

## What NOT to Do

- **Do not** run training scripts with the host Python (`python3 scripts/...` from the terminal). Host Python 3.13 has no PyTorch.
- **Do not** `pip install` packages inside the running container without also updating the Dockerfile. Changes to a running container are lost on restart.
- **Do not** use `requirements.txt` alone to reconstruct the environment.
- **Do not** write output files outside `/workspace` from inside the container — they will not be visible on the host and will be lost on restart.
