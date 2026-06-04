# Q46 Feature Extractor Benchmark — Failure Recovery Playbook

**Slice ID**: Q47-BENCHMARK-FAILURE-RECOVERY-PLAYBOOK  
**Date**: 2026-06-01  
**Branch**: feature/q47_benchmark_failure_recovery_playbook  
**Scope**: All known and anticipated failure modes for `scripts/run_q46b_feature_extractor_benchmark.py`,
their detection signatures, and step-by-step recovery procedures.  
**Purpose**: Enable overnight failures to be diagnosed and recovered without human intervention delays.

---

## Quick-Reference Triage Decision Tree

```
Did the script exit with a non-zero status or produce no output CSV?
│
├─ No output CSV at all?
│   ├─ Script never started → FM-01 (Docker container down)
│   ├─ "Cannot import torch/torchvision" → FM-02 (run outside Docker)
│   ├─ "ERROR: Cannot import qcore" → FM-03 (qcore/path missing)
│   ├─ "MISSING: dataset_root" in dry-run → FM-04 (dataset mount broken)
│   ├─ "MISSING: baseline_checkpoint" → FM-05 (checkpoint missing)
│   └─ CUDA error on startup → FM-06 (GPU unavailable)
│
├─ Script ran but stopped early with partial/empty CSV?
│   ├─ "[WALL-TIME CAP]" in output → FM-07 (cap fired — normal or slow)
│   ├─ "CUDA out of memory" → FM-08 (VRAM exhausted)
│   ├─ "ERROR building backbone" for one candidate → FM-09 (model build fail)
│   ├─ DataLoader worker crash / "Killed" → FM-10 (host RAM or worker OOM)
│   └─ Zero rows in CSV / empty file → FM-11 (write-path failure)
│
├─ Script completed but results look wrong?
│   ├─ All AUROC values identical or suspiciously uniform → FM-12 (seed not set)
│   ├─ AUROC = 0.5 for all runs → FM-13 (eval bug / model on wrong device)
│   ├─ Network download hang / "ConnectionError" → FM-14 (ImageNet weights DL)
│   ├─ NumPy ABI warning + silent metric corruption → FM-15 (numpy version)
│   └─ CSV present but JSON absent → FM-16 (JSON write not implemented)
│
└─ Benchmark completed Phase 1 but Phase 2 decision unclear?
    └─ FM-17 (decision rule interpretation)
```

---

## Failure Mode Catalogue

---

### FM-01 — Docker Container Not Running

**Severity**: BLOCKING  
**Likelihood**: High (overnight: container may have been stopped or restarted)

**Detection signatures**:
```
Error response from daemon: No such container: docker-qstrata-gpu-1
# or
Error response from daemon: Container is not running
# or
docker: Error response from daemon: ...
```

**Root cause**: Container `docker-qstrata-gpu-1` exited or was stopped since the Q46E audit
(container was Up 47h at audit time, which does not guarantee overnight persistence on desktop).

**Recovery procedure**:
```bash
# 1. Check current container status
docker ps -a | grep docker-qstrata-gpu

# 2a. If status is "Exited" — restart without rebuild
docker start docker-qstrata-gpu-1

# 2b. If container does not exist at all — recreate from compose
cd /home/mike/research-projects/QStrata
docker compose -f docker-compose.gpu.yml up -d

# 3. Verify GPU is accessible
docker exec docker-qstrata-gpu-1 python3 -c "import torch; print(torch.cuda.is_available())"
# Expected: True

# 4. Re-run dry-run to confirm readiness
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --dry-run

# 5. Resume benchmark at interrupted phase
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --smoke
```

**Notes**: If container restart fails with GPU errors, see FM-06. The container uses the
`nvidia` runtime and will fail to start if the host driver is unavailable (suspend/resume
cycle, driver update, kernel upgrade).

---

### FM-02 — Script Run Outside Docker (torch Not Found)

**Severity**: BLOCKING  
**Likelihood**: Medium (accidental host invocation)

**Detection signatures**:
```
ERROR: Cannot import torch/torchvision: No module named 'torch'
Run inside docker-qstrata-gpu-1:
  docker exec docker-qstrata-gpu-1 python3 /workspace/scripts/...
```
or
```
ModuleNotFoundError: No module named 'torch'
```

**Root cause**: Script invoked via host Python (`.venv/`, Python 3.13) which intentionally
has no torch/torchvision. Host venv is minimal by design.

**Recovery procedure**:
```bash
# Always prefix with docker exec — never invoke directly from host shell
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --smoke

# Verify you are NOT in the host venv accidentally
which python3  # should NOT point to .venv/bin/python3 for benchmark runs
```

**Prevention**: Add a shebang check or wrapper script that refuses execution outside Docker.

---

### FM-03 — qcore / Project Module Import Failure

**Severity**: BLOCKING  
**Likelihood**: Low (workspace mount usually stable)

**Detection signatures**:
```
ERROR: Cannot import qcore/project modules: No module named 'qcore'
# or
ERROR: Cannot import qcore/project modules: cannot import name 'eval_split' from 'run_q38a_preprocessing_benchmark'
```

**Root cause**: One of:
- Volume mount `../..:/workspace` is not active (container restarted without compose)
- `qcore/` package missing from project root (git branch mismatch)
- `scripts/run_q38a_preprocessing_benchmark.py` missing or renamed

**Recovery procedure**:
```bash
# 1. Verify workspace mount inside container
docker exec docker-qstrata-gpu-1 ls /workspace/scripts/run_q38a_preprocessing_benchmark.py
docker exec docker-qstrata-gpu-1 ls /workspace/qcore/__init__.py

# 2. If /workspace is empty or partial — mount is broken
# Restart container via compose (which re-applies volume binds):
cd /home/mike/research-projects/QStrata
docker compose -f docker-compose.gpu.yml down
docker compose -f docker-compose.gpu.yml up -d

# 3. Verify again, then re-run
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --dry-run
```

---

### FM-04 — Dataset Root Missing or Splits Incomplete

**Severity**: BLOCKING  
**Likelihood**: Low (dataset verified at Q46E, but mount/path issues possible)

**Detection signatures** (from `--dry-run` or startup):
```
[✗] dataset_root                       MISSING  → /workspace/data/processed/vindr_binary_roi_224
# or
[⚠] dataset_root                       WARN  (missing train/val/test splits)
# or
FileNotFoundError: [Errno 2] No such file or directory: '.../vindr_binary_roi_224/train'
```

**Root cause**: Dataset bind-mount not active, or path changed in config.

**Recovery procedure**:
```bash
# 1. Check dataset from host
ls /home/mike/research-projects/QStrata/data/processed/vindr_binary_roi_224/{train,val,test}

# 2. Check from inside container
docker exec docker-qstrata-gpu-1 \
    ls /workspace/data/processed/vindr_binary_roi_224/train/

# 3. If host path is fine but container can't see it — mount broken
# Recreate container via compose (see FM-03 step 2)

# 4. Confirm split counts match audit (train: 6712, val: 1677, test: 2077 total images)
docker exec docker-qstrata-gpu-1 bash -c \
    "find /workspace/data/processed/vindr_binary_roi_224 -name '*.png' | wc -l"
# Expected: 10466
```

---

### FM-05 — Baseline Checkpoint Missing

**Severity**: BLOCKING (for baseline candidate only)  
**Likelihood**: Very low (checkpoint present at Q46E audit)

**Detection signatures**:
```
[✗] baseline_checkpoint                MISSING  → .../checkpoints/c006_d040_classical_anchor.pt
# or
FileNotFoundError: .../c006_d040_classical_anchor.pt
```

**Recovery procedure**:
```bash
# 1. Verify on host
ls -lh /home/mike/research-projects/QStrata/checkpoints/c006_d040_classical_anchor.pt
# Expected: ~52 KB

# 2. If missing — check git history (checkpoint may not be tracked)
git log --all --full-history -- checkpoints/c006_d040_classical_anchor.pt

# 3. Workaround if checkpoint is unrecoverable: skip baseline candidate
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py \
    --smoke --candidate efficientnet_b0 mobilenetv3_small mobilenetv3_large convnext_tiny
```

**Note**: The baseline is the reference comparator; losing it does not prevent evaluating
the alternative backbones. Document any skip in the Q46B report.

---

### FM-06 — GPU Unavailable / CUDA Not Accessible

**Severity**: BLOCKING (benchmark will fall back to CPU, causing 10–50× slowdown)  
**Likelihood**: Low but non-zero (desktop GPU, driver refresh after kernel update)

**Detection signatures**:
```
docker: Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
# or inside container:
torch.cuda.is_available() → False
# or
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

**Root cause**: NVIDIA driver not loaded (kernel module unloaded after suspend/resume or
driver update), or Docker nvidia runtime misconfigured.

**Recovery procedure**:
```bash
# 1. Check GPU from host
nvidia-smi

# 2. If nvidia-smi fails — driver not loaded
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm

# 3. Verify docker nvidia runtime
docker info | grep -i runtime

# 4. If nvidia runtime missing — reinstall nvidia-container-toolkit
# (system-level fix; document and escalate if needed)

# 5. Verify GPU accessible after fix
docker exec docker-qstrata-gpu-1 \
    python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 2060 SUPER

# 6. If VRAM is occupied by other processes (< 2 GB free) — check usage
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# Kill competing GPU processes if appropriate, or wait for them to finish
```

**CPU fallback**: If GPU cannot be restored, the benchmark will run on CPU. Smoke phase
(5 candidates × 1 seed × 4 epochs) is estimated at 30–60× longer. Do not run `--full` on
CPU without adjusting `FULL_CAP_MIN` (120 min will fire immediately). Document CPU execution
in the report if this path is taken.

---

### FM-07 — Wall-Time Cap Fired Mid-Run

**Severity**: NON-BLOCKING (expected behaviour for slow candidates)  
**Likelihood**: Medium (cap is 60 min smoke / 120 min full; ConvNeXt-Tiny is largest)

**Detection signatures**:
```
[WALL-TIME CAP] 60 min reached after 61.3 min — stopping.
# or
[CAP] Wall-time cap reached during epoch 3.
```

**What it means**:
- The cap is a safety rail, not a failure. Any completed rows were written to the CSV.
- Rows for the in-progress candidate/seed at cap time are NOT written (partial epoch abandoned).
- The cap resets when you re-invoke the script with `--candidate` to resume missing entries.

**Recovery procedure**:
```bash
# 1. Inspect what was written
cat experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv
# Note which candidate_ids are missing

# 2. Re-run only the missing candidates with explicit --candidate flag
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py \
    --smoke --candidate convnext_tiny

# 3. Manually merge the two CSVs (the script overwrites the output file):
# Save the first CSV, collect rows from the second run, combine, re-sort by auroc, re-rank.
python3 - <<'EOF'
import pandas as pd

first = pd.read_csv("experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv")
second = pd.read_csv("experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv")  # after re-run
# In practice: save first CSV as smoke_part1.csv, re-run writes smoke_part2.csv
# Then:
merged = pd.concat([first, second], ignore_index=True).drop_duplicates(subset=["candidate_id","seed"])
merged = merged.sort_values("auroc", ascending=False).reset_index(drop=True)
merged["rank"] = range(1, len(merged) + 1)
merged.to_csv("experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv", index=False)
EOF
```

**Prevention for Phase 2 (--full)**: If smoke reveals ConvNeXt-Tiny is very slow (> 15 min
per candidate×seed), either exclude it from Phase 2 or increase `FULL_CAP_MIN` to 180 min.

---

### FM-08 — CUDA Out of Memory

**Severity**: BLOCKING for the affected candidate  
**Likelihood**: Low (Q46E estimated peak VRAM well within 8 GB for all candidates)

**Detection signatures**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X.XX GiB
# or
RuntimeError: CUDA out of memory...
```

**Root cause**: Desktop background processes (Xwayland, Zoom, browser) increased VRAM
usage, leaving less than expected. ConvNeXt-Tiny (~28.6M params) is the most likely trigger.

**Recovery procedure**:
```bash
# 1. Check VRAM occupancy from host
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

# 2. Kill background GPU consumers if safe (example: GPU-accelerated browser tab)
# Xwayland and gnome-shell typically use 600–800 MiB; Zoom adds ~100 MiB

# 3. Reduce batch_size to 4 for the affected candidate (edit script constant or pass env var)
# The script does not currently expose --batch-size as a CLI arg.
# Temporary workaround: edit CANDIDATES list to run large candidates separately,
# or add a temporary override at the top of main():
#   _OPT_BATCH_SIZE = 4  # reduce for OOM recovery

# 4. Re-run affected candidate only
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py \
    --smoke --candidate convnext_tiny

# 5. If OOM persists at batch_size=4, exclude ConvNeXt-Tiny from Q46B and document
```

---

### FM-09 — Backbone Build Failure (Single Candidate)

**Severity**: NON-BLOCKING (script `continue`s to next candidate)  
**Likelihood**: Low

**Detection signatures**:
```
    ERROR building backbone: <some exception message>
# Script continues to next candidate; affected candidate produces no row
```

**Root cause possibilities**:
- `torchvision.models` API changed (unlikely — pinned to 0.17.2+cu121)
- `run_q38a_preprocessing_benchmark.load_backbone` signature changed for baseline
- Checkpoint file truncated or corrupt (baseline only)
- Unknown backbone name in `_FACTORY` dict

**Recovery procedure**:
```bash
# 1. Isolate the failing candidate
docker exec docker-qstrata-gpu-1 python3 - <<'EOF'
import sys; sys.path.insert(0, "/workspace/scripts"); sys.path.insert(0, "/workspace")
from run_q46b_feature_extractor_benchmark import build_backbone_extractor, CANDIDATE_MAP
spec = CANDIDATE_MAP["efficientnet_b0"]  # change to failing candidate_id
bb, dim = build_backbone_extractor(spec, "/workspace")
print(f"OK — feature_dim={dim}")
EOF

# 2. For baseline checkpoint corruption:
python3 -c "import torch; ck=torch.load('/workspace/checkpoints/c006_d040_classical_anchor.pt'); print(type(ck))"

# 3. If torchvision API mismatch — pin torchvision back:
docker exec docker-qstrata-gpu-1 pip install "torchvision==0.17.2+cu121" --index-url https://download.pytorch.org/whl/cu121

# 4. Skip affected candidate and document in report
```

---

### FM-10 — DataLoader Worker Crash / Host RAM OOM

**Severity**: BLOCKING for affected run  
**Likelihood**: Low-Medium (4 workers × persistent_workers=True can hold ~2 GB per candidate)

**Detection signatures**:
```
Killed
# or
RuntimeError: DataLoader worker (pid XXXXX) is killed by signal: Killed.
# or
BrokenPipeError / EOFError from DataLoader
```

**Root cause**: Host system RAM exhausted by 4 persistent DataLoader workers, especially
when multiple candidates run back-to-back without worker teardown. Each worker holds a copy
of dataset metadata and image buffers.

**Recovery procedure**:
```bash
# 1. Check host RAM at time of failure
free -h
# If < 2 GB free, reduce workers

# 2. Reduce num_workers to 2 (edit script or apply in-container):
# No CLI flag currently — temporarily patch the constant before re-run:
docker exec docker-qstrata-gpu-1 bash -c \
    "sed -i 's/_OPT_NUM_WORKERS        = 4/_OPT_NUM_WORKERS        = 2/' \
     /workspace/scripts/run_q46b_feature_extractor_benchmark.py"

# 3. Re-run affected candidate(s)
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py \
    --smoke --candidate mobilenetv3_large

# 4. Revert worker count after run
docker exec docker-qstrata-gpu-1 bash -c \
    "sed -i 's/_OPT_NUM_WORKERS        = 2/_OPT_NUM_WORKERS        = 4/' \
     /workspace/scripts/run_q46b_feature_extractor_benchmark.py"
```

**Note**: `persistent_workers=True` keeps workers alive between epochs, which is efficient
but means 4 workers × 5 candidates can accumulate. Disabling `persistent_workers` is a
lighter-weight alternative if RAM is tight.

---

### FM-11 — CSV Not Written / Empty Output File

**Severity**: HIGH (data loss risk)  
**Likelihood**: Very low

**Detection signatures**:
```
# Script exits normally but CSV is absent or 0 bytes
ls -lh experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv
# shows 0 bytes or "No such file"
```

**Root cause**: `rows` list is empty because all candidates failed (see FM-09) or cap fired
before any candidate completed. The `if rows:` guard in the script suppresses the CSV write
when empty.

**Recovery procedure**:
```bash
# 1. Check script stdout/stderr for "ERROR building backbone" lines
# If all candidates failed — diagnose via FM-09

# 2. If cap fired before any candidate completed:
# Smoke: a single candidate × 4 epochs at normal speed should take < 10 min
# If cap fired in < 10 min, the clock was already running before the script started
# (check for stale WALL_TIME_CAP from a prior invocation sharing the process)
# → Simply re-invoke; the cap timer resets on each script invocation

# 3. Verify output directory permissions
docker exec docker-qstrata-gpu-1 \
    ls -la /workspace/experiments/leaderboards/

# 4. If path is a directory (not a file) — stale artifact; remove and re-run
docker exec docker-qstrata-gpu-1 \
    rm -rf /workspace/experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv
```

---

### FM-12 — All AUROC Values Identical or Suspiciously Uniform

**Severity**: HIGH (results invalid — seeds not being applied)  
**Likelihood**: Very low

**Detection signatures**:
```csv
rank,candidate_id,seed,auroc,...
1,efficientnet_b0,42,0.712345,...
2,efficientnet_b0,7,0.712345,...   ← identical to seed 42
3,efficientnet_b0,123,0.712345,...  ← identical across seeds
```

**Root cause**: `set_seeds()` not called correctly, or `torch.backends.cudnn.deterministic`
is `False` while `benchmark=True` overrides seeds for CUDA operations.

**Recovery procedure**:
```bash
# 1. Verify set_seeds works correctly inside container
docker exec docker-qstrata-gpu-1 python3 - <<'EOF'
import sys; sys.path.insert(0, "/workspace/scripts")
from run_q38a_preprocessing_benchmark import set_seeds
import torch
set_seeds(42); a = torch.randn(3)
set_seeds(7);  b = torch.randn(3)
set_seeds(42); c = torch.randn(3)
print("Seeds differ:", not torch.allclose(a, b))
print("Same seed reproducible:", torch.allclose(a, c))
EOF

# 2. If seeds produce identical outputs — check if torch.use_deterministic_algorithms(True)
# interferes; try adding torch.backends.cudnn.benchmark = False to the training loop

# 3. Results with truly identical AUROC across seeds suggest eval_split returns a cached
# value — unlikely but check for module-level caching in run_q38a_preprocessing_benchmark
```

---

### FM-13 — AUROC = 0.5 for All Runs (Random Baseline)

**Severity**: HIGH (model not training)  
**Likelihood**: Low

**Detection signatures**:
```
AUROC=0.5000  F1=0.xxxx  Δvs_q45a=-0.2196  wall=...
```
AUROC ≈ 0.5 is random chance — model is not learning.

**Root cause possibilities**:
- Model ended up on CPU but data on GPU (or vice versa), causing silent no-op
- All `trainable` params list is empty (backbone frozen AND head accidentally frozen)
- `BCEWithLogitsLoss` receiving wrong shape (logits not squeezed correctly)
- `eval_split` computing AUROC on logits instead of sigmoid probabilities

**Recovery procedure**:
```bash
# 1. Add a debug run with a single epoch, print loss trajectory
docker exec docker-qstrata-gpu-1 python3 - <<'EOF'
# Paste a minimal training loop snippet that prints loss per batch for 1 epoch
# Expected: loss should decrease from ~0.69 (BCE random) within first few batches
EOF

# 2. Check trainable parameter count
# In the main loop: print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Expected: ~2250 (head only). If 0 — head is accidentally frozen.

# 3. Check device consistency
# Add: print(f"Model device: {next(model.parameters()).device}, Data device: {imgs.device}")

# 4. If projection layer is not trainable (requires_grad=False by default from backbone freeze loop)
# The projection `nn.Linear(feat_dim, head_in_dim)` is initialized fresh — it should be trainable
# Verify: projection parameters do NOT have requires_grad=False
```

---

### FM-14 — ImageNet Weight Download Failure

**Severity**: BLOCKING for torchvision candidates  
**Likelihood**: Medium (overnight run in air-gapped or throttled network)

**Detection signatures**:
```
Downloading: "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
urllib.error.URLError: <urlopen error [Errno -2] Name or service not known>
# or
ConnectionError / TimeoutError during model instantiation
```

**Root cause**: torchvision downloads pretrained weights on first use per environment.
Inside `docker-qstrata-gpu-1`, the weights may not be cached yet (cache is at
`~/.cache/torch/hub/checkpoints/` inside the container, which is not bind-mounted).

**Recovery procedure**:
```bash
# 1. Check if weights are already cached inside container
docker exec docker-qstrata-gpu-1 \
    ls ~/.cache/torch/hub/checkpoints/

# 2. Pre-download weights manually while network is available
docker exec docker-qstrata-gpu-1 python3 - <<'EOF'
from torchvision.models import efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large, convnext_tiny
for fn in [efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large, convnext_tiny]:
    m = fn(weights="IMAGENET1K_V1")
    print(f"Downloaded: {fn.__name__}")
    del m
print("All weights cached.")
EOF

# 3. If network is genuinely unavailable, use weights=None (random init) as a fallback
# Note: results with weights=None are not comparable to Q46A protocol — document explicitly
# Modify build_backbone_extractor: change "IMAGENET1K_V1" → None for offline runs

# 4. Bind-mount the torch cache directory to survive container restarts
# Add to docker-compose.gpu.yml volumes:
#   - ~/.cache/torch:/root/.cache/torch
```

---

### FM-15 — NumPy ABI Warning / Silent Metric Corruption

**Severity**: LOW (currently non-blocking per Q46E audit)  
**Likelihood**: High (numpy 2.2.6 installed in container; Dockerfile constraint was numpy<2)

**Detection signatures**:
```
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
  (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:84.)
```

**Current status**: Non-fatal — torch operates correctly despite the warning. CUDA active,
metrics computed correctly. See Q46E §5.2.

**Recovery procedure** (recommended before any Phase 2 or Phase 3 run):
```bash
# Downgrade numpy inside container (no image rebuild required)
docker exec docker-qstrata-gpu-1 pip install "numpy<2" --force-reinstall

# Verify warning is gone
docker exec docker-qstrata-gpu-1 python3 -c "import torch; print('OK')"
# Expected: OK (no UserWarning)

# Verify CUDA still active after numpy downgrade
docker exec docker-qstrata-gpu-1 python3 -c \
    "import torch; print(torch.cuda.get_device_name(0))"
```

**If numpy downgrade breaks scikit-learn** (unlikely but possible):
```bash
docker exec docker-qstrata-gpu-1 pip install "scikit-learn" --upgrade
```

---

### FM-16 — Results JSON Absent After Completion

**Severity**: LOW (expected — not yet implemented)  
**Likelihood**: Certain (by design as of Q46C scaffold)

**Detection signatures**:
```
# Script completes, leaderboard CSV is present, but:
ls experiments/results/q46b_feature_extractor_benchmark.json
# No such file or directory
```

**Root cause**: The JSON write step is declared in `RESULTS_JSON_PATH` but not implemented
in `main()` as of the Q46C scaffold. The leaderboard CSV is the authoritative output.

**Workaround** (manual JSON generation from CSV):
```python
import pandas as pd, json, datetime

df = pd.read_csv("experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv")

results = {
    "slice_id": "Q46",
    "phase": "smoke",
    "run_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "git_commit": "see git log",
    "config_path": "configs/q46_feature_extractor_benchmark.yaml",
    "baselines": {
        "q38c_auroc": 0.723922, "q38c_f1": 0.677858,
        "q45a_mean_auroc": 0.7196, "q45a_mean_f1": 0.6360,
    },
    "rows": df.to_dict(orient="records"),
    "verdict": "PENDING",
    "winner_candidate_id": None,
    "wall_time_total_s": df["wall_time_s"].sum(),
}

with open("experiments/results/q46b_feature_extractor_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
```

**Action item**: Implement JSON write in script as part of Q46B execution slice.

---

### FM-17 — Decision Rule Interpretation After Phase 2

**Severity**: N/A (documentation / process failure, not runtime)  
**Likelihood**: Medium (rule has two conditions; easy to misread)

**Decision rule** (from config `decision_rule` block):
```
WINNER   : candidate mean_auroc > 0.7196 (Q45A) AND beats baseline in ≥ 2/3 seeds
NEGATIVE : no candidate meets both conditions
EXTENDED_TRIGGER : candidate mean_auroc within 0.005 of Q38C ceiling (0.7239) → run Phase 3
```

**Correct interpretation procedure**:
```python
import pandas as pd

Q45A_MEAN_AUROC = 0.7196
Q38C_BEST_AUROC = 0.723922
EXTENDED_DELTA  = 0.005

df = pd.read_csv("experiments/leaderboards/q46b_extractor_full_leaderboard.csv")
summary = df.groupby("candidate_id")["auroc"].agg(["mean", "std", "count"])
summary.columns = ["mean_auroc", "std_auroc", "n_seeds"]
summary["seeds_beat_q45a"] = df.groupby("candidate_id")["auroc"].apply(
    lambda x: (x > Q45A_MEAN_AUROC).sum()
)
summary["winner"] = (summary["mean_auroc"] > Q45A_MEAN_AUROC) & (summary["seeds_beat_q45a"] >= 2)
summary["extended"] = (summary["mean_auroc"] >= Q38C_BEST_AUROC - EXTENDED_DELTA)
print(summary[["mean_auroc", "std_auroc", "seeds_beat_q45a", "winner", "extended"]])
```

**Common mistake**: Reading only `mean_auroc > 0.7196` without checking `seeds_beat_q45a >= 2`.
A candidate can have mean > 0.7196 but only beat baseline on 1/3 seeds — that is NEGATIVE,
not WINNER.

---

## Pre-Launch Verification Checklist

Run this before any overnight benchmark invocation to catch FM-01 through FM-06 proactively:

```bash
# Step 1: Container running?
docker ps | grep docker-qstrata-gpu-1

# Step 2: GPU accessible?
docker exec docker-qstrata-gpu-1 \
    python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Step 3: Weights cached?
docker exec docker-qstrata-gpu-1 ls ~/.cache/torch/hub/checkpoints/ 2>/dev/null | head

# Step 4: VRAM headroom?
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

# Step 5: Dry-run passes?
docker exec docker-qstrata-gpu-1 \
    python3 /workspace/scripts/run_q46b_feature_extractor_benchmark.py --dry-run

# Step 6: NumPy ABI clean? (recommended)
docker exec docker-qstrata-gpu-1 \
    python3 -c "import torch" 2>&1 | grep -i warning || echo "No warnings"
```

All six steps should pass before launching `--smoke` or `--full` unattended.

---

## Output Artifact Recovery Map

| Artifact | Expected path | Recovery if absent |
|---|---|---|
| Smoke leaderboard | `experiments/leaderboards/q46b_extractor_smoke_leaderboard.csv` | Re-run `--smoke` |
| Full leaderboard | `experiments/leaderboards/q46b_extractor_full_leaderboard.csv` | Re-run `--full` |
| Extended leaderboard | `experiments/leaderboards/q46b_extractor_extended_leaderboard.csv` | Only needed if `EXTENDED_TRIGGER`; re-run with winner + EXT_SEEDS |
| Results JSON | `experiments/results/q46b_feature_extractor_benchmark.json` | Generate manually (FM-16 workaround) |
| Benchmark report | `reports/q46b_feature_extractor_benchmark.md` | Write post-execution from leaderboard CSVs |

---

## Severity Classification

| Code | Severity | Meaning |
|---|---|---|
| BLOCKING | Script cannot run or produces no output | Must resolve before any execution |
| HIGH | Script runs but results are invalid or data lost | Results must not be used until fixed |
| NON-BLOCKING | Script produces valid output for unaffected candidates | Document and continue |
| LOW | Advisory only | Fix before Phase 2 or extended run |

---

```
Playbook version: 1.0
Defined: Q47 (failure recovery playbook)
Script source: scripts/run_q46b_feature_extractor_benchmark.py (Q46C scaffold)
Environment source: reports/q46e_execution_environment_audit.md
Schema source: docs/specs/q46_benchmark_output_schema.md
```
