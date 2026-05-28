# Q36A — SkyPilot Single-Node Backend Pilot

**Slice:** Q36A  
**Branch:** `feature/q36a-skypilot-single-node`  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Status:** PASS (infra-validation design) — YAML created, no cloud execution performed

---

## 1. Objective

Validate a minimal SkyPilot single-node backend for QStrata CV NAS experiments. Measure (by design analysis) expected runtime, cost, setup friction, and reproducibility characteristics before committing to any cloud execution. This slice produces a validated SkyPilot YAML definition and execution protocol — it does not perform a live cloud run.

**Scope constraint:** No Ray, no distributed execution, no multi-node, no full NAS run.

---

## 2. YAML Artifact

**File:** `infra/skypilot/q36a_single_node_smoke.yaml`  
**Status:** Created and committed

| Field | Value |
|---|---|
| Name | `qstrata-q36a-cv-smoke-cpu` |
| Cloud | AWS |
| Instance | `c6i.xlarge` (4 vCPUs, 8 GB RAM) |
| Region | `us-east-1` |
| Spot | No (on-demand — smoke test reliability) |
| Disk | 30 GB |
| Workdir | `.` (rsync from repo root) |

---

## 3. Smoke Command

```bash
python3 scripts/run_q34c_cv_nas_pilot.py \
  --trials 1 \
  --seed 42 \
  --epochs 1 \
  --parallel \
  --max-workers 1 \
  --thread-cap 2
```

This is the Q34C CV NAS pilot at minimum viable scale: 1 trial, 1 epoch, 1 worker. It exercises the full execution path — config generation, symplectic CV circuit training, stability checks, metric emission, Pareto CSV update — without incurring full NAS cost.

---

## 4. Setup Steps

The YAML's `setup:` block executes on first launch:

| Step | Action | Expected Time |
|---|---|---|
| 1 | pip install CPU-only PyTorch 2.2.2 + torchvision 0.17.2 | ~60–90s |
| 2 | pip install numpy<2, medmnist, scikit-learn, pyyaml, matplotlib | ~30–60s |
| 3 | Verify torch import + CPU device | <5s |
| 4 | Download + verify PneumoniaMNIST (~4 MB) | ~5–15s |
| 5 | Verify PYTHONPATH + repo script paths exist | <5s |
| **Total setup** | | **~3–5 min** |

**Setup is idempotent.** Subsequent `sky launch` on the same stopped instance skips pip if the environment is persisted on the disk volume.

**Environment delta vs local (docker-qstrata-gpu-1):**

| Dimension | Local (Docker GPU) | Cloud (c6i.xlarge) | Impact |
|---|---|---|---|
| PyTorch build | CUDA 12.1 wheel | CPU-only wheel | No GPU acceleration |
| Backbone device | CUDA (RTX 2060 Super) | CPU | Backbone ~3× slower |
| CV ops | CPU (symplectic) | CPU (symplectic) | **Identical** |
| Dataset | medmnist PneumoniaMNIST | medmnist PneumoniaMNIST | **Identical** |
| Thread cap | 2/worker | 2/worker | **Identical** |
| Python version | 3.11 (Docker) | System Python3 (Ubuntu 22.04 = 3.10) | Minor version delta |

**Reproducibility risk:** Minor. The CPU torch wheel and Python 3.10 vs 3.11 delta are unlikely to affect metric outputs. The CV symplectic ops, dataset, and random seed are identical.

---

## 5. Estimated Runtime

### Smoke (1 trial, 1 epoch)

| Phase | Estimate | Basis |
|---|---|---|
| Setup | 3–5 min | pip install + medmnist download |
| Run | 2–4 min | 1 epoch; backbone on CPU (~3× local GPU slowdown) |
| **Total** | **5–9 min** | |

**Local reference (Q34C smoke, 1 trial 1 epoch, GPU backbone):** ~3 min.  
**Cloud CPU estimate:** ~2–3× slower backbone → ~6–9 min total including setup.

### Full 5-Trial 2-Epoch CV Pilot (reference — not Q36A scope)

| Instance | Estimate | Cost Estimate |
|---|---|---|
| c6i.xlarge (4 vCPUs) | 25–45 min | ~$0.07–$0.13 |
| c6i.2xlarge (8 vCPUs) | 15–30 min | ~$0.09–$0.17 |

**vs local (Q34C full, GPU backbone, 335.6s = 5.6 min):**  
Cloud CPU will be 3–5× slower due to backbone CUDA → CPU regression. Thread-cap=2 per worker limits BLAS parallelism in the same way as local, but without GPU backbone acceleration. At c6i.2xlarge scale, 8 vCPUs support 5 workers × 2 threads comfortably.

---

## 6. Cost Estimate

All estimates use us-east-1 on-demand pricing (2026):

| Scenario | Instance | Runtime | Cost |
|---|---|---|---|
| Smoke (Q36A) | c6i.xlarge ($0.17/hr) | 5–9 min | **< $0.03** |
| Full CV pilot | c6i.xlarge ($0.17/hr) | 25–45 min | ~$0.07–$0.13 |
| Full CV pilot | c6i.2xlarge ($0.34/hr) | 15–30 min | ~$0.09–$0.17 |
| GPU smoke comparison | g4dn.xlarge ($0.526/hr) | 5–8 min | ~$0.04–$0.07 |

**Recommendation for Q36B (Local vs SkyPilot Comparison):** Run the smoke on `c6i.xlarge` first. If setup + run validates end-to-end, run the full 5-trial 2-epoch pilot on `c6i.2xlarge`. Total Q36B budget: < $0.25.

**Cost ceiling:** No single run should exceed $0.50. The YAML uses `use_spot: false` for smoke reliability; switching to `use_spot: true` reduces cost by ~70% (c6i.xlarge spot: ~$0.05/hr) for longer runs.

---

## 7. Code Sync Validation

SkyPilot's `workdir: .` rsync syncs all non-`.gitignore`d files from the local repo root to `/workspace` on the instance. This covers:

| Synced | Path | Purpose |
|---|---|---|
| ✅ | `scripts/run_q34c_cv_nas_pilot.py` | Smoke orchestrator |
| ✅ | `scripts/train_q34c_cv_candidate.py` | Per-trial training script |
| ✅ | `qcore/` | CV circuit library |
| ✅ | `configs/` | Experiment configs |
| ✅ | `experiments/leaderboards/` | Pareto CSVs (pre-existing) |
| ❌ | `data/`, `*.zip`, experiment logs | gitignored — not synced |
| ❌ | Docker image layers | Not synced — pip install used instead |

**Dataset access:** PneumoniaMNIST is auto-downloaded by `medmnist` at runtime (~4 MB). No pre-staged S3 data required for smoke.

**Key sync verification:** The setup block checks `scripts/run_q34c_cv_nas_pilot.py` and `scripts/train_q34c_cv_candidate.py` exist at `/workspace/` after sync. Missing files fail fast.

---

## 8. Result Artifact Creation

The smoke run is expected to produce:

| Artifact | Path | Committed? |
|---|---|---|
| Summary JSON | `experiments/results/q34c_cv_nas_pilot_summary.json` | No (gitignored) |
| Pareto CSV | `experiments/leaderboards/q34c_cv_pareto.csv` | Pre-existing |
| Smoke log | `/tmp/q36a_smoke_<timestamp>.log` | No (temp) |

The `run:` block verifies artifact creation after the smoke completes. The Pareto CSV is expected to be updated (or remain unchanged if the single trial is non-Pareto). The summary JSON is always written by the pilot script.

---

## 9. Shutdown Procedure

**MANDATORY: Always terminate or stop the instance after the run completes to prevent cost accumulation.**

```bash
# After smoke completes — check logs first
sky logs qstrata-q36a-cv-smoke-cpu

# Option A: Terminate (destroys instance + data; billed storage stops)
sky down qstrata-q36a-cv-smoke-cpu

# Option B: Stop (paused; EBS volume persists; minimal storage cost only)
sky stop qstrata-q36a-cv-smoke-cpu

# Auto-stop safety net (add to launch command)
sky launch infra/skypilot/q36a_single_node_smoke.yaml \
  --cloud aws --yes --idle-minutes-to-autostop 10
```

**Recommended:** use `--idle-minutes-to-autostop 10` on every launch. The instance auto-terminates 10 minutes after the run finishes, even if the operator forgets to `sky down` manually.

---

## 10. Execution Protocol (Human-Runnable)

The following sequence must be run manually by the operator. **Do not automate cloud launch.**

```bash
# Step 1: Verify SkyPilot installation and AWS credentials
pip show skypilot
sky check

# Step 2: Dry-run — confirm instance type and cost before launch
sky launch infra/skypilot/q36a_single_node_smoke.yaml --dryrun

# Step 3: Launch smoke (requires explicit human approval)
# Add --idle-minutes-to-autostop 10 as safety net
sky launch infra/skypilot/q36a_single_node_smoke.yaml \
  --cloud aws \
  --yes \
  --idle-minutes-to-autostop 10

# Step 4: Monitor (SkyPilot streams logs automatically after launch)
sky logs qstrata-q36a-cv-smoke-cpu

# Step 5: Terminate after completion
sky down qstrata-q36a-cv-smoke-cpu

# Step 6: Record wall time, exit code, and metric values in Q36B report
```

---

## 11. Validation Checklist

| Check | Result |
|---|---|
| SkyPilot YAML created | ✅ `infra/skypilot/q36a_single_node_smoke.yaml` |
| Smoke command defined (1 trial, 1 epoch) | ✅ |
| Setup steps documented | ✅ (5-step setup block + table) |
| Estimated runtime documented | ✅ (5–9 min smoke; 15–45 min full pilot) |
| Estimated cost documented | ✅ (< $0.03 smoke; < $0.25 Q36B budget) |
| Shutdown command documented | ✅ (`sky down` + autostop) |
| No Ray | ✅ |
| No distributed execution | ✅ (single-node only) |
| No full NAS run executed | ✅ (design-only slice) |
| Roadmap updated | ✅ |

**Verdict: PASS** — all 10 pass criteria met.

---

## 12. Q36A Status: COMPLETE

Q36A defines the SkyPilot single-node execution path for QStrata CV NAS experiments. The YAML is validated by design (matches existing `q34b_dv_smoke_gpu.yaml` pattern, uses established Q34C smoke command, documents all required environment and cost characteristics).

**Q36B (Local vs SkyPilot Runtime/Cost Comparison) is now the next slice.** Q36B performs the actual cloud launch, records measured wall time, compares with local baseline (Q34C: 335.6s for 5 trials × 2 epochs), and produces a go/no-go recommendation for cloud-scaled NAS.

| Slice | Scope | Status |
|---|---|---|
| Q36A | YAML definition + cost/runtime design | **COMPLETE** |
| Q36B | Live smoke launch + local vs cloud comparison | **NEXT** |
| Q36C+ | Full CV NAS pilot on cloud (if Q36B passes) | PLANNED |

**Hard blocks remain:**
- Ray: BLOCKED
- Multiclass: BLOCKED
- Object detection: BLOCKED
