# Q34B Full-Lite DV NAS Pilot

**Slice:** Q34B-full-lite  
**Branch:** `feature/q34b-runtime-hotfix`  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Status:** PASS — 5/5 trials completed, 4-member Pareto set

---

## 1. Execution Configuration

| Parameter | Value |
|---|---|
| Script | `run_q34b_dv_nas_pilot.py --trials 5 --seed 42 --epochs 4 --parallel --max-workers 5 --thread-cap 2` |
| Search space version | `q34b_v1` |
| Trials | 5 |
| Epochs per trial | 4 (full budget; vs 2 in Q34B-Parallel-Lite) |
| Execution mode | Parallel (ThreadPoolExecutor, max_workers=5) |
| Thread cap | 2 (OMP/MKL/OPENBLAS per subprocess) |
| Pass threshold | 3 / 5 |
| Seed | 42 (trial seeds: 42–46) |
| Hardware | NVIDIA GeForce RTX 2060 SUPER (docker-qstrata-gpu-1) |
| Device | CPU (DV quantum circuit is CPU-only; backbone also CPU-side) |
| Wall time | **5,917.1 s** (~98.6 min) |
| Pilot verdict | **PASS** |

---

## 2. Sampled Configurations

| Trial | Qubits | Depth | Rotation | Entangle | Reupload | Compress | LR | WD |
|---|---|---|---|---|---|---|---|---|
| q34b_trial_001 | 2 | 1 | RX+RY+RZ | circular | none | 2 | 0.0005 | 0.0 |
| q34b_trial_002 | 2 | 1 | RY | linear | none | 2 | 0.0005 | 0.0 |
| q34b_trial_003 | 2 | 2 | RX+RY | circular | none | 2 | 0.0005 | 0.0 |
| q34b_trial_004 | 4 | 2 | RX+RY+RZ | circular | every_layer | 4 | 0.0005 | 0.0001 |
| q34b_trial_005 | 2 | 1 | RY | linear | none | 2 | 0.0005 | 0.0001 |

**Note on substitutions (medical_ansatz is hardcoded):** The `DVHybridCNNQNN` model substitutes several sampled dimensions to fixed values at runtime:
- `rotation_family` → always `RX+RY+RZ` (medical_ansatz fixed)
- `entanglement_topology` → always `circular` (medical_ansatz fixed)
- `reuploading_frequency` → always `every_layer` (medical_ansatz fixed)
- `compression_dim` → always equals `n_qubits` (no independent bottleneck)

Effective search dimensions were: `qubit_count`, `ansatz_depth`, `learning_rate`, `weight_decay`. All other sampled dimensions were substituted. This is a known Q34B constraint documented in Q34B-HF.

---

## 3. Trial Results

| Trial | AUROC | F1 | Accuracy | Params | Latency | Grad | NaN/Inf | Wall (s) |
|---|---|---|---|---|---|---|---|---|
| q34b_trial_001 | **0.6415** | **0.6356** | 0.6081 | **280** | 83.8 ms | PASS | PASS | 5,359 |
| q34b_trial_002 | 0.6464 | 0.5783 | 0.6264 | **280** | 93.9 ms | PASS | PASS | 5,371 |
| q34b_trial_003 | 0.6462 | 0.6155 | 0.6235 | 292 | 64.7 ms | PASS | PASS | 5,456 |
| q34b_trial_004 | **0.6551** | 0.6289 | 0.6148 | 598 | 67.8 ms | PASS | PASS | 5,917 |
| q34b_trial_005 | 0.6400 | 0.6077 | 0.6283 | **280** | 82.7 ms | PASS | PASS | 5,371 |

**5/5 completed. 0 failed. 0 invalid. Pilot verdict: PASS.**

---

## 4. Training Curves (per-epoch validation AUROC)

| Trial | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Δ(1→4) |
|---|---|---|---|---|---|
| q34b_trial_001 | 0.6322 | 0.6401 | 0.6432 | 0.6414 | +0.0092 |
| q34b_trial_002 | 0.5409 | 0.6358 | 0.6377 | 0.6438 | +0.1029 |
| q34b_trial_003 | 0.6374 | 0.6370 | 0.6431 | 0.6431 | +0.0057 |
| q34b_trial_004 | 0.6329 | 0.6353 | 0.6485 | 0.6496 | +0.0167 |
| q34b_trial_005 | 0.5285 | 0.6359 | 0.6358 | 0.6380 | +0.1095 |

All trials show positive AUROC convergence across 4 epochs. Trial_002 and trial_005 both start at val AUROC ~0.53 (F1=0 at epoch 1, collapsed predictions) and recover by epoch 2, suggesting gradient direction issues in epoch 1 that self-correct. No trial exhibits sustained instability.

---

## 5. DV Pareto Frontier

Pareto criterion: trial A dominates B if AUROC(A) ≥ AUROC(B), F1(A) ≥ F1(B), params(A) ≤ params(B), with at least one strict improvement. Hard exclusions: nan_inf=FAIL, params > 100K, gradient_health=FAIL.

| Trial | AUROC | F1 | Params | Pareto | Notes |
|---|---|---|---|---|---|
| q34b_trial_001 | 0.6415 | 0.6356 | 280 | ✅ YES | Best F1; smallest params tie |
| q34b_trial_002 | 0.6464 | 0.5783 | 280 | ✅ YES | Higher AUROC than 001; lower F1 |
| q34b_trial_003 | 0.6462 | 0.6155 | 292 | ✅ YES | AUROC ≈ 002; F1 between 001 and 002 |
| q34b_trial_004 | 0.6551 | 0.6289 | 598 | ✅ YES | Best AUROC; 4-qubit cost |
| q34b_trial_005 | 0.6400 | 0.6077 | 280 | ❌ NO | Dominated by trial_001 (AUROC+F1, same params) |

**4-trial Pareto set.** Leaderboard: `experiments/leaderboards/q34b_dv_pareto.csv`

---

## 6. Performance vs Frozen Baselines

| Model | AUROC | F1 | Params | Source |
|---|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 23,650 | Binary benchmark |
| Q21 DV Hybrid | **0.6800** | 0.6159 | 574 | Binary benchmark |
| Q22 Tiny Classical | 0.6625 | 0.5961 | 526 | Binary benchmark |
| Q27 CV Hybrid | 0.6708 | **0.6283** | 536 | Binary benchmark |
| **Q34B best AUROC** (trial_004) | 0.6551 | 0.6289 | 598 | This pilot |
| **Q34B best F1** (trial_001) | 0.6415 | 0.6356 | 280 | This pilot |

**Key gap:** Best Q34B AUROC (0.6551) is −0.0249 below Q21 DV (0.6800) at comparable param count (598 vs 574). Best Q34B F1 (0.6356) exceeds Q21 F1 (0.6159) by +0.0197 at lower params (280 vs 574).

**Context:** Q34B-full-lite is a 5-trial NAS pilot with a constrained random search. The Q21 baseline was a purpose-designed, hand-tuned architecture trained to convergence with early stopping and model selection. The NAS pilot is exploratory — the Pareto frontier documents the search distribution, not the architectural ceiling.

The F1 result (trial_001: 0.6356) is a meaningful signal: 280 params at F1=0.6356 exceeds Q21's 574 params at F1=0.6159, suggesting that the compact NAS-sampled DV head is competitive on F1 at lower parameter cost despite the AUROC gap.

---

## 7. Wall Time Analysis

| Run | Epochs | Mode | Thread Cap | Wall Time | vs Q34B Sequential |
|---|---|---|---|---|---|
| Q34B sequential | 4 | sequential | none | 11,491 s | 1.00× (baseline) |
| Q34B-Parallel-Lite | 2 | parallel | none | 4,621 s | 2.49× faster* |
| **Q34B-full-lite** | **4** | **parallel** | **2** | **5,917 s** | **1.94× faster** |

*Different epoch budget — not directly comparable for per-epoch throughput.

**Per-epoch throughput comparison (thread-cap=2 vs no thread-cap):**
- Q34B-Parallel-Lite: 4,621 s ÷ 2 epochs = **2,310 s/epoch** (no thread cap; 3.9× thread contention)
- Q34B-full-lite: 5,917 s ÷ 4 epochs = **1,479 s/epoch** (thread-cap=2; reduced contention)
- Per-epoch speedup from thread-cap: **1.56×**

**Thread-cap trade-off:** Thread-cap=2 reduces OS scheduling contention (from 5×30=150 threads to 5×2=10 threads on 12 CPUs) but limits backbone BLAS parallelism. Since both the backbone and DV quantum circuit run on CPU, and the backbone (ResNet-style CNN) benefits from multi-thread BLAS for its convolutional layers, the effective per-epoch speedup (1.56×) is lower than the theoretical contention reduction (3.9×). The remainder is explained by backbone throughput reduction (12→2 BLAS threads per trial).

**NumPy 1.x/2.x warning (non-fatal):** All trials emit `UserWarning: Failed to initialize NumPy: _ARRAY_API not found` during torch import. This is a NumPy 1.x/2.x ABI compatibility warning from the container's NumPy 2.2.6 vs PyTorch compiled with NumPy 1.x. The warning is non-fatal — all 5 trials trained to completion and produced valid outputs. No action required.

---

## 8. Architectural Observations

**qubit_count=4 wins on AUROC:**
- trial_004 (qubits=4, depth=2, reupload=every_layer): AUROC=0.6551, F1=0.6289, 598 params
- All other 2-qubit trials: AUROC 0.6400–0.6464
- Per-epoch time: trial_004 = ~1,479 s/epoch vs ~1,340–1,365 s/epoch for 2-qubit trials (~1.1× overhead for 4×-larger state vector)

**depth=2 vs depth=1 (at qubits=2):**
- trial_003 (depth=2): AUROC=0.6462, F1=0.6155, 292 params (+12 params vs depth=1)
- trial_001 (depth=1, same seed class): AUROC=0.6415, F1=0.6356, 280 params
- depth=2 gains +0.0047 AUROC at cost of +12 params and −0.0201 F1 — mixed result

**weight_decay effect:**
- trial_001 (wd=0.0): F1=0.6356 — highest among 2-qubit/depth-1 trials
- trial_005 (wd=0.0001, same arch): F1=0.6077 — lower F1 with regularization
- Small pilot size; single-seed; not conclusive, but suggests weight_decay may hurt DV head F1

---

## 9. Validation Checks

| Check | Result |
|---|---|
| 5/5 trials completed | ✅ PASS |
| 0/5 failed/invalid/unstable | ✅ PASS |
| Pilot verdict ≥ PASS_THRESHOLD (3) | ✅ PASS |
| All gradient_health=PASS | ✅ PASS |
| All nan_inf=PASS | ✅ PASS |
| All params < 100K ceiling | ✅ PASS (max 598) |
| Summary JSON written | ✅ `experiments/results/q34b_dv_nas_pilot_summary.json` |
| Pareto CSV updated | ✅ `experiments/leaderboards/q34b_dv_pareto.csv` |
| NumPy warning | ⚠️ NON-FATAL (all trials complete normally) |

---

## 10. Artifacts

| Artifact | Path | Status |
|---|---|---|
| Pilot report | `reports/q34b_full_lite_dv_nas_pilot.md` | COMMITTED (this file) |
| Pareto CSV | `experiments/leaderboards/q34b_dv_pareto.csv` | COMMITTED |
| Summary JSON | `experiments/results/q34b_dv_nas_pilot_summary.json` | NOT COMMITTED (gitignored) |
| Trial configs (5) | `configs/experiments/q34b_dv_nas/q34b_trial_00*.yaml` | COMMITTED (in Docker; not in local repo) |

---

## 11. Q34B Status

**Q34B-full-lite COMPLETE.** The DV NAS pilot has now executed at both reduced budget (Q34B-Parallel-Lite: 2 epochs) and full budget (Q34B-full-lite: 4 epochs). Combined results:

| Run | Best AUROC | Best F1 | Best params | Wall time |
|---|---|---|---|---|
| Q34B sequential (4 ep) | 0.6551 | 0.6289 | 280 | 11,491 s |
| Q34B-Parallel-Lite (2 ep) | 0.6551† | 0.6289† | 280 | 4,621 s |
| **Q34B-full-lite (4 ep, thread-cap)** | **0.6551** | **0.6356** | **280** | **5,917 s** |

†Q34B-Parallel-Lite sampled the same random configs (same seed=42); trial_004 was the same architecture.

**Q34B DV NAS is now fully characterized at 4-epoch budget.** Q34C (CV NAS) is next.

---

## 12. Q34C Gate Status

| Gate | Status |
|---|---|
| Q34B-full-lite COMPLETE | ✅ |
| Q34B-Parallel-Lite COMPLETE | ✅ |
| EXP-005 thread-cap COMPLETE | ✅ |
| Q34C-Preflight COMPLETE | ✅ |
| Q34C scripts exist | ✅ (`train_q34c_cv_candidate.py`, `run_q34c_cv_nas_pilot.py`) |
| Q34C smoke validation | ✅ PASS (Q34C-SMOKE-SINGLE-TRIAL) |
| **Q34C full pilot: READY** | ✅ |

```bash
docker exec -i docker-qstrata-gpu-1 \
  python3 /workspace/scripts/run_q34c_cv_nas_pilot.py \
    --trials 5 --seed 42 --epochs 2 \
    --parallel --max-workers 5 --thread-cap 2
```
