# Q34C CV NAS Pilot — MVP Results

**Slice:** Q34C  
**Branch:** `feature/q34b-runtime-hotfix`  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Status:** PASS — 5/5 trials completed, 2-member Pareto set, all stability=valid

---

## 1. Execution Configuration

| Parameter | Value |
|---|---|
| Script | `run_q34c_cv_nas_pilot.py --trials 5 --seed 42 --epochs 2 --parallel --max-workers 5 --thread-cap 2` |
| Search space | `q34c_v1` |
| Trials | 5 |
| Epochs per trial | 2 |
| Execution | Parallel (ThreadPoolExecutor, max_workers=5) |
| Thread cap | 2 (OMP/MKL/OPENBLAS per subprocess) |
| Pass threshold | 3 / 5 |
| Displacement cap | 2.0 (pilot-fixed; `tanh(disp_raw) × 2.0` via `CappedGaussianAnsatz`) |
| Wall time | **335.6 s** (~5.6 min) |
| Verdict | **PASS** |

---

## 2. Sampled Configurations

| Trial | n_modes | depth | sq_cap | comp_dim | LR | WD |
|---|---|---|---|---|---|---|
| q34c_trial_001 | 4 | 1 | 0.5 | 8 | 0.0005 | 0.0 |
| q34c_trial_002 | 1 | 1 | 0.5 | 2 | 0.001 | 0.0001 |
| q34c_trial_003 | 1 | 1 | 0.5 | 2 | 0.001 | 0.0 |
| q34c_trial_004 | 4 | 3 | 0.5 | 8 | 0.001 | 0.0001 |
| q34c_trial_005 | 1 | 2 | 1.5 | 2 | 0.001 | 0.0 |

**Pilot substitutions applied (all trials):**
- `beam_splitter_topology` → `circular` (hardcoded in ansatz)
- `readout_strategy` → `first_moments` (mu_final only)
- `encoding_strategy` → `displacement_encoding` (basic, no reuploading)
- `covariance_parameterization` → `symplectic_parameterization`
- `displacement_cap` → `2.0` (pilot-fixed; not searched)

**Constraint enforced:** `compression_dim = 2 × n_modes` at config-generation time.

---

## 3. Trial Results

| Trial | AUROC | F1 | Accuracy | Params | Latency | Stability | Pareto |
|---|---|---|---|---|---|---|---|
| q34c_trial_001 (m=4 d=1) | 0.6593 | 0.6390 | 0.6153 | 1,070 | 3.35 ms | valid | ❌ |
| q34c_trial_002 (m=1 d=1) | 0.6617 | 0.6373 | 0.6278 | **269** | **1.86 ms** | valid | ✅ |
| q34c_trial_003 (m=1 d=1) | 0.6575 | 0.6106 | 0.6278 | **269** | 2.26 ms | valid | ❌ |
| q34c_trial_004 (m=4 d=3) | 0.6554 | 0.5600 | 0.6413 | 1,110 | 4.89 ms | valid | ❌ |
| q34c_trial_005 (m=1 d=2) | **0.6623** | **0.6463** | 0.6201 | 274 | 2.00 ms | valid | ✅ |

**5/5 completed. 0 failed. 0 invalid. 0 unstable. Verdict: PASS.**  
All trials passed Gaussian-state stability checks: covariance PSD, finite mu, finite cov, gradient health.

---

## 4. CV Pareto Frontier (Stability-Filtered)

| Trial | AUROC | F1 | Params | Latency | n_modes | depth | sq_cap |
|---|---|---|---|---|---|---|---|
| q34c_trial_002 | 0.6617 | 0.6373 | 269 | 1.86 ms | 1 | 1 | 0.5 |
| q34c_trial_005 | **0.6623** | **0.6463** | 274 | 2.00 ms | 1 | 2 | 1.5 |

**Pareto dominance:**
- trial_002 stays: fewer params (269 vs 274); not dominated by trial_005 on params
- trial_005 stays: strictly better AUROC (+0.0006) and F1 (+0.009) vs trial_002; not dominated
- trial_001 dominated by trial_002: same Pareto criterion direction, 1070 vs 269 params with lower AUROC/F1
- trial_003 dominated by trial_002: same params (269), lower AUROC (0.6575 < 0.6617), lower F1
- trial_004 dominated by trial_002: 1110 vs 269 params, lower AUROC, lower F1

**Stability filter:** All 5 trials have `stability_taxonomy=valid`; no stability-based exclusions applied.

Leaderboard: `experiments/leaderboards/q34c_cv_pareto.csv`

---

## 5. Stability Analysis

All 5 trials passed all Gaussian-state validity checks:

| Check | Result |
|---|---|
| Covariance PSD (`eigvalsh` per batch) | ✅ PASS — all 5 trials |
| Covariance finite (no NaN/Inf in cov) | ✅ PASS — all 5 trials |
| First-moment finite (no NaN/Inf in mu) | ✅ PASS — all 5 trials |
| Gradient health | ✅ PASS — all 5 trials |
| NaN/Inf global check | ✅ PASS — all 5 trials |
| Stability taxonomy | `valid` — all 5 trials |
| Displacement tanh cap | ✅ Applied — `tanh(disp_raw) × 2.0` in `CappedGaussianAnsatz.apply()` |

No covariance instability, no Gaussian-state collapse, no symplectic violations observed at any epoch. The bounded squeezing cap (0.5–1.5) and fixed displacement cap (2.0) successfully constrained state evolution.

---

## 6. Performance vs Frozen Baselines

| Model | AUROC | F1 | Params | Source |
|---|---|---|---|---|
| Q17 Classical | 0.6224 | 0.5355 | 23,650 | Binary benchmark |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 574 | Binary benchmark |
| Q22 Tiny Classical | 0.6625 | 0.5961 | 526 | Binary benchmark |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 536 | Binary benchmark |
| **Q34C best AUROC** (trial_005) | **0.6623** | **0.6463** | **274** | This pilot |
| **Q34C best params** (trial_002) | 0.6617 | 0.6373 | 269 | This pilot |

**Key findings vs Q27 CV baseline (536 params):**
- AUROC gap: −0.0085 (Q34C trial_005: 0.6623 vs Q27: 0.6708)
- F1 delta: **+0.0180** (Q34C trial_005: 0.6463 vs Q27: 0.6283) ← Q34C NAS exceeds Q27 F1
- Params: 274 vs 536 → **49% fewer parameters** at better F1

**Key findings vs Q22 Tiny Classical (526 params):**
- Q34C trial_005: AUROC 0.6623 < Q22 0.6625 (−0.0002; near-parity)
- Q34C trial_005: F1 0.6463 > Q22 0.5961 (+0.0502; substantial improvement)
- Params: 274 vs 526 → 48% fewer parameters

**Latency advantage:** CV circuits (1.86–4.89 ms/sample) are 14–50× faster than DV circuits (Q34B: 64–94 ms/sample). The CV head's symplectic matrix ops on small covariance matrices (4×4 for n_modes=2, 2×2 for n_modes=1) are far more efficient than DV statevector simulation with per-sample Python loop.

---

## 7. Wall Time Analysis

| Trial | Wall (s) | n_modes | depth | Per-epoch (~s) |
|---|---|---|---|---|
| q34c_trial_003 | 182.1 | 1 | 1 | ~80 |
| q34c_trial_002 | 182.6 | 1 | 1 | ~80 |
| q34c_trial_005 | 198.1 | 1 | 2 | ~87 |
| q34c_trial_001 | 223.5 | 4 | 1 | ~100 |
| q34c_trial_004 | **335.6** | 4 | 3 | ~154 |

**Total pilot wall time: 335.6 s** (~5.6 min) — determined by the slowest trial (trial_004: n_modes=4, depth=3).  
**Expected was 300–600 s.** Actual: 335.6 s — within expected range, toward lower bound.

**Scaling observations:**
- n_modes=1 vs n_modes=4 (depth=1): ~80s vs ~100s/epoch → ~1.25× overhead (covariance matrix 2×2 vs 8×8)
- depth=1 vs depth=3 (n_modes=4): ~100s vs ~154s/epoch → ~1.54× overhead (3× symplectic layer count, sub-linear due to fixed data loading/backbone cost)

**vs Q34B-full-lite** (5917 s, DV, 4 epochs, thread-cap): Q34C is **17.6× faster** at 2 epochs. Normalized to per-epoch: Q34C ≈ 168 s/epoch-equivalent vs Q34B ≈ 1479 s/epoch. CV circuits are ~8.8× faster per epoch than DV circuits.

---

## 8. Validation Checklist

| Check | Result |
|---|---|
| 5 CV configs generated | ✅ PASS |
| Configs use 2 epochs | ✅ PASS |
| ≥ 3/5 trials completed (5/5) | ✅ PASS |
| Summary JSON created | ✅ `experiments/results/q34c_cv_nas_pilot_summary.json` |
| CV Pareto CSV created/updated | ✅ `experiments/leaderboards/q34c_cv_pareto.csv` |
| `stability_taxonomy` recorded per trial | ✅ PASS (all=`valid`) |
| Invalid Gaussian states excluded from Pareto | ✅ PASS (no exclusions needed) |
| PSD checks executed | ✅ PASS — all trials |
| Finite-state checks executed | ✅ PASS — all trials |
| Wall time documented | ✅ 335.6 s |
| No data/checkpoints/large logs committed | ✅ PASS |

---

## 9. Artifacts

| Artifact | Path | Status |
|---|---|---|
| Pilot report | `reports/q34c_cv_nas_pilot_mvp.md` | COMMITTED (this file) |
| CV Pareto CSV | `experiments/leaderboards/q34c_cv_pareto.csv` | COMMITTED |
| Summary JSON | `experiments/results/q34c_cv_nas_pilot_summary.json` | NOT COMMITTED (gitignored) |

---

## 10. Q34C Status: COMPLETE

Q34C PASS. The CV NAS pilot completes the three-frontier NAS program:

| Pilot | Best AUROC | Best F1 | Best Params | Wall Time |
|---|---|---|---|---|
| Q34A Classical | 0.6835 | 0.6398 | 2,250 | — |
| Q34B DV | 0.6551 | 0.6356 | 280 | 5,917 s |
| **Q34C CV** | **0.6623** | **0.6463** | **269** | **335.6 s** |

**Q35 (Unified Pareto Analysis) is now unblocked.**
