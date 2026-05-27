# Q35 — Unified Pareto Frontier Analysis

**Slice:** Q35  
**Branch:** `feature/q34b-runtime-hotfix`  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Status:** COMPLETE — 3-frontier analysis; 6-trial cross-frontier Pareto set; DV fully dominated; CV is the quantum-efficient frontier

---

## 1. Executive Summary

Q35 performs the unified comparative analysis across all three NAS frontiers completed in Phase 5:

- **Q34A Classical:** 5-trial pilot, 4-epoch budget, 4 Pareto-optimal trials
- **Q34B DV Quantum:** 5-trial pilot, 4-epoch budget, 4 Pareto-optimal trials (within DV frontier)
- **Q34C CV Quantum:** 5-trial pilot, 2-epoch budget, 2 Pareto-optimal trials (stability-filtered)

**Primary finding:** All 4 DV Pareto-optimal trials are strictly dominated by CV trial_002 across all four optimization objectives simultaneously (AUROC, F1, params, latency). DV quantum contributes zero cross-frontier Pareto-optimal configurations. The cross-frontier Pareto set contains 6 trials: 4 classical + 2 CV.

**Canonical recommendations:**
- **Best F1 at minimal params:** Q34C trial_005 — AUROC 0.6623, F1 **0.6463**, 274 params, 2.0 ms/sample
- **Best compact classical:** Q34A trial_004 — AUROC **0.6835**, F1 0.6398, 2,250 params, 1.6 ms/sample
- **Absolute AUROC ceiling (pilot):** Q34A trial_003 — AUROC **0.6869**, F1 0.499, 17,486 params

**Gate release:** Q36 (AWS/Ray Distributed Scaling Design) is now unblocked.

---

## 2. Per-Frontier Pareto Summaries

### 2.1 Classical Frontier (Q34A)

4 Pareto-optimal trials from a 5-trial pilot. All classical trials are cross-frontier Pareto-optimal — no classical trial is dominated by any quantum trial because classical AUROC (0.6826–0.6869) exceeds all quantum AUROC values (≤ 0.6623) across the frontier.

| Trial | Arch | AUROC | F1 | Accuracy | Params | Latency | Epochs |
|---|---|---|---|---|---|---|---|
| q34a_trial_002 | standard_cnn shallow | 0.6826 | 0.5843 | 0.6452 | 2,174 | 1.62 ms | 4 |
| q34a_trial_003 | depthwise_sep medium | **0.6869** | 0.4987 | 0.6302 | 17,486 | 1.47 ms | 4 |
| q34a_trial_004 | depthwise_sep shallow | 0.6835 | **0.6398** | 0.6389 | 2,250 | 1.60 ms | 4 |
| q34a_trial_005 | standard_cnn GELU | 0.6867 | 0.6287 | **0.6519** | 4,754 | **1.40 ms** | 4 |

**Classical frontier characteristics:**
- AUROC range: 0.6826–0.6869 (43 bps spread)
- F1 range: 0.4987–0.6398 (140 bps spread) — substantial variation; trial_003 sacrifices F1 for AUROC
- Params range: 2,174–17,486 (8× spread)
- Latency: 1.40–1.62 ms/sample (CPU inference)
- No stability constraint applies (purely classical architectures)

**Classical trade-off axis:** AUROC vs F1 are negatively correlated at high-AUROC values — trial_003 (highest AUROC, 0.6869) has the lowest F1 (0.4987). The compact trials (002, 004) achieve near-peak AUROC with better F1 balance.

### 2.2 DV Quantum Frontier (Q34B)

4 within-frontier Pareto-optimal trials from a 5-trial pilot (trial_005 dominated within DV). 0 cross-frontier Pareto-optimal trials.

| Trial | n_qubits | depth | AUROC | F1 | Accuracy | Params | Latency | Epochs |
|---|---|---|---|---|---|---|---|---|
| q34b_trial_001 | 2 | 1 | 0.6415 | 0.6356 | 0.6081 | 280 | 83.8 ms | 4 |
| q34b_trial_002 | 2 | 1 (RY) | 0.6464 | 0.5783 | 0.6264 | 280 | 93.9 ms | 4 |
| q34b_trial_003 | 2 | 2 | 0.6462 | 0.6155 | 0.6235 | 292 | 64.7 ms | 4 |
| q34b_trial_004 | 4 | 2 (reupload) | **0.6551** | 0.6289 | 0.6148 | 598 | 67.8 ms | 4 |

**DV frontier characteristics:**
- AUROC range: 0.6415–0.6551 (136 bps spread; well below classical)
- F1 range: 0.5783–0.6356 (573 bps spread)
- Params range: 280–598 (compact)
- Latency: **64.7–93.9 ms/sample** — 14–50× slower than CV, 40–65× slower than classical
- All DV trials dominated cross-frontier by CV trial_002 on all four objectives (see §3)

**DV constraint:** `DVHybridCNNQNN.forward()` executes a per-sample Python loop over statevector simulation. Circuit.matrix() gate embedding chain (~1 ms/sample no-grad, ~334× autograd overhead) makes DV fundamentally latency-limited relative to CV symplectic ops. GPU migration blocked (qcore has no device= propagation). This is not addressable within current architecture without qcore refactor.

### 2.3 CV Quantum Frontier (Q34C)

2 stability-filtered Pareto-optimal trials from a 5-trial pilot. Both are cross-frontier Pareto-optimal.

| Trial | n_modes | depth | sq_cap | AUROC | F1 | Accuracy | Params | Latency | Stability | Epochs |
|---|---|---|---|---|---|---|---|---|---|---|
| q34c_trial_002 | 1 | 1 | 0.5 | 0.6617 | 0.6373 | 0.6278 | **269** | **1.86 ms** | valid | 2 |
| q34c_trial_005 | 1 | 2 | 1.5 | **0.6623** | **0.6463** | 0.6201 | 274 | 2.00 ms | valid | 2 |

**CV frontier characteristics:**
- AUROC range: 0.6617–0.6623 (6 bps spread — tight cluster)
- F1: 0.6373–0.6463 (best F1 in the three-frontier cross-comparison)
- Params: 269–274 (smallest in cross-frontier set — 8.1× fewer than best classical compact)
- Latency: 1.86–2.00 ms/sample (symplectic matrix ops on 2×2 covariance; competitive with classical)
- Stability: all valid; PSD checks, finite-state checks, gradient health all passed
- Note: CV pilot used 2-epoch budget vs 4-epoch for classical and DV — CV AUROC may improve with longer training

---

## 3. Cross-Frontier Dominance Analysis

### 3.1 Dominance Criterion

Trial A **cross-frontier dominates** trial B if:
- `AUROC(A) ≥ AUROC(B)` AND `F1(A) ≥ F1(B)` AND `params(A) ≤ params(B)` AND `latency_ms(A) ≤ latency_ms(B)`
- With at least one strict inequality

Hard exclusions: nan_inf=FAIL, params > 100K, gradient_health=FAIL, stability_taxonomy ≠ 'valid' (CV only).

### 3.2 DV Dominated by CV — Full Dominance Table

CV trial_002 (AUROC=0.6617, F1=0.6373, params=269, latency=1.86ms) dominates all 4 DV Pareto trials on all four objectives simultaneously:

| DV Trial | DV AUROC | CV002 AUROC | ΔA | DV F1 | CV002 F1 | ΔF | DV Params | CV002 Params | ΔP | DV Latency | CV002 Latency | ΔL | Dominated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q34b_trial_001 | 0.6415 | 0.6617 | **+0.0202** | 0.6356 | 0.6373 | **+0.0017** | 280 | 269 | **−11** | 83.8 ms | 1.86 ms | **−81.8 ms** | ✅ YES |
| q34b_trial_002 | 0.6464 | 0.6617 | **+0.0153** | 0.5783 | 0.6373 | **+0.0590** | 280 | 269 | **−11** | 93.9 ms | 1.86 ms | **−92.0 ms** | ✅ YES |
| q34b_trial_003 | 0.6462 | 0.6617 | **+0.0155** | 0.6155 | 0.6373 | **+0.0218** | 292 | 269 | **−23** | 64.7 ms | 1.86 ms | **−62.8 ms** | ✅ YES |
| q34b_trial_004 | 0.6551 | 0.6617 | **+0.0066** | 0.6289 | 0.6373 | **+0.0084** | 598 | 269 | **−329** | 67.8 ms | 1.86 ms | **−65.9 ms** | ✅ YES |

**Verdict:** CV trial_002 strictly dominates all DV Pareto trials on every objective. DV contributes zero cross-frontier Pareto-optimal trials. This is a strong cross-frontier result: the symplectic CV architecture, even at a 2-epoch training budget, outperforms the DV statevector architecture at 4-epoch budget across all measurable dimensions.

### 3.3 Classical vs CV — Complementary Frontiers

Classical and CV are complementary — neither frontier fully dominates the other:

- **Classical wins AUROC:** Best classical AUROC (0.6869) exceeds best CV AUROC (0.6623) by **+0.0246**. Classical AUROC dominates CV AUROC across the entire frontier.
- **CV wins F1 at the top:** Best CV F1 (0.6463, trial_005) exceeds best classical F1 (0.6398, trial_004) by **+0.0065**.
- **CV wins on params:** CV trials (269–274) have 8.0–63.8× fewer parameters than classical compact trials (2,174–17,486).
- **Latency:** Classical (1.40–1.62 ms) is slightly faster than CV (1.86–2.00 ms) — both are sub-3 ms; latency is not a discriminating factor between these frontiers.

**No cross-frontier dominance** between classical and CV trials: classical cannot dominate CV because params/latency favor CV; CV cannot dominate classical because AUROC strongly favors classical.

### 3.4 Cross-Frontier Pareto Set (6 Trials)

| Rank | Trial | Frontier | AUROC | F1 | Params | Latency | Notes |
|---|---|---|---|---|---|---|---|
| 1 | q34a_trial_003 | classical | **0.6869** | 0.4987 | 17,486 | 1.47 ms | Peak AUROC; F1-accuracy trade-off |
| 2 | q34a_trial_005 | classical | 0.6867 | 0.6287 | 4,754 | **1.40 ms** | High AUROC + lowest latency + highest accuracy |
| 3 | q34a_trial_004 | classical | 0.6835 | **0.6398** | 2,250 | 1.60 ms | **Canonical compact classical** |
| 4 | q34a_trial_002 | classical | 0.6826 | 0.5843 | 2,174 | 1.62 ms | Compact; lowest classical params |
| 5 | q34c_trial_005 | cv_quantum | 0.6623 | **0.6463** | 274 | 2.00 ms | **Best F1 cross-frontier; canonical CV** |
| 6 | q34c_trial_002 | cv_quantum | 0.6617 | 0.6373 | **269** | **1.86 ms** | Fewest params cross-frontier; canonical compact CV |

**DV trials (q34b_trial_001–004):** ❌ ALL excluded from cross-frontier Pareto (dominated by cv_trial_002)

---

## 4. Canonical Candidates

Three canonical candidates emerge from the cross-frontier analysis for use as reference configurations in future runs:

### Candidate 1: Compact CV (Efficiency Champion)
**q34c_trial_005** — `n_modes=1, cv_depth=2, sq_cap=1.5, comp_dim=2, LR=0.001, WD=0.0, 2 epochs`
- AUROC: 0.6623 | F1: **0.6463** (best cross-frontier) | Params: **274** | Latency: 2.00 ms
- Stability: valid | Displacement cap: tanh×2.0 | Encoding: displacement | Topology: circular
- **Use case:** Maximum F1 at minimum parameter count. Recommended for embedded/resource-constrained settings.
- **Config:** `configs/experiments/q34c_cv_nas/q34c_trial_005.yaml`

### Candidate 2: Compact Classical (AUROC/F1 Balance)
**q34a_trial_004** — `depthwise_sep_shallow, LR=0.0001, WD=0.001, 4 epochs`
- AUROC: **0.6835** | F1: 0.6398 | Params: 2,250 | Latency: 1.60 ms
- **Use case:** Best classical AUROC/F1 balance at compact scale. Recommended as classical control for any future quantum comparison experiment.
- **Config:** `configs/experiments/q34a_classical_nas/q34a_trial_004.yaml`

### Candidate 3: Peak AUROC Classical (Research Reference)
**q34a_trial_005** — `standard_cnn GELU, 4 epochs`
- AUROC: 0.6867 | F1: 0.6287 | Params: 4,754 | Latency: 1.40 ms | Accuracy: **0.6519**
- **Use case:** Highest multi-metric classical performance (AUROC + accuracy + lowest latency). Reference ceiling for AUROC in any future NAS run.
- Note: trial_003 has marginally higher AUROC (0.6869 vs 0.6867) but trial_005 is strictly better on F1 (0.6287 vs 0.4987), accuracy (0.6519 vs 0.6302), params (4754 vs 17486), and latency. trial_003 is Pareto-optimal only because params and latency can favor it in some formulations; trial_005 is the practical ceiling reference.

---

## 5. Performance vs Frozen Benchmarks

The NAS pilots are compared against the fixed binary benchmarks from Phase 4 (Q17–Q27):

| Model | AUROC | F1 | Params | Source | Status |
|---|---|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 23,650 | Phase 4 | Frozen baseline |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 574 | Phase 4 | Frozen baseline |
| Q22 Tiny Classical | 0.6625 | 0.5961 | 526 | Phase 4 | Frozen baseline |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 536 | Phase 4 | Frozen baseline |
| **Q34A trial_004** | **0.6835** | 0.6398 | 2,250 | Q34A NAS pilot | Cross-frontier Pareto |
| **Q34A trial_005** | **0.6867** | 0.6287 | 4,754 | Q34A NAS pilot | Cross-frontier Pareto |
| **Q34C trial_005** | 0.6623 | **0.6463** | **274** | Q34C NAS pilot | Cross-frontier Pareto |

**Key observations vs frozen benchmarks:**

- **vs Q21 DV (0.6800 AUROC, 574 params):** Best NAS AUROC (Q34A trial_005: 0.6867) exceeds Q21 by +0.0067. Q21 was hand-tuned with early stopping; the NAS result at 5-trial pilot scale suggests classical NAS has found configurations exceeding the hand-tuned DV baseline on AUROC.
- **vs Q22 Tiny Classical (0.6625 AUROC, 526 params):** Q34C trial_005 (0.6623 AUROC, 274 params) nearly matches Q22 AUROC (−0.0002) while using **48% fewer parameters** and achieving +0.0502 higher F1 (0.6463 vs 0.5961).
- **vs Q27 CV (0.6708 AUROC, 536 params):** Q34C NAS pilot does not match Q27 AUROC (−0.0085 gap). However, Q34C trial_005 achieves **+0.018 higher F1** (0.6463 vs 0.6283) with **49% fewer params** (274 vs 536). Q27 was trained to convergence with early stopping; Q34C ran only 2 epochs.
- **vs Q17 Classical (0.6224 AUROC, 23,650 params):** All NAS trials exceed Q17 AUROC. NAS finds better classical architectures than the Phase 4 hand-designed CNN, at a fraction of the parameter count.

---

## 6. Stability Analysis Summary (CV)

CV stability taxonomy was enforced prior to Pareto computation. All 5 Q34C trials passed all stability checks:

| Stability Check | Q34C Result |
|---|---|
| Covariance PSD (eigvalsh per batch) | ✅ PASS — all 5 trials |
| Covariance finite (no NaN/Inf) | ✅ PASS — all 5 trials |
| First-moment finite (no NaN/Inf in mu) | ✅ PASS — all 5 trials |
| Gradient health | ✅ PASS — all 5 trials |
| NaN/Inf global check | ✅ PASS — all 5 trials |
| Stability taxonomy | `valid` — all 5 trials |
| Displacement tanh cap | ✅ Applied — `tanh(disp_raw) × 2.0` |
| Squeezing cap range | 0.5–1.5 (searched) |

No stability exclusions applied. All 5 trials qualified for Pareto computation; 2 entered the Pareto set. The `CappedGaussianAnsatz` displacement cap and bounded squeezing ranges successfully constrained Gaussian state evolution across all trials and epochs.

---

## 7. Latency and Runtime Analysis

### 7.1 Inference Latency (per-sample, production-relevant)

| Frontier | Range | Mechanism | Scalability |
|---|---|---|---|
| Classical | 1.40–1.62 ms | Conv + FC; BLAS-parallel | CPU-linear in depth/width |
| CV Quantum | 1.86–2.00 ms | Symplectic ops on 2×2–8×8 covariance | Sub-linear: O(n_modes³) |
| DV Quantum | 64.7–93.9 ms | Statevector simulation; per-sample Python loop | Exponential: O(2^n_qubits) |

CV latency (1.86–2.00 ms) is nearly indistinguishable from classical (1.40–1.62 ms) at n_modes=1. DV latency (64.7–93.9 ms) is 35–65× slower than CV and 40–67× slower than classical. DV is not viable for real-time inference applications without fundamental architectural change.

### 7.2 Training Wall Time (NAS pilot totals)

| NAS Pilot | Trials | Epochs | Wall Time | Per-epoch (slowest trial) |
|---|---|---|---|---|
| Q34A Classical | 5 | 4 | — | — |
| Q34B DV | 5 | 4 | 5,917 s | ~154 s |
| Q34C CV | 5 | 2 | **335.6 s** | ~87 s |

CV training is **17.6× faster** than DV at comparable trial counts (Q34C 2-epoch vs Q34B 4-epoch). Normalized per-epoch: CV ~168 s/epoch-equivalent vs DV ~1,479 s/epoch (**8.8× faster per epoch**). The CV speedup comes from symplectic matrix operations on compact covariance matrices (2×2 for n_modes=1) vs statevector simulation with 2^n_qubits complex amplitudes and per-sample Python loop.

---

## 8. NAS Hardening Recommendations (for Q36+)

Based on the three-frontier analysis, the following recommendations apply to future NAS execution:

### R1: Prioritize CV over DV for quantum NAS (Strong)
All DV Pareto trials are strictly dominated by CV trial_002 on all four objectives simultaneously. Any future NAS budget allocated to DV circuits produces lower expected returns than CV on AUROC, F1, params, and latency combined. Recommended: allocate quantum NAS budget exclusively to CV until DV architecture fundamentals are addressed (qcore GPU support, vectorized forward pass).

### R2: Expand CV search space (High priority)
Current Q34C search space explored `n_modes ∈ {1, 4}`, `cv_depth ∈ {1, 2, 3}`, `sq_cap ∈ {0.5, 1.5}`. All Pareto CV configs use `n_modes=1, cv_depth ∈ {1, 2}`. Recommended extensions:
- Expand `n_modes` to {1, 2, 3} for finer granularity
- Extend `cv_depth` to {1, 2, 3, 4}
- Add `squeezing_cap ∈ {0.5, 1.0, 1.5, 2.0}`
- Increase trial budget to 20+ for better Pareto coverage
- Increase epoch budget to 4 (matching Q34A/Q34B) for fair AUROC comparison

### R3: Use CV trial_005 as the quantum reference config (High priority)
Q34C trial_005 (n_modes=1, depth=2, sq=1.5, 274 params, F1=0.6463) is the canonical quantum candidate. In any multi-frontier comparison, use trial_005 as the CV anchor rather than Q27 (hand-tuned, 536 params) — the NAS-found config is more compact at competitive F1.

### R4: Classical compact sweep (Medium priority)
Q34A classical Pareto shows that `compact` architectures (2,174–2,250 params) achieve AUROC within 6 bps of the peak-params trial (17,486 params). A focused classical NAS sweep over the 1,000–5,000 param range with 20+ trials and 4 epochs may identify configurations that challenge Q34A trial_004/005 while reducing params further.

### R5: Epoch parity for cross-frontier comparison (Medium priority)
CV ran 2 epochs vs 4 epochs for classical/DV. CV trial_005 AUROC (0.6623) may improve with 4 epochs — Q34C epoch curves showed positive AUROC trends across epochs for all trials. A 4-epoch CV NAS run (or dedicated CV training run with trial_005 config) would close the epoch-budget gap for a more rigorous cross-frontier comparison.

### R6: Retain stability taxonomy for all future CV runs (Required)
The stability-aware Pareto filter is essential for CV circuits. While all 5 Q34C trials were `valid`, expanding the search space (larger n_modes, higher sq_cap) may produce `covariance_not_psd` or `nan_state` failures. The existing `CappedGaussianAnsatz` + stability taxonomy infrastructure should be preserved and applied to any future CV NAS run.

---

## 9. Q35 Artifact Index

| Artifact | Path | Status |
|---|---|---|
| Q35 unified report | `reports/q35_unified_pareto_frontier_analysis.md` | COMMITTED (this file) |
| Unified frontier CSV | `experiments/leaderboards/q35_unified_frontier.csv` | COMMITTED |
| Q34A Pareto CSV | `experiments/leaderboards/q34a_classical_pareto.csv` | Pre-existing (committed) |
| Q34B Pareto CSV | `experiments/leaderboards/q34b_dv_pareto.csv` | Pre-existing (committed) |
| Q34C Pareto CSV | `experiments/leaderboards/q34c_cv_pareto.csv` | Pre-existing (committed) |

---

## 10. Q35 Status: COMPLETE

**Cross-frontier Pareto set: 6 trials (4 classical + 2 CV + 0 DV)**

| Ranking | Trial | Frontier | AUROC | F1 | Params | Latency | Cross-Frontier Pareto |
|---|---|---|---|---|---|---|---|
| 1 (peak AUROC) | q34a_trial_003 | Classical | 0.6869 | 0.4987 | 17,486 | 1.47 ms | ✅ |
| 2 | q34a_trial_005 | Classical | 0.6867 | 0.6287 | 4,754 | 1.40 ms | ✅ |
| 3 | q34a_trial_004 | Classical | 0.6835 | 0.6398 | 2,250 | 1.60 ms | ✅ |
| 4 | q34a_trial_002 | Classical | 0.6826 | 0.5843 | 2,174 | 1.62 ms | ✅ |
| 5 (best F1) | q34c_trial_005 | CV Quantum | 0.6623 | **0.6463** | 274 | 2.00 ms | ✅ |
| 6 (min params) | q34c_trial_002 | CV Quantum | 0.6617 | 0.6373 | **269** | 1.86 ms | ✅ |
| — (dominated) | q34b_trial_001–004 | DV Quantum | 0.6415–0.6551 | 0.5783–0.6356 | 280–598 | 64.7–93.9 ms | ❌ |

**Key conclusions:**
1. DV quantum is fully dominated cross-frontier — zero cross-frontier Pareto-optimal trials
2. CV quantum achieves best F1 and fewest params in the three-frontier cross-comparison
3. Classical quantum holds the AUROC advantage across the entire frontier
4. The classical vs CV trade-off axis (AUROC vs F1+params) is the operative architectural decision for future work

**Q36 (AWS/Ray Distributed Scaling Design) is now unblocked.** Phase 5b gate: Q34A ✓ + Q34B ✓ + Q34C ✓ + Q35 ✓.
