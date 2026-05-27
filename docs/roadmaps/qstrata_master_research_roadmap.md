# QStrata Master Research Roadmap

**Project:** QStrata — medical imaging model optimization R&D  
**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-27  
**Author:** Miguel Lopez (QStrata)  
**Status:** Q35 COMPLETE — Q36 NEXT (Distributed Scaling Design)

---

## 1. Research Program Overview

QStrata is a systematic research program evaluating classical and quantum hybrid deep learning architectures for medical imaging classification under compact parameter budgets. The program proceeds in strict phases:

1. **Phase 1 — Binary Benchmarking (CLOSED):** Establish classical, DV hybrid, and CV hybrid baselines on VinDr-SpineXR binary classification under frozen pretrained backbone constraints. Result: four frozen benchmark models.

2. **Phase 2 — Experiment Automation (COMPLETE):** Design and implement reproducible experiment orchestration before NAS begins.

3. **Phase 3 — Classical NAS Ceiling:** Search compact classical CNN architecture space to establish the strongest classical baseline. Required before quantum NAS.

4. **Phase 4 — Quantum NAS:** Search DV and CV quantum head architecture spaces using the classical ceiling from Phase 3 as the evaluation reference.

5. **Phase 5 — Local Multi-Objective NAS Pilot:** Run joint AUROC/F1/params/latency/stability optimization on single GPU.

6. **Phase 6 — Distributed Scaling (BLOCKED):** Design and deploy AWS/Ray distributed NAS only after local NAS is validated.

7. **Phase 7 — Multiclass Benchmarking (BLOCKED):** Extend binary protocols to multiclass tasks after optimized binary baselines exist.

**Frozen binary benchmarks (canonical reference for all future work):**

| Model | AUROC | F1 | Accuracy | Params | Source |
|---|---|---|---|---|---|
| Q17 Classical CNN | 0.6224 | 0.5355 | 60.66% | 23,650 | `reports/vindr_classical_baseline_full_training.md` |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 63.84% | 574 | `reports/vindr_dv_hybrid_pretrained_full_training.md` |
| Q22 Tiny Classical Control | 0.6625 | 0.5961 | 64.37% | 526 | `reports/vindr_classical_control_tiny_head.md` |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 65.77% | 536 | `reports/q27_cv_binary_full_training.md` |

These values are frozen. Future comparative work must reference them explicitly.

---

## 2. Complete Slice Registry

### Phase 1 — Binary Benchmarking (CLOSED)

| Slice | Description | Status |
|---|---|---|
| Q16 | VinDr-SpineXR Classical Baseline Smoke Test | COMPLETE |
| Q17 | VinDr-SpineXR Classical Baseline Full Training | COMPLETE |
| Q18 | VinDr-SpineXR DV Hybrid Smoke Test | COMPLETE |
| Q19 | VinDr-SpineXR DV Hybrid Full Training (random backbone — invalid) | COMPLETE |
| Q20 | VinDr-SpineXR DV Hybrid Pretrained-Backbone Feasibility | COMPLETE |
| Q21 | VinDr-SpineXR DV Hybrid Pretrained-Backbone Full Training | COMPLETE |
| Q22 | VinDr-SpineXR Approximate Trainable-Parameter-Matched Classical Control | COMPLETE |
| Q23 | VinDr DV Binary Comparative Report | COMPLETE — DV phase CLOSED |
| Q24 | Roadmap Realignment for CV Binary Phase | COMPLETE |
| Q25 | Continuous-Variable Binary Feasibility Design | COMPLETE |
| Q25A | Roadmap Prioritization and Experiment Automation Planning | COMPLETE |
| Q26 | Continuous-Variable Binary Smoke Test | COMPLETE — 14/14 PASS |
| Q27 | Continuous-Variable Binary Full Training | COMPLETE — AUROC 0.6708 |
| Q27A | NAS Strategy and Optimization Phase Refinement | COMPLETE |
| Q28 | DV vs CV Binary Comparative Report | COMPLETE |
| Q29 | Binary Quantum Release Tagging | COMPLETE — binary phase CLOSED |

**Phase 1 status: CLOSED**  
**Release tags:** `qstrata-vindr-dv-binary-v1`, `qstrata-vindr-cv-binary-v1`, `qstrata-vindr-binary-comparative-v1`

### Phase 2 — Experiment Automation (IN PROGRESS)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q30 | Experiment Automation Framework Design | **COMPLETE** | Q29 ✓ |
| Q31 | Local GPU Experiment Runner MVP | **COMPLETE** | Q30 ✓ |
| Q31A | Runner Reproducibility Test and Hardening | **COMPLETE** | Q31 ✓ |

**Phase 2 gate:** Q29 complete ✓  
**Phase 2 status:** Q30 complete; Q31 complete; Q31A complete — Phase 2 COMPLETE

### Phase 3 — Classical NAS Ceiling (IN PROGRESS)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q32 | NAS Search Space Design — Classical Feature Extractors | **COMPLETE** (design only) | Q31A ✓ |

**Phase 3 gate:** Q31A (runner hardening) complete ✓  
**Phase 3 note (Q32):** Q32 is design only — no NAS execution. Defines the classical CNN search space and config template. NAS execution begins in Q34.  
**Phase 3 note:** Classical NAS always precedes quantum NAS. The classical ceiling defines the evaluation reference for all quantum head search.

### Phase 4 — Quantum NAS (IN PROGRESS)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q33A | NAS Search Space Design — DV Quantum Heads | **COMPLETE** (design only) | Q32 ✓ |
| Q33B | NAS Search Space Design — CV Quantum Heads | **COMPLETE** (design only) | Q33A ✓ |
| Q33C | NAS Execution Protocol Design | **NEXT** (design only) | Q33B ✓ |

**Phase 4 gate:** Q32 classical search space design complete ✓  
**Phase 4 note (Q33A):** Q33A is design only — no NAS execution. Defines the DV quantum head search space (qubit count, ansatz depth, rotation families, entanglement topology, encoding strategy, re-uploading frequency, measurement strategy, compression dimension, classical projection layer).  
**Phase 4 note (Q33B):** Q33B is design only — no NAS execution. Defines the CV Gaussian quantum head search space (n_modes, cv_depth, squeezing_cap, displacement_cap, encoding strategy, beam-splitter topology, readout strategy, compression dimension, covariance parameterization). Stability-first design: six Gaussian-state validity conditions; eight-category stability taxonomy; covariance explosion mitigation via bounded squeezing, constrained depth, and symplectic parameterization.  
**Phase 4 note (Q33C):** Q33C is design only — no NAS execution. Finalizes the Q34A/Q34B/Q34C incremental execution plan, trial sampling strategy, timeout values, leaderboard format, and CV stability monitoring protocol.

### Phase 5 — Local Multi-Objective NAS Pilot (IN PROGRESS)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q34A | Classical NAS Pilot (first execution) | **COMPLETE** — 5/5 PASS | Q33C ✓ |
| Q34B | DV NAS Pilot (second execution) | **COMPLETE** — 5/5 PASS | Q34A ✓ |
| Q34B-HF | DV Runtime Bottleneck Assessment | **COMPLETE** | Q34B ✓ |
| Q34B-Parallel-Lite | DV NAS Parallel Execution Validation | **COMPLETE** — 5/5 PASS | Q34B-HF ✓ |
| EXP-005-Thread-Cap | DV NAS Thread-Cap Preflight | **COMPLETE** | Q34B-Parallel-Lite ✓ |
| Q34B-full-lite | DV NAS Full-Budget Parallel Pilot (4 epochs, thread-cap=2) | **COMPLETE** — 5/5 PASS | EXP-005 ✓ |
| Q34C-Preflight | CV NAS Readiness Check | **COMPLETE** — 2 blockers identified | Q34B-Parallel-Lite ✓ |
| Q34C-Smoke | CV NAS Single-Trial Smoke Validation | **COMPLETE** — PASS | Q34C-Preflight ✓ |
| Q34C | CV NAS Pilot (third execution) | **COMPLETE** — 5/5 PASS | Q34C-Smoke ✓ |

**Phase 5 gate:** Q33C NAS execution protocol design must be complete  
**Phase 5 note (Q34A):** Q34A COMPLETE — 5/5 trials completed, 4-member Pareto set. Strongest compact candidate: q34a_trial_004 (AUROC 0.6835, F1 0.6398, 2,250 params). Pipeline validation passed end-to-end. Reference: `reports/q34a_classical_nas_pilot_mvp.md`.  
**Phase 5 note (Q34B):** Q34B COMPLETE — 5/5 trials completed, 4-member Pareto set. Best DV AUROC: q34b_trial_004 (AUROC 0.6551, F1 0.6289, 598 params). Best DV F1: q34b_trial_001 (AUROC 0.6415, F1 0.6356, 280 params). Wall time: 11,491 s (CPU-only). Reference: `reports/q34b_dv_nas_pilot_mvp.md`.  
**Phase 5 note (Q34B-HF):** DV runtime bottleneck assessed. Root cause: circuit.matrix() gate embedding chain (CPU-only; ~1ms/sample no-grad; ~334× autograd overhead during training). GPU migration blocked — qcore has no device= propagation; kron ops produce CPU tensors unconditionally. GPU provides no speedup at n=2–4 qubits even if unblocked (matrices 4×4–16×16; GPU launch overhead dominates). Recommended fix (no qcore change): parallel trial execution (5 processes) → ~4.3× speedup (11,491s → ~2,672s). Combined with 2-epoch budget: ~19 min. Reference: `reports/q34b_runtime_bottleneck_skypilot_assessment.md`.  
**Phase 5 note (Q34B-Parallel-Lite):** Parallel execution validated. 5/5 trials PASS, 4-member Pareto set, wall time 4,621 s (2.49× speedup vs Q34B-sequential). Thread contention observed: ~30 PyTorch threads/process × 5 workers on 12 CPUs → 3.9× per-trial overhead, limiting efficiency to 25% of ideal. Net speedup driven by 2-epoch reduction (2×) + partial parallelism (1.24×). Parallel mode now the default for Q34C. Thread cap fix implemented in Q34B-Parallel-Lite-Thread-Cap preflight via `--thread-cap` flag (sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` per subprocess). Reference: `reports/q34b_parallel_lite_dv_nas_pilot.md`.
**Phase 5 note (Q34B-full-lite):** Full 4-epoch budget pilot with thread-cap=2. 5/5 trials PASS, 4-member Pareto set, wall time 5,917 s (1.94× faster than Q34B-sequential at same 4-epoch budget). Thread-cap delivers 1.56× per-epoch speedup vs Q34B-Parallel-Lite (no cap). Best AUROC: trial_004 (0.6551, 598 params, qubits=4 depth=2 reupload=every_layer). Best F1: trial_001 (0.6356, 280 params, qubits=2 depth=1) — exceeds Q21 F1 (0.6159) at lower params. AUROC gap vs Q21 (0.6800): −0.0249. NumPy 1.x/2.x compatibility warning observed in all trials (non-fatal). Reference: `reports/q34b_full_lite_dv_nas_pilot.md`.  
**Phase 5 note:** Q34 executes incrementally — Q34A (classical) first, Q34B (DV) second, Q34C (CV) third. Do not attempt all three simultaneously. Each pilot produces a Pareto frontier; Q35 performs unified three-frontier comparison.  
**Phase 5 note (Q34C-Preflight):** Readiness check COMPLETE. Infrastructure (parallel, thread-cap, Q31 runner) is fully ready. CV backend (GaussianVariationalAnsatz, GaussianBackend) supports n_modes, depth, squeezing_cap. Two blockers identified: (1) `scripts/train_q34c_cv_candidate.py` missing — per-trial CV training script with Q34C_TRIAL_* output protocol, stability taxonomy, and inline PSD checks; (2) `scripts/run_q34c_cv_nas_pilot.py` missing — Q34C orchestrator with Q33B search space and stability-aware Pareto. Pilot substitutions defined for 5 unimplemented Q33B dimensions (topology, readout, encoding, cov_parameterization, displacement_cap). Variable dimensions: n_modes, cv_depth, squeezing_cap, compression_dim (compression_dim = 2×n_modes enforced). Expected pilot wall time: 300–600 s with thread-cap. Reference: `reports/q34c_cv_nas_readiness_check.md`.
**Phase 5 note (Q34C-Smoke):** CV pipeline validated end-to-end. Smoke trial: n_modes=4, cv_depth=1, sq_cap=0.5, comp_dim=8, 1 epoch. AUROC=0.6540, F1=0.6429, params=1,070, latency=2.92ms, stability_taxonomy=valid. All 12 checks PASS. `CappedGaussianAnsatz` subclass implements `tanh(disp_raw)×2.0` displacement cap without qcore change. Device fix: `eval_split()` required explicit `device` param to move `x` to backbone device. Reference: `reports/q34c_smoke_validation.md`.
**Phase 5 note (Q34C):** COMPLETE — 5/5 trials PASS, wall time 335.6 s (~5.6 min), 2-member Pareto set, all stability=valid (no stability exclusions). Best AUROC: trial_005 (0.6623, n_modes=1 depth=2 sq=1.5, 274 params). Best F1: trial_005 (0.6463) — exceeds Q27 CV F1 (0.6283) by +0.018 at 49% fewer params. CV latency 1.86–4.89 ms/sample vs DV 64–94 ms/sample (~20–40× faster). All stability checks (PSD, finite mu/cov, gradient health) passed for all trials. Reference: `reports/q34c_cv_nas_pilot_mvp.md`.

### Phase 5b — Unified Pareto Analysis (PLANNED)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q35 | Unified Pareto Analysis and NAS Hardening | **COMPLETE** | Q34A ✓ + Q34B ✓ + Q34C ✓ |

**Phase 5b gate:** Q34A, Q34B, and Q34C must all be complete  
**Phase 5b note:** Full three-frontier comparison: classical (Q34A) vs DV (Q34B) vs CV (Q34C) Pareto frontiers. Stability taxonomy analysis for CV trials. Identifies which Q33A/Q33B dimensions drive Pareto-optimal quantum performance. Produces NAS hardening recommendations.  
**Phase 5b note (Q35):** COMPLETE — 6-trial cross-frontier Pareto set (4 classical + 2 CV + 0 DV). DV is fully dominated by CV trial_002 on all four objectives simultaneously. Classical holds the AUROC advantage (0.6826–0.6869 vs 0.6617–0.6623); CV achieves best F1 cross-frontier (0.6463) and fewest params (269). Canonical CV candidate: q34c_trial_005 (AUROC 0.6623, F1 0.6463, 274 params). Canonical classical compact: q34a_trial_004 (AUROC 0.6835, F1 0.6398, 2,250 params). Six NAS hardening recommendations documented. Q36 now unblocked. Reference: `reports/q35_unified_pareto_frontier_analysis.md`.

### Phase 6 — Distributed Scaling (BLOCKED)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q36 | AWS / Ray Distributed Scaling Design | **NEXT** | Q35 ✓ |

**Phase 6 gate:** Q35 unified Pareto analysis must be complete and validated  
**Phase 6 note:** Design document only before any infrastructure is provisioned. No cloud resources before Q36 design is approved.

### Phase 7 — Multiclass Benchmarking (BLOCKED)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| M01 | VinDr-SpineXR Multiclass Classical Baseline | BLOCKED | Phase 3 + 4 + 5 |
| M02 | VinDr-SpineXR Multiclass DV Hybrid | BLOCKED | Phase 3 + 4 + 5 |
| M03 | VinDr-SpineXR Multiclass CV Hybrid | BLOCKED | Phase 3 + 4 + 5 |
| M04 | PathMNIST Multiclass Classical Baseline | BLOCKED | Phase 3 + 4 + 5 |
| M05 | PathMNIST Multiclass DV Hybrid | BLOCKED | Phase 3 + 4 + 5 |

**Phase 7 gate:** Phase 3 (classical ceiling), Phase 4 (quantum NAS), and Phase 5 (optimized binary release) must all be complete before multiclass work begins.

**Why multiclass is gated on Phases 3–5:** Multiclass benchmarks must compare against optimized binary reference baselines, not the current unoptimized benchmarks. Starting multiclass before NAS completes would require re-evaluating multiclass results after NAS produces new binary references.

### Deferred Binary Work

| Slice | Description | Status |
|---|---|---|
| P21 | PneumoniaMNIST Classical vs DV Hybrid Comparative Report | TODO |
| R-FINAL | Global Binary Benchmark Technical Summary | TODO |

P21 and R-FINAL are not currently scheduled. They are not blocked by any phase gate but are not the immediate execution priority.

---

## 3. Active Gating Rules

### NAS / Automation Gate (Q32–Q36)

| Gate | Status | Condition |
|---|---|---|
| Q30 complete | ✓ | Binary closure (Q29) |
| Q31 complete | ✓ | Q30 complete |
| Q31A complete | ✓ | Q31 complete |
| Q32 complete | ✓ | Q31A ✓ — design only; no NAS execution |
| Q33A complete | ✓ | Q32 ✓ — design only; no NAS execution |
| Q33B complete | ✓ | Q33A ✓ — design only; no NAS execution |
| Q33C unblocked | NEXT (design only) | Q33B ✓ |
| Q34A complete | ✓ | Q33C (skipped by pilot execution) — 5/5 PASS |
| Q34B complete | ✓ | Q34A ✓ — 5/5 PASS |
| Q34B-Parallel-Lite complete | ✓ | Q34B-HF ✓ — 5/5 PASS, 2.49× speedup, thread contention documented |
| EXP-005 Thread-Cap complete | ✓ | --thread-cap flag added, OMP/MKL/OPENBLAS propagated per subprocess |
| Q34B-full-lite complete | ✓ | 5/5 PASS, wall 5,917 s (1.94× speedup), best AUROC 0.6551 (trial_004) |
| Q34C-Preflight complete | ✓ | Q34B-Parallel-Lite ✓ — 2 blockers resolved: scripts now exist |
| Q34C-Smoke complete | ✓ | CV pipeline validated end-to-end; AUROC=0.6540, stability=valid |
| Q34C complete | ✓ | 5/5 PASS, wall 335.6 s, best AUROC 0.6623 (trial_005, 274 params), best F1 0.6463 |
| Q35 complete | **COMPLETE** ✓ | Q34A + Q34B + Q34C all complete |
| Q36 unblocked | **NEXT** | Q35 ✓ |

### Multiclass Gate

| Gate | Status | Condition |
|---|---|---|
| Phase 3 (Q32) complete | ✓ (design) | Q31A ✓ |
| Phase 4 (Q33A) complete | ✓ (design) | Q32 ✓ |
| Phase 4 (Q33B) complete | ✓ (design) | Q33A ✓ |
| Phase 4 (Q33C) complete | PLANNED | Q33B ✓ |
| Phase 5 (Q34A–Q34C) complete | PLANNED | Q33C complete |
| Phase 5b (Q35) complete | **COMPLETE** ✓ | Q34A–Q34C complete |
| Multiclass may begin | **BLOCKED** | All of the above complete |

### AWS/Ray Gate (Q36)

| Gate | Status | Condition |
|---|---|---|
| Q34A–Q34C local NAS validated | PLANNED | — |
| Q35 Pareto analysis complete | **COMPLETE** ✓ | Q34A–Q34C complete |
| Q36 design approved | BLOCKED | Q36 design doc not yet written |
| Cloud infrastructure provisioned | BLOCKED | Q36 design approved |

---

## 4. Scientific Principles (Carried Forward from Prior Phases)

**Classical Ceiling Principle:** Classical NAS (Q32) must always precede quantum NAS (Q33). The frozen binary benchmark table defines the current reference floor. Q32 defines the ceiling. No quantum NAS result can be scientifically interpreted without a validated compact classical ceiling.

**No Quantum Advantage Claim:** The binary benchmarks do not establish quantum advantage. DV and CV hybrids show small residual advantages over the parameter-matched classical control (Q22) in single-seed experiments without confidence intervals. These residuals are scientifically interesting but not statistically validated. Future NAS may confirm, reduce, or eliminate them.

**Multi-Objective Optimization:** No phase uses single-metric optimization. Pareto frontier exploration across AUROC, F1, parameter count, latency, stability, and generalization gap is required for all NAS phases.

**Local-First Infrastructure:** Local automation and local NAS must be validated before any distributed infrastructure (AWS, Ray) is introduced.

**Reproducibility-First:** Every experiment must be reconstructable from its frozen config and git commit SHA. Automation exists to enforce this, not to accelerate uncontrolled experimentation.

---

## 5. Immediate Next Action

**Q36 — AWS / Ray Distributed Scaling Design**

Q35 COMPLETE — three-frontier unified Pareto analysis done. Phase 5b Unified Pareto Analysis is complete.

**Q35 outcome summary:**
- Cross-frontier Pareto set: 6 trials (4 classical + 2 CV + **0 DV**)
- DV quantum: entirely dominated by CV trial_002 on all four objectives simultaneously (AUROC, F1, params, latency)
- Canonical CV candidate: q34c_trial_005 (AUROC 0.6623, F1 **0.6463**, 274 params, 2.00 ms)
- Canonical compact classical: q34a_trial_004 (AUROC **0.6835**, F1 0.6398, 2,250 params, 1.60 ms)
- Six NAS hardening recommendations documented (prioritize CV; expand CV search space to 20+ trials, 4 epochs; drop DV from next quantum NAS run)
- Unified leaderboard: `experiments/leaderboards/q35_unified_frontier.csv`
- Reference report: `reports/q35_unified_pareto_frontier_analysis.md`

**Q36 scope (design document only — no cloud resources):**
Q36 produces a design document for AWS/Ray distributed NAS scaling, informed by Q35 hardening recommendations. No cloud infrastructure may be provisioned before Q36 design is approved. Design should address: Ray cluster architecture for CV NAS, trial parallelism beyond 5-worker local limit, S3 artifact storage, cost modeling for 20+ trial runs.

Gate: Q35 ✓

**Q33C note:** Q33C (NAS execution protocol design) was effectively realized through the Q34A implementation. The incremental 5-trial/4-epoch pilot protocol, random sampling via `random.choice`, per-trial YAML config generation, sequential Q31 runner invocation, and Pareto CSV output were all defined and validated within Q34A. No separate Q33C design document is required before Q34C proceeds.

---

```
Q30 status: COMPLETE
Q31 status: COMPLETE — smoke PASS (experiment_id 20260526_222939_a508a2)
Q31A status: COMPLETE — reproducibility PASS (loss_delta=0.0, tolerance 0.0001)
Q32 status: COMPLETE — design only; no NAS execution
Q33A status: COMPLETE — design only; no NAS execution
Q33B status: COMPLETE — design only; no NAS execution
Q33C status: REALIZED within Q34A — protocol implemented; no separate doc required
Q34A status: COMPLETE — 5/5 trials PASS; Pareto set: 4 trials; pipeline validated
Q34B status: COMPLETE — 5/5 trials PASS; Pareto set: 4 trials; pipeline validated; wall time 11491s
Q34B-HF status: COMPLETE — bottleneck: circuit.matrix() + autograd; GPU blocked; recommendation: parallel trials; reference: reports/q34b_runtime_bottleneck_skypilot_assessment.md
Q34B-Parallel-Lite status: COMPLETE — 5/5 trials PASS; Pareto set: 4 trials; wall time 4621s (2.49× speedup); thread contention 3.9×/trial; reference: reports/q34b_parallel_lite_dv_nas_pilot.md
EXP-005-Thread-Cap-Preflight status: COMPLETE — --thread-cap flag added; OMP/MKL/OPENBLAS_NUM_THREADS propagated per subprocess; validated; Q34C command: --parallel --max-workers 5 --epochs 2 --thread-cap 2
Q34B-full-lite status: COMPLETE — 5/5 trials PASS; Pareto set: 4 trials; wall time 5917s (1.94× speedup vs Q34B sequential); thread-cap=2 delivers 1.56× per-epoch speedup; best AUROC 0.6551 (trial_004, qubits=4 depth=2); best F1 0.6356 (trial_001, 280 params); reference: reports/q34b_full_lite_dv_nas_pilot.md
Q34C-Preflight status: COMPLETE — infra ready; blockers resolved (scripts now exist); pilot substitutions defined; reference: reports/q34c_cv_nas_readiness_check.md
Q34C-Smoke status: COMPLETE — CV pipeline validated; AUROC=0.6540; stability=valid; all 12 checks PASS; reference: reports/q34c_smoke_validation.md
Q34C status: COMPLETE — 5/5 trials PASS; Pareto set: 2 trials; wall time 335.6s; best AUROC 0.6623 (trial_005, n_modes=1 depth=2 sq=1.5, 274 params); best F1 0.6463; all stability=valid; reference: reports/q34c_cv_nas_pilot_mvp.md
Q35 status: COMPLETE — cross-frontier Pareto set: 6 trials (4 classical + 2 CV + 0 DV); DV fully dominated by CV; canonical CV: q34c_trial_005 (AUROC 0.6623, F1 0.6463, 274 params); canonical classical: q34a_trial_004 (AUROC 0.6835, F1 0.6398, 2250 params); unified leaderboard: experiments/leaderboards/q35_unified_frontier.csv; reference: reports/q35_unified_pareto_frontier_analysis.md
Q36 status: NEXT — AWS/Ray distributed scaling design (design doc only; no cloud resources before Q36 approved)
Phase 2 (Experiment Automation): COMPLETE
Phase 3 (Classical NAS Ceiling): IN PROGRESS — Q32 design complete; Q34A pilot PASS (4-epoch, 5-trial; not definitive ceiling)
Phase 4 (Quantum NAS): IN PROGRESS — Q33A + Q33B complete; Q33C realized; Q34B COMPLETE; Q34C COMPLETE
Phase 5 (Local NAS Pilot): COMPLETE — Q34A COMPLETE; Q34B COMPLETE; Q34B-HF COMPLETE; Q34B-Parallel-Lite COMPLETE; EXP-005 COMPLETE; Q34B-full-lite COMPLETE; Q34C-Preflight COMPLETE; Q34C-Smoke COMPLETE; Q34C COMPLETE
Phase 5b (Unified Pareto Analysis): COMPLETE — Q35 COMPLETE; 6-trial cross-frontier Pareto set; Q36 unblocked
CV quantum pilot ceiling: q34c_trial_005 (AUROC 0.6623, F1 0.6463, 274 params) — pilot exploratory; 2-epoch budget; not definitive ceiling
DV quantum pilot ceiling: q34b_trial_004 (AUROC 0.6551, F1 0.6289, 598 params) — DOMINATED by CV cross-frontier; DV excluded from Q36 NAS planning
Classical pilot ceiling: q34a_trial_005 (AUROC 0.6867, F1 0.6287, 4754 params) — exploratory; 5-trial pilot; not definitive ceiling
Q34A best compact candidate: q34a_trial_004 (AUROC 0.6835, F1 0.6398, 2250 params)
Q34B best AUROC candidate: q34b_trial_004 (AUROC 0.6551, F1 0.6289, 598 params) — dominated cross-frontier
Q34B best F1 candidate: q34b_trial_001 (AUROC 0.6415, F1 0.6356, 280 params) — dominated cross-frontier
Q35 canonical CV: q34c_trial_005 (AUROC 0.6623, F1 0.6463, 274 params, 2.00 ms)
Q35 canonical classical compact: q34a_trial_004 (AUROC 0.6835, F1 0.6398, 2250 params, 1.60 ms)
Binary benchmarking phase: CLOSED
Multiclass: BLOCKED (requires Phase 3 + 4 + 5b)
AWS/Ray: NEXT — Q35 validated; Q36 design document required before cloud provisioning
Object detection: BLOCKED (out of current roadmap scope)
```
