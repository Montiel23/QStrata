# QStrata Master Research Roadmap

**Project:** QStrata — medical imaging model optimization R&D  
**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)  
**Status:** Q33B COMPLETE — Q33C NEXT (design only)

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

### Phase 5 — Local Multi-Objective NAS Pilot (PLANNED)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q34A | Classical NAS Pilot (first execution) | PLANNED | Q33C ✓ |
| Q34B | DV NAS Pilot (second execution) | PLANNED | Q34A complete |
| Q34C | CV NAS Pilot (third execution) | PLANNED | Q34B complete |

**Phase 5 gate:** Q33C NAS execution protocol design must be complete  
**Phase 5 note:** Q34 executes incrementally — Q34A (classical) first, Q34B (DV) second, Q34C (CV) third. Do not attempt all three simultaneously on the first execution day. Each pilot produces a Pareto frontier; Q35 performs unified three-frontier comparison.  
**Phase 5 note (Q34C):** CV NAS pilot records stability taxonomy for every trial. Stability-aware Pareto filtering excludes trials with invalid Gaussian states from the CV frontier regardless of AUROC/F1 values. No AWS or Ray. All three pilots execute on local single GPU.

### Phase 5b — Unified Pareto Analysis (PLANNED)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q35 | Unified Pareto Analysis and NAS Hardening | PLANNED | Q34A + Q34B + Q34C complete |

**Phase 5b gate:** Q34A, Q34B, and Q34C must all be complete  
**Phase 5b note:** Full three-frontier comparison: classical (Q34A) vs DV (Q34B) vs CV (Q34C) Pareto frontiers. Stability taxonomy analysis for CV trials. Identifies which Q33A/Q33B dimensions drive Pareto-optimal quantum performance. Produces NAS hardening recommendations.

### Phase 6 — Distributed Scaling (BLOCKED)

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q36 | AWS / Ray Distributed Scaling Design | BLOCKED | Q35 (Pareto analysis validated) |

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
| Q34A unblocked | PLANNED | Q33C complete |
| Q34B unblocked | PLANNED | Q34A complete |
| Q34C unblocked | PLANNED | Q34B complete |
| Q35 unblocked | PLANNED | Q34A + Q34B + Q34C complete |
| Q36 unblocked | BLOCKED | Q35 Pareto analysis validated |

### Multiclass Gate

| Gate | Status | Condition |
|---|---|---|
| Phase 3 (Q32) complete | ✓ (design) | Q31A ✓ |
| Phase 4 (Q33A) complete | ✓ (design) | Q32 ✓ |
| Phase 4 (Q33B) complete | ✓ (design) | Q33A ✓ |
| Phase 4 (Q33C) complete | PLANNED | Q33B ✓ |
| Phase 5 (Q34A–Q34C) complete | PLANNED | Q33C complete |
| Phase 5b (Q35) complete | PLANNED | Q34A–Q34C complete |
| Multiclass may begin | **BLOCKED** | All of the above complete |

### AWS/Ray Gate (Q36)

| Gate | Status | Condition |
|---|---|---|
| Q34A–Q34C local NAS validated | PLANNED | — |
| Q35 Pareto analysis complete | PLANNED | Q34A–Q34C complete |
| Q36 design approved | BLOCKED | Q35 complete |
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

**Q33C — NAS Execution Protocol Design**

Purpose: finalize the Q34A/Q34B/Q34C incremental execution plan. Define trial sampling strategy (random search vs. lightweight Bayesian), per-trial timeout values for each search space, leaderboard format, and CV stability monitoring protocol. Q33C is design only — no NAS execution. Q34A is the first local NAS execution phase and must not begin before Q33C is complete.

Reference documents:
- `docs/architecture/q33b_cv_quantum_nas_search_space.md`
- `reports/q33b_cv_quantum_nas_search_space_design.md`
- `docs/architecture/q33a_dv_quantum_nas_search_space.md`
- `docs/architecture/q32_classical_nas_search_space.md`
- `docs/specs/qstrata_experiment_config_schema.md`
- `reports/q31a_runner_reproducibility_test_and_hardening.md`

Gate: Q33B complete ✓ (CV quantum NAS search space design — design only; no NAS execution)  
Execution ordering: Q34A (classical) → Q34B (DV) → Q34C (CV). Do not attempt all three simultaneously.

---

```
Q30 status: COMPLETE
Q31 status: COMPLETE — smoke PASS (experiment_id 20260526_222939_a508a2)
Q31A status: COMPLETE — reproducibility PASS (loss_delta=0.0, tolerance 0.0001)
Q32 status: COMPLETE — design only; no NAS execution
Q33A status: COMPLETE — design only; no NAS execution
Q33B status: COMPLETE — design only; no NAS execution
Q33C status: NEXT — NAS Execution Protocol Design (design only)
Q34A status: PLANNED — classical NAS pilot (first execution)
Q34B status: PLANNED — DV NAS pilot (second execution)
Q34C status: PLANNED — CV NAS pilot (third execution)
Q35 status: PLANNED — unified Pareto analysis (after Q34A–Q34C)
Q36 status: BLOCKED — requires Q35 validated
Phase 2 (Experiment Automation): COMPLETE
Phase 3 (Classical NAS Ceiling): IN PROGRESS — Q32 design complete; execution in Q34A
Phase 4 (Quantum NAS): IN PROGRESS — Q33A + Q33B complete; Q33C NEXT (design only)
CV quantum ceiling: UNDEFINED — will be produced by Q34C
DV quantum ceiling: UNDEFINED — will be produced by Q34B
Classical ceiling: UNDEFINED — will be produced by Q34A
Binary benchmarking phase: CLOSED
Multiclass: BLOCKED (requires Phase 3 + 4 + 5b)
AWS/Ray: BLOCKED (requires Q35 validated)
Object detection: BLOCKED (out of current roadmap scope)
```
