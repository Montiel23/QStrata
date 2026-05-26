# Binary Classical vs Quantum Closure Plan

**Project:** QStrata — medical imaging model optimization R&D  
**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-25  
**Author:** Miguel Lopez (QStrata)

---

## 1. Objective

Close all binary classification experiments for both datasets before proceeding to multiclass or continuous-variable quantum work. Each dataset requires a complete classical CNN baseline and a complete DV hybrid CNN-QNN baseline with full comparative reporting.

**Scope:**

- **Dataset 1:** PneumoniaMNIST (28×28 grayscale, binary: Pneumonia vs Normal)
- **Dataset 2:** VinDr-SpineXR (224×224 grayscale, binary: Any Pathology vs No Finding)

**Models to compare per dataset:**

| Model | Description |
|---|---|
| Classical CNN baseline | Standard convolutional architecture; no quantum components |
| DV hybrid CNN-QNN | Classical CNN feature extractor + discrete-variable quantum circuit readout |

Binary closure is complete only when both datasets have finished both model types and a comparative report is produced for each dataset, followed by a global benchmark summary.

---

## 2. Current Status

### PneumoniaMNIST Binary

| Work item | Status |
|---|---|
| Classical baseline | DONE |
| DV hybrid baseline | DONE |
| Gradient fix (`torch.atan`) | DONE |
| Pretrained backbone integration | DONE |
| Multi-seed stability validation | DONE |
| Comparative final report | TODO |

### VinDr-SpineXR Binary

| Work item | Status |
|---|---|
| EDA | DONE |
| Binary task decision | DONE |
| ROI dataset design | DONE |
| Dataset exporter | DONE |
| Full dataset export | DONE |
| PyTorch Dataset loader | DONE |
| Classical baseline smoke test | DONE |
| Classical full baseline | DONE |
| DV hybrid smoke test | DONE |
| DV hybrid full baseline (random backbone) | DONE |
| Pretrained-backbone feasibility | DONE |
| DV hybrid full baseline (pretrained backbone) | DONE |
| Comparative report | DONE |

---

## 3. Remaining Slices

| Slice | Goal | Status |
|---|---|---|
| Q16 | VinDr-SpineXR Classical Baseline Smoke Test | DONE |
| Q17 | VinDr-SpineXR Classical Baseline Full Training | DONE |
| Q18 | VinDr-SpineXR DV Hybrid Smoke Test | DONE |
| Q19 | VinDr-SpineXR DV Hybrid Full Training (random backbone) | DONE |
| Q20 | VinDr-SpineXR DV Hybrid Pretrained-Backbone Feasibility | DONE |
| Q21 | VinDr-SpineXR DV Hybrid Pretrained-Backbone Full Training | DONE |
| Q22 | VinDr-SpineXR Approximate Trainable-Parameter-Matched Classical Control | COMPLETE |
| Q23 | VinDr DV Binary Comparative Report | COMPLETE |
| — | **VinDr DV binary benchmarking** | **CLOSED** |
| Q24 | Roadmap Realignment for CV Binary Quantum Phase | COMPLETE |
| Q25 | Continuous-Variable Binary Feasibility Design | COMPLETE |
| Q25A | Roadmap Prioritization and Experiment Automation Planning | COMPLETE |
| Q26 | Continuous-Variable Binary Smoke Test | COMPLETE |
| Q27 | Continuous-Variable Binary Full Training | COMPLETE |
| Q27A | NAS Strategy and Optimization Phase Refinement | COMPLETE |
| Q28 | DV vs CV Binary Comparative Report | COMPLETE |
| Q29 | Binary Quantum Release Tagging | COMPLETE |
| — | **VinDr CV binary benchmarking** | **CLOSED** |
| — | **Overall VinDr binary quantum benchmarking** | **CLOSED** |
| P21 | PneumoniaMNIST Classical vs DV Hybrid Comparative Report | TODO |
| R-FINAL | Global Binary Benchmark Technical Summary | TODO |
| — | **NEXT PHASE — Optimization / NAS** | **UNBLOCKED — Q30 is NEXT** |
| — | **NEXT PHASE — Multiclass benchmarking** | **UNBLOCKED — scheduling pending** |

**Slice descriptions — CV binary phase:**

- **Q24** — Roadmap Realignment for CV Binary Quantum Phase. Corrects roadmap to reflect DV binary closure only; inserts CV binary phase Q25–Q29 before multiclass. Documentation only.
- **Q25** — Continuous-Variable Binary Feasibility Design. Design CV binary experiment architecture for VinDr-SpineXR. Covers Gaussian ansatz, quadrature outputs, moment-based readout, symplectic formalism, and QStrata-only integration. No training.
- **Q25A** — Roadmap Prioritization and Experiment Automation Planning. Separated immediate scientific execution priorities from future automation priorities. Gated NAS, AWS, and Ray work until CV baseline is validated. Corrected Q26 assumptions from final Q25 design decisions.
- **Q26** — Continuous-Variable Binary Smoke Test. Validate minimal CV pipeline with forward pass, gradient flow, numerical stability, probability sanity, and optimizer update verification.
- **Q27** — Continuous-Variable Binary Full Training. Train CV binary hybrid benchmark on VinDr-SpineXR using validated CV pipeline.
- **Q27A** — NAS Strategy and Optimization Phase Refinement. Documentation-only slice. Replaced generic Q30–Q35 automation placeholders with a structured optimization roadmap separating classical NAS (Q32) from quantum NAS (Q33). Added classical ceiling principle and multi-objective optimization philosophy. No training, no scripts.
- **Q28** — DV vs CV Binary Comparative Report. Scientific comparison of DV hybrid, CV hybrid, and classical controls across VinDr binary benchmarks.
- **Q29** — Binary Quantum Release Tagging. Create binary benchmark release tags after DV and CV binary phases are both complete.

---

## 3b. Immediate Scientific Priorities

The following four slices were the exclusive execution focus. Q26, Q27, Q27A, and Q28 are now complete.
No automation, NAS, or distributed infrastructure work begins before Q29 is complete.

| Slice | Description | Status |
|---|---|---|
| Q26 | CV Binary Smoke Test | COMPLETE — PASS (2026-05-26) |
| Q27 | CV Binary Full Training | COMPLETE — PASS (2026-05-26) |
| Q28 | DV vs CV Binary Comparative Report | COMPLETE (2026-05-26) |
| Q29 | Binary Quantum Release Tagging | COMPLETE (2026-05-26) |

**Q26 — Confirmed results (from `reports/q26_cv_binary_smoke_test.md`):**

| Parameter | Value | Confirmed |
|---|---|---|
| Compression layer | `nn.Linear(128 → 4)` | ✓ |
| CV encoding | `compressed * sqrt(2*hbar)` (gradient-safe) | ✓ |
| Ansatz | `GaussianVariationalAnsatz(n_modes=2, depth=1, squeezing_cap=1.5)` | ✓ |
| Readout | Deterministic first-moment readout (`mu_final`) | ✓ |
| Readout layer | `nn.Linear(4 → 2)` | ✓ |
| Actual trainable params | 536 (exact match to Q25A spec) | ✓ |
| CV backend | `GaussianBackend(n_modes=2, hbar=2.0, device='cpu')` | ✓ |
| Health checks | 14 / 14 PASS | ✓ |
| Gradient flow | compression, ansatz, readout all received non-zero gradient | ✓ |
| Backbone frozen | zero gradient confirmed | ✓ |

**Q27 — Confirmed results (from `reports/q27_cv_binary_full_training.md`):**

| Metric | Value |
|---|---|
| Epochs run | 15 of 15 (max epochs reached) |
| Best epoch | 15 |
| Best val AUROC | 0.6946 |
| Best val F1 | 0.6382 |
| Best val loss | 0.6440 |
| Test AUROC | 0.6708 |
| Test F1 | 0.6283 |
| Test Accuracy | 65.77% |
| Test Confusion | [[765, 305], [406, 601]] (non-degenerate) |
| CV health (all 15 epochs) | COV_PSD PASS, COV_SYMMETRIC PASS, QUAD_FINITE PASS, NO_NAN_INF PASS |
| Latency | 2.15 ms/sample (100 single-sample passes, CUDA backbone + CPU CV) |
| Trainable params | 536 |
| Verdict | CV_BINARY_FULL_TRAINING: PASS |

---

## 3c. Future Experiment Automation Phase

All slices in this phase are **PLANNED**.
Q29 is now complete — this phase is **UNBLOCKED**. Q30 is the immediate next slice.

The gating condition has been met: Q29 (Binary Quantum Release Tagging) is COMPLETE (2026-05-26).

| Slice | Description | Status | Blocked Until |
|---|---|---|---|
| Q30 | Experiment Automation Framework Design | NEXT | Q29 complete ✓ |
| Q31 | Local GPU Experiment Runner | PLANNED | Q30 complete |
| Q32 | NAS Search Space Design — Classical Feature Extractors | PLANNED | Q31 complete |
| Q33 | NAS Search Space Design — Quantum Heads (DV/CV) | PLANNED | Q32 complete |
| Q34 | Local Multi-Objective NAS Pilot | PLANNED | Q33 complete |
| Q35 | AWS / Ray Distributed Scaling Design | PLANNED | Q34 complete |

**Slice descriptions — Optimization Phase:**

- **Q30** — Experiment Automation Framework Design. Design reproducible local experiment orchestration covering metric tracking, checkpoint management, artifact naming, and sweep configuration. Output: automation design document. No training.
- **Q31** — Local GPU Experiment Runner. Implement YAML-driven runner supporting: run queuing, resume of failed runs, metric collection, and result ranking by configurable criteria. Output: working local runner with tested examples.
- **Q32** — NAS Search Space Design — Classical Feature Extractors. Define and search a compact classical CNN architecture space (convolution block types, channel widths, depth, pooling variants). Goal: establish the strongest compact classical performance ceiling before quantum architecture search. Classical NAS always precedes quantum NAS.
- **Q33** — NAS Search Space Design — Quantum Heads (DV/CV). Define search spaces for DV (qubits, depth, ansatz type) and CV (modes, depth, squeezing, encoding) quantum heads. Requires classical ceiling from Q32 to bound evaluation criteria and search stopping conditions.
- **Q34** — Local Multi-Objective NAS Pilot. Execute joint optimization across AUROC, F1, parameter count, inference latency, and training stability on a single GPU. Multi-objective, not single-metric ranking. No AWS or Ray.
- **Q35** — AWS / Ray Distributed Scaling Design. Design distributed NAS only after local NAS (Q34) is validated. Covers AWS GPU provisioning, Ray orchestration, artifact management, cost controls. Output: distributed scaling design document. No code until design is approved.

---

## 3d. NAS / AWS / Ray Gating Rules

NAS, AWS, and Ray work was **BLOCKED** until ALL of the following were complete:

- Q26: CV binary smoke test **PASSES** — ✓ COMPLETE (2026-05-26)
- Q27: CV binary full training **COMPLETE** — ✓ COMPLETE (2026-05-26)
- Q28: DV vs CV binary comparative analysis **COMPLETE** — ✓ COMPLETE (2026-05-26)
- Q29: Binary quantum release tagging **COMPLETE** — ✓ COMPLETE (2026-05-26)

**Gate cleared.** All four conditions are now satisfied. NAS, AWS, and Ray work is UNBLOCKED. Q30 is the immediate next slice.

**Reason:** Do not automate search before the CV baseline is scientifically validated and
binary benchmarking is formally closed. Premature scaling increases uncertainty and wastes
resources. NAS requires a validated baseline to define meaningful search bounds, evaluation
criteria, and stopping conditions.

**Additional gate — Q35:** AWS / Ray distributed scaling is additionally blocked until Q34
(local multi-objective NAS pilot) is complete and validated. Distributed infrastructure must
not be provisioned before local NAS confirms the search procedure is sound.

---

## 3e. Optimization Philosophy

The following principles govern all optimization and NAS work in the Q30–Q35 phase.
These are research-level constraints, not implementation preferences.

### Classical Ceiling Principle

> **Classical NAS (Q32) must always precede quantum NAS (Q33).**

Before exploring quantum architecture search, the strongest compact classical architecture
achievable under the same parameter and latency budget must be established. This ceiling
defines the performance reference against which quantum advantage claims are evaluated.
A quantum result that does not beat a well-optimized classical baseline under equivalent
resource constraints does not support quantum advantage conclusions.

Quantum NAS (Q33) begins only after Q32 has produced a validated compact classical ceiling.

### Multi-Objective Optimization Principle

Single-metric optimization (e.g., maximize AUROC only) is insufficient for medical imaging
research. Q34 and later slices optimize jointly across:

| Objective | Rationale |
|---|---|
| Test AUROC | Primary discriminative performance metric |
| Test F1 | Class-balance-aware performance; critical for imbalanced datasets |
| Trainable parameter count | Model compactness; resource efficiency |
| Inference latency (ms/sample) | Clinical deployment feasibility |
| Training stability | Variance across seeds; reproducibility |
| Generalization gap | val loss − train loss at convergence; overfitting signal |

No single trade-off point is pre-selected. The Pareto frontier is explored. Final model
selection is a scientific decision, not an automated argmax.

### NAS Philosophy

The NAS procedure in Q32–Q34 is:

- **Constrained**: search spaces are bounded by hardware budget, not unbounded
- **Reproducible**: all trials are seeded and logged; result tables are exact, not approximate
- **Interpretable**: each search dimension has a stated scientific motivation
- **Budget-aware**: wall-clock and GPU cost are tracked as first-class trial metadata
- **Non-exhaustive by design**: search is guided, not grid-over-everything

### Infrastructure Sequencing Principle

Automation and infrastructure complexity must grow in proportion to validated scientific need:

1. **Validated manual baseline first** — Q17 through Q27
2. **Design automation** before building it — Q30 produces design documents, not code
3. **Local single-GPU execution** before distributed — Q31/Q32/Q33/Q34 are local-only
4. **Distributed infrastructure only after local is validated** — Q35 requires Q34 pass
5. **AWS and Ray provisioned only after design approval** — no cloud cost before design sign-off
6. **Never automate a scientifically unvalidated procedure** — the Q29 gate enforces this

---

## 3f. Tagging Strategy

| Tag | Trigger condition |
|---|---|
| `vindr-binary-dv-v1` | After Q23 — VinDr DV binary phase closure |
| `vindr-binary-cv-v1` | After Q28 — VinDr CV binary phase closure |
| `vindr-binary-complete-v1` | After Q29 — full VinDr binary quantum closure |

Tags must not be created until all required slices in each phase are complete.

---

## 3g. Multiclass Phase Gate

**Status: UNBLOCKED — all binary phase prerequisites are now complete.**

- VinDr DV binary benchmarking (Q17–Q23): **CLOSED** ✓
- VinDr CV binary benchmarking (Q25–Q28): **CLOSED** ✓ (2026-05-26)
- DV vs CV binary comparative report (Q28): **COMPLETE** ✓ (2026-05-26)
- Binary quantum release tagging (Q29): **COMPLETE** ✓ (2026-05-26)

Multiclass benchmarking may now be scheduled. The immediate execution priority remains Q30 (Experiment Automation Framework Design). Multiclass slices should be planned in parallel but not executed before Q30 design is complete.

---

## 4. Explicit Ordering Rules

The following ordering constraints are hard requirements, not preferences:

1. **Do not start multiclass work until binary closure is complete for both datasets.**  
   Binary closure requires Q20, P21, and R-FINAL all completed.

2. **CV binary experiments begin after DV binary closure — not after full binary closure.**  
   VinDr DV binary closure (Q17–Q23) is now COMPLETE. CV binary benchmarking (Q25–Q28) is the next active phase. Do NOT start CV experiments before completing the DV phase (now satisfied).

3. **Do not start VinDr-SpineXR DV hybrid full training (Q19) until the classical VinDr baseline (Q17) is validated and stable.**  
   The classical baseline establishes the performance reference that the DV hybrid result is measured against. A flawed or incomplete classical baseline invalidates the comparative report.

---

## 5. Metrics Required for Full Baselines

All full baseline runs (Q17, Q19, and the PneumoniaMNIST comparative re-run if needed) must collect and report the following metrics.

### Machine Learning Metrics (required for all full baselines)

| Metric | Notes |
|---|---|
| Accuracy | Per-epoch (train + val); final test value (analysis only) |
| Precision | Reported on val and test |
| Recall | Reported on val and test |
| F1-score | Reported on val and test |
| AUROC | Area under the ROC curve; val and test |
| AUPRC | Area under the precision-recall curve; val and test |
| Confusion matrix | Val and test |
| Train loss | Per epoch |
| Val loss | Per epoch |
| Test loss | Analysis only — never used as a fitness signal or gate criterion |
| Epoch time | Wall-clock seconds |
| Inference time | Per-batch and per-sample |
| Parameter count | Trainable parameters |

### Quantum Metrics (required where applicable — DV hybrid baselines)

| Metric | Notes |
|---|---|
| Theta gradient norm | Per epoch — tracks quantum parameter update health |
| Projection gradient norm | Per epoch |
| Readout gradient norm | Per epoch |
| Probability sum check | Per epoch — confirms circuit outputs are valid probability distributions |
| State fidelity | If available in the DV measurement backend |
| Gate fidelity | If available |
| Entropy | If available |
| Purity | If available |
| State evolution by epoch | If available — tracks quantum state change across training |

---

## 5b. Q19 Backbone Guardrail

> **Added after Q20 (Slice Q20 — 2026-05-26)**

```
Q19 Backbone Guardrail:
The Q19 DV hybrid result used a frozen random CNN backbone and must not be treated
as the final DV benchmark for VinDr-SpineXR. A pretrained or architecturally
compatible classical backbone must be validated before the VinDr comparative report
(Q22) is produced.
```

**Background:** Q19 full training produced degenerate results (all-class-0, F1=0, confusion
[[1070,0],[1007,0]]) because the quantum head was trained on random convolutional features.
Q20 confirmed that `checkpoints/c006_d040_classical_anchor.pt` (Slice Q6, PneumoniaMNIST-trained
depthwise_sep [64,128] backbone) is architecturally compatible and loads correctly into
`DVHybridCNNQNN`. Q21 will run full training with this pretrained backbone.

---

## 5c. Q20 Interpretation Guardrail

> **Added after Q18 (Slice Q18 — 2026-05-26)**

```
Q20 Interpretation Guardrail:
If the DV hybrid model outperforms the current VinDr-SpineXR classical baseline
(Q17: AUROC 0.6224, F1 0.5355), do NOT claim quantum advantage. The Q17 classical
baseline is potentially weak due to missing inter-block spatial downsampling.
A classical ablation with MaxPool/inter-block downsampling must be run and compared
before any architecture-level conclusions are drawn from the Q20 comparative report.
```

**Background:** The Q17 classical baseline (CNN3Block, 23,650 params) exhibited training
instability — validation loss spiked while training loss decreased, suggesting the
architecture without inter-block MaxPool is not the strongest classical reference.
Any Q20 comparison must account for this architectural limitation before attributing
performance differences to quantum vs classical effects.

---

## 6. Definition of Done

Binary closure is complete when **ALL** of the following conditions are true:

- [ ] **P21** (PneumoniaMNIST Classical vs DV Hybrid Comparative Report) is completed
- [x] **Q20** (VinDr-SpineXR DV Hybrid Pretrained-Backbone Feasibility) is completed
- [x] **Q21** (VinDr-SpineXR DV Hybrid Pretrained-Backbone Full Training) is completed
- [x] **Q22** (VinDr-SpineXR Approximate Trainable-Parameter-Matched Classical Control) is completed
- [x] **Q23** (VinDr DV Binary Comparative Report) is completed — VinDr **DV** binary phase CLOSED
- [x] **Q24** (Roadmap Realignment for CV Binary Quantum Phase) is completed
- [x] **Q25** (CV binary feasibility design) is completed
- [x] **Q25A** (Roadmap prioritization and automation planning) is completed
- [x] **Q26** (CV binary smoke test) is completed — PASS (2026-05-26)
- [x] **Q27** (CV binary full training) is completed — PASS (2026-05-26): Test AUROC 0.6708
- [x] **Q27A** (NAS strategy and optimization phase refinement) is completed — COMPLETE (2026-05-26)
- [x] **Q28** (DV vs CV binary comparative report) is completed — COMPLETE (2026-05-26)
- [x] **Q29** (Binary Quantum Release Tagging) is completed — COMPLETE (2026-05-26)
- [x] VinDr-SpineXR classical full baseline (Q17) is completed and validated
- [x] VinDr-SpineXR DV hybrid full baseline with random backbone (Q19) is completed
- [x] VinDr-SpineXR DV hybrid full baseline with pretrained backbone (Q21) is completed
- [ ] **R-FINAL** (Global Binary Benchmark Technical Summary) is completed
- [ ] No multiclass work has been started prematurely
- [ ] No continuous-variable quantum work has been started prematurely

A dataset's comparative report is only valid if both the classical and DV hybrid baselines for that dataset used the same: train/val/test split, seed, preprocessing policy, and evaluation protocol.

---

## 7. Deferred Work

### Multiclass (unblocked — scheduling pending)

Q29 is now complete. Multiclass benchmarking may be scheduled. Immediate execution priority is Q30.

| Item | Dataset | Status |
|---|---|---|
| PathMNIST multiclass | PathMNIST | Unblocked — scheduling pending |
| VinDr-SpineXR multiclass | VinDr-SpineXR | Unblocked — scheduling pending |

### Continuous-Variable Quantum — Closed

VinDr-SpineXR binary CV phase (Q25–Q28) is now **CLOSED**.

| Item | Dataset | Status |
|---|---|---|
| VinDr-SpineXR binary CV (Q25–Q28) | VinDr-SpineXR | **CLOSED** — Q29 complete |
| PneumoniaMNIST binary CV | PneumoniaMNIST | Deferred — scheduling pending after Q30 |
| PathMNIST multiclass CV | PathMNIST | Deferred — scheduling pending |
| VinDr-SpineXR multiclass CV | VinDr-SpineXR | Deferred — scheduling pending |

---

## 8. Immediate Next Action

VinDr **DV** binary phase is **CLOSED** (Q23 complete).
VinDr **CV** binary phase is **CLOSED** — Q26 PASS, Q27 PASS, Q28 COMPLETE (2026-05-26).
Q27A **COMPLETE** (2026-05-26). Q28 **COMPLETE** (2026-05-26). Q29 **COMPLETE** (2026-05-26).
**Overall VinDr binary quantum benchmarking: CLOSED.**
Q30 (Experiment Automation Framework Design) is the immediate next slice.

**Q28 — Confirmed results (from `reports/q28_dv_vs_cv_binary_comparative_report.md`):**

| Model | AUROC | F1 | Accuracy | Params |
|---|---|---|---|---|
| Q17 Classical | 0.6224 | 0.5355 | 60.66% | 23,650 |
| Q21 DV Hybrid | 0.6800 | 0.6159 | 63.84% | 574 |
| Q22 Tiny Classical Control | 0.6625 | 0.5961 | 64.37% | 526 |
| Q27 CV Hybrid | 0.6708 | 0.6283 | 65.77% | 536 |

**Key finding:** Compact bottleneck with frozen pretrained backbone is the dominant contributor to improvement over Q17. DV and CV hybrids both exceed the parameter-matched classical control (Q22) by small margins (+0.0175 and +0.0083 AUROC respectively). No quantum advantage is claimed.

**Q29 — COMPLETE (2026-05-26)**

Three annotated release tags created on commit dc7fff6 (Q28):
- `qstrata-vindr-dv-binary-v1` — VinDr DV binary benchmarking package (Q17–Q23)
- `qstrata-vindr-cv-binary-v1` — VinDr CV binary benchmarking package (Q25–Q28)
- `qstrata-vindr-binary-comparative-v1` — Full binary comparative release (Q28)

Binary benchmarking phase: **CLOSED**.

```
Execute:
Slice Q30 — Experiment Automation Framework Design

Goal:
Design reproducible local experiment orchestration framework covering:
  - metric tracking schema
  - checkpoint management conventions
  - sweep configuration format (YAML)
  - artifact naming and logging structure

Output: design document only. No training runs. No code committed.
Gate: Q29 COMPLETE ✓

Reference (frozen benchmarks):
  Q17 Classical:        AUROC 0.6224, F1 0.5355, params 23,650
  Q21 DV Hybrid:        AUROC 0.6800, F1 0.6159, params 574
  Q22 Tiny Classical:   AUROC 0.6625, F1 0.5961, params 526
  Q27 CV Hybrid:        AUROC 0.6708, F1 0.6283, params 536
```
