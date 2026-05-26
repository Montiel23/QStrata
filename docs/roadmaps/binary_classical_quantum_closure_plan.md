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
| Q25 | Continuous-Variable Binary Feasibility Design | PLANNED |
| Q26 | Continuous-Variable Binary Smoke Test | PLANNED |
| Q27 | Continuous-Variable Binary Full Training | PLANNED |
| Q28 | DV vs CV Binary Comparative Report | PLANNED |
| Q29 | Binary Quantum Release Tagging | PLANNED |
| — | **VinDr CV binary benchmarking** | **PENDING** |
| — | **Overall VinDr binary quantum benchmarking** | **IN PROGRESS** |
| P21 | PneumoniaMNIST Classical vs DV Hybrid Comparative Report | TODO |
| R-FINAL | Global Binary Benchmark Technical Summary | TODO |
| — | **NEXT PHASE — Multiclass benchmarking** | **BLOCKED until Q29 complete** |

**Slice descriptions — CV binary phase:**

- **Q24** — Roadmap Realignment for CV Binary Quantum Phase. Corrects roadmap to reflect DV binary closure only; inserts CV binary phase Q25–Q29 before multiclass. Documentation only.
- **Q25** — Continuous-Variable Binary Feasibility Design. Design CV binary experiment architecture for VinDr-SpineXR. Covers Gaussian ansatz, quadrature outputs, moment-based readout, symplectic formalism, and QStrata-only integration. No training.
- **Q26** — Continuous-Variable Binary Smoke Test. Validate minimal CV pipeline with forward pass, gradient flow, numerical stability, probability sanity, and optimizer update verification.
- **Q27** — Continuous-Variable Binary Full Training. Train CV binary hybrid benchmark on VinDr-SpineXR using validated CV pipeline.
- **Q28** — DV vs CV Binary Comparative Report. Scientific comparison of DV hybrid, CV hybrid, and classical controls across VinDr binary benchmarks.
- **Q29** — Binary Quantum Release Tagging. Create binary benchmark release tags after DV and CV binary phases are both complete.

---

## 3b. Tagging Strategy

| Tag | Trigger condition |
|---|---|
| `vindr-binary-dv-v1` | After Q23 — VinDr DV binary phase closure |
| `vindr-binary-cv-v1` | After Q28 — VinDr CV binary phase closure |
| `vindr-binary-complete-v1` | After Q29 — full VinDr binary quantum closure |

Tags must not be created until all required slices in each phase are complete.

---

## 3c. Multiclass Phase Gate

**Status: BLOCKED — must not start until all of the following are complete:**

- VinDr DV binary benchmarking (Q17–Q23): **CLOSED** ✓
- VinDr CV binary benchmarking (Q25–Q28): PENDING
- DV vs CV binary comparative report (Q28): PENDING
- Binary quantum release tagging (Q29): PENDING

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
- [ ] **Q25–Q28** (VinDr CV binary benchmarking) is completed — PENDING
- [ ] **Q29** (Binary Quantum Release Tagging) is completed — PENDING
- [x] VinDr-SpineXR classical full baseline (Q17) is completed and validated
- [x] VinDr-SpineXR DV hybrid full baseline with random backbone (Q19) is completed
- [x] VinDr-SpineXR DV hybrid full baseline with pretrained backbone (Q21) is completed
- [ ] **R-FINAL** (Global Binary Benchmark Technical Summary) is completed
- [ ] No multiclass work has been started prematurely
- [ ] No continuous-variable quantum work has been started prematurely

A dataset's comparative report is only valid if both the classical and DV hybrid baselines for that dataset used the same: train/val/test split, seed, preprocessing policy, and evaluation protocol.

---

## 7. Deferred Work

### Multiclass (BLOCKED until Q29)

| Item | Dataset |
|---|---|
| PathMNIST multiclass | PathMNIST |
| VinDr-SpineXR multiclass | VinDr-SpineXR |

### Continuous-Variable Quantum — Active (Q25–Q29)

VinDr-SpineXR binary CV is now the active next phase. It is no longer deferred.

| Item | Dataset | Status |
|---|---|---|
| VinDr-SpineXR binary CV (Q25–Q28) | VinDr-SpineXR | **ACTIVE — next phase** |
| PneumoniaMNIST binary CV | PneumoniaMNIST | Deferred until VinDr CV binary complete |
| PathMNIST multiclass CV | PathMNIST | Deferred until Q29 |
| VinDr-SpineXR multiclass CV | VinDr-SpineXR | Deferred until Q29 |

No resources or design work should be allocated to multiclass items until Q29 is complete.

---

## 8. Immediate Next Action

VinDr **DV** binary phase is **CLOSED** (Q23 complete).
VinDr **CV** binary phase is **PENDING** (Q25 next).

```
Run:
Slice Q25 — Continuous-Variable Binary Feasibility Design

Goal:
Design the CV binary experiment architecture for VinDr-SpineXR using:
  - Gaussian ansatz (Gaussian boson sampling or Gaussian circuit formalism)
  - Quadrature outputs (position/momentum quadrature measurements)
  - Moment-based readout (first and second moments of output distribution)
  - Symplectic formalism for gate operations
  - QStrata-only integration (no external quantum libraries)

No training. No implementation. Design and documentation only.

References:
  VinDr DV binary report: reports/vindr_binary_comparative_report.md
  Q21 (DV benchmark):      reports/vindr_dv_hybrid_pretrained_full_training.md
  Q22 (classical control): reports/vindr_classical_control_tiny_head.md
  Q24 (this realignment):  reports/q24_roadmap_realignment_cv_binary.md
```
