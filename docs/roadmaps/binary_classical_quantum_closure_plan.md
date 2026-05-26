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
| Classical baseline smoke test | NEXT |
| Classical full baseline | TODO |
| DV hybrid smoke test | TODO |
| DV hybrid full baseline | TODO |
| Comparative report | TODO |

---

## 3. Remaining Slices

| Slice | Goal |
|---|---|
| Q16 | VinDr-SpineXR Classical Baseline Smoke Test |
| Q17 | VinDr-SpineXR Classical Baseline Full Training |
| Q18 | VinDr-SpineXR DV Hybrid Smoke Test |
| Q19 | VinDr-SpineXR DV Hybrid Full Training |
| Q20 | VinDr-SpineXR Classical vs DV Hybrid Comparative Report |
| P21 | PneumoniaMNIST Classical vs DV Hybrid Comparative Report |
| R-FINAL | Global Binary Benchmark Technical Summary |

---

## 4. Explicit Ordering Rules

The following ordering constraints are hard requirements, not preferences:

1. **Do not start multiclass work until binary closure is complete for both datasets.**  
   Binary closure requires Q20, P21, and R-FINAL all completed.

2. **Do not start continuous-variable quantum experiments until DV binary closure is complete.**  
   DV binary closure requires Q19 (VinDr-SpineXR DV hybrid full training) and the equivalent PneumoniaMNIST DV work to be finished and reported.

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

## 6. Definition of Done

Binary closure is complete when **ALL** of the following conditions are true:

- [ ] **P21** (PneumoniaMNIST Classical vs DV Hybrid Comparative Report) is completed
- [ ] **Q20** (VinDr-SpineXR Classical vs DV Hybrid Comparative Report) is completed
- [ ] VinDr-SpineXR classical full baseline (Q17) is completed and validated
- [ ] VinDr-SpineXR DV hybrid full baseline (Q19) is completed and validated
- [ ] **R-FINAL** (Global Binary Benchmark Technical Summary) is completed
- [ ] No multiclass work has been started prematurely
- [ ] No continuous-variable quantum work has been started prematurely

A dataset's comparative report is only valid if both the classical and DV hybrid baselines for that dataset used the same: train/val/test split, seed, preprocessing policy, and evaluation protocol.

---

## 7. Deferred Work

The following work is explicitly deferred until after binary closure (R-FINAL complete):

### Multiclass

| Item | Dataset |
|---|---|
| PathMNIST multiclass | PathMNIST |
| VinDr-SpineXR multiclass | VinDr-SpineXR |

### Continuous-Variable Quantum

| Item | Dataset |
|---|---|
| PneumoniaMNIST binary CV | PneumoniaMNIST |
| VinDr-SpineXR binary CV | VinDr-SpineXR |
| PathMNIST multiclass CV | PathMNIST |
| VinDr-SpineXR multiclass CV | VinDr-SpineXR |

No resources or design work should be allocated to deferred items until the Definition of Done (Section 6) is satisfied.

---

## 8. Immediate Next Action

```
Run:
Slice Q16 — VinDr-SpineXR Classical Baseline Smoke Test

Goal:
Validate end-to-end classical CNN training mechanics on the VinDr-SpineXR
binary dataset before full baseline training.
```

Entry point: `scripts/smoke_test_vindr_classical_baseline.py`  
Loader: `qcore/data/vindr_spinexr.py`  
Dataset root: `data/processed/vindr_binary_roi_224/`  
Hard caps: 1 epoch, 5 train batches, 3 val batches — mechanics only, no checkpoint.
