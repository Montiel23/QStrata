# Q27: Continuous-Variable Binary Full Training

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Q26 passed all 14 health checks on 2026-05-26, validating that the CV pipeline is numerically stable, gradient-healthy, and correctly integrated with the QStrata Gaussian backend. Q27 trains the exact same architecture — no architectural changes — to produce the first CV binary benchmark result in the QStrata research program.

---

## 2. Training Scope

- Maximum epochs: 15
- Early stopping patience: 4 (monitor: val_loss, minimize)
- Seed: 42 (single seed only)
- No NAS, no architecture search, no hyperparameter sweep
- No quantum advantage claim

---

## 3. Model Architecture

**Frozen backbone:** C006-D040 pretrained on PneumoniaMNIST (`build_model(CNN_CONFIG)[:4]`). Outputs `(B, 128)` features. Frozen (`requires_grad=False`, `eval()` mode enforced throughout training via `model.train()` override). Device: CUDA.

**Feature extraction path:** `with torch.no_grad(): features = self.backbone(x)` — exact Q21 semantics, identical to Q26.

**Feature transfer:** `features_cpu = features.detach().cpu()` — device transfer between backbone (CUDA) and CV circuit (CPU).

**Compression layer:** `nn.Linear(128, 4)` — trainable, CPU. Projects 128-dim backbone features to 4-dim phase-space encoding vector.

**Complex displacement encoding:** `encoded_input = compressed * sqrt(2 * hbar)` where `hbar=2.0`. Gradient-safe — no in-place tensor index operations; gradient flows cleanly from loss through readout → mu_final → encoded_input → compression layer.

**GaussianVariationalAnsatz:** `GaussianVariationalAnsatz(n_modes=2, depth=1, squeezing_cap=1.5)` from `qcore/ansatz/cv_spine_ansatz.py`. Trainable parameters: `disp_real`, `disp_imag`, `squeezing_raw` (bounded via `tanh(r) * squeezing_cap`), `bs_theta`, `rot_phi` — each shape `(1, 2)`. Gate sequence per depth layer: displacement → squeezing → rotation (single-mode); rotation → beamsplitter (two-mode circular).

**GaussianBackend:** `GaussianBackend(n_modes=2, hbar=2.0, device='cpu')` from `qcore/backends/cvBackend.py`. CPU-only. Vacuum state: `mu=zeros(4)`, `cov=eye(4)*(hbar/2)`.

**Readout:** Deterministic first-moment readout — `readout_vec = mu_final` (the final phase-space mean vector, shape `(4,)`). No stochastic homodyne noise.

**Readout layer:** `nn.Linear(4, 2)` — trainable, CPU. Maps `mu_final` to binary logits `(2,)`.

---

## 4. Parameter Count

| Component | Formula | Count |
|---|---|---|
| Compression `nn.Linear(128, 4)` | 128×4 + 4 | 516 |
| Ansatz `GaussianVariationalAnsatz(n_modes=2, depth=1)` | 5 params × 2 modes | 10 |
| Readout `nn.Linear(4, 2)` | 4×2 + 2 | 10 |
| **Total trainable** | | **536** |
| Frozen backbone | C006-D040 | 9612 |

Expected trainable: 536 — Actual trainable: 536 (exact match)

---

## 5. Backbone Loading Summary

| Item | Value |
|---|---|
| Checkpoint | `checkpoints/c006_d040_classical_anchor.pt` |
| Format | `model_state_dict` |
| Matched keys | 28 |
| Skipped keys | 2 (classifier head `5.*` — expected) |
| Unexpected keys | 0 |
| Backbone frozen | YES |

---

## 6. Training Configuration

| Parameter | Value |
|---|---|
| Dataset | `vindr_binary_roi_224` |
| Checkpoint | `c006_d040_classical_anchor.pt` |
| batch_size | 4 |
| epochs | 15 (max) |
| optimizer | AdamW |
| lr | 1e-3 |
| loss | CrossEntropyLoss (unweighted) |
| seed | 42 |
| early stopping | patience 4, monitor val_loss |
| backbone device | CUDA |
| CV backend device | CPU |

---

## 7. Per-Epoch Training Table

| Ep | TrLoss | TrAcc | VlLoss | VlAcc | VlPrec | VlRec | VlF1 | VlAUROC | VlAUPRC | CompGrad | AnsGrad | ReadGrad | TotGrad | muMean | muMax | CovDiag | SqzNorm | AnsPNorm | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  1 | 0.6783 | 59.89% | 0.6648 | 63.98% | 0.6187 | 0.6982 | 0.6560 | 0.6692 | 0.6566 | 1.57e+00 | 2.82e-01 | 4.96e-01 | 1.67e+00 | 0.6384 | 1.6176 | 1.0091 | 0.0640 | 15.5122 | 82.1s |
|  2 | 0.6686 | 61.50% | 0.6608 | 64.64% | 0.6429 | 0.6327 | 0.6378 | 0.6706 | 0.6597 | 1.47e+00 | 8.01e-02 | 1.55e-01 | 1.48e+00 | 0.4988 | 1.3187 | 1.0163 | 0.0852 | 15.2967 | 82.9s |
|  3 | 0.6654 | 62.37% | 0.6648 | 61.60% | 0.6231 | 0.5552 | 0.5872 | 0.6514 | 0.6545 | 4.36e+00 | 4.12e-01 | 1.34e+00 | 4.58e+00 | 0.7816 | 2.8590 | 1.0190 | 0.0918 | 15.0868 | 81.8s |
|  4 | 0.6635 | 62.50% | 0.6570 | 63.86% | 0.6154 | 0.7079 | 0.6584 | 0.6749 | 0.6639 | 1.13e+00 | 1.93e-01 | 4.62e-01 | 1.24e+00 | 0.3699 | 1.4636 | 1.0214 | 0.0976 | 14.8620 | 82.5s |
|  5 | 0.6630 | 62.80% | 0.6566 | 64.64% | 0.6292 | 0.6848 | 0.6558 | 0.6775 | 0.6656 | 8.70e-01 | 1.40e-01 | 3.86e-01 | 9.62e-01 | 0.3408 | 0.7761 | 1.0231 | 0.1012 | 14.6429 | 79.4s |
|  6 | 0.6633 | 61.90% | 0.6561 | 64.34% | 0.6238 | 0.6933 | 0.6567 | 0.6794 | 0.6667 | 8.30e-01 | 1.13e-01 | 4.12e-01 | 9.33e-01 | 1.1943 | 3.2152 | 1.0179 | 0.0891 | 14.4028 | 85.2s |
|  7 | 0.6611 | 62.51% | 0.6551 | 63.63% | 0.6085 | 0.7309 | 0.6641 | 0.6826 | 0.6687 | 9.99e-01 | 1.53e-01 | 6.21e-01 | 1.19e+00 | 0.5551 | 1.4071 | 1.0126 | 0.0748 | 14.2097 | 85.9s |
|  8 | 0.6604 | 62.77% | 0.6520 | 64.58% | 0.6624 | 0.5709 | 0.6133 | 0.6807 | 0.6690 | 2.57e+00 | 1.07e-01 | 3.07e-01 | 2.59e+00 | 0.9259 | 2.2550 | 1.0106 | 0.0687 | 14.0301 | 88.3s |
|  9 | 0.6583 | 62.95% | 0.6516 | 63.57% | 0.6256 | 0.6461 | 0.6357 | 0.6790 | 0.6689 | 6.06e-01 | 5.58e-02 | 1.43e-01 | 6.25e-01 | 0.3962 | 1.4301 | 1.0114 | 0.0709 | 13.8462 | 87.7s |
| 10 | 0.6582 | 62.75% | 0.6565 | 63.15% | 0.6556 | 0.5285 | 0.5852 | 0.6650 | 0.6640 | 1.37e+00 | 1.71e-01 | 3.62e-01 | 1.42e+00 | 0.5118 | 1.9963 | 1.0235 | 0.1025 | 13.6831 | 86.8s |
| 11 | 0.6574 | 62.53% | 0.6503 | 63.63% | 0.6055 | 0.7479 | 0.6692 | 0.6918 | 0.6752 | 2.88e+00 | 1.99e-01 | 5.29e-01 | 2.94e+00 | 0.4107 | 1.2527 | 1.0540 | 0.1547 | 13.5249 | 86.9s |
| 12 | 0.6557 | 63.01% | 0.6576 | 63.15% | 0.6751 | 0.4836 | 0.5636 | 0.6667 | 0.6664 | 1.93e+00 | 3.87e-01 | 7.95e-01 | 2.12e+00 | 0.4844 | 1.6912 | 1.0607 | 0.1644 | 13.3368 | 87.6s |
| 13 | 0.6557 | 63.23% | 0.6503 | 63.80% | 0.6473 | 0.5806 | 0.6121 | 0.6818 | 0.6737 | 1.48e+00 | 8.48e-02 | 3.71e-01 | 1.53e+00 | 0.5248 | 1.4348 | 1.0437 | 0.1395 | 13.1664 | 87.2s |
| 14 | 0.6545 | 63.07% | 0.6512 | 64.22% | 0.6718 | 0.5333 | 0.5946 | 0.6742 | 0.6703 | 9.26e-01 | 3.15e-01 | 1.04e+00 | 1.43e+00 | 0.7946 | 3.0347 | 1.0475 | 0.1452 | 13.0086 | 88.5s |
| 15 | 0.6549 | 62.72% | 0.6440 | 67.14% | 0.6963 | 0.5891 | 0.6382 | 0.6946 | 0.6769 | 1.18e+00 | 3.06e-01 | 8.39e-01 | 1.48e+00 | 0.6383 | 1.9479 | 1.0628 | 0.1674 | 12.8502 | 86.2s |

---

## 8. Best Validation Epoch

| Metric | Value |
|---|---|
| Best epoch | 15 of 15 |
| Val loss | 0.6440 |
| Val AUROC | 0.6946 |
| Val F1 | 0.6382 |
| Val accuracy | 67.14% |
| Val precision | 0.6963 |
| Val recall | 0.5891 |
| Val AUPRC | 0.6769 |
| Stop reason | max epochs reached |

---

## 9. Final Test Metrics

Evaluated on test split at best checkpoint (epoch 15).
**Reported for analysis only. Not used as fitness signal or gate criterion.**

| Metric | Value |
|---|---|
| Test loss | 0.6548 |
| Test accuracy | 65.77% |
| Test precision | 0.6634 |
| Test recall | 0.5968 |
| Test F1 | 0.6283 |
| Test AUROC | 0.6708 |
| Test AUPRC | 0.6560 |

---

## 10. Confusion Matrix

Test split at best checkpoint (epoch 15).
Labels: 0 = No Finding (negative), 1 = Any Pathology (positive).

```
[[ 765   305]    (TN  FP)
 [ 406   601]]   (FN  TP)
```

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | TN = 765 | FP = 305 |
| **Actual Positive (1)** | FN = 406 | TP = 601 |

Confusion matrix is non-degenerate — predictions span both classes.

---

## 11. CV Health Analysis

Per-epoch CV health checks:

| Epoch | COV_PSD | COV_SYMMETRIC | QUAD_FINITE | NO_NAN_INF |
|---|---|---|---|---|
|  1 | PASS | PASS | PASS | PASS |
|  2 | PASS | PASS | PASS | PASS |
|  3 | PASS | PASS | PASS | PASS |
|  4 | PASS | PASS | PASS | PASS |
|  5 | PASS | PASS | PASS | PASS |
|  6 | PASS | PASS | PASS | PASS |
|  7 | PASS | PASS | PASS | PASS |
|  8 | PASS | PASS | PASS | PASS |
|  9 | PASS | PASS | PASS | PASS |
| 10 | PASS | PASS | PASS | PASS |
| 11 | PASS | PASS | PASS | PASS |
| 12 | PASS | PASS | PASS | PASS |
| 13 | PASS | PASS | PASS | PASS |
| 14 | PASS | PASS | PASS | PASS |
| 15 | PASS | PASS | PASS | PASS |

**Overall (all 15 epochs):**
- COV_PSD:          PASS
- COV_SYMMETRIC:    PASS
- QUAD_FINITE:      PASS
- NO_NAN_INF:       PASS

**mu magnitude trends:** Mean mu magnitude started at 0.6384 and ended at 0.6383 (epoch 15). Max mu: 1.9479. These values indicate the Gaussian state is being displaced non-trivially from vacuum throughout training.

**Squeezing evolution:** Squeezing norm started at 0.0640 and ended at 0.1674, reflecting adaptation of the squeezing parameters.

**Ansatz parameter norm:** Started at 15.5122 and ended at 12.8502, confirming active parameter updates throughout training.

**Gradient health:** All gradient norms remained finite across all epochs. No gradient explosion or vanishing detected.

---

## 12. Latency Analysis

| Metric | Value |
|---|---|
| Mean latency | 2.1538 ms/sample |
| Std latency | 0.1523 ms/sample |
| Measurement | 100 single-sample forward passes |
| Backbone device | CUDA |
| CV backend | CPU-only |

**Note:** CV circuit simulation is CPU-bound (GaussianBackend constraint). Backbone runs on CUDA for speed but the Gaussian circuit per-sample loop dominates latency. For comparison: Q21 DV Hybrid measured 54.79 ms/sample (CPU), Q22 Tiny Classical measured 1.48 ms/sample (GPU). Direct cross-model latency comparison is not architecturally meaningful due to different device assignments.

---

## 13. Comparison Anchor Table

**Preliminary anchor table only. Formal DV vs CV interpretation is deferred to Q28.**

| Metric | Q21 DV Hybrid | Q22 Tiny Classical | Q27 CV Hybrid |
|---|---|---|---|
| Test AUROC | 0.6800 | 0.6625 | 0.6708 |
| Test F1 | 0.6159 | 0.5961 | 0.6283 |
| Test Accuracy | 63.84% | 64.37% | 65.77% |
| Trainable Params | 574 | 526 | 536 |
| Backbone | frozen pretrained | frozen pretrained | frozen pretrained |
| Best epoch | 15 of 15 | 15 of 15 | 15 of 15 |

This table is an anchor only. No interpretation of the Q27 vs Q21 delta should be drawn here. See Q28 for formal analysis.

---

## 14. Interpretation

**Training completion:** Training ran the full 15 epochs (max epochs reached).

**Numerical stability:** The CV pipeline remained numerically stable throughout all 15 epochs. No NaN or inf was detected in loss, logits, first moments, or gradient norms at any point.

**Prediction quality:** The model produced non-degenerate predictions on the test set — both classes are represented in the confusion matrix.

**Gradient health:** All three trainable components (compression, ansatz, readout) received non-zero gradients throughout training, confirming end-to-end gradient flow through the Gaussian circuit. Backbone received zero gradient throughout.

**What cannot be concluded:** The Q27 result alone does not establish any claim about quantum advantage, quantum inductive bias, or superiority of CV over DV quantum circuits. Single-seed results without statistical validation cannot support such conclusions. Formal comparative analysis is deferred to Q28.

---

## 15. Required Scientific Guardrail

> Q27 establishes the first VinDr-SpineXR CV binary benchmark under the QStrata framework. This result does NOT establish quantum advantage. Formal DV vs CV interpretation is deferred to Q28.

---

## 16. Next Slice

**Q28 — DV vs CV Binary Comparative Report**

Purpose: formal scientific comparison of DV hybrid (Q21), CV hybrid (Q27), and classical controls (Q17, Q22) on VinDr-SpineXR binary classification.

---

```
Q27 status: COMPLETE — PASS
Q28 status: NEXT — DV vs CV Binary Comparative Report
```
