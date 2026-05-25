# DV Hybrid PneumoniaMNIST Multi-Seed Stability Validation

- **Status:** Complete
- **Date:** 2026-05-25
- **Branch:** feature/qnn-integration
- **Slice:** Q9

---

## 1. Title

DV Hybrid PneumoniaMNIST Multi-Seed Stability Validation — Slice Q9

Multi-seed (4 seeds × 10 epochs) stability screening of `DVHybridCNNQNN` on PneumoniaMNIST.
Q8 validated the architecture at seed=42 over 30 epochs (best val acc 92.18%); this slice
checks whether the result is stable across seeds before committing to more expensive
experiments or new datasets.

---

## 2. Context

Q8 full baseline (seed=42, 30 epochs) achieved 92.18% best val acc, with projection
gradients active throughout and no majority-class collapse. The result was meaningful but
based on a single seed. Q9 answers: was Q8 stable, or was it seed luck?

Four seeds (42, 7, 123, 999) are run for 10 epochs each. Q8 showed the model plateaus in
the 90–92% range by epoch 10 before further improvement requires 20+ epochs; 10 epochs is
therefore sufficient for stability screening before committing to expensive full reruns per
seed.

Architecture is unchanged from Q8. All training conditions are identical (same checkpoint,
same class weights, same batch size, same LR) except for the per-seed random initialisation
of the trainable hybrid parameters.

---

## 3. Architecture Summary

Architecture unchanged from Q8 (`DVHybridCNNQNN`).

| Component | Type | Frozen / Trainable | Parameters | Source |
|---|---|---|---|---|
| CNN backbone (`model[:4]`) | 2× depthwise-sep + AdaptiveAvgPool2d + Flatten | **Frozen** | 9,612 | Pretrained C006-D040 (Q6) |
| Projection layer | `nn.Linear(128, 4)`, no activation | **Trainable** | 516 | Random init per seed |
| Quantum theta | `nn.Parameter` shape `(1, 2, 4, 3)` | **Trainable** | 24 | Random init per seed |
| Readout layer | `nn.Linear(16, 2)` | **Trainable** | 34 | Random init per seed |
| **Total trainable** | | | **574** | |
| **Total frozen** | | | **9,612** | |

---

## 4. Run Configuration

| Parameter | Value |
|---|---|
| Seeds | 42, 7, 123, 999 |
| Epochs per seed | 10 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | `nn.CrossEntropyLoss(weight=balanced)` |
| Class weights | [0.742141, 0.257859] |
| Checkpoint | `checkpoints/c006_d040_classical_anchor.pt` |
| Device | cpu (quantum simulator CPU-only) |
| Test accuracy role | **Analysis only — not a fitness gate** |

---

## 5. Per-Seed Results

| Seed | Best Val Acc | Best Ep | Test Acc* | Train Acc @ Best | Val Loss @ Best | Precision | Recall | F1 | AUROC | AUPRC | Confusion Matrix | Proj GN Ep1 | Proj All Active | Prob Valid | Collapse Absent | Mean Ep Time | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 91.98% | 1 | 88.14% | 74.81% | 0.4920 | 0.8873 | 0.9282 | 0.9073 | 0.9433 | 0.9558 | TN=188, FP=46, FN=28, TP=362 | 2.62e-02 | YES | YES | YES | 61.7s | COMPLETED |
| 7 | 91.98% | 7 | 88.46% | 92.10% | 0.2176 | 0.8859 | 0.9359 | 0.9102 | 0.9483 | 0.9639 | TN=187, FP=47, FN=25, TP=365 | 2.05e-02 | YES | YES | YES | 62.6s | COMPLETED |
| 123 | 92.18% | 8 | 88.14% | 92.29% | 0.2196 | 0.8780 | 0.9410 | 0.9084 | 0.9472 | 0.9621 | TN=183, FP=51, FN=23, TP=367 | 2.63e-02 | YES | YES | YES | 60.7s | COMPLETED |
| 999 | 91.98% | 4 | 88.30% | 91.53% | 0.2387 | 0.8783 | 0.9436 | 0.9098 | 0.9464 | 0.9601 | TN=183, FP=51, FN=22, TP=368 | 2.09e-02 | YES | YES | YES | 60.0s | COMPLETED |

*Test accuracy is analysis-only — not a fitness gate.

---

## 6. Aggregate Statistics

| Metric | Value |
|---|---|
| mean_best_val_acc | 92.03% |
| std_best_val_acc | 0.08% |
| min_best_val_acc | 91.98% |
| max_best_val_acc | 92.18% |
| mean_test_acc (analysis only) | 88.26% |
| std_test_acc (analysis only) | 0.13% |
| mean_f1 | 0.9089 |
| std_f1 | 0.0012 |
| mean_auroc | 0.9463 |
| std_auroc | 0.0019 |
| mean_epoch_time | 61.2s |

---

## 7. Stability Gate

| Criterion | Threshold | Actual | Result |
|---|---|---|---|
| std(best_val_acc) | ≤ 1.0% | 0.08% | PASS |
| Max seed gap below mean val acc | ≤ 2.5% | 0.05% | PASS |
| std(test_acc) [analysis] | ≤ 2.0% | 0.13% | PASS |
| All runs completed | 0 failures | 0 failures | PASS |
| Proj grad > 0 epoch 1 all seeds | All seeds | 4/4 seeds | PASS |
| Proj grads active all epochs all seeds | All seeds | 4/4 seeds | PASS |
| Prob sums valid all seeds | All seeds | 4/4 seeds | PASS |
| Collapse absent all seeds | All seeds | 4/4 seeds | PASS |

---

## 8. Comparison Against Q8 Seed=42 Full Baseline

| Metric | Q8 (seed=42, 30ep) | Q9 mean (4 seeds, 10ep) |
|---|---|---|
| best_val_acc | 92.18% | 92.03% ± 0.08% |
| test_acc (analysis) | 87.98% | 88.26% ± 0.13% |
| mean_f1 | — | 0.9089 ± 0.0012 |
| mean_auroc | — | 0.9463 ± 0.0019 |

---

## 9. Interpretation

All eight stability gate criteria passed. The DV hybrid model achieves mean best val acc 92.03% ± 0.08% across four seeds (range 91.98–92.18%), within the ≤1.0% std threshold. Gradient flow through the quantum projection layer remained active across all epochs for all seeds, and quantum probability sums remained valid (≈1.0) throughout all runs. No majority-class collapse was observed in any seed. The DV hybrid baseline is considered stable and suitable for progression to the next experimental stage.

---

## 10. Explicit Verdict

VERDICT: Stable enough for VinDr-SpineXR binary planning

---

## 11. Recommended Next Step

The DV hybrid baseline is stable across seeds. The recommended next step (Q10) is to begin VinDr-SpineXR binary classification planning: identify the target binary task (e.g., normal vs. abnormal), prepare the data pipeline (resizing, normalisation, class balancing), confirm that `DVHybridCNNQNN` is compatible with the new dataset's image dimensions, and run a 3-epoch sanity check analogous to Slice Q7 on PneumoniaMNIST.
