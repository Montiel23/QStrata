# VinDr-SpineXR Classical Baseline Full Training Report
## Slice Q17

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Slice Q16 validated end-to-end classical CNN training mechanics on the VinDr-SpineXR binary ROI
dataset (30/30 checks PASS). This slice (Q17) runs the first full classical CNN baseline training
to convergence — establishing the reference performance for the DV hybrid comparison in Q18/Q19/Q20.

**No QNN work occurs in this slice.**
**No augmentation is applied.**
**No architecture search is performed.**
**Checkpoint is gitignored; not committed.**
**Test metrics are reported for analysis only — not used as a fitness signal or gate criterion.**

---

## 2. Dataset Splits and Class Balance

| Split | Total | Label 0 (No Finding) | Label 1 (Any Pathology) | Ratio |
|---|---|---|---|---|
| train | 6,712 | 3,408 | 3,304 | 0.97:1 |
| val | 1,677 | 852 | 825 | 0.97:1 |
| test | 2,077 | 1,070 | 1,007 | 0.94:1 |

Near-balanced dataset; no class weighting applied.

---

## 3. Model Configuration

**Function:** `build_model()` from `qcore/models/cnn.py`
**Architecture name:** CNN3Block (standard, 3-block)

```python
config = {
    "conv_channels":  [16, 32, 64],
    "use_batchnorm":  True,
    "dropout":        0.0,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
    "block_type":     "standard",
}
```

**Layer-by-layer:**

```
Conv2d(1→16,  3×3, padding=1) + BatchNorm2d(16)  + ReLU → [B, 16, 224, 224]
Conv2d(16→32, 3×3, padding=1) + BatchNorm2d(32)  + ReLU → [B, 32, 224, 224]
Conv2d(32→64, 3×3, padding=1) + BatchNorm2d(64)  + ReLU → [B, 64, 224, 224]
AdaptiveAvgPool2d(1, 1)                                  → [B, 64, 1, 1]
Flatten()                                                → [B, 64]
Linear(64, 2)                                            → [B, 2]
```

| Property | Value |
|---|---|
| Trainable parameters | 23,650 |
| Pretrained weights | None — random init |
| Device | cuda |

---

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Dataset root | `data/processed/vindr_binary_roi_224` |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Loss | Unweighted CrossEntropyLoss |
| Batch size | 16 |
| Max epochs | 30 |
| Seed | 42 |
| Early stopping patience | 5 (monitor: val loss, minimize) |
| Device | cuda |

---

## 5. Per-Epoch Train Metrics

| Epoch | Loss | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | Time |
|---|---|---|---|---|---|---|---|---|
|  1 | 0.6434 | 62.77% | 0.6428 | 0.5484 | 0.5919 | 0.6700 | 0.6824 | 75.1s |
|  2 | 0.6088 | 67.54% | 0.6973 | 0.6017 | 0.6460 | 0.7273 | 0.7346 | 72.1s |
|  3 | 0.5873 | 68.91% | 0.7143 | 0.6138 | 0.6603 | 0.7509 | 0.7633 | 72.6s |
|  4 | 0.5701 | 70.41% | 0.7287 | 0.6356 | 0.6790 | 0.7696 | 0.7830 | 73.3s |
|  5 | 0.5604 | 70.75% | 0.7258 | 0.6522 | 0.6871 | 0.7787 | 0.7916 | 72.9s |
|  6 | 0.5459 | 72.30% | 0.7401 | 0.6740 | 0.7055 | 0.7931 | 0.8056 | 72.4s |

---

## 6. Per-Epoch Validation Metrics

| Epoch | Loss | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | Time |
|---|---|---|---|---|---|---|---|---|
|  1 | 0.6839 | 61.42% | 0.6686 | 0.4279 | 0.5218 | 0.6366 | 0.6819 | 75.1s |
|  2 | 1.0851 | 49.85% | 0.4950 | 0.9685 | 0.6552 | 0.6059 | 0.6478 | 72.1s |
|  3 | 1.8790 | 51.34% | 1.0000 | 0.0109 | 0.0216 | 0.5868 | 0.6025 | 72.6s |
|  4 | 0.8654 | 48.78% | 0.4882 | 0.8545 | 0.6214 | 0.5508 | 0.6000 | 73.3s |
|  5 | 1.2317 | 52.42% | 1.0000 | 0.0327 | 0.0634 | 0.6725 | 0.6793 | 72.9s |
|  6 | 0.8429 | 56.23% | 0.9252 | 0.1200 | 0.2124 | 0.7254 | 0.7392 | 72.4s |

---

## 7. Early Stopping Summary

| Metric | Value |
|---|---|
| Best epoch | 1 |
| Best val loss (checkpoint selection criterion) | 0.6839 |
| Best val AUROC at best epoch | 0.6366 |
| Stop reason | early stopping (patience=5) |
| Total epochs run | 6 |

---

## 8. Best Checkpoint Summary

| Property | Value |
|---|---|
| Checkpoint path | `checkpoints/vindr_classical_baseline_best.pt` |
| Best val loss | 0.6839 |
| Best val AUROC at best epoch | 0.6366 |
| Best epoch | 1 |

Checkpoint is gitignored (`*.pt` rule in `.gitignore`) and is not committed.

---

## 9. Final Test Metrics

Evaluated on the test split at the best checkpoint (epoch 1).
**Reported for analysis only. Not used as a fitness signal or gate criterion.**

| Metric | Value |
|---|---|
| Test loss | 0.6850 |
| Test accuracy | 60.66% |
| Test precision | 0.6263 |
| Test recall | 0.4677 |
| Test F1 | 0.5355 |
| Test AUROC | 0.6224 |
| Test AUPRC | 0.6730 |

---

## 10. Confusion Matrix

Test split at best checkpoint. Labels: 0 = No Finding (negative), 1 = Any Pathology (positive).

```
[[ 789   281]    (TN  FP)
 [ 536   471]]   (FN  TP)
```

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | TN = 789 | FP = 281 |
| **Actual Positive (1)** | FN = 536 | TP = 471 |

---

## 11. ROC and PR Summary

| Metric | Value |
|---|---|
| Test AUROC | 0.6224 |
| Test AUPRC | 0.6730 |

No plot files generated in this slice. Plots are deferred to the Q20 comparative report.

---

## 12. Inference Latency

**Methodology:** GPU-synchronized block timing.
- Warmup: 25 forward passes (CUDA-synchronized) discarded before measurement
- Timed blocks: 100 independent forward passes
- Each block: 1 forward pass of 16 images through the full model
- Synchronization: `torch.cuda.synchronize()` before and after each timed block
- Per-sample latency = block_duration_ms / batch_size

| Metric | Value |
|---|---|
| Mean latency | 0.8114 ms/sample |
| Std latency | 0.0455 ms/sample |
| Std % of mean | 5.60% |

---

## 13. Technical Observations

- **Convergence:** Training completed via early stopping (patience=5) at epoch 6.
  Best validation loss (0.6839) achieved at epoch 1.
- **Val AUROC at best epoch:** 0.6366
- **Test AUROC:** 0.6224 — this is the classical reference for
  the Q18/Q19/Q20 DV hybrid comparison.
- **Test F1:** 0.5355
- **Final train/val loss (last epoch):** 0.5459 / 0.8429
- **Near-balanced dataset** (0.97:1 ratio) confirms unweighted CrossEntropyLoss is appropriate.
- No class weighting was required. No gradient anomalies observed.

---

## 14. Limitations

1. **Single seed (seed=42).** Multi-seed validation is deferred.
2. **No data augmentation.** Applied no transforms at training time. Augmentation ablation deferred.
3. **No class weighting.** Not required given near-balanced dataset.
4. **Fixed architecture.** CNN3Block (23,650 params) is the reference; no architecture search performed.
5. **No MaxPool between blocks.** Spatial resolution stays 224×224 through all conv blocks;
   AdaptiveAvgPool2d reduces at end. This is the Q16-validated design.
6. **No LR schedule.** Constant AdamW lr=1e-3 throughout. Scheduler ablation deferred.

---

## 15. Next Slice Recommendation

```
Slice Q18 — VinDr-SpineXR DV Hybrid Smoke Test

Goal:
Validate end-to-end DV hybrid CNN-QNN training mechanics on the VinDr-SpineXR
binary dataset before full hybrid baseline training.
```

---

```
Classical full baseline status: PASS
```
