# VinDr-SpineXR DV Hybrid Pretrained-Backbone Full Training Report
## Slice Q21

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

This is the first scientifically valid VinDr-SpineXR DV hybrid benchmark.

Slice Q19 ran the initial DV hybrid full training but used a randomly initialized frozen CNN
backbone — the quantum head was effectively learning from noise. Q20 confirmed that the
C006-D040 PneumoniaMNIST-pretrained backbone (depthwise_sep [64,128], 91.98% val acc) is
fully compatible with `DVHybridCNNQNN` (28 matched keys, 0 missing, frozen validation PASS).

This slice (Q21) runs the first valid VinDr-SpineXR DV hybrid training: same architecture,
same training procedure, but with meaningful pretrained features feeding the quantum head.

**No quantum advantage is claimed.**
**Test metrics are reported for analysis only — not used as fitness signal or gate criterion.**
**Checkpoint is gitignored; not committed.**

---

## 2. Q19 Limitation Reminder

Q19 full training (random backbone) produced degenerate results:

| Metric | Q19 Value | Interpretation |
|---|---|---|
| Test AUROC | 0.5442 | Marginally above chance |
| Test F1 | 0.0000 | All predictions = class 0 |
| Confusion matrix | TN=1070, FP=0, FN=1007, TP=0 | Complete class collapse |
| Val AUROC (best) | 0.5285 | At epoch 2 of 6 |

**Q19 is not a valid DV benchmark.** The quantum head had no discriminative signal to
learn from because the frozen backbone was randomly initialized. Q21 corrects this.

---

## 3. Q20 Feasibility Summary

| Check | Q20 Result |
|---|---|
| Checkpoint | `checkpoints/c006_d040_classical_anchor.pt` |
| Source | Slice Q6 — C006-D040 PneumoniaMNIST classical anchor |
| Matched backbone keys | 28 |
| Missing backbone keys | 0 |
| Unexpected keys | 0 |
| Skipped classifier keys | 2 |
| Backbone frozen | PASS |
| Theta/proj/readout grad | Non-zero (2.80e-02 / 1.43e-02 / 2.88e-01) |
| Backbone grad | None (frozen, correct) |
| Probability conservation | PASS (1.000000) |
| Feasibility verdict | PRETRAINED_BACKBONE_DV_READY: YES |

---

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Dataset root | `data/processed/vindr_binary_roi_224` |
| Backbone checkpoint | `checkpoints/c006_d040_classical_anchor.pt` |
| Backbone source | Slice Q6 — C006-D040 PneumoniaMNIST pretrained |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Loss | Unweighted CrossEntropyLoss |
| Batch size | 4 |
| Max epochs | 15 |
| Seed | 42 |
| Early stopping patience | 4 (monitor: val loss, minimize) |
| Device | CPU (CUDA verified available; quantum circuit is CPU-only) |
| Class weights | None |
| Augmentation | None |

---

## 5. Backbone Loading Summary

| Check | Result |
|---|---|
| Checkpoint format | `model_state_dict` dict |
| Matched backbone keys | 28 |
| Skipped classifier keys | 2 |
| Unexpected keys | 0 |
| Backbone frozen after load | YES |

**Architecture:**
```
backbone.0.*  — depthwise_sep block 0 (1 → 64 channels)   [14 keys, 9,612→ partial]
backbone.1.*  — depthwise_sep block 1 (64 → 128 channels) [14 keys]
backbone.2    — AdaptiveAvgPool2d(1,1)                     [no parameters]
backbone.3    — Flatten                                     [no parameters]
```

| Component | Parameters |
|---|---|
| Backbone (frozen, pretrained) | 9,612 |
| Projection Linear(128, 4) | 516 |
| Theta (4-qubit variational) | 24 |
| Readout Linear(16, 2) | 34 |
| **Trainable total** | **574** |
| **All parameters** | **10186** |

**Device note:** CUDA confirmed available. Quantum circuit simulation is CPU-only
(QStrata Backend constraint); model runs on CPU throughout.

---

## 6. Per-Epoch Training Table

### Train metrics

| Epoch | Loss | Accuracy | Time |
|---|---|---|---|
|  1 | 0.6907 | 54.25% | 547.2s |
|  2 | 0.6747 | 61.81% | 543.4s |
|  3 | 0.6629 | 62.26% | 544.2s |
|  4 | 0.6566 | 63.20% | 546.7s |
|  5 | 0.6530 | 62.87% | 545.1s |
|  6 | 0.6506 | 62.99% | 544.2s |
|  7 | 0.6488 | 63.17% | 543.9s |
|  8 | 0.6477 | 63.26% | 548.6s |
|  9 | 0.6473 | 63.33% | 546.2s |
| 10 | 0.6459 | 63.83% | 546.0s |
| 11 | 0.6452 | 63.99% | 543.8s |
| 12 | 0.6435 | 63.66% | 542.4s |
| 13 | 0.6441 | 63.86% | 545.6s |
| 14 | 0.6433 | 64.00% | 546.4s |
| 15 | 0.6430 | 64.00% | 545.1s |

### Validation metrics

| Epoch | Loss | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | Time |
|---|---|---|---|---|---|---|---|---|
|  1 | 0.6850 | 59.69% | 0.7159 | 0.2994 | 0.4222 | 0.6416 | 0.6572 | 547.2s |
|  2 | 0.6713 | 61.00% | 0.6535 | 0.4412 | 0.5268 | 0.6434 | 0.6558 | 543.4s |
|  3 | 0.6612 | 62.79% | 0.6297 | 0.5915 | 0.6100 | 0.6477 | 0.6447 | 544.2s |
|  4 | 0.6579 | 62.91% | 0.6236 | 0.6206 | 0.6221 | 0.6482 | 0.6432 | 546.7s |
|  5 | 0.6521 | 63.33% | 0.6513 | 0.5479 | 0.5951 | 0.6580 | 0.6503 | 545.1s |
|  6 | 0.6501 | 62.91% | 0.6550 | 0.5200 | 0.5797 | 0.6686 | 0.6564 | 544.2s |
|  7 | 0.6485 | 63.21% | 0.6557 | 0.5309 | 0.5867 | 0.6723 | 0.6577 | 543.9s |
|  8 | 0.6477 | 63.86% | 0.6527 | 0.5673 | 0.6070 | 0.6703 | 0.6554 | 548.6s |
|  9 | 0.6472 | 63.92% | 0.6558 | 0.5612 | 0.6048 | 0.6723 | 0.6569 | 546.2s |
| 10 | 0.6460 | 63.98% | 0.6637 | 0.5430 | 0.5973 | 0.6748 | 0.6603 | 546.0s |
| 11 | 0.6455 | 63.27% | 0.6595 | 0.5236 | 0.5838 | 0.6769 | 0.6623 | 543.8s |
| 12 | 0.6471 | 63.80% | 0.6352 | 0.6206 | 0.6278 | 0.6715 | 0.6560 | 542.4s |
| 13 | 0.6435 | 64.22% | 0.6697 | 0.5382 | 0.5968 | 0.6801 | 0.6632 | 545.6s |
| 14 | 0.6438 | 64.10% | 0.6428 | 0.6085 | 0.6252 | 0.6751 | 0.6600 | 546.4s |
| 15 | 0.6422 | 64.04% | 0.6480 | 0.5891 | 0.6171 | 0.6800 | 0.6621 | 545.1s |

---

## 7. Best Validation Epoch

| Metric | Value |
|---|---|
| Best epoch | 15 |
| Best val loss | 0.6422 |
| Best val AUROC | 0.6800 |
| Stop reason | max epochs reached |
| Total epochs run | 15 |

---

## 8. Final Test Metrics

Evaluated on test split at best checkpoint (epoch 15).
**Reported for analysis only. Not used as fitness signal or gate criterion.**

| Metric | Value |
|---|---|
| Test loss | 0.6429 |
| Test accuracy | 63.84% |
| Test precision | 0.6350 |
| Test recall | 0.5978 |
| Test F1 | 0.6159 |
| Test AUROC | 0.6800 |
| Test AUPRC | 0.6571 |

---

## 9. Confusion Matrix

Test split at best checkpoint. Labels: 0 = No Finding (negative), 1 = Any Pathology (positive).

```
[[ 724   346]    (TN  FP)
 [ 405   602]]   (FN  TP)
```

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | TN = 724 | FP = 346 |
| **Actual Positive (1)** | FN = 405 | TP = 602 |

---

## 10. Gradient Health Analysis

Per-epoch quantum/hybrid gradient norms (last batch of each train epoch):

| Epoch | θ grad norm | Proj grad norm | Readout grad norm | Total grad norm |
|---|---|---|---|---|
|  1 | 1.66e-02 | 1.29e-02 | 5.23e-02 | 5.63e-02 |
|  2 | 6.63e-02 | 4.87e-02 | 1.66e-01 | 1.85e-01 |
|  3 | 5.26e-02 | 4.65e-02 | 1.15e-01 | 1.35e-01 |
|  4 | 1.12e-01 | 4.48e-02 | 1.86e-01 | 2.22e-01 |
|  5 | 4.21e-02 | 3.48e-02 | 8.53e-02 | 1.01e-01 |
|  6 | 6.87e-02 | 1.38e-01 | 6.73e-02 | 1.68e-01 |
|  7 | 1.25e-01 | 1.85e-01 | 2.77e-01 | 3.56e-01 |
|  8 | 3.92e-01 | 3.82e-01 | 4.97e-01 | 7.39e-01 |
|  9 | 8.83e-02 | 1.89e-01 | 3.20e-01 | 3.82e-01 |
| 10 | 2.99e-01 | 1.36e-01 | 2.98e-01 | 4.44e-01 |
| 11 | 2.18e-01 | 8.44e-02 | 3.22e-01 | 3.98e-01 |
| 12 | 4.66e-02 | 1.47e-01 | 6.73e-02 | 1.69e-01 |
| 13 | 2.11e-01 | 1.12e-01 | 2.35e-01 | 3.35e-01 |
| 14 | 2.28e-01 | 1.09e-01 | 2.71e-01 | 3.70e-01 |
| 15 | 2.30e-01 | 1.01e-01 | 1.24e-01 | 2.80e-01 |

**Backbone gradient confirmation:** Backbone received **no gradients at any epoch** (frozen).
Per-epoch backbone gradient check performed after every `loss.backward()` call.

**Gradient health assessment:**
- **Theta:** Non-zero throughout training → quantum parameters actively updated
- **Projection:** Non-zero throughout training → projection layer actively updated
- **Readout:** Non-zero throughout training → readout layer actively updated
- **Total:** Non-zero throughout training → end-to-end gradient flow confirmed
- No NaN, no inf, no exploding gradients detected

---

## 11. Probability Conservation

Measured across full test set at best checkpoint:

| Metric | Value |
|---|---|
| Probability sum mean | 1.000000 |
| Max deviation from 1.0 | 7.15e-07 |

DV quantum circuit outputs valid probability distributions with pretrained backbone input.
Deviation < 1e-5 is numerical noise from float32 arithmetic.

---

## 12. Latency Analysis

**Methodology:** Wall-clock block timing with `torch.cuda.synchronize()` (no-op on CPU path;
CUDA is confirmed available in the environment).
- Warmup: 10 forward passes before timed measurements
- Timed blocks: 30 independent forward passes
- Each block: 1 forward pass of 4 images through the full DV hybrid model
- Per-sample latency = block_duration_ms / batch_size

| Metric | Value |
|---|---|
| Mean latency | 54.7855 ms/sample |
| Std latency | 3.0217 ms/sample |
| Std % of mean | 5.52% |

**Note:** Latency measures CPU quantum circuit simulation speed, not GPU inference.
Q17 classical baseline latency (0.8114 ms/sample) was measured on GPU. Direct comparison
is not valid. Q22 comparative report should note this asymmetry explicitly.

---

## 13. Classical vs DV Comparison

| Metric | Q17 classical | Q19 DV (random backbone) | Q21 DV (pretrained backbone) |
|---|---|---|---|
| Test AUROC | 0.6224 | 0.5442 | 0.6800 |
| Test F1 | 0.5355 | 0.0000 | 0.6159 |
| Test accuracy | 60.66% | 51.52% | 63.84% |
| Backbone | N/A | Random frozen | Pretrained frozen (C006-D040) |
| Best epoch | 1 of 6 | 2 of 6 | 15 of 15 |

---

## 14. Interpretation — Case A

Strong follow-up candidate. Pretrained features enabled the quantum head to surpass both the random-backbone DV baseline (Q19) and the classical baseline (Q17). Q22 comparative report is the immediate priority.

```
A better DV result than Q19 demonstrates the value of pretrained classical features
feeding the quantum head. This alone does NOT establish quantum advantage over the
classical baseline. Comparative interpretation is deferred to Slice Q22.
```

---

## 15. Scientific Guardrail

```
A better DV result than Q19 demonstrates the value of pretrained classical features
feeding the quantum head. This alone does NOT establish quantum advantage over the
classical baseline. Comparative interpretation is deferred to Slice Q22.
```

---

## 16. Next Slice Recommendation

```
Slice Q22 — VinDr-SpineXR Classical vs DV Hybrid Comparative Report

Goal:
Compare Q17 classical baseline and Q21 DV hybrid pretrained baseline with full
metrics, caveats, and the Q20 interpretation guardrail.
```

---

```
DV pretrained full training status: PASS
```
