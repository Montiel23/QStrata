# Q22: VinDr-SpineXR Approximate Trainable-Parameter-Matched Classical Control

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Slice Q17 established the VinDr-SpineXR classical CNN baseline (CNN3Block, 23,650 trainable parameters, unfrozen backbone, test AUROC 0.6224, F1 0.5355). Slice Q21 produced the first scientifically valid VinDr-SpineXR DV hybrid benchmark: a frozen, pretrained C006-D040 backbone feeding a DV quantum head with only 574 trainable parameters (projection, theta, readout), achieving test AUROC 0.6800 and F1 0.6159 — improvements over both Q17 and the degenerate random-backbone Q19 result. Before any comparative interpretation of Q17 vs Q21 is meaningful, Slice Q22 introduces a necessary scientific control: a tiny classical head of approximately equivalent trainable parameter count (~574) attached to the same frozen pretrained backbone, holding all other experimental conditions constant.

---

## 2. Scientific Rationale

The Q21 DV hybrid improvement over Q17 admits at least three candidate explanations that cannot be disentangled without appropriate controls:

1. **Compact trainable bottleneck effect** — forcing the network through a very small trainable module (574 params) may act as a strong regulariser, preventing overfitting on a relatively small dataset. This effect would be fully explained by compact bottleneck geometry regardless of whether the head is quantum or classical.

2. **Regularization via frozen backbone** — freezing a pretrained backbone eliminates the risk of destabilising the feature extractor during fine-tuning. This is a classical transfer-learning effect and would benefit any head, quantum or classical.

3. **Quantum inductive bias** — the DV quantum circuit (variational ansatz, entanglement structure, Hilbert space geometry) may provide a qualitatively different inductive bias that leads to better generalisation on medical image features than a classical linear map of the same parameter count.

An approximate trainable-parameter-matched classical control directly tests whether hypothesis (1) — the compact bottleneck — can account for the Q21 improvement, independent of any quantum properties. If Q22 ≈ Q21, the bottleneck effect is a sufficient explanation. If Q22 < Q21, the gap is not explained by parameter count alone, increasing the scientific interest of the DV hybrid result — though not establishing quantum advantage.

---

## 3. Q22 Control Objective

Replace only the DV quantum head with a tiny classical head of approximately equivalent trainable parameter count, holding all else constant:

- **Backbone**: same frozen pretrained C006-D040 checkpoint
- **Backbone loading**: same key remapping ("0.*" → "backbone.0.*", "1.*" → "backbone.1.*", skip "5.*"), same freeze-and-assert procedure
- **Dataset**: same `VinDrSpineXRBinaryDataset` with identical train/val/test splits
- **Optimizer**: AdamW, lr = 1e-3 (identical to Q21)
- **Loss**: unweighted CrossEntropyLoss (identical to Q21)
- **Batch size**: 4 (identical to Q21)
- **Max epochs**: 15 (identical to Q21)
- **Early stopping patience**: 4 on val_loss (identical to Q21)
- **Seed**: 42 (identical to Q21)
- **Head**: TinyClassicalHead — replaces the DV quantum head with a two-layer classical MLP of approximately equal trainable parameter count

---

## 4. Model Architecture Summary

**Backbone (frozen, pretrained C006-D040):**

```
backbone[0]  — DepthwiseSepBlock (1 → 64 channels, BN, ReLU)
backbone[1]  — DepthwiseSepBlock (64 → 128 channels, BN, ReLU)
backbone[2]  — AdaptiveAvgPool2d(1, 1)
backbone[3]  — Flatten()
```
Output shape: (B, 128) — identical to the feature shape entering the Q21 quantum head.

**Tiny classical head (trainable):**

| Layer | Type | In → Out | Activation |
|---|---|---|---|
| projection | nn.Linear | 128 → 4 | — |
| act | nn.ReLU | — | — |
| readout | nn.Linear | 4 → 2 | — |

**Head dim computation:**
```
target_params = 574  (Q21 trainable count)
formula: H = round((574 - 2) / (in_features + 3))
       = round(572 / (128 + 3))
       = round(572 / 131)
       = round(4.366)
       = 4
actual params = H × (in_features + 3) + 2
              = 4 × 131 + 2
              = 526
```

**Full forward path:**
```
Input (B, 1, 224, 224)
  → [with torch.no_grad()] backbone  → (B, 128)   [frozen]
  → head.projection  → (B, 4)
  → head.act (ReLU)  → (B, 4)
  → head.readout     → (B, 2)       [logits]
```

**Total trainable params: 526** (all in head; backbone contributes 0 trainable parameters)

---

## 5. Parameter Count Verification

| Component | Q21 Count | Q22 Count |
|---|---|---|
| Backbone | 9,612 (frozen) | 9,612 (frozen) |
| Head (trainable) | 574 | 526 |
| **Trainable total** | **574** | **526** |
| **All parameters** | **10,186** | **10,138** |

| Metric | Value |
|---|---|
| Q21 trainable params | 574 |
| Q22 trainable params | 526 |
| Absolute delta | −48 |
| Relative delta | −8.4% |

Target range: 500–700 trainable params. **Q22 is within range (526). ✓**

---

## 6. Training Configuration

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
| Device | CUDA |
| Class weights | None |
| Augmentation | None |

---

## 7. Per-Epoch Training Table

All 13 tracked fields per epoch.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Precision | Val Recall | Val F1 | Val AUROC | Val AUPRC | Head Grad Norm | Total Grad Norm | Epoch Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  1 | 0.6842 | 0.5724 | 0.6746 | 0.6154 | 0.6271 | 0.5382 | 0.5793 | 0.6454 | 0.6492 | 1.1145e+00 | 1.1145e+00 | 69.5s |
|  2 | 0.6723 | 0.6101 | 0.6738 | 0.6219 | 0.6436 | 0.5188 | 0.5745 | 0.6470 | 0.6504 | 1.1784e+00 | 1.1784e+00 | 67.3s |
|  3 | 0.6707 | 0.6196 | 0.6791 | 0.6041 | 0.6472 | 0.4291 | 0.5160 | 0.6387 | 0.6502 | 4.8077e-01 | 4.8077e-01 | 67.2s |
|  4 | 0.6693 | 0.6217 | 0.6672 | 0.6351 | 0.6132 | 0.6994 | 0.6535 | 0.6636 | 0.6499 | 5.2395e-01 | 5.2395e-01 | 68.1s |
|  5 | 0.6685 | 0.6210 | 0.6647 | 0.6452 | 0.6427 | 0.6279 | 0.6352 | 0.6660 | 0.6520 | 1.6714e+00 | 1.6714e+00 | 68.5s |
|  6 | 0.6665 | 0.6244 | 0.6644 | 0.6446 | 0.6201 | 0.7164 | 0.6648 | 0.6720 | 0.6447 | 8.7849e-01 | 8.7849e-01 | 67.6s |
|  7 | 0.6656 | 0.6281 | 0.6621 | 0.6410 | 0.6485 | 0.5903 | 0.6180 | 0.6656 | 0.6545 | 2.5587e+00 | 2.5587e+00 | 68.0s |
|  8 | 0.6650 | 0.6310 | 0.6609 | 0.6357 | 0.6259 | 0.6448 | 0.6352 | 0.6639 | 0.6549 | 4.3348e-01 | 4.3348e-01 | 69.5s |
|  9 | 0.6637 | 0.6223 | 0.6591 | 0.6416 | 0.6390 | 0.6242 | 0.6315 | 0.6687 | 0.6572 | 1.2808e+00 | 1.2808e+00 | 69.9s |
| 10 | 0.6631 | 0.6301 | 0.6601 | 0.6267 | 0.5954 | 0.7527 | 0.6649 | 0.6765 | 0.6563 | 4.0014e-01 | 4.0014e-01 | 70.3s |
| 11 | 0.6627 | 0.6278 | 0.6581 | 0.6297 | 0.6253 | 0.6170 | 0.6211 | 0.6664 | 0.6576 | 7.6463e-01 | 7.6463e-01 | 70.0s |
| 12 | 0.6620 | 0.6292 | 0.6580 | 0.6380 | 0.6086 | 0.7406 | 0.6681 | 0.6826 | 0.6590 | 1.0809e+00 | 1.0809e+00 | 68.3s |
| 13 | 0.6616 | 0.6314 | 0.6573 | 0.6303 | 0.6306 | 0.6000 | 0.6149 | 0.6668 | 0.6601 | 9.0944e-01 | 9.0944e-01 | 68.2s |
| 14 | 0.6607 | 0.6296 | 0.6654 | 0.5886 | 0.5555 | 0.8194 | 0.6621 | 0.6807 | 0.6498 | 1.5340e+00 | 1.5340e+00 | 68.8s |
| 15 | 0.6595 | 0.6277 | 0.6550 | 0.6541 | 0.6917 | 0.5358 | 0.6038 | 0.6771 | 0.6655 | 1.4169e+00 | 1.4169e+00 | 68.8s |

**Stop reason:** max epochs reached (no early stopping triggered)

---

## 8. Best Validation Epoch

| Metric | Value |
|---|---|
| Best epoch | 15 |
| Best val loss | 0.6550 |
| Best val AUROC | 0.6771 |
| Best val F1 | 0.6038 |
| Stop reason | max epochs reached |
| Total epochs run | 15 |

---

## 9. Final Test Metrics

Evaluated on test split at best checkpoint (epoch 15).
**Reported for analysis only. Not used as fitness signal or gate criterion.**

| Metric | Value |
|---|---|
| Test loss | 0.6627 |
| Test accuracy | 64.37% |
| Test precision | 0.6618 |
| Test recall | 0.5422 |
| Test F1 | 0.5961 |
| Test AUROC | 0.6625 |
| Test AUPRC | 0.6559 |

---

## 10. Confusion Matrix

Test split at best checkpoint. Labels: 0 = No Finding (negative), 1 = Any Pathology (positive).

```
[[ 791   279]    (TN  FP)
 [ 461   546]]   (FN  TP)
```

| | Predicted Negative (0) | Predicted Positive (1) |
|---|---|---|
| **Actual Negative (0)** | TN = 791 | FP = 279 |
| **Actual Positive (1)** | FN = 461 | TP = 546 |

Confusion matrix is non-degenerate: both TN and TP are nonzero, both FP and FN are nonzero. **PASS.**

---

## 11. Latency Analysis

**Methodology:** 10 warmup passes, then 100 timed single-sample forward passes on CUDA with `torch.cuda.synchronize()`.

| Metric | Q22 Value | Q21 Value |
|---|---|---|
| Mean latency | **1.48 ms/sample** | 54.79 ms/sample |
| Measurement device | CUDA | CPU (quantum constraint) |

**Note:** Q21 latency is CPU-bound by quantum circuit simulation (QStrata Backend constraint). Q22 runs entirely on CUDA. Direct latency comparison is not architecturally valid — the 37× gap reflects circuit simulation overhead, not a meaningful speed advantage of one head design over another. The Q22 latency (1.48 ms/sample) is comparable to the Q17 classical baseline (0.81 ms/sample), as expected for GPU classical inference.

---

## 12. Q17 vs Q21 vs Q22 Comparison Table

| Metric | Q17 Classical | Q21 DV Hybrid | Q22 Tiny Classical |
|---|---|---|---|
| Test AUROC | 0.6224 | 0.6800 | **0.6625** |
| Test F1 | 0.5355 | 0.6159 | **0.5961** |
| Test Accuracy | 60.66% | 63.84% | **64.37%** |
| Test AUPRC | 0.6196 | 0.6571 | **0.6559** |
| Test Precision | 0.5625 | 0.6350 | **0.6618** |
| Test Recall | 0.5130 | 0.5978 | **0.5422** |
| Trainable Params | 23,650 | 574 | **526** |
| Backbone | unfrozen | frozen pretrained | frozen pretrained |
| Head type | classical MLP | DV quantum | tiny classical MLP |
| Best epoch | 1 of 6 | 15 of 15 | 15 of 15 |
| Latency | 0.81 ms/sample (GPU) | 54.79 ms/sample (CPU) | 1.48 ms/sample (GPU) |

---

## 13. Interpretation

**Q22 test AUROC (0.6625) is meaningfully lower than Q21 (0.6800), with a gap of ~0.018 (> 0.01 threshold).**

Q22 underperforms Q21, which increases scientific interest in the DV hybrid benchmark. However, this result does NOT establish quantum advantage. Further controls are required.

*This approximate trainable-parameter-matched classical control tests whether compact trainable bottleneck behavior can explain the Q21 improvement. A weaker result than Q21 increases scientific interest in the DV hybrid benchmark, but does NOT establish quantum advantage.*

**Contextual observations:**

1. **Q22 > Q17 in AUROC (+0.040)**: The frozen pretrained backbone alone — regardless of head type — substantially improves over the Q17 unfrozen baseline. This confirms that backbone pretraining and freezing is the dominant factor separating Q17 from Q21/Q22.

2. **Q22 < Q21 in AUROC (−0.018)**: The DV quantum head outperforms the approximately parameter-matched classical head. This gap is not attributable to parameter count alone, and is consistent with (but does not confirm) a quantum inductive bias effect.

3. **Q22 ≈ Q21 in accuracy (64.37% vs 63.84%)**: Accuracy is nearly identical. The AUROC gap reflects distributional differences in predicted probabilities and is a more nuanced discriminator than accuracy alone.

4. **The Q20 interpretation guardrail still applies**: The Q17 classical baseline may be architecturally weak (unfrozen backbone, no inter-block MaxPool). Q22 holds the backbone and training procedure constant with Q21, making the Q22 vs Q21 comparison architecturally clean — but no comparative interpretation should conflate the Q17 vs Q21 gap with the Q22 vs Q21 gap. They test different things.

5. **Scientific status**: No quantum advantage is claimed. The Q22 result motivates further study of the DV hybrid architecture — specifically, whether the head geometry, the entanglement structure, or the variational encoding provides the observed AUROC advantage over a classical bottleneck of the same size.

---

## 14. Next Slice Recommendation

Q23 = VinDr Classical vs DV Hybrid vs Tiny Classical Comparative Report.

This report should:
- Consolidate Q17, Q21, and Q22 results with full metric tables
- Apply all active guardrails (Q20 interpretation guardrail, Q19 backbone guardrail)
- Formally assess the three candidate explanations (compact bottleneck, frozen backbone regularization, quantum inductive bias)
- Document the latency asymmetry between CPU quantum circuit simulation (Q21) and GPU classical inference (Q22, Q17)
- Recommend next experimental steps if quantum inductive bias remains a viable hypothesis

---

```
Classical control tiny head status: PASS
```
