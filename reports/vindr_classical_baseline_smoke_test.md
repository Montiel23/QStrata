# VinDr-SpineXR Classical Baseline Smoke Test Report
## Slice Q16

**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-25  
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Slice Q15 implemented and validated `VinDrSpineXRBinaryDataset` — a PyTorch Dataset reading from the full 10,466-image export in `data/processed/vindr_binary_roi_224/`. All Q15 loader checks (29/29) passed.

This slice (Q16) validates the complete end-to-end classical training pipeline using that loader: data loading → model forward pass → loss computation → backward pass → optimizer step → tiny train loop → tiny val loop. This is a **mechanics validation only** — not a performance baseline, not full training, not QNN work.

**No model training to convergence occurs in this slice.**  
**No test evaluation occurs in this slice.**  
**No checkpoint is saved.**  
**No QNN work occurs in this slice.**

---

## 2. Dataset Root and Splits Used

| Parameter | Value |
|---|---|
| Dataset root | `data/processed/vindr_binary_roi_224/` |
| Manifest | `data/processed/vindr_binary_roi_224/manifest.csv` |
| Splits used | train, val only |
| Test split | Not evaluated in this slice |
| train length | 6,712 |
| val length | 1,677 |

---

## 3. Model Used

**Reused existing infrastructure:** `build_model` from `qcore/models/cnn.py`  
No new model file created.

**Configuration applied:**

```python
{
    "conv_channels":  [16, 32, 64],
    "use_batchnorm":  True,
    "dropout":        0.0,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
    "block_type":     "standard",
}
```

**Architecture (CNN3Block):**

```
Conv2d(1→16, 3×3, padding=1) + BatchNorm2d(16) + ReLU  → [B, 16, 224, 224]
Conv2d(16→32, 3×3, padding=1) + BatchNorm2d(32) + ReLU → [B, 32, 224, 224]
Conv2d(32→64, 3×3, padding=1) + BatchNorm2d(64) + ReLU → [B, 64, 224, 224]
AdaptiveAvgPool2d(1, 1)                                 → [B, 64, 1, 1]
Flatten()                                               → [B, 64]
Linear(64, 2)                                           → [B, 2]
```

| Property | Value |
|---|---|
| Trainable parameters | 23,650 |
| Input compatibility | Any `[B, 1, H, W]` |
| Output | Logits `[B, 2]` |
| Pretrained weights | None — random init |

---

## 4. Training Smoke Configuration

| Parameter | Value |
|---|---|
| Dataset root | `data/processed/vindr_binary_roi_224` |
| Input shape | `[8, 1, 224, 224]` |
| Labels | Binary: 0 (No Finding), 1 (Any Pathology) |
| Loss | `nn.CrossEntropyLoss` — unweighted |
| Optimizer | `Adam` (lr=1e-3) |
| Learning rate | `1e-3` |
| Batch size | 8 |
| Max train batches | 5 |
| Max val batches | 3 |
| Epochs | 1 |
| Seed | 42 |
| Device | `cuda` (confirmed available) |

---

## 5. Batch Validation

One batch sampled from the train DataLoader at the start of validation:

| Check | Observed | Expected | Status |
|---|---|---|---|
| x shape | `(8, 1, 224, 224)` | `(B, 1, 224, 224)` | ✅ PASS |
| y shape | `(8,)` | `(B,)` | ✅ PASS |
| y dtype | `torch.long` | `torch.long` | ✅ PASS |
| label values | `{0, 1}` | `{0, 1}` | ✅ PASS |

**Batch validation: PASS**

---

## 6. Forward / Backward Validation

| Check | Observed | Expected | Status |
|---|---|---|---|
| Logits shape | `(8, 2)` | `(B, 2)` | ✅ PASS |
| Logits finite | All finite | All finite | ✅ PASS |
| Loss value | `0.6121` | Finite | ✅ PASS |
| Grad norm after backward | `1.5614` | > 0 | ✅ PASS |

**Forward/backward validation: PASS**

---

## 7. Optimizer Update Validation

Procedure:
1. All trainable parameter tensors cloned before `optimizer.step()`
2. `optimizer.step()` called
3. Maximum absolute parameter delta computed across all parameters: `max(|param_after − param_before|)`

| Metric | Value |
|---|---|
| Max parameter delta | `1.00 × 10⁻³` |
| At least one parameter changed | ✅ YES |
| Param update status | ✅ PASS |

The maximum delta of `1.00e-03` equals the learning rate (1e-3), consistent with Adam's first step from zero momentum/variance for the largest gradient component. Parameters were updated.

---

## 8. Tiny Train-Loop Results

| Metric | Value |
|---|---|
| Batches processed | 5 (hard cap = 5) |
| Mean train loss | 0.6926 |
| Train accuracy | 0.6250 (62.5%) |

The model starts from random initialization. The train accuracy of 62.5% after 5 batches is above random chance (50%), consistent with early-stage gradient descent on a near-balanced binary dataset. No convergence is expected or claimed.

---

## 9. Tiny Val-Loop Results

| Metric | Value |
|---|---|
| Batches processed | 3 (hard cap = 3) |
| Mean val loss | 0.7429 |
| Val accuracy | 0.0417 (4.2%) |

Val accuracy of 4.2% on 3 batches (24 samples) at random initialization reflects the model predicting a single class for all samples in those batches — expected behaviour with minimal training (only the standalone validation step + 5 train batches ran before val). This is a **mechanics check only**; accuracy is not a gate criterion.

---

## 10. Guardrails Confirmed

| Guardrail | Status |
|---|---|
| No test evaluation | ✅ Confirmed — test split never loaded or evaluated |
| No full training | ✅ Confirmed — 1 epoch, capped at 5 train batches + 3 val batches |
| No QNN work | ✅ Confirmed — no `qcore/circuit`, `qcore/ansatz`, or quantum code used |
| No checkpoint saved | ✅ Confirmed — no `torch.save()` call; no `.pt` file written |
| No dataset modification | ✅ Confirmed — loader is read-only |
| No `data/processed/` files staged | ✅ Confirmed — gitignored |
| No branch switch | ✅ Confirmed |
| No push | ✅ Confirmed |

---

## 11. Environment Note

The GPU container has NumPy 2.2.6 installed against a PyTorch build targeting NumPy 1.x. This produces a non-fatal `UserWarning` on `torch` import (`Failed to initialize NumPy: _ARRAY_API not found`). The warning does not affect training mechanics — PyTorch operations run correctly on CUDA. The loader avoids `torch.from_numpy()` (uses `torch.tensor()`) as documented in Q15. This container-level dependency conflict is pre-existing and deferred to infrastructure work.

---

## 12. Known Limitations

1. **5 train batches / 3 val batches is not representative.** This smoke test validates mechanics only — loss curves, accuracy, and generalisation are not meaningful at this scale.

2. **No class-weighted loss in smoke test.** `nn.CrossEntropyLoss` is used unweighted. The dataset is near-balanced (0.97:1 ratio), so class weighting is not critical, but for a full baseline it should be evaluated.

3. **No augmentation.** No transforms are applied at the DataLoader level. The full baseline run (Q17) should evaluate augmentation strategies.

4. **No model checkpoint.** No `.pt` file is saved in this slice. Full baseline training (Q17) will introduce checkpointing with val-loss-based model selection.

5. **Val accuracy at 3 batches is not meaningful.** The model is essentially untrained at this point (6 gradient steps total). Val accuracy will only be meaningful after full training.

---

## 13. Next Slice Recommendation

```
Slice Q17 — VinDr-SpineXR Classical Baseline Full Training

Goal:
Run a controlled classical CNN baseline on the full VinDr-SpineXR binary dataset
with metrics, checkpointing, and a technical report.
```

---

## 14. Smoke Test Output (Verbatim)

```
=== VinDr-SpineXR Classical Baseline Smoke Test ===
Root: data/processed/vindr_binary_roi_224
Device: cuda
Seed: 42

Dataset:
  train length: 6712
  val length:   1677

Model:
  name: CNN3Block  (build_model from qcore.models.cnn)
  params: 23,650

Batch validation:
  x shape: (8, 1, 224, 224)
  y shape: (8,)
  labels valid: PASS

Forward/backward:
  logits shape: (8, 2)
  loss finite:  PASS  (loss=0.6121)
  grad norm:    1.5614
  max param delta: 1.00e-03
  param update: PASS

Tiny training (1 epoch, 5 batches):
  train batches processed: 5
  train loss: 0.6926
  train acc:  0.6250

Tiny validation (3 batches):
  val batches processed: 3
  val loss: 0.7429
  val acc:  0.0417

Smoke test: PASS
```

---

```
Classical smoke status: PASS
```
