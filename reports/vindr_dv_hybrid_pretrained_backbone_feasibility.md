# VinDr-SpineXR DV Hybrid Pretrained-Backbone Feasibility Report
## Slice Q20

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Slice Q19 established the first full DV hybrid CNN-QNN training on VinDr-SpineXR (6 epochs,
early stopping patience=4). However, that run used a **randomly initialized, frozen CNN backbone**
— it did not load any pretrained weights. This means the quantum head was trained on top of
random features, not learned visual representations.

This slice (Q20) answers the prerequisite question before running a scientifically valid DV
hybrid training: **Is there a pretrained classical backbone checkpoint that is architecturally
compatible with `DVHybridCNNQNN`, and does it survive end-to-end loading + gradient validation?**

This is feasibility and smoke validation only. No full training. No test evaluation. No
comparative report. No quantum advantage claim.

---

## 2. Q19 Finding Summary

| Metric | Value |
|---|---|
| Best epoch | 2 of 6 |
| Val loss (best) | 0.6930 |
| Val AUROC (best epoch) | 0.5285 |
| Test AUROC | 0.5442 |
| Test F1 | 0.0000 |
| Test confusion | TN=1070, FP=0, FN=1007, TP=0 |
| Backbone | **Random initialization (not pretrained)** |
| Gradient health | theta, projection, readout all non-zero ✓ |
| Probability conservation | mean=0.999999, max dev=1.01e-06 ✓ |

**Key limitation:** The Q19 DV hybrid result is not a scientifically final benchmark because
the QNN was trained on random convolutional features. The model collapsed to all-class-0
predictions (F1=0, confusion matrix all-negative) — this is consistent with weak frozen
features providing no discriminative signal to the quantum head.

**The Q19 DV result must not be treated as the final DV benchmark for VinDr-SpineXR.**

---

## 3. Checkpoint Discovery

The following prioritized search was performed:

### Priority 1 — C006-D040 Classical Anchor (PneumoniaMNIST pretrained backbone)

```
checkpoints/c006_d040_classical_anchor.pt
```

**Found:** YES (52K, May 24 23:49)

This is the C006-D040 classical anchor checkpoint from Slice Q6. It was trained specifically
as the pretrained backbone for `DVHybridCNNQNN` on the PneumoniaMNIST binary dataset
(depthwise-separable CNN, conv_channels=[64,128]). Slice Q6 explicitly validated that 28
backbone keys transfer to `DVHybridCNNQNN` with 9,612 parameters loaded.

| Property | Value |
|---|---|
| Source slice | Q6 — C006-D040 Classical Anchor Checkpoint Training |
| Checkpoint path | `checkpoints/c006_d040_classical_anchor.pt` |
| Architecture | depthwise_sep, conv_channels=[64, 128], pooling=adaptive_avg |
| Training dataset | PneumoniaMNIST (binary, 28×28 grayscale) |
| Best epoch | 9 of 10 |
| Best val accuracy | 91.98% (PneumoniaMNIST) |
| Format | `model_state_dict`, `epoch`, `best_val_acc`, `config` |

### Priority 2 — VinDr Classical Baseline Checkpoint

```
checkpoints/vindr_classical_baseline_best.pt
```

**Found:** YES — but **INCOMPATIBLE** with `DVHybridCNNQNN` backbone.

The Q17 VinDr classical baseline used a different architecture:
- Block type: standard (not depthwise_sep)
- Channels: [16, 32, 64] → 64-dim output (DVHybridCNNQNN requires 128-dim)
- Keys: `0.weight (16,1,3,3)`, `3.weight (32,16,3,3)`, `6.weight (64,32,3,3)` — no match

Key shape mismatch makes this checkpoint **architecturally incompatible**. It was correctly
excluded.

### Selected Checkpoint

**`checkpoints/c006_d040_classical_anchor.pt` (Priority 1)**

Selected over the VinDr Q17 checkpoint because:
1. Architecturally identical to `DVHybridCNNQNN` backbone (depthwise_sep, [64,128], 128-dim)
2. Explicitly designed and validated for `DVHybridCNNQNN` in Slice Q6
3. The Q17 checkpoint is architecturally incompatible (different block type and channel dims)

**Domain note:** C006-D040 was trained on PneumoniaMNIST (chest X-ray, 28×28) rather than
VinDr-SpineXR (spine X-ray, 224×224). However, because the backbone uses `AdaptiveAvgPool2d(1,1)`
followed by `Flatten`, the spatial dimensions are irrelevant — the convolutional filter weights
(3×3 kernels) are shape-compatible regardless of input resolution. This is an expected and
accepted cross-domain backbone reuse.

---

## 4. Checkpoint Format

| Property | Value |
|---|---|
| Format | `model_state_dict` dict |
| Top-level keys | `model_state_dict`, `epoch`, `best_val_acc`, `config` |
| State dict key count | 30 total (28 backbone + 2 classifier head) |
| Epoch saved | 9 |
| Best val accuracy | 91.98473% |
| Config architecture | `depthwise_sep`, `conv_channels=[64, 128]` |

---

## 5. Backbone Compatibility Analysis

The `DVHybridCNNQNN` backbone (`model[:4]`) uses keys prefixed with `backbone.`:

```
backbone.0.*  — depthwise_sep block 0 (1 → 64 channels)
backbone.1.*  — depthwise_sep block 1 (64 → 128 channels)
backbone.2    — AdaptiveAvgPool2d(1,1) (no parameters)
backbone.3    — Flatten (no parameters)
```

The C006-D040 state dict uses the same keys without the `backbone.` prefix:

```
0.*  — block 0 (14 keys)
1.*  — block 1 (14 keys)
5.*  — classifier Linear(128,2) head (2 keys — SKIPPED)
```

Key remapping: `"0.X" → "backbone.0.X"`, `"1.X" → "backbone.1.X"`, skip `"5.*"`.

| Category | Count |
|---|---|
| Matched backbone keys | **28** |
| Missing backbone keys | **0** |
| Unexpected keys | **0** |
| Skipped classifier keys | **2** (`5.weight`, `5.bias`) |
| Compatible | **YES** |

All 28 backbone keys shape-match exactly (verified in container). Zero mismatches.

---

## 6. Backbone Weight Loading Summary

Loading used `strict=False` to permit the 5 trainable keys (`theta`, `proj.weight`,
`proj.bias`, `readout.weight`, `readout.bias`) to remain randomly initialized.

```
load_state_dict(remapped_backbone_sd, strict=False)
```

| Result | Keys |
|---|---|
| Loaded successfully | 28 backbone keys |
| Missing (expected — trainable) | theta, proj.weight, proj.bias, readout.weight, readout.bias |
| Unexpected | 0 |

The 5 "missing" keys are the trainable quantum/readout components — they intentionally retain
random initialization and will be trained during Q21.

Backbone parameters were re-frozen after loading:
```python
for param in model.backbone.parameters():
    param.requires_grad = False
model.backbone.eval()
```

---

## 7. Frozen Backbone Validation

All backbone parameters confirmed `requires_grad=False` after load:

| Check | Result |
|---|---|
| `backbone.0.*` `requires_grad` | False for all 14 keys |
| `backbone.1.*` `requires_grad` | False for all 14 keys |
| `model.backbone.training` | False (eval mode) |
| Backbone frozen | **YES** |

Verified by assertion: `all(not p.requires_grad for p in model.backbone.parameters())` — passed.

---

## 8. Forward/Backward Validation

Single forward + backward pass on a batch of 4 VinDr-SpineXR 224×224 images:

| Check | Result |
|---|---|
| Logits shape | `(4, 2)` ✓ |
| Loss finite | **PASS** |
| Backbone gradient | **None** (frozen backbone, no grad computed) |
| Theta grad norm | **2.80e-02** |
| Projection grad norm | **1.43e-02** |
| Readout grad norm | **2.88e-01** |
| Max parameter delta | **1.00e-03** |
| Optimizer update | **PASS** |
| Probability sum mean | **1.000000** |

All checks pass. End-to-end gradient flow confirmed: loss → readout → quantum circuit →
projection layer. Backbone correctly received no gradient.

---

## 9. Gradient Health

All three trainable components received non-zero gradients:

| Component | Grad norm | Status |
|---|---|---|
| Theta (quantum parameters, 24 values) | 2.80e-02 | ✓ Non-zero |
| Projection Linear(128→4) | 1.43e-02 | ✓ Non-zero |
| Readout Linear(16→2) | 2.88e-01 | ✓ Non-zero |
| Backbone | None | ✓ Frozen (correct) |

Gradient pattern matches Q18 smoke test and Q19 full training. DV hybrid gradient path
is intact with pretrained backbone.

---

## 10. Optimizer Update Validation

Trainable parameter snapshot taken before `optimizer.step()`. All trainable parameters compared
after step:

| Check | Result |
|---|---|
| Max parameter delta | 1.00e-03 |
| At least one trainable parameter changed | **PASS** |
| Backbone parameters changed | **No** (frozen) |

Only trainable parameters (projection, theta, readout) were updated. Backbone weights were
unchanged, confirming the freeze is enforced through the optimizer as well.

---

## 11. Probability Conservation

Measured across the 4 forward-pass samples:

| Metric | Value |
|---|---|
| Probability sum mean | 1.000000 |
| Max deviation from 1.0 | < 1e-06 |

DV quantum circuit outputs valid probability distributions with pretrained backbone input.
Consistent with Q18 smoke test (1.000000 exactly) and Q19 full training (mean=0.999999,
max_dev=1.01e-06).

---

## 12. Tiny Train/Val Smoke Results

**These are sanity-only metrics. They are NOT performance signals.**

The tiny loop used 3 train batches and 2 val batches (4 samples/batch = 12 train, 8 val
total). Results reflect a partially optimized model on a tiny subset — they cannot be
interpreted as accuracy or loss trends.

### Tiny Training (1 epoch, 3 batches)

| Metric | Value |
|---|---|
| Batches completed | 3 |
| Train loss | 0.7341 |
| Train accuracy | 33.33% |

### Tiny Validation (2 batches)

| Metric | Value |
|---|---|
| Batches completed | 2 |
| Val loss | 0.8020 |
| Val accuracy | 0.00% |

_These values are expected to be noisy at 3 batches. They confirm the pipeline completes
without crash, not model performance. Full performance results will come from Q21 full training._

---

## 13. Feasibility Verdict

### 11 Key Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is the best compatible pretrained classical backbone checkpoint? | `checkpoints/c006_d040_classical_anchor.pt` — Slice Q6 C006-D040 anchor, depthwise_sep [64,128], trained on PneumoniaMNIST (91.98% best val acc). Selected over Q17 VinDr checkpoint (architecturally incompatible). |
| 2 | What checkpoint path was selected? | `checkpoints/c006_d040_classical_anchor.pt` |
| 3 | Is it compatible with the DV hybrid backbone architecture? | **YES** — 28/28 backbone keys matched, 0 missing, 0 unexpected |
| 4 | Can matching backbone weights be loaded into `DVHybridCNNQNN`? | **YES** — `load_state_dict(strict=False)` loaded all 28 keys successfully |
| 5 | Are loaded backbone parameters frozen after loading? | **YES** — `requires_grad=False` for all backbone params; `backbone.eval()` confirmed |
| 6 | Does a forward pass work? | **YES** — logits shape `(4, 2)`, loss finite |
| 7 | Does backward pass work? | **YES** — loss.backward() completes without error |
| 8 | Are theta/projection/readout gradients non-zero? | **YES** — theta=2.80e-02, proj=1.43e-02, readout=2.88e-01 |
| 9 | Does the optimizer update only trainable parameters? | **YES** — max delta 1.00e-03, backbone unchanged |
| 10 | Is probability conservation preserved? | **YES** — prob_sum_mean=1.000000 |
| 11 | Is pretrained-backbone DV full training justified as the next slice? | **YES** — all 10 prior questions answered affirmatively |

```
PRETRAINED_BACKBONE_DV_READY: YES
```

---

## 14. Roadmap Implication

Q19 used a randomly initialized frozen backbone and produced degenerate results (all-class-0,
F1=0). This does not invalidate Q19 as a pipeline validation — it confirms that gradient flow
and probability conservation work — but it does establish that a scientifically valid DV
benchmark requires pretrained features.

This slice confirms:
- A compatible pretrained backbone checkpoint exists (`c006_d040_classical_anchor.pt`)
- It loads correctly into `DVHybridCNNQNN`
- End-to-end training mechanics work with pretrained features
- **The next step is Q21: full DV hybrid training on VinDr-SpineXR with the pretrained backbone**

The VinDr comparative report (formerly Q20 in the roadmap) is now Q22, gated on both
Q21 (pretrained-backbone DV full training) and a stronger classical baseline if needed.

---

## 15. Next Slice Recommendation

Compatible checkpoint found and smoke passed:

```
Slice Q21 — VinDr-SpineXR DV Hybrid Pretrained-Backbone Full Training

Goal:
Run full DV hybrid training on VinDr-SpineXR using checkpoints/c006_d040_classical_anchor.pt
as the frozen pretrained feature extractor. This provides a scientifically valid DV
baseline for the Q22 comparative report.
```

---

```
Pretrained-backbone DV feasibility status: PASS
```
