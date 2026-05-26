# VinDr-SpineXR DV Hybrid Smoke Test Report
## Slice Q18

**Branch:** `feature/qnn-integration`  
**Date:** 2026-05-26  
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Slice Q17 ran the first full classical CNN baseline on the VinDr-SpineXR binary ROI dataset.
Training completed via early stopping at epoch 6 (best epoch=1), with Test AUROC=0.6224 and
Test F1=0.5355. The Q17 report noted convergence instability — validation loss spiked while
training loss fell — suggesting the CNN3Block architecture (lacking inter-block spatial
downsampling) is a weak classical reference.

This slice (Q18) validates the complete DV hybrid CNN-QNN pipeline end-to-end on the VinDr-SpineXR
binary dataset: loader → CNN backbone → projection → DV quantum circuit → readout → loss →
backward → optimizer step → tiny train/val loop. This is **mechanics validation only** — not full
training, not a performance comparison.

**No full training occurs in this slice.**  
**No test evaluation occurs in this slice.**  
**No checkpoint is saved.**  
**No PennyLane or external quantum libraries used.**  
**No dataset modification.**

---

## 2. Dataset Root and Splits Used

| Parameter | Value |
|---|---|
| Dataset root | `data/processed/vindr_binary_roi_224/` |
| Splits used | train, val only |
| Test split | Not evaluated in this slice |
| train length | 6,712 |
| val length | 1,677 |

---

## 3. Q17 Classical Baseline Reference

| Metric | Q17 Classical |
|---|---|
| Test AUROC | 0.6224 |
| Test F1 | 0.5355 |
| Test accuracy | 60.66% |
| Best epoch | 1 of 6 run |
| Stop reason | Early stopping (patience=5) |
| Status | PASS |

---

## 4. Q20 Interpretation Guardrail

```
Q20 Interpretation Guardrail:
If the DV hybrid model outperforms the current VinDr classical baseline, do not claim
quantum advantage. The Q17 classical baseline is potentially weak due to missing
inter-block spatial downsampling. A classical ablation with MaxPool/inter-block
downsampling must be run and compared before any architecture-level conclusions are drawn.
```

This guardrail is also recorded in `docs/roadmaps/binary_classical_quantum_closure_plan.md`
(Section 5b, added in Q18).

---

## 5. DV Hybrid Model Used

**Class:** `DVHybridCNNQNN` from `qcore/models/dv_hybrid_cnn_qnn.py`  
**No modification to `dv_hybrid_cnn_qnn.py` was required or made.**

### 224×224 Compatibility

The `DVHybridCNNQNN` backbone is `build_model(cnn_config)[:4]`, which contains:
- Layer [0]: `build_block("depthwise_sep", 1, 64)` — spatial filtering
- Layer [1]: `build_block("depthwise_sep", 64, 128)` — spatial filtering
- Layer [2]: `AdaptiveAvgPool2d((1, 1))` — collapses **any** `(H, W)` → `(1, 1)`
- Layer [3]: `Flatten()` → output always `(B, 128)` regardless of input spatial size

Since `AdaptiveAvgPool2d(1,1)` is resolution-agnostic, VinDr 224×224 input is **fully
compatible** with the existing `DVHybridCNNQNN` interface without any code changes. The docstring
says "(B, 1, 28, 28)" for PneumoniaMNIST but this is documentation, not a constraint.

### Configuration Applied

```python
CNN_CONFIG = {
    "block_type":     "depthwise_sep",
    "conv_channels":  [64, 128],
    "dropout":        0.3,           # irrelevant — backbone ends at layer [:4]
    "use_batchnorm":  True,
    "pooling":        "adaptive_avg",
    "input_channels": 1,
    "num_classes":    2,
}

model = DVHybridCNNQNN(
    cnn_config=CNN_CONFIG,
    n_qubits=4,
    depth=1,
    alpha=0.1,
    n_classes=2,
)
```

**Note:** No pretrained C006-D040 backbone checkpoint was loaded — the smoke test uses
randomly initialised backbone weights. The backbone is frozen (all backbone parameters
have `requires_grad=False`).

### Parameter Count

| Component | Parameters |
|---|---|
| Backbone (frozen) | 9,612 |
| Projection Linear(128, 4) | 516 |
| Theta (quantum variational) | 24 |
| Readout Linear(16, 2) | 34 |
| **Trainable total** | **574** |

### Device

**CPU** — the DV quantum circuit simulator (`Backend.compile` + `Backend.run`) is
CPU-only; CUDA is not used in the quantum path.

---

## 6. Smoke Test Configuration

| Parameter | Value |
|---|---|
| Dataset root | `data/processed/vindr_binary_roi_224` |
| Batch size | 4 |
| Max train batches | 3 |
| Max val batches | 2 |
| Epochs | 1 |
| Seed | 42 |
| Loss | Unweighted CrossEntropyLoss |
| Optimizer | Adam (lr=1e-3) |
| Device | CPU |
| Per-batch timeout | 120 seconds (hard abort) |
| Class weights | None |
| Augmentation | None |

---

## 7. Batch Validation

One batch sampled from train DataLoader (batch_size=4):

| Check | Observed | Expected | Status |
|---|---|---|---|
| x shape | `(4, 1, 224, 224)` | `(B, 1, 224, 224)` | ✅ PASS |
| y shape | `(4,)` | `(B,)` | ✅ PASS |
| label values | `{0, 1}` | `{0, 1}` | ✅ PASS |

**Batch validation: PASS**

---

## 8. Forward / Backward Validation

| Check | Observed | Status |
|---|---|---|
| Logits shape | `(4, 2)` | ✅ PASS |
| Logits finite | All finite | ✅ PASS |
| Loss value | 0.6459 | ✅ PASS (finite) |
| Forward time | 0.2s (4 samples × 4-qubit circuit) | ✅ PASS (<120s) |
| Backward | Completed without error | ✅ PASS |
| Theta grad norm | 2.78e-02 (> 0) | ✅ PASS |
| Projection grad norm | 4.53e-03 (> 0) | ✅ PASS |
| Readout grad norm | 2.88e-01 (> 0) | ✅ PASS |
| Probability sum (mean) | 1.000000 | ✅ PASS |

All three gradient paths are active: quantum theta, projection layer, and readout layer all
receive non-zero gradient updates — confirming the full end-to-end differentiable path from
loss through readout → circuit probs → quantum circuit → projection layer.

**Forward/backward validation: PASS**

---

## 9. Optimizer Update Validation

| Metric | Value |
|---|---|
| Parameters cloned before step | All 574 trainable parameters |
| Max parameter delta (post-step) | 1.00e-03 |
| At least one parameter changed | ✅ YES |
| Param update status | ✅ PASS |

Max delta of 1.00e-03 equals the learning rate (Adam first step behaviour from zero
momentum). Optimizer update confirmed.

---

## 10. Tiny Train-Loop Results

| Metric | Value |
|---|---|
| Batches processed | 3 (hard cap = 3) |
| Batch 1 loss | 0.7530 | 
| Batch 2 loss | 0.7527 |
| Batch 3 loss | 0.6988 |
| Mean train loss | 0.7348 |
| Train accuracy | 0.3333 (4/12 correct) |
| Per-batch wall time | 0.2–0.3s |

> **Note:** These metrics are sanity-only. With batch_size=4 and 3 batches (12 samples
> total), they are NOT performance signals. Low accuracy at random initialisation is expected.

---

## 11. Tiny Val-Loop Results

| Metric | Value |
|---|---|
| Batches processed | 2 (hard cap = 2) |
| Batch 1 loss | 0.8038 |
| Batch 2 loss | 0.8037 |
| Mean val loss | 0.8038 |
| Val accuracy | 0.0000 (0/8 correct) |
| Per-batch wall time | 0.2s |

> **Note:** These metrics are sanity-only. Val accuracy of 0.0 on 8 samples at random
> initialisation reflects the model predicting the same class for all samples — expected
> behaviour before any meaningful training. This is a **mechanics check only**.

---

## 12. Guardrails Confirmed

| Guardrail | Status |
|---|---|
| No test evaluation | ✅ Confirmed — test split never loaded |
| No full training | ✅ Confirmed — 1 epoch, 3 train batches, 2 val batches |
| No PennyLane or external quantum framework | ✅ Confirmed — QStrata internal framework only |
| No checkpoint saved | ✅ Confirmed — no `torch.save()` call |
| No dataset modification | ✅ Confirmed — loader is read-only |
| `data/processed/` not staged | ✅ Confirmed — gitignored |
| No branch switch | ✅ Confirmed |
| No push | ✅ Confirmed |

---

## 13. Known Limitations

1. **Input size documentation mismatch.** `DVHybridCNNQNN` docstring says `(B, 1, 28, 28)`. The
   actual constraint is the backbone output dimension (128), not the input spatial size — because
   `AdaptiveAvgPool2d(1,1)` collapses any spatial resolution. The 224×224 adaptation is
   zero-cost and requires no code change.

2. **No pretrained backbone.** The smoke test uses randomly initialised backbone weights. The
   full Q19 baseline should consider loading the C006-D040 pretrained checkpoint (if available)
   or training the backbone from scratch — documented in Q19 design.

3. **Per-sample quantum loop.** The forward pass runs one 4-qubit circuit per sample in a
   Python loop. For batch_size=4 this is fast (0.2s). For the full Q19 training run
   (~420 batches/epoch × 16 samples = ~6,720 quantum evaluations/epoch), wall-clock time
   will dominate. Q19 design should assess epoch time before committing to 30 epochs.

4. **CPU-only execution.** The quantum circuit simulator cannot use CUDA. The CNN backbone
   and projection layer also run on CPU in this configuration. Q19 will require epoch-time
   monitoring to ensure training remains practical.

5. **Tiny validation accuracy (0.0) is not meaningful.** 8 samples at random initialisation.
   Not a signal of model quality.

---

## 14. Next Slice Recommendation

```
Slice Q19 — VinDr-SpineXR DV Hybrid Full Training

Goal:
Run a full DV hybrid CNN-QNN training baseline on the VinDr-SpineXR binary
dataset using the validated smoke test configuration.
```

**Q19 design notes from Q18:**
- Confirm epoch wall-clock time on first epoch before committing to 30 epochs
- Consider reducing batch size to 4 (proven fast per-sample) or 8
- Monitor theta/proj/readout grad norms each epoch
- Apply Q20 interpretation guardrail before drawing any comparison conclusions

---

```
DV hybrid smoke status: PASS
```
