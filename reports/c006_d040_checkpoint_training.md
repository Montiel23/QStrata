# C006-D040 Classical Anchor Checkpoint Training

- **Status:** Complete
- **Date:** 2026-05-25
- **Branch:** feature/qnn-integration
- **Slice:** Q6

---

## 1. Title

C006-D040 Classical Anchor Checkpoint Training — Slice Q6

Training the C006-D040 depthwise-separable CNN architecture on PneumoniaMNIST
to produce a pretrained backbone for `DVHybridCNNQNN`.

---

## 2. Training Configuration

| Parameter | Value |
|---|---|
| Config path | `/workspace/experiments/configs/binary_baseline_depthwise_sep_wide.yaml` |
| Dataset | PneumoniaMNIST binary (`pneumoniamnist`) |
| Split sizes | Train: 4,708 / Val: 524 / Test: 624 |
| Seed | 42 |
| Epochs | 10 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss (balanced class weights) |
| Class weights | [1.93904447555542, 0.6737263798713684] |
| Device | cuda |
| Architecture | depthwise_sep, conv_channels=[64, 128], params=9,870 |

---

## 3. Per-Epoch Results

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Time |
|---|---|---|---|---|---|
|  1 | 0.5178 | 0.4769 | 77.27% | 89.12% | 0.9s |
|  2 | 0.3198 | 0.2853 | 88.34% | 87.02% | 0.6s |
|  3 | 0.2750 | 0.2520 | 89.08% | 90.46% | 0.6s |
|  4 | 0.2551 | 0.2555 | 89.38% | 90.65% | 0.5s |
|  5 | 0.2435 | 0.2611 | 90.29% | 91.60% | 0.6s |
|  6 | 0.2383 | 0.3218 | 89.97% | 91.60% | 0.6s |
|  7 | 0.2387 | 0.2318 | 90.63% | 91.22% | 0.6s |
|  8 | 0.2349 | 0.2391 | 90.70% | 90.46% | 0.6s |
|  9 | 0.2329 | 0.2449 | 90.42% | 91.98% | 0.6s |
| 10 | 0.2201 | 0.2395 | 91.53% | 86.64% | 0.6s |

---

## 4. Best Epoch Summary

| Item | Value |
|---|---|
| Best epoch | 9 |
| Best val accuracy | 91.98% |
| Checkpoint path | `checkpoints/c006_d040_classical_anchor.pt` |

Checkpoint contents: `model_state_dict`, `epoch`, `best_val_acc`, `config`.

---

## 5. Test Accuracy

**Test accuracy: 86.22%** — analysis only; not used as fitness signal or gate criterion.

Evaluated at the best-val checkpoint (epoch 9).

---

## 6. Checkpoint Verification

| Check | Result | Notes |
|---|---|---|
| Checkpoint file exists | PASS | `/workspace/checkpoints/c006_d040_classical_anchor.pt` |
| Checkpoint reloads without error | PASS | `build_model()` + `load_state_dict()` |
| Reloaded model val acc matches saved val acc | PASS | reloaded=91.98%, saved=91.98% |
| Backbone state dict loads into `DVHybridCNNQNN` | PASS | 28 keys loaded, 9612 params |

---

## 7. Notes

- **Architecture:** C006-D040 uses depthwise-separable convolution blocks (`build_block("depthwise_sep", ...)`). The model[:4] backbone slice (blocks 0–1 + AdaptiveAvgPool2d + Flatten) maps to `DVHybridCNNQNN.backbone`.
- **Backbone compatibility:** 28 backbone keys were transferred. Layers 2 (AdaptiveAvgPool2d) and 3 (Flatten) carry no learnable parameters — only layers 0 and 1 (the depthwise-sep blocks) contribute to the backbone state dict.
- **Val acc reference:** The classical anchor `v1_classical_anchor` achieved best_val_acc = 91.79% in the Slice 29–30 stability validation (4 seeds, epochs varied). The result here uses seed=42 and epochs=10.
- **Next step:** `DVHybridCNNQNN.__init__` should be updated in Q7 to load `checkpoints/c006_d040_classical_anchor.pt` during backbone construction, replacing the current random initialisation.
