# DV Hybrid PneumoniaMNIST Full Baseline Report

- **Status:** Complete
- **Date:** 2026-05-25
- **Branch:** feature/qnn-integration
- **Slice:** Q8

---

## 1. Title

DV Hybrid PneumoniaMNIST Full Training Baseline — Slice Q8

Full 30-epoch training run of `DVHybridCNNQNN` with both Q5 (differentiable
`torch.atan` encoding) and Q6 (pretrained C006-D040 backbone) fixes in place.
Establishes the first statistically meaningful training curve for the DV hybrid
model on PneumoniaMNIST, with complete ML and quantum metrics.

---

## 2. Context

**Prior slices:**
- **Q4** — Initial DV hybrid run: random backbone + broken projection gradient
  (`np.arctan` detached autograd). Val acc 74.24% = majority-class base rate.
- **Q5** — Gradient restoration: `np.arctan` → `torch.atan`; `.detach()` removed.
  Smoke test confirmed proj grad norm = 1.39e-03 (was 0.0).
- **Q6** — C006-D040 classical anchor trained (10 epochs, best val 91.98%),
  checkpoint saved to `checkpoints/c006_d040_classical_anchor.pt`.
- **Q7** — 3-epoch sanity rerun with Q5+Q6 fixes. Confirmed: collapse resolved,
  proj gradient active (2.62e-02), backbone load verified. Val acc 91.98% at
  epoch 1; too short to assess convergence.
- **Q8 (this run)** — 30-epoch full baseline. Same architecture, same config.

**Research question:** Given a correctly wired DV hybrid (gradient path restored,
pretrained backbone loaded), does the model converge meaningfully over 30 epochs,
and how close can it approach the 91.98% classical anchor val acc?

---

## 3. Model Summary

Architecture is identical to Q7 (`DVHybridCNNQNN`, unchanged).

| Component | Type | Frozen / Trainable | Parameters | Source |
|---|---|---|---|---|
| CNN backbone (`model[:4]`) | 2× depthwise-sep + AdaptiveAvgPool2d + Flatten | **Frozen** | 9,612 | Pretrained C006-D040 (Q6) |
| Projection layer | `nn.Linear(128, 4)`, no activation | **Trainable** | 516 | Random init; Q5 gradient restored |
| Quantum theta | `nn.Parameter` shape `(1, 2, 4, 3)` | **Trainable** | 24 | Random init |
| Readout layer | `nn.Linear(16, 2)` | **Trainable** | 34 | Random init |
| **Total trainable** | | | **574** | |
| **Total frozen** | | | **9,612** | |

---

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Checkpoint | `checkpoints/c006_d040_classical_anchor.pt` |
| Dataset | PneumoniaMNIST binary |
| Train / Val / Test | 4,708 / 524 / 624 |
| Seed | 42 |
| Epochs | 30 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | `nn.CrossEntropyLoss(weight=balanced)` |
| Class weights | [0.742141, 0.257859] |
| Device | cpu (quantum simulator CPU-only) |
| Test accuracy role | **Analysis only — not a fitness gate** |

---

## 5. Backbone and Gradient Verification

| Check | Result | Detail |
|---|---|---|
| Checkpoint loaded — no key mismatch | PASS | 28 keys, 9,612 params — checkpoint epoch 9, val acc 91.98% |
| Backbone frozen — no trainable backbone params | PASS | All 16 backbone params have `requires_grad=False` |
| Best checkpoint reloaded before test eval | YES | epoch 21, val acc 92.18% |
| Projection grad norm > 0 at epoch 1 | PASS | `2.62e-02` |
| Theta grad norm > 0 at epoch 1 | PASS | `5.81e-02` |
| Readout grad norm > 0 at epoch 1 | PASS | `2.68e-01` |

---

## 6. Per-Epoch Results

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Theta ∇ | Proj ∇ | Readout ∇ | Time |
|---|---|---|---|---|---|---|---|---|
|  1 | 0.6271 | 0.4920 | 74.81% | 91.98% | 5.81e-02 | 2.62e-02 | 2.68e-01 | 59.0s |
|  2 | 0.3673 | 0.3053 | 90.97% | 90.46% | 9.12e-02 | 5.00e-02 | 1.96e-01 | 58.1s |
|  3 | 0.2692 | 0.2577 | 90.48% | 90.84% | 1.05e-01 | 6.05e-02 | 1.49e-01 | 58.0s |
|  4 | 0.2379 | 0.2378 | 90.76% | 91.22% | 1.21e-01 | 6.99e-02 | 1.30e-01 | 57.8s |
|  5 | 0.2218 | 0.2271 | 91.08% | 91.03% | 1.32e-01 | 7.50e-02 | 1.25e-01 | 57.6s |
|  6 | 0.2165 | 0.2215 | 91.08% | 91.41% | 1.45e-01 | 8.20e-02 | 1.21e-01 | 55.6s |
|  7 | 0.2080 | 0.2203 | 91.84% | 91.60% | 1.57e-01 | 8.56e-02 | 1.22e-01 | 55.9s |
|  8 | 0.2045 | 0.2159 | 91.84% | 91.41% | 1.59e-01 | 8.63e-02 | 1.18e-01 | 55.5s |
|  9 | 0.2010 | 0.2169 | 91.89% | 90.65% | 1.62e-01 | 8.82e-02 | 1.14e-01 | 55.6s |
| 10 | 0.1977 | 0.2137 | 92.10% | 91.41% | 1.61e-01 | 8.67e-02 | 1.10e-01 | 55.9s |
| 11 | 0.1973 | 0.2130 | 92.44% | 91.22% | 1.68e-01 | 8.98e-02 | 1.11e-01 | 55.9s |
| 12 | 0.1956 | 0.2134 | 92.40% | 91.60% | 1.74e-01 | 9.29e-02 | 1.13e-01 | 55.6s |
| 13 | 0.1943 | 0.2124 | 92.63% | 91.60% | 1.65e-01 | 8.89e-02 | 1.08e-01 | 55.7s |
| 14 | 0.1913 | 0.2117 | 92.63% | 91.60% | 1.71e-01 | 9.12e-02 | 1.09e-01 | 55.0s |
| 15 | 0.1918 | 0.2115 | 92.59% | 91.60% | 1.76e-01 | 9.30e-02 | 1.11e-01 | 54.9s |
| 16 | 0.1905 | 0.2114 | 92.52% | 91.60% | 1.72e-01 | 9.20e-02 | 1.09e-01 | 55.4s |
| 17 | 0.1924 | 0.2125 | 92.69% | 91.98% | 1.80e-01 | 9.52e-02 | 1.14e-01 | 55.7s |
| 18 | 0.1910 | 0.2105 | 92.69% | 91.60% | 1.75e-01 | 9.33e-02 | 1.08e-01 | 55.4s |
| 19 | 0.1880 | 0.2102 | 92.54% | 91.79% | 1.79e-01 | 9.43e-02 | 1.09e-01 | 55.2s |
| 20 | 0.1922 | 0.2106 | 92.71% | 91.98% | 1.80e-01 | 9.54e-02 | 1.11e-01 | 55.6s |
| 21 | 0.1904 | 0.2106 | 92.78% | 92.18% | 1.75e-01 | 9.24e-02 | 1.07e-01 | 55.6s |
| 22 | 0.1862 | 0.2112 | 92.63% | 91.60% | 1.81e-01 | 9.60e-02 | 1.10e-01 | 55.6s |
| 23 | 0.1916 | 0.2096 | 92.88% | 91.79% | 1.84e-01 | 9.67e-02 | 1.11e-01 | 55.3s |
| 24 | 0.1909 | 0.2105 | 92.61% | 91.60% | 1.90e-01 | 9.89e-02 | 1.13e-01 | 55.9s |
| 25 | 0.1875 | 0.2109 | 92.74% | 91.60% | 1.85e-01 | 9.56e-02 | 1.10e-01 | 55.8s |
| 26 | 0.1893 | 0.2091 | 92.67% | 92.18% | 1.86e-01 | 9.90e-02 | 1.11e-01 | 55.3s |
| 27 | 0.1891 | 0.2090 | 92.78% | 91.79% | 1.81e-01 | 9.44e-02 | 1.06e-01 | 55.8s |
| 28 | 0.1903 | 0.2100 | 92.91% | 91.60% | 1.83e-01 | 9.51e-02 | 1.08e-01 | 56.2s |
| 29 | 0.1884 | 0.2104 | 92.71% | 91.79% | 1.84e-01 | 9.63e-02 | 1.10e-01 | 55.8s |
| 30 | 0.1897 | 0.2099 | 92.80% | 92.18% | 1.93e-01 | 1.01e-01 | 1.14e-01 | 55.4s |

---

## 7. Best-Epoch Analysis

| Item | Value |
|---|---|
| Best val epoch | 21 / 30 |
| Best val acc | 92.18% |
| Train acc at best-val epoch | 92.78% |
| Train loss at best-val epoch | 0.1904 |
| Val loss at best-val epoch | 0.2106 |

---

## 8. Final ML Metrics (at best-val checkpoint)

| Metric | Value |
|---|---|
| Val accuracy (best epoch 21) | 92.18% |
| Test accuracy **(analysis only)** | 87.98% |
| Precision | 0.8832 |
| Recall | 0.9308 |
| F1-score | 0.9064 |
| AUROC | 0.9508 |
| AUPRC | 0.9676 |

---

## 9. Confusion Matrix (test set, at best-val checkpoint)

| | Predicted Normal (0) | Predicted Pneumonia (1) |
|---|---|---|
| Actual Normal (0)    | 186 (TN) | 48 (FP) |
| Actual Pneumonia (1) | 27 (FN) | 363 (TP) |

Q4 reference (all predicted pneumonia): TN=0, FP=234, FN=0, TP=390

---

## 10. Training Curve Summary

### 10.1 Val Acc at Milestones

| Epoch | Val Acc |
|---|---|
| 1 | 91.98% |
| 5 | 91.03% |
| 10 | 91.41% |
| 15 | 91.60% |
| 20 | 91.98% |
| 25 | 91.60% |
| 30 | 92.18% |

### 10.2 Threshold Crossing

| Threshold | First Epoch Crossed |
|---|---|
| 80% val acc | 1 |
| 85% val acc | 1 |
| 90% val acc | 1 |

### 10.3 Convergence and Overfitting

Mean train-val gap (last 5 epochs) = 0.87 pp — no clear overfitting.

Val acc at final epoch (30): 92.18%.
Val acc at best epoch (21): 92.18%.
Peak-to-final delta: 0.00 pp.

---

## 11. Quantum Metrics

### 11.1 Gradient Norm Evolution

| Epoch | Theta ∇ | Proj ∇ | Readout ∇ |
|---|---|---|---|
|  1 | 5.81e-02 | 2.62e-02 | 2.68e-01 |
|  2 | 9.12e-02 | 5.00e-02 | 1.96e-01 |
|  3 | 1.05e-01 | 6.05e-02 | 1.49e-01 |
|  4 | 1.21e-01 | 6.99e-02 | 1.30e-01 |
|  5 | 1.32e-01 | 7.50e-02 | 1.25e-01 |
|  6 | 1.45e-01 | 8.20e-02 | 1.21e-01 |
|  7 | 1.57e-01 | 8.56e-02 | 1.22e-01 |
|  8 | 1.59e-01 | 8.63e-02 | 1.18e-01 |
|  9 | 1.62e-01 | 8.82e-02 | 1.14e-01 |
| 10 | 1.61e-01 | 8.67e-02 | 1.10e-01 |
| 11 | 1.68e-01 | 8.98e-02 | 1.11e-01 |
| 12 | 1.74e-01 | 9.29e-02 | 1.13e-01 |
| 13 | 1.65e-01 | 8.89e-02 | 1.08e-01 |
| 14 | 1.71e-01 | 9.12e-02 | 1.09e-01 |
| 15 | 1.76e-01 | 9.30e-02 | 1.11e-01 |
| 16 | 1.72e-01 | 9.20e-02 | 1.09e-01 |
| 17 | 1.80e-01 | 9.52e-02 | 1.14e-01 |
| 18 | 1.75e-01 | 9.33e-02 | 1.08e-01 |
| 19 | 1.79e-01 | 9.43e-02 | 1.09e-01 |
| 20 | 1.80e-01 | 9.54e-02 | 1.11e-01 |
| 21 | 1.75e-01 | 9.24e-02 | 1.07e-01 |
| 22 | 1.81e-01 | 9.60e-02 | 1.10e-01 |
| 23 | 1.84e-01 | 9.67e-02 | 1.11e-01 |
| 24 | 1.90e-01 | 9.89e-02 | 1.13e-01 |
| 25 | 1.85e-01 | 9.56e-02 | 1.10e-01 |
| 26 | 1.86e-01 | 9.90e-02 | 1.11e-01 |
| 27 | 1.81e-01 | 9.44e-02 | 1.06e-01 |
| 28 | 1.83e-01 | 9.51e-02 | 1.08e-01 |
| 29 | 1.84e-01 | 9.63e-02 | 1.10e-01 |
| 30 | 1.93e-01 | 1.01e-01 | 1.14e-01 |

Projection grad norm at epoch 1: `2.62e-02` (Q5 fix confirmed; was `0.00e+00` in Q4).
Proj gradient active across all 30 epochs: **YES**.

### 11.2 Probability Distribution Validity

| Epoch | Prob Sum Mean | Prob Sum Std |
|---|---|---|
|  1 | 1.000000 | 0.000000 |
|  2 | 1.000000 | 0.000000 |
|  3 | 1.000000 | 0.000000 |
|  4 | 1.000000 | 0.000000 |
|  5 | 1.000000 | 0.000000 |
|  6 | 1.000000 | 0.000000 |
|  7 | 1.000000 | 0.000000 |
|  8 | 1.000000 | 0.000000 |
|  9 | 1.000000 | 0.000000 |
| 10 | 1.000000 | 0.000000 |
| 11 | 1.000000 | 0.000000 |
| 12 | 1.000000 | 0.000000 |
| 13 | 1.000000 | 0.000000 |
| 14 | 1.000000 | 0.000000 |
| 15 | 1.000000 | 0.000000 |
| 16 | 1.000000 | 0.000000 |
| 17 | 1.000000 | 0.000000 |
| 18 | 1.000000 | 0.000000 |
| 19 | 1.000000 | 0.000000 |
| 20 | 1.000000 | 0.000000 |
| 21 | 1.000000 | 0.000000 |
| 22 | 1.000000 | 0.000000 |
| 23 | 1.000000 | 0.000000 |
| 24 | 1.000000 | 0.000000 |
| 25 | 1.000000 | 0.000000 |
| 26 | 1.000000 | 0.000000 |
| 27 | 1.000000 | 0.000000 |
| 28 | 1.000000 | 0.000000 |
| 29 | 1.000000 | 0.000000 |
| 30 | 1.000000 | 0.000000 |

`prob_sum` ≈ 1.0 across all epochs — unitary preservation by the quantum circuit backend confirmed.

### 11.3 State Entropy

Not tracked in this baseline. `_quantum_forward_single` returns `|ψ|²` probabilities
only. Von Neumann entropy monitoring is deferred to a future slice when
`DVHybridCNNQNN` exposes `_states`.

---

## 12. Comparison Table

| Metric | Q4 DV hybrid (random) | Q7 DV hybrid (pretrained, 3ep) | Q8 DV hybrid (pretrained, 30ep) | C006-D040 classical |
|---|---|---|---|---|
| Best val acc | 74.24% | 91.98% | **92.18%** | 91.98% |
| Test acc (analysis) | 62.50% | 88.14% | 87.98% | 86.22% |
| F1-score | — | — | 0.9064 | — |
| AUROC | — | — | 0.9508 | — |
| AUPRC | — | — | 0.9676 | — |
| Proj grad norm (ep 1) | 0.00e+00 | 2.62e-02 | 2.62e-02 | N/A |
| Backbone | Random init | Pretrained C006-D040 | Pretrained C006-D040 | N/A (full model) |
| Epochs trained | 3 | 3 | 30 | 10 |
| Trainable params | 574 (58 effective) | 574 (all effective) | 574 (all effective) | 9,870 |
| Majority-class collapse | YES | NO (TN=188) | YES (TN=186) | N/A |

Val acc gap Q7 → Q8 : **+0.20 pp**
Val acc gap to classical anchor: **-0.20 pp**

---

## 13. Explicit Verdicts

1. **Majority-class collapse resolved:** `YES` —
   Q4 predicted pneumonia for every sample (TN=0, FP=234, FN=0, TP=390).
   Q8 confusion matrix: TN=186, FP=48, FN=27, TP=363.
   Both classes predicted throughout training; collapse definitively resolved.

2. **Projection gradient active:** `YES (all epochs)` —
   Epoch-1 proj grad norm = `2.62e-02` (Q5 fix confirmed; was `0.00e+00` in Q4).
   All three trainable components (proj 516 params, theta 24 params,
   readout 34 params) receive gradient updates across all 30 epochs.

3. **Convergence over 30 epochs:** `YES` —
   Best val acc 92.18% reached at epoch 21.
   Val acc at epoch 30: 92.18%.
   Mean train-val gap (last 5 epochs) = 0.87 pp — no clear overfitting.

4. **Pretrained backbone benefit vs Q7 (3-epoch):** `EQUAL` —
   Q7 best val acc over 3 epochs: 91.98%.
   Q8 best val acc over 30 epochs: 92.18%.
   Extended training with pretrained backbone maintains or improves on the Q7 3-epoch result.
   Val acc gap to C006-D040 classical anchor: -0.20 pp.

5. **Ready for Q9 (next slice):** `YES` —
   Full gradient flow confirmed on all three trainable components across all epochs. Pipeline is in a sound state to proceed to the next experimental slice.

---

## 14. Limitations and Next-Step Recommendation

**Limitations:**
1. **CPU-only quantum simulator.** The `Backend` runs matrix-multiplication
   per sample on CPU; epoch times reflect this (~56s/epoch average at batch_size=8).
   Batched circuit execution would significantly improve throughput.
2. **Single seed.** All results are seed=42. Multi-seed variance is not yet quantified.
3. **Frozen backbone.** No selective unfreezing was evaluated. Fine-tuning the
   final CNN layers jointly with the quantum head may close the gap to the
   classical anchor.
4. **Depth=1, n_qubits=4.** Minimal quantum circuit. Increasing depth or qubits
   may improve expressibility at higher compute cost.
5. **No learning rate schedule.** Fixed LR=0.001 across all 30 epochs. A
   cosine or step-decay schedule may improve late-training convergence.

**Recommended next step:**
With the 30-epoch training curve established, the immediate recommended next steps
are: (a) multi-seed validation (seeds 0, 1, 2) to quantify variance, or (b)
architecture ablations (depth, n_qubits) to assess expressibility vs compute
trade-offs, guided by the convergence behaviour observed in this baseline.
