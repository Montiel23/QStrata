# Q26: VinDr-SpineXR CV Binary Smoke Test

**Branch:** `feature/qnn-integration`
**Date:** 2026-05-26
**Author:** Miguel Lopez (QStrata)

---

## 1. Context

Q26 validates the minimal continuous-variable (CV) hybrid pipeline designed in Q25 and locked in Q25A. The architecture maps VinDr-SpineXR binary classification through a frozen pretrained CNN backbone → a trainable linear compression layer → a Gaussian variational ansatz → deterministic first-moment readout → a trainable linear readout. This is the first time a CV quantum layer has been executed in the QStrata pipeline. The smoke test runs one batch only: no training loop, no validation, no test evaluation. The sole purpose is to confirm that the CV pipeline is numerically stable, gradient-healthy, and correctly integrated with the QStrata Gaussian backend.

---

## 2. Architecture

All parameters sourced from Q25A Section 7 (locked assumptions):

| Component | Specification | Trainable Params |
|---|---|---|
| Backbone | C006-D040 (depthwise_sep [64,128]), `build_model(CNN_CONFIG)[:4]`, frozen | 0 (frozen, 9,612 total) |
| Compression | `nn.Linear(128, 4)` | 128×4 + 4 = 516 |
| CV Ansatz | `GaussianVariationalAnsatz(n_modes=2, depth=1, squeezing_cap=1.5)` | 10 |
| Backend | `GaussianBackend(n_modes=2, hbar=2.0, device='cpu')` | — |
| Readout | `nn.Linear(4, 2)` | 4×2 + 2 = 10 |
| **Total trainable** | | **536** |
| **Total frozen** | | **9,612** |

**Device split:** Backbone on CUDA (GPU), all CV computation on CPU (GaussianBackend constraint). Feature transfer: `features.detach().cpu()` after backbone forward pass.

**Encoding:** `encoded_input = compressed * math.sqrt(2.0 * hbar)` — gradient-safe multiplication (no in-place index operations), ensures gradient flows from loss back to compression layer.

**Readout:** Deterministic first-moment readout — `readout_vec = mu_final` directly (no stochastic homodyne noise). Consistent with Q25 Section 9.

---

## 3. Execution Parameters

| Parameter | Value |
|---|---|
| Script | `scripts/smoke_test_vindr_cv_binary.py` |
| Dataset root | `data/processed/vindr_binary_roi_224` |
| Checkpoint | `checkpoints/c006_d040_classical_anchor.pt` |
| Batch size | 4 |
| Seed | 42 |
| Optimizer | AdamW, lr=1e-3 |
| Loss | CrossEntropyLoss |
| Execution environment | Docker container (`qstrata-gpu`), CUDA available |

---

## 4. Backbone Loading

| Item | Value |
|---|---|
| Checkpoint format | `model_state_dict` |
| Matched keys | 28 |
| Skipped keys | 2 (classifier head: `5.*` — expected) |
| Unexpected keys | 0 |
| Missing backbone keys | 0 |

Key remapping applied (identical to Q21/Q22):
- `"0.*"` → `"backbone.0.*"` — depthwise_sep block 0 (1 → 64 channels)
- `"1.*"` → `"backbone.1.*"` — depthwise_sep block 1 (64 → 128 channels)
- `"5.*"` → skipped — classifier head not used in CV hybrid model

---

## 5. Forward Pass Values

| Item | Value |
|---|---|
| Input shape | `(4, 1, 224, 224)` |
| Labels (y_batch) | `[0, 0, 1, 0]` |
| Logits shape | `(4, 2)` |
| Loss (CrossEntropy) | `0.781537` |

**Logits (raw):**

| Sample | Logit[0] | Logit[1] | Pred |
|---|---|---|---|
| 0 (label=0) | +0.01355 | +0.41921 | 1 |
| 1 (label=0) | −0.09984 | −0.03550 | 1 |
| 2 (label=1) | +0.42328 | +1.12508 | 1 |
| 3 (label=0) | +0.24151 | +0.90860 | 1 |

*Note: All four samples predict class 1 on this single batch. This is expected behaviour for an untrained model with randomly initialised ansatz and compression parameters — no interpretation should be drawn from single-batch predictions.*

**First moments mu_final (per sample, 4-element phase-space vector):**

| Sample | mu[0] | mu[1] | mu[2] | mu[3] |
|---|---|---|---|---|
| 0 | −0.5703 | +0.4235 | −1.1952 | +0.7807 |
| 1 | −0.4076 | +0.3803 | +0.0904 | +0.1588 |
| 2 | −0.3461 | +0.5810 | −2.7812 | +1.7096 |
| 3 | −0.4565 | +0.4670 | −2.3194 | +1.3987 |

All first moments are finite. No NaN or inf detected.

---

## 6. Gradient Health

**Gradient norms after `loss.backward()`:**

| Component | Gradient Norm |
|---|---|
| Compression (`nn.Linear 128→4`) | 4.1564e+00 |
| Ansatz (`GaussianVariationalAnsatz`) | 9.6710e−01 |
| Readout (`nn.Linear 4→2`) | 8.2346e−01 |
| Backbone (frozen) | 0.0 (confirmed zero) |

All three trainable components received non-zero gradient. Gradient flow confirmed through the full path: loss → readout → mu_final (Gaussian circuit) → ansatz parameters + encoded_input → compression layer.

Backbone gradient confirmed zero: no backbone parameter received any gradient after `loss.backward()`. The `torch.no_grad()` context and `features.detach().cpu()` transfer are both effective.

---

## 7. Health Check Results

All 14 health checks passed.

| Check | Result | Notes |
|---|---|---|
| FORWARD_PASS | **PASS** | Forward executed without error |
| LOGITS_SHAPE | **PASS** | `(4, 2)` — correct |
| LOGITS_FINITE | **PASS** | No NaN or inf |
| MU_FINITE | **PASS** | No NaN or inf in first moments |
| COV_FINITE | **PASS** | No NaN or inf in covariance |
| LOSS_FINITE | **PASS** | Loss = 0.781537 |
| GRAD_COMPRESSION | **PASS** | Norm = 4.1564e+00 > 0 |
| GRAD_ANSATZ | **PASS** | Norm = 9.6710e−01 > 0 |
| GRAD_READOUT | **PASS** | Norm = 8.2346e−01 > 0 |
| BACKBONE_FROZEN | **PASS** | Zero backbone gradient confirmed |
| GRAD_FINITE | **PASS** | All gradient norms finite |
| PARAMS_UPDATED | **PASS** | Parameters changed after AdamW step |
| COV_SYMMETRIC | **PASS** | |cov − covᵀ| < 1e−5 |
| COV_PSD | **PASS** | All eigenvalues ≥ −1e−6 |

---

## 8. Parameter Count Verification

| Component | Count | Expected | Match |
|---|---|---|---|
| Compression `nn.Linear(128, 4)` | 516 | 516 | ✓ |
| Ansatz `GaussianVariationalAnsatz(n_modes=2, depth=1)` | 10 | 10 | ✓ |
| Readout `nn.Linear(4, 2)` | 10 | 10 | ✓ |
| **Total trainable** | **536** | **536** | **✓** |
| Frozen backbone | 9,612 | 9,612 | ✓ |

Trainable parameter count matches the Q25A Section 7 specification exactly.

Ansatz parameter breakdown (10 total):

| Parameter | Shape | Count |
|---|---|---|
| `disp_real` | (1, 2) | 2 |
| `disp_imag` | (1, 2) | 2 |
| `squeezing_raw` | (1, 2) | 2 |
| `bs_theta` | (1, 2) | 2 |
| `rot_phi` | (1, 2) | 2 |

---

## 9. Numerical Environment Note

A NumPy 2.x / compiled-module compatibility warning appears at import time:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 ...
UserWarning: Failed to initialize NumPy: _ARRAY_API not found
```

This is a non-fatal environment warning. PyTorch's core tensor operations — including all CV circuit computations, gradient flow, and the `torch.linalg.eigvalsh` PSD check — are unaffected. The smoke test explicitly avoids `.numpy()` calls, using `torch.linalg.eigvalsh` for the covariance PSD check instead. All 14 health checks passed in this environment. The warning was present in Q22 training as well and did not affect that run.

---

## 10. Verdict

```
SMOKE TEST RESULT: PASS
Checks passed: 14 / 14
Hard stops triggered: 0
```

The CV pipeline defined in Q25 and locked in Q25A is:
- **Numerically stable** — no NaN or inf at any point in forward, backward, or optimizer step
- **Gradient-healthy** — all three trainable components (compression, ansatz, readout) received non-zero gradients; backbone received zero gradient
- **Correctly integrated** — GaussianBackend, GaussianVariationalAnsatz, and symplectic gate sequence operate as designed
- **Parameter-correct** — exactly 536 trainable parameters, matching the Q25 design specification

**Q27 (CV Binary Full Training) is unblocked.**

---

## 11. Q27 Preconditions Confirmed

All Q27 preconditions established by this smoke test:

| Precondition | Status |
|---|---|
| Forward pass executes without error | ✓ CONFIRMED |
| All 14 health checks PASS | ✓ CONFIRMED |
| Gradient flow through compression, ansatz, readout | ✓ CONFIRMED |
| Backbone receives zero gradient | ✓ CONFIRMED |
| One optimizer step executes and parameters update | ✓ CONFIRMED |
| No NaN or inf in forward or backward pass | ✓ CONFIRMED |
| Trainable param count = 536 (matches Q25A spec) | ✓ CONFIRMED |
| Backbone loads correctly (28 matched, 2 skipped, 0 unexpected) | ✓ CONFIRMED |

---

```
Q26 status: COMPLETE — PASS
Q27 status: NEXT — CV Binary Full Training
```
